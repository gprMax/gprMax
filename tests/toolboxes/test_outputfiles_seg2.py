# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom

import struct

import h5py
import numpy as np
import pytest

from toolboxes.Utilities.outputfiles_seg2 import export_seg2


def _write_output(filename, trace_number=1, *, dt=4.7e-12, samples=6, component="Ez"):
    with h5py.File(filename, "w") as output:
        output.attrs["Title"] = "SEG-2 export test"
        output.attrs["dt"] = dt
        source = output.create_group("srcs/src1")
        source.attrs["Position"] = (0.1 + 0.001 * trace_number, 0.2, 0.3)
        receiver = output.create_group("rxs/rx1")
        receiver.attrs["Position"] = (0.4 + 0.002 * trace_number, 0.5, 0.6)
        data = receiver.create_dataset(
            component,
            data=np.linspace(-trace_number, trace_number, samples, dtype=np.float64),
        )
        data.attrs["SampleInterval"] = dt
        data.attrs["TimeSampleOffset"] = 0.0


def _free_strings(data, start, size):
    values = {}
    offset = start
    end = start + size
    while offset + 2 <= end:
        length = struct.unpack_from("<H", data, offset)[0]
        if length == 0:
            break
        text = data[offset + 2 : offset + length].split(b"\x00", 1)[0].decode("ascii")
        key, _, value = text.partition(" ")
        values[key] = value
        offset += length
    return values


def test_export_seg2_exact_sampling_geometry_and_float_data(tmp_path):
    files = [tmp_path / f"scan{number}.h5" for number in range(1, 3)]
    for number, filename in enumerate(files, start=1):
        _write_output(filename, number)

    destination = tmp_path / "scan_Ez.sg2"
    summary = export_seg2(files, destination, 1, "Ez")
    raw = destination.read_bytes()

    assert summary.trace_count == 2
    assert summary.sample_count == 6
    assert summary.sample_interval == 4.7e-12
    assert struct.unpack_from("<H", raw, 0)[0] == 0x3A55
    assert struct.unpack_from("<H", raw, 2)[0] == 1
    pointer_size = struct.unpack_from("<H", raw, 4)[0]
    trace_count = struct.unpack_from("<H", raw, 6)[0]
    assert pointer_size == 8
    assert trace_count == 2
    assert raw[8:14] == bytes((1, 0, 0, 1, 10, 0))

    pointers = struct.unpack_from("<2I", raw, 32)
    file_strings = _free_strings(raw, 32 + pointer_size, pointers[0] - 32 - pointer_size)
    assert file_strings["GPRMAX_COMPONENT"] == "Ez"
    assert file_strings["GPRMAX_UNITS"] == "V/m"
    assert file_strings["TRACE_SORT"] == "AS_ACQUIRED"
    assert file_strings["UNITS"] == "METERS"

    for index, pointer in enumerate(pointers, start=1):
        assert struct.unpack_from("<H", raw, pointer)[0] == 0x4422
        descriptor_size = struct.unpack_from("<H", raw, pointer + 2)[0]
        data_size = struct.unpack_from("<I", raw, pointer + 4)[0]
        sample_count = struct.unpack_from("<I", raw, pointer + 8)[0]
        assert raw[pointer + 12] == 4
        assert data_size == 24
        assert sample_count == 6
        strings = _free_strings(raw, pointer + 32, descriptor_size - 32)
        assert float(strings["SAMPLE_INTERVAL"]) == 4.7e-12
        assert strings["TRACE_TYPE"] == "RADAR_DATA"
        assert strings["GPRMAX_UNITS"] == "V/m"
        assert np.fromstring(strings["SOURCE_LOCATION"], sep=" ")[0] == pytest.approx(0.1 + 0.001 * index)
        assert np.fromstring(strings["RECEIVER_LOCATION"], sep=" ")[0] == pytest.approx(0.4 + 0.002 * index)
        samples = np.frombuffer(raw, dtype="<f4", count=6, offset=pointer + descriptor_size)
        np.testing.assert_allclose(samples, np.linspace(-index, index, 6), rtol=1e-7)


def test_export_seg2_preserves_half_step_magnetic_time_offset(tmp_path):
    filename = tmp_path / "scan1.h5"
    _write_output(filename, component="Hy", dt=2e-12)
    with h5py.File(filename, "a") as output:
        del output["rxs/rx1/Hy"].attrs["TimeSampleOffset"]

    destination = tmp_path / "scan_Hy.sg2"
    summary = export_seg2([filename], destination, 1, "Hy")
    raw = destination.read_bytes()
    pointer = struct.unpack_from("<I", raw, 32)[0]
    descriptor_size = struct.unpack_from("<H", raw, pointer + 2)[0]
    strings = _free_strings(raw, pointer + 32, descriptor_size - 32)

    assert summary.time_sample_offset == pytest.approx(-1e-12)
    assert float(strings["DELAY"]) == pytest.approx(-1e-12)
    assert strings["GPRMAX_UNITS"] == "A/m"


def test_export_seg2_refuses_overwrite(tmp_path):
    filename = tmp_path / "scan1.h5"
    _write_output(filename)
    destination = tmp_path / "scan.sg2"

    export_seg2([filename], destination, 1, "Ez")
    with pytest.raises(FileExistsError):
        export_seg2([filename], destination, 1, "Ez")
    export_seg2([filename], destination, 1, "Ez", overwrite=True)
