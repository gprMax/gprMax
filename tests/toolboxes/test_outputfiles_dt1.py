# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom

import re
import struct

import h5py
import numpy as np
import pytest

from toolboxes.Utilities.outputfiles_dt1 import TRACE_HEADER_FORMAT, export_dt1


def _write_output(filename, trace_number=1, *, dt=4.7e-12, samples=6):
    with h5py.File(filename, "w") as output:
        output.attrs["Title"] = "DT1 export test"
        output.attrs["dt"] = dt
        source = output.create_group("srcs/src1")
        source.attrs["Position"] = (0.1 + 0.01 * trace_number, 0.2, 0.3)
        receiver = output.create_group("rxs/rx1")
        receiver.attrs["Position"] = (0.4 + 0.02 * trace_number, 0.5, 0.6)
        data = receiver.create_dataset(
            "Ez",
            data=np.linspace(-2 * trace_number, trace_number, samples, dtype=np.float64),
        )
        data.attrs["SampleInterval"] = dt
        data.attrs["TimeSampleOffset"] = 0.0


def _hd_values(filename):
    values = {}
    for line in filename.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def test_export_dt1_headers_geometry_quantisation_and_hd_metadata(tmp_path):
    files = [tmp_path / f"scan{number}.h5" for number in range(1, 4)]
    for number, filename in enumerate(files, start=1):
        _write_output(filename, number)

    summary = export_dt1(
        files,
        tmp_path / "survey",
        1,
        "Ez",
        nominal_frequency_mhz=1000,
        stacks=4,
    )
    raw = summary.dt1_outputfile.read_bytes()
    trace_bytes = 128 + 6 * 2

    assert summary.trace_count == 3
    assert summary.sample_count == 6
    assert len(raw) == 3 * trace_bytes
    assert summary.dt1_outputfile.name == "survey.DT1"
    assert summary.hd_outputfile.name == "survey.HD"

    previous_position = -1.0
    for index in range(3):
        offset = index * trace_bytes
        header = struct.unpack_from(TRACE_HEADER_FORMAT, raw, offset)
        samples = np.frombuffer(raw, dtype="<i2", count=6, offset=offset + 128)
        trace_number = index + 1
        assert header[0] == trace_number
        assert header[1] > previous_position
        assert header[2] == 6
        assert header[5] == 2
        assert header[6] == pytest.approx(4.7e-12 * 6 * 1e9, rel=1e-6)
        assert header[7] == 4
        assert header[8:11] == pytest.approx((0.4 + 0.02 * trace_number, 0.5, 0.6))
        assert header[11:14] == pytest.approx((0.4 + 0.02 * trace_number, 0.5, 0.6))
        assert header[14:17] == pytest.approx((0.1 + 0.01 * trace_number, 0.2, 0.3))
        expected = np.linspace(-2 * trace_number, trace_number, 6)
        reconstructed = samples.astype(np.float64) * summary.count_scale
        np.testing.assert_allclose(reconstructed, expected, atol=summary.count_scale / 2, rtol=0)
        previous_position = header[1]

    values = _hd_values(summary.hd_outputfile)
    assert values["NUMBER OF TRACES"] == "3"
    assert values["NUMBER OF PTS/TRC"] == "6"
    assert float(values["TOTAL TIME WINDOW"]) == pytest.approx(4.7e-12 * 6 * 1e9)
    assert values["POSITION UNITS"] == "m"
    assert values["NOMINAL FREQUENCY"] == "1000"
    assert values["NUMBER OF STACKS"] == "4"
    assert values["GPRMAX COMPONENT"] == "Ez"
    assert values["GPRMAX ORIGINAL UNITS"] == "V/m"
    assert float(values["GPRMAX COUNT SCALE"]) == summary.count_scale
    assert re.match(r"SI_VALUE = INTEGER_COUNT \* GPRMAX_COUNT_SCALE", values["GPRMAX AMPLITUDE CONVERSION"])


def test_export_dt1_is_readable_by_common_32_float_header_layout(tmp_path):
    """Exercise the offsets used by the open-source GPRPy DT1 reader."""

    files = [tmp_path / "scan1.h5", tmp_path / "scan2.h5"]
    _write_output(files[0], 1)
    _write_output(files[1], 2)
    summary = export_dt1(files, tmp_path / "survey.DT1", 1, "Ez")

    raw = summary.dt1_outputfile.read_bytes()
    sample_count = int(struct.unpack_from("<f", raw, 8)[0])
    trace_bytes = 128 + 2 * sample_count
    last_trace_number = int(struct.unpack_from("<f", raw, len(raw) - trace_bytes)[0])
    data = np.empty((sample_count, last_trace_number), dtype=np.int16)
    for trace in range(last_trace_number):
        data[:, trace] = np.frombuffer(raw, dtype="<i2", count=sample_count, offset=trace * trace_bytes + 128)

    assert sample_count == 6
    assert last_trace_number == 2
    assert data.shape == (6, 2)
    assert np.max(np.abs(data)) == 32_767


def test_export_dt1_refuses_partial_pair_overwrite(tmp_path):
    filename = tmp_path / "scan1.h5"
    _write_output(filename)
    (tmp_path / "survey.HD").write_text("existing")

    with pytest.raises(FileExistsError):
        export_dt1([filename], tmp_path / "survey", 1, "Ez")
    assert not (tmp_path / "survey.DT1").exists()

    summary = export_dt1([filename], tmp_path / "survey", 1, "Ez", overwrite=True)
    assert summary.dt1_outputfile.is_file()
    assert summary.hd_outputfile.is_file()
