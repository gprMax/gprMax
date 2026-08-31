# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import struct

import h5py
import numpy as np
import pytest

from toolboxes.Utilities.outputfiles_segy import export_segy


def _write_output(
    filename,
    trace_number=1,
    *,
    dt=2.5e-12,
    samples=5,
    sources=1,
    component="Ez",
    grid_path="/",
):
    with h5py.File(filename, "w") as output:
        output.attrs["Title"] = "SEG-Y export test"
        grid = output if grid_path == "/" else output.create_group(grid_path.strip("/"))
        grid.attrs["Iterations"] = samples
        grid.attrs["nrx"] = 1
        grid.attrs["nsrc"] = sources
        grid.attrs["dt"] = dt
        for source_number in range(1, sources + 1):
            source = grid.create_group(f"srcs/src{source_number}")
            source.attrs["Position"] = (
                0.1 + 0.001 * trace_number,
                0.2 + 0.01 * source_number,
                0.3,
            )
        receiver = grid.create_group("rxs/rx1")
        receiver.attrs["Position"] = (
            0.4 + 0.002 * trace_number,
            0.5,
            0.6,
        )
        data = receiver.create_dataset(component, data=np.arange(samples, dtype=np.float64) + 10 * trace_number)
        data.attrs["SampleInterval"] = dt
        data.attrs["TimeSampleOffset"] = 0.0


def _read(fmt, data, offset):
    return struct.unpack_from(">" + fmt, data, offset)[0]


def test_export_segy_revision_2_headers_geometry_and_samples(tmp_path):
    files = [tmp_path / f"scan{number}.h5" for number in range(1, 4)]
    for number, filename in enumerate(files, start=1):
        _write_output(filename, number)

    destination = tmp_path / "scan_Ez.sgy"
    summary = export_segy(files, destination, 1, "Ez", line_number=23)
    raw = destination.read_bytes()

    assert summary.trace_count == 3
    assert summary.sample_count == 5
    assert summary.profile == "standard"
    assert len(raw) == 3200 + 400 + 3 * (240 + 5 * 4)
    assert raw[:3] == b"C01"
    assert b"GPRMAX SYNTHETIC GPR DATA" in raw[:3200]

    binary = raw[3200:3600]
    assert _read("i", binary, 4) == 23
    assert _read("H", binary, 16) == 0  # 2.5 ps is not an integer microsecond
    assert _read("H", binary, 24) == 5  # IEEE float32
    assert _read("I", binary, 68) == 5
    assert _read("d", binary, 72) == pytest.approx(2.5e-6)  # microseconds
    assert _read("I", binary, 96) == 0x01020304
    assert binary[300:302] == bytes((2, 1))
    assert _read("Q", binary, 312) == 3

    trace_size = 240 + 5 * 4
    for index in range(3):
        start = 3600 + index * trace_size
        header = raw[start : start + 240]
        samples = np.frombuffer(raw[start + 240 : start + trace_size], dtype=">f4")
        number = index + 1
        assert _read("i", header, 0) == number
        assert _read("h", header, 28) == 27  # vertical electric component
        assert _read("h", header, 68) == -10_000
        assert _read("h", header, 70) == -10_000
        assert _read("i", header, 72) == round((0.1 + 0.001 * number) * 10_000)
        assert _read("i", header, 80) == round((0.4 + 0.002 * number) * 10_000)
        np.testing.assert_array_equal(samples, np.arange(5) + 10 * number)


def test_export_gpr_profile_uses_revision_1_picoseconds_and_resamples(tmp_path):
    files = [tmp_path / f"scan{number}.h5" for number in range(1, 3)]
    for number, filename in enumerate(files, start=1):
        _write_output(filename, number, dt=4.7e-12, samples=6)

    destination = tmp_path / "scan_Ez_gpr.sgy"
    summary = export_segy(files, destination, 1, "Ez", profile="gpr")
    raw = destination.read_bytes()

    # The 23.5 ps input duration is represented by five samples at 5 ps. The
    # legacy interval is deliberately interpreted as picoseconds by GPR tools.
    assert summary.profile == "gpr"
    assert summary.sample_interval == pytest.approx(5e-12)
    assert summary.sample_count == 5
    assert len(raw) == 3200 + 400 + 2 * (240 + 5 * 4)
    assert b"GPR SEG-Y REVISION 1 PROFILE" in raw[:3200]
    assert b"PICOSECONDS, NOT SEGY MICROSECONDS" in raw[:3200]

    binary = raw[3200:3600]
    assert _read("H", binary, 16) == 5
    assert _read("H", binary, 18) == 5
    assert _read("H", binary, 20) == 5
    assert binary[300:302] == bytes((1, 0))
    assert binary[60:100] == bytes(40)  # no revision-2 extended values

    trace_size = 240 + 5 * 4
    for index in range(2):
        start = 3600 + index * trace_size
        header = raw[start : start + 240]
        samples = np.frombuffer(raw[start + 240 : start + trace_size], dtype=">f4")
        assert _read("h", header, 28) == 1  # generic live trace for legacy readers
        assert _read("h", header, 34) == 1  # production rather than test data
        assert _read("H", header, 114) == 5
        assert _read("H", header, 116) == 5
        expected = np.interp(
            np.arange(5) * 5e-12,
            np.arange(6) * 4.7e-12,
            np.arange(6) + 10 * (index + 1),
        )
        np.testing.assert_allclose(samples, expected, rtol=1e-6)


def test_export_gpr_profile_rejects_unrepresentable_interval(tmp_path):
    filename = tmp_path / "scan1.h5"
    _write_output(filename, dt=0.4e-12)

    with pytest.raises(ValueError, match="integer from 1 to 65535 picoseconds"):
        export_segy([filename], tmp_path / "invalid.sgy", 1, "Ez", profile="gpr")


def test_export_requires_source_selection_when_multiple_sources_exist(tmp_path):
    filename = tmp_path / "scan1.h5"
    _write_output(filename, sources=2)

    with pytest.raises(ValueError, match="More than one source"):
        export_segy([filename], tmp_path / "ambiguous.sgy", 1, "Ez")

    summary = export_segy([filename], tmp_path / "selected.sgy", 1, "Ez", source_path="srcs/src2")
    assert summary.source_path == "srcs/src2"


@pytest.mark.parametrize("difference", ["dt", "samples"])
def test_export_rejects_inconsistent_trace_sampling_without_partial_file(tmp_path, difference):
    files = [tmp_path / "scan1.h5", tmp_path / "scan2.h5"]
    _write_output(files[0])
    kwargs = {"dt": 3e-12} if difference == "dt" else {"samples": 6}
    _write_output(files[1], 2, **kwargs)
    destination = tmp_path / "invalid.sgy"

    with pytest.raises(ValueError, match="Sample interval|samples"):
        export_segy(files, destination, 1, "Ez")
    assert not destination.exists()


def test_export_rejects_legacy_merged_receiver_dataset(tmp_path):
    filename = tmp_path / "merged.h5"
    _write_output(filename)
    with h5py.File(filename, "a") as output:
        del output["rxs/rx1/Ez"]
        output["rxs/rx1"].create_dataset("Ez", data=np.ones((5, 2)))

    with pytest.raises(ValueError, match="original one-dimensional A-scan"):
        export_segy([filename], tmp_path / "merged.sgy", 1, "Ez")


def test_export_supports_subgrid_output_and_refuses_overwrite(tmp_path):
    filename = tmp_path / "scan1.h5"
    _write_output(filename, grid_path="subgrids/fine")
    destination = tmp_path / "fine.sgy"

    export_segy([filename], destination, 1, "Ez", grid_path="subgrids/fine")
    with pytest.raises(FileExistsError):
        export_segy([filename], destination, 1, "Ez", grid_path="subgrids/fine")
    export_segy([filename], destination, 1, "Ez", grid_path="subgrids/fine", overwrite=True)


def test_export_rejects_complex_trace_data(tmp_path):
    filename = tmp_path / "scan1.h5"
    _write_output(filename)
    with h5py.File(filename, "a") as output:
        del output["rxs/rx1/Ez"]
        output["rxs/rx1"].create_dataset("Ez", data=np.ones(5, dtype=np.complex128))

    with pytest.raises(ValueError, match="Complex-valued"):
        export_segy([filename], tmp_path / "complex.sgy", 1, "Ez")


def test_export_accepts_real_time_domain_voltage_but_rejects_s_parameters(tmp_path):
    filename = tmp_path / "scan1.h5"
    _write_output(filename)
    with h5py.File(filename, "a") as output:
        line = output.create_group("tls/tl1")
        line.attrs["Position"] = (0.1, 0.2, 0.3)
        line.create_dataset("Vtotal", data=np.linspace(0, 1, 5))

    summary = export_segy(
        [filename],
        tmp_path / "voltage.sgy",
        1,
        "Vtotal",
        trace_group="tls/tl1",
        source_path="tls/tl1",
    )
    assert summary.component == "Vtotal"

    with pytest.raises(ValueError, match="not a supported real time-domain quantity"):
        export_segy([filename], tmp_path / "s11.sgy", 1, "S11")
