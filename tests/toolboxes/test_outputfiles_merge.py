# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import h5py
import numpy as np
import pytest

from toolboxes.Utilities.outputfiles_merge import get_output_data, merge_files


def _write_output(filename, offset=0, iterations=3, dtype=np.float64):
    with h5py.File(filename, "w") as output:
        output.attrs["Title"] = "merge test"
        output.attrs["Iterations"] = iterations
        output.attrs["nrx"] = 1
        output.attrs["dt"] = 1e-10
        source = output.create_group("srcs/src1")
        source.attrs["Position"] = (0.5 + offset, 1.0, 1.5)
        source.attrs["GridPosition"] = (5 + offset, 10, 15)
        receiver = output.create_group("rxs/rx1")
        receiver.attrs["Name"] = "surface"
        receiver.attrs["Position"] = (1.0 + offset, 2.0, 3.0)
        receiver.attrs["GridPosition"] = (10 + offset, 20, 30)
        field = receiver.create_dataset("Ez", data=np.asarray(np.arange(iterations) + offset, dtype=dtype))
        field.attrs["SampleInterval"] = 1e-10
        field.attrs["TimeSampleOffset"] = 0.0
        field.attrs["Quantity"] = "Ez"

        port = output.create_group("ports/receive")
        port.attrs["Name"] = "receive"
        port.attrs["Position"] = (1.5 + offset, 2.0, 3.0)
        port.attrs["GridPosition"] = (15 + offset, 20, 30)
        port.attrs["PortMode"] = "resistive_thevenin"
        port.attrs["TimeSampleOffset"] = 0.5e-10
        port.create_dataset("time", data=(np.arange(iterations - 1) + 0.5) * 1e-10)
        port.create_dataset(
            "Vtotal",
            data=np.asarray(np.arange(iterations - 1) + 100 * offset, dtype=dtype),
        )

        transmission_line = output.create_group("tls/tl1")
        transmission_line.attrs["Position"] = (2.5 + offset, 2.0, 3.0)
        transmission_line.attrs["TimeVoltageOffset"] = 0.0
        transmission_line.create_dataset("time_voltage", data=np.arange(iterations) * 1e-10)
        transmission_line.create_dataset(
            "Vtotal",
            data=np.asarray(np.arange(iterations) + 200 * offset, dtype=dtype),
        )

        frill = output.create_group("frills/frill1")
        frill.attrs["Position"] = (3.5 + offset, 2.0, 3.0)
        frill.attrs["TimeOffset"] = 0.0
        frill.create_dataset("time", data=np.arange(iterations) * 1e-10)
        frill.create_dataset(
            "Vtotal",
            data=np.asarray(np.arange(iterations + 1) + 300 * offset, dtype=dtype),
        )

        subgrid = output.create_group("subgrids/fine")
        subgrid.attrs["Iterations"] = 2 * iterations
        subgrid.attrs["nrx"] = 1
        subgrid.attrs["dt"] = 0.5e-10
        subgrid_receiver = subgrid.create_group("rxs/rx1")
        subgrid_receiver.attrs["Name"] = "fine surface"
        subgrid_receiver.attrs["Position"] = (2.0 + offset, 3.0, 4.0)
        subgrid_receiver.create_dataset("Ex", data=np.asarray(np.arange(2 * iterations) + 10 * offset, dtype=dtype))
        subgrid_port = subgrid.create_group("ports/subgrid_receive")
        subgrid_port.attrs["Position"] = (4.0 + offset, 5.0, 6.0)
        subgrid_port.attrs["TimeSampleOffset"] = 0.25e-10
        subgrid_port.create_dataset("time", data=(np.arange(2 * iterations - 1) + 0.5) * 0.5e-10)
        subgrid_port.create_dataset(
            "Vtotal",
            data=np.asarray(
                np.arange(2 * iterations - 1) + 400 * offset,
                dtype=dtype,
            ),
        )


def test_merge_preserves_receiver_metadata_and_subgrid_outputs(tmp_path):
    files = [tmp_path / "model1.h5", tmp_path / "model2.h5"]
    _write_output(files[0])
    _write_output(files[1], offset=1)

    merged = merge_files(files)

    main, _ = get_output_data(merged, 1, "Ez")
    fine, _ = get_output_data(merged, 1, "Ex", "subgrids/fine")
    voltage, _ = get_output_data(merged, 1, "Vtotal", trace_group="ports/receive")
    fine_voltage, _ = get_output_data(
        merged,
        1,
        "Vtotal",
        "subgrids/fine",
        "ports/subgrid_receive",
    )
    np.testing.assert_array_equal(main, [[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(fine[:, 1] - fine[:, 0], np.full(6, 10))
    np.testing.assert_array_equal(voltage, [[0, 100], [1, 101]])
    np.testing.assert_array_equal(fine_voltage[:, 1] - fine_voltage[:, 0], np.full(5, 400))
    with h5py.File(merged, "r") as output:
        assert output["rxs/rx1"].attrs["Name"] == "surface"
        np.testing.assert_array_equal(output["rxs/rx1"].attrs["Position"], [1, 2, 3])
        assert output["rxs/rx1/Ez"].attrs["SampleInterval"] == 1e-10
        assert output["rxs/rx1/Ez"].attrs["TimeSampleOffset"] == 0.0
        assert output.attrs["MergedOutput"]
        assert output.attrs["MergedOutputSchemaVersion"] == 1
        assert output.attrs["ntraces"] == 2
        np.testing.assert_array_equal(
            output["trace_metadata/rxs/rx1/Position"],
            [[1, 2, 3], [2, 2, 3]],
        )
        np.testing.assert_array_equal(
            output["trace_metadata/srcs/src1/Position"],
            [[0.5, 1, 1.5], [1.5, 1, 1.5]],
        )
        np.testing.assert_array_equal(
            output["trace_metadata/rxs/rx1/GridPosition"],
            [[10, 20, 30], [11, 20, 30]],
        )
        np.testing.assert_array_equal(
            output["tls/tl1/Vtotal"],
            [[0, 200], [1, 201], [2, 202]],
        )
        # The extra raw frill endpoint is not a physical FDTD output sample.
        np.testing.assert_array_equal(
            output["frills/frill1/Vtotal"],
            [[0, 300], [1, 301], [2, 302]],
        )
        assert output["ports/receive/Vtotal"].attrs["Units"] == "V"
        assert output["ports/receive"].attrs["MergedTimeDomainOnly"]
        np.testing.assert_array_equal(
            output["trace_metadata/ports/receive/Position"],
            [[1.5, 2, 3], [2.5, 2, 3]],
        )
        assert [name.decode() for name in output["trace_metadata/InputFiles"]] == [
            "model1.h5",
            "model2.h5",
        ]


def test_merge_rejects_inconsistent_iterations(tmp_path):
    files = [tmp_path / "model1.h5", tmp_path / "model2.h5"]
    _write_output(files[0])
    _write_output(files[1], iterations=4)

    destination = tmp_path / "merged.h5"
    with pytest.raises(ValueError, match="Inconsistent Iterations"):
        merge_files(files, destination)

    assert not destination.exists()


def test_merge_rejects_files_without_receiver_ascans(tmp_path):
    filename = tmp_path / "frequency_only.h5"
    with h5py.File(filename, "w") as output:
        output.attrs["Iterations"] = 3
        output.attrs["nrx"] = 0
        output.attrs["dt"] = 1e-10
        output.create_dataset("study/frequency", data=[1e9, 2e9])
        output.create_dataset("study/S", data=np.ones(2, dtype=np.complex128))

    with pytest.raises(ValueError, match="does not merge frequency-domain"):
        merge_files([filename])


def test_merge_accepts_terminal_voltage_without_point_receiver(tmp_path):
    files = [tmp_path / "antenna1.h5", tmp_path / "antenna2.h5"]
    for index, filename in enumerate(files):
        with h5py.File(filename, "w") as output:
            output.attrs["Iterations"] = 4
            output.attrs["nrx"] = 0
            output.attrs["dt"] = 1e-10
            port = output.create_group("ports/passive_antenna")
            port.attrs["Position"] = (index, 0, 0)
            port.attrs["TimeSampleOffset"] = 0.5e-10
            port.create_dataset("time", data=(np.arange(3) + 0.5) * 1e-10)
            port.create_dataset("Vtotal", data=np.asarray(np.arange(3) + 10 * index, dtype=np.float32))
            port.create_dataset("S11", data=np.ones(2, dtype=np.complex64))

    merged = merge_files(files)
    voltage, dt = get_output_data(
        merged,
        1,
        "Vtotal",
        trace_group="ports/passive_antenna",
    )

    np.testing.assert_array_equal(voltage, [[0, 10], [1, 11], [2, 12]])
    assert dt == 1e-10
    with h5py.File(merged, "r") as output:
        assert "S11" not in output["ports/passive_antenna"]
        assert output.attrs["nrx"] == 0


def test_merge_ignores_non_receiver_frequency_outputs(tmp_path):
    files = [tmp_path / "model1.h5", tmp_path / "model2.h5"]
    for index, filename in enumerate(files):
        _write_output(filename, offset=index)
        with h5py.File(filename, "a") as output:
            output.create_dataset("ntff/surface/frequency/frequencies", data=[1e9])
            output.create_dataset("ntff/surface/frequency/Etheta", data=[1 + 2j])

    merged = merge_files(files)
    with h5py.File(merged, "r") as output:
        assert "ntff" not in output
        np.testing.assert_array_equal(output["rxs/rx1/Ez"], [[0, 1], [1, 2], [2, 3]])


def test_merge_rejects_non_time_domain_receiver_dataset(tmp_path):
    files = [tmp_path / "model1.h5", tmp_path / "model2.h5"]
    for filename in files:
        _write_output(filename)
        with h5py.File(filename, "a") as output:
            output["rxs/rx1"].create_dataset("S11", data=np.ones(3, dtype=np.complex128))

    with pytest.raises(ValueError, match="not a supported time-domain receiver component"):
        merge_files(files)


def test_merge_rejects_merged_file_as_an_ascan(tmp_path):
    files = [tmp_path / "model1.h5", tmp_path / "model2.h5"]
    _write_output(files[0])
    _write_output(files[1], offset=1)
    merged = merge_files(files)

    with pytest.raises(ValueError, match="original one-dimensional A-scan"):
        merge_files([merged], tmp_path / "merged_twice.h5")


def test_merge_rejects_mixed_precision(tmp_path):
    files = [tmp_path / "model1.h5", tmp_path / "model2.h5"]
    _write_output(files[0], dtype=np.float32)
    _write_output(files[1], offset=1, dtype=np.float64)

    with pytest.raises(ValueError, match="Inconsistent dtype"):
        merge_files(files)
