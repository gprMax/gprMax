# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
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
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Tests for plotting authoritative RxPort output."""

import h5py
import numpy as np
import pytest

from toolboxes.Plotting.plot_antenna_params import discover_terminal_outputs, read_port_params


def _write_port(output, port_id, offset=0):
    port = output.require_group(f"ports/{port_id}")
    frequency = np.array([0, 1e9, 2e9], dtype=np.float32)
    s11 = np.array([0.5, 0.2 + 0.1j, 0.1], dtype=np.complex64) + offset
    zin = np.array([10, 50 + 5j, 70], dtype=np.complex64) + offset
    port.create_dataset("frequency", data=frequency)
    port.create_dataset("S11", data=s11)
    port.create_dataset("Zin", data=zin)
    port.create_dataset("Yin", data=1 / zin)
    port.create_dataset("valid_S11", data=np.array([0, 1, 1], dtype=np.uint8))
    port.create_dataset("valid_Zin", data=np.array([0, 1, 0], dtype=np.uint8))
    port.create_dataset("valid_Yin", data=np.array([0, 1, 0], dtype=np.uint8))
    port.create_dataset("time", data=np.array([0, 1e-9, 2e-9], dtype=np.float32))
    port.create_dataset("Vgenerator", data=np.array([0, 1, 0], dtype=np.float32))
    port.create_dataset("Vtotal", data=np.array([0, 0.8, 0.1], dtype=np.float32))
    port.create_dataset("Vincident_spectrum", data=np.ones(3, dtype=np.complex64))
    port.create_dataset("Vtotal_spectrum", data=np.full(3, 0.8, dtype=np.complex64))
    port.attrs["ReferenceImpedance"] = 50.0
    port.attrs["SourceType"] = "VoltageSource"


def _write_transmission_line(output):
    line = output.require_group("tls/tl1")
    frequency = np.array([0, 1e9, 2e9], dtype=np.float32)
    line.create_dataset("frequency", data=frequency)
    line.create_dataset("S11", data=np.full(3, 0.2, dtype=np.complex64))
    line.create_dataset("Zin", data=np.full(3, 75 + 5j, dtype=np.complex64))
    line.create_dataset("Yin", data=np.full(3, 1 / (75 + 5j), dtype=np.complex64))
    for name in ("valid_S11", "valid_Zin", "valid_Yin"):
        line.create_dataset(name, data=np.ones(3, dtype=np.uint8))
    line.create_dataset("time_voltage", data=np.array([0, 1e-9, 2e-9]))
    line.create_dataset("time_current", data=np.array([-0.5e-9, 0.5e-9, 1.5e-9]))
    line.create_dataset("Vinc", data=np.array([0, 1, 0], dtype=np.float32))
    line.create_dataset("Vtotal", data=np.array([0, 0.9, 0.1], dtype=np.float32))
    line.create_dataset("Iinc", data=np.array([0, 0.02, 0], dtype=np.float32))
    line.create_dataset("Itotal", data=np.array([0, 0.018, 0.002], dtype=np.float32))
    line.create_dataset("Vincident_spectrum", data=np.ones(3, dtype=np.complex64))
    line.create_dataset("Iincident_spectrum", data=np.ones(3, dtype=np.complex64))
    line.attrs["ReferenceImpedance"] = 50.0


def test_read_port_params_selects_only_port_and_preserves_stored_data(tmp_path):
    filename = tmp_path / "single.h5"
    with h5py.File(filename, "w") as output:
        _write_port(output, "feed")

    result = read_port_params(filename)

    assert result["port_id"] == "feed"
    assert result["reference_impedance"] == 50
    np.testing.assert_array_equal(result["frequency"], [0, 1e9, 2e9])
    np.testing.assert_array_equal(result["valid_s11"], [False, True, True])
    np.testing.assert_array_equal(result["valid_zin"], [False, True, False])
    assert [trace["label"] for trace in result["time_traces"]["voltage"]] == [
        "Generator voltage",
        "Total voltage",
    ]
    assert result["time_traces"]["current"] == []
    assert result["source_type"] == "VoltageSource"


def test_read_port_params_requires_selection_for_multiple_ports(tmp_path):
    filename = tmp_path / "multiple.h5"
    with h5py.File(filename, "w") as output:
        _write_port(output, "feed1")
        _write_port(output, "feed2", offset=1)

    with pytest.raises(ValueError, match="multiple terminal outputs"):
        read_port_params(filename)

    result = read_port_params(filename, "feed2")
    assert result["port_id"] == "feed2"
    np.testing.assert_allclose(result["s11"], [1.5, 1.2 + 0.1j, 1.1])


def test_read_port_params_rejects_file_without_terminal_outputs(tmp_path):
    filename = tmp_path / "empty.h5"
    with h5py.File(filename, "w"):
        pass

    with pytest.raises(ValueError, match="does not contain terminal"):
        read_port_params(filename)


def test_reader_adapts_to_transmission_line_voltage_and_current(tmp_path):
    filename = tmp_path / "line.h5"
    with h5py.File(filename, "w") as output:
        _write_transmission_line(output)

    result = read_port_params(filename)

    assert result["port_path"] == "tls/tl1"
    assert result["source_type"] == "Transmission line"
    assert len(result["time_traces"]["voltage"]) == 2
    assert len(result["time_traces"]["current"]) == 2
    assert len(result["spectral_traces"]["voltage"]) == 1
    assert len(result["spectral_traces"]["current"]) == 1


def test_discovery_includes_subgrid_ports(tmp_path):
    filename = tmp_path / "subgrid.h5"
    with h5py.File(filename, "w") as output:
        port = output.require_group("subgrids/fine_grid/ports/feed")
        for name in ("frequency", "S11", "Zin", "Yin"):
            port.create_dataset(name, data=np.ones(2))

    assert discover_terminal_outputs(filename) == ["subgrids/fine_grid/ports/feed"]
