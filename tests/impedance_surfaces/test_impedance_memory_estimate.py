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

"""Structural memory accounting for local impedance-pole state."""

from types import SimpleNamespace

import numpy as np

from gprMax import config
from gprMax.grid.fdtd_grid import FDTDGrid


def _grid(monkeypatch, *, order: int | None) -> FDTDGrid:
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(dtypes={"float_or_double": np.dtype(np.float64)}),
    )
    grid = FDTDGrid()
    grid.size[:] = 4
    grid.set_pml_thickness(0)
    grid.solid = np.zeros((4, 4, 4), dtype=np.uint32)
    # Keep the dense-material contribution identical in the reference grid.
    grid.solid[1, 1, 1] = 1
    grid.materials = [SimpleNamespace(), SimpleNamespace()]
    if order is not None:
        grid.impedance_marker_models = {1: "wall"}
        grid.surface_impedance_models = {"wall": SimpleNamespace(order=order)}
    return grid


def _isolated_voxel_impedance_bytes(order: int) -> int:
    """Mirror the documented conservative isolated-voxel upper bound."""

    int32 = np.dtype(np.int32).itemsize
    int8 = np.dtype(np.int8).itemsize
    real = np.dtype(np.float64).itemsize
    edges = 12
    ports = 24
    owner = 4 * 4 * 4 * int32
    edge_records = edges * (24 * int32 + 9 * real)
    port_records_and_state = ports * (2 * int32 + 2 * int8 + (4 + order) * real)
    model_info = 2 * int32
    model_coefficients = (2 * max(1, order) + 1) * real
    state_sentinel = real if order == 0 else 0
    return (
        owner
        + edge_records
        + port_records_and_state
        + model_info
        + model_coefficients
        + state_sentinel
    )


def test_resistive_impedance_memory_includes_only_packed_sentinels(monkeypatch):
    baseline = _grid(monkeypatch, order=None)
    resistive = _grid(monkeypatch, order=0)

    assert resistive.mem_est_basic() - baseline.mem_est_basic() == (
        _isolated_voxel_impedance_bytes(0)
    )


def test_dynamic_impedance_memory_is_linear_in_port_and_model_poles(monkeypatch):
    baseline = _grid(monkeypatch, order=None)
    three_pole = _grid(monkeypatch, order=3)

    assert three_pole.mem_est_basic() - baseline.mem_est_basic() == (
        _isolated_voxel_impedance_bytes(3)
    )

    # Relative to order zero: one state for each of 24 estimated ports and
    # two shared model coefficients per added pole. The order-zero f/q
    # sentinels already account for one pole slot in each vector, while the
    # separate zero-order state sentinel is no longer needed.
    expected_growth = 24 * 3 * 8 + 2 * (3 - 1) * 8 - 8
    assert (
        _isolated_voxel_impedance_bytes(3) - _isolated_voxel_impedance_bytes(0) == expected_growth
    )
