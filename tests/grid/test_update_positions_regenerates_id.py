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

"""Regression test for FDTDGrid._update_positions() not regenerating .ID:

A source/receiver's auto-generated .ID (e.g. "Rx(40,85,0)", set once in
cmds_multiuse.py's _create_receiver/_create_hertzian_dipole/
_create_magnetic_dipole from the pre-step point) is never refreshed when
_update_positions() moves the item for a later model run in a #src_steps/
#rx_steps scan. .coord is updated correctly every run; .ID stays frozen at
the model-1 position for the rest of the scan. Confirmed live: running
examples/gpr/basic/cylinder_Bscan_2D.in with -n 10 produces 10 output
files whose rxs/rx1 GridPosition correctly steps 40->49 but whose Name
attribute reads "Rx(40,85,0)" in every single one of them.

Receivers distinguish an auto-generated .ID from a user-supplied one via
.name_kind ("generated" vs "user", set in RxUser.build()); simple sources
(HertzianDipole/MagneticDipole) have no naming option at all and are
always auto-named, so they carry no .name_kind attribute - the fix's
`getattr(item, "name_kind", "generated")` must default to "generated" for
them, not skip the regeneration.

Uses the same minimal FDTDGrid/_FakeItem pattern as
test_src_rx_steps_validation.py in this directory.
"""
import numpy as np
import pytest

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid


class _FakeSource:
    """A simple source: always auto-named, no .name_kind at all."""

    def __init__(self, coord):
        self.coord = np.array(coord, dtype=np.int32)
        self.coordorigin = self.coord.copy()
        self.ID = f"{self.__class__.__name__}({coord[0]},{coord[1]},{coord[2]})"


class _FakeGeneratedReceiver:
    """A receiver with no user-supplied name (RxUser.build() -> 'generated')."""

    def __init__(self, coord):
        self.coord = np.array(coord, dtype=np.int32)
        self.coordorigin = self.coord.copy()
        self.ID = f"{self.__class__.__name__}({coord[0]},{coord[1]},{coord[2]})"
        self.name_kind = "generated"


class _FakeUserNamedReceiver:
    """A receiver the user gave an explicit name to (RxUser.build() -> 'user')."""

    def __init__(self, coord, user_id):
        self.coord = np.array(coord, dtype=np.int32)
        self.coordorigin = self.coord.copy()
        self.ID = user_id
        self.name_kind = "user"


def _make_grid(nx=100, ny=100, nz=100):
    grid = FDTDGrid.__new__(FDTDGrid)
    grid.size = np.array([nx, ny, nz], dtype=np.int32)
    return grid


@pytest.fixture(autouse=True)
def _fake_sim_config(monkeypatch):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.model_end = 5
    config.sim_config.model_start = 0


def test_generated_receiver_id_tracks_stepped_position():
    grid = _make_grid()
    item = _FakeGeneratedReceiver((40, 85, 0))
    assert item.ID == "_FakeGeneratedReceiver(40,85,0)"

    step_size = np.array([1, 0, 0], dtype=np.int32)
    grid._update_positions([item], step_size, 0)
    assert item.ID == "_FakeGeneratedReceiver(40,85,0)"  # model 0: unmoved

    grid._update_positions([item], step_size, 3)
    assert tuple(item.coord) == (43, 85, 0)
    assert item.ID == "_FakeGeneratedReceiver(43,85,0)", (
        "a receiver with no user-supplied name must have its auto-generated "
        "ID track its current stepped position, not stay frozen at model 1's"
    )


def test_user_named_receiver_id_is_never_overwritten():
    grid = _make_grid()
    item = _FakeUserNamedReceiver((40, 85, 0), user_id="my_receiver")
    step_size = np.array([1, 0, 0], dtype=np.int32)

    grid._update_positions([item], step_size, 3)
    assert tuple(item.coord) == (43, 85, 0)
    assert item.ID == "my_receiver", (
        "a receiver the user explicitly named must keep that name "
        "regardless of stepping - only an auto-generated ID is refreshed"
    )


def test_simple_source_id_tracks_stepped_position_with_no_name_kind_attribute():
    # Sources have no naming option at all, so they carry no .name_kind
    # attribute whatsoever. The fix must still treat them as "generated"
    # by default rather than silently skipping the regeneration.
    grid = _make_grid()
    item = _FakeSource((10, 20, 0))
    assert not hasattr(item, "name_kind")

    step_size = np.array([2, 0, 0], dtype=np.int32)
    grid._update_positions([item], step_size, 2)
    assert tuple(item.coord) == (14, 20, 0)
    assert item.ID == "_FakeSource(14,20,0)"
