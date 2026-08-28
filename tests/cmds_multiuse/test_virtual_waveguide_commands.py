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

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.hash_cmds_multiuse import process_multicmds
from gprMax.user_objects.cmds_multiuse import EigenmodeExcitation, VirtualWaveguide


def test_hash_virtual_waveguide_parses_defaults_and_custom_profile():
    commands = defaultdict(lambda: None)
    commands["#eigenmode_band"] = ["wg 4e9 6e9 21"]
    commands["#eigenmode_port"] = [
        "1 0.01 0.005 0 0.01 0.035 0.01 + 1 auto",
        "2 0.09 0.005 0 0.09 0.035 0.01 - 1 auto",
    ]
    commands["#virtual_waveguide"] = ["1", "2 40 10 5 quiet"]
    commands["#eigenmode_excitation"] = ["1 1 auto"]

    objects = process_multicmds(commands)

    guides = [obj for obj in objects if isinstance(obj, VirtualWaveguide)]
    assert guides[0].kwargs == {
        "port": 1,
        "length_cells": 30,
        "pml_cells": 12,
        "source_clearance_cells": 6,
        "pml_profile": None,
    }
    assert guides[1].kwargs == {
        "port": 2,
        "length_cells": 40,
        "pml_cells": 10,
        "source_clearance_cells": 5,
        "pml_profile": "quiet",
    }


def test_hash_allows_passive_ports_only_when_each_has_virtual_waveguide():
    commands = defaultdict(lambda: None)
    commands["#eigenmode_band"] = ["wg 4e9 6e9 21"]
    commands["#eigenmode_port"] = [
        "1 0.01 0.005 0 0.01 0.035 0.01 + 1 auto",
        "2 0.09 0.005 0 0.09 0.035 0.01 - 1 auto",
    ]
    commands["#virtual_waveguide"] = ["1", "2"]

    objects = process_multicmds(commands)

    assert len([obj for obj in objects if isinstance(obj, VirtualWaveguide)]) == 2
    assert not any(isinstance(obj, EigenmodeExcitation) for obj in objects)

    commands["#virtual_waveguide"] = ["1"]
    with pytest.raises(ValueError, match="every port has a passive"):
        process_multicmds(commands)


def test_virtual_waveguide_python_api_rejects_fractional_cell_counts(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(general={"solver": "cpu"}, mpi=False),
    )
    grid = FDTDGrid()
    grid.eigenmodeportdefs[1] = object()

    with pytest.raises(ValueError, match="length_cells must be an integer"):
        VirtualWaveguide(port=1, length_cells=np.float64(30.5)).build(grid)


def test_virtual_waveguide_rejects_short_guide_before_modal_solve(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(general={"solver": "cpu"}, mpi=False),
    )
    grid = FDTDGrid()
    grid.eigenmodeportdefs[1] = object()

    with pytest.raises(ValueError, match="length_cells must be at least"):
        VirtualWaveguide(
            port=1,
            length_cells=20,
            pml_cells=12,
            source_clearance_cells=6,
        ).build(grid)

    with pytest.raises(ValueError, match="unknown PML profile"):
        VirtualWaveguide(port=1, pml_profile="missing").build(grid)
