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

"""Regression tests for source and receiver parity across local solvers.

The tests replace ``config.sim_config`` so that command validation can be
checked without requiring accelerator hardware.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.user_objects.cmds_multiuse import Rx, TransmissionLine


def _set_solver(monkeypatch, solver):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": solver}
    config.sim_config.dtypes = {"float_or_double": np.float64}
    config.sim_config.em_consts = {"z0": 376.730313668}


@pytest.mark.parametrize("solver", ["cpu", "cuda", "opencl", "metal"])
def test_transmission_line_is_allowed_on_every_local_solver(monkeypatch, solver):
    _set_solver(monkeypatch, solver)
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(mode="3D"),
    )
    grid = SimpleNamespace(waveforms=[SimpleNamespace(ID="w")])
    tl = TransmissionLine(
        p1=(0.01, 0.01, 0.01), polarisation="x", resistance=50, waveform_id="w"
    )

    tl._validate_parameters(grid)


@pytest.mark.parametrize("solver", ["cuda", "opencl", "metal", "cpu"])
def test_current_output_allowed_on_every_solver(monkeypatch, solver):
    _set_solver(monkeypatch, solver)

    rx = Rx(p1=(0.01, 0.01, 0.01), id="r1", outputs=["Ix"])

    class _Grid:
        iterations = 5

    r = rx._create_receiver(grid=_Grid(), coord=np.array([1, 1, 1], dtype=np.int32))

    assert "Ix" in r.outputs
