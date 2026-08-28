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

"""Validation of optional start/stop time pairs for conventional sources."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
import gprMax.config as config
from gprMax.user_objects.cmds_multiuse import _configure_dpw_time_window


@pytest.fixture(autouse=True)
def source_config(monkeypatch):
    monkeypatch.setattr(config, "sim_config", SimpleNamespace())
    config.sim_config.general = {"solver": "cpu", "subgrid": False}
    config.sim_config.mpi = False
    config.sim_config.dtypes = {"float_or_double": np.float64}
    config.sim_config.em_consts = {"z0": 376.730313668}
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D"))


def _grid():
    return SimpleNamespace(waveforms=[SimpleNamespace(ID="pulse")])


@pytest.mark.parametrize(
    "source",
    (
        gprMax.VoltageSource(
            p1=(0, 0, 0),
            polarisation="z",
            resistance=50,
            waveform_id="pulse",
            start=0,
        ),
        gprMax.HertzianDipole(
            p1=(0, 0, 0), polarisation="z", waveform_id="pulse", stop=1e-9
        ),
        gprMax.MagneticDipole(
            p1=(0, 0, 0), polarisation="z", waveform_id="pulse", start=0
        ),
        gprMax.TransmissionLine(
            p1=(0, 0, 0),
            polarisation="z",
            resistance=50,
            waveform_id="pulse",
            stop=1e-9,
        ),
        gprMax.MagneticFrillSource(
            p1=(0, 0, 0),
            polarisation="z",
            zcoax=50,
            waveform_id="pulse",
            start=0,
        ),
    ),
)
def test_lone_start_or_stop_is_rejected_instead_of_silently_ignored(source):
    with pytest.raises(ValueError, match="start and stop times must be supplied together"):
        source._validate_parameters(_grid())


@pytest.mark.parametrize(
    "source",
    (
        gprMax.DiscretePlaneWaveAngles(start=0),
        gprMax.DiscretePlaneWaveVector(stop=1e-9),
        gprMax.DiscretePlaneWaveAxial(start=0),
    ),
)
def test_plane_wave_lone_start_or_stop_is_rejected(source):
    runtime_source = SimpleNamespace()
    grid = SimpleNamespace(timewindow=2e-9)
    with pytest.raises(ValueError, match="start and stop times must be supplied together"):
        _configure_dpw_time_window(source, runtime_source, grid)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_plane_wave_non_finite_time_is_rejected(value):
    source = gprMax.DiscretePlaneWaveAxial(start=0, stop=value)
    with pytest.raises(ValueError, match="must be finite"):
        _configure_dpw_time_window(source, SimpleNamespace(), SimpleNamespace(timewindow=2e-9))
