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

"""Tests for the -cpu_precision/-gpu_precision CLI args and cpu_precision=/
gpu_precision= Python API kwargs (gprMax/gprMax.py, gprMax/config.py).

Precision is fixed in SimulationConfig.__init__(), which runs before any
input-file/hash-command is ever read (config.sim_config = SimulationConfig(args)
happens at the top of run_main(), before context.run() gets anywhere near
parsing a Scene) - so this can only be a CLI/API-level setting, not a hash
command. These tests lock in the defaults and override behaviour at that
level.
"""
import argparse

import numpy as np
import pytest

import gprMax
import gprMax.config as config
from gprMax import gprMax as gprmax_mod


def _make_args(**overrides):
    args = argparse.Namespace(**gprmax_mod.args_defaults)
    args.inputfile = "test.in"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def mock_cuda_devices(monkeypatch):
    """Avoid requiring CUDA hardware for configuration-only tests."""

    monkeypatch.setattr(config, "detect_cuda_gpus", lambda: {0: object()})


def test_cpu_precision_defaults_to_single():
    sim_config = config.SimulationConfig(_make_args())
    assert sim_config.general["precision"] == "single"
    assert sim_config.dtypes["float_or_double"] is np.float32


def test_cpu_precision_can_be_raised_to_double():
    sim_config = config.SimulationConfig(_make_args(cpu_precision="double"))
    assert sim_config.general["precision"] == "double"
    assert sim_config.dtypes["float_or_double"] is np.float64


def test_gpu_precision_defaults_to_single(mock_cuda_devices):
    sim_config = config.SimulationConfig(_make_args(gpu=[0]))
    assert sim_config.general["solver"] == "cuda"
    assert sim_config.general["precision"] == "single"


def test_gpu_precision_can_be_raised_to_double(mock_cuda_devices):
    sim_config = config.SimulationConfig(_make_args(gpu=[0], gpu_precision="double"))
    assert sim_config.general["precision"] == "double"


def test_subgrid_always_forces_double_regardless_of_cpu_precision(caplog):
    sim_config = config.SimulationConfig(_make_args(subgrid=True, cpu_precision="single"))
    assert sim_config.general["precision"] == "double"
    assert any("overriding the requested single precision" in r.message for r in caplog.records)


def test_ratio_one_subgrid_inherits_requested_cpu_precision():
    scene = gprMax.Scene()
    scene.add(
        gprMax.SubGridHSG(
            p1=(0.01, 0.01, 0.01),
            p2=(0.02, 0.02, 0.02),
            ratio=1,
            id="local_grid",
        )
    )
    sim_config = config.SimulationConfig(
        _make_args(subgrid=True, scenes=[scene], cpu_precision="single")
    )
    assert sim_config.general["precision"] == "single"
    assert sim_config.dtypes["float_or_double"] is np.float32


def test_metal_solver_rejects_double_precision():
    """Apple GPU hardware and the Metal Shading Language have no native
    double type at all - unlike CUDA/OpenCL, double precision on Metal
    isn't just unimplemented, it's a hard platform constraint. Without
    this guard, requesting it would silently generate invalid Metal
    shader source that fails to compile, surfacing later as an opaque
    AttributeError rather than a clear diagnostic at startup."""
    with pytest.raises(ValueError):
        config.SimulationConfig(_make_args(metal=[0], gpu_precision="double"))


def test_metal_solver_allows_single_precision(monkeypatch):
    monkeypatch.setattr(config, "detect_metal", lambda: {})
    sim_config = config.SimulationConfig(_make_args(metal=[0], gpu_precision="single"))
    assert sim_config.general["solver"] == "metal"
    assert sim_config.general["precision"] == "single"


def test_invalid_cpu_precision_string_rejected():
    """The CLI is protected by argparse's choices=["single","double"] on
    -cpu_precision/-gpu_precision, but the Python API (cpu_precision=/
    gpu_precision= kwargs) bypasses argparse entirely and used to accept
    any string - SimulationConfig._set_precision() had no else branch, so
    an invalid value left self.dtypes completely unset, failing later
    with a confusing, unrelated AttributeError instead of a clear
    upfront error."""
    with pytest.raises(ValueError):
        config.SimulationConfig(_make_args(cpu_precision="triple"))


def test_invalid_gpu_precision_string_rejected(mock_cuda_devices):
    with pytest.raises(ValueError):
        config.SimulationConfig(_make_args(gpu=[0], gpu_precision="triple"))


def test_run_api_cpu_precision_kwarg_reaches_sim_config(tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.05, 0.05, 0.05)))
    scene.add(gprMax.TimeWindow(time=1e-12))

    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        cpu_precision="double",
        outputfile=tmp_path / "run",
        hide_progress_bars=True,
    )

    assert config.sim_config.general["precision"] == "double"
    assert config.sim_config.dtypes["float_or_double"] is np.float64
