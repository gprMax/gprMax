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

"""Regression test for gprMax/contexts.py's TaskfarmContext device
assignment (Codex-reported, "Medium"): _create_model_config() only
overrode the per-worker device for solver == "cuda". OpenCL workers fell
through and kept whatever device ordinary ModelConfig chose (always the
first requested device - see project_python_api_device_selection_bug),
so every OpenCL task-farm worker contended for the same single device
instead of each getting its own, contradicting the documented
"OpenCL/MPI ... mixed mode" task-farm support (docs/source/
accelerators.rst).

Fixed by extending the existing per-rank indexing
(devices["devs"][self.rank - 1]) to also cover "opencl" - detect_opencl()
enumerates every visible device across all platforms into devices["devs"],
exactly like CUDA's detect_cuda_gpus(), so the same indexing scheme
applies safely.

Metal is deliberately NOT included: detect_metal() (gprMax/utilities/
host_info.py) only ever calls MTLCreateSystemDefaultDevice(), so
devices["devs"] always has exactly one entry (key 0) - there is no
multi-GPU enumeration to index into. Adding the same rank-based indexing
for Metal would raise KeyError for every worker rank beyond the first
instead of genuinely fixing anything, so it's left as a separate, deeper
gap rather than "fixed" in a way that would newly crash real task farms.

This test bypasses TaskfarmContext.__init__ (which needs a real MPI
communicator) via __new__, sets a fake rank directly, and monkeypatches
the parent class's _create_model_config to avoid needing a full
ModelConfig/SimulationConfig construction.
"""
import pytest

import gprMax.config as config
from gprMax.contexts import Context, TaskfarmContext


class _FakeModelConfig:
    def __init__(self):
        self.device = "unset"


@pytest.fixture
def taskfarm_ctx(monkeypatch):
    ctx = TaskfarmContext.__new__(TaskfarmContext)
    ctx.rank = 3  # worker rank 3 -> should select deviceID 2 (rank - 1)
    monkeypatch.setattr(Context, "_create_model_config", lambda self, model_num: _FakeModelConfig())
    return ctx


def _set_solver(monkeypatch, solver, devs):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": solver}
    config.sim_config.devices = {"devs": devs}


def test_opencl_worker_gets_device_by_rank(taskfarm_ctx, monkeypatch):
    _set_solver(monkeypatch, "opencl", devs={0: "dev0", 1: "dev1", 2: "dev2", 3: "dev3"})

    model_config = taskfarm_ctx._create_model_config(0)

    assert model_config.device == {"dev": "dev2", "deviceID": 2, "snapsgpu2cpu": False}


def test_cuda_worker_gets_device_by_rank(taskfarm_ctx, monkeypatch):
    _set_solver(monkeypatch, "cuda", devs={0: "dev0", 1: "dev1", 2: "dev2", 3: "dev3"})

    model_config = taskfarm_ctx._create_model_config(0)

    assert model_config.device == {"dev": "dev2", "deviceID": 2, "snapsgpu2cpu": False}


def test_metal_worker_left_untouched(taskfarm_ctx, monkeypatch):
    """Metal's device dict only ever has one entry (key 0) - confirm the
    guard deliberately does NOT try rank-based indexing (which would
    KeyError here), leaving whatever ordinary ModelConfig already set."""
    _set_solver(monkeypatch, "metal", devs={0: "dev0"})

    model_config = taskfarm_ctx._create_model_config(0)

    assert model_config.device == "unset"


def test_cpu_worker_left_untouched(taskfarm_ctx, monkeypatch):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": "cpu"}

    model_config = taskfarm_ctx._create_model_config(0)

    assert model_config.device == "unset"
