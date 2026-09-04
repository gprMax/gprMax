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

"""Regression test for Metal's geometry_fixed stale-device-field bug
(Codex-reported).

MetalUpdates.update_magnetic()/update_electric_a()/store_outputs() used to
guard their host->device field uploads with `if not hasattr(self.grid,
"Ex_dev"): ...`. A fresh MetalUpdates instance is constructed for every model
run (gprMax/solvers.py), but with geometry_fixed=True the same grid object
(and any device buffers already attached to it) survives across runs, while
Model.reuse_geometry() only resets the *host* field arrays
(grid.reset_fields()). So on run 2+, the guard saw the still-present
Ex_dev/ID_dev attributes from run 1 and skipped the upload entirely -
run 2 silently resumed computation from run 1's final GPU field values
instead of the freshly-zeroed host arrays. CUDA/OpenCL never had this bug:
their equivalent htod_field_arrays()/htod_geometry_arrays() calls are
unconditional, inside _set_field_knls(), which runs once per fresh Updates
object regardless of what the grid already carries.

Fixed by moving the geometry/field/material uploads into
MetalUpdates._set_field_knls() unconditionally (matching CUDA/OpenCL's
placement exactly) and removing the per-iteration hasattr guards.

This test builds a fake grid that already carries Ex_dev/ID_dev (simulating
a reused geometry_fixed grid on run 2) and confirms _set_field_knls() still
uploads fresh geometry/field/material arrays regardless.
"""
import numpy as np

from gprMax import config
from gprMax.updates.metal_updates import MetalUpdates


class _FakePSO:
    def maxTotalThreadsPerThreadgroup(self):
        return 64


class _FakeLib:
    def newFunctionWithName_(self, name):
        return f"function:{name}"


class _FakeDevice:
    def newLibraryWithSource_options_error_(self, source, opts, error):
        return _FakeLib(), None

    def newComputePipelineStateWithFunction_error_(self, func, error):
        return (_FakePSO(), None)


class _FakeMetalModule:
    def MTLSizeMake(self, x, y, z):
        return (x, y, z)


def _make_grid_with_stale_device_buffers():
    class _Grid:
        nx = ny = nz = 4
        maxpoles = 0
        pmls = {"slabs": []}
        rxs = []
        voltagesources = hertziandipoles = magneticdipoles = []
        snapshots = []

        # Simulate a grid reused across geometry_fixed runs: it already
        # carries device buffer attributes from a previous run.
        ID_dev = "stale_ID_dev_from_run_1"
        Ex_dev = Ey_dev = Ez_dev = "stale_E_dev_from_run_1"
        Hx_dev = Hy_dev = Hz_dev = "stale_H_dev_from_run_1"

        def set_threads_per_thread_group(self):
            pass

        def set_thread_group_size(self, pso):
            pass

        def htod_geometry_arrays(self, dev):
            self.geometry_upload_calls = getattr(self, "geometry_upload_calls", 0) + 1

        def htod_field_arrays(self, dev):
            self.field_upload_calls = getattr(self, "field_upload_calls", 0) + 1

        def htod_material_arrays(self, dev):
            self.material_upload_calls = getattr(self, "material_upload_calls", 0) + 1

    return _Grid()


def test_set_field_knls_uploads_arrays_even_when_grid_already_has_device_buffers(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: type("_MC", (), {"materials": {"maxpoles": 0}})(),
    )

    updates = MetalUpdates.__new__(MetalUpdates)
    updates.dev = _FakeDevice()
    updates.opts = None
    updates.metal = _FakeMetalModule()
    updates.knl_common = ""
    updates.subs_func = {
        "REAL": "float",
        "CUDA_IDX": "",
        "NX_FIELDS": 5,
        "NY_FIELDS": 5,
        "NZ_FIELDS": 5,
        "NX_ID": 4,
        "NY_ID": 4,
        "NZ_ID": 4,
    }
    updates.subs_name_args = {"REAL": "float", "COMPLEX": "float"}

    grid = _make_grid_with_stale_device_buffers()
    updates.grid = grid

    updates._set_field_knls()

    assert grid.geometry_upload_calls == 1
    assert grid.field_upload_calls == 1
    assert grid.material_upload_calls == 1


def test_update_magnetic_no_longer_guards_upload_with_hasattr():
    import inspect

    from gprMax.updates import metal_updates

    source = inspect.getsource(metal_updates.MetalUpdates.update_magnetic)
    assert "hasattr" not in source


def test_update_electric_a_no_longer_guards_upload_with_hasattr():
    import inspect

    from gprMax.updates import metal_updates

    source = inspect.getsource(metal_updates.MetalUpdates.update_electric_a)
    assert "hasattr" not in source


def test_store_outputs_no_longer_guards_upload_with_hasattr():
    import inspect

    from gprMax.updates import metal_updates

    source = inspect.getsource(metal_updates.MetalUpdates.store_outputs)
    assert "hasattr" not in source
