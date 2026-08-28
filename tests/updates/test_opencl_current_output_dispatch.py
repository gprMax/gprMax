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

"""OpenCL receiver-current kernel construction and dispatch tests."""

from types import SimpleNamespace

import numpy as np

import gprMax.config as config
import gprMax.updates.opencl_updates as opencl_updates
from gprMax.updates.opencl_updates import OpenCLUpdates


def _set_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={
                "C_float_or_double": "double",
                "float_or_double": np.float64,
            },
            devices={"compiler_opts": []},
        ),
    )


def test_opencl_receiver_setup_builds_requested_current_kernel(monkeypatch):
    _set_config(monkeypatch)
    device_arrays = tuple(f"device:{name}" for name in ("coords", "fields", "info", "current"))
    monkeypatch.setattr(
        opencl_updates,
        "htod_rx_arrays",
        lambda grid, queue: device_arrays,
    )
    monkeypatch.setattr(
        opencl_updates,
        "requested_current_outputs",
        lambda grid: [(0, "Ix"), (0, "Iz")],
    )

    kernels = {}

    def fake_elementwise(context, arguments, body, name, **kwargs):
        kernels[name] = SimpleNamespace(arguments=arguments, body=body)
        return f"kernel:{name}"

    updates = OpenCLUpdates.__new__(OpenCLUpdates)
    updates.grid = SimpleNamespace()
    updates.queue = object()
    updates.ctx = object()
    updates.knl_common = ""
    updates.elwiseknl = fake_elementwise
    updates._set_rx_knl()

    assert updates.nrxcurrent == 2
    assert set(kernels) == {"store_outputs", "store_current_outputs"}
    assert updates.store_current_outputs_dev == "kernel:store_current_outputs"
    assert "$" not in kernels["store_current_outputs"].arguments
    assert "$" not in kernels["store_current_outputs"].body


def test_opencl_store_outputs_dispatches_requested_currents(monkeypatch):
    _set_config(monkeypatch)
    field_call = []
    current_call = []

    updates = OpenCLUpdates.__new__(OpenCLUpdates)
    updates.nrxcurrent = 2
    updates.rxcoords_dev = "rxcoords"
    updates.rxs_dev = "rxfields"
    updates.rxcurrentinfo_dev = "rxcurrentinfo"
    updates.rxcurrents_dev = "rxcurrents"
    updates.store_outputs_dev = lambda *args: field_call.append(args)
    updates.store_current_outputs_dev = lambda *args: current_call.append(args)
    updates.grid = SimpleNamespace(
        rxs=[object()],
        dx=0.1,
        dy=0.2,
        dz=0.3,
        Ex_dev="Ex",
        Ey_dev="Ey",
        Ez_dev="Ez",
        Hx_dev="Hx",
        Hy_dev="Hy",
        Hz_dev="Hz",
    )

    updates.store_outputs(7)

    assert len(field_call) == 1
    assert len(current_call) == 1
    arguments = current_call[0]
    assert tuple(int(value) for value in arguments[:2]) == (2, 7)
    assert arguments[2:4] == ("rxcurrentinfo", "rxcurrents")
    assert arguments[4:7] == (np.float64(0.1), np.float64(0.2), np.float64(0.3))
    assert arguments[7:] == ("Hx", "Hy", "Hz")
