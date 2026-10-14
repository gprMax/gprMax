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

"""Regression tests for Metal solver throughput (gprMax issue #897).

The Metal solver committed every command buffer and immediately waited for
it, so each iteration made one blocking CPU-GPU round trip per update stage
and per PML slab. MetalCommandQueue.submit() now commits without waiting,
and MetalCommandQueue.synchronise() waits only before the host touches
device memory. MetalPML also launched one thread per cell of the whole grid
for every slab update; it now launches one thread per cell of the slab.

Like the other Metal dispatch tests, these use fake Metal objects.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax import config
from gprMax.pml import MetalPML
from gprMax.updates import metal_updates as metal_updates_module
from gprMax.updates.metal_updates import MetalCommandQueue, MetalUpdates

_SCHEDULED = 3
_COMPLETED = 4
_ERROR = 5


class _Command:
    def __init__(self, log, name, status=_SCHEDULED, fails=False):
        self.log = log
        self.name = name
        self._status = status
        self.fails = fails

    def commit(self):
        self.log.append(("commit", self.name))

    def waitUntilCompleted(self):
        self.log.append(("wait", self.name))
        self._status = _ERROR if self.fails else _COMPLETED

    def status(self):
        return self._status

    def error(self):
        return "GPU fault"


def test_submit_commits_without_waiting():
    log = []
    queue = MetalCommandQueue(SimpleNamespace())

    queue.submit(_Command(log, "a"))

    assert log == [("commit", "a")]


def test_synchronise_waits_for_every_submitted_command_in_order():
    log = []
    queue = MetalCommandQueue(SimpleNamespace())
    queue.submit(_Command(log, "a"))
    queue.submit(_Command(log, "b"))

    queue.synchronise()
    queue.synchronise()

    assert log == [("commit", "a"), ("commit", "b"), ("wait", "a"), ("wait", "b")]


def test_submit_releases_commands_the_gpu_has_completed():
    log = []
    queue = MetalCommandQueue(SimpleNamespace())
    queue.submit(_Command(log, "done", status=_COMPLETED))
    queue.submit(_Command(log, "pending"))

    queue.synchronise()

    assert ("wait", "done") not in log
    assert ("wait", "pending") in log


def test_synchronise_raises_for_a_failed_command():
    queue = MetalCommandQueue(SimpleNamespace())
    queue.submit(_Command([], "a", fails=True))

    with pytest.raises(RuntimeError, match="GPU fault"):
        queue.synchronise()


def test_submit_raises_for_a_failed_command_it_releases():
    queue = MetalCommandQueue(SimpleNamespace())

    with pytest.raises(RuntimeError, match="GPU fault"):
        queue.submit(_Command([], "a", status=_ERROR))


def test_command_buffer_passes_through_to_the_wrapped_queue():
    command = object()
    queue = MetalCommandQueue(SimpleNamespace(commandBuffer=lambda: command))

    assert queue.commandBuffer() is command


def test_finalise_synchronises_before_reading_receivers(monkeypatch):
    events = []
    updates = MetalUpdates.__new__(MetalUpdates)
    updates.cmdqueue = SimpleNamespace(synchronise=lambda: events.append("synchronise"))
    updates.grid = SimpleNamespace(rxs=[object()], magneticfrillsources=[], snapshots=[])
    updates.rxs_dev = updates.rxcoords_dev = object()
    updates.nrxcurrent = 0
    monkeypatch.setattr(
        metal_updates_module, "dtoh_rx_array", lambda *args: events.append("dtoh_rx_array")
    )

    updates.finalise()

    assert events == ["synchronise", "dtoh_rx_array"]


class _Encoder:
    def __init__(self):
        self.dispatch = None
        self.ended = False

    def setComputePipelineState_(self, pipeline):
        pass

    def setBytes_length_atIndex_(self, data, length, index):
        pass

    def setBuffer_offset_atIndex_(self, buffer, offset, index):
        pass

    def dispatchThreads_threadsPerThreadgroup_(self, grid_size, group_size):
        self.dispatch = (grid_size, group_size)

    def endEncoding(self):
        self.ended = True


def _metal_pml(log):
    """An x-slab PML whose PHI arrays have unequal shapes, on a 100^3 grid."""
    pml = MetalPML.__new__(MetalPML)
    pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf = 0, 10, 0, 100, 0, 100
    pml.thickness = 10
    pml.d = 0.002
    pml.pmc_mask = 0
    pml.EPhi1_shape = (1, 10, 100, 101)
    pml.EPhi2_shape = (1, 10, 101, 100)
    pml.HPhi1_shape = (1, 10, 101, 100)
    pml.HPhi2_shape = (1, 10, 100, 101)
    pml.ERA = np.zeros((2, 10))
    for name in (
        "EPhi1",
        "EPhi2",
        "ERA",
        "ERB",
        "ERE",
        "ERF",
        "HPhi1",
        "HPhi2",
        "HRA",
        "HRB",
        "HRE",
        "HRF",
    ):
        setattr(pml, f"{name}_dev", object())
    pml.psoE = pml.psoH = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: 1024)
    pml._electric_update_bounds = lambda: (0, 10, 0, 100, 0, 100)

    encoder = _Encoder()
    command = SimpleNamespace(computeCommandEncoder=lambda: encoder, commit=lambda: None)
    pml.queue = SimpleNamespace(
        commandBuffer=lambda: command, submit=lambda submitted: log.append(submitted)
    )
    pml.G = SimpleNamespace(
        nx=100,
        ny=100,
        nz=100,
        metal=SimpleNamespace(MTLSizeMake=lambda x, y, z: (x, y, z)),
        **{f"{name}_dev": object() for name in ("ID", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")},
    )
    return pml, encoder, command


@pytest.mark.parametrize("update", ["update_electric", "update_magnetic"])
def test_metal_pml_launches_one_thread_per_slab_cell(update, monkeypatch):
    monkeypatch.setattr(
        config, "sim_config", SimpleNamespace(dtypes={"float_or_double": np.float32})
    )
    submitted = []
    pml, encoder, command = _metal_pml(submitted)

    getattr(pml, update)()

    # The larger PHI array has 10 * 101 * 100 cells; the grid has 101^3.
    assert encoder.dispatch == ((10 * 101 * 100, 1, 1), (1024, 1, 1))
    assert encoder.ended
    assert submitted == [command]
