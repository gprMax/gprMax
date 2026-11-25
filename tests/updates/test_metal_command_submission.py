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

Because work is now left in flight, a solver error must not abandon it:
a failed command buffer is reported only after the buffers behind it have
completed, and MetalUpdates.cleanup() drains the queue without replacing the
solver's exception.

Like the other Metal dispatch tests, these use fake Metal objects.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax import config, eigenmode_device
from gprMax.eigenmode_ports import DFT_PHASE_REANCHOR_INTERVAL
from gprMax.pml import MetalPML
from gprMax.solvers import Solver
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
        if self._status != _ERROR:
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


def test_synchronise_waits_for_commands_behind_a_failed_command():
    log = []
    queue = MetalCommandQueue(SimpleNamespace())
    queue.submit(_Command(log, "failed", fails=True))
    queue.submit(_Command(log, "pending"))

    with pytest.raises(RuntimeError, match="GPU fault"):
        queue.synchronise()

    assert log[2:] == [("wait", "failed"), ("wait", "pending")]
    assert not queue._submitted


def test_submit_waits_for_commands_behind_a_failed_command_before_raising():
    log = []
    queue = MetalCommandQueue(SimpleNamespace())
    failed = _Command(log, "failed")
    queue.submit(failed)
    queue.submit(_Command(log, "pending"))
    _fail_on_gpu(failed)

    with pytest.raises(RuntimeError, match="GPU fault"):
        queue.submit(_Command(log, "next"))

    assert log[3:] == [("wait", "failed"), ("wait", "pending"), ("wait", "next")]
    assert not queue._submitted


def test_drain_waits_for_every_command_and_logs_failures(caplog):
    log = []
    queue = MetalCommandQueue(SimpleNamespace())
    queue.submit(_Command(log, "failed", fails=True))
    queue.submit(_Command(log, "pending"))

    with caplog.at_level(logging.WARNING, logger=metal_updates_module.__name__):
        queue.drain()

    assert log[2:] == [("wait", "failed"), ("wait", "pending")]
    assert not queue._submitted
    assert "Metal command buffer failed: GPU fault" in caplog.text


def _fail_on_gpu(command):
    """Marks a submitted command buffer as failed by the GPU."""
    command._status = _ERROR


def _updates_with_pending_work(log, *commands):
    """A MetalUpdates whose solve submits commands and then reaches store_outputs."""
    updates = MetalUpdates.__new__(MetalUpdates)
    updates.cmdqueue = MetalCommandQueue(SimpleNamespace())

    def time_start():
        for command in commands:
            updates.cmdqueue.submit(command)

    updates.time_start = time_start
    return updates


@pytest.mark.parametrize("error", [ValueError("host failure"), KeyboardInterrupt()])
def test_cleanup_drains_pending_work_after_a_host_exception(error):
    log = []
    updates = _updates_with_pending_work(log, _Command(log, "a"), _Command(log, "b"))

    def store_outputs(iteration):
        raise error

    updates.store_outputs = store_outputs

    with pytest.raises(type(error)) as raised:
        Solver(updates).solve(range(3))

    assert raised.value is error
    assert log == [("commit", "a"), ("commit", "b"), ("wait", "a"), ("wait", "b")]


def test_cleanup_drains_a_failed_command_without_replacing_a_host_exception(caplog):
    log = []
    failed = _Command(log, "failed")
    updates = _updates_with_pending_work(log, failed, _Command(log, "pending"))
    error = ValueError("host failure")

    def store_outputs(iteration):
        _fail_on_gpu(failed)
        raise error

    updates.store_outputs = store_outputs

    with caplog.at_level(logging.WARNING, logger=metal_updates_module.__name__):
        with pytest.raises(ValueError) as raised:
            Solver(updates).solve(range(3))

    assert raised.value is error
    assert log[2:] == [("wait", "failed"), ("wait", "pending")]
    assert "Metal command buffer failed: GPU fault" in caplog.text


def test_a_failed_command_followed_by_pending_work_leaves_nothing_in_flight():
    log = []
    failed = _Command(log, "failed")
    updates = _updates_with_pending_work(log, failed, _Command(log, "pending"))

    def store_outputs(iteration):
        _fail_on_gpu(failed)
        updates.cmdqueue.submit(_Command(log, "next"))

    updates.store_outputs = store_outputs

    with pytest.raises(RuntimeError, match="Metal command buffer failed: GPU fault"):
        Solver(updates).solve(range(3))

    assert log == [
        ("commit", "failed"),
        ("commit", "pending"),
        ("commit", "next"),
        ("wait", "failed"),
        ("wait", "pending"),
        ("wait", "next"),
    ]
    assert not updates.cmdqueue._submitted


def test_cleanup_can_be_called_repeatedly(caplog):
    log = []
    failed = _Command(log, "failed")
    updates = _updates_with_pending_work(log, failed, _Command(log, "pending"))

    def store_outputs(iteration):
        _fail_on_gpu(failed)
        raise ValueError("host failure")

    updates.store_outputs = store_outputs

    with caplog.at_level(logging.WARNING, logger=metal_updates_module.__name__):
        with pytest.raises(ValueError, match="host failure"):
            Solver(updates).solve(range(3))
        updates.cleanup()
        updates.cleanup()

    assert log[2:] == [("wait", "failed"), ("wait", "pending")]
    assert caplog.text.count("Metal command buffer failed") == 1


def test_cleanup_without_a_command_queue_does_nothing():
    MetalUpdates.__new__(MetalUpdates).cleanup()


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


@pytest.mark.unit
@pytest.mark.parametrize("monitor_count", [1, 2])
def test_eigenmode_monitors_wait_only_before_phase_reanchors(monkeypatch, monitor_count):
    """Exercise two reset boundaries without Metal; this checks host scheduling."""
    monkeypatch.setattr(
        config, "sim_config", SimpleNamespace(dtypes={"float_or_double": np.float32})
    )
    monkeypatch.setattr(
        eigenmode_device, "upload_device_arrays", lambda arrays, *args, **kwargs: arrays
    )
    monitors = []
    for _ in range(monitor_count):
        monitor = SimpleNamespace(
            _next_iteration=0,
            owner=SimpleNamespace(
                normal_axis=2,
                direction="+",
                transverse_start=(0, 0),
                transverse_stop=(1, 1),
                plane_index=1,
            ),
            magnetic_side=0,
            measure=1e-6,
            handedness=1,
            electric_dft=np.zeros((2, 1), dtype=np.complex64),
            magnetic_dft=np.zeros((2, 1), dtype=np.complex64),
        )
        for name in (
            "electric_phase",
            "magnetic_phase",
            "phase_step",
            "conj_eu",
            "conj_ev",
            "conj_hu",
            "conj_hv",
        ):
            setattr(monitor, name, np.ones(2, dtype=np.complex64))
        eigenmode_device.prepare_device_eigenmode_monitor(monitor, "metal")
        monitors.append(monitor)

    log = []
    commands = []

    def command_buffer():
        command = _Command(log, len(commands))
        command.computeCommandEncoder = _Encoder
        commands.append(command)
        return command

    updates = MetalUpdates.__new__(MetalUpdates)
    updates.cmdqueue = MetalCommandQueue(SimpleNamespace(commandBuffer=command_buffer))
    updates.metal = SimpleNamespace(MTLSizeMake=lambda *args: args)
    updates.pso_eigenmode_dft = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: 64)
    updates.grid = SimpleNamespace(dt=1e-12, eigenmodeports=monitors)
    for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        setattr(updates.grid, f"{field}_dev", object())
    reanchors = []

    def reanchor(monitor, grid, backend):
        assert grid is updates.grid
        assert backend == "metal"
        assert not updates.cmdqueue._submitted
        assert all(command.status() == _COMPLETED for command in commands)
        reanchors.append(monitor._next_iteration)
        log.append(("reanchor", monitor._next_iteration))

    monkeypatch.setattr(metal_updates_module, "reanchor_device_eigenmode_monitor", reanchor)
    interval = DFT_PHASE_REANCHOR_INTERVAL
    for iteration in range(2 * interval + 2):
        log_start = len(log)
        updates.observe_eigenmode_ports(iteration)
        if (iteration + 1) % interval:
            assert [event for event, _ in log[log_start:]] == ["commit"] * monitor_count

    assert reanchors == [interval] * monitor_count + [2 * interval] * monitor_count
    assert all(monitor._next_iteration == 2 * interval + 2 for monitor in monitors)
    assert len(updates.cmdqueue._submitted) == 2 * monitor_count
    updates.cleanup()
    assert not updates.cmdqueue._submitted
    assert all(command.status() == _COMPLETED for command in commands)


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
