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

"""Real CPU/Metal parity for snapshot timing and field values.

The Metal solver submits command buffers without waiting for them, so the
host must synchronise before it copies snapshots from the device, and a model
that reuses its geometry must start from freshly uploaded state. These tests
cover both snapshot transfer modes and stepped geometry_fixed runs.
"""

import logging

import h5py
import numpy as np
import pytest

import gprMax
from gprMax import config
from gprMax.updates.metal_updates import MetalUpdates

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

try:
    import Metal

    HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS_METAL = False


def _scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.TimeWindow(iterations=100))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(
            p1=(0.010, 0.010, 0.010),
            polarisation="z",
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.Snapshot(
            p1=(0.004, 0.004, 0.004),
            p2=(0.016, 0.016, 0.016),
            dl=(0.002, 0.002, 0.002),
            filename="fields",
            iterations=80,
            fileext=".h5",
        )
    )
    return scene


@pytest.mark.skipif(not HAS_METAL, reason="No Apple Metal device/PyObjC available")
def test_metal_snapshot_matches_cpu_at_the_same_electric_timestep(tmp_path):
    cpu_path = tmp_path / "cpu"
    metal_path = tmp_path / "metal"
    common = dict(n=1, hide_progress_bars=True, log_level=logging.WARNING)

    gprMax.run(
        scenes=[_scene()],
        outputfile=cpu_path,
        cpu_precision="double",
        **common,
    )
    gprMax.run(
        scenes=[_scene()],
        outputfile=metal_path,
        metal=True,
        gpu_precision="single",
        **common,
    )

    with h5py.File(tmp_path / "cpu_snaps" / "fields.h5", "r") as cpu_file, h5py.File(
        tmp_path / "metal_snaps" / "fields.h5", "r"
    ) as metal_file:
        assert cpu_file.attrs["iteration"] == metal_file.attrs["iteration"] == 80
        assert cpu_file.attrs["time"] == pytest.approx(metal_file.attrs["time"])
        assert cpu_file.attrs["magnetic_time"] == pytest.approx(
            metal_file.attrs["magnetic_time"]
        )

        cpu = np.concatenate([cpu_file[name][...].ravel() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")])
        metal = np.concatenate(
            [metal_file[name][...].ravel() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
        )

    scale = np.max(np.abs(cpu))
    assert scale > 1e-6
    assert np.isfinite(metal).all()
    assert np.max(np.abs(metal - cpu)) / scale < 2e-4
    assert np.linalg.norm(metal - cpu) / np.linalg.norm(cpu) < 2e-4


_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_SNAPSHOT_ITERATIONS = (10, 35, 60, 85)
_STEP = 0.002


def _ordinary_source_scene(offset=0.0, steps=False):
    """Dipole and receiver beside a dielectric block, with several snapshots."""

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.024, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.TimeWindow(iterations=100))
    scene.add(gprMax.Material(er=4, se=0.01, mr=1, sm=0, id="block"))
    scene.add(gprMax.Box(p1=(0.004, 0.004, 0.004), p2=(0.02, 0.016, 0.008), material_id="block"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(
            p1=(0.008 + offset, 0.010, 0.010), polarisation="z", waveform_id="pulse"
        )
    )
    scene.add(gprMax.Rx(p1=(0.012 + offset, 0.010, 0.010)))
    if steps:
        scene.add(gprMax.SrcSteps(p1=(_STEP, 0, 0)))
        scene.add(gprMax.RxSteps(p1=(_STEP, 0, 0)))
    for iteration in _SNAPSHOT_ITERATIONS:
        scene.add(
            gprMax.Snapshot(
                p1=(0.004, 0.004, 0.004),
                p2=(0.02, 0.016, 0.016),
                dl=(0.002, 0.002, 0.002),
                filename=f"fields{iteration}",
                iterations=iteration,
                fileext=".h5",
            )
        )
    return scene


def _outputs(path):
    """Receiver traces and, per snapshot, its timestamps and six fields."""

    with h5py.File(path.with_name(path.name + ".h5"), "r") as output:
        traces = {name: output[f"rxs/rx1/{name}"][...] for name in _FIELDS}
    snapshots = {}
    for iteration in _SNAPSHOT_ITERATIONS:
        with h5py.File(path.with_name(path.name + "_snaps") / f"fields{iteration}.h5") as snap:
            times = {name: snap.attrs[name] for name in ("iteration", "time", "magnetic_time")}
            snapshots[iteration] = times, {name: snap[name][...] for name in _FIELDS}
    return traces, snapshots


def _assert_identical(actual, expected):
    actual_traces, actual_snapshots = actual
    expected_traces, expected_snapshots = expected
    for name in _FIELDS:
        np.testing.assert_array_equal(actual_traces[name], expected_traces[name], err_msg=name)
    for iteration in _SNAPSHOT_ITERATIONS:
        actual_times, actual_fields = actual_snapshots[iteration]
        expected_times, expected_fields = expected_snapshots[iteration]
        assert actual_times == expected_times
        for name in _FIELDS:
            np.testing.assert_array_equal(
                actual_fields[name], expected_fields[name], err_msg=f"{iteration} {name}"
            )


def _assert_close(metal, cpu):
    metal_traces, metal_snapshots = metal
    cpu_traces, cpu_snapshots = cpu
    groups = [(metal_traces, cpu_traces)]
    for iteration in _SNAPSHOT_ITERATIONS:
        metal_times, metal_fields = metal_snapshots[iteration]
        cpu_times, cpu_fields = cpu_snapshots[iteration]
        assert metal_times["iteration"] == cpu_times["iteration"] == iteration
        assert metal_times["time"] == pytest.approx(cpu_times["time"])
        assert metal_times["magnetic_time"] == pytest.approx(cpu_times["magnetic_time"])
        groups.append((metal_fields, cpu_fields))
    for metal_values, cpu_values in groups:
        metal_all = np.concatenate([metal_values[name].ravel() for name in _FIELDS])
        cpu_all = np.concatenate([cpu_values[name].ravel() for name in _FIELDS])
        scale = np.max(np.abs(cpu_all))
        assert scale > 1e-6
        assert np.isfinite(metal_all).all()
        assert np.max(np.abs(metal_all - cpu_all)) / scale < 2e-4


@pytest.fixture
def snapshot_read_backs(monkeypatch):
    """Records how many snapshot slots each device read-back copies."""

    read_back = MetalUpdates._metal_snapshot_buffers_to_numpy
    slots = []

    def recording_read_back(self):
        arrays = read_back(self)
        slots.append(arrays[0].shape[0])
        return arrays

    monkeypatch.setattr(MetalUpdates, "_metal_snapshot_buffers_to_numpy", recording_read_back)
    return slots


def _copy_snapshots_during_the_run(monkeypatch):
    """Selects the snapsgpu2cpu path, which Metal's unified memory never needs."""

    initialise = config.ModelConfig.__init__

    def __init__(self, *args, **kwargs):
        initialise(self, *args, **kwargs)
        self.device["snapsgpu2cpu"] = True

    monkeypatch.setattr(config.ModelConfig, "__init__", __init__)


_METAL = dict(
    metal=True, gpu_precision="single", hide_progress_bars=True, log_level=logging.WARNING
)


@pytest.mark.skipif(not HAS_METAL, reason="No Apple Metal device/PyObjC available")
def test_metal_snapshot_transfer_modes_match_each_other_and_cpu(
    tmp_path, monkeypatch, snapshot_read_backs
):
    gprMax.run(
        scenes=[_ordinary_source_scene()],
        outputfile=tmp_path / "cpu",
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=logging.WARNING,
    )
    gprMax.run(scenes=[_ordinary_source_scene()], outputfile=tmp_path / "kept", **_METAL)
    # Kept on the device, all snapshots are copied back together at the end.
    assert snapshot_read_backs == [len(_SNAPSHOT_ITERATIONS)]

    _copy_snapshots_during_the_run(monkeypatch)
    gprMax.run(scenes=[_ordinary_source_scene()], outputfile=tmp_path / "copied", **_METAL)
    # With snapsgpu2cpu, each snapshot is copied back when it is taken.
    assert snapshot_read_backs[1:] == [1] * len(_SNAPSHOT_ITERATIONS)

    kept = _outputs(tmp_path / "kept")
    _assert_identical(_outputs(tmp_path / "copied"), kept)
    _assert_close(kept, _outputs(tmp_path / "cpu"))


@pytest.mark.skipif(not HAS_METAL, reason="No Apple Metal device/PyObjC available")
@pytest.mark.parametrize("snapsgpu2cpu", [False, True])
def test_metal_geometry_fixed_steps_match_independently_built_models(
    tmp_path, monkeypatch, snapshot_read_backs, snapsgpu2cpu
):
    if snapsgpu2cpu:
        _copy_snapshots_during_the_run(monkeypatch)
    gprMax.run(
        scenes=[_ordinary_source_scene(steps=True)],
        n=3,
        geometry_fixed=True,
        outputfile=tmp_path / "stepped",
        **_METAL,
    )
    for model in range(3):
        gprMax.run(
            scenes=[_ordinary_source_scene(offset=model * _STEP)],
            outputfile=tmp_path / f"rebuilt{model}",
            **_METAL,
        )
    slots = [1] * len(_SNAPSHOT_ITERATIONS) if snapsgpu2cpu else [len(_SNAPSHOT_ITERATIONS)]
    assert snapshot_read_backs == slots * 6

    stepped = [_outputs(tmp_path / f"stepped{model + 1}") for model in range(3)]
    for model in range(3):
        _assert_identical(stepped[model], _outputs(tmp_path / f"rebuilt{model}"))
    # Each step moved the source, so no model may repeat another's output.
    peaks = {float(np.max(np.abs(traces["Ez"]))) for traces, _ in stepped}
    assert len(peaks) == 3
