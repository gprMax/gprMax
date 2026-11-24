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

"""Transient absorption windows: delayed signals, spatial weights and guards."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.hash_cmds_file import get_user_objects
from gprMax.model import Model
from gprMax.sar import RadiometryMonitor, SARMonitor, _window

from .test_radiometry import _lossy_scene


def _sample_two_cells(kind, window, *, count=101, source_index=10, arrivals=(20, 30)):
    """Two equal-magnitude transfer functions with distinct pure delays.

    Exercise the real sparse streaming DFT, collocation, source normalisation
    and absorption path. These are prescribed samples, not an FDTD solution.
    """
    monitor = kind.__new__(kind)
    monitor.grid_iterations = count
    monitor.grid_dt = 1e-12
    monitor.frequencies = np.asarray([1e9])
    monitor.complex_dtype = np.dtype(np.complex128)
    monitor.real_dtype = np.dtype(np.float64)
    monitor.nthreads = 1
    monitor.window = _window(window, count, monitor.real_dtype)
    monitor._source_samples = np.zeros(count)
    monitor._source_samples[source_index] = 1
    monitor._source_window = _window(window, count, monitor.real_dtype)
    monitor.target_amplitude = 1.0
    monitor.source_floor_db = -40.0
    monitor.mesh_valid = np.asarray([True])
    monitor.cells_per_wavelength = np.asarray([20.0])
    monitor.limiting_material = np.asarray(["tissue"])
    monitor.cells = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    monitor.cell_tag_ids = np.asarray([1, 1], dtype=np.uint8)
    monitor.cell_material_ids = np.asarray([2, 2], dtype=np.uint32)
    monitor.density = np.asarray([1000.0, 1000.0])
    monitor.cell_material_loss = np.asarray([[0.5, 0.5]])
    monitor.cell_edge_indices = {c: np.arange(8).reshape(2, 4) for c in ("Ex", "Ey", "Ez")}
    monitor.edge_flat_indices = {c: np.arange(8, dtype=np.int64) for c in monitor.cell_edge_indices}
    monitor.accumulators = {c: np.zeros((1, 8), dtype=np.complex128) for c in monitor.cell_edge_indices}
    monitor._phase = np.ones(1, dtype=np.complex128)
    monitor._phase_step = np.exp(-2j * np.pi * monitor.frequencies * monitor.grid_dt)
    monitor._next_iteration = 0
    monitor.averaging_masses = ()
    monitor.normalisation = "waveform"
    monitor.result = None
    zero = np.zeros((2, 2, 2))
    for iteration in range(count):
        ex = np.zeros(8)
        for cell, arrival in enumerate(arrivals):
            if iteration == arrival:
                ex[4 * cell : 4 * cell + 4] = 10
        monitor.observe_electric(iteration, ex.reshape(2, 2, 2), zero, zero)
    return monitor.finalise()


@pytest.mark.parametrize("kind", (SARMonitor, RadiometryMonitor))
@pytest.mark.parametrize("count", (101, 201))
@pytest.mark.parametrize("window", ("rectangular", "hann_end"))
def test_early_delayed_pulses_preserve_absorption_and_spatial_weights(kind, count, window):
    result = _sample_two_cells(kind, window, count=count)
    power = result.absorbed_power_density[0]
    np.testing.assert_allclose(power, [25, 25], rtol=1e-13)  # .5 * sigma * |E|^2
    np.testing.assert_allclose(power / power.sum(), [0.5, 0.5], rtol=1e-13)
    if kind is SARMonitor:
        np.testing.assert_allclose(result.sar, [[0.025, 0.025]], rtol=1e-13)
    else:
        np.testing.assert_allclose(result.normalised_absorption_density, [power], rtol=1e-13)


@pytest.mark.parametrize("kind", (SARMonitor, RadiometryMonitor))
def test_full_hann_remains_compatible_but_does_not_preserve_delayed_response(kind):
    result = _sample_two_cells(kind, "hann")
    weights = np.hanning(101)
    expected = 25 * (weights[[20, 30]] / weights[10]) ** 2
    np.testing.assert_allclose(result.absorbed_power_density[0], expected, rtol=1e-13)
    # A scalar re-normalisation cannot fix the altered spatial distribution.
    assert not np.allclose(expected / expected.sum(), [0.5, 0.5])


@pytest.mark.parametrize("kind", (SARMonitor, RadiometryMonitor))
def test_end_taper_still_suppresses_real_late_arrivals(kind):
    result = _sample_two_cells(kind, "hann_end", arrivals=(20, 90))
    np.testing.assert_allclose(result.absorbed_power_density[0], [25, 6.25], rtol=1e-13)


@pytest.mark.parametrize("kind", (SARMonitor, RadiometryMonitor))
def test_end_taper_preserves_time_zero_impulse_reference(kind):
    result = _sample_two_cells(kind, "hann_end", source_index=0)
    assert result.valid[0]
    np.testing.assert_allclose(result.absorbed_power_density, [[25, 25]], rtol=1e-13)


@pytest.mark.parametrize("kind", (gprMax.SAR, gprMax.Radiometry))
@pytest.mark.parametrize("normalisation", ("incident_power", "accepted_power"))
@pytest.mark.parametrize("window", ("hann", "hann_end", "hanning", "hann-end"))
def test_port_power_rejects_tapers_before_solving(tmp_path, monkeypatch, kind, normalisation, window):
    def unexpected_solve(*args, **kwargs):
        pytest.fail("unsupported port window reached time stepping")

    monkeypatch.setattr(Model, "solve", unexpected_solve)
    scene = _lossy_scene()
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="lossy"))
    next(o for o in scene.grid_objects if isinstance(o, gprMax.VoltageSource)).id = "feed"
    scene.add(
        kind(
            frequencies=[1e9], tags="target", window=window, normalisation=normalisation, port_id="feed", target_power=1
        )
    )
    with pytest.raises(ValueError, match="port-power normalisation requires a rectangular"):
        gprMax.run(scenes=[scene], outputfile=tmp_path / "invalid", hide_progress_bars=True)


@pytest.mark.parametrize("command", ("sar", "radiometry"))
def test_hash_commands_remain_rectangular(command):
    (output,) = get_user_objects([f"#{command}: 1e9 1e9 1 pulse 1 10 output target"], checkessential=False)
    assert output.window == "rectangular"


def test_decayed_lossy_target_preserves_sar_and_radiometry_with_end_taper(tmp_path):
    previous = None
    for time_ns in (4, 8):
        scene = _lossy_scene()
        scene.single_use_objects = [o for o in scene.single_use_objects if not isinstance(o, gprMax.TimeWindow)]
        scene.add(gprMax.TimeWindow(time=time_ns * 1e-9))
        scene.add(gprMax.MaterialDensity(density=1000, material_ids="lossy"))
        outputs = {}
        for window in ("rectangular", "hann", "hann_end"):
            for kind in (gprMax.SAR, gprMax.Radiometry):
                name = f"{kind.__name__.lower()}_{window}"
                output = kind(
                    frequencies=(0.75e9, 1e9, 1.25e9), tags="target", waveform_id="pulse", id=name, window=window
                )
                scene.add(output)
                outputs[name] = output
        path = tmp_path / f"lossy_{time_ns}ns"
        gprMax.run(scenes=[scene], outputfile=path, cpu_precision="double", hide_progress_bars=True)
        with h5py.File(path.with_suffix(".h5")) as data:
            for window in ("rectangular", "hann", "hann_end"):
                sar, rad = outputs[f"sar_{window}"].result, outputs[f"radiometry_{window}"].result
                assert np.all(sar.valid) and np.all(rad.valid)
                np.testing.assert_allclose(sar.sar * 1000, rad.absorbed_power_density, rtol=2e-14)
                for kind in ("sar", "radiometry"):
                    group = data[f"{kind}/{kind}_{window}"]
                    assert group.attrs["Window"] == window
                    if window == "hann_end":
                        assert group.attrs["WindowTaperFraction"] == 0.2
                    else:
                        assert "WindowTaperFraction" not in group.attrs
        rect = outputs["sar_rectangular"].result.sar.sum(axis=1)
        end = outputs["sar_hann_end"].result.sar.sum(axis=1)
        np.testing.assert_allclose(end, rect, rtol=5e-3)
        assert np.all(outputs["sar_hann"].result.sar.sum(axis=1) / rect > 1.01)
        if previous is not None:
            np.testing.assert_allclose(rect, previous, rtol=1e-3)
        previous = rect
