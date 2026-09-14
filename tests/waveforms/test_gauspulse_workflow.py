"""Exercise the documented gauspulse input, source/port histories and plots."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.signal import gausspulse

import gprMax
from examples.features.waveforms.modulated_gaussian import plot_and_export, sample_waveform

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "examples/features/waveforms/gauspulse.in"
COMMON = dict(hide_progress_bars=True, log_level=30, cpu_precision="double")


def _scene(reference=False, resistance=50):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.06,) * 3))
    scene.add(gprMax.Discretisation(p1=(0.002,) * 3))
    scene.add(gprMax.TimeWindow(time=6e-9))
    scene.add(gprMax.PMLThickness(thickness=6))
    scene.add(gprMax.OMPThreads(n=1))
    if reference:
        delay = gausspulse("cutoff", fc=1e9)
        scene.add(
            gprMax.Waveform(
                wave_type="user",
                id="pulse",
                user_func=lambda t: float(gausspulse(t - delay, fc=1e9)),
            )
        )
    else:
        scene.add(gprMax.Waveform(wave_type="gauspulse", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.VoltageSource(
            p1=(0.03,) * 3,
            polarisation="z",
            resistance=resistance,
            waveform_id="pulse",
            start=0,
            stop=6e-9,
            id="feed",
            spectrum_limit=10,
        )
    )
    scene.add(gprMax.Rx(p1=(0.034, 0.03, 0.03), id="probe", outputs=["Ez"]))
    return scene


def _compare(first, second, tolerance=2e-12):
    with h5py.File(first) as a, h5py.File(second) as b:
        for path in ("rxs/rx1/Ez", "ports/feed/Vtotal", "ports/feed/Vgenerator"):
            av, bv = a[path][...], b[path][...]
            assert np.isfinite(av).all() and np.max(np.abs(av)) > 1e-8
            np.testing.assert_allclose(av, bv, rtol=tolerance, atol=tolerance * np.max(np.abs(av)))


@pytest.mark.integration
def test_hash_api_and_reference_agree(tmp_path):
    gprMax.run(inputfile=INPUT, outputfile=tmp_path / "hash", **COMMON)
    gprMax.run(scenes=[_scene()], outputfile=tmp_path / "api", **COMMON)
    gprMax.run(scenes=[_scene(reference=True)], outputfile=tmp_path / "reference", **COMMON)
    _compare(tmp_path / "hash.h5", tmp_path / "api.h5")
    _compare(tmp_path / "api.h5", tmp_path / "reference.h5")


@pytest.mark.integration
def test_hard_source_matches_reference_including_time_zero(tmp_path):
    for name, reference in (("builtin", False), ("reference", True)):
        gprMax.run(scenes=[_scene(reference=reference, resistance=0)], outputfile=tmp_path / name, **COMMON)
    _compare(tmp_path / "builtin.h5", tmp_path / "reference.h5")
    with h5py.File(tmp_path / "builtin.h5") as out:
        expected = gausspulse(-gausspulse("cutoff", fc=1e9), fc=1e9)
        assert out["ports/feed/Vtotal"][0] == pytest.approx(expected)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("resistance", [0, 50])
def test_cuda_matches_cpu(tmp_path, gpu_device, resistance):
    gprMax.run(scenes=[_scene(resistance=resistance)], outputfile=tmp_path / "cpu", **COMMON)
    gprMax.run(
        scenes=[_scene(resistance=resistance)],
        outputfile=tmp_path / "gpu",
        gpu=[gpu_device],
        gpu_precision="double",
        **COMMON
    )
    _compare(tmp_path / "cpu.h5", tmp_path / "gpu.h5", tolerance=2e-10)


def test_gallery_export_and_plot(tmp_path):
    table, plot = plot_and_export(tmp_path)
    assert table.read_text().splitlines()[0] == "time modulated_gaussian"
    times, values = sample_waveform()
    np.testing.assert_array_equal(np.loadtxt(table, skiprows=1), np.column_stack((times, values)))
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plot.stat().st_size > 1000


def test_standard_plotting_tool(tmp_path, monkeypatch):
    from gprMax.toolboxes.Plotting.plot_source_wave import mpl_plot
    from examples.features.waveforms.modulated_gaussian import waveform

    monkeypatch.chdir(tmp_path)
    plot = mpl_plot(waveform(), 6e-9, 1.926e-12, 3117, fft=True, show=False)
    plot.close("all")
    assert (tmp_path / "gauspulse.png").stat().st_size > 1000
