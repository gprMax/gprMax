"""MATLAB-default Gaussian RF pulse, including the independent DPW evaluator."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import gausspulse

from gprMax.cython.plane_wave import getSource
from gprMax.sources import DiscretePlaneWave
from gprMax.user_objects.cmds_multiuse import Waveform as WaveformCommand

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("frequency", [50e3, 1e9, 2e9, 10e9])
@pytest.mark.parametrize("amplitude", [0.0, 1.0, -2.5])
def test_matches_reference_on_whole_and_half_steps(make_waveform, frequency, amplitude):
    w = make_waveform("gauspulse", freq=frequency, amp=amplitude)
    dt = 1 / (200 * frequency)
    delay = gausspulse("cutoff", fc=frequency)
    times = np.arange(2401) * dt / 2
    expected = amplitude * gausspulse(times - delay, fc=frequency)
    actual = [w.calculate_value(t, dt) for t in times]
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-13)
    assert w.chi == pytest.approx(delay)
    assert w.calculate_value(w.chi, dt) == pytest.approx(amplitude)
    assert np.exp(-w.zeta * w.chi**2) == pytest.approx(1e-3)


def test_measured_bandwidth(make_waveform):
    w = make_waveform("gauspulse", freq=1e9)
    dt = 1e-12
    time = np.arange(8001) * dt
    values = np.array([w.calculate_value(t, dt) for t in time])
    # Evaluate the sampled Fourier transform at the carrier and specified
    # half-band edges, avoiding FFT-bin quantisation in the bandwidth check.
    frequencies = w.freq * np.array([0.75, 1.0, 1.25])
    spectrum = np.abs(np.exp(-2j * np.pi * frequencies[:, None] * time) @ values)
    db = 20 * np.log10(spectrum / spectrum[1])
    np.testing.assert_allclose(db[[0, 2]], [-6, -6], atol=0.01)


def test_cython_point_evaluator_matches_python(make_waveform):
    w = make_waveform("gauspulse", freq=2e9)
    times = np.linspace(0, 3e-9, 401)
    np.testing.assert_allclose(
        [getSource(t, w.freq, b"gauspulse", 1e-12) for t in times],
        [w.calculate_value(t, 1e-12) for t in times],
        atol=1e-14,
        rtol=1e-12,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dpw_precomputed_histories_match_reference(make_waveform, dtype, monkeypatch):
    from gprMax import config

    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"float_or_double": dtype},
            get_model_config=lambda: SimpleNamespace(mode="3D"),
        ),
    )
    dpw = DiscretePlaneWave(G=None)
    dpw.m = np.array([1, 2, 3, 3], dtype=np.int32)
    dpw.ds, dpw.speed = 1e-3, 299792458.0
    dpw.start, dpw.stop = 0.2e-9, 5.8e-9
    dpw.waveform = make_waveform("gauspulse", freq=1e9, amp=-2.5)
    grid = SimpleNamespace(iterations=1200, dt=5e-12)
    dpw.calculate_waveform_values(grid, cythonize=False)
    whole = dpw.waveformvalues_wholedt.copy()
    half = dpw.waveformvalues_halfdt.copy()
    dpw.calculate_waveform_values(grid, cythonize=True)
    tol = 2e-7 if dtype == np.float32 else 2e-13
    np.testing.assert_allclose(dpw.waveformvalues_wholedt, whole, rtol=tol, atol=tol)
    np.testing.assert_allclose(dpw.waveformvalues_halfdt, half, rtol=tol, atol=tol)
    assert np.max(np.abs(whole)) > 2.4


@pytest.mark.parametrize("frequency", [0, -1, np.inf, np.nan])
def test_command_rejects_invalid_frequency(frequency):
    with pytest.raises(ValueError, match="finite excitation frequency greater than zero"):
        WaveformCommand(wave_type="gauspulse", amp=1, freq=frequency, id="pulse").build(None)
