from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.sources as sources_module
from gprMax.eigenmode_config import (
    EigenmodeBandSpec,
    EigenmodeBandpassWaveform,
    EigenmodePortSpec,
    sampled_waveform_spectrum,
)
from gprMax.sources import (
    EigenmodeAnchorMismatchError,
    EigenmodeSource,
)
from gprMax.waveforms import Waveform


def test_automatic_bandpass_tracks_requested_band_and_avoids_dc_nyquist():
    dt = 0.2e-12
    sample_count = 2048
    waveform = EigenmodeBandpassWaveform(
        band_id='wg',
        fmin=45e9,
        fmax=65e9,
        amplitude=1.0,
        dt=dt,
        sample_count=sample_count,
        spectral_threshold=1e-3,
    )

    _, frequencies, spectrum = sampled_waveform_spectrum(waveform, dt, sample_count)
    magnitude = np.abs(spectrum)
    magnitude /= np.max(magnitude)
    passband = (frequencies >= 45e9) & (frequencies <= 65e9)

    assert np.min(magnitude[passband]) > 0.25
    assert magnitude[0] < 1e-3
    assert magnitude[-1] < 1e-3
    assert waveform.significant_low < 45e9
    assert waveform.significant_high > 65e9


def test_bad_custom_waveform_recommends_automatic_bandpass():
    waveform = Waveform()
    waveform.type = 'gaussiandot'
    waveform.amp = 1.0
    waveform.freq = 55e9
    grid = SimpleNamespace(dt=0.2e-12, iterations=2048)
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)

    with pytest.raises(ValueError, match='waveform=\'auto\''):
        band.resolve_spectrum(grid, waveform, generated_waveform=False)


def _port(port, anchors):
    return EigenmodePortSpec(
        port=port,
        p1=(0.0, 0.0, 0.0),
        p2=(0.0, 1.0, 1.0),
        normal='x',
        direction='+',
        normal_axis=0,
        transverse_axes=(1, 2),
        invariant_axis=None,
        modes=(1,),
        anchors=anchors,
        plot_fields=None,
    )


def test_auto_source_anchors_cover_spectrum_but_passive_anchors_cover_dft_band():
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)
    band.significant_range = (32e9, 78e9)
    source = _port(1, 'auto')
    receiver = _port(2, 'auto')

    source.resolve_anchors(band, is_source=True)
    receiver.resolve_anchors(band, is_source=False)

    assert source.resolved_anchors[0] == 32e9
    assert source.resolved_anchors[-1] == 78e9
    assert receiver.resolved_anchors[0] == 45e9
    assert receiver.resolved_anchors[-1] == 65e9


def test_explicit_multiple_anchors_require_coverage_but_single_is_allowed():
    band = EigenmodeBandSpec(id='wg', fmin=45e9, fmax=65e9, points=81)
    band.significant_range = (32e9, 78e9)

    with pytest.raises(ValueError, match='Suggested coverage anchors'):
        _port(1, (45e9, 55e9, 65e9)).resolve_anchors(band, is_source=True)

    single = _port(1, (55e9,))
    single.resolve_anchors(band, is_source=True)
    assert single.resolved_anchors == (55e9,)


def test_auto_anchor_mode_mismatch_falls_back_to_band_centre(monkeypatch):
    source = EigenmodeSource(None)
    source.plane_index = 1
    source.port_index = 3
    source.frequency = 45e9
    source.frequencies = (45e9, 55e9, 65e9)
    source.anchor_policy = 'auto'
    source.fallback_frequency = 55e9
    calls = []

    def fail_broadband(grid, frequencies):
        raise EigenmodeAnchorMismatchError('mode mismatch')

    source._solve_broadband_eigenmode = fail_broadband
    source._extract_frequency_dependent_materials = lambda grid: calls.append('extract')
    source._solve_eigenmode = lambda grid: calls.append('solve')
    source._prepare_single_frequency_injection = lambda grid: calls.append('prepare')
    source._register_port_monitor = lambda grid: calls.append('monitor')
    warnings = []
    monkeypatch.setattr(sources_module.logger, 'warning', warnings.append)

    source.grid_init(SimpleNamespace())

    assert source.frequencies == (55e9,)
    assert calls == ['extract', 'solve', 'prepare', 'monitor']
    assert 'may be inaccurate toward frequencies far from this anchor' in warnings[0]


def test_explicit_anchor_mode_mismatch_remains_an_error():
    with pytest.raises(
        EigenmodeAnchorMismatchError,
        match='single explicit frequency anchor',
    ):
        EigenmodeSource._check_anchor_overlap(
            0.2,
            45e9,
            65e9,
            1,
            'Eigenmode port 1',
        )
