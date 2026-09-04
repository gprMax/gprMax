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

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
import gprMax.sources as sources_module
from gprMax.eigenmode_device import eigenmode_source_envelopes, eigenmode_source_profiles
from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver
from gprMax.sources import EigenmodeSource
from gprMax.waveforms import Waveform


def _field_set(values):
    values = np.asarray(values, dtype=np.complex128)
    return [values.copy(), np.zeros_like(values), np.zeros_like(values)]


def _trace_source(monkeypatch, *, modal_power=1.0):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "z0": 376.730313668,
                "c": 299792458.0,
            },
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.start = 0.0
    source.normal_axis = 0
    source.spectral_threshold = 1e-3
    source.anchor_complex_neff = np.ones(2, dtype=np.complex128)
    modal_fields = [
        np.ones((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
    ]
    source.anchor_modal_e = [
        [field.copy() for field in modal_fields],
        [field.copy() for field in modal_fields],
    ]
    source.anchor_modal_h = [
        [field.copy() for field in modal_fields],
        [field.copy() for field in modal_fields],
    ]
    source._modal_cross_power = lambda electric, magnetic, grid: modal_power
    source.waveform = Waveform()
    source.waveform.type = "ricker"
    source.waveform.amp = 1.0
    source.waveform.freq = 5e9
    grid = SimpleNamespace(
        iterations=2048,
        dt=1e-12,
        dl=np.asarray([0.5e-3, 0.5e-3, 0.5e-3]),
    )
    return source, grid


def _set_tagged_power_anchors(source):
    def fields(value):
        field = np.full((1, 1), value, dtype=np.complex128)
        zero = np.zeros_like(field)
        return [field, zero.copy(), zero.copy()]

    source.anchor_modal_e = [fields(0.0), fields(1.0)]
    source.anchor_modal_h = [fields(0.0), fields(1.0)]
    source._modal_cross_power = lambda electric, magnetic, grid: (
        electric[0][0, 0] * np.conj(magnetic[0][0, 0])
    )


def test_broadband_anchor_phase_alignment_is_phase_invariant(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.mode_index = 1
    reference_e = _field_set([1.0, 2.0])
    reference_h = _field_set([0.25, 0.5])
    phase = np.exp(0.73j)
    anchor_e = [reference_e, [field * phase for field in reference_e]]
    anchor_h = [reference_h, [field * phase for field in reference_h]]

    overlaps = source._align_and_validate_anchors(anchor_e, anchor_h, (1e9, 2e9))

    assert overlaps == pytest.approx([1.0])
    for expected, actual in zip(reference_e, anchor_e[1]):
        assert actual == pytest.approx(expected)
    for expected, actual in zip(reference_h, anchor_h[1]):
        assert actual == pytest.approx(expected)


def test_broadband_anchor_overlap_warns_and_continues(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.mode_index = 3
    anchor_e = [
        _field_set([1.0, 0.0]),
        _field_set([0.75, np.sqrt(1 - 0.75**2)]),
    ]
    anchor_h = [_field_set([0.0, 0.0]), _field_set([0.0, 0.0])]
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    overlaps = source._align_and_validate_anchors(anchor_e, anchor_h, (1e9, 2e9))
    output = "\n".join(warnings)

    assert overlaps == pytest.approx([0.75])
    assert "below the warning threshold 0.900000" in output
    assert "The run will continue" in output


def test_broadband_anchor_overlap_below_minimum_raises(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.mode_index = 2
    anchor_e = [
        _field_set([1.0, 0.0]),
        _field_set([0.59, np.sqrt(1 - 0.59**2)]),
    ]
    anchor_h = [_field_set([0.0, 0.0]), _field_set([0.0, 0.0])]

    with pytest.raises(
        ValueError,
        match="Use a single-frequency eigenmode solver instead",
    ):
        source._align_and_validate_anchors(anchor_e, anchor_h, (1e9, 2e9))


def test_broadband_anchor_overlap_at_minimum_warns_but_does_not_raise(
    monkeypatch,
):
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    EigenmodeSource._check_anchor_overlap(
        0.6,
        1e9,
        2e9,
        1,
        "Broadband eigenmode source",
    )

    assert "below the warning threshold 0.900000" in "\n".join(warnings)


def test_broadband_anchor_overlap_at_warning_threshold_does_not_warn(
    monkeypatch,
):
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    EigenmodeSource._check_anchor_overlap(
        0.9,
        1e9,
        2e9,
        1,
        "Broadband eigenmode source",
    )

    assert not warnings


def test_broadband_invalid_anchor_norm_raises(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.mode_index = 1
    zero_fields = _field_set([0.0, 0.0])
    anchor_e = [
        [field.copy() for field in zero_fields],
        [field.copy() for field in zero_fields],
    ]
    anchor_h = [
        [field.copy() for field in zero_fields],
        [field.copy() for field in zero_fields],
    ]
    with pytest.raises(
        ValueError,
        match="Use a single-frequency eigenmode solver instead",
    ):
        source._align_and_validate_anchors(anchor_e, anchor_h, (1e9, 2e9))


def test_broadband_anchor_coverage_warns_and_continues(monkeypatch):
    source, grid = _trace_source(monkeypatch)
    source.spectrum_coverage_policy = "warn"
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    source._prepare_broadband_time_traces(grid, (4.9e9, 5.1e9))
    output = "\n".join(warnings)

    assert "do not cover the significant waveform spectrum" in output
    assert np.all(np.isfinite(source.broadband_e_envelopes))
    assert source.broadband_waveform_error < 1e-8


def test_broadband_invalid_injected_modal_power_raises(monkeypatch):
    source, grid = _trace_source(monkeypatch, modal_power=0.0)

    with pytest.raises(
        ValueError,
        match="Invalid modal power affects .* injected FFT bin",
    ):
        source._prepare_broadband_time_traces(grid, (0.1e9, 25e9))


@pytest.mark.parametrize("sample", (np.nan, np.inf, -np.inf))
def test_broadband_nonfinite_waveform_sample_raises(sample):
    source = EigenmodeSource(None)
    source.waveform = SimpleNamespace(
        calculate_value=lambda time, dt: sample,
    )
    grid = SimpleNamespace(iterations=8, dt=1e-12)

    with pytest.raises(ValueError, match="contains non-finite samples"):
        source._prepare_broadband_time_traces(grid, (1e9, 2e9))


def test_broadband_zero_spectral_energy_raises():
    source = EigenmodeSource(None)
    source.waveform = SimpleNamespace(
        calculate_value=lambda time, dt: 0.0,
    )
    grid = SimpleNamespace(iterations=8, dt=1e-12)

    with pytest.raises(ValueError, match="zero or non-finite spectral energy"):
        source._prepare_broadband_time_traces(grid, (1e9, 2e9))


@pytest.mark.parametrize(
    "samples",
    (np.ones(8), (-1.0) ** np.arange(8)),
    ids=("dc", "nyquist"),
)
@pytest.mark.parametrize("single_frequency_iq", (False, True))
def test_eigenmode_iq_significant_dc_or_nyquist_warns_and_discards(
    monkeypatch,
    single_frequency_iq,
    samples,
):
    source, _ = _trace_source(monkeypatch)
    source.waveform = SimpleNamespace(
        calculate_value=lambda time, dt: float(samples[int(round(time / dt))]),
    )
    source.spectrum_coverage_policy = "allow"
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)
    grid = SimpleNamespace(iterations=8, dt=1.0, dl=np.ones(3))

    source._prepare_broadband_time_traces(
        grid,
        (1 / 16, 7 / 16),
        single_frequency_iq=single_frequency_iq,
    )

    warning = "\n".join(warnings)
    assert "significant DC or Nyquist content" in warning
    assert "will be discarded" in warning
    assert "band-limited waveform" in warning
    assert "waveform='auto'" in warning
    assert np.all(np.isfinite(source.broadband_e_envelopes))
    assert np.all(np.isfinite(source.broadband_h_envelopes))
    assert source.broadband_waveform_error > 0


def test_broadband_invalid_anchor_weight_partition_raises(monkeypatch):
    source, grid = _trace_source(monkeypatch)
    source._linear_anchor_weights = lambda bins, anchors: np.zeros(
        (len(anchors), len(bins)),
        dtype=np.float64,
    )

    with pytest.raises(RuntimeError, match="do not form a partition of unity"):
        source._prepare_broadband_time_traces(grid, (0.1e9, 25e9))


def test_broadband_modal_power_ignores_bins_not_injected(monkeypatch):
    source, _ = _trace_source(monkeypatch)
    sample_count = 8
    padded_count = 16
    samples = np.zeros(sample_count, dtype=np.float64)
    samples[0] = 1.0
    samples[4] -= 1.0
    spectrum = np.fft.rfft(samples, n=padded_count)
    zero_spectrum_index = 4
    assert spectrum[zero_spectrum_index] == 0.0

    source.waveform = SimpleNamespace(
        calculate_value=lambda time, dt: float(samples[int(round(time / dt))]),
    )
    grid = SimpleNamespace(
        iterations=sample_count,
        dt=1.0,
        dl=np.ones(3),
    )
    frequencies = (1 / padded_count, 7 / padded_count)
    _set_tagged_power_anchors(source)

    def weights(bin_frequencies, anchors):
        result = np.zeros((2, bin_frequencies.size), dtype=np.float64)
        result[1] = 1.0
        result[:, (0, zero_spectrum_index, bin_frequencies.size - 1)] = np.asarray(
            [[1.0], [0.0]]
        )
        return result

    source._linear_anchor_weights = weights

    source._prepare_broadband_time_traces(grid, frequencies)

    assert np.all(np.isfinite(source.broadband_e_envelopes))
    assert np.all(np.isfinite(source.broadband_h_envelopes))


def test_broadband_modal_power_ignores_discarded_endpoint_bins(monkeypatch):
    source, grid = _trace_source(monkeypatch)
    sample_count = grid.iterations
    padded_count = 1 << int(np.ceil(np.log2(2 * sample_count)))
    times = np.arange(sample_count, dtype=np.float64) * grid.dt
    samples = np.asarray(
        [source.waveform.calculate_value(time, grid.dt) for time in times]
    )
    magnitude = np.abs(np.fft.rfft(samples, n=padded_count))
    peak = float(np.max(magnitude))
    assert 0 < magnitude[0] < source.spectral_threshold * peak
    assert 0 < magnitude[-1] < source.spectral_threshold * peak
    _set_tagged_power_anchors(source)

    def weights(bin_frequencies, anchors):
        result = np.zeros((2, bin_frequencies.size), dtype=np.float64)
        result[1] = 1.0
        result[:, (0, bin_frequencies.size - 1)] = np.asarray([[1.0], [0.0]])
        return result

    source._linear_anchor_weights = weights

    source._prepare_broadband_time_traces(grid, (0.1e9, 25e9))

    assert np.all(np.isfinite(source.broadband_e_envelopes))
    assert np.all(np.isfinite(source.broadband_h_envelopes))


def test_linear_anchor_weights_are_local_and_sum_to_one():
    bins = np.asarray([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    anchors = np.asarray([1.0, 2.0, 4.0])

    weights = EigenmodeSource._linear_anchor_weights(bins, anchors)

    assert weights[:, 0] == pytest.approx([1.0, 0.0, 0.0])
    assert weights[:, 1] == pytest.approx([1.0, 0.0, 0.0])
    assert weights[:, 2] == pytest.approx([0.5, 0.5, 0.0])
    assert weights[:, 3] == pytest.approx([0.0, 1.0, 0.0])
    assert weights[:, 4] == pytest.approx([0.0, 0.5, 0.5])
    assert weights[:, 5] == pytest.approx([0.0, 0.0, 1.0])
    assert weights[:, 6] == pytest.approx([0.0, 0.0, 1.0])
    assert np.sum(weights, axis=0) == pytest.approx(np.ones(7))


@pytest.mark.parametrize("wave_type", ("ricker", "gaussiandot"))
def test_broadband_ifft_recovers_original_waveform(monkeypatch, wave_type):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "z0": 376.730313668,
                "c": 299792458.0,
            },
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.start = 0.0
    source.normal_axis = 0
    source.spectral_threshold = 1e-3
    source.anchor_complex_neff = np.ones(2, dtype=np.complex128)
    modal_fields = [
        np.ones((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
        np.zeros((1, 1), dtype=np.complex128),
    ]
    source.anchor_modal_e = [
        [field.copy() for field in modal_fields],
        [field.copy() for field in modal_fields],
    ]
    source.anchor_modal_h = [
        [field.copy() for field in modal_fields],
        [field.copy() for field in modal_fields],
    ]
    source._modal_cross_power = lambda electric, magnetic, grid: 1.0
    source.waveform = Waveform()
    source.waveform.type = wave_type
    source.waveform.amp = 1.0
    source.waveform.freq = 5e9
    grid = SimpleNamespace(
        iterations=2048,
        dt=1e-12,
        dl=np.asarray([0.5e-3, 0.5e-3, 0.5e-3]),
    )

    source._prepare_broadband_time_traces(grid, (0.1e9, 25e9))

    reconstructed = np.sum(source.broadband_e_envelopes[:, 0, :], axis=0)
    times = np.arange(grid.iterations) * grid.dt
    expected = np.asarray([source.waveform.calculate_value(time, grid.dt) for time in times])
    peak = np.max(np.abs(expected))
    assert np.max(np.abs(reconstructed - expected)) / peak < 1e-8
    assert source.broadband_waveform_error < 1e-8


def test_magnetic_stagger_factor_uses_each_frequency_and_beta():
    omega = 2 * np.pi * np.asarray([1e9, 4e9])
    beta = np.asarray([12.0, 31.0])
    dt = 2e-12
    spacing = 0.5e-3

    factors = EigenmodeSource._magnetic_stagger_factor(omega, beta, dt, spacing)

    assert factors == pytest.approx(np.exp(1j * (omega * dt / 2 + beta * spacing / 2)))
    assert factors[0] != pytest.approx(factors[1])


def _coarse_tem_trace_source(monkeypatch, operator_neff=1.0, carrier=10e9):
    """A matched TEM profile isolates the longitudinal Yee phase correction."""
    source, grid = _trace_source(monkeypatch)
    grid.iterations = 512
    grid.dt = 6e-12
    grid.dl[:] = 5e-3
    source.spectrum_coverage_policy = "allow"
    source.waveform = SimpleNamespace(
        calculate_value=lambda time, dt: (
            np.sin(2 * np.pi * carrier * time)
            * np.exp(-((time - 1.5e-9) / 0.13e-9) ** 2)
        ),
    )
    anchors = np.asarray([5e9, 15e9])
    source.anchor_operator_neff = np.full(2, operator_neff, dtype=np.complex128)
    anchor_symbol = (
        2 * np.sin(np.pi * anchors * grid.dt) / grid.dt
        * source.anchor_operator_neff / config.sim_config.em_consts["c"]
    )
    anchor_beta = 2 * np.arcsin(grid.dl[0] * anchor_symbol / 2) / grid.dl[0]
    source.anchor_complex_neff = (
        anchor_beta * config.sim_config.em_consts["c"] / (2 * np.pi * anchors)
    )
    times = np.arange(grid.iterations) * grid.dt
    waveform = np.asarray([source.waveform.calculate_value(t, grid.dt) for t in times])
    padded_count = 2 * grid.iterations
    spectrum = np.fft.rfft(waveform, n=padded_count)
    bins = np.fft.rfftfreq(padded_count, grid.dt)
    return source, grid, anchors, bins, spectrum


@pytest.mark.parametrize("operator_neff", [1.0, -1.0, 1.0 - 0.1j, -1.0 - 0.1j, 1.0 + 0.1j])
def test_broadband_tem_stagger_obeys_yee_dispersion_between_sparse_anchors(
    monkeypatch, operator_neff
):
    source, grid, anchors, bins, spectrum = _coarse_tem_trace_source(
        monkeypatch, operator_neff
    )
    omega = 2 * np.pi * bins
    # A homogeneous TEM mode has constant operator index, so this analytic
    # dispersion relation holds at every FFT bin, including outside anchors.
    symbol = (
        np.sin(np.pi * bins * grid.dt) * grid.dl[0] * complex(operator_neff)
        / (config.sim_config.em_consts["c"] * grid.dt)
    )
    beta = 2 * np.arcsin(symbol) / grid.dl[0]
    retained = np.ones(bins.size, dtype=bool)
    retained[[0, -1]] = False
    if np.imag(operator_neff) == 0:
        retained &= np.abs(symbol.real) < 1
    spectrum = np.where(retained, spectrum, 0)
    phase = np.exp(1j * (omega * grid.dt / 2 + beta * grid.dl[0] / 2))

    source._prepare_broadband_time_traces(grid, anchors)

    for quadrature, factor in enumerate((1, 1j)):
        expected_e = np.fft.irfft(factor * spectrum, n=2 * grid.iterations)[:grid.iterations]
        expected_h = np.fft.irfft(factor * spectrum * phase, n=2 * grid.iterations)[
            :grid.iterations
        ]
        np.testing.assert_allclose(
            np.sum(source.broadband_e_envelopes[:, quadrature], axis=0),
            expected_e, rtol=2e-13, atol=2e-14,
        )
        np.testing.assert_allclose(
            np.sum(source.broadband_h_envelopes[:, quadrature], axis=0),
            expected_h, rtol=2e-13, atol=2e-14,
        )
    # The coarse mesh must distinguish this oracle from the old interpolation
    # of physical index after the nonlinear arcsine conversion.
    midpoint_beta = 2 * np.arcsin(
        np.sin(np.pi * 10e9 * grid.dt) * grid.dl[0] * complex(operator_neff)
        / (config.sim_config.em_consts["c"] * grid.dt)
    ) / grid.dl[0]
    old_midpoint_beta = (
        2 * np.pi * 10e9 * np.mean(source.anchor_complex_neff)
        / config.sim_config.em_consts["c"]
    )
    assert abs(old_midpoint_beta / midpoint_beta - 1) > 0.01


@pytest.mark.parametrize("single_frequency_iq", [False, True])
def test_one_anchor_with_operator_metadata_keeps_constant_physical_index(
    monkeypatch, single_frequency_iq
):
    source, grid, anchors, bins, spectrum = _coarse_tem_trace_source(monkeypatch)
    source.anchor_complex_neff = source.anchor_complex_neff[:1]
    source.anchor_operator_neff = source.anchor_operator_neff[:1]
    source.anchor_modal_e = source.anchor_modal_e[:1]
    source.anchor_modal_h = source.anchor_modal_h[:1]
    spectrum[[0, -1]] = 0
    omega = 2 * np.pi * bins
    beta = omega * source.anchor_complex_neff[0] / config.sim_config.em_consts["c"]
    phase = np.exp(1j * (omega * grid.dt / 2 + beta * grid.dl[0] / 2))

    source._prepare_broadband_time_traces(
        grid, anchors[:1], single_frequency_iq=single_frequency_iq
    )

    np.testing.assert_allclose(
        source.broadband_h_envelopes[0, 0],
        np.fft.irfft(spectrum * phase, n=2 * grid.iterations)[:grid.iterations],
        rtol=2e-13, atol=2e-14,
    )


def test_operator_interpolation_uses_field_weights_at_anchors_and_endpoints(monkeypatch):
    source, grid, _, bins, spectrum = _coarse_tem_trace_source(monkeypatch)
    anchors = bins[[32, 96]]
    source.anchor_operator_neff = np.asarray([0.8, 1.05], dtype=np.complex128)
    source.anchor_complex_neff = (
        2 * np.arcsin(
            np.sin(np.pi * anchors * grid.dt) * grid.dl[0]
            * source.anchor_operator_neff / (config.sim_config.em_consts["c"] * grid.dt)
        ) / grid.dl[0] * config.sim_config.em_consts["c"] / (2 * np.pi * anchors)
    )
    upper_weight = np.clip((bins - anchors[0]) / (anchors[1] - anchors[0]), 0, 1)
    weights = np.stack((1 - upper_weight, upper_weight))
    operator_index = 0.8 * weights[0] + 1.05 * weights[1]
    symbol = np.sin(np.pi * bins * grid.dt) * grid.dl[0] * operator_index / (
        config.sim_config.em_consts["c"] * grid.dt
    )
    retained = symbol < 1
    retained[[0, -1]] = False
    spectrum = np.where(retained, spectrum, 0)
    beta = 2 * np.arcsin(symbol.astype(np.complex128)) / grid.dl[0]
    phase = np.exp(1j * (np.pi * bins * grid.dt + beta * grid.dl[0] / 2))

    source._prepare_broadband_time_traces(grid, anchors)

    for anchor in range(2):
        expected = np.fft.irfft(
            spectrum * weights[anchor] * phase, n=2 * grid.iterations
        )[:grid.iterations]
        np.testing.assert_allclose(
            source.broadband_h_envelopes[anchor, 0], expected, rtol=2e-13, atol=2e-14
        )


def test_legacy_broadband_without_operator_metadata_keeps_phase_interpolation(monkeypatch):
    source, grid, anchors, bins, spectrum = _coarse_tem_trace_source(monkeypatch)
    source.anchor_operator_neff = None
    spectrum[[0, -1]] = 0
    phase_index = np.interp(bins, anchors, source.anchor_complex_neff)
    omega = 2 * np.pi * bins
    beta = omega * phase_index / config.sim_config.em_consts["c"]
    phase = np.exp(1j * (omega * grid.dt / 2 + beta * grid.dl[0] / 2))

    source._prepare_broadband_time_traces(grid, anchors)

    np.testing.assert_allclose(
        np.sum(source.broadband_h_envelopes[:, 0], axis=0),
        np.fft.irfft(spectrum * phase, n=2 * grid.iterations)[:grid.iterations],
        rtol=2e-13, atol=2e-14,
    )


def test_broadband_significant_spatial_stop_band_energy_is_rejected(monkeypatch):
    source, grid, anchors, _, _ = _coarse_tem_trace_source(monkeypatch, carrier=25e9)

    with pytest.raises(ValueError, match="(?i)spatial.*(stop.band|unresolved)"):
        source._prepare_broadband_time_traces(grid, anchors)


def test_broadband_small_stop_band_tails_are_discarded_and_counted_in_error(monkeypatch):
    source, grid, anchors, bins, original = _coarse_tem_trace_source(monkeypatch)
    symbol = np.sin(np.pi * bins * grid.dt) * grid.dl[0] / (
        config.sim_config.em_consts["c"] * grid.dt
    )
    stop_band = symbol >= 1
    stop_band[[0, -1]] = False
    peak = np.max(np.abs(original[1:-1]))
    assert 0 < np.max(np.abs(original[stop_band])) < source.spectral_threshold * peak
    filtered = original.copy()
    filtered[stop_band] = 0
    filtered[[0, -1]] = 0
    expected = np.fft.irfft(filtered, n=2 * grid.iterations)[:grid.iterations]
    input_waveform = np.fft.irfft(original, n=2 * grid.iterations)[:grid.iterations]
    expected_error = np.max(np.abs(expected - input_waveform)) / np.max(np.abs(input_waveform))

    source._prepare_broadband_time_traces(grid, anchors)

    np.testing.assert_allclose(source.broadband_reconstructed_waveform, expected, atol=2e-14)
    np.testing.assert_allclose(source.broadband_input_waveform, input_waveform, atol=2e-14)
    assert source.broadband_waveform_error == pytest.approx(expected_error, rel=1e-8)
    assert source.broadband_waveform_error > 1e-9


def test_cached_mode_switch_keeps_operator_indices_with_selected_anchors(monkeypatch):
    source, grid = _trace_source(monkeypatch)
    grid.timewindow = grid.iterations * grid.dt
    source.mode_indices = (1, 2)
    source.port_anchor_frequencies = (5e9, 10e9, 15e9)
    source.port_anchor_mode_valid = np.asarray([[True, False], [True, True], [True, True]])
    source.port_anchor_neff = np.asarray([[1.1, 2.1], [1.2, 2.2], [1.3, 2.3]])
    source.port_anchor_operator_neff = np.asarray([[0.9, 1.8], [1.0, 1.9], [1.1, 2.0]])
    source.port_anchor_e = [[source.anchor_modal_e[0] for _ in range(2)] for _ in range(3)]
    source.port_anchor_h = [[source.anchor_modal_h[0] for _ in range(2)] for _ in range(3)]
    source.port_mode_solvers = tuple(object() for _ in range(3))
    source.port_mode_anchor_policies = ("explicit", "explicit_nonpropagating_trimmed")
    prepared = []

    def capture_prepared(grid, frequencies):
        prepared.append((frequencies, source.anchor_operator_neff.copy()))

    monkeypatch.setattr(source, "_prepare_broadband_time_traces", capture_prepared)

    source.configure_cached_excitation(grid, 2, source.waveform)
    source.configure_cached_excitation(grid, 1, source.waveform)

    assert prepared[0][0] == (10e9, 15e9)
    np.testing.assert_array_equal(prepared[0][1], [1.9, 2.0])
    assert prepared[1][0] == (5e9, 10e9, 15e9)
    np.testing.assert_array_equal(prepared[1][1], [0.9, 1.0, 1.1])


def test_modal_cross_power_is_independent_of_requested_direction(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.invariant_axis = 2
    source.physical_transverse_axis = 1
    source.domain_polarization = "TM"
    electric = [
        np.zeros((4, 2), dtype=np.complex128),
        np.zeros((3, 2), dtype=np.complex128),
        np.ones((4, 1), dtype=np.complex128),
    ]
    magnetic = [
        np.zeros((3, 1), dtype=np.complex128),
        -np.ones((4, 1), dtype=np.complex128),
        np.zeros((3, 2), dtype=np.complex128),
    ]
    grid = SimpleNamespace(dl=np.asarray([0.5, 0.25, 1.0]))

    source.direction = "+"
    forward_basis_power = source._modal_cross_power(electric, magnetic, grid)
    source.direction = "-"
    backward_basis_power = source._modal_cross_power(electric, magnetic, grid)

    assert forward_basis_power == pytest.approx(0.375)
    assert backward_basis_power == pytest.approx(forward_basis_power)


def test_modal_cross_power_2d_te_uses_only_live_invariant_layer(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 376.730313668, "c": 299792458.0}),
    )
    source = EigenmodeSource(None)
    source.direction = "+"
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.invariant_axis = 2
    source.physical_transverse_axis = 1
    source.domain_polarization = "TE"
    electric = [
        np.zeros((4, 2), dtype=np.complex128),
        np.zeros((3, 3), dtype=np.complex128),
        np.zeros((4, 2), dtype=np.complex128),
    ]
    magnetic = [
        np.zeros((4, 2), dtype=np.complex128),
        np.zeros((3, 3), dtype=np.complex128),
        np.zeros((3, 3), dtype=np.complex128),
    ]
    electric[1][:, 1] = 2.0
    magnetic[2][:, 1] = 0.5
    grid = SimpleNamespace(dl=np.asarray([0.5, 0.25, 1.0]))

    assert source._modal_cross_power(electric, magnetic, grid) == pytest.approx(0.375)


@pytest.mark.parametrize("polarization", ("TM", "TE"))
@pytest.mark.parametrize(
    ("invariant_axis", "normal_axis"),
    ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
)
def test_1d_solver_mapping_has_positive_forward_power(
    invariant_axis,
    normal_axis,
    polarization,
):
    physical_transverse_axis = next(
        axis for axis in range(3) if axis not in (invariant_axis, normal_axis)
    )
    transverse_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    source = EigenmodeSource(None)
    source.normal_axis = normal_axis
    source.transverse_axes = transverse_axes
    source.invariant_axis = invariant_axis
    source.physical_transverse_axis = physical_transverse_axis
    source.domain_polarization = polarization
    source.transverse_start = np.zeros(2, dtype=np.int32)
    source.transverse_stop = np.asarray(
        [2 if axis == physical_transverse_axis else 1 for axis in transverse_axes],
        dtype=np.int32,
    )

    solver = object.__new__(FDFD_1D_mode_solver)
    solver.num_modes = 1
    solver.complex_neff = np.asarray([1.0 + 0j])
    if polarization == "TM":
        solver.Ea = np.ones((3, 1), dtype=np.complex128)
        solver.Ht = -np.ones((3, 1), dtype=np.complex128)
        solver.Hw = np.zeros((2, 1), dtype=np.complex128)
    else:
        solver.Et = np.ones((2, 1), dtype=np.complex128)
        solver.Ha = np.ones((2, 1), dtype=np.complex128)
        solver.Ew = np.zeros((3, 1), dtype=np.complex128)

    electric, magnetic, _ = source._fields_from_solver_mode(solver, 1)
    power = source._modal_cross_power(
        electric,
        magnetic,
        SimpleNamespace(dl=np.ones(3)),
    )

    assert power == pytest.approx(1.0)


def test_single_frequency_global_phase_shift_uses_real_profile(monkeypatch):
    impedance = 376.730313668
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={"z0": impedance, "c": 299792458.0},
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.frequency = 5e9
    source.complex_neff = 1.0 + 0.0j
    phase = np.exp(0.73j)
    source.modal_e = [
        1j * np.asarray([1.0, 2.0]),
        phase * np.asarray([1.0, -0.5, 0.25]),
        phase * np.asarray([0.75, -1.25]),
    ]
    source.modal_h = [
        -1j * np.asarray([0.2, 0.4]),
        phase * np.asarray([0.4, -0.2]) / impedance,
        phase * np.asarray([0.3, 0.1, -0.5]) / impedance,
    ]

    source._prepare_single_frequency_injection(SimpleNamespace(dl=np.full(3, 0.5e-3)))

    residual = source.complex_profile_residual
    assert residual < 1e-12
    assert not source.uses_quadrature
    assert source.broadband_e_envelopes is None
    assert source.broadband_h_envelopes is None
    assert source.broadband_modal_e_real is None
    assert source.broadband_modal_e_imag is None
    assert source.broadband_modal_h_real is None
    assert source.broadband_modal_h_imag is None
    for axis in source.transverse_axes:
        assert np.max(np.abs(np.imag(source.modal_e[axis]))) < 1e-12
        assert np.max(np.abs(np.imag(source.modal_h[axis]))) < 1e-12
    for complex_field, real_field in zip(source.modal_e, source.modal_e_real):
        assert real_field.dtype == np.float64
        assert real_field.flags.c_contiguous
        assert real_field == pytest.approx(np.real(complex_field))
    for complex_field, real_field in zip(source.modal_h, source.modal_h_real):
        assert real_field.dtype == np.float64
        assert real_field.flags.c_contiguous
        assert real_field == pytest.approx(np.real(complex_field))
    # The normal components are deliberately in quadrature and do not force
    # I/Q injection because the TF/SF plane never injects them.
    assert np.max(np.abs(np.imag(source.modal_e[source.normal_axis]))) > 0.1


def test_single_frequency_phase_residual_matches_rotated_imaginary_norm(
    monkeypatch,
):
    impedance = 2.5
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={"z0": impedance, "c": 299792458.0},
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.modal_e = [
        np.zeros(1, dtype=np.complex128),
        np.asarray([np.exp(0.2j), 0.6 * np.exp(1.0j)]),
        np.asarray([0.4 * np.exp(-0.35j)]),
    ]
    source.modal_h = [
        np.zeros(1, dtype=np.complex128),
        np.asarray([0.25 * np.exp(0.65j)]) / impedance,
        np.asarray([0.8 * np.exp(0.45j), 0.3 * np.exp(-0.8j)]) / impedance,
    ]
    total_energy = sum(
        np.sum(np.abs(source.modal_e[axis]) ** 2)
        + np.sum(np.abs(impedance * source.modal_h[axis]) ** 2)
        for axis in source.transverse_axes
    )

    residual = source._align_tangential_mode_for_real_injection()

    rotated_imaginary_energy = sum(
        np.sum(np.imag(source.modal_e[axis]) ** 2)
        + np.sum(np.imag(impedance * source.modal_h[axis]) ** 2)
        for axis in source.transverse_axes
    )
    independently_measured_residual = np.sqrt(rotated_imaginary_energy / total_energy)
    assert 0.1 < abs(source.complex_profile_phase) < 1.0
    assert 0.1 < residual < 0.7
    assert residual == pytest.approx(independently_measured_residual)
    assert source.complex_profile_residual == pytest.approx(independently_measured_residual)


def test_single_frequency_spatial_phase_requires_quadrature(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={"z0": 1.0, "c": 299792458.0},
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.modal_e = [
        np.zeros(1, dtype=np.complex128),
        np.asarray([1.0, 1.0j]),
        np.zeros(1, dtype=np.complex128),
    ]
    source.modal_h = [
        np.zeros(1, dtype=np.complex128),
        np.zeros(1, dtype=np.complex128),
        np.asarray([1.0, 1.0j]),
    ]

    residual = source._align_tangential_mode_for_real_injection()

    assert residual == pytest.approx(1 / np.sqrt(2))
    assert residual > source.COMPLEX_PROFILE_TOLERANCE


@pytest.mark.parametrize(
    ("complex_profile", "phase", "delay", "neff", "expected_reasons"),
    (
        (True, 0.0, 0.0, 1.0, ("complex modal profile",)),
        (False, 45.0, 0.0, 1.0, ("drive phase/delay",)),
        (False, 0.0, 1e-12, 1.0, ("drive phase/delay",)),
        (False, 0.0, 0.0, 1.0 - 0.1j, ("complex longitudinal staggering",)),
        (
            True, 45.0, 1e-12, 1.0 - 0.1j,
            ("complex modal profile", "drive phase/delay", "complex longitudinal staggering"),
        ),
        (False, 0.0, 0.0, -1.0, ()),
        (False, 360.0, 0.0, 1.0, ()),
    ),
)
def test_single_frequency_iq_diagnostic_records_actual_reasons(
    monkeypatch, complex_profile, phase, delay, neff, expected_reasons
):
    source, grid = _trace_source(monkeypatch)
    source.transverse_axes = (1, 2)
    source.frequency = 5e9
    source.complex_neff = neff
    source.drive_phase_deg, source.drive_delay_s = phase, delay
    profile = np.asarray([1.0, 1.0j if complex_profile else 1.0])
    zero = np.zeros_like(profile)
    source.modal_e = [zero.copy(), profile.copy(), zero.copy()]
    source.modal_h = [zero.copy(), zero.copy(), profile / config.sim_config.em_consts["z0"]]
    source.single_frequency_iq_reasons = ("stale reason",)
    messages = []
    monkeypatch.setattr(sources_module.logger, "info", messages.append)

    source._prepare_single_frequency_injection(grid)

    assert source.single_frequency_iq_reasons == expected_reasons
    assert source.uses_quadrature is bool(expected_reasons)
    if not complex_profile:
        assert source.complex_profile_residual < source.COMPLEX_PROFILE_TOLERANCE
    diagnostic = next(message for message in messages if message.startswith("Single-frequency"))
    assert "above tolerance" not in diagnostic
    if expected_reasons:
        assert f"(reasons: {', '.join(expected_reasons)})" in diagnostic
    else:
        assert "using real-only injection" in diagnostic
        assert "reasons:" not in diagnostic


def test_single_frequency_complex_mode_reuses_fft_quadrature(monkeypatch):
    speed = 299792458.0
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={"z0": 1.0, "c": speed},
            dtypes={"float_or_double": np.float64},
        ),
    )
    source = EigenmodeSource(None)
    source.normal_axis = 0
    source.transverse_axes = (1, 2)
    source.frequency = 5e9
    source.complex_neff = 1.0 + 0.0j
    source.mode_solver = object()
    source.modal_e = [
        np.zeros(1, dtype=np.complex128),
        np.asarray([1.0, 1.0j]),
        np.zeros(1, dtype=np.complex128),
    ]
    source.modal_h = [
        np.zeros(1, dtype=np.complex128),
        np.zeros(1, dtype=np.complex128),
        np.asarray([1.0, 1.0j]),
    ]
    source._modal_cross_power = lambda electric, magnetic, grid: 1.0
    source.waveform = Waveform()
    source.waveform.type = "user"
    source.waveform.userfunc = lambda time: (
        np.sin(2 * np.pi * source.frequency * time) * np.exp(-(((time - 1.0e-9) / 0.3e-9) ** 2))
    )
    grid = SimpleNamespace(
        iterations=256,
        dt=1e-11,
        dl=np.asarray([0.5e-3, 0.5e-3, 0.5e-3]),
    )
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    source._prepare_single_frequency_injection(grid)
    output = "\n".join(warnings)

    assert source.waveform.freq is None
    assert source.uses_quadrature
    assert source.broadband_e_envelopes.shape == (1, 2, grid.iterations)
    assert source.broadband_h_envelopes.shape == (1, 2, grid.iterations)
    assert "anchor frequencies do not cover" not in output

    sample_count = grid.iterations
    padded_count = 1 << int(np.ceil(np.log2(max(2, 2 * sample_count))))
    times = np.arange(sample_count) * grid.dt
    waveform = np.asarray([source.waveform.calculate_value(time, grid.dt) for time in times])
    spectrum = np.fft.rfft(waveform, n=padded_count)
    spectrum[0] = 0
    spectrum[-1] = 0

    electric_real = source.broadband_modal_e_real[0][1][1]
    electric_imag = source.broadband_modal_e_imag[0][1][1]
    actual_electric = (
        electric_real * source.broadband_e_envelopes[0, 0]
        + electric_imag * source.broadband_e_envelopes[0, 1]
    )
    electric_field = source.anchor_modal_e[0][1][1]
    expected_electric = np.fft.irfft(electric_field * spectrum, n=padded_count)[:sample_count]
    assert actual_electric == pytest.approx(expected_electric)

    bin_frequencies = np.fft.rfftfreq(padded_count, d=grid.dt)
    omega = 2 * np.pi * bin_frequencies
    beta = omega * source.complex_neff / speed
    magnetic_phase = np.exp(1j * (omega * grid.dt / 2 + beta * grid.dl[source.normal_axis] / 2))
    magnetic_real = source.broadband_modal_h_real[0][2][1]
    magnetic_imag = source.broadband_modal_h_imag[0][2][1]
    actual_magnetic = (
        magnetic_real * source.broadband_h_envelopes[0, 0]
        + magnetic_imag * source.broadband_h_envelopes[0, 1]
    )
    magnetic_field = source.anchor_modal_h[0][2][1]
    expected_magnetic = np.fft.irfft(
        magnetic_field * spectrum * magnetic_phase,
        n=padded_count,
    )[:sample_count]
    assert actual_magnetic == pytest.approx(expected_magnetic)


@pytest.mark.parametrize("normal_axis", (0, 2))
@pytest.mark.parametrize("speed", (299792458.0, 0.8 * 299792458.0))
@pytest.mark.parametrize(
    ("operator_neff", "requires_quadrature"),
    (
        pytest.param(1.0, False, id="positive-phase"),
        pytest.param(-1.0, False, id="negative-phase"),
        pytest.param(1.0 - 0.1j, True, id="lossy-real-profile"),
        pytest.param(1.0 + 0.1j, True, id="growing-real-profile"),
        pytest.param(1.0 - 1e-16j, False, id="roundoff-loss"),
    ),
)
def test_single_frequency_real_profile_preserves_propagation_stagger(
    monkeypatch, normal_axis, speed, operator_neff, requires_quadrature
):
    """CPU and device H injection must preserve phase sign and attenuation."""
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={"z0": 1.0, "c": speed},
            dtypes={"float_or_double": np.float64, "C_float_or_double": "double"},
        ),
    )
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(ompthreads=1))
    grid = SimpleNamespace(
        iterations=128,
        dt=6e-12,
        dl=np.full(3, 1e-3),
        updatecoeffsE=None,
        ID=None,
        Ex=None,
        Ey=None,
        Ez=None,
    )
    grid.dl[normal_axis] = 5e-3
    source = EigenmodeSource(None)
    source.normal_axis = normal_axis
    source.transverse_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    source.transverse_start = np.zeros(2, dtype=np.int32)
    source.transverse_stop = np.ones(2, dtype=np.int32)
    source.plane_index = 1
    source.direction = "+"
    source.start, source.stop = 0.0, np.inf
    source.frequency = 10e9
    omega = 2 * np.pi * source.frequency
    spacing = grid.dl[normal_axis]
    discrete_beta = operator_neff * 2 * np.sin(omega * grid.dt / 2) / (speed * grid.dt)
    beta = 2 * np.arcsin(discrete_beta * spacing / 2) / spacing
    source.complex_neff = beta * speed / omega
    source.neff = float(np.real(source.complex_neff))
    source.mode_solver = object()
    source.modal_e = [np.zeros((1, 1), dtype=np.complex128) for _ in range(3)]
    source.modal_h = [np.zeros((1, 1), dtype=np.complex128) for _ in range(3)]
    electric_axis, magnetic_axis = source.transverse_axes
    # Matched impedance keeps both tangential profiles real even for complex beta.
    source.modal_e[electric_axis].fill(1.0)
    source.modal_h[magnetic_axis].fill(1.0)
    source._modal_cross_power = lambda electric, magnetic, grid: 1.0
    source.waveform = Waveform()
    source.waveform.type = "user"
    source.waveform.userfunc = lambda time: (
        np.sin(omega * time) * np.exp(-(((time - 0.35e-9) / 0.12e-9) ** 2))
    )

    source._prepare_single_frequency_injection(grid)

    assert source.complex_profile_residual < 1e-14
    assert source.uses_quadrature is requires_quadrature
    times = np.arange(grid.iterations) * grid.dt
    if requires_quadrature:
        padded_count = 2 * grid.iterations
        waveform = np.asarray([source.waveform.calculate_value(t, grid.dt) for t in times])
        spectrum = np.fft.rfft(waveform, n=padded_count)
        spectrum[0] = spectrum[-1] = 0
        bin_omega = 2 * np.pi * np.fft.rfftfreq(padded_count, d=grid.dt)
        # The single-frequency model holds phase index constant across the pulse.
        bin_beta = beta * bin_omega / omega
        stagger = np.exp(1j * (bin_omega * grid.dt / 2 + bin_beta * spacing / 2))
        expected = np.fft.irfft(spectrum * stagger, n=padded_count)[: grid.iterations]
    else:
        shifted_times = times + grid.dt / 2 + np.real(beta) * spacing / (2 * omega)
        expected = np.asarray(
            [
                source.waveform.calculate_value(t, grid.dt) if t >= source.start else 0.0
                for t in shifted_times
            ]
        )

    applied = []

    def capture_electric_update(*args):
        # The electric TF/SF update consumes the incident magnetic profile.
        applied.append(args[10] * args[11 + magnetic_axis][0, 0])

    monkeypatch.setattr(
        sources_module, "updateEigenmode_electric", {"double": capture_electric_update}
    )
    _, magnetic_profiles = eigenmode_source_profiles(source)
    cpu_trace, device_trace = [], []
    for iteration in range(grid.iterations):
        applied.clear()
        source.update_eigenmode_electric(iteration, grid)
        cpu_trace.append(sum(applied))
        device_trace.append(
            sum(
                magnetic_profiles[basis, magnetic_axis, 0, 0] * envelope
                for basis, envelope in eigenmode_source_envelopes(
                    source, grid, iteration, magnetic=False
                )
            )
        )
    np.testing.assert_allclose(cpu_trace, expected, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(device_trace, expected, rtol=2e-13, atol=2e-14)
