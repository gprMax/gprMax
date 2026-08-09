from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
import gprMax.sources as sources_module
from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver
from gprMax.sources import EigenmodeSource
from gprMax.waveforms import Waveform


def _field_set(values):
    values = np.asarray(values, dtype=np.complex128)
    return [values.copy(), np.zeros_like(values), np.zeros_like(values)]


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


def test_broadband_quality_safeguards_warn_and_continue(monkeypatch):
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
    source.spectrum_coverage_policy = "warn"
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
    source._modal_cross_power = lambda electric, magnetic, grid: 0.0
    source.waveform = Waveform()
    source.waveform.type = "ricker"
    source.waveform.amp = 1.0
    source.waveform.freq = 5e9
    grid = SimpleNamespace(
        iterations=2048,
        dt=1e-12,
        dl=np.asarray([0.5e-3, 0.5e-3, 0.5e-3]),
    )
    warnings = []
    monkeypatch.setattr(sources_module.logger, "warning", warnings.append)

    source._prepare_broadband_time_traces(grid, (4.9e9, 5.1e9))
    output = "\n".join(warnings)

    assert "do not cover the significant waveform spectrum" in output
    assert "fallback normalization" in output
    assert np.all(np.isfinite(source.broadband_e_envelopes))
    assert source.broadband_waveform_error < 1e-8


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

    source._prepare_single_frequency_injection(SimpleNamespace())

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
