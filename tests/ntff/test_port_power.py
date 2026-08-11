"""Exact-frequency antenna-port power tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax.config as config
import gprMax.ports as ports
from gprMax.eigenmode_ports import EigenmodePortMonitor, EigenmodePortResult
from gprMax.ntff.conventions import engineering_dft
from gprMax.ports import (
    MagneticFrillPortOutput,
    RationalNetworkPortOutput,
    TransmissionLinePortOutput,
    VoltageSourcePortMonitor,
    evaluate_port_power_spectrum,
    modal_power_spectrum,
)


@pytest.fixture(autouse=True)
def port_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"float_or_double": np.float64, "complex": np.complex128},
        ),
    )


def _voltage_port(total_voltage, generator_voltage, *, dt, resistance=50.0, capacitance=0.0):
    source = SimpleNamespace(polarisation="z")
    receiver = SimpleNamespace()
    output = VoltageSourcePortMonitor("feed", source, receiver, 10.0)
    output.reference_impedance = resistance
    output.hard_source = False
    output.background_conductance = 0.0
    output.gap_capacitance = capacitance
    output.minimum_wavelength_cells = 10.0
    output.result = SimpleNamespace(
        total_voltage=np.asarray(total_voltage, dtype=np.float64),
        generator_voltage=np.asarray(generator_voltage, dtype=np.float64),
    )
    grid = SimpleNamespace(dt=dt)
    return output, grid


def _hard_voltage_port(total_voltage, loop_current, *, dt, resistance=50.0):
    output, grid = _voltage_port(
        total_voltage,
        total_voltage,
        dt=dt,
        resistance=resistance,
    )
    output.hard_source = True
    output._hard_loop_current = np.asarray(loop_current, dtype=np.float64)
    output.hard_voltage_time_offset = dt
    output.hard_current_time_offset = 0.5 * dt
    return output, grid


def _rational_network_port(total_voltage, network_current, *, dt, resistance=50.0):
    terminal = SimpleNamespace()
    output = RationalNetworkPortOutput("feed", terminal, resistance)
    output.background_conductance = 0.0
    output.gap_capacitance = 0.0
    output.minimum_wavelength_cells = 10.0
    output.result = SimpleNamespace(
        total_voltage=np.asarray(total_voltage, dtype=np.float64),
        network_current=np.asarray(network_current, dtype=np.float64),
    )
    return output, SimpleNamespace(dt=dt)


def _eigenmode_port(*, is_source):
    monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
    monitor.port_id = "modal"
    monitor.port_index = 1
    monitor.owner = SimpleNamespace()
    monitor.mode_indices = (1, 2)
    monitor.is_source = is_source
    monitor.excitation_mode_index = 1 if is_source else None
    monitor.mode_power_valid = np.ones((1, 2), dtype=bool)
    monitor.power_matrix_valid = np.ones(1, dtype=bool)
    monitor.power_matrix = np.asarray(
        [[[2.0, 0.5 - 0.25j], [0.5 + 0.25j, 1.0]]],
        dtype=np.complex128,
    )
    monitor.electric_gram = monitor.power_matrix.copy()
    monitor.result = EigenmodePortResult(
        frequency=np.asarray([5.0]),
        incident=np.asarray([[2.0 + 0.0j], [1.0 - 0.5j]]),
        outgoing=np.asarray([[0.25 + 0.0j], [0.5 + 0.25j]]),
        valid=np.ones((2, 1), dtype=bool),
        condition_number=np.ones(1),
    )
    return monitor


def test_modal_power_quadratic_form_is_invariant_under_basis_change():
    amplitudes = np.asarray([[1.0 + 0.5j], [-0.25 + 0.75j]])
    matrix = np.asarray([[[1.5, 0.2j], [-0.2j, 0.8]]])
    transform = np.asarray([[1.0, 0.25j], [-0.3, 1.2]])
    transformed_amplitudes = np.linalg.solve(transform, amplitudes)
    transformed_matrix = np.asarray([np.conj(transform.T) @ matrix[0] @ transform])

    assert_allclose(
        modal_power_spectrum(amplitudes, matrix),
        modal_power_spectrum(transformed_amplitudes, transformed_matrix),
    )


@pytest.mark.parametrize("is_source", (True, False))
def test_eigenmode_port_power_uses_full_modal_matrix(monkeypatch, is_source):
    monitor = _eigenmode_port(is_source=is_source)
    monkeypatch.setattr(
        ports,
        "_port_mesh_valid",
        lambda output, grid, frequency: np.ones(frequency.shape, dtype=bool),
    )

    spectrum = evaluate_port_power_spectrum(monitor, SimpleNamespace(), [5.0])
    expected_accepted = modal_power_spectrum(
        monitor.result.incident, monitor.power_matrix
    ) - modal_power_spectrum(monitor.result.outgoing, monitor.power_matrix)

    assert spectrum.representation == "modal_power_waves"
    assert_allclose(spectrum.accepted_power, expected_accepted)
    if is_source:
        driven = np.zeros_like(monitor.result.incident)
        driven[0] = monitor.result.incident[0]
        assert_allclose(
            spectrum.incident_power,
            modal_power_spectrum(driven, monitor.power_matrix),
        )
    else:
        assert_allclose(spectrum.incident_power, 0)
    assert spectrum.terminal_valid.all()


def test_lossy_eigenmode_accepted_power_keeps_interference_term(monkeypatch):
    monitor = _eigenmode_port(is_source=True)
    monitor.mode_indices = (1,)
    monitor.mode_power_valid = np.ones((1, 1), dtype=bool)
    monitor.power_matrix_valid = np.ones(1, dtype=bool)
    monitor.power_matrix = np.asarray([[[1.0]]], dtype=np.complex128)
    monitor.electric_gram = np.asarray([[[1.0 + 0.4j]]], dtype=np.complex128)
    incident = 1.2 + 0.7j
    outgoing = -0.3 + 0.8j
    monitor.result = EigenmodePortResult(
        frequency=np.asarray([5.0]),
        incident=np.asarray([[incident]]),
        outgoing=np.asarray([[outgoing]]),
        valid=np.ones((1, 1), dtype=bool),
        condition_number=np.ones(1),
    )
    monkeypatch.setattr(
        ports,
        "_port_mesh_valid",
        lambda output, grid, frequency: np.ones(frequency.shape, dtype=bool),
    )

    spectrum = evaluate_port_power_spectrum(monitor, SimpleNamespace(), [5.0])
    expected = np.real(
        np.conj(incident - outgoing) * monitor.electric_gram[0, 0, 0] * (incident + outgoing)
    )
    lossy_formula_without_interference = abs(incident) ** 2 - abs(outgoing) ** 2

    assert_allclose(spectrum.accepted_power, [expected])
    assert not np.isclose(expected, lossy_formula_without_interference)


def test_nyquist_research_override_retains_full_port_mesh_band(monkeypatch):
    output = SimpleNamespace(
        spectrum_limit="nyquist",
        minimum_wavelength_cells=10.0,
    )
    monkeypatch.setattr(
        ports,
        "minimum_wavelength_sampling",
        lambda grid, frequency: (
            np.zeros(np.asarray(frequency).shape),
            np.full(np.asarray(frequency).shape, "material"),
        ),
    )

    assert ports._port_mesh_valid(output, SimpleNamespace(), [1.0, 2.0]).all()


def test_zero_amplitude_voltage_port_retains_negative_coupled_power(monkeypatch):
    dt = 1e-3
    nsamples = 32
    output, grid = _voltage_port(
        np.full(nsamples, 2.0),
        np.zeros(nsamples),
        dt=dt,
    )
    monkeypatch.setattr(
        ports,
        "_port_mesh_valid",
        lambda output, grid, frequency: np.ones(frequency.shape, dtype=bool),
    )

    result = evaluate_port_power_spectrum(output, grid, [0.0])
    voltage_spectrum = 2 * nsamples * dt

    assert result.incident_power[0] == 0
    assert result.accepted_power[0] < 0
    assert_allclose(
        result.accepted_power[0],
        -(voltage_spectrum**2) / (2 * output.reference_impedance),
    )
    assert result.terminal_valid[0]


def test_voltage_port_terminal_current_reproduces_gap_corrected_s11(monkeypatch):
    dt = 1e-3
    nsamples = 64
    frequency = 1 / (nsamples * dt)
    time = (np.arange(nsamples) + 0.5) * dt
    generator = 4.0 * np.cos(2 * np.pi * frequency * time)
    total = 1.4 * np.cos(2 * np.pi * frequency * time + 0.35)
    output, grid = _voltage_port(
        total,
        generator,
        dt=dt,
        capacitance=2e-3,
    )
    monkeypatch.setattr(
        ports,
        "_port_mesh_valid",
        lambda output, grid, values: np.ones(values.shape, dtype=bool),
    )

    result = evaluate_port_power_spectrum(output, grid, [frequency])
    generator_spectrum = engineering_dft(
        generator,
        [frequency],
        dt,
        time_offset=0.5 * dt,
    )
    total_spectrum = engineering_dft(
        total,
        [frequency],
        dt,
        time_offset=0.5 * dt,
    )
    source_s11 = (total_spectrum - 0.5 * generator_spectrum) / (0.5 * generator_spectrum)
    omega_discrete = (2 / dt) * np.tan(np.pi * frequency * dt)
    correction = output.reference_impedance * 1j * omega_discrete * output.gap_capacitance
    expected_s11, _ = ports.correct_s11_for_parallel_gap(source_s11, correction)
    terminal_s11 = (
        result.terminal_voltage - output.reference_impedance * result.terminal_current
    ) / (result.terminal_voltage + output.reference_impedance * result.terminal_current)

    assert_allclose(terminal_s11, expected_s11, rtol=2e-13, atol=2e-13)


def test_rational_network_port_uses_external_network_current_sign(monkeypatch):
    dt = 1e-3
    nsamples = 64
    frequency = 1 / (nsamples * dt)
    time = (np.arange(nsamples) + 0.5) * dt
    voltage = 2.0 * np.cos(2 * np.pi * frequency * time + 0.2)
    # Inetwork is defined from the FDTD gap into the external network, so a
    # positive current entering the antenna is stored with the opposite sign.
    terminal_current = 0.04 * np.cos(2 * np.pi * frequency * time - 0.1)
    output, grid = _rational_network_port(
        voltage,
        -terminal_current,
        dt=dt,
    )
    monkeypatch.setattr(
        ports,
        "_port_mesh_valid",
        lambda output, grid, values: np.ones(values.shape, dtype=bool),
    )

    result = evaluate_port_power_spectrum(output, grid, [frequency])
    expected_voltage = engineering_dft(voltage, [frequency], dt, time_offset=0.5 * dt)
    expected_current = engineering_dft(
        terminal_current,
        [frequency],
        dt,
        time_offset=0.5 * dt,
    )

    assert_allclose(result.terminal_voltage, expected_voltage)
    assert_allclose(result.terminal_current, expected_current)
    assert result.accepted_power[0] > 0


def test_hard_voltage_port_uses_time_aligned_loop_current(monkeypatch):
    dt = 1e-3
    nsamples = 64
    frequency = 1 / (nsamples * dt)
    voltage_time = (np.arange(nsamples) + 1) * dt
    current_time = (np.arange(nsamples) + 0.5) * dt
    voltage = 2.0 * np.cos(2 * np.pi * frequency * voltage_time + 0.2)
    current = 0.04 * np.cos(2 * np.pi * frequency * current_time - 0.1)
    output, grid = _hard_voltage_port(voltage, current, dt=dt)
    monkeypatch.setattr(
        ports,
        "_port_mesh_valid",
        lambda output, grid, values: np.ones(values.shape, dtype=bool),
    )

    result = evaluate_port_power_spectrum(output, grid, [frequency])
    expected_voltage = engineering_dft(voltage, [frequency], dt, time_offset=dt)
    expected_current = engineering_dft(
        current,
        [frequency],
        dt,
        time_offset=0.5 * dt,
    )
    expected_incident = 0.5 * (expected_voltage + output.reference_impedance * expected_current)

    assert_allclose(result.terminal_voltage, expected_voltage)
    assert_allclose(result.terminal_current, expected_current)
    assert_allclose(result.incident_voltage, expected_incident)
    assert result.terminal_valid[0]


@pytest.mark.parametrize("output_type", [TransmissionLinePortOutput, MagneticFrillPortOutput])
def test_automatic_source_ports_use_their_terminal_wave_definitions(monkeypatch, output_type):
    dt = 1e-3
    nsamples = 32
    frequency = 1 / (nsamples * dt)
    time = np.arange(nsamples) * dt
    incident = np.cos(2 * np.pi * frequency * time)
    terminal = 1.4 * np.cos(2 * np.pi * frequency * time + 0.2)
    current = 0.03 * np.cos(2 * np.pi * frequency * time - 0.1)
    source = SimpleNamespace(
        Vinc=incident,
        Vtotal=terminal,
        Itot=current,
    )
    output = output_type.__new__(output_type)
    output.source = source
    output.source_index = 1
    output.result = SimpleNamespace()
    output.reference_impedance = 50.0
    output.minimum_wavelength_cells = 10.0
    grid = SimpleNamespace(dt=dt)
    monkeypatch.setattr(
        ports,
        "_port_mesh_valid",
        lambda output, grid, values: np.ones(values.shape, dtype=bool),
    )

    result = evaluate_port_power_spectrum(output, grid, [frequency])
    expected_incident = engineering_dft(incident, [frequency], dt)
    expected_terminal = engineering_dft(terminal, [frequency], dt)
    if output_type is TransmissionLinePortOutput:
        expected_current = (2 * expected_incident - expected_terminal) / 50.0
    else:
        expected_current = engineering_dft(current, [frequency], dt)

    assert_allclose(result.incident_voltage, expected_incident)
    assert_allclose(result.terminal_voltage, expected_terminal)
    assert_allclose(result.terminal_current, expected_current)
    assert result.terminal_valid[0]
