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

"""Canonical analytical and FDTD checks for rational network terminals."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.cython.network_port import update_rational_network_terminal
from gprMax.network_ports import RationalNetworkModel, linear_interval_coefficients


def _series_rlc_terms(resistance, inductance, capacitance):
    alpha = -resistance / (2 * inductance)
    beta = np.sqrt(1 / (inductance * capacitance) - alpha**2)
    pole_positive = alpha + 1j * beta
    pole_negative = np.conj(pole_positive)
    residue_positive = pole_positive / (inductance * (pole_positive - pole_negative))
    return (pole_positive, pole_negative), (residue_positive, np.conj(residue_positive))


def test_canonical_r_c_l_and_series_network_admittances():
    resistance = 50.0
    capacitance = 0.8e-12
    inductance = 3.2e-9
    frequency = np.geomspace(1e6, 30e9, 501)
    s = 2j * np.pi * frequency

    models_and_expected = (
        (
            RationalNetworkModel("resistor", conductance=1 / resistance),
            np.full(frequency.shape, 1 / resistance, dtype=np.complex128),
        ),
        (
            RationalNetworkModel("capacitor", capacitance=capacitance),
            s * capacitance,
        ),
        (
            RationalNetworkModel("inductor", poles=(0,), residues=(1 / inductance,)),
            1 / (s * inductance),
        ),
        (
            RationalNetworkModel(
                "series_rc",
                conductance=1 / resistance,
                poles=(-1 / (resistance * capacitance),),
                residues=(-1 / (resistance**2 * capacitance),),
            ),
            1 / (resistance + 1 / (s * capacitance)),
        ),
        (
            RationalNetworkModel(
                "series_rl",
                poles=(-resistance / inductance,),
                residues=(1 / inductance,),
            ),
            1 / (resistance + s * inductance),
        ),
    )

    for model, expected in models_and_expected:
        # Series RC requires cancellation of its direct and pole terms at low
        # frequency, so use a tolerance appropriate to that conditioning.
        np.testing.assert_allclose(model.admittance(frequency), expected, rtol=2e-12, atol=1e-17)
        model.validate_passivity(frequency)


def test_canonical_series_rlc_resonance_matches_closed_form():
    resistance = 18.0
    inductance = 2.1e-9
    capacitance = 0.48e-12
    poles, residues = _series_rlc_terms(resistance, inductance, capacitance)
    model = RationalNetworkModel("series_rlc", poles=poles, residues=residues)
    resonant_frequency = 1 / (2 * np.pi * np.sqrt(inductance * capacitance))
    frequency = np.geomspace(resonant_frequency / 8, resonant_frequency * 8, 801)
    s = 2j * np.pi * frequency

    expected = 1 / (resistance + s * inductance + 1 / (s * capacitance))
    calculated = model.admittance(frequency)

    np.testing.assert_allclose(calculated, expected, rtol=2e-14, atol=2e-17)
    peak_index = int(np.argmax(np.abs(calculated)))
    assert frequency[peak_index] == pytest.approx(resonant_frequency, rel=3e-3)
    assert calculated[peak_index].real == pytest.approx(1 / resistance, rel=3e-3)
    assert abs(calculated[peak_index].imag) < 3e-3 / resistance
    model.validate_passivity(frequency)


@pytest.mark.parametrize(
    ("samples_per_period", "maximum_exact_error"),
    ((6, 0.09), (10, 0.04), (20, 0.009), (40, 0.0022)),
)
def test_analytic_half_step_outperforms_classic_plrc_over_resolution(
    samples_per_period, maximum_exact_error
):
    resistance = 10 * np.pi
    inductance = 10e-9
    frequency = 1e9
    omega = 2 * np.pi * frequency
    dt = 1 / (frequency * samples_per_period)
    pole = -resistance / inductance
    residue = 1 / inductance
    exp_half, half_new, half_old = linear_interval_coefficients(pole, residue, dt, 0.5)
    exp_full, full_new, full_old = linear_interval_coefficients(pole, residue, dt, 1.0)

    state = 0j
    exact_current = []
    classic_current = []
    sample_times = []
    for iteration in range(120 * samples_per_period):
        voltage_old = np.sin(omega * iteration * dt)
        voltage_new = np.sin(omega * (iteration + 1) * dt)
        state_half = exp_half * state + half_new * voltage_new + half_old * voltage_old
        state_new = exp_full * state + full_new * voltage_new + full_old * voltage_old
        if iteration >= 100 * samples_per_period:
            exact_current.append(state_half.real)
            classic_current.append(0.5 * (state + state_new).real)
            sample_times.append((iteration + 0.5) * dt)
        state = state_new

    basis = np.column_stack(
        (np.sin(omega * np.asarray(sample_times)), np.cos(omega * np.asarray(sample_times)))
    )
    expected = 1 / (resistance + 1j * omega * inductance)
    expected_phasor = np.asarray((expected.real, expected.imag))
    errors = []
    for current in (exact_current, classic_current):
        phasor = np.linalg.lstsq(basis, current, rcond=None)[0]
        errors.append(np.linalg.norm(phasor - expected_phasor) / abs(expected))

    exact_error, classic_error = errors
    assert exact_error < maximum_exact_error
    assert exact_error < 0.41 * classic_error


@pytest.mark.parametrize("network", ("series_rl", "series_rc"))
def test_canonical_first_order_step_response_is_exact(network):
    resistance = 40.0
    inductance = 4e-9
    capacitance = 1e-12
    dt = 8e-12
    if network == "series_rl":
        pole = -resistance / inductance
        residue = 1 / inductance
        conductance = 0.0
        expected = lambda time: (1 / resistance) * (1 - np.exp(-resistance * time / inductance))
    else:
        pole = -1 / (resistance * capacitance)
        residue = -1 / (resistance**2 * capacitance)
        conductance = 1 / resistance
        expected = lambda time: (1 / resistance) * np.exp(-time / (resistance * capacitance))

    exp_half, coeff_half_new, coeff_half_old = linear_interval_coefficients(pole, residue, dt, 0.5)
    exp_full, coeff_full_new, coeff_full_old = linear_interval_coefficients(pole, residue, dt, 1.0)
    state = 0j
    for iteration in range(80):
        half_state = exp_half * state + coeff_half_new + coeff_half_old
        time = (iteration + 0.5) * dt
        assert conductance + half_state.real == pytest.approx(expected(time), rel=3e-14, abs=2e-16)
        state = exp_full * state + coeff_full_new + coeff_full_old


def test_canonical_capacitor_local_implicit_update():
    electric = np.zeros((2, 2, 2), dtype=np.float64)
    electric[1, 1, 1] = -20.0
    empty = np.empty(0, dtype=np.complex128)
    capacitance = 1e-12
    dt = 1e-10
    dl = 0.01
    area = 1e-4
    source_coefficient = 0.2
    voltage_old = 0.1
    generator_old = 0.4
    generator_new = 0.6
    generator_half = 0.5
    alpha = capacitance / dt
    denominator = 1 + source_coefficient * alpha * dl / area
    history = -capacitance * voltage_old / dt - capacitance * (generator_new - generator_old) / dt
    expected_electric = (electric[1, 1, 1] + source_coefficient * history / area) / denominator
    expected_voltage = -dl * expected_electric
    expected_current = (
        capacitance * (expected_voltage - voltage_old - generator_new + generator_old) / dt
    )

    current = update_rational_network_terminal(
        1,
        1,
        1,
        dl,
        area,
        source_coefficient,
        denominator,
        alpha,
        0.0,
        capacitance,
        dt,
        voltage_old,
        generator_old,
        generator_new,
        generator_half,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        electric,
    )

    assert electric[1, 1, 1] == pytest.approx(expected_electric, rel=2e-15)
    assert current == pytest.approx(expected_current, rel=2e-15, abs=1e-17)


def test_nonpassive_network_is_rejected_unless_explicitly_enabled():
    frequency = np.geomspace(1e6, 20e9, 301)
    model = RationalNetworkModel("active", conductance=0.01, poles=(-1e9,), residues=(-4e7,))
    with pytest.raises(ValueError, match="non-passive"):
        model.validate_passivity(frequency)

    active_model = RationalNetworkModel(
        "active_allowed",
        conductance=0.01,
        poles=(-1e9,),
        residues=(-4e7,),
        allow_active=True,
    )
    active_model.validate_passivity(frequency)


@pytest.mark.integration
def test_canonical_passive_networks_remain_stable_in_fdtd_and_ports_are_consistent(tmp_path):
    output = tmp_path / "canonical_networks"
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.032, 0.024, 0.024)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=2e-9))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))

    rlc_poles, rlc_residues = _series_rlc_terms(25.0, 2e-9, 0.5e-12)
    network_specs = (
        ("resistor", {"conductance": 1 / 50}),
        ("capacitor", {"capacitance": 0.2e-12}),
        ("inductor", {"poles": (0,), "residues": (1 / 2e-9,)}),
        ("series_rlc", {"poles": rlc_poles, "residues": rlc_residues}),
    )
    positions = (0.008, 0.014, 0.020, 0.026)
    for (terminal_id, kwargs), x_position in zip(network_specs, positions):
        scene.add(gprMax.RationalNetwork(id=terminal_id, **kwargs))
        scene.add(
            gprMax.NetworkTerminal(
                p1=(x_position, 0.012, 0.012),
                polarisation="z",
                network_id=terminal_id,
                id=terminal_id,
            )
        )
        scene.add(gprMax.NetworkExcitation(terminal_id, "pulse", stop=0.5e-9))
        scene.add(gprMax.NetworkPort(terminal_id, reference_impedance=50, spectrum_limit="nyquist"))

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(output.with_suffix(".h5"), "r") as result:
        assert result.attrs["nsrc"] == len(network_specs)
        assert result.attrs["nports"] == len(network_specs)
        for terminal_id, _ in network_specs:
            port = result[f"ports/{terminal_id}"]
            voltage = port["Vtotal"][...]
            current = port["Inetwork"][...]
            assert np.isfinite(voltage).all()
            assert np.isfinite(current).all()
            assert np.max(np.abs(voltage)) < 100
            assert np.max(np.abs(current)) < 100
            assert np.max(np.abs(voltage[-50:])) <= np.max(np.abs(voltage))

            valid_s11 = port["valid_S11"][...].astype(bool)
            valid_zin = port["valid_Zin"][...].astype(bool)
            incident = port["Vincident_spectrum"][...]
            reflected = port["Vreflected_spectrum"][...]
            total_spectrum = port["Vtotal_spectrum"][...]
            terminal_current = port["Iterminal_spectrum"][...]
            np.testing.assert_allclose(
                port["S11"][...][valid_s11],
                (reflected / incident)[valid_s11],
                rtol=2e-13,
                atol=2e-13,
            )
            np.testing.assert_allclose(
                port["Zin"][...][valid_zin],
                (total_spectrum / terminal_current)[valid_zin],
                rtol=2e-13,
                atol=2e-13,
            )
