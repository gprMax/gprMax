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

"""Rational-network recurrence and small CPU integration tests."""

import h5py
import numpy as np
import pytest
from scipy.integrate import quad

import gprMax
from gprMax.cython.network_port import update_rational_network_terminal
from gprMax.hash_cmds_file import get_user_objects
from gprMax.network_ports import RationalNetworkModel, linear_interval_coefficients
from gprMax.user_objects.cmds_multiuse import NetworkExcitation, NetworkTerminal, RationalNetwork
from gprMax.user_objects.cmds_output import NetworkPort


def _quadrature_state(pole, residue, dt, fraction, state, old, new):
    upper = fraction * dt

    def integrand(time):
        value = old + (new - old) * time / dt
        return np.exp(pole * (upper - time)) * residue * value

    real = quad(lambda time: np.real(integrand(time)), 0, upper, epsabs=1e-14)[0]
    imag = quad(lambda time: np.imag(integrand(time)), 0, upper, epsabs=1e-14)[0]
    return np.exp(pole * upper) * state + real + 1j * imag


@pytest.mark.parametrize("fraction", (0.5, 1.0))
@pytest.mark.parametrize("pole", (0j, -2.5e9 + 0j, -1.5e9 + 4e9j))
def test_linear_interval_coefficients_match_direct_convolution(fraction, pole):
    residue = 3.2e8 - 1.1e8j
    dt = 7e-11
    state = 0.2 - 0.4j
    old = -0.7
    new = 1.3

    exponential, coeff_new, coeff_old = linear_interval_coefficients(
        pole,
        residue,
        dt,
        fraction,
    )
    calculated = exponential * state + coeff_new * new + coeff_old * old
    expected = _quadrature_state(pole, residue, dt, fraction, state, old, new)

    np.testing.assert_allclose(calculated, expected, rtol=2e-13, atol=2e-13)


def test_rational_model_matches_series_rl_admittance():
    resistance = 37.0
    inductance = 2.4e-9
    model = RationalNetworkModel(
        "series_rl",
        poles=(-resistance / inductance,),
        residues=(1 / inductance,),
    )
    frequency = np.geomspace(1e6, 20e9, 301)

    expected = 1 / (resistance + 2j * np.pi * frequency * inductance)
    np.testing.assert_allclose(model.admittance(frequency), expected, rtol=2e-15)
    model.validate_passivity(frequency)


def test_rational_model_matches_parallel_rlc_admittance():
    resistance = 75.0
    inductance = 4.7e-9
    capacitance = 1.2e-12
    model = RationalNetworkModel(
        "parallel_rlc",
        conductance=1 / resistance,
        capacitance=capacitance,
        poles=(0,),
        residues=(1 / inductance,),
    )
    frequency = np.geomspace(1e6, 40e9, 401)

    omega = 2 * np.pi * frequency
    expected = 1 / resistance + 1j * omega * capacitance + 1 / (1j * omega * inductance)
    np.testing.assert_allclose(model.admittance(frequency), expected, rtol=3e-15)


def test_exact_half_step_recurrence_reduces_series_rl_phasor_error():
    """Compare the new analytic half-step current with classic midpoint PLRC."""

    resistance = 10 * np.pi
    inductance = 10e-9
    frequency = 1e9
    samples_per_period = 10
    dt = 1 / (frequency * samples_per_period)
    omega = 2 * np.pi * frequency
    pole = -resistance / inductance
    residue = 1 / inductance
    exp_half, half_new, half_old = linear_interval_coefficients(pole, residue, dt, 0.5)
    exp_full, full_new, full_old = linear_interval_coefficients(pole, residue, dt, 1.0)

    state = 0j
    exact_half = []
    classic_half = []
    times = []
    for iteration in range(120 * samples_per_period):
        old = np.sin(omega * iteration * dt)
        new = np.sin(omega * (iteration + 1) * dt)
        analytic_half = exp_half * state + half_new * new + half_old * old
        state_new = exp_full * state + full_new * new + full_old * old
        if iteration >= 100 * samples_per_period:
            exact_half.append(analytic_half.real)
            classic_half.append(0.5 * (state + state_new).real)
            times.append((iteration + 0.5) * dt)
        state = state_new

    basis = np.column_stack((np.sin(omega * np.asarray(times)), np.cos(omega * np.asarray(times))))
    exact_phasor = np.linalg.lstsq(basis, exact_half, rcond=None)[0]
    classic_phasor = np.linalg.lstsq(basis, classic_half, rcond=None)[0]
    expected = 1 / (resistance + 1j * omega * inductance)
    expected_phasor = np.asarray((expected.real, expected.imag))
    exact_error = np.linalg.norm(exact_phasor - expected_phasor) / abs(expected)
    classic_error = np.linalg.norm(classic_phasor - expected_phasor) / abs(expected)

    assert exact_error < 0.04
    assert exact_error < 0.5 * classic_error


def test_complex_terms_require_conjugate_pole_residue_pairs():
    with pytest.raises(ValueError, match="conjugate pairs"):
        RationalNetworkModel(
            "invalid",
            poles=(-1e9 + 2e9j,),
            residues=(3e8 + 4e8j,),
        )


def test_hash_commands_create_reusable_network_terminal_excitation_and_port():
    objects = get_user_objects(
        [
            "#rational_network: series_rl 0 0 1 -2e9 0 5e7 0\n",
            "#network_terminal: z 0.01 0.02 0.03 series_rl feed\n",
            "#network_excitation: feed pulse 1e-10 9e-10\n",
            "#network_port: feed 75 nyquist\n",
        ],
        checkessential=False,
    )

    assert [type(item) for item in objects] == [
        RationalNetwork,
        NetworkTerminal,
        NetworkExcitation,
        NetworkPort,
    ]
    assert objects[0].poles == (-2e9 + 0j,)
    assert objects[0].residues == (5e7 + 0j,)
    assert objects[2].start == pytest.approx(1e-10)
    assert objects[3].reference_impedance == pytest.approx(75)
    assert objects[3].spectrum_limit == "nyquist"


def test_cython_kernel_resistor_sign_and_implicit_edge_update():
    electric = np.zeros((2, 2, 2), dtype=np.float32)
    electric[1, 1, 1] = -20
    empty = np.empty(0, dtype=np.complex64)
    conductance = 0.02
    dl = 0.01
    area = 1e-4
    source_coefficient = 0.2
    alpha = conductance / 2
    denominator = 1 + source_coefficient * alpha * dl / area

    current = update_rational_network_terminal(
        1,
        1,
        1,
        dl,
        area,
        source_coefficient,
        denominator,
        alpha,
        conductance,
        0,
        1,
        0.3,
        1,
        1,
        1,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
        electric,
    )

    assert electric[1, 1, 1] == pytest.approx(-45)
    assert current == pytest.approx(-0.0125)


def _scene(use_network):
    dl = 0.002
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    if use_network:
        scene.add(gprMax.RationalNetwork(id="source_50", conductance=1 / 50))
        scene.add(
            gprMax.NetworkTerminal(
                p1=(0.01, 0.01, 0.01),
                polarisation="z",
                network_id="source_50",
                id="feed",
            )
        )
        scene.add(gprMax.NetworkExcitation("feed", "pulse"))
        port = gprMax.NetworkPort("feed", reference_impedance=50)
        scene.add(port)
    else:
        port = gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse", id="feed")
        scene.add(port)
    scene.add(gprMax.Rx((0.012, 0.01, 0.01), id="field"))
    return scene, port


def _run_scene(tmp_path, name, use_network):
    scene, port = _scene(use_network)
    output = tmp_path / name
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
        cpu_precision="single",
    )
    return output.with_suffix(".h5"), port


@pytest.mark.integration
def test_pure_resistance_network_matches_finite_resistance_voltage_source(tmp_path):
    network_file, network_port = _run_scene(tmp_path, "network", True)
    voltage_file, _ = _run_scene(tmp_path, "voltage", False)

    with h5py.File(network_file, "r") as network, h5py.File(voltage_file, "r") as voltage:
        np.testing.assert_allclose(
            network["rxs/rx1/Ez"][...],
            voltage["rxs/rx1/Ez"][...],
            rtol=2e-5,
            atol=2e-7,
        )
        port = network["ports/feed"]
        assert port.attrs["SourceType"] == "RationalNetworkTerminal"
        assert port.attrs["NetworkModelID"] == "source_50"
        assert port["S11"].shape == port["frequency"].shape
        assert port["valid_S11"][...].astype(bool).any()
        np.testing.assert_allclose(
            port["Zin"][...][port["valid_Zin"][...].astype(bool)],
            network_port.result.zin[network_port.result.valid_zin],
        )


@pytest.mark.integration
def test_rational_network_terminal_and_port_run_inside_subgrid(tmp_path):
    output = tmp_path / "subgrid_network"
    source_position = (0.045, 0.045, 0.045)
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=3,
        id="fine_grid",
    )
    scene.add(subgrid)
    subgrid.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    subgrid.add(gprMax.RationalNetwork(id="source_50", conductance=1 / 50))
    subgrid.add(
        gprMax.NetworkTerminal(
            p1=source_position,
            polarisation="z",
            network_id="source_50",
            id="feed",
        )
    )
    subgrid.add(gprMax.NetworkExcitation("feed", "pulse"))
    port = gprMax.NetworkPort("feed", reference_impedance=50, spectrum_limit="nyquist")
    subgrid.add(port)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="single",
    )

    with h5py.File(output.with_suffix(".h5"), "r") as result:
        assert result.attrs["nsrc"] == 0
        fine = result["subgrids/fine_grid"]
        assert fine.attrs["nsrc"] == 1
        assert fine.attrs["nports"] == 1
        np.testing.assert_allclose(fine["srcs/src1"].attrs["Position"], source_position)
        network_port = fine["ports/feed"]
        np.testing.assert_allclose(network_port.attrs["Position"], source_position)
        assert np.max(np.abs(network_port["Vtotal"][...])) > 0
        assert np.max(np.abs(network_port["Inetwork"][...])) > 0
        assert network_port["valid_S11"][...].astype(bool).any()
        assert port.result.valid_zin.any()


@pytest.mark.integration
def test_multiple_independent_network_ports_share_one_fdtd_model(tmp_path):
    output = tmp_path / "two_network_ports"
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.024, 0.02, 0.02)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(gprMax.RationalNetwork(id="port_50", conductance=1 / 50))
    scene.add(gprMax.RationalNetwork(id="port_75", conductance=1 / 75))
    for terminal_id, network_id, position, impedance in (
        ("feed1", "port_50", (0.008, 0.01, 0.01), 50),
        ("feed2", "port_75", (0.016, 0.01, 0.01), 75),
    ):
        scene.add(
            gprMax.NetworkTerminal(
                p1=position,
                polarisation="z",
                network_id=network_id,
                id=terminal_id,
            )
        )
        scene.add(gprMax.NetworkExcitation(terminal_id, "pulse"))
        scene.add(gprMax.NetworkPort(terminal_id, reference_impedance=impedance))

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
        cpu_precision="single",
    )

    with h5py.File(output.with_suffix(".h5"), "r") as result:
        assert result.attrs["nsrc"] == 2
        assert result.attrs["nports"] == 2
        assert set(result["ports"]) == {"feed1", "feed2"}
        for terminal_id in ("feed1", "feed2"):
            port = result[f"ports/{terminal_id}"]
            assert np.max(np.abs(port["Vtotal"][...])) > 0
            assert np.max(np.abs(port["Inetwork"][...])) > 0
