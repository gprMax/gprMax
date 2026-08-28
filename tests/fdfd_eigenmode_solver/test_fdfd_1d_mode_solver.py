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
from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver


@pytest.fixture(autouse=True)
def _solver_constants(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "e0": 8.8541878128e-12,
                "m0": 1.25663706212e-6,
                "c": 299792458.0,
                "z0": 376.73031366686166,
            }
        ),
    )


def _solver(polarization, n=80, frequency=10e9, **kwargs):
    defaults = {
        "frequency": frequency,
        "dt": 0.5e-3,
        "mode_index": 0,
        "polarization": polarization,
        "eps_r_t": np.ones(n),
        "eps_r_a": np.ones(n + 1),
        "eps_r_w": np.ones(n + 1),
        "mu_r_t": np.ones(n + 1),
        "mu_r_a": np.ones(n),
        "mu_r_w": np.ones(n),
    }
    defaults.update(kwargs)
    return FDFD_1D_mode_solver(**defaults)


@pytest.mark.parametrize("solver_class", [FDFD_1D_mode_solver, FDFD_2D_mode_solver])
def test_passive_neff_branch_preserves_loss_and_evanescent_decay(solver_class):
    epsilon = 9 - 7.19004143381j
    expected = np.sqrt(epsilon)

    neff = solver_class._passive_positive_neff(epsilon)
    evanescent_neff = solver_class._passive_positive_neff(-4 + 0j)

    assert neff == pytest.approx(expected)
    assert np.real(neff) > 0
    assert np.imag(neff) < 0
    assert evanescent_neff == pytest.approx(-2j)


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_lossy_homogeneous_mode_uses_passive_forward_neff(polarization):
    n = 40
    frequency = 5e9
    epsilon = 9 - 1j * 2 / (2 * np.pi * frequency * 8.8541878128e-12)
    kwargs = {
        "eps_r_t": np.full(n, epsilon),
        "eps_r_a": np.full(n + 1, epsilon),
        "eps_r_w": np.full(n + 1, epsilon),
    }
    if polarization == "TE":
        pec_w = np.zeros(n + 1, dtype=bool)
        pec_w[[0, -1]] = True
        kwargs["pec_w_mask"] = pec_w

    solver = _solver(
        polarization,
        n=n,
        frequency=frequency,
        **kwargs,
    )
    solver.solve()

    expected = np.sqrt(epsilon)
    forward_factor = np.exp(-1j * solver.k0 * solver.modal_complex_neff * 0.5e-3)
    assert solver.modal_complex_neff == pytest.approx(expected, rel=1e-9)
    assert np.real(solver.modal_complex_neff) > 0
    assert np.imag(solver.modal_complex_neff) < 0
    assert abs(forward_factor) < 1
    assert solver.modal_power_valid
    assert solver.modal_forward_power_metric > solver.FORWARD_POWER_METRIC_TOLERANCE
    assert np.real(solver.modal_raw_power) > 0
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)
    if polarization == "TM":
        active = np.abs(solver.modal_Ea) > 1e-12 * np.max(np.abs(solver.modal_Ea))
        ratio = solver.modal_Ht[active] / solver.modal_Ea[active]
        expected_ratio = -solver.modal_complex_neff / solver.eta0
    else:
        active = np.abs(solver.modal_Ha) > 1e-12 * np.max(np.abs(solver.modal_Ha))
        ratio = solver.modal_Et[active] / solver.modal_Ha[active]
        expected_ratio = solver.eta0 * solver.modal_complex_neff / epsilon
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_backward_wave_uses_passive_forward_power_branch(polarization):
    n = 40
    frequency = 5e9
    material = -2 - 0.05j
    solver = _solver(
        polarization,
        n=n,
        frequency=frequency,
        eps_r_t=np.full(n, material),
        eps_r_a=np.full(n + 1, material),
        eps_r_w=np.full(n + 1, material),
        mu_r_t=np.full(n + 1, material),
        mu_r_a=np.full(n, material),
        mu_r_w=np.full(n, material),
    )

    solver.solve()

    forward_factor = np.exp(-1j * solver.k0 * solver.modal_complex_neff * 0.5e-3)
    assert np.real(solver.modal_complex_neff) < 0
    assert np.imag(solver.modal_complex_neff) < 0
    assert abs(forward_factor) < 1
    assert solver.modal_power_valid
    assert solver.modal_forward_power_metric > solver.FORWARD_POWER_METRIC_TOLERANCE
    assert np.real(solver.modal_raw_power) > 0
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)


def test_below_cutoff_mode_is_kept_with_finite_balanced_normalization():
    n = 40
    pec_a = np.zeros(n + 1, dtype=bool)
    pec_a[[0, -1]] = True
    solver = _solver(
        "TM",
        n=n,
        frequency=5e9,
        pec_a_mask=pec_a,
    )

    solver.solve()

    assert solver.modal_complex_neff.real == 0
    assert solver.modal_complex_neff.imag < 0
    assert not solver.modal_power_valid
    assert abs(solver.modal_forward_power_metric) < solver.FORWARD_POWER_METRIC_TOLERANCE
    assert abs(solver.modal_raw_power.imag) > abs(solver.modal_raw_power.real)
    assert solver._calculate_mode_balanced_power(0) == pytest.approx(1.0, rel=1e-12)
    assert solver.modal_power == pytest.approx(solver.modal_forward_power_metric, abs=1e-14)
    for field in (
        solver.modal_Et,
        solver.modal_Ea,
        solver.modal_Ew,
        solver.modal_Ht,
        solver.modal_Ha,
        solver.modal_Hw,
    ):
        assert np.all(np.isfinite(field))


def test_propagation_validity_is_classified_per_mode():
    n = 40
    pec_a = np.zeros(n + 1, dtype=bool)
    pec_a[[0, -1]] = True
    solver = _solver(
        "TM",
        n=n,
        frequency=10e9,
        mode_index=1,
        pec_a_mask=pec_a,
    )

    solver.solve()

    np.testing.assert_array_equal(solver.power_valid, (True, False))
    assert solver.powers[0] == pytest.approx(1.0, rel=1e-12)
    assert abs(solver.powers[1]) < solver.FORWARD_POWER_METRIC_TOLERANCE
    assert np.all(np.isfinite(solver.Ea))
    assert np.all(np.isfinite(solver.Ht))


def test_tm_pec_parallel_plate_neff_and_yee_shapes():
    n = 80
    pec_a = np.zeros(n + 1, dtype=bool)
    pec_a[[0, -1]] = True
    solver = _solver("TM", n=n, pec_a_mask=pec_a)
    solver.solve()

    width = n * solver.dt
    expected = np.sqrt(1 - (np.pi / (solver.k0 * width)) ** 2)
    assert solver.modal_real_neff == pytest.approx(expected, rel=2e-3)
    assert solver.modal_Ea.shape == (n + 1,)
    assert solver.modal_Ht.shape == (n + 1,)
    assert solver.modal_Hw.shape == (n,)
    assert solver.modal_Ea[0] == 0
    assert solver.modal_Ea[-1] == 0
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)


def test_te_uniform_mode_uses_pec_longitudinal_boundary():
    n = 60
    pec_w = np.zeros(n + 1, dtype=bool)
    pec_w[[0, -1]] = True
    solver = _solver("TE", n=n, pec_w_mask=pec_w)
    solver.solve()

    assert solver.modal_real_neff == pytest.approx(1.0, rel=1e-10)
    assert solver.modal_Ha.shape == (n,)
    assert solver.modal_Et.shape == (n,)
    assert solver.modal_Ew.shape == (n + 1,)
    assert np.max(np.abs(solver.modal_Ew)) < 1e-8
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)


def test_default_guess_targets_magnetic_medium_fundamental():
    n = 80
    epsilon = 4.0
    permeability = 4.0
    solver = _solver(
        "TE",
        n=n,
        eps_r_t=np.full(n, epsilon),
        eps_r_a=np.full(n + 1, epsilon),
        eps_r_w=np.full(n + 1, epsilon),
        mu_r_t=np.full(n + 1, permeability),
        mu_r_a=np.full(n, permeability),
        mu_r_w=np.full(n, permeability),
    )

    assert solver.guess == pytest.approx(-(epsilon * permeability))
    solver.solve()

    assert solver.modal_real_neff == pytest.approx(
        np.sqrt(epsilon * permeability),
        rel=5e-3,
    )


@pytest.mark.parametrize("polarization", ("TM", "TE"))
def test_real_profile_alignment_canonicalizes_global_mode_sign(polarization):
    phase = np.exp(0.37j)
    aligned_fields = []
    for sign in (1.0, -1.0):
        solver = object.__new__(FDFD_1D_mode_solver)
        solver.num_modes = 1
        solver.polarization = polarization
        solver.dt = 1.0
        solver.Et = np.zeros((2, 1), dtype=np.complex128)
        solver.Ea = np.zeros((3, 1), dtype=np.complex128)
        solver.Ew = np.zeros((3, 1), dtype=np.complex128)
        solver.Ht = np.zeros((3, 1), dtype=np.complex128)
        solver.Ha = np.zeros((2, 1), dtype=np.complex128)
        solver.Hw = np.zeros((2, 1), dtype=np.complex128)
        if polarization == "TM":
            solver.Ea[:, 0] = sign * phase * np.asarray((1.0, 2.0, 1.0))
            solver.Ht[:, 0] = -sign * phase * np.asarray((0.5, 1.0, 0.5))
        else:
            solver.Et[:, 0] = sign * phase * np.asarray((1.0, 2.0))
            solver.Ha[:, 0] = sign * phase * np.asarray((0.5, 1.0))

        solver._align_modes_for_real_profile_power()
        aligned_fields.append(
            tuple(
                field.copy()
                for field in (solver.Et, solver.Ea, solver.Ew, solver.Ht, solver.Ha, solver.Hw)
            )
        )

    for positive, negative in zip(*aligned_fields):
        np.testing.assert_allclose(positive, negative, rtol=0, atol=1e-14)

    pivot_vector = np.concatenate((aligned_fields[0][0].ravel(), aligned_fields[0][1].ravel()))
    pivot = pivot_vector[np.argmax(np.abs(pivot_vector))]
    assert np.real(pivot) > 0


@pytest.mark.parametrize("polarization", ("TM", "TE"))
def test_longitudinal_field_satisfies_discrete_curl_equation(polarization):
    n = 40
    kwargs = {}
    if polarization == "TM":
        pec_a = np.zeros(n + 1, dtype=bool)
        pec_a[[0, -1]] = True
        kwargs["pec_a_mask"] = pec_a
    else:
        pmc_a = np.zeros(n, dtype=bool)
        pmc_a[[0, -1]] = True
        kwargs["pmc_a_mask"] = pmc_a
    solver = _solver(polarization, n=n, **kwargs)

    solver.solve()

    if polarization == "TM":
        expected = 1j * (solver.D_NODE_TO_CELL @ solver.modal_Ea) / (solver.eta0 * solver.mu_r_w)
        actual = solver.modal_Hw
    else:
        expected = -1j * solver.eta0 * (solver.D_CELL_TO_NODE @ solver.modal_Ha) / solver.eps_r_w
        actual = solver.modal_Ew

    assert np.max(np.abs(expected)) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_negative_real_power_fallback_rotates_all_fields_together():
    solver = object.__new__(FDFD_1D_mode_solver)
    solver.num_modes = 1
    solver.polarization = "TE"
    solver.dt = 1.0
    solver.Et = np.asarray(((1.0,), (2.0,)), dtype=np.complex128)
    solver.Ea = np.asarray(((0.1,), (0.2,), (0.1,)), dtype=np.complex128)
    solver.Ew = np.asarray(((0.2,), (0.3,), (0.2,)), dtype=np.complex128)
    solver.Ht = np.asarray(((0.1,), (0.2,), (0.1,)), dtype=np.complex128)
    solver.Ha = np.asarray(((0.5,), (1.0,)), dtype=np.complex128)
    solver.Hw = np.asarray(((0.2,), (0.4,)), dtype=np.complex128)
    fields = (solver.Et, solver.Ea, solver.Ew, solver.Ht, solver.Ha, solver.Hw)
    original_fields = tuple(field.copy() for field in fields)
    original_power = solver._calculate_mode_power(0)
    solver._real_profile_power_from_fields = lambda mode: -1.0

    solver._align_modes_for_real_profile_power()

    for actual, original in zip(fields, original_fields):
        np.testing.assert_allclose(actual, 1j * original, rtol=0, atol=1e-14)
    assert solver._calculate_mode_power(0) == pytest.approx(original_power)


def test_tm_dielectric_slab_mode_decays_across_air_margin():
    n = 70
    dt = 1e-3
    node_coordinate = np.arange(n + 1) * dt
    cell_coordinate = (np.arange(n) + 0.5) * dt
    eps_a = np.ones(n + 1)
    eps_t = np.ones(n)
    eps_w = np.ones(n + 1)
    eps_a[(node_coordinate >= 0.025) & (node_coordinate <= 0.045)] = 9
    eps_t[(cell_coordinate >= 0.025) & (cell_coordinate <= 0.045)] = 9
    eps_w[(node_coordinate >= 0.025) & (node_coordinate <= 0.045)] = 9

    solver = _solver(
        "TM",
        n=n,
        frequency=5e9,
        dt=dt,
        eps_r_t=eps_t,
        eps_r_a=eps_a,
        eps_r_w=eps_w,
    )
    solver.solve()

    field_magnitude = np.abs(solver.modal_Ea)
    edge_magnitude = max(field_magnitude[0], field_magnitude[-1])
    assert 1 < solver.modal_real_neff < 3
    assert edge_magnitude / np.max(field_magnitude) < 2e-3
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)


def test_te_pmc_parallel_plate_constrains_scalar_field():
    n = 40
    pmc_a = np.zeros(n, dtype=bool)
    pmc_a[[0, -1]] = True
    solver = _solver("TE", n=n, pmc_a_mask=pmc_a)
    solver.solve()

    assert 0 < solver.modal_real_neff < 1
    assert solver.modal_Ha[0] == 0
    assert solver.modal_Ha[-1] == 0
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_line_plot_contains_all_three_fields(tmp_path, polarization):
    n = 30
    kwargs = {}
    if polarization == "TM":
        mask = np.zeros(n + 1, dtype=bool)
        mask[[0, -1]] = True
        kwargs["pec_a_mask"] = mask
    else:
        mask = np.zeros(n + 1, dtype=bool)
        mask[[0, -1]] = True
        kwargs["pec_w_mask"] = mask
    solver = _solver(polarization, n=n, **kwargs)
    solver.solve()

    output = tmp_path / f"{polarization.lower()}_fields.png"
    assert solver.plot_fields(output) == output
    assert output.is_file()
    assert output.stat().st_size > 0
