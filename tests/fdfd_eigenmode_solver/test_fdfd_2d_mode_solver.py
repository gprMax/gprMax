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
from scipy.sparse import diags

import gprMax.config as config
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


def test_lossy_pec_waveguide_mode_uses_same_passive_branch_for_neff_and_h():
    nu, nv = 8, 6
    spacing = 5e-3
    frequency = 5e9
    epsilon = 9 - 1j * 2 / (2 * np.pi * frequency * 8.8541878128e-12)

    eps_uu = np.full((nu, nv + 1), epsilon)
    eps_vv = np.full((nu + 1, nv), epsilon)
    eps_ww = np.full((nu + 1, nv + 1), epsilon)
    mu_uu = np.ones((nu + 1, nv))
    mu_vv = np.ones((nu, nv + 1))
    mu_ww = np.ones((nu, nv))

    pec_u = np.zeros_like(eps_uu, dtype=bool)
    pec_v = np.zeros_like(eps_vv, dtype=bool)
    pec_w = np.zeros_like(eps_ww, dtype=bool)
    pec_u[:, [0, -1]] = True
    pec_v[[0, -1], :] = True
    pec_w[[0, -1], :] = True
    pec_w[:, [0, -1]] = True

    solver = FDFD_2D_mode_solver(
        frequency=frequency,
        du=spacing,
        dv=spacing,
        mode_index=0,
        eps_r_uu=eps_uu,
        eps_r_vv=eps_vv,
        eps_r_ww=eps_ww,
        mu_r_uu=mu_uu,
        mu_r_vv=mu_vv,
        mu_r_ww=mu_ww,
        pec_u_mask=pec_u,
        pec_v_mask=pec_v,
        pec_w_mask=pec_w,
    )
    solver.solve()

    width = nu * spacing
    expected = np.sqrt(epsilon - (np.pi / (solver.k0 * width)) ** 2)
    forward_factor = np.exp(-1j * solver.k0 * solver.modal_complex_neff * 0.5e-3)

    assert solver.modal_complex_neff == pytest.approx(expected, rel=1e-3)
    assert np.real(solver.modal_complex_neff) > 0
    assert np.imag(solver.modal_complex_neff) < 0
    assert abs(forward_factor) < 1
    assert solver.modal_power_valid
    assert solver.modal_forward_power_metric > solver.FORWARD_POWER_METRIC_TOLERANCE
    assert np.real(solver.modal_raw_power) > 0
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(
        np.square(1j * solver.complex_neff),
        solver.eigenvalues,
        rtol=1e-12,
        atol=1e-12,
    )


def test_backward_wave_uses_passive_forward_power_branch():
    nu, nv = 8, 6
    spacing = 5e-3
    frequency = 5e9
    material = -2 - 0.05j
    eps_uu = np.full((nu, nv + 1), material)
    eps_vv = np.full((nu + 1, nv), material)
    eps_ww = np.full((nu + 1, nv + 1), material)
    mu_uu = np.full((nu + 1, nv), material)
    mu_vv = np.full((nu, nv + 1), material)
    mu_ww = np.full((nu, nv), material)
    pec_u = np.zeros_like(eps_uu, dtype=bool)
    pec_v = np.zeros_like(eps_vv, dtype=bool)
    pec_w = np.zeros_like(eps_ww, dtype=bool)
    pec_u[:, [0, -1]] = True
    pec_v[[0, -1], :] = True
    pec_w[[0, -1], :] = True
    pec_w[:, [0, -1]] = True
    solver = FDFD_2D_mode_solver(
        frequency=frequency,
        du=spacing,
        dv=spacing,
        mode_index=0,
        eps_r_uu=eps_uu,
        eps_r_vv=eps_vv,
        eps_r_ww=eps_ww,
        mu_r_uu=mu_uu,
        mu_r_vv=mu_vv,
        mu_r_ww=mu_ww,
        pec_u_mask=pec_u,
        pec_v_mask=pec_v,
        pec_w_mask=pec_w,
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
    np.testing.assert_allclose(
        np.square(1j * solver.complex_neff),
        solver.eigenvalues,
        rtol=1e-12,
        atol=1e-12,
    )


def test_below_cutoff_mode_is_kept_with_finite_balanced_normalization():
    nu, nv = 12, 8
    spacing = 1e-3
    frequency = 10e9
    eps_uu = np.ones((nu, nv + 1))
    eps_vv = np.ones((nu + 1, nv))
    eps_ww = np.ones((nu + 1, nv + 1))
    mu_uu = np.ones((nu + 1, nv))
    mu_vv = np.ones((nu, nv + 1))
    mu_ww = np.ones((nu, nv))
    pec_u = np.zeros_like(eps_uu, dtype=bool)
    pec_v = np.zeros_like(eps_vv, dtype=bool)
    pec_w = np.zeros_like(eps_ww, dtype=bool)
    pec_u[:, [0, -1]] = True
    pec_v[[0, -1], :] = True
    pec_w[[0, -1], :] = True
    pec_w[:, [0, -1]] = True
    cutoff = config.sim_config.em_consts["c"] / (2 * nu * spacing)
    expected_neff = -1j * np.sqrt((cutoff / frequency) ** 2 - 1)
    solver = FDFD_2D_mode_solver(
        frequency=frequency,
        du=spacing,
        dv=spacing,
        mode_index=0,
        eps_r_uu=eps_uu,
        eps_r_vv=eps_vv,
        eps_r_ww=eps_ww,
        mu_r_uu=mu_uu,
        mu_r_vv=mu_vv,
        mu_r_ww=mu_ww,
        pec_u_mask=pec_u,
        pec_v_mask=pec_v,
        pec_w_mask=pec_w,
        guess=-(expected_neff**2),
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
        solver.modal_Eu,
        solver.modal_Ev,
        solver.modal_Ew,
        solver.modal_Hu,
        solver.modal_Hv,
        solver.modal_Hw,
    ):
        assert np.all(np.isfinite(field))


def test_rectangular_te10_enforces_pec_normal_h_and_wave_impedance():
    nu, nv = 60, 40
    spacing = 0.1e-3
    frequency = 55e9

    eps_uu = np.ones((nu, nv + 1))
    eps_vv = np.ones((nu + 1, nv))
    eps_ww = np.ones((nu + 1, nv + 1))
    mu_uu = np.ones((nu + 1, nv))
    mu_vv = np.ones((nu, nv + 1))
    mu_ww = np.ones((nu, nv))

    pec_u = np.zeros_like(eps_uu, dtype=bool)
    pec_v = np.zeros_like(eps_vv, dtype=bool)
    pec_w = np.zeros_like(eps_ww, dtype=bool)
    pec_u[:, [0, -1]] = True
    pec_v[[0, -1], :] = True
    pec_w[[0, -1], :] = True
    pec_w[:, [0, -1]] = True

    width = nu * spacing
    expected_neff = np.sqrt(1 - (config.sim_config.em_consts["c"] / (2 * width * frequency)) ** 2)
    solver = FDFD_2D_mode_solver(
        frequency=frequency,
        du=spacing,
        dv=spacing,
        mode_index=0,
        eps_r_uu=eps_uu,
        eps_r_vv=eps_vv,
        eps_r_ww=eps_ww,
        mu_r_uu=mu_uu,
        mu_r_vv=mu_vv,
        mu_r_ww=mu_ww,
        pec_u_mask=pec_u,
        pec_v_mask=pec_v,
        pec_w_mask=pec_w,
        guess=-(expected_neff**2),
    )
    solver.solve()

    np.testing.assert_array_equal(solver.hu_constraint_mask, pec_v)
    np.testing.assert_array_equal(solver.hv_constraint_mask, pec_u)
    np.testing.assert_array_equal(solver.modal_Hu[pec_v], 0.0)
    np.testing.assert_array_equal(solver.modal_Hv[pec_u], 0.0)

    cell_hu = 0.5 * (solver.modal_Hu[:-1, :] + solver.modal_Hu[1:, :])
    cell_ev = 0.5 * (solver.modal_Ev[:-1, :] + solver.modal_Ev[1:, :])
    fitted_impedance = abs(np.vdot(cell_hu, cell_ev) / np.vdot(cell_hu, cell_hu))
    expected_impedance = solver.eta0 / solver.modal_real_neff
    assert fitted_impedance == pytest.approx(expected_impedance, rel=1e-3)


def test_rectangular_pmc_tm10_enforces_normal_e_and_matches_theory():
    nu, nv = 61, 41
    spacing = 0.1e-3
    frequency = 55e9

    eps_uu = np.ones((nu, nv + 1))
    eps_vv = np.ones((nu + 1, nv))
    eps_ww = np.ones((nu + 1, nv + 1))
    mu_uu = np.ones((nu + 1, nv))
    mu_vv = np.ones((nu, nv + 1))
    mu_ww = np.ones((nu, nv))

    pmc_u = np.zeros_like(mu_uu, dtype=bool)
    pmc_v = np.zeros_like(mu_vv, dtype=bool)
    pmc_w = np.zeros_like(mu_ww, dtype=bool)
    pmc_u[:, [0, -1]] = True
    pmc_v[[0, -1], :] = True
    pmc_w[[0, -1], :] = True
    pmc_w[:, [0, -1]] = True

    # Transverse PMC samples lie at cell centres, so their plane separation
    # is one cell smaller than the corresponding array extent.
    width = (nu - 1) * spacing
    expected_neff = np.sqrt(1 - (config.sim_config.em_consts["c"] / (2 * width * frequency)) ** 2)
    solver = FDFD_2D_mode_solver(
        frequency=frequency,
        du=spacing,
        dv=spacing,
        mode_index=0,
        eps_r_uu=eps_uu,
        eps_r_vv=eps_vv,
        eps_r_ww=eps_ww,
        mu_r_uu=mu_uu,
        mu_r_vv=mu_vv,
        mu_r_ww=mu_ww,
        pmc_u_mask=pmc_u,
        pmc_v_mask=pmc_v,
        pmc_w_mask=pmc_w,
        guess=-(expected_neff**2),
    )
    solver.solve()

    np.testing.assert_array_equal(solver.eu_constraint_mask, pmc_v)
    np.testing.assert_array_equal(solver.ev_constraint_mask, pmc_u)
    np.testing.assert_array_equal(solver.modal_Eu[pmc_v], 0.0)
    np.testing.assert_array_equal(solver.modal_Ev[pmc_u], 0.0)
    assert solver.modal_real_neff == pytest.approx(expected_neff, rel=1e-4)

    cell_eu = 0.5 * (solver.modal_Eu[:, :-1] + solver.modal_Eu[:, 1:])
    cell_hv = 0.5 * (solver.modal_Hv[:, :-1] + solver.modal_Hv[:, 1:])
    fitted_impedance = abs(np.vdot(cell_eu, cell_eu) / np.vdot(cell_eu, cell_hv))
    expected_impedance = solver.eta0 * solver.modal_real_neff
    assert fitted_impedance == pytest.approx(expected_impedance, rel=1e-3)


def test_default_guess_targets_magnetic_medium_fundamental():
    nu, nv = 30, 25
    epsilon = 4.0
    permeability = 4.0
    solver = FDFD_2D_mode_solver(
        frequency=10e9,
        du=0.5e-3,
        dv=0.5e-3,
        mode_index=0,
        eps_r_uu=np.full((nu, nv + 1), epsilon),
        eps_r_vv=np.full((nu + 1, nv), epsilon),
        eps_r_ww=np.full((nu + 1, nv + 1), epsilon),
        mu_r_uu=np.full((nu + 1, nv), permeability),
        mu_r_vv=np.full((nu, nv + 1), permeability),
        mu_r_ww=np.full((nu, nv), permeability),
    )

    assert solver.guess == pytest.approx(-(epsilon * permeability))
    solver.solve()

    assert solver.modal_real_neff == pytest.approx(
        np.sqrt(epsilon * permeability),
        rel=4e-2,
    )


def test_real_profile_alignment_canonicalizes_global_mode_sign():
    phase = np.exp(0.37j)
    aligned_fields = []
    for sign in (1.0, -1.0):
        solver = object.__new__(FDFD_2D_mode_solver)
        solver.num_modes = 1
        solver.Nu = 2
        solver.Nv = 2
        solver.du = 1.0
        solver.dv = 1.0
        solver.Eu = sign * phase * np.ones((2, 3, 1), dtype=np.complex128)
        solver.Ev = np.zeros((3, 2, 1), dtype=np.complex128)
        solver.Ew = np.zeros((3, 3, 1), dtype=np.complex128)
        solver.Hu = np.zeros((3, 2, 1), dtype=np.complex128)
        solver.Hv = sign * phase * np.ones((2, 3, 1), dtype=np.complex128)
        solver.Hw = np.zeros((2, 2, 1), dtype=np.complex128)

        solver._align_modes_for_real_profile_power()
        aligned_fields.append(
            tuple(
                field.copy()
                for field in (solver.Eu, solver.Ev, solver.Ew, solver.Hu, solver.Hv, solver.Hw)
            )
        )

    for positive, negative in zip(*aligned_fields):
        np.testing.assert_allclose(positive, negative, rtol=0, atol=1e-14)

    pivot_vector = np.concatenate((aligned_fields[0][0].ravel(), aligned_fields[0][1].ravel()))
    pivot = pivot_vector[np.argmax(np.abs(pivot_vector))]
    assert np.real(pivot) > 0


def test_small_reduced_system_uses_dense_eigensolve():
    solver = object.__new__(FDFD_2D_mode_solver)
    solver.num_modes = 2
    solver.guess = -4.0
    matrix = diags((-9.0, -4.0, -1.0), format="csr")

    eigenvalues, eigenvectors = solver._solve_reduced(matrix)

    np.testing.assert_allclose(eigenvalues, (-4.0, -1.0), rtol=0, atol=1e-12)
    assert eigenvectors.shape == (3, 2)


def test_sparse_eigensolve_retries_perturbed_singular_shift():
    solver = object.__new__(FDFD_2D_mode_solver)
    solver.num_modes = 1
    solver.guess = -4.0
    matrix = diags((-4.0, -2.0, -1.0), format="csr")

    eigenvalues, eigenvectors = solver._solve_reduced(matrix)

    np.testing.assert_allclose(eigenvalues, (-4.0,), rtol=0, atol=1e-12)
    assert eigenvectors.shape == (3, 1)
