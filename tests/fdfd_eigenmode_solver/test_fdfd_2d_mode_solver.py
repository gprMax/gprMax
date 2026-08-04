from types import SimpleNamespace

import numpy as np
import pytest

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
    forward_factor = np.exp(
        -1j * solver.k0 * solver.modal_complex_neff * 0.5e-3
    )

    assert solver.modal_complex_neff == pytest.approx(expected, rel=1e-3)
    assert np.real(solver.modal_complex_neff) > 0
    assert np.imag(solver.modal_complex_neff) < 0
    assert abs(forward_factor) < 1
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(
        np.square(1j * solver.complex_neff),
        solver.eigenvalues,
        rtol=1e-12,
        atol=1e-12,
    )
