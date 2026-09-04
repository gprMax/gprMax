"""Check modal phase and impedance against discrete Yee plane-wave relations."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver


C = 299792458.0
Z0 = 376.73031366686166
FREQUENCY = 10e9
TRANSVERSE_SPACING = 3e-3
PROPAGATION_SPACING = 5e-3
FDTD_DT = 6e-12


@pytest.fixture(autouse=True)
def _solver_constants(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "e0": 8.8541878128e-12,
                "m0": 1.25663706212e-6,
                "c": C,
                "z0": Z0,
            }
        ),
    )


def _one_dimensional(polarization="TM", *, guide=True, epsilon=1, mu=1, **kwargs):
    n = 12
    endpoint_mask = np.zeros(n + 1, dtype=bool)
    endpoint_mask[[0, -1]] = True
    defaults = dict(
        frequency=FREQUENCY,
        dt=TRANSVERSE_SPACING,
        mode_index=0,
        polarization=polarization,
        eps_r_t=np.full(n, epsilon),
        eps_r_a=np.full(n + 1, epsilon),
        eps_r_w=np.full(n + 1, epsilon),
        mu_r_t=np.full(n + 1, mu),
        mu_r_a=np.full(n, mu),
        mu_r_w=np.full(n, mu),
    )
    if polarization == "TM" and guide:
        defaults["pec_a_mask"] = endpoint_mask
    elif polarization == "TE":
        # PEC longitudinal E gives Neumann boundaries for the scalar H field.
        defaults["pec_w_mask"] = endpoint_mask
        defaults["mode_index"] = 1 if guide else 0
    defaults.update(kwargs)
    return FDFD_1D_mode_solver(**defaults)


def _rectangular_guide(**kwargs):
    nu, nv = 12, 8
    eu = np.ones((nu, nv + 1))
    ev = np.ones((nu + 1, nv))
    ew = np.ones((nu + 1, nv + 1))
    pec_u, pec_v, pec_w = [np.zeros_like(a, dtype=bool) for a in (eu, ev, ew)]
    pec_u[:, [0, -1]] = True
    pec_v[[0, -1], :] = True
    pec_w[:, [0, -1]] = True
    pec_w[[0, -1], :] = True
    defaults = dict(
        frequency=FREQUENCY,
        du=TRANSVERSE_SPACING,
        dv=TRANSVERSE_SPACING,
        mode_index=0,
        eps_r_uu=eu,
        eps_r_vv=ev,
        eps_r_ww=ew,
        mu_r_uu=np.ones_like(ev),
        mu_r_vv=np.ones_like(eu),
        mu_r_ww=np.ones((nu, nv)),
        pec_u_mask=pec_u,
        pec_v_mask=pec_v,
        pec_w_mask=pec_w,
    )
    defaults.update(kwargs)
    return FDFD_2D_mode_solver(**defaults)


def _assert_discrete_dispersion(solver, transverse_order=1, epsilon_mu=1):
    """The Yee sine dispersion law includes time and every varying space axis."""
    beta = solver.beta[solver.mode_index]
    lhs = (np.sin(np.pi * solver.frequency * FDTD_DT) / (C * FDTD_DT)) ** 2
    transverse = np.sin(transverse_order * np.pi / (2 * 12)) / TRANSVERSE_SPACING
    longitudinal = np.sin(beta * PROPAGATION_SPACING / 2) / PROPAGATION_SPACING
    assert transverse**2 + longitudinal**2 == pytest.approx(epsilon_mu * lhs, rel=2e-10)
    assert solver.modal_complex_neff == pytest.approx(beta / (2 * np.pi * solver.frequency / C))


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_coarse_1d_guide_obeys_yee_dispersion_and_modal_impedance(polarization):
    solver = _one_dimensional(
        polarization, fdtd_dt=FDTD_DT, propagation_spacing=PROPAGATION_SPACING
    )
    solver.solve()

    _assert_discrete_dispersion(solver)
    omega_d = 2 * np.sin(np.pi * FREQUENCY * FDTD_DT) / FDTD_DT
    transverse_symbol = 2 * np.sin(np.pi / 24) / TRANSVERSE_SPACING
    propagation_symbol = np.sqrt((omega_d / C) ** 2 - transverse_symbol**2)
    if polarization == "TM":
        electric, magnetic = solver.modal_Ea, solver.modal_Ht
        impedance = -Z0 * omega_d / (C * propagation_symbol)
    else:
        electric, magnetic = solver.modal_Et, solver.modal_Ha
        impedance = Z0 * C * propagation_symbol / omega_d
    active = np.abs(magnetic) > 1e-8 * np.max(np.abs(magnetic))
    np.testing.assert_allclose(electric[active] / magnetic[active], impedance, rtol=2e-10)
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)
    # This mesh is coarse enough that confusing phase and operator indices matters.
    assert abs(solver.modal_complex_neff - solver.operator_neff[solver.mode_index]) > 0.01


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_homogeneous_tem_keeps_material_impedance_with_corrected_phase(polarization):
    epsilon, mu = 2.25, 1.0
    solver = _one_dimensional(
        polarization,
        guide=False,
        epsilon=epsilon,
        mu=mu,
        fdtd_dt=FDTD_DT,
        propagation_spacing=PROPAGATION_SPACING,
    )
    solver.solve()

    _assert_discrete_dispersion(solver, transverse_order=0, epsilon_mu=epsilon * mu)
    electric = solver.modal_Ea if polarization == "TM" else solver.modal_Et
    magnetic = -solver.modal_Ht if polarization == "TM" else solver.modal_Ha
    np.testing.assert_allclose(electric / magnetic, Z0 * np.sqrt(mu / epsilon), rtol=2e-10)
    assert solver.operator_neff[0] == pytest.approx(np.sqrt(epsilon * mu), rel=2e-10)


def test_rectangular_te10_obeys_3d_yee_dispersion_and_impedance():
    solver = _rectangular_guide(fdtd_dt=FDTD_DT, propagation_spacing=PROPAGATION_SPACING)
    solver.solve()

    _assert_discrete_dispersion(solver)
    omega_d = 2 * np.sin(np.pi * FREQUENCY * FDTD_DT) / FDTD_DT
    transverse_symbol = 2 * np.sin(np.pi / 24) / TRANSVERSE_SPACING
    propagation_symbol = np.sqrt((omega_d / C) ** 2 - transverse_symbol**2)
    active = np.abs(solver.modal_Ev) > 1e-8 * np.max(np.abs(solver.modal_Ev))
    np.testing.assert_allclose(
        solver.modal_Hu[active] / solver.modal_Ev[active],
        -C * propagation_symbol / (Z0 * omega_d),
        rtol=2e-10,
    )
    np.testing.assert_allclose(-(solver.operator_neff**2), solver.eigenvalues, rtol=1e-11)
    assert solver.modal_power_valid


def test_below_cutoff_phase_remains_passively_evanescent():
    solver = _one_dimensional(
        frequency=2e9, fdtd_dt=FDTD_DT, propagation_spacing=PROPAGATION_SPACING
    )
    solver.solve()

    _assert_discrete_dispersion(solver)
    assert solver.beta[0].real == pytest.approx(0, abs=1e-10)
    assert solver.beta[0].imag < 0
    assert abs(np.exp(-1j * solver.beta[0] * PROPAGATION_SPACING)) < 1
    assert not solver.modal_power_valid


@pytest.mark.parametrize("epsilon,mu", [(4 - 0.4j, 1), (-2 - 0.05j, -2 - 0.05j)])
def test_lossy_and_backward_phase_branches_preserve_decay_and_forward_power(epsilon, mu):
    solver = _one_dimensional(
        guide=False,
        epsilon=epsilon,
        mu=mu,
        fdtd_dt=FDTD_DT,
        propagation_spacing=PROPAGATION_SPACING,
    )
    solver.solve()

    _assert_discrete_dispersion(solver, transverse_order=0, epsilon_mu=epsilon * mu)
    assert np.sign(solver.beta[0].real) == np.sign(np.real(epsilon))
    assert solver.beta[0].imag < 0
    assert abs(np.exp(-1j * solver.beta[0] * PROPAGATION_SPACING)) < 1
    assert solver.modal_power_valid
    assert solver.modal_power == pytest.approx(1.0, rel=1e-12)
    np.testing.assert_allclose(-(solver.operator_neff**2), solver.eigenvalues, rtol=1e-11)


def test_spatial_stop_band_uses_decaying_branch():
    solver = _one_dimensional(
        guide=False, epsilon=25, fdtd_dt=FDTD_DT, propagation_spacing=PROPAGATION_SPACING
    )
    solver.solve()

    _assert_discrete_dispersion(solver, transverse_order=0, epsilon_mu=25)
    assert solver.beta[0].real == pytest.approx(np.pi / PROPAGATION_SPACING, rel=1e-10)
    assert solver.beta[0].imag < 0
    assert not solver.modal_power_valid
    assert np.all(np.isfinite(solver.modal_Ea))
    assert np.all(np.isfinite(solver.modal_Ht))


@pytest.mark.parametrize("factory", [_one_dimensional, _rectangular_guide], ids=["1d", "2d"])
@pytest.mark.parametrize("fdtd_dt", [None, FDTD_DT], ids=["continuous-time", "yee-time"])
def test_physical_and_operator_frequency_conventions(factory, fdtd_dt):
    solver = factory(fdtd_dt=fdtd_dt, propagation_spacing=PROPAGATION_SPACING)
    omega = 2 * np.pi * FREQUENCY
    operator_omega = omega if fdtd_dt is None else 2 * np.sin(omega * fdtd_dt / 2) / fdtd_dt

    assert solver.omega == pytest.approx(omega, rel=1e-14)
    assert solver.k0 == pytest.approx(omega / C, rel=1e-14)
    assert solver.operator_omega == pytest.approx(operator_omega, rel=1e-14)
    assert solver.operator_k0 == pytest.approx(operator_omega / C, rel=1e-14)
    if fdtd_dt is not None:
        assert solver.operator_omega < solver.omega
        assert solver.operator_k0 < solver.k0

    solver.solve()
    np.testing.assert_allclose(solver.beta, solver.k0 * solver.complex_neff, rtol=1e-13)
    np.testing.assert_allclose(
        2 * np.sin(solver.beta * PROPAGATION_SPACING / 2) / PROPAGATION_SPACING,
        operator_omega / C * solver.operator_neff,
        rtol=1e-13,
    )


@pytest.mark.parametrize("factory", [_one_dimensional, _rectangular_guide], ids=["1d", "2d"])
def test_legacy_defaults_and_fine_step_limit(factory):
    legacy = factory()
    fine = factory(fdtd_dt=2e-17, propagation_spacing=1e-8)
    legacy.solve()
    fine.solve()

    assert legacy.omega == pytest.approx(2 * np.pi * FREQUENCY)
    assert legacy.k0 == pytest.approx(2 * np.pi * FREQUENCY / C)
    assert legacy.operator_omega == legacy.omega
    assert legacy.operator_k0 == legacy.k0
    np.testing.assert_allclose(legacy.complex_neff, legacy.operator_neff, rtol=1e-13)
    np.testing.assert_allclose(legacy.beta, legacy.k0 * legacy.complex_neff, rtol=1e-13)
    np.testing.assert_allclose(fine.complex_neff, legacy.complex_neff, rtol=1e-10)


@pytest.mark.parametrize("factory", [_one_dimensional, _rectangular_guide], ids=["1d", "2d"])
@pytest.mark.parametrize("parameter", ["fdtd_dt", "propagation_spacing"])
@pytest.mark.parametrize("value", [0.0, -1e-12, np.nan, np.inf, -np.inf])
def test_dispersion_parameters_must_be_finite_and_positive(factory, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        factory(**{parameter: value})


@pytest.mark.parametrize("factory", [_one_dimensional, _rectangular_guide], ids=["1d", "2d"])
@pytest.mark.parametrize("fdtd_dt", [0.5 / FREQUENCY, 1.0 / FREQUENCY])
def test_temporal_nyquist_and_aliased_frequencies_are_rejected(factory, fdtd_dt):
    with pytest.raises(ValueError, match="[Nn]yquist"):
        factory(fdtd_dt=fdtd_dt)
