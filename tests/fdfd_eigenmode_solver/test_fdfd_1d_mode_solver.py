from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver


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
