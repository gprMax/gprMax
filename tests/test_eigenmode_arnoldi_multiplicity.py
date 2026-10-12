"""ARPACK must retain complete repeated Maxwell eigenspaces."""
import numpy as np
import pytest

from testing.validation.eigenmode_tracking.subspaces.expanded_mode_tracking import fd_solver
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants


@pytest.mark.parametrize('seed', [0, 10, 12, 18, 19])
def test_square_quartet_survives_arnoldi_restarts(monkeypatch, seed):
    import gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver as solver_module

    original = solver_module.eigs

    def solve_with_start(operator, **kwargs):
        kwargs['v0'] = np.random.default_rng(seed).standard_normal(operator.shape[0])
        return original(operator, **kwargs)

    monkeypatch.setattr(solver_module, 'eigs', solve_with_start)
    with _configured_solver_constants():
        solver = fd_solver(21e9, (12, 12), .02 / 12, (2, 2, 2), count=12)
        k0 = 2 * np.pi * solver.frequency / solver_module.config.sim_config.em_consts['c']
        cutoff_squared = sum((2 * np.sin(m * np.pi / 24) / solver.du)**2 for m in (1, 2))
        expected = -2 + cutoff_squared / k0**2
        np.testing.assert_allclose(solver.eigenvalues[6:10], expected, atol=1e-11, rtol=0)
        assert np.linalg.matrix_rank(solver.eigenvectors[:, 6:10]) == 4
        assert max(solver.tracking_field_residuals(range(6, 10))) < 1e-10
