"""Coupled finite-SIBC, bulk-polarization and native-PML equation checks."""

import numpy as np
import pytest

import gprMax.impedance_surfaces as implementation
from gprMax.updates.cpu_updates import CPUUpdates
from testing.validation.impedance_surface.validate_sibc_pml import build_grid, scene_for

pytestmark = pytest.mark.usefixtures("suppress_sibc_fit_plots")


@pytest.mark.parametrize("kind", ("lossy", "debye", "lorentz", "drude"))
@pytest.mark.parametrize("backend", ("python", "cython"))
def test_finite_surface_bulk_and_pml_histories_match_one_coupled_dense_solve(tmp_path, monkeypatch, kind, backend):
    grid = build_grid(
        scene_for(40, kind="foster", host_kind=kind, layered=True, pml_cells=8),
        tmp_path / "coupled",
    )
    system = grid.impedance_surfaces
    rng = np.random.default_rng(492)
    electric = (grid.Ex, grid.Ey, grid.Ez)
    magnetic = (grid.Hx, grid.Hy, grid.Hz)
    for field in (*electric, *magnetic):
        field[:] = rng.normal(size=field.shape)
    system.state_y[:] = rng.normal(size=system.state_y.shape)
    system.state_p[:] = rng.normal(size=system.state_p.shape) * 1e-12
    for slab in grid.pmls["slabs"]:
        for name in ("EPhi1", "EPhi2"):
            values = getattr(slab, name)
            values[:] = rng.normal(size=values.shape) * 100

    old_e = np.asarray([electric[c][i, j, k] for c, i, j, k, *_ in system.edge_info])
    old_y, old_p = system.state_y.copy(), system.state_p.copy()
    updates = CPUUpdates(grid)
    if grid.maxpoles:
        updates.set_dispersive_updates()
    updates.update_electric_a()
    updates.update_electric_pml()
    updates.update_electric_b()
    forcing = system._pending_pml_rhs.copy()
    assert np.max(np.abs(forcing)) > 1e-6
    expected = []
    for index, rhs_pml in zip(system.pml_edge_indices, forcing):
        c, i, j, k, hs, hc, ps, pc = system.edge_info[index]
        # The ordinary bulk A/B pass and PML capture must preserve old E.
        assert electric[c][i, j, k] == old_e[index]
        if grid.maxpoles:
            assert not np.any((grid.Tx, grid.Ty, grid.Tz)[c][:, i, j, k])
        plus, minus = system.edge_params[index].copy()
        start, stop = (system.pole_offsets[index:index + 2] if system.pole_coeffs.size else (0, 0))
        coefficients = system.pole_coeffs[start:stop]
        if coefficients.size:
            plus, minus = np.asarray((plus, minus)) + system.edge_dispersion[index]
        phi = np.sum(coefficients[:, 4] * old_p[start:stop, 0]
                     - coefficients[:, 5] * old_p[start:stop, 1]) if coefficients.size else 0
        circulation = sum(
            system.h_weight[h] * magnetic[system.h_info[h, 0]][tuple(system.h_info[h, 1:])]
            for h in range(hs, hs + hc)
        )
        matrix, rhs = np.zeros((1 + pc, 1 + pc)), np.zeros(1 + pc)
        matrix[0, 0] = plus
        rhs[0] = minus * old_e[index] + circulation - phi + rhs_pml
        for row, port in enumerate(range(ps, ps + pc), 1):
            model, offset = system.port_info[port]
            count = system.model_info[model, 0]
            matrix[0, row] = -system.port_g[port]
            matrix[row, 0], matrix[row, row] = -0.5, system.model_Z0[model]
            rhs[row] = 0.5 * old_e[index] - np.sum(old_y[offset:offset + count])
        expected.append((index, np.linalg.solve(matrix, rhs)))

    if backend == "python":
        monkeypatch.setattr(implementation, "_cython_update", None)
    else:
        assert implementation._cython_update is not None
    system.update(grid)
    for index, solved in expected:
        c, i, j, k, _, _, ps, pc = system.edge_info[index]
        np.testing.assert_allclose(electric[c][i, j, k], solved[0], rtol=2e-12, atol=1e-12)
        for row, port in enumerate(range(ps, ps + pc), 1):
            model, offset = system.port_info[port]
            count, cf = system.model_info[model]
            expected_y = (system.model_f[cf:cf + count] * old_y[offset:offset + count]
                          + system.model_q[cf:cf + count] * solved[row])
            np.testing.assert_allclose(system.state_y[offset:offset + count], expected_y,
                                       rtol=2e-12, atol=1e-12)
        if system.pole_coeffs.size:
            start, stop = system.pole_offsets[index:index + 2]
            coefficients = system.pole_coeffs[start:stop]
            f = coefficients[:, 0] + 1j * coefficients[:, 1]
            b = coefficients[:, 2] + 1j * coefficients[:, 3]
            expected_p = f * (old_p[start:stop, 0] + 1j * old_p[start:stop, 1]) + b * (old_e[index] - solved[0])
            np.testing.assert_allclose(system.state_p[start:stop, 0] + 1j * system.state_p[start:stop, 1],
                                       expected_p, rtol=2e-12, atol=1e-24)
    grid.reset_fields()
    assert not np.any(system.state_p)
    assert not np.any(system.state_y)
    assert system._pending_pml_rhs is None


@pytest.mark.integration
@pytest.mark.parametrize("kind", ("lossy", pytest.param("lorentz", marks=pytest.mark.slow)))
def test_copper_microstrip_substrate_pml_matches_longer_line(tmp_path, kind):
    from testing.validation.impedance_surface.validate_microstrip_pml import compare_microstrip

    result = compare_microstrip(tmp_path, kind)
    assert result["passed"], result
    assert result["relative_reflection_peak"] < 1e-3
    assert result["mixed_host_pml_edges"] > 0
