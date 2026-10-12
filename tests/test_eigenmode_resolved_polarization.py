"""Explicit single-direction orientation follows automatic subgroup selection."""
from copy import deepcopy

import h5py
import numpy as np
import pytest

from gprMax.eigenmode_tracking import _moments, write_diagnostics, aligned_plot_solvers
from testing.validation.expanded_mode_tracking import fd_solver
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
from tests.test_eigenmode_degenerate import bank, circular_solvers
from types import SimpleNamespace


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('sign', ('+', '-'))
@pytest.mark.parametrize('diagonal', (False, True))
def test_auto_polarization_after_symmetry_is_gauge_invariant(circular_solvers, axis, sign, diagonal):
    transverse = [a for a in range(3) if a != axis]
    primary = np.eye(3)[transverse[0]].copy()
    if diagonal:
        primary += np.eye(3)[transverse[1]]
    primary /= np.linalg.norm(primary)
    partner = np.cross(np.eye(3)[axis], primary)
    changed = deepcopy(circular_solvers)
    rng = np.random.default_rng(73)
    for solver in changed:
        rotation = np.linalg.qr(rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)))[0]
        for name in ('Eu', 'Ev', 'Ew', 'Hu', 'Hv', 'Hw'):
            setattr(solver, name, getattr(solver, name) @ rotation)
    with _configured_solver_constants():
        owner, grid, e, h = bank(changed, axis, sign, primary, tracking='auto')
        _, _, expected_e, expected_h = bank(circular_solvers, axis, sign, primary, tracking='auto')
    assert owner.symmetry_diagnostics
    assert owner.polarization_diagnostics[0]['modes'] == (1, 2)
    for k in range(len(e)):
        moments = _moments(owner, grid, e[k])
        np.testing.assert_allclose(moments / np.linalg.norm(moments, axis=0),
                                   np.asarray([primary, partner]).T[transverse], atol=1e-10)
        for actual, expected in ((e[k], expected_e[k]), (h[k], expected_h[k])):
            for amode, emode in zip(actual, expected):
                for a, b in zip(amode, emode):
                    np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('labels,reason', [((3, 4), 'no eligible doublet'),
                                         ((7, 8, 9, 10), 'integrated transverse E')])
def test_subdivision_precedes_polarization_and_rejects_ineligible_patterns(monkeypatch, labels, reason):
    import gprMax.eigenmode_symmetry as symmetry
    original = symmetry.align_symmetry_subspaces
    observed = []

    def inspect(*args, **kwargs):
        original(*args, **kwargs)
        observed.extend(args[0].symmetry_diagnostics)

    monkeypatch.setattr(symmetry, 'align_symmetry_subspaces', inspect)
    with _configured_solver_constants():
        solvers = [fd_solver(f*1e9, (16, 16), .02/16, (2, 2, 2), count=12)
                   for f in (39, 40, 41)]
        with pytest.raises(ValueError, match=reason):
            bank(solvers, references='x', groups=(labels,), tracking='auto')
    assert observed
    assert [len(b) for b in observed[0]['blocks']] == ([1, 1] if len(labels) == 2 else [2, 2])


def test_export_distinguishes_native_and_final_polarized_bases(circular_solvers, tmp_path):
    with _configured_solver_constants():
        owner, grid, e, h = bank(circular_solvers, references=(1, 1, 0), tracking='auto')
    plotters = aligned_plot_solvers(owner, circular_solvers)
    np.testing.assert_allclose(plotters[0].mode_polarizations[1], [1/np.sqrt(2), 1/np.sqrt(2), 0])
    for k, solver in enumerate(circular_solvers):
        raw = [owner._fields_from_solver_mode(solver, m) for m in (1, 2)]
        transform = owner.degenerate_diagnostics[0]['anchors'][k]['transform']
        for family, final in ((0, e), (1, h)):
            for component in range(3):
                actual = np.stack([mode[family][component] for mode in raw], axis=-1) @ transform
                np.testing.assert_allclose(actual, np.stack([mode[component] for mode in final[k]], axis=-1), atol=1e-10)
    with h5py.File(tmp_path/'diagnostics.h5', 'w') as output:
        write_diagnostics(output, owner)
        assert 'polarization_subspaces/1/anchor0/transform' in output
        assert output['degenerate_groups/1'].attrs['PhysicalPolarization']
        assert 'degenerate_groups/1/requested_directions' in output
        assert 'before any explicit' in output['symmetry_subspaces'].attrs['BasisConvention']
        np.testing.assert_allclose(output['polarization_subspaces/1/requested_directions'][:],
                                   [[1/np.sqrt(2), 1/np.sqrt(2), 0], [-1/np.sqrt(2), 1/np.sqrt(2), 0]])


@pytest.mark.parametrize('tracking', ('auto', 'legacy'))
@pytest.mark.parametrize('value', [('x', 'y'), ((1, 1, 0), (-1, 1, 0)),
                                  {1: 'x', 2: 'y'}, {(1, 2): 'x'}])
def test_public_port_rejects_removed_forms(monkeypatch, tracking, value):
    import gprMax.config as config
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.user_objects.cmds_multiuse import EigenmodePort
    monkeypatch.setattr(config, 'sim_config', SimpleNamespace(general={'solver': 'cpu'}, mpi=False))
    monkeypatch.setattr(config, 'get_model_config', lambda: SimpleNamespace(mode='3D'))
    grid = FDTDGrid()
    grid.eigenmodeband = SimpleNamespace()
    port = EigenmodePort(port=1, p1=(0, 0, .02), p2=(.05, .05, .02), direction='+',
                         modes=(1, 2), anchors='auto', degenerate=(1, 2),
                         tracking=tracking, mode_polarizations=value)
    with pytest.raises(ValueError, match='one transverse direction only'):
        port.build(grid)
