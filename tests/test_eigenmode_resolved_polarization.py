"""Explicit single-direction orientation follows automatic subgroup selection."""
from copy import deepcopy

import h5py
import numpy as np
import pytest

from gprMax.eigenmode_tracking import _moments, write_diagnostics, aligned_plot_solvers
from testing.eigenmode_support.subspaces.expanded_mode_tracking import fd_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
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
def test_public_port_pairs_warn_and_auto_mappings_are_rejected(monkeypatch, tracking, value):
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
    if tracking == 'auto' and isinstance(value, dict):
        with pytest.raises(ValueError, match="require tracking='legacy'"):
            port.build(grid)
    else:
        with pytest.warns(FutureWarning, match='removed in gprMax 5.0'):
            port.build(grid)
        result = grid.eigenmodeportdefs[1].mode_polarizations
        primary = np.array((1., 1., 0.)) / np.sqrt(2) if isinstance(value, tuple) and not isinstance(value[0], str) else (1., 0., 0.)
        partner = (-1 / np.sqrt(2), 1 / np.sqrt(2), 0.) if isinstance(value, tuple) and not isinstance(value[0], str) else (0., 1., 0.)
        if tracking == 'auto':
            np.testing.assert_allclose(result, [primary, partner])
        else:
            np.testing.assert_allclose(result[1], primary)
            np.testing.assert_allclose(result[2], partner)


@pytest.mark.parametrize('sign', ('+', '-'))
@pytest.mark.parametrize('randomize', (False, True))
@pytest.mark.parametrize('requested', [('y', 'x'), ('y', (-1, 0, 0)),
                                      ((0, -1, 0), (-1, 0, 0)), ('x', (1, 1, 0))])
def test_auto_explicit_pair_preserves_directions_signs_and_power_gram(circular_solvers, sign, randomize, requested):
    from gprMax.eigenmode_tracking import normalize_polarizations, _power
    changed = deepcopy(circular_solvers)
    if randomize:
        rng = np.random.default_rng(91)
        for solver in changed:
            rotation = np.linalg.qr(rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)))[0]
            for name in ('Eu', 'Ev', 'Ew', 'Hu', 'Hv', 'Hw'):
                setattr(solver, name, getattr(solver, name) @ rotation)
    with pytest.warns(FutureWarning, match='removed in gprMax 5.0') as warning:
        with _configured_solver_constants():
            owner, grid, e, h = bank(changed, direction=sign, references=requested, tracking='auto')
    assert len(warning) == 1
    wanted = normalize_polarizations(requested, (), 2, unresolved=True, warn_deprecated=False)
    for k in range(len(e)):
        moments = _moments(owner, grid, e[k])
        np.testing.assert_allclose(moments / np.linalg.norm(moments, axis=0),
                                   np.asarray(wanted).T[:2], atol=1e-10)
        power = _power(owner, grid, e[k], h[k])
        np.testing.assert_allclose(np.diag(power), np.ones(2), atol=1e-10)
        assert np.min(np.linalg.eigvalsh(power)) > 0
        expected_overlap = np.dot(*wanted)
        assert abs(power[0, 1]) == pytest.approx(abs(expected_overlap), abs=1e-10)
    np.testing.assert_allclose(owner.polarization_diagnostics[0]['directions'], wanted)
