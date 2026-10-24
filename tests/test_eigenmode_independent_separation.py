"""Independent separation must survive absent/rejected spatial symmetry."""
from copy import deepcopy

import h5py
import numpy as np
import pytest

import gprMax.eigenmode_symmetry as symmetry
from gprMax.eigenmode_tracking import _frame, _polarize_resolved_pairs, write_diagnostics
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from tests.test_eigenmode_auto_symmetry import aligned_bank
from testing.eigenmode_support.subspaces.general_subspace_tracking import fd_solver, stress_rotation
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
from testing.eigenmode_support.identification.tracking_classification_edges import rectangular_templates


def shifted_rectangle(frequency):
    """A PEC rectangle away from both native mirror planes of the aperture."""
    nu, nv = 20, 18
    eu = np.full((nu, nv+1), np.inf, dtype=complex)
    ev = np.full((nu+1, nv), np.inf, dtype=complex)
    ew = np.full((nu+1, nv+1), np.inf, dtype=complex)
    eu[2:14, 4:13] = 2
    ev[3:14, 3:13] = 2
    ew[3:14, 4:13] = 2
    solver = FDFD_2D_mode_solver(frequency, .02/12, .02/12, 7, eu, ev, ew,
        np.ones_like(ev), np.ones_like(eu), np.ones((nu, nv)))
    solver.retain_tracking_operator = True
    solver.calculate_diagnostics = True
    solver.solve()
    return solver


def no_symmetry(solvers):
    return dict(detected='none', candidates=[], anchors=[], groups=[])


def test_physical_te_tm_separation_precedes_detection_without_native_symmetry(monkeypatch, tmp_path):
    selected = []
    select = symmetry.longitudinal_gradient_blocks
    detect = symmetry.detect_symmetry

    def track_selection(*args, **kwargs):
        result = select(*args, **kwargs)
        selected.append(result)
        return result

    def detect_after_selection(solvers):
        assert len(selected) == len(solvers)
        return detect(solvers)

    monkeypatch.setattr(symmetry, 'longitudinal_gradient_blocks', track_selection)
    monkeypatch.setattr(symmetry, 'detect_symmetry', detect_after_selection)
    with _configured_solver_constants():
        bank = [shifted_rectangle(f*1e9) for f in (19, 20, 21)]
        for solver in bank:
            stress_rotation(solver, [2, 3])
        owner, grid, e, h = aligned_bank(bank, (3, 4))
        assert owner.symmetry_detection['detected'] == 'none'
        assert owner.symmetry_diagnostics == []
        assert owner.separation_attempts == []
        record = owner.separation_diagnostics[0]
        assert [len(b) for b in record['blocks']] == [1, 1]
        for k, row in record['anchors'].items():
            le = np.array([np.linalg.norm(a[2])**2 for a in e[k]])
            lh = owner._tracking_impedance**2*np.array([np.linalg.norm(a[2])**2 for a in h[k]])
            np.testing.assert_allclose(sorted(le/(le+lh)), [0, 1], atol=1e-10)
            assert max(row['residual']) < 1e-10
            assert row['field_residual'] < 1e-10
            raw = [owner._fields_from_solver_mode(bank[k], m) for m in (3, 4)]
            raw_frame = _frame([a[0] for a in raw], [a[1] for a in raw], owner._tracking_impedance)
            np.testing.assert_allclose(raw_frame @ row['transform'],
                                       _frame(e[k], h[k], owner._tracking_impedance), atol=1e-9)
        with h5py.File(tmp_path/'separation.h5', 'w') as output:
            write_diagnostics(output, owner)
            assert 'mode_separation/1/anchor0/transform' in output
            assert 'symmetry_subspaces' not in output
            assert output['mode_separation'].attrs['RejectedTrials'] == '[]'
        # A full-rank moment check cannot bypass missing C4v certification.
        with pytest.raises(ValueError, match='certified C4v E'):
            _polarize_resolved_pairs(owner, grid, bank, (3, 4), e, h, 'x')


def test_gradient_axis_separation_without_symmetry_is_gauge_invariant(monkeypatch):
    monkeypatch.setattr(symmetry, 'detect_symmetry', no_symmetry)
    with _configured_solver_constants():
        bank = [fd_solver(f*1e9, (12, 12), .02/12, (2, 2, 2), count=8) for f in (19, 20)]
        changed = deepcopy(bank)
        for solver in changed:
            stress_rotation(solver, [4, 5])
        frames = []
        for solvers in (bank, changed):
            owner, _, e, h = aligned_bank(solvers, (5, 6))
            assert owner.symmetry_diagnostics == []
            assert owner.separation_attempts == []
            for k in range(2):
                best = [rectangular_templates(a, b)[0] for a, b in zip(e[k], h[k])]
                assert {a['name'] for a in best} == {'TE_20', 'TE_02'}
                assert min(a['score'] for a in best) > 1-1e-10
            frames.append([_frame(a, b, owner._tracking_impedance) for a, b in zip(e, h)])
        for actual, expected in zip(*frames):
            np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_total_gradient_splits_only_unresolved_directional_block():
    # A selector-kernel example, not permission to mix distinct physical cutoffs.
    x = np.linspace(0, 1, 33)
    patterns = np.stack([np.sin(n*np.pi*x[:, None])*np.sin(n*np.pi*x[None, :]) for n in (1, 2)], axis=-1)
    rotation = np.array([[1, -1j], [-1j, 1]])/np.sqrt(2)
    fields = patterns @ rotation
    blocks, record = symmetry.longitudinal_gradient_blocks(fields, np.zeros_like(fields), (1/32, 1/32), 1.)
    assert [b.shape[1] for b in blocks] == [1, 1]
    assert all(r['sum_split'] and not r['directional_split'] for r in record['gradients'])
    np.testing.assert_allclose(abs(rotation @ np.column_stack(blocks)), np.eye(2), atol=1e-12)
    assert len(record['candidates']) == 2


@pytest.mark.parametrize('m,expected_sizes', [(2, [1, 1]), (3, [2])])
def test_exact_circular_modes_need_symmetry_beyond_three_scalar_operators(m, expected_sizes):
    from testing.eigenmode_support.identification.gradient_first_investigation import circular_forms
    metric, difference, total, _, _ = circular_forms('TE', [(m, 1, 'cos'), (m, 1, 'sin')])
    blocks, record = symmetry._separate_longitudinal_forms(np.zeros_like(metric), metric,
                                                         difference, total, 2.)
    assert [b.shape[1] for b in blocks] == [2]
    assert len(record['candidates']) == 1
    rotation = -np.eye(2) if m == 2 else np.array([[0., 1.], [-1., 0.]])
    mirror = np.diag([-1., 1.])
    refined, leakage = symmetry.irreducible_blocks((rotation, mirror))
    assert [b.shape[1] for b in refined] == expected_sizes
    assert leakage < 1e-12
    transform = np.column_stack(refined)
    columns = [np.array([0]), np.array([1])] if m == 2 else [np.array([0, 1])]
    names = symmetry.c4v_irrep_names([transform.conj().T @ a @ transform for a in (rotation, mirror)], columns)
    assert set(names) == ({'B1', 'B2'} if m == 2 else {'E'})


@pytest.mark.parametrize('metric_case', ['zero', 'singular', 'unresolved'])
def test_unresolved_longitudinal_gradient_subspace_is_not_arbitrarily_split(metric_case):
    fields = np.zeros((4, 4, 2))
    if metric_case == 'singular':
        fields[..., 0] = 1
    elif metric_case == 'unresolved':
        x = np.arange(4)-1.5
        fields[..., 0] = x[:, None]
        fields[..., 1] = x[None, :]
        # Equal directional scores from equally weighted electric and magnetic
        # patterns with exchanged spatial axes.
        blocks, record = symmetry.longitudinal_gradient_blocks(fields, fields.transpose(1, 0, 2), (1, 1), 1.)
        assert len(blocks) == 1 and blocks[0].shape == (2, 2)
        return
    blocks, record = symmetry.longitudinal_gradient_blocks(fields, np.zeros_like(fields), (1, 1), 1.)
    assert len(blocks) == 1 and blocks[0].shape == (2, 2)
    assert not record['longitudinal']['applied']


@pytest.mark.parametrize('certification', ['none', 'c2v', 'singlets'])
def test_user_polarization_requires_certified_e_pair_even_with_good_moments(certification):
    with _configured_solver_constants():
        bank = [fd_solver(20e9, (12, 12), .02/12, (2, 2, 2), count=4)]
        owner, grid, e, h = aligned_bank(bank, (1, 2))
        from gprMax.eigenmode_tracking import _moments
        assert np.linalg.cond(_moments(owner, grid, e[0])) < 1.01
        if certification == 'none':
            owner.symmetry_diagnostics = []
        elif certification == 'c2v':
            owner.symmetry_diagnostics[0]['symmetry'] = 'c2v'
        else:
            owner.symmetry_diagnostics[0]['irreps'] = ('A2+B2',)
        before = deepcopy(e)
        with pytest.raises(ValueError, match='certified C4v E'):
            _polarize_resolved_pairs(owner, grid, bank, (1, 2), e, h, 'x')
        for a, b in zip(e[0], before[0]):
            for ac, bc in zip(a, b):
                np.testing.assert_array_equal(ac, bc)


@pytest.mark.parametrize('moment', [np.eye(2)*1e-24, np.array([[1., 1.], [0., 0.]]),
                                   np.diag([1., 1e-10])])
def test_certified_e_pair_still_requires_significant_well_conditioned_rank_two_moments(monkeypatch, moment):
    import gprMax.eigenmode_tracking as tracking
    with _configured_solver_constants():
        bank = [fd_solver(20e9, (12, 12), .02/12, (2, 2, 2), count=4)]
        owner, grid, e, h = aligned_bank(bank, (1, 2))
        assert owner.symmetry_diagnostics[0]['irreps'] == ('E',)
        monkeypatch.setattr(tracking, '_moments', lambda *a: moment)
        with pytest.raises(ValueError, match='integrated transverse E'):
            _polarize_resolved_pairs(owner, grid, bank, (1, 2), e, h, 'x')


@pytest.mark.parametrize('failure', ['mixed', 'maxwell', 'measurement', 'rank'])
def test_failed_separation_is_transactional_and_records_physical_failure(monkeypatch, failure):
    import gprMax.eigenmode_tracking as tracking
    with _configured_solver_constants():
        bank = [fd_solver(20e9, (12, 12), .02/12, (2, 2, 2), count=12)]
        owner, grid, e, h = aligned_bank(bank, (7, 8, 9, 10), symmetry=None)
        before_e, before_h = deepcopy(e), deepcopy(h)
        monkeypatch.setattr(symmetry, 'detect_symmetry', no_symmetry)
        if failure == 'mixed':
            monkeypatch.setattr(tracking, '_residuals', lambda *a: np.ones(4))
        elif failure == 'maxwell':
            monkeypatch.setattr(type(bank[0]), 'tracking_field_residuals', lambda *a: np.ones(4))
        elif failure == 'measurement':
            def reject(*args):
                raise ValueError('Test measurement precision budget rejection')
            monkeypatch.setattr(symmetry, '_validate_measurement_basis', reject)
        else:
            original = symmetry.longitudinal_gradient_blocks
            def rank_loss(*args, **kwargs):
                blocks, record = original(*args, **kwargs)
                return (blocks[0],)*4, record
            monkeypatch.setattr(symmetry, 'longitudinal_gradient_blocks', rank_loss)
        symmetry.align_symmetry_subspaces(owner, grid, [20e9], bank, (7, 8, 9, 10), e, h)
        assert owner.separation_diagnostics == []
        assert len(owner.separation_attempts) == 1
        for actual, expected in ((e, before_e), (h, before_h)):
            for a, b in zip(actual[0], expected[0]):
                for ac, bc in zip(a, b):
                    if owner.transport_diagnostics:
                        np.testing.assert_allclose(ac, bc, atol=1e-9)
                    else:
                        np.testing.assert_array_equal(ac, bc)


def test_unstable_split_at_neighbor_keeps_entire_previous_bank(monkeypatch):
    with _configured_solver_constants():
        bank = [fd_solver(f*1e9, (12, 12), .02/12, (2, 2, 2), count=12) for f in (19, 20)]
        owner, grid, e, h = aligned_bank(bank, (7, 8, 9, 10), symmetry=None)
        before_e, before_h = deepcopy(e), deepcopy(h)
        select = symmetry.longitudinal_gradient_blocks
        calls = []

        def changed_split(*args, **kwargs):
            blocks, record = select(*args, **kwargs)
            calls.append(record)
            if len(calls) == 2:
                record['candidates'] = [np.arange(4)]
                return (np.eye(4),), record
            return blocks, record

        monkeypatch.setattr(symmetry, 'longitudinal_gradient_blocks', changed_split)
        monkeypatch.setattr(symmetry, 'detect_symmetry', no_symmetry)
        symmetry.align_symmetry_subspaces(owner, grid, [19e9, 20e9], bank, (7, 8, 9, 10), e, h)
        assert owner.separation_diagnostics == []
        assert 'closer anchors' in owner.separation_attempts[0]['reason']
        assert owner.transport_diagnostics[0]['blocks'] == ((7, 8, 9, 10),)
        for actual, expected in ((e, before_e), (h, before_h)):
            for row, original in zip(actual, expected):
                for a, b in zip(row, original):
                    for ac, bc in zip(a, b):
                        np.testing.assert_allclose(ac, bc, atol=1e-9)


def test_final_procrustes_transports_only_unresolved_block_after_symmetry(monkeypatch, tmp_path):
    select = symmetry.longitudinal_gradient_blocks
    align = symmetry._align_symmetry_subspaces
    stages = []
    before_transport = []
    rotate_unresolved = [False]

    def partially_resolved(*args, **kwargs):
        blocks, record = select(*args, **kwargs)
        # Keep two physically resolved singlets and deliberately withhold a
        # directional split in the other longitudinal family.
        record['gradients'] = record['gradients'][:2]
        return (blocks[0], blocks[1], np.column_stack(blocks[2:])), record

    def observe(owner, grid, frequencies, solvers, modes, e, h, *, symmetry=None):
        stages.append(symmetry)
        if symmetry in ('c4v', 'c2v'):
            raise ValueError('Withheld symmetry certification for transport regression')
        if symmetry == 'procrustes':
            before_transport.append((deepcopy(e), deepcopy(h)))
            if rotate_unresolved[0]:
                # A harmless complex gauge at one neighbor must be removed by
                # final transport without touching the two resolved singlets.
                u = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
                for fields in (e, h):
                    for component in range(3):
                        mixed = np.stack([fields[2][j][component] for j in (2, 3)], axis=-1) @ u
                        for j in (2, 3):
                            fields[2][j] = list(fields[2][j])
                            fields[2][j][component] = mixed[..., j-2]
        return align(owner, grid, frequencies, solvers, modes, e, h, symmetry=symmetry)

    monkeypatch.setattr(symmetry, 'longitudinal_gradient_blocks', partially_resolved)
    monkeypatch.setattr(symmetry, '_align_symmetry_subspaces', observe)
    with _configured_solver_constants():
        bank = [fd_solver(f*1e9, (12, 12), .02/12, (2, 2, 2), count=12) for f in (19, 20, 21)]
        _, _, expected_e, expected_h = aligned_bank(bank, (7, 8, 9, 10))
        rotate_unresolved[0] = True
        stages.clear()
        owner, _, e, h = aligned_bank(bank, (7, 8, 9, 10))
        assert stages == [None, 'c4v', 'c2v', 'procrustes']
        assert owner.symmetry_diagnostics == []
        assert owner.transport_attempts == []
        record = owner.transport_diagnostics[0]
        assert tuple(map(len, record['blocks'])) == (1, 1, 2)
        assert owner.degenerate_diagnostics[0]['orientation'] == 'unresolved_procrustes'
        for actual, expected in ((e, expected_e), (h, expected_h)):
            for row, reference in zip(actual, expected):
                for a, b in zip(row, reference):
                    for ac, bc in zip(a, b):
                        np.testing.assert_allclose(ac, bc, atol=1e-9)
        for actual, previous in zip((e, h), before_transport[-1]):
            for row, reference in zip(actual, previous):
                for a, b in zip(row[:2], reference[:2]):
                    for ac, bc in zip(a, b):
                        np.testing.assert_allclose(ac, bc, atol=1e-9)
        for row in record['anchors'].values():
            assert max(row['residual']) < 1e-10
            assert row['field_residual'] < 1e-10
            assert np.linalg.eigvalsh(row['power']).min() > .9
        with h5py.File(tmp_path/'transport.h5', 'w') as output:
            write_diagnostics(output, owner)
            np.testing.assert_array_equal(output['unresolved_transport/1'].attrs['BlockMethods'],
                                           [b'phase', b'phase', b'procrustes'])
            np.testing.assert_allclose(output['unresolved_transport/1/anchor2/transform'],
                                       owner.degenerate_diagnostics[0]['anchors'][2]['transform'])


def test_certified_e_pair_keeps_symmetry_orientation_instead_of_procrustes():
    with _configured_solver_constants():
        bank = [fd_solver(f*1e9, (12, 12), .02/12, (2, 2, 2), count=4) for f in (19, 20)]
        owner, grid, e, h = aligned_bank(bank, (1, 2))
        assert owner.symmetry_diagnostics[0]['irreps'] == ('E',)
        assert owner.transport_diagnostics == []
        _polarize_resolved_pairs(owner, grid, bank, (1, 2), e, h, 'x')
        assert owner.polarization_diagnostics[0]['modes'] == (1, 2)
