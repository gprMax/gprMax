"""Counterexamples and positive controls for the proposed mode identifiers."""
import numpy as np
import pytest
from scipy.linalg import eigh

from gprMax.eigenmode_symmetry import c4v_irrep_names, orient_irreducible_block
from testing.eigenmode_support.identification.gradient_first_investigation import (
    circular_forms, gradient_basis, native_square_cases,
)


@pytest.mark.parametrize('family', ['TE', 'TM'])
def test_sum_recovers_radial_orders_when_difference_is_zero(family):
    metric, difference, total, _, roots = circular_forms(family, [(0, 1, 'cos'), (0, 2, 'cos')])
    np.testing.assert_allclose(difference, 0, atol=1e-12)
    np.testing.assert_allclose(eigh(total, metric, eigvals_only=True), np.square(roots), atol=1e-11)
    # Use invertible complex, nonorthogonal input gauges, not only rotations.
    rng = np.random.default_rng(329)
    for _ in range(4):
        gauge = np.eye(2)+.2*(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        m, d, s = [gauge.conj().T @ a @ gauge for a in (metric, difference, total)]
        c = gradient_basis(np.zeros_like(m) if family == 'TE' else m, m, d, s, use_sum=True)
        actual = gauge @ c
        actual /= np.linalg.norm(actual, axis=0)
        np.testing.assert_allclose(abs(actual), np.eye(2), atol=1e-12)


@pytest.mark.parametrize('family', ['TE', 'TM'])
@pytest.mark.parametrize('m', [2, 3, 4])
def test_both_gradients_fail_to_orient_higher_circular_pairs(family, m):
    metric, difference, total, _, roots = circular_forms(family, [(m, 1, 'cos'), (m, 1, 'sin')])
    np.testing.assert_allclose(difference, 0, atol=1e-12)
    np.testing.assert_allclose(total, roots[0]**2*metric, atol=1e-11)


@pytest.mark.parametrize('family,m,nonzero', [('TE', 1, True), ('TE', 3, False), ('TM', 1, False)])
def test_e_irrep_does_not_guarantee_nonzero_electric_moments(family, m, nonzero):
    _, _, _, moments, _ = circular_forms(family, [(m, 1, 'cos'), (m, 1, 'sin')])
    if nonzero:
        assert np.linalg.svd(moments, compute_uv=False).min() > 1
    else:
        assert np.linalg.norm(moments) < 1e-12


def test_equal_absolute_difference_is_not_an_e_irrep_certificate():
    # Axis partners in a square TE20/TE02 span swap under a quarter turn,
    # whereas a genuine E doublet acquires a minus sign after a half turn.
    swap = np.array([[0., 1.], [1., 0.]])
    rotation = np.array([[0., -1.], [1., 0.]])
    difference = np.diag([-2., 2.])
    for action in (swap, rotation):
        np.testing.assert_allclose(action.conj().T @ difference @ action, -difference)
    assert c4v_irrep_names((rotation, np.diag([-1., 1.])), [[0, 1]]) == ('E',)
    with pytest.raises(ValueError, match='E doublet'):
        c4v_irrep_names((swap, -np.eye(2)), [[0, 1]])


def test_zero_moment_e_pair_still_has_a_valid_parity_basis():
    rotation = np.array([[0., -1.], [1., 0.]])
    mirrors = (np.diag([-1., 1.]), np.diag([1., -1.]))
    basis, record = orient_irreducible_block(np.eye(2), np.eye(2), np.zeros((2, 2)), 1.,
                                            mirrors, rotation=rotation)
    np.testing.assert_allclose(basis, np.eye(2))
    assert record['method'] == 'mirror_parity'


def test_invariant_sum_need_not_commute_with_absolute_difference_across_e_copies():
    # A symmetry-respecting positive-gradient matrix example, not a Maxwell
    # geometry: symmetry alone does not guarantee simultaneous identifiers.
    r = np.kron(np.eye(2), [[0., -1.], [1., 0.]])
    mirror = np.kron(np.eye(2), np.diag([-1., 1.]))
    d = np.kron(np.diag([1., 2.]), np.diag([1., -1.]))
    s = np.kron([[4., 1.], [1., 4.]], np.eye(2))
    assert np.linalg.eigvalsh(s+d).min() > 0
    assert np.linalg.eigvalsh(s-d).min() > 0
    np.testing.assert_allclose(r.T @ d @ r, -d)
    np.testing.assert_allclose(mirror.T @ d @ mirror, d)
    for action in (r, mirror):
        np.testing.assert_allclose(action.T @ s @ action, s)
        np.testing.assert_allclose(action.T @ (d@d) @ action, d@d)
    assert np.linalg.norm(s@d@d-d@d@s) > 1


def test_native_square_proposal_matches_current_basis_but_does_not_improve_it():
    for row in native_square_cases():
        for name in ('proposed_templates', 'current_templates'):
            assert min(a['score'] for a in row[name]) > 1-1e-10
        assert max(row['mixed_residual']) < 1e-10
        assert row['rotation_closure'] < 1e-10
        expected_trace = 2 if row['labels'] == (5, 6) else -len(row['labels'])
        assert abs(row['rotation_squared_trace']-expected_trace) < 1e-10
