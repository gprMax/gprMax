"""Whole-cluster recovery must preserve physical spans, not raw columns."""
from copy import deepcopy
import h5py
import numpy as np
import pytest

from gprMax.eigenmode_tracking import track_solver_bank, write_diagnostics
from gprMax.sources import EigenmodeAnchorMismatchError
from testing.validation.persistent_mode_groups import make_owner, make_solver, pair_crossing_bank, unitary


def projector(a):
    q = np.linalg.qr(a)[0]
    return q @ q.conj().T


@pytest.mark.parametrize('angle', [0, 10, 15, 20])
@pytest.mark.parametrize('reference', [-1., 0., 1.])
def test_partially_mixed_crossing_preserves_subgroup_projectors(angle, reference):
    theta = np.deg2rad(angle)
    identity = np.eye(2)
    rotation = np.block([[np.cos(theta)*identity, -np.sin(theta)*identity],
                         [np.sin(theta)*identity, np.cos(theta)*identity]])
    solvers = [make_solver([-4, -4, -4.04, -4.04]),
               make_solver([-4]*4, rotation),
               make_solver([-4, -4, -3.96, -3.96])]
    owner = make_owner(reference)
    track_solver_bank(owner, [-1, 0, 1], solvers, 4)
    assert sorted(map(len, owner.degenerate)) == [2, 2]
    for group in owner.degenerate:
        columns = np.asarray(group) - 1
        expected = projector(solvers[0].Ea[:, columns])
        for solver in solvers:
            np.testing.assert_allclose(projector(solver.Ea[:, columns]), expected, atol=1e-12)
    assert bool(owner.tracking_diagnostics['subspace_recovery']) == bool(angle)


@pytest.mark.parametrize('reference', [-1., 0., 1.])
@pytest.mark.parametrize('seed', [3, 19, 71])
def test_fully_mixed_pair_pair_crossing(reference, seed):
    axis = [-1., 0., 1.]
    solvers = pair_crossing_bank(axis, seed=seed, two_pairs=True, mixed_crossing=True)
    owner = make_owner(reference)
    track_solver_bank(owner, axis, solvers, 4)
    assert sorted(map(len, owner.degenerate)) == [2, 2]
    truth = [np.diag([1, 1, 0, 0]), np.diag([0, 0, 1, 1])]
    for group in owner.degenerate:
        indices = np.asarray(group) - 1
        target = min(truth, key=lambda p: np.linalg.norm(projector(solvers[0].Ea[:, indices])-p))
        for solver in solvers:
            np.testing.assert_allclose(projector(solver.Ea[:, indices]), target, atol=1e-12)
    record = owner.tracking_diagnostics['subspace_recovery']
    assert any(r['anchor'] == 1 for r in record)
    assert np.max(record[0]['residual']) < 1e-13


def test_insufficient_history_does_not_invent_two_pairs():
    rng = np.random.default_rng(7)
    solvers = [make_solver([-4]*4, unitary(rng, 4)) for _ in range(3)]
    owner = make_owner()
    track_solver_bank(owner, [-1,0,1], solvers, 4)
    assert owner.degenerate == ((1,2,3,4),)
    assert owner.tracking_diagnostics['subspace_recovery'] == ()


def test_failed_bank_is_transactional_after_successful_recovery():
    solvers = pair_crossing_bank([-1,0,1], mixed_crossing=True)
    # A bad final residual fails after the middle anchor could be recovered.
    solvers[2].field_residuals[:] = 1.
    originals = [deepcopy(s.__dict__) for s in solvers]
    with pytest.raises(EigenmodeAnchorMismatchError):
        track_solver_bank(make_owner(-1), [-1,0,1], solvers, 3)
    for solver, original in zip(solvers, originals):
        for name in ('Ea','Ht','eigenvectors','eigenvalues','field_residuals'):
            np.testing.assert_array_equal(getattr(solver,name), original[name])


def test_resolved_split_cannot_be_rotated_to_pass_overlap():
    rng=np.random.default_rng(3)
    values=[-4.,-4.+1e-6,-4.+2e-6,-4.+3e-6]
    solvers=[make_solver(values),make_solver(values,unitary(rng,4))]
    # Both records describe exact, distinct eigenpairs in different operators.
    for solver in solvers:
        from scipy.sparse import csr_matrix
        solver.operator=csr_matrix(solver.Ea @ np.diag(solver.eigenvalues) @ solver.Ea.conj().T)
    with pytest.raises(EigenmodeAnchorMismatchError):
        track_solver_bank(make_owner(-1),[-1,1],solvers,4)


def test_recovery_transform_is_persisted_and_reconstructs_fields(tmp_path):
    solvers=pair_crossing_bank([-1,0,1],two_pairs=True,mixed_crossing=True)
    original=solvers[1].Ea.copy()
    owner=make_owner(-1)
    mapping=track_solver_bank(owner,[-1,0,1],solvers,4)
    record=owner.tracking_diagnostics['subspace_recovery'][0]
    reconstructed=original[:,record['candidate_indices']] @ record['transform']
    np.testing.assert_allclose(reconstructed[:,mapping[1]],solvers[1].Ea,atol=1e-13)
    owner.degenerate_diagnostics=[]
    owner.tracking='auto'
    owner.verification='none'
    with h5py.File(tmp_path/'tracking.h5','w') as f:
        write_diagnostics(f,owner)
    with h5py.File(tmp_path/'tracking.h5') as f:
        np.testing.assert_allclose(f['mode_tracking/subspace_recovery/0/transform'],record['transform'])


def test_projection_preserves_nonorthogonal_subgroups():
    from scipy.sparse import csr_matrix
    shear=np.eye(4,dtype=complex); shear[0,2]=.15
    solvers=pair_crossing_bank([-1,0,1],two_pairs=True,mixed_crossing=True)
    for solver in solvers:
        solver.Ea=shear @ solver.Ea; solver.Ht=solver.Ea.copy()
        solver.eigenvectors=solver.Ea.copy()
        solver.operator=csr_matrix(solver.Ea @ np.diag(solver.eigenvalues) @ np.linalg.inv(solver.Ea))
    owner=make_owner(-1)
    track_solver_bank(owner,[-1,0,1],solvers,4)
    for group in owner.degenerate:
        members=np.asarray(group)-1
        p=projector(solvers[0].Ea[:,members])
        for solver in solvers:
            np.testing.assert_allclose(projector(solver.Ea[:,members]),p,atol=1e-12)


def test_multiple_consecutive_coincident_anchors_retain_subgroups():
    rng=np.random.default_rng(123)
    solvers=pair_crossing_bank([-1,0,0,1],two_pairs=True,mixed_crossing=True)
    for k in (1,2):
        solvers[k]=make_solver([-4]*4,unitary(rng,4))
    owner=make_owner(-1)
    track_solver_bank(owner,[-1,-.1,.1,1],solvers,4)
    assert sorted(map(len,owner.degenerate))==[2,2]
    for group in owner.degenerate:
        ix=np.asarray(group)-1
        truth=projector(solvers[0].Ea[:,ix])
        for solver in solvers:
            np.testing.assert_allclose(projector(solver.Ea[:,ix]),truth,atol=1e-12)


def test_physical_hybrid_crossing_is_invariant_to_eigensolver_rotation():
    from testing.validation.general_subspace_tracking import loaded_solver, stress_rotation
    from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
    from gprMax.eigenmode_tracking import _tracking_vectors
    import gprMax.config as config

    with _configured_solver_constants():
        bank = [loaded_solver(t, cells=12) for t in (-.01, 0., .01)]
        rotated = deepcopy(bank)
        stress_rotation(rotated[1], [6,7,8,9])
        results = []
        for solvers in (bank, rotated):
            owner = make_owner(-.01)
            owner._tracking_impedance = config.sim_config.em_consts['z0']
            track_solver_bank(owner, [-.01,0.,.01], solvers, 10)
            assert (7,8) in owner.degenerate and (9,10) in owner.degenerate
            frames = [_tracking_vectors(s, owner._tracking_impedance) for s in solvers]
            results.append([projector(frames[1][:,g]) for g in ((6,7),(8,9))])
            for s in solvers:
                assert max(s.tracking_field_residuals(np.arange(10))) < 1e-10
        np.testing.assert_allclose(results[0], results[1], atol=1e-11)


@pytest.mark.parametrize('component', ['Ew', 'Hw', 'Hv'])
def test_reconstructed_maxwell_check_detects_corrupted_field(component):
    from testing.validation.general_subspace_tracking import loaded_solver
    from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants

    with _configured_solver_constants():
        solver = loaded_solver(.1, cells=12)
        assert max(solver.tracking_field_residuals([6,7,8,9])) < 1e-10
        getattr(solver, component)[..., 6] *= 1.2
        assert solver.tracking_field_residuals([6])[0] > 1e-3
