"""Moment/parity orientation must remove raw gauges and preserve E/H fields."""
from copy import deepcopy

import numpy as np
import pytest

from gprMax.eigenmode_symmetry import orient_irreducible_block, mirror_u, mirror_v
from gprMax.eigenmode_tracking import _frame, _moments
from testing.eigenmode_support.subspaces.general_subspace_tracking import fd_solver, _configured_solver_constants
from tests.test_eigenmode_auto_symmetry import aligned_bank


@pytest.mark.parametrize('moment,method',[(np.eye(2),'electric_moment'),
                                        (np.zeros((2,2)),'mirror_parity'),
                                        (np.diag([1.,1e-14]),'mirror_parity')])
def test_complex_basis_rotation_has_unique_oriented_partners(moment,method):
    rng=np.random.default_rng(523)
    mirrors=(np.diag([-1,1]),np.diag([1,-1]))
    rotation=np.array([[0,-1],[1,0]])
    for _ in range(8):
        raw=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
        result,diagnostic=orient_irreducible_block(raw,np.eye(2),moment,2.,mirrors,rotation=rotation)
        np.testing.assert_allclose(result,np.eye(2),atol=1e-12)
        assert diagnostic['method']==method
        np.testing.assert_array_equal(diagnostic['parity'],[[-1,1],[1,-1]])
        assert diagnostic['parity_residual']<1e-12


def test_phase_continuation_does_not_mix_parity_channels():
    previous=np.diag(np.exp(1j*np.array([.8,-.3])))
    result,record=orient_irreducible_block(np.eye(2),np.eye(2),np.eye(2),2.,
        (np.diag([-1,1]),np.diag([1,-1])),previous=previous,previous_parity=[[-1,1],[1,-1]])
    np.testing.assert_allclose(result,previous,atol=1e-12)
    np.testing.assert_allclose(record['channel_overlaps'],1.,atol=1e-12)


def test_wrong_vector_parity_is_rejected():
    with pytest.raises(ValueError,match='parity'):
        orient_irreducible_block(np.eye(2),np.eye(2),np.eye(2),2.,
                                (np.array([[0,1],[1,0]]),np.diag([1,-1])))


def test_spurious_moment_direction_uses_mirror_fallback():
    angle=.4
    moment=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    result,record=orient_irreducible_block(np.eye(2),np.eye(2),moment,2.,
        (np.diag([-1,1]),np.diag([1,-1])),rotation=np.array([[0,-1],[1,0]]))
    assert record['method']=='mirror_parity'
    np.testing.assert_allclose(result,np.eye(2),atol=1e-12)


def assert_field_parities(owner,bank,e,h,labels):
    for record in owner.symmetry_diagnostics:
        for k,row in record['anchors'].items():
            for group,oriented in zip(record['blocks'],row['orientation']):
                for m,parities in zip(group,oriented['parity']):
                    j=labels.index(m)
                    original=_frame([e[k][j]],[h[k][j]],owner._tracking_impedance)
                    for action,sign in zip((mirror_u,mirror_v),parities):
                        te,th=action(bank[k],e[k][j],h[k][j])
                        changed=_frame([te],[th],owner._tracking_impedance)
                        assert np.linalg.norm(changed-sign*original)/np.linalg.norm(original)<1e-10


@pytest.mark.parametrize('labels,method',[((1,2),'electric_moment'),((7,8,9,10),'mirror_parity')])
def test_physical_fundamental_and_higher_order_modes(labels,method):
    with _configured_solver_constants():
        bank=[fd_solver(f*1e9,(12,12),.02/12,(2,2,2),count=12) for f in (19,20,21)]
        owner,grid,e,h=aligned_bank(bank,labels)
        assert owner.symmetry_detection['groups'][0]['applied']=='c4v'
        assert_field_parities(owner,bank,e,h,labels)
        for row in owner.symmetry_diagnostics[0]['anchors'].values():
            assert all(d['method']==method for d in row['orientation'])
            assert all(np.min(d['channel_overlaps'])>.99 for d in row['orientation'])
            assert row['field_residual']<1e-10
        if method=='electric_moment':
            moment=_moments(owner,grid,e[0])
            np.testing.assert_allclose(moment/np.linalg.norm(moment,axis=0),np.eye(2),atol=1e-10)


def test_physical_moment_basis_agrees_after_independent_complex_raw_gauges():
    with _configured_solver_constants():
        bank=[fd_solver(f*1e9,(12,12),.02/12,(2,2,2),count=12) for f in (19,20)]
        changed=deepcopy(bank)
        rng=np.random.default_rng(901)
        for solver in changed:
            gauge=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
            for name in ('Eu','Ev','Ew','Hu','Hv','Hw'):
                getattr(solver,name)[...,:2]=getattr(solver,name)[...,:2] @ gauge
        a,_,ae,ah=aligned_bank(bank,(1,2))
        b,_,be,bh=aligned_bank(changed,(1,2))
        for k in range(2):
            np.testing.assert_allclose(_frame(ae[k],ah[k],a._tracking_impedance),
                                       _frame(be[k],bh[k],b._tracking_impedance),rtol=1e-10,atol=1e-8)


@pytest.mark.parametrize('labels',[(1,2),(7,8,9,10)])
def test_without_certified_symmetry_keeps_separated_power_basis(monkeypatch,labels):
    import gprMax.eigenmode_symmetry as symmetry
    # Isolate fallback policy while retaining a real positive-power Maxwell
    # eigenspace. The control deliberately withholds symmetry certification.
    monkeypatch.setattr(symmetry,'detect_symmetry',lambda solvers:dict(detected='none',candidates=[],anchors=[],groups=[]))
    with _configured_solver_constants():
        bank=[fd_solver(f*1e9,(12,12),.02/12,(2,2,2),count=12) for f in (19,20)]
        owner,grid,e,h=aligned_bank(bank,labels)
        assert owner.symmetry_detection['groups'][0]['applied']=='longitudinal_gradient'
        assert [len(b) for b in owner.separation_diagnostics[0]['blocks']] == [1]*len(labels)
        assert owner.symmetry_diagnostics==[]
        assert owner.degenerate_diagnostics[0]['modes']==labels
        from gprMax.eigenmode_tracking import _power
        power = _power(owner,grid,e[0],h[0])
        np.testing.assert_allclose(np.diag(power),np.ones(len(labels)),atol=1e-10)
        # Families retain their cross-power instead of being remixed by global whitening.
        assert np.linalg.eigvalsh(power).min() > .9



def test_automatic_moment_selection_is_not_a_user_override_on_rebuild():
    from gprMax.eigenmode_tracking import align_groups
    with _configured_solver_constants():
        bank=[fd_solver(f*1e9,(12,12),.02/12,(2,2,2),count=12) for f in (19,20)]
        owner,grid,e,h=aligned_bank(bank,(1,2))
        assert owner.mode_polarizations=={}
        align_groups(owner,grid,[s.frequency for s in bank],bank,(1,2),e,h,np.ones((2,2),bool))
        assert owner.symmetry_detection['groups'][0]['applied']=='c4v'
        assert all(row['orientation'][0]['method']=='electric_moment'
                   for row in owner.symmetry_diagnostics[0]['anchors'].values())
