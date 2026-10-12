"""Separate equivalent symmetry copies without analytical mode templates."""
from copy import deepcopy

import numpy as np
import pytest

from gprMax.eigenmode_symmetry import longitudinal_blocks, axis_sensitive_blocks
from gprMax.eigenmode_tracking import _frame
from tests.test_eigenmode_auto_symmetry import aligned_bank
from testing.eigenmode_support.subspaces.general_subspace_tracking import fd_solver, stress_rotation
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
from testing.eigenmode_support.identification.tracking_classification_edges import rectangular_templates


@pytest.mark.parametrize('dimension,copies', [(1,2), (2,2), (2,3), (2,5)])
def test_longitudinal_families_are_invariant_under_complex_raw_gauges(dimension,copies):
    n = dimension*copies
    rng = np.random.default_rng(152)
    r = np.kron(np.eye(copies), [[0,-1],[1,0]]) if dimension == 2 else np.eye(n)
    s = np.kron(np.eye(copies), np.diag([-1,1])) if dimension == 2 else np.eye(n)
    # Families need not be power-orthogonal: this is why global whitening
    # must not be applied again after resolving longitudinal content.
    mixing = np.eye(copies) + .08*np.triu(np.ones((copies,copies)),1)
    physical = np.kron(mixing,np.eye(dimension))
    rho = np.repeat(np.linspace(0,1,copies),dimension)
    ez = np.diag(np.sqrt(rho)) @ np.linalg.inv(physical)
    hz = np.diag(np.sqrt(1-rho)) @ np.linalg.inv(physical)
    for _ in range(4):
        gauge = np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
        blocks, error, record = longitudinal_blocks(
            [gauge.conj().T @ d @ gauge for d in (r,s)],ez@gauge,hz@gauge,100.)
        assert record['applied'] and error < 1e-12
        assert [b.shape[1] for b in blocks] == [dimension]*copies
        for j,b in enumerate(blocks):
            actual = gauge @ b
            expected = np.linalg.qr(physical[:,j*dimension:(j+1)*dimension])[0]
            np.testing.assert_allclose(actual@actual.conj().T,expected@expected.conj().T,atol=1e-12)


@pytest.mark.parametrize('ez,hz,reason', [
    (np.zeros((2,2)),np.zeros((2,2)),'insignificant'),
    (np.diag([0.,1.]),np.zeros((2,2)),'singular'),
    (np.eye(2),np.eye(2),'unresolved'),
])
def test_unresolved_longitudinal_fields_keep_existing_decomposition(ez,hz,reason):
    blocks, error, record = longitudinal_blocks([np.eye(2),np.eye(2)],ez,hz,10.)
    assert not record['applied'] and reason in record['reason']
    assert sum(b.shape[1] for b in blocks) == 2
    assert error < 1e-12


def test_a_longitudinal_split_must_preserve_symmetry():
    rotation = np.array([[0.,-1.],[1.,0.]])
    blocks, _, record = longitudinal_blocks([rotation,np.diag([-1,1])],
                                            np.diag([0.,1.]),np.diag([1.,0.]),10.)
    assert not record['applied']
    assert record['reason'] == 'longitudinal_split_not_symmetry_invariant'
    assert len(blocks) == 1 and blocks[0].shape == (2,2)


@pytest.mark.parametrize('shape,labels,expected', [
    ((16,12),(3,4),{'TE_11','TM_11'}),
    ((16,16),(7,8,9,10),{'TE_12','TE_21','TM_12','TM_21'}),
])
def test_physical_modes_recover_analytic_fields_without_runtime_templates(shape,labels,expected):
    with _configured_solver_constants():
        bank = [fd_solver(f*1e9,shape,.02/16,(2,2,2),count=12) for f in (19,20,21)]
        rotated = deepcopy(bank)
        for s in rotated:
            stress_rotation(s,np.asarray(labels)-1)
        outputs = []
        for solvers in (bank,rotated):
            owner,_,e,h = aligned_bank(solvers,labels)
            assert owner.symmetry_diagnostics
            for k,row in enumerate(e):
                scores = [rectangular_templates(ee,hh)[0] for ee,hh in zip(row,h[k])]
                assert {s['name'] for s in scores} == expected
                assert min(s['score'] for s in scores) > 1-1e-10
            assert all(r['field_residual']<1e-10 for r in owner.symmetry_diagnostics[0]['anchors'].values())
            outputs.append([_frame(ee,hh,owner._tracking_impedance) for ee,hh in zip(e,h)])
        for a,b in zip(*outputs):
            overlap = abs(a.conj().T@b)**2 / (np.sum(abs(a)**2,axis=0)[:,None]*np.sum(abs(b)**2,axis=0))
            assert np.min(np.max(overlap,axis=1)) > 1-1e-10


@pytest.mark.parametrize('labels,expected', [
    ((5,6), {'TE_20','TE_02'}),
    ((15,16,17,18), {'TE_13','TE_31','TM_13','TM_31'}),
    ((31,32,33,34), {'TE_24','TE_42','TM_24','TM_42'}),
])
def test_axis_orientation_recovers_remaining_metallic_indices(labels,expected):
    with _configured_solver_constants():
        bank=[fd_solver(f*1e9,(24,24),.02/24,(2,2,2),count=max(labels)+4) for f in (40,42,44)]
        changed=deepcopy(bank)
        for s in changed:stress_rotation(s,np.asarray(labels)-1)
        for solvers in (bank,changed):
            owner,_,e,h=aligned_bank(solvers,labels)
            assert owner.symmetry_diagnostics
            for k,row in enumerate(e):
                scores=[rectangular_templates(ee,hh)[0] for ee,hh in zip(row,h[k])]
                assert {s['name'] for s in scores} == expected
                assert min(s['score'] for s in scores)>1-1e-10
            for row in owner.symmetry_diagnostics[0]['anchors'].values():
                assert any(r['applied'] for r in row['axis_sensitive'])
                assert row['field_residual']<1e-10
                assert row['irreducible_basis']['transform'].shape == (len(labels),len(labels))


@pytest.mark.parametrize('electric_fraction',[0.,.3,1.])
def test_axis_measure_orients_metallic_or_hybrid_longitudinal_fields(electric_fraction):
    x=(np.arange(32)+.5)/16-1
    patterns=np.stack(np.broadcast_arrays(np.cos(np.pi*x)[:,None],np.cos(np.pi*x)[None,:]),axis=-1)
    patterns/=np.linalg.norm(patterns,axis=(0,1))
    rng=np.random.default_rng(781)
    for _ in range(4):
        u=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
        fields=patterns@u
        blocks,records=axis_sensitive_blocks((np.eye(2)[:,:1],np.eye(2)[:,1:]),
            np.sqrt(electric_fraction)*fields,np.sqrt(1-electric_fraction)*fields,
            (np.eye(2),np.eye(2)),(.0625,.0625),2.)
        assert records[0]['applied']
        actual=u@np.column_stack(blocks)
        np.testing.assert_allclose(abs(actual),[[0,1],[1,0]],atol=1e-12)


def test_axis_measure_retains_unresolved_or_different_family_fields():
    x=np.linspace(-1,1,33)
    patterns=np.stack(np.broadcast_arrays(np.cos(np.pi*x)[:,None],np.cos(np.pi*x)[None,:]),axis=-1)
    blocks=(np.eye(2)[:,:1],np.eye(2)[:,1:])
    # Different TE/TM content must never be recombined to select an axis.
    result,records=axis_sensitive_blocks(blocks,patterns*np.array([1,0]),patterns*np.array([0,1]),
                                         (np.eye(2),np.eye(2)),(1,1),100.)
    assert records == []
    np.testing.assert_array_equal(np.column_stack(result),np.eye(2))


def test_axis_source_and_original_irreducible_basis_are_saved_separately(tmp_path):
    import h5py
    import json
    from gprMax.eigenmode_tracking import write_diagnostics
    with _configured_solver_constants():
        s=fd_solver(20e9,(16,16),.02/16,(2,2,2),count=8)
        owner,_,e,h=aligned_bank([s],(5,6))
        with h5py.File(tmp_path/'axis.h5','w') as f:
            write_diagnostics(f,owner)
        with h5py.File(tmp_path/'axis.h5') as f:
            g=f['symmetry_subspaces/1/anchor0']
            records=json.loads(g['axis_sensitive'].attrs['Records'])
            assert records[0]['applied']
            original=g['irreducible_basis/transform'][:]
            source=g['transform'][:]
            assert abs(np.linalg.det(source))>0
            assert not np.allclose(original,source)
            raw=[owner._fields_from_solver_mode(s,m) for m in (5,6)]
            rawframe=_frame([v[0] for v in raw],[v[1] for v in raw],owner._tracking_impedance)
            np.testing.assert_allclose(rawframe@source,_frame(e[0],h[0],owner._tracking_impedance),atol=1e-9)


def test_axis_measure_retains_equal_directional_scores():
    x=np.linspace(-1,1,33)
    blocks=(np.eye(2)[:,:1],np.eye(2)[:,1:])
    # Same directional scores: retain the established basis.
    a=np.sin(np.pi*x)[:,None]*np.sin(np.pi*x)[None,:]
    b=np.cos(np.pi*x)[:,None]*np.cos(np.pi*x)[None,:]
    fields=np.stack([a,b],axis=-1)
    result,records=axis_sensitive_blocks(blocks,fields,np.zeros_like(fields),
                                         (np.eye(2),np.eye(2)),(1,1),100.)
    assert not records[0]['applied']
    assert records[0]['reason']=='unresolved_directional_scores'
    np.testing.assert_array_equal(np.column_stack(result),np.eye(2))
