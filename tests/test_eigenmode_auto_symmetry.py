"""Automatic symmetry is inferred from the sampled operator, then certified."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.eigenmode_symmetry import detect_symmetry, align_symmetry_subspaces
from gprMax.eigenmode_tracking import align_groups, _power
from gprMax.sources import EigenmodeSource
from testing.eigenmode_support.subspaces.expanded_mode_tracking import fd_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
import gprMax.config as config


def test_retired_waveguide_presets_are_not_accepted():
    from gprMax.eigenmode_tracking import parse_port_options
    from gprMax.user_objects.cmds_multiuse import EigenmodePort
    with pytest.raises(ValueError, match='waveguide_type'):
        parse_port_options(['waveguide_type=square_pec'])
    with pytest.raises((ValueError, TypeError), match='waveguide_type'):
        EigenmodePort(port=1, waveguide_type='square_pec')


def sampled_port(nu=8, nv=8, du=1., dv=1.):
    arrays = dict(eps_r_uu=np.ones((nu,nv+1)), eps_r_vv=np.ones((nu+1,nv)),
                  eps_r_ww=np.ones((nu+1,nv+1)), mu_r_uu=np.ones((nu+1,nv)),
                  mu_r_vv=np.ones((nu,nv+1)), mu_r_ww=np.ones((nu,nv)))
    for kind, prefix in [('pec','eps_r'), ('pmc','mu_r')]:
        for axis in 'uvw':
            arrays[f'{kind}_{axis}_mask'] = np.zeros_like(arrays[f'{prefix}_{axis}{axis}'],dtype=bool)
    return SimpleNamespace(Nu=nu,Nv=nv,du=du,dv=dv,surface_boundary=None,**arrays)


def test_detection_uses_common_sampled_symmetry_and_masks():
    square=sampled_port()
    rectangle=sampled_port(nv=6)
    assert detect_symmetry([square])['detected']=='c4v'
    assert detect_symmetry([rectangle])['detected']=='c2v'
    assert detect_symmetry([sampled_port(dv=2.)])['detected']=='c2v'
    assert detect_symmetry([square,rectangle])['detected']=='c2v'
    anisotropic=sampled_port()
    anisotropic.eps_r_uu *= 2
    assert detect_symmetry([anisotropic])['detected']=='c2v'
    square.pec_u_mask[2,3]=True
    assert detect_symmetry([square])['detected']=='none'


@pytest.mark.parametrize('epsilon,permeability,expected', [
    ((2, 2, 3), (1, 1, 1), 'c4v'),
    ((2, 3, 2), (1, 1, 1), 'c2v'),
    ((2-.02j, 2-.02j, 3-.1j), (1, 1, 1), 'c4v'),
    ((2-.02j, 2-.03j, 3-.1j), (1, 1, 1), 'c2v'),
    ((2, 2, 3), (1, 1, 1.5), 'c4v'),
    ((2, 2, 3), (1, 1.5, 1), 'c2v'),
])
def test_transverse_tensor_equality_controls_square_port_symmetry(epsilon, permeability, expected):
    port=sampled_port()
    for axis,eps,mu in zip('uvw',epsilon,permeability):
        for prefix,value in [('eps_r',eps),('mu_r',mu)]:
            name=f'{prefix}_{axis}{axis}'
            setattr(port,name,np.full(getattr(port,name).shape,value,dtype=complex))
    assert detect_symmetry([port])['detected']==expected


@pytest.mark.parametrize('change', ['loss', 'nan', 'sibc', 'missing'])
def test_uncertified_materials_fall_back(change):
    port=sampled_port()
    if change=='loss':
        port.eps_r_ww=port.eps_r_ww.astype(complex)
        port.eps_r_ww[2,3] += .01j
    elif change=='nan':
        port.eps_r_ww[2,3]=np.nan
    elif change=='sibc':
        port.surface_boundary=object()
    else:
        del port.mu_r_ww
    result=detect_symmetry([port])
    assert result['detected']=='none'
    assert result['anchors'][0]['reasons']


def aligned_bank(bank, labels, symmetry='auto', tracking='auto'):
    owner=EigenmodeSource(None)
    owner.tracking=tracking
    owner.normal_axis=2; owner.normal='z'; owner.direction='+'; owner.transverse_axes=(0,1)
    owner.degenerate=(tuple(labels),); owner.mode_polarizations={}
    owner._tracking_impedance=config.sim_config.em_consts['z0']
    fields=[[owner._fields_from_solver_mode(s,m) for m in labels] for s in bank]
    e=[[v[0] for v in row] for row in fields]; h=[[v[1] for v in row] for row in fields]
    grid=SimpleNamespace(dl=np.array([bank[0].du,bank[0].dv,bank[0].du]))
    axis=[s.frequency for s in bank]
    from testing.eigenmode_support.subspaces.general_subspace_tracking import align_validation_groups
    align_validation_groups(owner,grid,axis,bank,tuple(labels),e,h,np.ones((len(bank),len(labels)),bool),
                            symmetry=symmetry if tracking == 'auto' else None)
    return owner,grid,e,h


def test_legacy_declared_group_skips_automatic_symmetry(monkeypatch):
    import gprMax.eigenmode_symmetry as symmetry

    def unexpected_selection(*args, **kwargs):
        pytest.fail('Legacy tracking entered automatic symmetry selection')

    monkeypatch.setattr(symmetry, 'align_symmetry_subspaces', unexpected_selection)
    with _configured_solver_constants():
        bank = [fd_solver(20e9, (12,12), .02/12, (2,2,2), count=12)]
        owner, _, actual_e, actual_h = aligned_bank(bank, (7,8,9,10), tracking='legacy')
        _, _, expected_e, expected_h = aligned_bank(bank, (7,8,9,10), symmetry=None, tracking='legacy')
    assert not getattr(owner, 'symmetry_diagnostics', None)
    assert not getattr(owner, 'symmetry_detection', None)
    for actual, expected in ((actual_e, expected_e), (actual_h, expected_h)):
        for row, expected_row in zip(actual, expected):
            for field, expected_field in zip(row, expected_row):
                for component, expected_component in zip(field, expected_field):
                    np.testing.assert_array_equal(component, expected_component)


def test_default_square_quartet_decomposes_without_user_symmetry():
    with _configured_solver_constants():
        bank=[fd_solver(f*1e9,(12,12),.02/12,(2,2,2),count=12) for f in (19,20,21)]
        owner,grid,e,h=aligned_bank(bank,(7,8,9,10))
        assert not hasattr(EigenmodeSource(None), 'mode_symmetry')
        assert owner.symmetry_detection['detected']=='c4v'
        assert owner.symmetry_detection['groups'][0]['applied']=='c4v'
        record=owner.symmetry_diagnostics[0]
        assert [len(b) for b in record['blocks']]==[2,2]
        assert record['irreps']==('E','E')
        for k,row in record['anchors'].items():
            assert row['closure']<1e-10
            assert row['field_residual']<1e-10
            power = _power(owner,grid,e[k],h[k])
            assert np.linalg.eigvalsh(power).min() > .9
            for indices in ([0,1], [2,3]):
                np.testing.assert_allclose(power[np.ix_(indices,indices)],np.eye(2),atol=1e-10)
            assert row['longitudinal']['applied']


def test_rectangular_accidental_degeneracy_splits_into_c2v_singlets():
    with _configured_solver_constants():
        solver=fd_solver(20e9,(16,12),.02/16,(2,2,2),count=12)
        i=next(i for i in range(11) if abs(solver.eigenvalues[i]-solver.eigenvalues[i+1])<1e-9)
        owner,_,_,_=aligned_bank([solver],(i+1,i+2))
    assert owner.symmetry_detection['detected']=='c2v'
    assert owner.symmetry_detection['groups'][0]['applied']=='c2v'
    assert [len(b) for b in owner.symmetry_diagnostics[0]['blocks']]==[1,1]
    # Equivalent singlets need not have distinct names (e.g. TE11/TM11).
    assert all(name in ('A1','A2','B1','B2') for name in owner.symmetry_diagnostics[0]['irreps'])


def test_failed_automatic_trials_do_not_change_fallback_fields(monkeypatch):
    import gprMax.eigenmode_symmetry as symmetry
    with _configured_solver_constants():
        bank=[fd_solver(f*1e9,(12,12),.02/12,(2,2,2),count=12) for f in (19,20)]
        owner,grid,e,h=aligned_bank(bank,(7,8,9,10),symmetry=None)
        expected_e,expected_h=deepcopy(e),deepcopy(h)
        parents=deepcopy(owner.degenerate_diagnostics)
        def failed_trial(trial, grid, frequencies, solvers, modes, electric, magnetic, **kwargs):
            electric[0][0]=[a*7 for a in electric[0][0]]
            trial.degenerate_diagnostics[0]['orientation']='invalid'
            raise ValueError('Test rejection after a partial trial.')
        monkeypatch.setattr(symmetry,'_align_symmetry_subspaces',failed_trial)
        align_symmetry_subspaces(owner,grid,[19e9,20e9],bank,(7,8,9,10),e,h)
        for actual,expected in ((e,expected_e),(h,expected_h)):
            for arow,brow in zip(actual,expected):
                for amode,bmode in zip(arow,brow):
                    for a,b in zip(amode,bmode):
                        np.testing.assert_array_equal(a,b)
        assert owner.degenerate_diagnostics[0]['orientation']==parents[0]['orientation']
        assert owner.symmetry_diagnostics==[]
        record=owner.symmetry_detection['groups'][0]
        assert record['applied']=='power_whitening'
        assert [a['symmetry'] for a in record['attempts']]==['c4v','c2v']


def test_rejected_c4v_trial_can_use_certified_c2v(monkeypatch):
    import gprMax.eigenmode_symmetry as symmetry
    original=symmetry._align_symmetry_subspaces
    def reject_quarter_turn(owner,*args,**kwargs):
        if kwargs['symmetry']=='c4v':
            raise ValueError('Injected quarter-turn rejection.')
        return original(owner,*args,**kwargs)
    monkeypatch.setattr(symmetry,'_align_symmetry_subspaces',reject_quarter_turn)
    with _configured_solver_constants():
        solver=fd_solver(20e9,(12,12),.02/12,(2,2,2),count=12)
        owner,grid,e,h=aligned_bank([solver],(7,8,9,10))
        record=owner.symmetry_detection['groups'][0]
        assert record['applied']=='c2v'
        assert record['attempts'][0]['symmetry']=='c4v'
        assert [len(b) for b in owner.symmetry_diagnostics[0]['blocks']]==[1,1,1,1]
        power = _power(owner,grid,e[0],h[0])
        np.testing.assert_allclose(np.diag(power),np.ones(4),atol=1e-10)
        assert np.linalg.eigvalsh(power).min() > .9


@pytest.mark.parametrize('sides,detected,groups',[(6,'c2v',()),(8,'c4v',((1,2),))])
def test_polygon_detection_does_not_recombine_split_pairs(sides,detected,groups):
    from testing.eigenmode_support.symmetry.polygon_symmetry_breaking import polygon_solver
    from gprMax.eigenmode_tracking import track_solver_bank
    from gprMax.eigenmode_config import EigenmodeTrackingConfig
    with _configured_solver_constants():
        bank=[polygon_solver(sides,24,frequency=f*1e9) for f in (19.9,20,20.1)]
        owner=EigenmodeSource(None)
        owner.normal_axis=2; owner.normal='z'; owner.direction='+'; owner.transverse_axes=(0,1)
        owner.tracking='auto'; owner.verification='none'; owner.mode_polarizations={}
        owner.tracking_config=EigenmodeTrackingConfig(extra_candidates=0)
        owner.fallback_frequency=20e9; owner._tracking_impedance=config.sim_config.em_consts['z0']
        track_solver_bank(owner,[s.frequency for s in bank],bank,5)
        assert owner.degenerate==groups
        modes=tuple(range(1,owner._auto_tracking_mode_count+1))
        fields=[[owner._fields_from_solver_mode(s,m) for m in modes] for s in bank]
        e=[[v[0] for v in row] for row in fields]; h=[[v[1] for v in row] for row in fields]
        align_groups(owner,SimpleNamespace(dl=np.full(3,bank[0].du)),[s.frequency for s in bank],
                     bank,modes,e,h,np.ones((len(bank),len(modes)),bool))
        assert owner.symmetry_detection['detected']==detected
        assert tuple(r['modes'] for r in owner.symmetry_diagnostics)==groups


def test_default_detection_reaches_source_receiver_and_hdf5(tmp_path):
    import h5py
    import json
    import gprMax
    from testing.eigenmode_support.ports.degenerate_eigenmode_ports import circular_scene
    scene=circular_scene(normal_axis=2,steps=12,virtual=False,monitor=True,averaging='n')
    for obj in scene.grid_objects:
        if isinstance(obj,gprMax.EigenmodePort):
            obj.kwargs['mode_polarizations']=None
            obj.kwargs['tracking']='auto'
    path=tmp_path/'automatic_symmetry'
    gprMax.run(scenes=[scene],outputfile=path,cpu_precision='double',hide_progress_bars=True,log_level=40)
    with h5py.File(path.with_suffix('.h5')) as result:
        for port in ('port1','port2'):
            root=result[f'eigenmode_ports/{port}']
            assert root['symmetry_detection'].attrs['Detected']=='c4v'
            details=json.loads(root['symmetry_detection'].attrs['Details'])
            assert details['groups'][0]['applied']=='c4v'
            assert root['symmetry_subspaces/1'].attrs['Blocks']=='1,2'
            orientation=root['symmetry_subspaces/1/anchor0/orientation/1']
            assert orientation.attrs['Method']=='electric_moment'
            np.testing.assert_array_equal(orientation['parity'][...],[[-1,1],[1,-1]])
            assert orientation['parity_residual'][()]<1e-8
