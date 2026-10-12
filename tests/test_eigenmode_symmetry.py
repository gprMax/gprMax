"""Irreducible decomposition must support arbitrary multiplicities."""
import numpy as np
import pytest
from scipy.linalg import block_diag

from gprMax.eigenmode_symmetry import irreducible_blocks, commutant


@pytest.mark.parametrize('copies', [1, 2, 3, 4, 6, 10])
def test_arbitrary_number_of_equivalent_doublets(copies):
    rng = np.random.default_rng(27)
    n = 2 * copies
    u = np.linalg.qr(rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)))[0]
    rotation = np.kron(np.eye(copies), [[0,-1],[1,0]])
    mirror = np.kron(np.eye(copies), [[1,0],[0,-1]])
    actions = [u.conj().T @ d @ u for d in (rotation, mirror)]
    blocks, error = irreducible_blocks(actions)
    assert [b.shape[1] for b in blocks] == [2] * copies
    assert error < 1e-12
    for b in blocks:
        assert commutant([b.conj().T @ d @ b for d in actions]).shape[1] == 1


def test_singlets_are_not_forced_into_doublets():
    r = block_diag([[1]], [[-1]], [[0,-1],[1,0]], [[0,-1],[1,0]])
    s = block_diag([[1]], [[1]], [[1,0],[0,-1]], [[1,0],[0,-1]])
    blocks, _ = irreducible_blocks([r,s])
    assert sorted(b.shape[1] for b in blocks) == [1,1,2,2]


def test_neighbor_probe_recovers_specific_equivalent_copies():
    rng = np.random.default_rng(8)
    u = np.linalg.qr(rng.normal(size=(6,6))+1j*rng.normal(size=(6,6)))[0]
    r = np.kron(np.eye(3), [[0,-1],[1,0]])
    s = np.kron(np.eye(3), [[1,0],[0,-1]])
    probe = u.conj().T @ np.diag([1,1,2,2,3,3]) @ u
    blocks, _ = irreducible_blocks([u.conj().T @ d @ u for d in (r,s)], probe=probe)
    for k,b in enumerate(blocks):
        expected = u.conj().T[:,2*k:2*k+2]
        np.testing.assert_allclose(b @ b.conj().T, expected @ expected.conj().T, atol=1e-12)


def test_nonunitary_action_is_rejected():
    with pytest.raises(ValueError, match='unitary'):
        irreducible_blocks([np.diag([1,2])])


def test_physical_quartet_splits_only_with_supplied_symmetry(tmp_path, monkeypatch):
    from testing.validation import general_subspace_tracking as validation
    from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
    monkeypatch.setattr(validation, 'plot_case', lambda *a: None)
    with _configured_solver_constants():
        for symmetry, sizes in ((None,(4,)), ('c4v',(2,2))):
            bank = [validation.fd_solver(f*1e9,(12,12),.02/12,(2,2,2),count=12) for f in (19,20,21)]
            report = validation.run_case('c4v_metallic_quartet',np.array([19,20,21]),bank,10,
                                         [7,8,9,10],tmp_path,'GHz',expected_sizes=sizes,symmetry=symmetry)
            assert sorted(len(g['labels']) for g in report['groups']) == list(sizes)
            assert report['minimum_full_power_eigenvalue'] > .9
            if symmetry is None:
                assert report['maximum_full_power_identity_error'] < 1e-10


def test_supplied_action_must_preserve_the_actual_fields(tmp_path, monkeypatch):
    from testing.validation import general_subspace_tracking as validation
    from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
    monkeypatch.setattr(validation, 'plot_case', lambda *a: None)
    def invalid(solver, e, h):
        return (e[0]*2,e[1],e[2]),h
    with _configured_solver_constants():
        bank = [validation.fd_solver(f*1e9,(12,12),.02/12,(2,2,2),count=12) for f in (19,20,21)]
        with pytest.raises(ValueError, match='preserve|unitary'):
            validation.run_case('c4v_metallic_quartet',np.array([19,20,21]),bank,10,
                                [7,8,9,10],tmp_path,'GHz',symmetry={'invalid':invalid})


@pytest.mark.parametrize('value', ['auto', 'c4v', 'c2v', 'none'])
def test_hash_port_rejects_removed_symmetry_argument(value):
    from collections import defaultdict
    from gprMax.hash_cmds_multiuse import process_multicmds
    from gprMax.user_objects.cmds_multiuse import EigenmodePort
    commands = defaultdict(lambda: None)
    commands['#eigenmode_band'] = ['band 18e9 22e9 3']
    commands['#eigenmode_excitation'] = ['1 1 auto n']
    commands['#eigenmode_port'] = [f'1 0 0 0.02 0.02 0.02 0.02 + 1,2 auto tracking=auto mode_symmetry={value}']
    with pytest.raises(ValueError, match='mode_symmetry'):
        process_multicmds(commands)


@pytest.mark.parametrize('value', [None, 'auto', 'c4v', 'c2v', {}])
def test_python_port_rejects_removed_symmetry_argument(value):
    from gprMax.user_objects.cmds_multiuse import EigenmodePort
    with pytest.raises((ValueError, TypeError), match='mode_symmetry'):
        EigenmodePort(port=1, tracking='auto', mode_symmetry=value)


@pytest.mark.parametrize('axis', [0,1,2])
def test_automatic_symmetry_reaches_source_receiver_and_hdf5(tmp_path, axis):
    import h5py
    import gprMax
    from testing.validation.degenerate_eigenmode_ports import circular_scene
    scene = circular_scene(normal_axis=axis, steps=12, virtual=False, monitor=True, averaging='n')
    for obj in scene.grid_objects:
        if isinstance(obj, gprMax.EigenmodePort):
            obj.kwargs['mode_polarizations'] = None
            obj.kwargs['tracking'] = 'auto'
    path = tmp_path / f'symmetry_axis{axis}'
    gprMax.run(scenes=[scene], outputfile=path, cpu_precision='double',
               hide_progress_bars=True, log_level=40)
    with h5py.File(path.with_suffix('.h5')) as result:
        for port in ('port1','port2'):
            group = result[f'eigenmode_ports/{port}/symmetry_subspaces/1']
            assert group.attrs['Blocks'] == '1,2'
            assert group['anchor0/closure'][()] < 1e-8
            assert max(group['anchor0/residual']) < 1e-9
            np.testing.assert_allclose(group['anchor0/power'],np.eye(2),atol=1e-10)
            assert group['anchor0/longitudinal'].attrs['Reason']
            assert np.all(np.isfinite(group['anchor0/longitudinal/scores']))
