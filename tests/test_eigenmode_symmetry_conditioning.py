"""Selected symmetry banks must remain usable by the coefficient monitor."""
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
import gprMax.eigenmode_symmetry as symmetry
from gprMax.eigenmode_ports import EigenmodePortMonitor
from tests.test_eigenmode_auto_symmetry import aligned_bank
from testing.eigenmode_support.subspaces.expanded_mode_tracking import fd_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


def unequal_longitudinal_basis():
    x = (np.arange(32) + .5) / 16 - 1
    patterns = np.stack(np.broadcast_arrays(np.cos(np.pi*x)[:, None],
                                           np.cos(np.pi*x)[None, :]), axis=-1)
    patterns /= np.linalg.norm(patterns, axis=(0, 1))
    rotation = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    fields = (patterns @ rotation) * np.sqrt([1, 1e5])
    blocks, records = symmetry.axis_sensitive_blocks(
        (np.eye(2)[:, :1], np.eye(2)[:, 1:]), fields, np.zeros_like(fields),
        (np.eye(2), np.eye(2)), (.0625, .0625), 1e5)
    assert records[0]['applied']
    basis = np.column_stack(blocks)
    assert np.linalg.cond(basis.conj().T @ basis) == pytest.approx(1e5)
    return blocks, records


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_axis_selection_falls_back_when_monitor_precision_requires_it(monkeypatch, dtype):
    blocks, records = unequal_longitudinal_basis()
    # Exercise the real transactional selection with a directional result from
    # unequal longitudinal norms, isolating it from a particular guide geometry.
    monkeypatch.setattr(symmetry, 'axis_sensitive_blocks', lambda *a, **kw: (blocks, records))
    with _configured_solver_constants():
        config.sim_config.dtypes = {'complex': dtype}
        solver = fd_solver(20e9, (16,16), .02/16, (2,2,2), count=8)
        owner, grid, e, h = aligned_bank([solver], (5,6))
        selected = owner.symmetry_detection['groups'][0]
        if dtype == np.complex64:
            assert selected['applied'] == 'longitudinal_gradient'
            assert all('precision budget' in trial['reason'] for trial in selected['attempts'])
            assert owner.symmetry_diagnostics == []
            # Failed symmetry must retain the independently validated separation.
            with monkeypatch.context() as context:
                context.setattr(symmetry, 'detect_symmetry', lambda solvers: dict(
                    detected='none', candidates=[], anchors=[], groups=[]))
                _, _, fallback_e, fallback_h = aligned_bank([solver], (5,6))
            for actual, expected in ((e, fallback_e), (h, fallback_h)):
                for a, b in zip(actual[0], expected[0]):
                    for ac, bc in zip(a, b):
                        np.testing.assert_array_equal(ac, bc)
            with pytest.raises(ValueError, match='precision budget'):
                aligned_bank([solver], (5,6), symmetry='c4v')
        else:
            assert selected['applied'] == 'c4v'

        gram = np.array([[owner._modal_cross_power(ej, hi, grid) for ej in e[0]]
                         for hi in h[0]])
        powers = np.real(np.diag(gram))
        gram /= np.sqrt(powers[:, None] * powers[None, :])
        incident = np.array([[1., .2j], [.7j, 1.]])
        scattering = np.array([[.12, .65j], [.65j, -.08]])
        outgoing = scattering @ incident
        monitor = EigenmodePortMonitor.__new__(EigenmodePortMonitor)
        monitor.electric_gram = gram.astype(dtype)[None]
        monitor.magnetic_gram = gram.conj().T.astype(dtype)[None]
        monitor.frequency = np.array([20e9])
        monitor.beta = np.zeros((1,2))
        monitor.power_basis_valid = np.ones((2,1), dtype=bool)
        monitor.reference_basis_valid = monitor.power_basis_valid.copy()
        monitor.power_matrix_valid = np.ones(1, dtype=bool)
        monitor.owner = owner
        monitor.magnetic_side = -1
        results = []
        for drive in range(2):
            monitor.electric_dft = (gram @ (incident[:,drive] + outgoing[:,drive])).astype(dtype)[None]
            monitor.magnetic_dft = (gram.conj().T @ (incident[:,drive] - outgoing[:,drive])).astype(dtype)[None]
            result = monitor.finalise(SimpleNamespace(dl=np.zeros(3)))
            assert result.coefficient_valid.all()
            assert result.power_wave_valid.all()
            results.append(result)
        measured_incident = np.column_stack([r.incident[:,0] for r in results])
        measured_outgoing = np.column_stack([r.outgoing[:,0] for r in results])
        np.testing.assert_allclose(measured_incident, incident, atol=1e-6)
        np.testing.assert_allclose(measured_outgoing, outgoing, atol=1e-6)
        measured_scattering = np.linalg.solve(measured_incident.T, measured_outgoing.T).T
        np.testing.assert_allclose(measured_scattering, scattering, atol=1e-6)


def test_measurement_check_includes_modes_outside_selected_group(monkeypatch):
    monkeypatch.setattr(config, 'sim_config', SimpleNamespace(dtypes={'complex': np.complex64}))
    owner = SimpleNamespace(_modal_cross_power=lambda e, h, grid: np.vdot(h, e))
    # Each singleton has unit power, but the complete bank is unsafe.
    fields = [np.array([1., 0.]), np.array([1., 1e-3])]
    for field in fields:
        symmetry._validate_measurement_basis(owner, None, [field], [field])
    with pytest.raises(ValueError, match='precision budget'):
        symmetry._validate_measurement_basis(owner, None, fields, fields)


def test_measurement_check_uses_unsymmetrized_gram(monkeypatch):
    monkeypatch.setattr(config, 'sim_config', SimpleNamespace(dtypes={'complex': np.complex64}))
    gram = np.diag([1 + 1e5j, 1])
    np.testing.assert_array_equal((gram + gram.conj().T) / 2, np.eye(2))
    owner = SimpleNamespace(_modal_cross_power=lambda e, h, grid: gram[h, e])
    with pytest.raises(ValueError, match='precision budget'):
        symmetry._validate_measurement_basis(owner, None, [0, 1], [0, 1])
