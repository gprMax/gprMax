"""Native subgroup characters and translated PEC/SIBC port regressions."""
from copy import deepcopy
from dataclasses import replace

import h5py
import numpy as np
import pytest

from gprMax.eigenmode_symmetry import (
    detect_symmetry, field_actions, native_irrep_names, validate_material_symmetry,
)
from gprMax.eigenmode_symmetry_geometry import GROUPS, symmetry_bounds
from gprMax.eigenmode_tracking import write_diagnostics
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import FDFDSurfaceBoundary
from testing.eigenmode_support.subspaces.expanded_mode_tracking import fd_solver
from testing.eigenmode_support.identification.tracking_classification_edges import copper_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
from tests.test_eigenmode_auto_symmetry import sampled_port, aligned_bank


def pattern(key, n=10):
    x = np.linspace(-1, 1, n+1)[:, None]
    y = np.linspace(-1, 1, n+1)[None, :]
    return {
        'c4v': x*x+y*y,
        'c2v': x*x+.37*y*y,
        'c2v_diagonal': x*y+.13*(x*x+y*y),
        'c4': x*y*(x*x-y*y),
        'c2': x*y+.37*x**3*y,
        'cs_u': x*x+.37*y+.13*x*x*y,
        'cs_v': y*y+.37*x+.13*y*y*x,
        'cs_diagonal': x+y+.37*x*y,
        'cs_antidiagonal': x-y+.37*x*y,
        'c1': x+.37*y+.13*x*y,
    }[key]


def subgroup_port(key):
    port = sampled_port(10, 10)
    # A longitudinal tensor perturbation leaves the transverse tensors equal.
    port.eps_r_ww = 2 + .1*pattern(key)
    return port


def embed(array, padding, fill):
    (left, right), (bottom, top) = padding
    result = np.full((array.shape[0]+left+right, array.shape[1]+bottom+top, *array.shape[2:]),
                     fill, dtype=array.dtype)
    result[left:left+array.shape[0], bottom:bottom+array.shape[1]] = array
    return result


def padded_solver(core, padding=((2, 5), (3, 1))):
    """New independent solve, with exact physical constraints in the padding."""
    kwargs = {}
    for kind in ('eps_r', 'mu_r'):
        for axis in 'uvw':
            name = f'{kind}_{axis}{axis}'
            kwargs[name] = embed(getattr(core, name), padding, 1.)
    for kind in ('pec', 'pmc'):
        for axis in 'uvw':
            name = f'{kind}_{axis}_mask'
            kwargs[name] = embed(getattr(core, name), padding, kind == 'pec' and core.surface_boundary is None)
    if core.surface_boundary is not None:
        def shift(index):
            return tuple(i+p[0] for i, p in zip(index, padding))
        kwargs['surface_boundary'] = FDFDSurfaceBoundary.create(
            electric_retained=tuple(embed(a, padding, False) for a in core.surface_electric_retained),
            magnetic_retained=tuple(embed(a, padding, False) for a in core.surface_magnetic_retained),
            rows=tuple(replace(row, electric_index=shift(row.electric_index),
                magnetic_terms=tuple(replace(t, index=shift(t.index)) for t in row.magnetic_terms))
                for row in core.surface_boundary_rows))
    result = FDFD_2D_mode_solver(core.frequency, core.du, core.dv, core.num_modes-1, **kwargs)
    # The geometry is fixed in physical coordinates while its window moves.
    result.symmetry_origin = (-padding[0][0]*core.du, -padding[1][0]*core.dv)
    result.retain_tracking_operator = True
    result.calculate_diagnostics = True
    result.solve()
    return result


@pytest.mark.parametrize('key', GROUPS)
def test_detects_exact_subgroup_and_embedding(key):
    port = subgroup_port(key)
    result = detect_symmetry([port])
    assert result['embedding'] == key
    assert result['detected'] == key.split('_')[0]
    validate_material_symmetry(port, key)
    if key == 'c1':
        assert result['candidates'] == []


@pytest.mark.parametrize('key,expected', [('cs_diagonal', 'c1'), ('c4', 'c2'),
                                        ('c2v_diagonal', 'c2'), ('cs_u', 'cs_u')])
def test_unequal_spacing_rejects_axis_exchange_only(key, expected):
    port = subgroup_port(key)
    port.dv = 2.
    assert detect_symmetry([port])['embedding'] == expected


def test_common_mirror_is_a_positioned_line_not_a_group_name():
    first, second = subgroup_port('cs_u'), subgroup_port('cs_u')
    second.symmetry_origin = (0., 7.)  # Translation along the mirror is immaterial.
    assert detect_symmetry([first, second])['embedding'] == 'cs_u'
    second.symmetry_origin = (.5, 7.)
    result = detect_symmetry([first, second])
    assert result['detected'] == 'c1'
    assert 'mirror lines' in result['bank_reasons']['cs_u']
    assert detect_symmetry([subgroup_port('cs_u'), subgroup_port('cs_v')])['detected'] == 'c1'


def test_frequency_dependent_loss_reduces_to_common_subgroup():
    first, second = subgroup_port('c4v'), subgroup_port('cs_u')
    second.eps_r_ww = first.eps_r_ww + .01j*pattern('cs_u')
    assert detect_symmetry([first, second])['embedding'] == 'cs_u'


@pytest.mark.parametrize('shape,expected', [((10, 10), 'c4v'), ((9, 9), 'c4v'), ((10, 8), 'c2v')])
def test_off_centre_pec_solve_preserves_spectrum_and_symmetry(shape, expected):
    with _configured_solver_constants():
        core = fd_solver(20e9, shape, .002, (2, 2, 2), count=6)
        padded = padded_solver(core)
        result = detect_symmetry([padded])
        assert result['embedding'] == expected
        anchor = result['anchors'][0]
        assert anchor['bounds'] == ((2, 2+shape[0]), (3, 3+shape[1]))
        assert anchor['centre_cells'] == [2+shape[0]/2, 3+shape[1]/2]
        assert anchor['centre_cells'] != anchor['window_centre_cells']
        assert detect_symmetry([core, padded])['embedding'] == expected
        np.testing.assert_allclose(padded.complex_neff, core.complex_neff, atol=1e-10)
        assert np.max(padded.tracking_field_residuals(np.arange(6))) < 1e-10
        # Certify the transformed fields against the complete padded Maxwell
        # operator, independently of the material detector and modal aligner.
        for action in field_actions(expected).values():
            transformed = deepcopy(padded)
            e, h = action(padded, tuple(getattr(padded, 'E'+a) for a in 'uvw'),
                          tuple(getattr(padded, 'H'+a) for a in 'uvw'))
            for prefix, family in (('E', e), ('H', h)):
                for axis, array in zip('uvw', family):
                    setattr(transformed, prefix+axis, array)
            assert np.max(transformed.tracking_field_residuals(np.arange(6))) < 1e-10


def test_off_centre_square_tracks_doublet_across_anchors_and_writes_centre(tmp_path):
    with _configured_solver_constants():
        bank = [padded_solver(fd_solver(f*1e9, (10, 10), .002, (2, 2, 2), count=6)) for f in (19, 20, 21)]
        owner, _, _, _ = aligned_bank(bank, (1, 2))
    assert owner.symmetry_detection['groups'][0]['applied'] == 'c4v'
    record = owner.symmetry_diagnostics[0]
    assert record['irreps'] == ('E',)
    assert all(row['closure'] < 1e-10 and row['field_residual'] < 1e-10 for row in record['anchors'].values())
    with h5py.File(tmp_path/'symmetry.h5', 'w') as output:
        write_diagnostics(output, owner)
        import json
        details = json.loads(output['symmetry_detection'].attrs['Details'])
        assert details['anchors'][0]['centre_cells'] == [7., 8.]
        assert details['embedding'] == 'c4v'


@pytest.mark.parametrize('shape,expected', [((10, 10), 'c4v'), ((10, 8), 'c2v')])
def test_off_centre_sibc_retains_rows_and_spectrum(shape, expected):
    with _configured_solver_constants():
        core = copper_solver(42e9, shape, count=6)
        padded = padded_solver(core)
        assert detect_symmetry([padded])['embedding'] == expected
        np.testing.assert_allclose(padded.complex_neff, core.complex_neff, atol=1e-9)
        assert np.max(padded.tracking_field_residuals(np.arange(6))) < 1e-9
        if expected == 'c4v':
            owner, _, _, _ = aligned_bank([padded], (1, 2))
            assert owner.symmetry_detection['groups'][0]['applied'] == 'c4v'
            assert owner.symmetry_diagnostics[0]['anchors'][0]['field_residual'] < 1e-8
    broken = deepcopy(padded)
    rows = list(broken.surface_boundary_rows)
    rows[0] = replace(rows[0], relative_permittivity=1.03*rows[0].relative_permittivity)
    broken.surface_boundary_rows = tuple(rows)
    assert detect_symmetry([broken])['detected'] == 'c1'


@pytest.mark.parametrize('key', ['cs_u', 'cs_v', 'cs_diagonal', 'cs_antidiagonal', 'c2', 'c4', 'c2v_diagonal'])
@pytest.mark.parametrize('off_centre', [False, True])
def test_new_groups_align_physical_modes_with_correct_characters(key, off_centre):
    with _configured_solver_constants():
        bank = []
        for frequency in (19e9, 20e9):
            port = subgroup_port(key)
            port.eps_r_uu[:] = 2.
            port.eps_r_vv[:] = 2.
            port.pec_u_mask[:, [0, -1]] = True
            port.pec_v_mask[[0, -1], :] = True
            port.pec_w_mask[:, [0, -1]] = True
            port.pec_w_mask[[0, -1], :] = True
            kwargs = {name: value for name, value in vars(port).items()
                      if name.startswith(('eps_', 'mu_', 'pec_', 'pmc_'))}
            solver = FDFD_2D_mode_solver(frequency, .002, .002, 5, **kwargs)
            solver.retain_tracking_operator = True
            solver.calculate_diagnostics = True
            solver.solve()
            solver = padded_solver(solver) if off_centre else solver
            for action in field_actions(key).values():
                transformed = deepcopy(solver)
                e, h = action(solver, tuple(getattr(solver, 'E'+a) for a in 'uvw'),
                              tuple(getattr(solver, 'H'+a) for a in 'uvw'))
                for prefix, family in (('E', e), ('H', h)):
                    for axis, array in zip('uvw', family):
                        setattr(transformed, prefix+axis, array)
                assert np.max(transformed.tracking_field_residuals(np.arange(6))) < 1e-10
            bank.append(solver)
        owner, _, _, _ = aligned_bank(bank, (1, 2))
    assert owner.symmetry_detection['embedding'] == key
    assert owner.symmetry_detection['groups'][0]['applied'] == key
    record = owner.symmetry_diagnostics[0]
    expected = ({"A'", "A''"} if key.startswith('cs_') else {'E+', 'E-'} if key == 'c4'
                else {'B1', 'B2'} if key == 'c2v_diagonal' else {'B'})
    assert set(record['irreps']) == expected
    assert [len(b) for b in record['blocks']] == [1, 1]
    for row in record['anchors'].values():
        assert row['closure'] < 1e-10
        assert row['field_residual'] < 1e-10
        assert np.max(row['residual']) < 1e-10
        np.testing.assert_allclose(row['power'], np.eye(2), atol=1e-10)


def test_asymmetric_active_air_padding_is_not_discarded():
    port = sampled_port(14, 12)
    # Square dielectric core centred at (5, 5), inside an active air window
    # centred at (7, 6). It has no exact mirror of the complete operator.
    port.eps_r_ww[3:8, 3:8] = 2.
    assert symmetry_bounds(port) == ((0, 14), (0, 12))
    assert detect_symmetry([port])['detected'] == 'c1'


def test_diagonal_vector_action_has_axial_magnetic_sign():
    port = sampled_port(3, 3)
    e = tuple(np.arange(np.prod(a.shape)).reshape(a.shape)+10*i for i, a in enumerate(
        (port.eps_r_uu, port.eps_r_vv, port.eps_r_ww)))
    h = tuple(np.arange(np.prod(a.shape)).reshape(a.shape)+30*i for i, a in enumerate(
        (port.mu_r_uu, port.mu_r_vv, port.mu_r_ww)))
    action = field_actions('cs_diagonal')['mirror_diagonal']
    te, th = action(port, e, h)
    for actual, expected in zip(te, (e[1].T, e[0].T, e[2].T)):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(th, (-h[1].T, -h[0].T, -h[2].T)):
        np.testing.assert_array_equal(actual, expected)
    ee, hh = action(port, te, th)
    for original, restored in zip((*e, *h), (*ee, *hh)):
        np.testing.assert_array_equal(original, restored)


def test_bad_cyclic_representation_cannot_receive_an_irrep_label():
    with pytest.raises(ValueError, match='cyclic group'):
        native_irrep_names('c4', [np.array([[np.exp(.2j)]])], [[0]])


@pytest.mark.parametrize('attribute,value', [('du', 0.), ('dv', np.nan),
                                            ('symmetry_origin', (np.inf, 0.)),
                                            ('symmetry_origin', (1.,))])
def test_invalid_geometry_metadata_is_not_certified(attribute, value):
    port = sampled_port()
    setattr(port, attribute, value)
    result = detect_symmetry([port])
    assert result['detected'] == 'none'
    assert result['candidates'] == []
    assert result['anchors'][0]['reasons']['c4v']
