"""Certify lossy boundary rows and orient copper partners without templates."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from gprMax.eigenmode_symmetry import detect_symmetry, validate_material_symmetry
from testing.eigenmode_support.identification.tracking_classification_edges import copper_solver, rectangular_templates
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
from tests.test_eigenmode_auto_symmetry import aligned_bank


@pytest.fixture(scope='module')
def copper():
    with _configured_solver_constants():
        return copper_solver(42e9, 24, 28)


def test_copper_rows_preserve_c4v(copper):
    assert detect_symmetry([copper])['detected'] == 'c4v'


def test_rectangular_copper_rows_preserve_c2v():
    with _configured_solver_constants():
        solver = copper_solver(42e9, (24, 18), 20)
    assert detect_symmetry([solver])['detected'] == 'c2v'


@pytest.mark.parametrize('change', ['circulation', 'area', 'admittance', 'missing_row', 'retained_e', 'retained_h'])
def test_asymmetric_surface_changes_fail_certification(copper, change):
    solver = deepcopy(copper)
    rows = list(solver.surface_boundary_rows)
    row = rows[0]
    if change == 'circulation':
        rows[0] = replace(row, magnetic_terms=(replace(row.magnetic_terms[0], line_weight=-row.magnetic_terms[0].line_weight),))
    elif change == 'area':
        rows[0] = replace(row, retained_dual_area=1.01*row.retained_dual_area)
    elif change == 'admittance':
        rows[0] = replace(row, relative_permittivity=1.01*row.relative_permittivity)
    elif change == 'missing_row':
        rows.pop(0)
    else:
        masks = solver.surface_electric_retained if change == 'retained_e' else solver.surface_magnetic_retained
        masks[0][2, 3] = False
    solver.surface_boundary_rows = tuple(rows)
    assert detect_symmetry([solver])['detected'] == 'none'
    with pytest.raises(ValueError, match='SIBC'):
        validate_material_symmetry(solver, 'c4v')


def test_opposite_wall_admittances_can_reduce_c4v_to_c2v(copper):
    solver = deepcopy(copper)
    solver.surface_boundary_rows = tuple(
        replace(r, relative_permittivity=1.01*r.relative_permittivity) if r.electric_axis == 0 else r
        for r in solver.surface_boundary_rows)
    assert detect_symmetry([solver])['detected'] == 'c2v'


@pytest.mark.parametrize('symmetry', ['auto', 'c2v'])
@pytest.mark.parametrize('labels', [(5,6), (15,16), (17,18), (23,24)])
def test_copper_equal_parity_partners_are_individual_patterns(copper, labels, symmetry):
    with _configured_solver_constants():
        owner, _, e, h = aligned_bank([deepcopy(copper)], labels, symmetry=symmetry)
        scores = [rectangular_templates(ee, hh)[0] for ee, hh in zip(e[0], h[0])]
    if symmetry == 'auto':
        assert owner.symmetry_detection['groups'][0]['applied'] == 'c4v'
    assert min(s['score'] for s in scores) > .999
    assert scores[0]['name'] != scores[1]['name']
    row = next(iter(owner.symmetry_diagnostics[0]['anchors'].values()))
    assert any(r['applied'] for r in row['axis_sensitive'])
    assert row['field_residual'] < 1e-7
    assert np.max(row['residual']) < 1e-7


def test_loss_scaled_axis_keeps_distinct_longitudinal_families():
    from gprMax.eigenmode_symmetry import axis_sensitive_blocks
    x = (np.arange(16)+.5)/8-1
    fields = np.stack(np.broadcast_arrays(np.cos(np.pi*x)[:,None], np.cos(np.pi*x)[None,:]), axis=-1)
    blocks = (np.eye(2)[:,:1], np.eye(2)[:,1:])
    result, records = axis_sensitive_blocks(
        blocks, fields*np.sqrt([.2,.8]), fields*np.sqrt([.8,.2]),
        (np.eye(2), np.eye(2)), (1.,1.), 100., family_tolerance=1e-3)
    assert records == []
    np.testing.assert_array_equal(np.column_stack(result), np.eye(2))
