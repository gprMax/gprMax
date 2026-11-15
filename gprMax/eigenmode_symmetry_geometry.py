"""Exact transverse Yee symmetries and the electrically active port extent.

All coordinates are local to the port window. Bounds are cell-edge indices;
their midpoint can therefore lie on a node or halfway between nodes. Cropping
never changes a solve: it defines the support of a signed field permutation.
"""

from copy import copy
from dataclasses import replace
from functools import partial

import numpy as np


# Signed permutations fixing the propagation axis. The order keeps the legacy
# C4v -> axis-C2v fallback before trying the other (incomparable) subgroups.
OPERATIONS = {
    'identity': np.eye(3, dtype=int),
    'rotation_90': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    'rotation_180': np.diag([-1, -1, 1]),
    'mirror_u': np.diag([-1, 1, 1]),
    'mirror_v': np.diag([1, -1, 1]),
    'mirror_diagonal': np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    'mirror_antidiagonal': np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
}
GROUPS = {
    'c4v': ('rotation_90', 'mirror_u'),
    'c2v': ('mirror_u', 'mirror_v'),
    'c2v_diagonal': ('mirror_diagonal', 'mirror_antidiagonal'),
    'c4': ('rotation_90',),
    'c2': ('rotation_180',),
    'cs_u': ('mirror_u',),
    'cs_v': ('mirror_v',),
    'cs_diagonal': ('mirror_diagonal',),
    'cs_antidiagonal': ('mirror_antidiagonal',),
    'c1': ('identity',),
}
MATERIAL_FAMILIES = (
    ('eps_r_uu', 'eps_r_vv', 'eps_r_ww'),
    ('mu_r_uu', 'mu_r_vv', 'mu_r_ww'),
    ('pec_u_mask', 'pec_v_mask', 'pec_w_mask'),
    ('pmc_u_mask', 'pmc_v_mask', 'pmc_w_mask'),
)
OFFSETS = (((.5, 0), (0, .5), (0, 0)), ((0, .5), (.5, 0), (.5, .5)))


def group_name(key):
    return key.split('_')[0]


def component_slices(shape, cells, bounds):
    """Slice each staggered component at the same physical cell edges."""
    return tuple(slice(lo, hi + size - count)
                 for size, count, (lo, hi) in zip(shape, cells, bounds))


def symmetry_bounds(solver):
    """Remove only PEC-clamped exterior or unretained SIBC samples.

    In the PEC case every electric sample outside the box is constrained, and
    its moved faces have physical tangential PEC constraints. Faraday's curl
    then also vanishes outside the box; we do not infer H support from a PEC
    E mask alone. PMC/mixed enclosures conservatively keep the full window.
    SIBC bounds include *both* retained families and every installed row term.
    """
    full = ((0, solver.Nu), (0, solver.Nv))
    if getattr(solver, 'surface_boundary', None) is not None:
        families = (solver.surface_electric_retained, solver.surface_magnetic_retained)
    else:
        families = (tuple(~np.asarray(getattr(solver, f'pec_{a}_mask'), dtype=bool)
                          for a in 'uvw'),)
    lower, upper = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
    for family, offsets in zip(families, OFFSETS):
        for mask, offset in zip(family, offsets):
            indices = np.nonzero(mask)
            if len(indices[0]):
                lower = np.minimum(lower, [i.min() + o for i, o in zip(indices, offset)])
                upper = np.maximum(upper, [i.max() + o for i, o in zip(indices, offset)])
    if not np.all(np.isfinite(lower)):
        return full
    bounds = tuple((int(np.floor(lo)), int(np.ceil(hi))) for lo, hi in zip(lower, upper))
    if any(lo < 0 or hi > n or lo >= hi for (lo, hi), n in zip(bounds, (solver.Nu, solver.Nv))):
        return full
    if bounds == full:
        return full
    if getattr(solver, 'surface_boundary', None) is not None:
        # A retained row can refer to a sample outside the retained masks.
        # Do not silently lose that part of the boundary operator.
        for row in solver.surface_boundary_rows:
            addresses = [(0, row.electric_axis, row.electric_index)] + [
                (1, t.axis, t.index) for t in row.magnetic_terms]
            for family, axis, index in addresses:
                point = np.asarray(index) + OFFSETS[family][axis]
                if any(x < lo or x > hi for x, (lo, hi) in zip(point, bounds)):
                    return full
        return bounds
    (u0, u1), (v0, v1) = bounds
    masks = [np.asarray(getattr(solver, 'physical_pec_' + a + '_mask',
                                getattr(solver, 'pec_' + a + '_mask'))) for a in 'uvw']
    eu, ev, ew = masks
    walls = (np.all(ev[u0, v0:v1]) and np.all(ew[u0, v0:v1+1]),
             np.all(ev[u1, v0:v1]) and np.all(ew[u1, v0:v1+1]),
             np.all(eu[u0:u1, v0]) and np.all(ew[u0:u1+1, v0]),
             np.all(eu[u0:u1, v1]) and np.all(ew[u0:u1+1, v1]))
    moved = (u0 != 0, u1 != solver.Nu, v0 != 0, v1 != solver.Nv)
    return bounds if all(wall or not move for wall, move in zip(walls, moved)) else full


def symmetry_view(solver):
    """A shallow, cropped validation view; never mutate the solver arrays."""
    bounds = symmetry_bounds(solver)
    cells = (solver.Nu, solver.Nv)
    view = copy(solver)
    view.Nu, view.Nv = (hi - lo for lo, hi in bounds)
    def crop(array):
        return np.asarray(array)[component_slices(array.shape[:2], cells, bounds)]
    for names in MATERIAL_FAMILIES:
        for name in names:
            setattr(view, name, crop(getattr(solver, name)))
    if getattr(solver, 'surface_boundary', None) is not None:
        for family in ('electric', 'magnetic'):
            name = 'surface_' + family + '_retained'
            setattr(view, name, tuple(crop(a) for a in getattr(solver, name)))
        for name, values in zip(('e', 'h'), (view.surface_electric_retained, view.surface_magnetic_retained)):
            for axis, array in zip('uvw', values):
                setattr(view, f'shape_{name}{axis}', array.shape)
        shift = lambda index: tuple(int(i - lo) for i, (lo, _) in zip(index, bounds))
        view.surface_boundary_rows = tuple(replace(row, electric_index=shift(row.electric_index),
            magnetic_terms=tuple(replace(t, index=shift(t.index)) for t in row.magnetic_terms))
            for row in solver.surface_boundary_rows)
    return view, bounds


def transform_family(values, operation, *, axial=False, signed=True):
    """Transform vectors or diagonal tensors on a centred rectangular extent."""
    matrix = OPERATIONS[operation]
    permutation = np.argmax(abs(matrix), axis=1)
    determinant = int(round(np.linalg.det(matrix))) if axial else 1
    result = []
    for axis, source in enumerate(permutation):
        array = np.asarray(values[source])
        if permutation[0] == 1:
            array = array.swapaxes(0, 1)
        for coordinate in (0, 1):
            if matrix[coordinate, permutation[coordinate]] < 0:
                array = np.flip(array, axis=coordinate)
        if array.shape != np.shape(values[axis]):
            raise ValueError('Symmetry does not map the staggered Yee samples onto themselves.')
        result.append(array * (determinant * matrix[axis, source]) if signed else array)
    return tuple(result)


def check_spacing(solver, operation):
    if OPERATIONS[operation][0, 1] and not np.isclose(solver.du, solver.dv, rtol=1e-12, atol=0):
        raise ValueError('Axis-exchanging symmetry requires equal transverse spacing.')


def field_action(solver, electric, magnetic, *, operation):
    check_spacing(solver, operation)
    bounds = symmetry_bounds(solver)
    cells = (solver.Nu, solver.Nv)
    result = []
    for axial, family in enumerate((electric, magnetic)):
        slices = [component_slices(a.shape[:2], cells, bounds) for a in family]
        transformed = transform_family([a[s] for a, s in zip(family, slices)], operation, axial=bool(axial))
        output = [np.zeros_like(a) for a in family]
        for array, section, value in zip(output, slices, transformed):
            array[section] = value
        result.append(tuple(output))
    return tuple(result)


def native_field_actions(key):
    return {op: partial(field_action, operation=op) for op in GROUPS[key]}


def positioned_operations(solver, key, bounds):
    """Affine translations in metres distinguish mirror lines across anchors."""
    centre = np.array([sum(b) * d / 2 for b, d in zip(bounds, (solver.du, solver.dv))])
    centre += np.asarray(getattr(solver, 'symmetry_origin', (0., 0.)))
    return {op: ((np.eye(2) - OPERATIONS[op][:2, :2]) @ centre).tolist() for op in GROUPS[key]}
