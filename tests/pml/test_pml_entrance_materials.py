"""Boundary PML calibration uses the entrance-side voxel on either face.

Normal material changes here deliberately distinguish the candidate planes;
they are indexing regressions, not recommended absorbing-layer geometries.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.materials import Material, create_built_in_materials
from gprMax.pml import PML

pytestmark = pytest.mark.unit


def _grid(make_pml_grid):
    grid = make_pml_grid(nx=12, ny=10, nz=8, dl=(0.001, 0.002, 0.003))
    grid.materials = []
    create_built_in_materials(grid)
    host = Material(3, "host")
    host.er, host.mr = 15.0, 2.0
    grid.materials.append(host)
    grid.solid.fill(2)
    return grid


def _entrance(grid, face, thickness):
    axis = "xyz".index(face[0])
    index = thickness - 1 if face.endswith("0") else grid.size[axis] - thickness
    plane = [slice(None)] * 3
    plane[axis] = index
    return tuple(plane)


@pytest.mark.parametrize("face", PML.boundaryIDs)
@pytest.mark.parametrize("thickness", (1, 3))
def test_boundary_average_uses_entrance_voxel(make_pml_grid, face, thickness):
    grid = _grid(make_pml_grid)
    grid.solid[_entrance(grid, face, thickness)] = 3
    slab = grid._construct_pml(face, thickness)

    assert grid._calculate_average_pml_material_properties(slab) == (15.0, 2.0)


@pytest.mark.parametrize("axis", range(3))
def test_mirroring_geometry_preserves_automatic_sigma(make_pml_grid, axis):
    grid = _grid(make_pml_grid)
    minus, plus = "xyz"[axis] + "0", "xyz"[axis] + "max"
    plane = grid.solid[_entrance(grid, minus, 3)]
    plane[::2, :] = 3
    first = grid._construct_pml(minus, 3)
    first_average = grid._calculate_average_pml_material_properties(first)
    first.calculate_update_coeffs(*first_average)

    grid.solid = np.flip(grid.solid, axis=axis).copy()
    second = grid._construct_pml(plus, 3)
    second_average = grid._calculate_average_pml_material_properties(second)
    second.calculate_update_coeffs(*second_average)

    assert second_average == first_average
    assert second.CFS[0].sigma.max == first.CFS[0].sigma.max


@pytest.mark.parametrize("face", PML.boundaryIDs)
@pytest.mark.parametrize("conductors", (False, True))
def test_normal_invariant_cross_section_is_unchanged(make_pml_grid, face, conductors):
    grid = _grid(make_pml_grid)
    axis = "xyz".index(face[0])
    plane = np.take(grid.solid, 0, axis=axis)
    plane.fill(3)
    if conductors:
        # Keep existing PEC/PMC weighting: these built-ins contribute er=mr=1.
        plane[::2, ::2] = 0
        plane[1::2, ::2] = 1
    grid.solid = np.repeat(np.expand_dims(plane, axis), grid.size[axis], axis=axis)
    slab = grid._construct_pml(face, 3)
    lookup_er = np.array([material.er for material in grid.materials])
    lookup_mr = np.array([material.mr for material in grid.materials])
    expected = (lookup_er[plane].mean(), lookup_mr[plane].mean())

    assert grid._calculate_average_pml_material_properties(slab) == pytest.approx(expected)


@pytest.mark.parametrize("face", PML.boundaryIDs)
def test_explicit_sigma_is_not_recalibrated(make_pml_grid, face):
    grid = _grid(make_pml_grid)
    grid.solid[_entrance(grid, face, 3)] = 3
    grid.pmls["cfs"][0].sigma.max = 0.123
    slab = grid._construct_pml(face, 3)

    slab.calculate_update_coeffs(*grid._calculate_average_pml_material_properties(slab))

    assert slab.CFS[0].sigma.max == 0.123


@pytest.mark.parametrize("face", PML.boundaryIDs)
@pytest.mark.parametrize("halo", (0, 1))
def test_mpi_entrance_average_excludes_transverse_halos(make_pml_grid, face, halo):
    pytest.importorskip("mpi4py.MPI")
    from gprMax.grid.mpi_grid import MPIGrid

    grid = _grid(make_pml_grid)
    slab = grid._construct_pml(face, 3)
    axis = "xyz".index(face[0])
    grid.negative_halo_offset = np.full(3, halo, dtype=np.int32)
    grid.negative_halo_offset[axis] = 0
    # The excluded halo cells deliberately retain free-space values.
    grid.solid[_entrance(grid, face, 3)][halo:, halo:] = 3
    slab.comm = SimpleNamespace(allreduce=lambda value, op: value)

    assert MPIGrid._calculate_average_pml_material_properties(grid, slab) == (15.0, 2.0)
