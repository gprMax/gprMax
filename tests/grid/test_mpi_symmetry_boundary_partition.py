"""Rank-local ownership tests for MPI PEC/PMC symmetry boundaries."""

import numpy as np

from gprMax.grid.mpi_grid import MPIGrid
from gprMax.symmetry_boundaries import build_symmetry_boundary_edges


def _grid(neighbours, boundaries):
    grid = object.__new__(MPIGrid)
    grid.neighbours = np.asarray(neighbours, dtype=np.int32)
    grid.symmetry_boundaries = boundaries
    return grid


def test_mpi_grid_keeps_global_declarations_but_selects_owned_faces():
    # This rank touches x0 and zmax, but is internal in y.
    grid = _grid(
        [[-1, 1], [2, 3], [4, -1]],
        {"x0": "pmc", "y0": "pec", "zmax": "pmc"},
    )

    assert grid.symmetry_boundaries == {"x0": "pmc", "y0": "pec", "zmax": "pmc"}
    assert grid.get_local_symmetry_boundaries() == {"x0": "pmc", "zmax": "pmc"}


def test_mpi_internal_rank_seams_do_not_create_domain_edges():
    # Although x0 is a PMC face, this rank is internal in both transverse
    # directions. Its local y/z limits are MPI seams, not domain edges.
    grid = _grid([[-1, 1], [2, 3], [4, 5]], {"x0": "pmc"})

    assert build_symmetry_boundary_edges(grid) == []


def test_mpi_edge_dispatch_requires_both_global_faces():
    # The rank touches x0 and y0 but is internal in z. Only their shared Ez
    # edge belongs to this rank; x0-z and y0-z local limits remain MPI seams.
    grid = _grid([[-1, 1], [-1, 2], [3, 4]], {"x0": "pmc", "y0": "pmc"})

    edges = build_symmetry_boundary_edges(grid)
    assert len(edges) == 1
    func, x0_pmc, y0_pmc, *_ = edges[0]
    assert func.__name__ == "update_symmetry_boundary_electric_Ez_X0_Y0"
    assert x0_pmc and y0_pmc


def test_mpi_global_corner_dispatches_its_three_physical_edges():
    grid = _grid(
        [[-1, 1], [-1, 2], [-1, 3]],
        {"x0": "pmc", "y0": "pmc", "z0": "pmc"},
    )

    names = {edge[0].__name__ for edge in build_symmetry_boundary_edges(grid)}
    assert names == {
        "update_symmetry_boundary_electric_Ez_X0_Y0",
        "update_symmetry_boundary_electric_Ey_X0_Z0",
        "update_symmetry_boundary_electric_Ex_Y0_Z0",
    }
