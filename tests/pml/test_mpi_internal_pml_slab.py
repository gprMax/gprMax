"""Rank-local partitioning tests for MPI internal PML slabs."""

import numpy as np
import pytest

from gprMax.grid.mpi_grid import MPIGrid
from gprMax.pml import InternalPMLSpec


def _grid(lower, upper, negative_halo):
    grid = object.__new__(MPIGrid)
    grid.lower_extent = np.asarray(lower, dtype=np.int32)
    grid.upper_extent = np.asarray(upper, dtype=np.int32)
    grid.negative_halo_offset = np.asarray(negative_halo, dtype=np.int32)
    grid.size = grid.upper_extent - grid.lower_extent
    grid.global_size = np.asarray((40, 40, 40), dtype=np.int32)
    return grid


def _spec(maximum_face="xmax", **bounds):
    defaults = {
        "xs": 8,
        "xf": 28,
        "ys": 12,
        "yf": 28,
        "zs": 12,
        "zf": 28,
    }
    defaults.update(bounds)
    return InternalPMLSpec(
        ID="load",
        maximum_face=maximum_face,
        direction=f"{maximum_face[0]}{'minus' if maximum_face.endswith('0') else 'plus'}",
        **defaults,
    )


@pytest.mark.parametrize(
    ("maximum_face", "expected_offsets"),
    (("xmax", (0, 11)), ("x0", (8, 0))),
)
def test_normal_partition_preserves_global_profile_and_negative_halo(
    maximum_face, expected_offsets
):
    grids = (
        _grid((0, 0, 0), (20, 40, 40), (0, 0, 0)),
        _grid((19, 0, 0), (40, 40, 40), (1, 0, 0)),
    )

    localised = [grid._localise_internal_pml_spec(_spec(maximum_face)) for grid in grids]
    first, second = localised

    assert first is not None and second is not None
    assert (first[0].xs, first[0].xf) == (8, 20)
    assert (second[0].xs, second[0].xf) == (0, 9)
    assert (first[1], second[1]) == expected_offsets
    assert first[2] == second[2] == 20
    assert first[3] is second[3] is True

    # The global x=19 cell is the positive rank's negative halo. Keeping that
    # one-cell overlap gives the ordinary Yee plane at the seam one owner
    # after the field halo exchange.
    assert grids[0].local_to_global_coordinate(np.asarray((19, 0, 0)))[0] == 19
    assert grids[1].local_to_global_coordinate(np.asarray((0, 0, 0)))[0] == 19


def test_transverse_partition_keeps_complete_normal_profile_on_each_rank():
    grids = (
        _grid((0, 0, 0), (40, 20, 40), (0, 0, 0)),
        _grid((0, 19, 0), (40, 40, 40), (0, 1, 0)),
    )

    localised = [grid._localise_internal_pml_spec(_spec()) for grid in grids]
    first, second = localised

    assert first is not None and second is not None
    assert (first[0].xs, first[0].xf, first[0].ys, first[0].yf) == (8, 28, 12, 20)
    assert (second[0].xs, second[0].xf, second[0].ys, second[0].yf) == (8, 28, 0, 9)
    assert first[1] == second[1] == 0
    assert first[2] == second[2] == 20


def test_boundary_replacement_omits_internal_terminal_e_profile_sample():
    grid = _grid((0, 0, 0), (20, 40, 40), (0, 0, 0))
    spec = _spec(maximum_face="x0", xs=0, xf=8, ys=0, yf=40, zs=0, zf=40)

    localised = grid._localise_internal_pml_spec(spec)

    assert localised is not None
    local_spec, offset, thickness, endpoint = localised
    assert (local_spec.xs, local_spec.xf) == (0, 8)
    assert offset == 0
    assert thickness == 8
    assert endpoint is False


def test_rank_without_slab_overlap_has_no_local_pml():
    grid = _grid((29, 0, 0), (40, 40, 40), (1, 0, 0))

    assert grid._localise_internal_pml_spec(_spec()) is None
