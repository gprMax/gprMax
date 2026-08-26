from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.user_inputs import MPIUserInput, SubgridUserInput


class _BoundsGrid(SimpleNamespace):
    """Small grid stand-in implementing the normal three-axis bounds check."""

    def within_bounds(self, point):
        for index, dimension in enumerate("xyz"):
            if point[index] < 0 or point[index] > self.size[index]:
                raise ValueError(dimension)
        return True


def _subgrid_input(origin=(5, 7, 11)):
    grid = _BoundsGrid(
        dl=np.full(3, 1e-3),
        size=np.full(3, 20, dtype=np.int32),
        nx=20,
        ny=20,
        nz=20,
        ratio=3,
        i0=origin[0],
        j0=origin[1],
        k0=origin[2],
        n_boundary_cells_x=4,
        n_boundary_cells_y=4,
        n_boundary_cells_z=4,
    )
    return SubgridUserInput(grid)


def _global_extent(user_input, axis, local_index):
    grid = user_input.grid
    origin = (grid.i0, grid.j0, grid.k0)[axis] * grid.ratio
    boundary = (
        grid.n_boundary_cells_x,
        grid.n_boundary_cells_y,
        grid.n_boundary_cells_z,
    )[axis]
    return (origin + local_index - boundary) * grid.dl[axis]


@pytest.mark.parametrize(("dimension", "axis"), tuple(zip("xyz", range(3))))
def test_subgrid_thickness_ignores_transverse_carrier_coordinates(dimension, axis):
    user_input = _subgrid_input()
    lower_extent = _global_extent(user_input, axis, local_index=8)

    # Reproduce the old failure condition: the zeros used to carry a scalar
    # extent translate outside an off-origin subgrid in the other two axes.
    carrier = np.zeros(3)
    carrier[axis] = lower_extent
    translated_carrier = user_input.discretise_point(tuple(carrier))
    assert np.delete(translated_carrier, axis).max() < 0

    within_grid, local_lower, local_thickness = user_input.check_thickness(
        dimension,
        lower_extent,
        3e-3,
        "#test_object",
    )

    assert within_grid
    assert local_lower == pytest.approx(8e-3)
    assert local_thickness == pytest.approx(3e-3)


@pytest.mark.parametrize(("dimension", "axis"), tuple(zip("xyz", range(3))))
def test_subgrid_thickness_still_rejects_requested_axis_overflow(dimension, axis):
    user_input = _subgrid_input()
    lower_extent = _global_extent(user_input, axis, local_index=18)

    with pytest.raises(
        ValueError,
        match=rf"extends beyond the size of the model in the {dimension} dimension",
    ):
        user_input.check_thickness(
            dimension,
            lower_extent,
            3e-3,
            "#test_object",
        )


@pytest.mark.parametrize(("dimension", "axis"), tuple(zip("xyz", range(3))))
def test_subgrid_thickness_is_stable_at_large_absolute_offset(dimension, axis):
    """Carrier-coordinate cancellation must not depend on proximity to zero."""

    user_input = _subgrid_input(origin=(500_003, 700_007, 1_100_009))
    lower_extent = _global_extent(user_input, axis, local_index=8)

    within_grid, local_lower, local_thickness = user_input.check_thickness(
        dimension,
        lower_extent,
        3e-3,
        "#test_object",
    )

    assert within_grid
    assert local_lower == pytest.approx(8e-3)
    assert local_thickness == pytest.approx(3e-3)


class _MPIGrid(SimpleNamespace):
    def global_to_local_coordinate(self, point):
        return point - self.lower_extent

    def local_to_global_coordinate(self, point):
        return point + self.lower_extent

    def within_bounds(self, local_point):
        global_point = self.local_to_global_coordinate(local_point)
        for index, dimension in enumerate("xyz"):
            if global_point[index] < 0 or global_point[index] > self.global_size[index]:
                raise ValueError(dimension)
        return True


@pytest.mark.parametrize(("dimension", "axis"), tuple(zip("xyz", range(3))))
def test_mpi_thickness_crossing_lower_rank_face_remains_clipped(dimension, axis):
    lower_extent = np.zeros(3, dtype=np.int32)
    lower_extent[axis] = 9
    grid = _MPIGrid(
        dl=np.ones(3),
        size=np.full(3, 11, dtype=np.int32),
        global_size=np.full(3, 30, dtype=np.int32),
        lower_extent=lower_extent,
    )
    user_input = MPIUserInput(grid)

    within_grid, local_lower, local_thickness = user_input.check_thickness(
        dimension,
        lower_extent=5,
        thickness=10,
        cmd_str="#test_object",
    )

    assert within_grid
    assert local_lower == pytest.approx(0)
    assert local_thickness == pytest.approx(6)
