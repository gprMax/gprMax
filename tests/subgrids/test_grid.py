import numpy as np
import pytest

from gprMax.subgrids.grid import SubGridBaseGrid


class _ConcreteSubGrid(SubGridBaseGrid):
    """Minimal concrete SubGridBaseGrid for testing coordinate maths only.

    SubGridBaseGrid is abstract because it doesn't implement the FDTD
    update equations - those aren't needed to test local_to_global.
    """

    def update_magnetic_is(self, precursors):
        pass

    def update_electric_is(self, precursors):
        pass

    def update_electric_os(self, main_grid):
        pass

    def update_magnetic_os(self, main_grid):
        pass

    def print_info(self):
        pass


def make_subgrid(ratio=3, i0=10, j0=5, k0=2, main_dl=0.001):
    """Builds a subgrid with the same shape of attributes SubGridBase.setup()
    would have populated (set_discretisation + set_main_grid_indices), so
    local_to_global sees realistic state.
    """
    sg = _ConcreteSubGrid(
        ratio=ratio,
        id="sg",
        filter=True,
        is_os_sep=3,
        pml_separation=ratio // 2 + 2,
        subgrid_pml_thickness=6,
        interpolation=1,
    )
    sg.dx = main_dl / ratio
    sg.dy = main_dl / ratio
    sg.dz = main_dl / ratio
    sg.i0, sg.j0, sg.k0 = i0, j0, k0
    return sg


@pytest.mark.parametrize(
    "ratio,i0,j0,k0,global_index",
    [
        (3, 10, 5, 2, (50, 30, 12)),
        (5, 0, 0, 0, (7, 7, 7)),
        (1, 7, 3, 9, (100, 40, 60)),
        (9, 20, 20, 20, (500, 250, 300)),
    ],
)
def test_local_to_global_inverts_forward_placement(ratio, i0, j0, k0, global_index):
    """local_to_global must invert the forward placement transform used when
    building objects in a subgrid (SubgridUserInput.translate_to_gap):

        local = (global_index - i0 * ratio) + n_boundary_cells

    A source/receiver discretised at `global_index` in the fine (subgrid)
    resolution and placed via that forward transform must round-trip back
    to its original physical position through local_to_global. This is the
    bug that was previously silently under-reporting/mis-reporting subgrid
    source and receiver Positions in output and geometry-view metadata.
    """
    sg = make_subgrid(ratio=ratio, i0=i0, j0=j0, k0=k0)
    global_index = np.array(global_index)

    # Forward transform (mirrors SubgridUserInput.translate_to_gap).
    boundary = np.array([sg.n_boundary_cells_x, sg.n_boundary_cells_y, sg.n_boundary_cells_z])
    local_index = (global_index - np.array([i0, j0, k0]) * ratio) + boundary

    recovered_position = np.array(sg.local_to_global(local_index))
    expected_position = global_index * sg.dl

    np.testing.assert_allclose(recovered_position, expected_position)


def test_local_to_global_offset_matters():
    """A regression guard for the original bug: naively scaling the local
    index by dl (ignoring the subgrid's boundary/placement offset) must NOT
    equal local_to_global's result when the subgrid is not placed at the
    main-grid origin. If this test fails after a future refactor, it means
    local_to_global has regressed into the old (wrong) `coord * dl` behaviour.
    """
    sg = make_subgrid(ratio=3, i0=10, j0=5, k0=2)
    local_index = np.array([50, 30, 12])

    naive_position = local_index * sg.dl
    correct_position = np.array(sg.local_to_global(local_index))

    assert not np.allclose(naive_position, correct_position)
