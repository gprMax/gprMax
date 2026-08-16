"""``VtkUnstructuredGrid`` — explicit points and cells, for line geometry.

Where ImageData describes a grid implicitly, an unstructured grid stores
everything: N point coordinates, C cell types, and a connectivity array cut
into cells by an offsets array. gprMax uses it for geometry views in *line*
mode, where each cell is a two-point edge along a material boundary.

The four arrays have to agree, and none of the agreements is checkable by VTK
after the fact — a connectivity index one too large reads a point that is not
there. The constructor therefore validates three relationships before writing
anything:

* ``cell_offsets`` is one longer than ``cell_types`` — C cells need C+1
  boundaries;
* ``cell_offsets`` ascends — a cell cannot end before it starts;
* ``connectivity`` is at least as long as the final offset — every cell's
  points must exist.

The fourth case, a connectivity array *longer* than the offsets require, is a
warning rather than an error: the surplus is simply unreferenced.

**Seven datasets** are written on construction, and their names are fixed by
the VTKHDF specification. They are asserted as a set here, because a missing
one produces a file VTK opens and then renders as nothing at all — the most
expensive kind of failure to diagnose.

**``Points`` is the exception to the transpose.** It is written with
``xyz_data_ordering=False``, keeping the ``(N, 3)`` layout VTK expects for
coordinates. Every *other* dataset goes through the default path — which is
what makes the 2-D point-data case below worth pinning.

Only the serial path is exercised. The MPI branch needs a real communicator
with more than one rank; ``tests/unit/outputs/test_mpi_grid_view.py`` covers
that ground from above.
"""

import h5py
import numpy as np
import pytest

from gprMax.vtkhdf_filehandlers.vtk_unstructured_grid import VtkUnstructuredGrid
from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

from .conftest import UNSTRUCTURED_GRID_TYPE

# The seven datasets the VTKHDF specification requires at the root of an
# UnstructuredGrid file. Names and spelling are fixed by the format.
REQUIRED_DATASETS = [
    "VTKHDF/Connectivity",
    "VTKHDF/NumberOfCells",
    "VTKHDF/NumberOfConnectivityIds",
    "VTKHDF/NumberOfPoints",
    "VTKHDF/Offsets",
    "VTKHDF/Points",
    "VTKHDF/Types",
]


class TestConstruction:
    """A minimal grid: two points joined by one line."""

    def test_the_type_attribute_says_unstructured_grid(self, make_unstructured_grid, read_h5):
        with make_unstructured_grid() as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert attrs["VTKHDF/Type"] == UNSTRUCTURED_GRID_TYPE

    def test_all_seven_datasets_are_written(self, make_unstructured_grid, read_h5):
        with make_unstructured_grid() as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert sorted(datasets) == REQUIRED_DATASETS

    def test_the_dataset_names_are_the_specification_ones(self):
        """Pinned against the enum, so a rename is caught at the source."""
        assert sorted(member.value for member in VtkUnstructuredGrid.Dataset) == [
            "Connectivity",
            "NumberOfCells",
            "NumberOfConnectivityIds",
            "NumberOfPoints",
            "Offsets",
            "Points",
            "Types",
        ]

    def test_the_file_is_always_opened_for_writing(self, make_unstructured_grid):
        with make_unstructured_grid() as handler:
            assert handler.file_handler.mode == "r+"

    def test_the_serial_partition_is_zero(self, make_unstructured_grid):
        """Without a communicator there is one partition, numbered zero."""
        with make_unstructured_grid() as handler:
            assert handler.partition == 0


class TestCounts:
    """The three count datasets, and the properties that mirror them."""

    def test_the_cell_count_is_the_number_of_cell_types(
        self, make_unstructured_grid, line_grid_arrays
    ):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            assert handler.number_of_cells == 2

    def test_the_point_count_is_the_number_of_coordinates(
        self, make_unstructured_grid, line_grid_arrays
    ):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            assert handler.number_of_points == 3

    def test_the_connectivity_count_is_the_array_length(
        self, make_unstructured_grid, line_grid_arrays
    ):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            assert handler.number_of_connectivity_ids == 4

    def test_the_counts_are_written_to_disk(
        self, make_unstructured_grid, line_grid_arrays, read_h5
    ):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert (
            datasets["VTKHDF/NumberOfCells"][0],
            datasets["VTKHDF/NumberOfPoints"][0],
            datasets["VTKHDF/NumberOfConnectivityIds"][0],
        ) == (2, 3, 4)

    def test_the_counts_are_one_element_datasets(self, make_unstructured_grid, read_h5):
        """One entry per partition; serial writes a single element.

        The format expects an array here even when there is one rank, which
        is why the scalar is expanded rather than written as an attribute.
        """
        with make_unstructured_grid() as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/NumberOfCells"].shape == (1,)

    def test_the_global_counts_match_the_local_ones(self, make_unstructured_grid, line_grid_arrays):
        """Serial: no reduction happens, so global equals local."""
        with make_unstructured_grid(**line_grid_arrays) as handler:
            assert (
                handler.global_number_of_cells == handler.number_of_cells
                and handler.global_number_of_points == handler.number_of_points
            )

    def test_the_offsets_start_at_zero(self, make_unstructured_grid):
        """Serial has nothing before it to skip."""
        with make_unstructured_grid() as handler:
            assert list(handler.cells_offset) == [0]

    def test_the_point_offsets_are_a_pair(self, make_unstructured_grid):
        """Two dimensions, because ``Points`` is an ``(N, 3)`` array."""
        with make_unstructured_grid() as handler:
            assert list(handler.points_offset) == [0, 0]


class TestPointsDataset:
    """Coordinates — the one dataset written without the ZYX transpose."""

    def test_the_shape_is_points_by_three(self, make_unstructured_grid, line_grid_arrays, tmp_path):
        """``(N, 3)`` on disk, exactly as VTK expects for coordinates."""
        with make_unstructured_grid(**line_grid_arrays):
            pass

        with h5py.File(tmp_path / "grid.vtkhdf", "r") as f:
            assert f["VTKHDF/Points"].shape == (3, 3)

    def test_the_coordinates_are_not_transposed(self, make_unstructured_grid, tmp_path):
        """A distinctive point makes the orientation unambiguous."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with make_unstructured_grid(points=points):
            pass

        with h5py.File(tmp_path / "grid.vtkhdf", "r") as f:
            stored = f["VTKHDF/Points"][()]

        assert list(stored[0]) == [1.0, 2.0, 3.0]

    def test_the_coordinates_survive_the_round_trip(self, make_unstructured_grid, read_h5):
        points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        with make_unstructured_grid(points=points) as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Points"] == pytest.approx(points)


class TestCellsAndConnectivity:
    """How points are grouped into cells."""

    def test_the_cell_types_are_written(self, make_unstructured_grid, read_h5):
        with make_unstructured_grid() as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Types"]) == [VtkCellType.LINE]

    def test_the_cell_types_are_unsigned_bytes(self, make_unstructured_grid, read_h5):
        """One byte per cell; a geometry view can hold millions."""
        with make_unstructured_grid() as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Types"].dtype == np.uint8

    def test_the_connectivity_is_written(self, make_unstructured_grid, line_grid_arrays, read_h5):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Connectivity"]) == [0, 1, 1, 2]

    def test_the_offsets_are_written(self, make_unstructured_grid, line_grid_arrays, read_h5):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Offsets"]) == [0, 2, 4]

    def test_the_offsets_are_one_longer_than_the_cells(
        self, make_unstructured_grid, line_grid_arrays, read_h5
    ):
        """C+1 boundaries for C cells — the invariant the check enforces."""
        with make_unstructured_grid(**line_grid_arrays) as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert len(datasets["VTKHDF/Offsets"]) == len(datasets["VTKHDF/Types"]) + 1

    def test_a_grid_with_no_cells_is_accepted(self, make_unstructured_grid):
        """An empty geometry view is a legitimate, if dull, output."""
        with make_unstructured_grid(
            points=np.zeros((0, 3)),
            cell_types=np.array([], dtype=np.uint8),
            connectivity=np.array([], dtype=np.int32),
            cell_offsets=np.array([0], dtype=np.int32),
        ) as handler:
            assert handler.number_of_cells == 0


class TestValidation:
    """Three raises and one warning, before anything is written."""

    def test_too_few_offsets_raises(self, make_unstructured_grid, line_grid_arrays):
        line_grid_arrays["cell_offsets"] = np.array([0, 2], dtype=np.int32)

        with pytest.raises(ValueError, match="one longer than cell_types"):
            make_unstructured_grid(**line_grid_arrays)

    def test_too_many_offsets_raises(self, make_unstructured_grid, line_grid_arrays):
        line_grid_arrays["cell_offsets"] = np.array([0, 2, 4, 6], dtype=np.int32)

        with pytest.raises(ValueError, match="one longer than cell_types"):
            make_unstructured_grid(**line_grid_arrays)

    def test_unsorted_offsets_raise(self, make_unstructured_grid, line_grid_arrays):
        """A cell that ends before it starts would read backwards."""
        line_grid_arrays["cell_offsets"] = np.array([0, 4, 2], dtype=np.int32)

        with pytest.raises(ValueError, match="sorted in ascending order"):
            make_unstructured_grid(**line_grid_arrays)

    def test_equal_consecutive_offsets_are_allowed(self, make_unstructured_grid, line_grid_arrays):
        """A zero-length cell is degenerate but not out of order."""
        line_grid_arrays["cell_offsets"] = np.array([0, 2, 2], dtype=np.int32)

        with make_unstructured_grid(**line_grid_arrays) as handler:
            assert handler.number_of_cells == 2

    def test_a_short_connectivity_array_raises(self, make_unstructured_grid, line_grid_arrays):
        """The last cell would reference points past the end of the array."""
        line_grid_arrays["connectivity"] = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="shorter than final cell_offsets"):
            make_unstructured_grid(**line_grid_arrays)

    def test_a_long_connectivity_array_warns(
        self, make_unstructured_grid, line_grid_arrays, caplog
    ):
        """Surplus entries are unreferenced, not wrong — a warning suffices."""
        line_grid_arrays["connectivity"] = np.array([0, 1, 1, 2, 2, 0], dtype=np.int32)

        with make_unstructured_grid(**line_grid_arrays):
            pass

        assert "will be ignored" in caplog.text

    def test_a_long_connectivity_array_is_still_written_in_full(
        self, make_unstructured_grid, line_grid_arrays, read_h5
    ):
        """The surplus is stored; only the offsets decide what is read."""
        line_grid_arrays["connectivity"] = np.array([0, 1, 1, 2, 2, 0], dtype=np.int32)

        with make_unstructured_grid(**line_grid_arrays) as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert len(datasets["VTKHDF/Connectivity"]) == 6

    def test_a_valid_grid_does_not_warn(self, make_unstructured_grid, line_grid_arrays, caplog):
        with make_unstructured_grid(**line_grid_arrays):
            pass

        assert caplog.records == []

    def test_a_failed_construction_still_leaves_a_file(
        self, make_unstructured_grid, line_grid_arrays, tmp_path
    ):
        """The file is opened — and truncated — before validation runs.

        So a rejected grid destroys any earlier file of the same name and
        leaves an empty one in its place, with the handle never closed. Pinned
        as the current behaviour; written up in
        ``notes/bugs/vtkhdf-truncates-before-validating.md``.
        """
        line_grid_arrays["cell_offsets"] = np.array([0, 2], dtype=np.int32)

        with pytest.raises(ValueError):
            make_unstructured_grid(**line_grid_arrays)

        assert (tmp_path / "grid.vtkhdf").is_file()


class TestAddCellData:
    """One value, or one 3-vector, per cell."""

    def test_a_matching_array_is_written(self, make_unstructured_grid, line_grid_arrays, read_h5):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            handler.add_cell_data("Material", np.array([1, 2]))
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/CellData/Material"]) == [1, 2]

    def test_a_three_component_array_is_accepted(self, make_unstructured_grid, line_grid_arrays):
        """Vectors per cell — the ``(C, 3)`` layout VTK defines."""
        with make_unstructured_grid(**line_grid_arrays) as handler:
            handler.add_cell_data("Vectors", np.zeros((2, 3)))

            assert "CellData" in handler.root_group

    def test_a_single_component_array_is_accepted(self, make_unstructured_grid, line_grid_arrays):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            handler.add_cell_data("Scalars", np.zeros((2, 1)))

            assert "CellData" in handler.root_group

    def test_a_wrong_length_raises(self, make_unstructured_grid, line_grid_arrays):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            with pytest.raises(ValueError, match="must match the number of cells"):
                handler.add_cell_data("Material", np.array([1, 2, 3]))

    def test_the_error_names_the_partition(self, make_unstructured_grid, line_grid_arrays):
        """Under MPI the rank is what a user needs to know."""
        with make_unstructured_grid(**line_grid_arrays) as handler:
            with pytest.raises(ValueError, match="partition 0"):
                handler.add_cell_data("Material", np.array([1]))

    def test_a_three_dimensional_array_raises(self, make_unstructured_grid, line_grid_arrays):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            with pytest.raises(ValueError, match="1 or 2 dimensions"):
                handler.add_cell_data("Material", np.zeros((2, 3, 4)))

    def test_a_two_component_array_raises(self, make_unstructured_grid, line_grid_arrays):
        """VTK has scalars and 3-vectors; nothing in between."""
        with make_unstructured_grid(**line_grid_arrays) as handler:
            with pytest.raises(ValueError, match="shape 1 or 3"):
                handler.add_cell_data("Material", np.zeros((2, 2)))

    def test_two_dimensional_cell_data_is_stored_transposed(
        self, make_unstructured_grid, line_grid_arrays, tmp_path
    ):
        """``(C, 3)`` goes in as ``(3, C)`` — the opposite of VTK's layout.

        ``Points`` is written with ``xyz_data_ordering=False`` for exactly
        this reason, but ``add_cell_data`` and ``add_point_data`` are not, so
        vector-valued cell data comes out with components and tuples swapped.
        gprMax only ever writes scalar cell data, where a 1-D transpose is a
        no-op, which is why it has never been noticed. Pinned as the current
        behaviour; written up in
        ``notes/bugs/vtkhdf-unstructured-vector-data-transposed.md``.
        """
        with make_unstructured_grid(**line_grid_arrays) as handler:
            handler.add_cell_data("Vectors", np.zeros((2, 3)))

        with h5py.File(tmp_path / "grid.vtkhdf", "r") as f:
            assert f["VTKHDF/CellData/Vectors"].shape == (3, 2)


class TestAddPointData:
    """One value, or one 3-vector, per point."""

    def test_a_matching_array_is_written(self, make_unstructured_grid, line_grid_arrays, read_h5):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            handler.add_point_data("Field", np.array([1, 2, 3]))
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/PointData/Field"]) == [1, 2, 3]

    def test_a_three_component_array_is_accepted(self, make_unstructured_grid, line_grid_arrays):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            handler.add_point_data("Field", np.zeros((3, 3)))

            assert "PointData" in handler.root_group

    def test_a_wrong_length_raises(self, make_unstructured_grid, line_grid_arrays):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            with pytest.raises(ValueError, match="must match the number of points"):
                handler.add_point_data("Field", np.array([1, 2]))

    def test_a_three_dimensional_array_raises(self, make_unstructured_grid, line_grid_arrays):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            with pytest.raises(ValueError, match="1 or 2 dimensions"):
                handler.add_point_data("Field", np.zeros((3, 3, 3)))

    def test_a_two_component_array_raises(self, make_unstructured_grid, line_grid_arrays):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            with pytest.raises(ValueError, match="shape 1 or 3"):
                handler.add_point_data("Field", np.zeros((3, 2)))

    def test_point_and_cell_data_coexist(self, make_unstructured_grid, line_grid_arrays, read_h5):
        with make_unstructured_grid(**line_grid_arrays) as handler:
            handler.add_cell_data("Material", np.array([1, 2]))
            handler.add_point_data("Field", np.array([1, 2, 3]))
            path = handler.filename

        _, datasets = read_h5(path)

        assert "VTKHDF/CellData/Material" in datasets
        assert "VTKHDF/PointData/Field" in datasets

    def test_the_seven_required_datasets_are_undisturbed(
        self, make_unstructured_grid, line_grid_arrays, read_h5
    ):
        """Adding data must not perturb the structural datasets."""
        with make_unstructured_grid(**line_grid_arrays) as handler:
            handler.add_point_data("Field", np.array([1, 2, 3]))
            path = handler.filename

        _, datasets = read_h5(path)

        assert set(REQUIRED_DATASETS) <= set(datasets)


pytestmark = pytest.mark.unit
