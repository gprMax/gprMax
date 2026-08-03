"""``GeometryViewLines`` — every cell edge drawn separately.

The bulky geometry export: a VTK UnstructuredGrid holding three line segments
per cell, one along each axis, each carrying the material of the corresponding
``ID`` component. It is what you need when debugging why an antenna is not
resonating where it should, because it shows the staircase discretisation of a
curved object rather than smoothing it into bricks.

**The step is forced to one.** ``__init__`` overrides the base signature and
hard-codes ``dx = dy = dz = 1``. Drawing individual cell edges at a stride
would be meaningless — you would be drawing edges that do not exist.

**The point-ID walk is the fiddly part.** ``get_line_properties`` numbers
points over a ``(nx+1, ny+1, nz+1)`` lattice while iterating cells over
``(nx, ny, nz)``, so it has to skip the far-edge points that are the source of
no line. The strides are ``z_step = 1``, ``y_step = nz + 1`` and
``x_step = (nz + 1)(ny + 1)``; after each k-loop the walk skips one point, and
after each j-loop it skips a further ``nz + 1``. Small grids make this
hand-checkable, and the tests below do exactly that.

**Materials are remapped here.** Unlike the voxel exporter, this one calls
``initialise_materials(filter_materials=False)`` and then maps the raw material
data through the resulting index, so cell values are positions in the metadata
table rather than grid material IDs.
"""

import numpy as np
import pytest

from gprMax.cython.geometry_outputs import get_line_properties
from gprMax.geometry_outputs.geometry_view_lines import (
    GeometryViewLines,
    MPIGeometryViewLines,
)
from gprMax.geometry_outputs.geometry_views import GeometryView

from .conftest import DL, DL_ANISO


@pytest.fixture
def make_line_view(make_view_grid):
    """Factory for a ``GeometryViewLines`` with its filename resolved."""

    def _make(
        start=(0, 0, 0),
        stop=(2, 2, 2),
        filename="lines",
        grid=None,
        prepare=True,
        **grid_kwargs,
    ):
        g = grid if grid is not None else make_view_grid(**grid_kwargs)
        view = GeometryViewLines(*start, *stop, filename, g)
        view.set_filename()
        if prepare:
            view.prep_vtk()
        return view

    return _make


class TestClassSurface:
    def test_extends_geometry_view(self):
        """Expects the shared base."""
        assert issubclass(GeometryViewLines, GeometryView)

    def test_the_constructor_takes_no_step(self, make_view_grid):
        """Expects six coordinates plus a filename, not nine — the step is not
        the caller's to choose."""
        view = GeometryViewLines(0, 0, 0, 2, 2, 2, "lines", make_view_grid())
        assert view.grid_view.step.tolist() == [1, 1, 1]

    def test_the_step_is_forced_to_one(self, make_view_grid):
        """Expects a unit stride however the view is built: individual cell
        edges only exist at the grid's own resolution."""
        view = GeometryViewLines(1, 2, 3, 5, 6, 7, "lines", make_view_grid(nx=8, ny=8, nz=8))
        assert view.grid_view.dx == view.grid_view.dy == view.grid_view.dz == 1


class TestLinePropertiesKernel:
    """``get_line_properties``, hand-checked on the smallest possible grids."""

    def test_a_single_cell_gives_three_lines(self):
        """Expects one line per axis for one cell."""
        ID = np.zeros((6, 2, 2, 2), dtype=np.uint32)
        connectivity, material_data = get_line_properties(3, 1, 1, 1, ID)
        assert len(material_data) == 3
        assert len(connectivity) == 6

    def test_a_single_cell_connectivity_is_hand_computable(self):
        """Expects ``[0,4, 0,2, 0,1]``.

        For ``nx=ny=nz=1`` the strides are ``x_step = (1+1)(1+1) = 4``,
        ``y_step = 1+1 = 2`` and ``z_step = 1``. All three edges start at point
        0 and end one stride away along their own axis."""
        ID = np.zeros((6, 2, 2, 2), dtype=np.uint32)
        connectivity, _ = get_line_properties(3, 1, 1, 1, ID)
        assert connectivity.tolist() == [0, 4, 0, 2, 0, 1]

    def test_two_cells_along_x_skip_the_far_edge_points(self):
        """Expects ``[0,4,0,2,0,1, 4,8,4,6,4,5]``.

        After the first cell the walk advances one point, then skips one for
        the ``(i, j, nz)`` plane and ``nz + 1`` more for the ``(i, ny, ·)``
        row — landing on point 4, which is exactly ``x_step``."""
        ID = np.zeros((6, 3, 2, 2), dtype=np.uint32)
        connectivity, _ = get_line_properties(6, 2, 1, 1, ID)
        assert connectivity.tolist() == [0, 4, 0, 2, 0, 1, 4, 8, 4, 6, 4, 5]

    def test_line_count_is_three_per_cell(self):
        """Expects ``3 * nx * ny * nz`` lines for any grid."""
        ID = np.zeros((6, 4, 3, 5), dtype=np.uint32)
        _, material_data = get_line_properties(3 * 3 * 2 * 4, 3, 2, 4, ID)
        assert len(material_data) == 72

    def test_each_line_takes_its_own_id_component(self):
        """Expects the x, y and z edges of a cell to read ``ID[0]``, ``ID[1]``
        and ``ID[2]`` respectively — the three components in order."""
        ID = np.zeros((6, 2, 2, 2), dtype=np.uint32)
        ID[0, 0, 0, 0] = 7
        ID[1, 0, 0, 0] = 8
        ID[2, 0, 0, 0] = 9
        _, material_data = get_line_properties(3, 1, 1, 1, ID)
        assert material_data.tolist() == [7, 8, 9]

    def test_higher_id_components_are_ignored(self):
        """Expects only the first three of the six ``ID`` components to be
        read — components 3-5 are the magnetic ones, and a cell edge is an
        electric quantity."""
        ID = np.zeros((6, 2, 2, 2), dtype=np.uint32)
        ID[3:, ...] = 99
        _, material_data = get_line_properties(3, 1, 1, 1, ID)
        assert material_data.tolist() == [0, 0, 0]

    def test_cells_are_visited_in_x_then_y_then_z_order(self):
        """Expects the innermost loop to be z, so consecutive line triples walk
        along z first."""
        ID = np.zeros((6, 2, 2, 3), dtype=np.uint32)
        ID[0, 0, 0, 0] = 1
        ID[0, 0, 0, 1] = 2
        _, material_data = get_line_properties(6, 1, 1, 2, ID)
        assert material_data[0] == 1
        assert material_data[3] == 2

    def test_connectivity_is_int32(self):
        """Expects ``int32`` point indices, as the VTKHDF connectivity array
        requires."""
        ID = np.zeros((6, 2, 2, 2), dtype=np.uint32)
        connectivity, _ = get_line_properties(3, 1, 1, 1, ID)
        assert connectivity.dtype == np.int32

    def test_material_data_is_uint32(self):
        """Expects ``uint32`` material IDs, matching the ``ID`` array."""
        ID = np.zeros((6, 2, 2, 2), dtype=np.uint32)
        _, material_data = get_line_properties(3, 1, 1, 1, ID)
        assert material_data.dtype == np.uint32

    def test_every_point_index_is_within_the_lattice(self):
        """Expects no index past ``(nx+1)(ny+1)(nz+1) - 1`` — an overrun would
        reference a point that was never written."""
        nx, ny, nz = 3, 4, 5
        ID = np.zeros((6, nx + 1, ny + 1, nz + 1), dtype=np.uint32)
        connectivity, _ = get_line_properties(3 * nx * ny * nz, nx, ny, nz, ID)
        assert connectivity.max() < (nx + 1) * (ny + 1) * (nz + 1)


class TestPrepVtk:
    def test_builds_one_point_per_lattice_node(self, make_line_view):
        """Expects ``(nx+1)(ny+1)(nz+1)`` points — lines join nodes, and there
        is one more node than cell along each axis."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        assert view.points.shape == (27, 3)

    def test_points_are_three_dimensional(self, make_line_view):
        """Expects an ``(n, 3)`` coordinate array."""
        assert make_line_view().points.shape[1] == 3

    def test_points_are_in_metres(self, make_line_view):
        """Expects lattice indices scaled by ``grid.dl``, so the drawing lands
        at the model's physical scale."""
        view = make_line_view(start=(0, 0, 0), stop=(1, 1, 1), nx=8, ny=8, nz=8)
        assert view.points.max() == pytest.approx(DL)

    def test_points_are_offset_by_the_view_start(self, make_line_view):
        """Expects a view of the model's interior to be drawn there rather than
        at the origin."""
        view = make_line_view(start=(2, 2, 2), stop=(4, 4, 4), nx=8, ny=8, nz=8)
        assert view.points.min() == pytest.approx(2 * DL)

    def test_points_follow_an_anisotropic_grid(self, make_line_view):
        """Expects each axis scaled independently."""
        view = make_line_view(start=(0, 0, 0), stop=(1, 1, 1), nx=8, ny=8, nz=8, dl=DL_ANISO)
        assert view.points.max(axis=0) == pytest.approx(list(DL_ANISO))

    def test_three_lines_per_cell(self, make_line_view):
        """Expects ``3 * nx * ny * nz`` cell types."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        assert len(view.cell_types) == 24

    def test_every_cell_is_a_line(self, make_line_view):
        """Expects a uniform VTK cell type — this exporter draws nothing but
        line segments."""
        view = make_line_view()
        assert len(set(view.cell_types.tolist())) == 1

    def test_cell_offsets_step_by_two(self, make_line_view):
        """Expects ``0, 2, 4, …`` — every line has exactly two endpoints, so
        the offsets into the connectivity array advance in pairs."""
        view = make_line_view(stop=(1, 1, 1), nx=8, ny=8, nz=8)
        assert view.cell_offsets.tolist() == [0, 2, 4, 6]

    def test_cell_offsets_have_one_more_entry_than_cells(self, make_line_view):
        """Expects ``n_lines + 1`` offsets, the standard VTK convention where
        the final entry closes the last cell."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        assert len(view.cell_offsets) == len(view.cell_types) + 1

    def test_connectivity_has_two_entries_per_line(self, make_line_view):
        """Expects ``2 * n_lines``."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        assert len(view.connectivity) == 2 * len(view.cell_types)

    def test_material_data_has_one_entry_per_line(self, make_line_view):
        """Expects one material per drawn edge."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        assert len(view.material_data) == len(view.cell_types)

    def test_materials_are_remapped_to_the_view_index(self, make_line_view, make_view_grid):
        """Expects the raw ``ID`` values replaced by positions in the metadata
        table.

        The exporter calls ``initialise_materials(filter_materials=False)``, so
        with the grid's own IDs already dense the map is the identity — but the
        mapping step is what makes the file self-describing when they are not."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        g.ID[...] = 2
        view = make_line_view(grid=g, stop=(2, 2, 2))
        assert set(np.unique(view.material_data)) == {2}

    def test_metadata_is_materials_only(self, make_line_view):
        """Expects PML, source and receiver information to be skipped — this
        exporter asks for ``materials_only``, unlike the voxel one."""
        view = make_line_view()
        assert not hasattr(view.metadata, "pml_thickness")

    def test_metadata_includes_averaged_materials(self, make_line_view, make_view_grid):
        """Expects dielectric-smoothed materials in the table, because the
        drawn edges can reference them."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        g.materials[1].type = "dielectric-smoothed"
        view = make_line_view(grid=g, stop=(2, 2, 2))
        assert len(view.metadata.materials) == 3

    def test_byte_count_sums_every_written_array(self, make_line_view):
        """Expects points, cell types, connectivity, offsets and materials —
        all five arrays this exporter writes."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        assert view.nbytes == (
            view.points.nbytes
            + view.cell_types.nbytes
            + view.connectivity.nbytes
            + view.cell_offsets.nbytes
            + view.material_data.nbytes
        )


class TestWriteVtk:
    def test_writes_a_readable_unstructured_grid(self, make_line_view, read_h5):
        """Expects a VTKHDF file declaring the UnstructuredGrid type."""
        view = make_line_view()
        view.write_vtk()
        attrs, _ = read_h5(view.filename)
        assert attrs["VTKHDF/Type"] == b"UnstructuredGrid"

    def test_writes_the_point_coordinates(self, make_line_view, read_h5):
        """Expects a ``Points`` dataset of shape ``(n, 3)``."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert data["VTKHDF/Points"].shape == (27, 3)

    def test_writes_the_connectivity(self, make_line_view, read_h5):
        """Expects the flat endpoint list, two entries per line."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert data["VTKHDF/Connectivity"].shape == (48,)

    def test_writes_the_cell_offsets(self, make_line_view, read_h5):
        """Expects the offsets array alongside the connectivity."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert data["VTKHDF/Offsets"].shape == (25,)

    def test_writes_the_cell_types(self, make_line_view, read_h5):
        """Expects a ``Types`` dataset, one entry per line."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert data["VTKHDF/Types"].shape == (24,)

    def test_declares_the_counts(self, make_line_view, read_h5):
        """Expects ``NumberOfCells``, ``NumberOfPoints`` and
        ``NumberOfConnectivityIds`` to agree with the arrays."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert int(np.ravel(data["VTKHDF/NumberOfCells"])[0]) == 24
        assert int(np.ravel(data["VTKHDF/NumberOfPoints"])[0]) == 27
        assert int(np.ravel(data["VTKHDF/NumberOfConnectivityIds"])[0]) == 48

    def test_cell_data_is_named_material(self, make_line_view, read_h5):
        """Expects the per-line material under ``VTKHDF/CellData/Material``,
        the same name the voxel exporter uses."""
        view = make_line_view()
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert "VTKHDF/CellData/Material" in data

    def test_material_values_survive_the_round_trip(self, make_line_view, read_h5):
        """Expects the per-line materials written verbatim — a 1D array needs
        no transpose, unlike the voxel exporter's 3D one."""
        view = make_line_view(stop=(2, 2, 2), nx=8, ny=8, nz=8)
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert data["VTKHDF/CellData/Material"] == pytest.approx(view.material_data)

    def test_metadata_is_attached(self, make_line_view, read_h5):
        """Expects the four core field-data entries and nothing more, since
        this exporter asks for ``materials_only``."""
        view = make_line_view()
        view.write_vtk()
        _, data = read_h5(view.filename)
        names = {k.rsplit("/", 1)[-1] for k in data}
        assert {"gprMax_version", "dx_dy_dz", "nx_ny_nz", "material_ids"} <= names
        assert "pml_thickness" not in names


class TestMpiVariant:
    def test_extends_the_serial_exporter(self):
        """Expects the MPI variant to override only the grid-view type and the
        two write-side methods."""
        assert issubclass(MPIGeometryViewLines, GeometryViewLines)
        overrides = {n for n in MPIGeometryViewLines.__dict__ if not n.startswith("__")}
        assert overrides <= {"GRID_VIEW_TYPE", "prep_vtk", "write_vtk"}

    def test_uses_an_mpi_grid_view(self, make_mpi_grid, make_materials):
        """Expects ``MPIGridView``, so points are generated for this rank's
        share of the domain only."""
        from gprMax.geometry_outputs.grid_view import MPIGridView

        grid = make_mpi_grid(
            size=(4, 4, 4),
            negative_halo_offset=(0, 0, 0),
            arrays={"ID": np.ones((6, 5, 5, 5), dtype=np.uint32)},
        )
        grid.materials = make_materials(2)
        view = MPIGeometryViewLines(0, 0, 0, 4, 4, 4, "mpi", grid)
        assert isinstance(view.grid_view, MPIGridView)

    def test_points_use_global_coordinates(self, make_mpi_grid, make_materials):
        """Expects ``global_start + offset`` rather than the local start, so
        each rank's edges land in the right place in the shared file."""
        grid = make_mpi_grid(
            size=(4, 4, 4),
            negative_halo_offset=(0, 0, 0),
            origin=(10, 10, 10),
            arrays={"ID": np.ones((6, 5, 5, 5), dtype=np.uint32)},
        )
        grid.materials = make_materials(2)
        view = MPIGeometryViewLines(0, 0, 0, 4, 4, 4, "mpi", grid)
        view.set_filename()
        view.prep_vtk()
        assert view.points.min() == pytest.approx(10 * DL)
