# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""``GeometryViewVoxels`` — one coloured brick per cell.

The compact geometry export: a VTK ImageData file holding a single ``Material``
value per cell, plus the metadata block. It is what you want for a quick look
at a whole domain.

**It writes raw ``solid`` values.** Unlike ``GeometryViewLines`` and
``GeometryObject``, this exporter never calls ``initialise_materials``, so the
cell data carries the grid's own material numbering rather than a compacted
per-view one. That is self-consistent — ``Metadata`` then falls back to the
grid's full unfiltered material list, so index *n* in the table still names the
material with ``numID == n``. But it means the two view types describe their
materials differently, and a reader that assumes a filtered table for one and
applies it to the other gets the wrong colours. The tests below pin both halves
of that consistency.

**Subgrids get an absolute origin.** A subgrid's own coordinates start at zero,
so a naive origin would stack every subgrid on top of the main grid at the
model origin. ``prep_vtk`` detects a ``SubGridBaseGrid`` and offsets by the
subgrid's position in the parent, scaled by the refinement ratio.
"""

import numpy as np
import pytest

from gprMax.geometry_outputs.geometry_view_voxels import GeometryViewVoxels, MPIGeometryViewVoxels
from gprMax.geometry_outputs.geometry_views import GeometryView
from gprMax.geometry_tags import GeometryTagMap, GeometryTagRegistry

from .conftest import DL, DL_ANISO


@pytest.fixture
def make_voxel_view(make_view_grid):
    """Factory for a ``GeometryViewVoxels`` with its filename resolved."""

    def _make(
        start=(0, 0, 0),
        stop=(4, 4, 4),
        step=(1, 1, 1),
        filename="voxels",
        grid=None,
        prepare=True,
        **grid_kwargs,
    ):
        g = grid if grid is not None else make_view_grid(**grid_kwargs)
        view = GeometryViewVoxels(*start, *stop, *step, filename, g)
        view.set_filename()
        if prepare:
            view.prep_vtk()
        return view

    return _make


class TestClassSurface:
    def test_extends_geometry_view(self):
        """Expects the shared base, so filename handling is inherited."""
        assert issubclass(GeometryViewVoxels, GeometryView)

    def test_does_not_override_the_constructor(self):
        """Expects the base nine-coordinate signature, unlike
        ``GeometryViewLines`` which forces a unit step."""
        assert "__init__" not in GeometryViewVoxels.__dict__

    def test_implements_both_abstract_methods(self, make_voxel_view):
        """Expects a concrete, instantiable exporter."""
        view = make_voxel_view(prepare=False)
        assert callable(view.prep_vtk)
        assert callable(view.write_vtk)


class TestPrepVtk:
    def test_material_data_is_the_solid_array(self, make_voxel_view, make_view_grid):
        """Expects the raw per-cell material IDs from ``get_solid()``."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        g.solid[...] = 2
        view = make_voxel_view(grid=g, stop=(4, 4, 4))
        assert np.all(view.material_data == 2)

    def test_material_data_is_cell_shaped(self, make_voxel_view):
        """Expects ``(nx, ny, nz)`` — one value per cell, with no extra node."""
        assert make_voxel_view(stop=(4, 5, 6), nx=8, ny=8, nz=8).material_data.shape == (4, 5, 6)

    def test_material_data_is_not_remapped(self, make_voxel_view, make_view_grid):
        """Expects the grid's own material numbering to survive.

        ``initialise_materials`` is never called here, so no compaction
        happens. A view containing only material 2 still reports 2, not 0."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        g.solid[...] = 2
        view = make_voxel_view(grid=g, stop=(4, 4, 4))
        assert set(np.unique(view.material_data)) == {2}

    def test_the_metadata_material_table_matches_that_numbering(
        self, make_voxel_view, make_view_grid
    ):
        """Expects the metadata to list the *grid's* full material set, so
        index *n* of the table still names material ``numID == n``.

        This is the consistency that makes the unremapped cell data usable."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        view = make_voxel_view(grid=g, stop=(4, 4, 4))
        assert view.metadata.materials == [m.ID for m in g.materials]

    def test_origin_is_the_physical_start_of_the_window(self, make_voxel_view):
        """Expects ``start * grid.dl`` in metres."""
        view = make_voxel_view(start=(2, 3, 4), stop=(6, 7, 8), nx=12, ny=12, nz=12)
        assert view.origin == pytest.approx([2 * DL, 3 * DL, 4 * DL])

    def test_origin_follows_an_anisotropic_grid(self, make_voxel_view):
        """Expects each axis scaled by its own discretisation."""
        view = make_voxel_view(start=(2, 2, 2), stop=(6, 6, 6), nx=12, ny=12, nz=12, dl=DL_ANISO)
        assert view.origin == pytest.approx([2 * d for d in DL_ANISO])

    def test_spacing_is_the_physical_cell_size(self, make_voxel_view):
        """Expects ``step * grid.dl``, so a strided view reports larger
        bricks."""
        view = make_voxel_view(start=(0, 0, 0), stop=(8, 8, 8), step=(2, 2, 2), nx=12, ny=12, nz=12)
        assert view.spacing == pytest.approx([2 * DL] * 3)

    def test_byte_count_is_the_material_array_size(self, make_voxel_view):
        """Expects ``nbytes`` to size the progress bar from the only array this
        exporter writes."""
        view = make_voxel_view(stop=(4, 4, 4), nx=8, ny=8, nz=8)
        assert view.nbytes == view.material_data.nbytes

    def test_metadata_is_the_full_form(self, make_voxel_view):
        """Expects PML, source and receiver information to be collected —
        ``materials_only`` is left at its default, unlike the lines exporter."""
        view = make_voxel_view()
        assert hasattr(view.metadata, "pml_thickness")


class TestSubgridOrigin:
    @pytest.fixture
    def subgrid_view(self, make_view_grid, make_materials):
        """A voxel view over a genuine ``SubGridBaseGrid`` subclass.

        ``SubGridBaseGrid`` is an ABC with five abstract methods and eight
        required constructor kwargs — all of that is PR 9's territory. The
        concrete subclass below supplies the abstract methods and sets the
        four attributes ``prep_vtk`` actually reads, so the subgrid branch is
        taken with values this test controls.
        """
        from gprMax.subgrids.grid import SubGridBaseGrid

        class StubSubGrid(SubGridBaseGrid):
            def __init__(self):
                # Deliberately bypass SubGridBaseGrid.__init__, whose kwarg
                # validation and derived sizes are covered in PR 9.
                super(SubGridBaseGrid, self).__init__()

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

        g = StubSubGrid()
        g.size = np.array([12, 12, 12], dtype=np.int64)
        g.dl = np.array([DL, DL, DL], dtype=np.float64)
        g.materials = make_materials(3)
        g.initialise_geometry_arrays()
        g.i0, g.j0, g.k0 = 3, 4, 5
        g.ratio = 3
        g.n_boundary_cells_x = 0
        g.n_boundary_cells_y = 0
        g.n_boundary_cells_z = 0

        view = GeometryViewVoxels(0, 0, 0, 4, 4, 4, 1, 1, 1, "sub", g)
        view.set_filename()
        view.prep_vtk()
        return view

    def test_origin_uses_the_parent_position(self, subgrid_view):
        """Expects ``i0 * dx * ratio`` — the subgrid's location in the parent,
        expressed in the subgrid's own (finer) spacing.

        Without this, every subgrid would be drawn at the model origin, stacked
        on top of the main grid."""
        assert subgrid_view.origin == pytest.approx([3 * DL * 3, 4 * DL * 3, 5 * DL * 3])

    def test_the_three_axes_use_their_own_indices(self, subgrid_view):
        """Expects ``i0``, ``j0`` and ``k0`` to drive x, y and z respectively —
        the distinct fixture values make a mix-up visible."""
        assert subgrid_view.origin[0] != subgrid_view.origin[1] != subgrid_view.origin[2]

    def test_spacing_is_unaffected(self, subgrid_view):
        """Expects the subgrid branch to change only the origin — the cell size
        is still ``step * dl``."""
        assert subgrid_view.spacing == pytest.approx([DL] * 3)


class TestWriteVtk:
    def test_writes_a_readable_file(self, make_voxel_view, read_h5):
        """Expects a complete VTKHDF file on disk."""
        view = make_voxel_view()
        view.write_vtk()
        attrs, _ = read_h5(view.filename)
        assert attrs["VTKHDF/Type"] == b"ImageData"

    def test_cell_data_is_named_material(self, make_voxel_view, read_h5):
        """Expects the single cell array under ``VTKHDF/CellData/Material`` —
        the name ParaView colours by."""
        view = make_voxel_view()
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert "VTKHDF/CellData/Material" in data

    def test_semantic_tags_and_registry_are_written_when_present(
        self, make_voxel_view, make_view_grid, read_h5
    ):
        grid = make_view_grid(nx=8, ny=8, nz=8)
        registry = GeometryTagRegistry()
        registry.register("housing")
        registry.freeze()
        grid.geometry_tag_registry = registry
        grid.geometry_tag_map = GeometryTagMap((8, 8, 8), registry)
        grid.geometry_tag_map.data[1:3, 1:3, 1:3] = 1

        view = make_voxel_view(grid=grid, stop=(4, 4, 4))
        view.write_vtk()
        _, data = read_h5(view.filename)

        assert "VTKHDF/CellData/TagID" in data
        assert data["VTKHDF/CellData/TagID"].shape == (4, 4, 4)
        assert "VTKHDF/FieldData/geometry_tag_ids" in data
        assert "VTKHDF/FieldData/geometry_tag_names" in data

    def test_tag_data_is_absent_when_geometry_has_no_tags(self, make_voxel_view, read_h5):
        view = make_voxel_view()
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert "VTKHDF/CellData/TagID" not in data

    def test_cell_data_is_written_in_zyx_order(self, make_voxel_view, read_h5):
        """Expects the transpose the VTKHDF specification requires, as for
        snapshots."""
        view = make_voxel_view(stop=(2, 3, 4), nx=8, ny=8, nz=8)
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert data["VTKHDF/CellData/Material"].shape == (4, 3, 2)
        assert data["VTKHDF/CellData/Material"] == pytest.approx(view.material_data.T)

    def test_whole_extent_spans_the_view(self, make_voxel_view, read_h5):
        """Expects ``[0, nx, 0, ny, 0, nz]``."""
        view = make_voxel_view(start=(2, 2, 2), stop=(6, 6, 6), nx=12, ny=12, nz=12)
        view.write_vtk()
        attrs, _ = read_h5(view.filename)
        assert list(attrs["VTKHDF/WholeExtent"]) == [0, 4, 0, 4, 0, 4]

    def test_origin_and_spacing_reach_the_file(self, make_voxel_view, read_h5):
        """Expects both geometry attributes written, so the brick lattice lands
        in the right physical place."""
        view = make_voxel_view(start=(2, 2, 2), stop=(6, 6, 6), nx=12, ny=12, nz=12)
        view.write_vtk()
        attrs, _ = read_h5(view.filename)
        assert attrs["VTKHDF/Origin"] == pytest.approx([2 * DL] * 3)
        assert attrs["VTKHDF/Spacing"] == pytest.approx([DL] * 3)

    def test_metadata_is_attached(self, make_voxel_view, read_h5):
        """Expects the four core field-data entries alongside the cell data, so
        the file is self-describing."""
        view = make_voxel_view()
        view.write_vtk()
        _, data = read_h5(view.filename)
        names = {k.rsplit("/", 1)[-1] for k in data}
        assert {"gprMax_version", "dx_dy_dz", "nx_ny_nz", "material_ids"} <= names

    def test_material_values_survive_the_round_trip(self, make_voxel_view, make_view_grid, read_h5):
        """Expects the exact material IDs written, so a reader can map them
        back through the metadata table."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        g.solid[...] = 0
        g.solid[1, 1, 1] = 2
        view = make_voxel_view(grid=g, stop=(4, 4, 4))
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert data["VTKHDF/CellData/Material"][1, 1, 1] == 2

    def test_a_strided_view_writes_the_reduced_shape(self, make_voxel_view, read_h5):
        """Expects one brick per sampled cell, not per grid cell."""
        view = make_voxel_view(start=(0, 0, 0), stop=(8, 8, 8), step=(2, 2, 2), nx=12, ny=12, nz=12)
        view.write_vtk()
        _, data = read_h5(view.filename)
        assert data["VTKHDF/CellData/Material"].shape == (4, 4, 4)


class TestMpiVariant:
    def test_extends_the_serial_exporter(self):
        """Expects only the grid-view type to be overridden — the MPI variant
        adds halo awareness and nothing else."""
        assert issubclass(MPIGeometryViewVoxels, GeometryViewVoxels)
        overrides = {n for n in MPIGeometryViewVoxels.__dict__ if not n.startswith("__")}
        assert overrides <= {"GRID_VIEW_TYPE", "prep_vtk", "write_vtk"}

    def test_uses_an_mpi_grid_view(self, make_mpi_grid, make_materials):
        """Expects ``MPIGridView``, so the exporter's coordinates are trimmed
        to this rank's share of the domain."""
        from gprMax.geometry_outputs.grid_view import MPIGridView

        grid = make_mpi_grid(
            size=(8, 8, 8),
            negative_halo_offset=(0, 0, 0),
            arrays={"solid": np.ones((8, 8, 8), dtype=np.uint32)},
        )
        grid.materials = make_materials(2)
        view = MPIGeometryViewVoxels(0, 0, 0, 8, 8, 8, 1, 1, 1, "mpi", grid)
        assert isinstance(view.grid_view, MPIGridView)

    def test_origin_uses_global_coordinates(self, make_mpi_grid, make_materials):
        """Expects ``global_start * dl`` rather than the local start, so each
        rank's block lands in the right place in the shared file."""
        grid = make_mpi_grid(
            size=(8, 8, 8),
            negative_halo_offset=(0, 0, 0),
            origin=(10, 20, 30),
            arrays={"solid": np.ones((8, 8, 8), dtype=np.uint32)},
        )
        grid.materials = make_materials(2)
        grid.pmls = {
            "slabs": [],
            "thickness": dict.fromkeys(["x0", "y0", "z0", "xmax", "ymax", "zmax"], 0),
        }
        grid.nx, grid.ny, grid.nz = 8, 8, 8
        grid.rxs = []
        for name in ("hertziandipoles", "magneticdipoles", "voltagesources", "transmissionlines"):
            setattr(grid, name, [])
        view = MPIGeometryViewVoxels(0, 0, 0, 8, 8, 8, 1, 1, 1, "mpi", grid)
        view.set_filename()
        view.prep_vtk()
        assert view.origin == pytest.approx([10 * DL, 20 * DL, 30 * DL])


pytestmark = pytest.mark.unit
