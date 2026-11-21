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

"""``ReadGeometryObject`` — importing a previously exported geometry.

The other half of the round trip ``test_geometry_objects.py`` writes. Given an
exported ``.h5``, this reader slices out the part that belongs to the importing
grid and writes it into that grid's ``solid``, ``rigidE``, ``rigidH`` and
``ID`` arrays.

**There are two files with this name.** PR 8 tested
``gprMax/user_objects/cmds_geometry/geometry_objects_read.py`` — the
``#geometry_objects_read`` *command*, which parses the materials text file and
drives the import. This is
``gprMax/geometry_outputs/geometry_objects_read.py``, the *file reader* that
command delegates to. Same idea, different layer, identical filename.

**Materials are re-based on import.** The importing model already has its own
materials, so the ``material_id_map`` array maps every file-local material
index to its numID in the importing grid. The caller builds this map by
matching material IDs by name.

**Ranks that do not overlap get no view at all.** Under MPI, a rank whose local
domain does not intersect the object's bounding box sets ``grid_view = None``
and every read method short-circuits. It still calls ``comm.Split`` first,
because ``MPIGridView`` would call it on the other ranks and an unmatched
collective deadlocks.
"""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from gprMax.geometry_outputs.geometry_objects_read import ReadGeometryObject

from .conftest import DL, DL_ANISO

# Identity material-id map large enough for any data in this suite.
# material_id_map[i] == i, so file-local index i maps to numID i.
ID_MAP = np.arange(100, dtype=np.int32)


@pytest.fixture
def geometry_file(tmp_path):
    """Write a geometry-object ``.h5`` the way ``GeometryObject`` does.

    Values are distinct per cell so a misplaced slice cannot go unnoticed.
    """

    def _make(
        name="obj.h5",
        shape=(3, 3, 3),
        dl=(DL, DL, DL),
        include_id=True,
        include_rigid=True,
        data_dtype=np.int16,
    ):
        path = tmp_path / name
        nx, ny, nz = shape
        with h5py.File(path, "w") as f:
            f.attrs["gprMax"] = "test"
            f.attrs["Title"] = "fixture"
            f.attrs["dx_dy_dz"] = dl
            f["/data"] = np.arange(nx * ny * nz, dtype=data_dtype).reshape(shape)
            if include_rigid:
                f["/rigidE"] = np.ones((12, nx, ny, nz), dtype=np.int8)
                f["/rigidH"] = np.full((6, nx, ny, nz), 2, dtype=np.int8)
            if include_id:
                f["/ID"] = np.full((6, nx + 1, ny + 1, nz + 1), 3, dtype=np.uint32)
        return path

    return _make


@pytest.fixture
def target_grid(make_view_grid):
    """A grid to import into, with its arrays cleared so writes are visible."""

    def _make(nx=8, ny=8, nz=8, dl=DL):
        g = make_view_grid(nx=nx, ny=ny, nz=nz, dl=dl)
        g.solid[...] = 0
        g.ID[...] = 0
        g.rigidE[...] = 0
        g.rigidH[...] = 0
        return g

    return _make


class TestConstruction:
    def test_opens_the_file(self, geometry_file, target_grid):
        """Expects a live h5py handle for the duration of the read."""
        with ReadGeometryObject(geometry_file(), target_grid(), np.zeros(3, np.int32), ID_MAP) as r:
            assert r.file_handler.id.valid

    def test_derives_the_extent_from_the_data_shape(self, geometry_file, target_grid):
        """Expects ``stop = start + data.shape`` — the caller supplies only the
        insertion point, and the file itself says how big the object is."""
        start = np.array([1, 2, 3], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(3, 4, 5)), target_grid(), start, ID_MAP) as r:
            assert r.grid_view.stop.tolist() == [4, 6, 8]

    def test_builds_a_serial_grid_view_for_a_plain_grid(self, geometry_file, target_grid):
        """Expects a ``GridView`` rather than the MPI variant."""
        from gprMax.geometry_outputs.grid_view import GridView

        with ReadGeometryObject(geometry_file(), target_grid(), np.zeros(3, np.int32), ID_MAP) as r:
            assert type(r.grid_view) is GridView

    def test_stores_the_material_map(self, geometry_file, target_grid):
        """Expects the material_id_map stored for use by every read."""
        seven_map = np.arange(7, dtype=np.int32)
        with ReadGeometryObject(
            geometry_file(), target_grid(), np.zeros(3, np.int32), seven_map
        ) as r:
            assert len(r.material_id_map) == 7
            assert np.array_equal(r.material_id_map, seven_map)

    def test_is_a_context_manager(self, geometry_file, target_grid):
        """Expects ``__enter__`` to return the reader itself."""
        reader = ReadGeometryObject(geometry_file(), target_grid(), np.zeros(3, np.int32), ID_MAP)
        with reader as r:
            assert r is reader

    def test_exiting_closes_the_file(self, geometry_file, target_grid):
        """Expects the handle released on exit, so the file is not left locked."""
        with ReadGeometryObject(geometry_file(), target_grid(), np.zeros(3, np.int32), ID_MAP) as r:
            handler = r.file_handler
        assert not handler.id.valid

    def test_close_can_be_called_directly(self, geometry_file, target_grid):
        """Expects the same effect without the ``with`` block, for callers that
        manage the lifetime themselves."""
        reader = ReadGeometryObject(geometry_file(), target_grid(), np.zeros(3, np.int32), ID_MAP)
        reader.close()
        assert not reader.file_handler.id.valid


class TestValidation:
    def test_matching_discretisation_is_accepted(self, geometry_file, target_grid):
        """Expects ``True`` when the file's spacing equals the grid's."""
        with ReadGeometryObject(
            geometry_file(dl=(DL, DL, DL)), target_grid(dl=DL), np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert r.has_valid_discritisation()

    def test_mismatched_discretisation_is_rejected(self, geometry_file, target_grid):
        """Expects ``False`` — a geometry object is a fixed lattice of cells,
        so importing it into a differently-discretised model would silently
        change its physical size."""
        with ReadGeometryObject(
            geometry_file(dl=(2 * DL, 2 * DL, 2 * DL)),
            target_grid(dl=DL),
            np.zeros(3, np.int32),
            ID_MAP,
        ) as r:
            assert not r.has_valid_discritisation()

    def test_a_single_mismatched_axis_is_rejected(self, geometry_file, target_grid):
        """Expects all three axes checked, not just the first."""
        with ReadGeometryObject(
            geometry_file(dl=(DL, DL, 2 * DL)), target_grid(dl=DL), np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert not r.has_valid_discritisation()

    def test_detects_an_id_array(self, geometry_file, target_grid):
        """Expects ``True`` when the file carries ``/ID``.

        The caller uses this to choose between a fast path that reads the
        stored arrays directly and a slow one that rebuilds them from ``data``
        with the voxel builder."""
        with ReadGeometryObject(
            geometry_file(include_id=True), target_grid(), np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert r.has_ID_array()

    def test_detects_a_missing_id_array(self, geometry_file, target_grid):
        """Expects ``False`` for an older file with only ``/data``."""
        with ReadGeometryObject(
            geometry_file(include_id=False), target_grid(), np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert not r.has_ID_array()

    def test_detects_rigid_arrays(self, geometry_file, target_grid):
        """Expects ``True`` only when *both* rigid arrays are present."""
        with ReadGeometryObject(
            geometry_file(include_rigid=True), target_grid(), np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert r.has_rigid_arrays()

    def test_detects_missing_rigid_arrays(self, geometry_file, target_grid):
        """Expects ``False`` when they are absent."""
        with ReadGeometryObject(
            geometry_file(include_rigid=False), target_grid(), np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert not r.has_rigid_arrays()


class TestReadData:
    def test_writes_the_solid_array_into_the_grid(self, geometry_file, target_grid):
        """Expects the imported values to land in ``grid.solid`` at the
        requested position."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            r.read_data()
        assert g.solid[0, 0, 0] == 0
        assert g.solid[2, 2, 2] == 26

    def test_places_the_object_at_the_requested_start(self, geometry_file, target_grid):
        """Expects the object offset into the grid, leaving everything before
        it untouched."""
        g = target_grid()
        start = np.array([2, 2, 2], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(3, 3, 3)), g, start, ID_MAP) as r:
            r.read_data()
        assert g.solid[4, 4, 4] == 26
        assert g.solid[0, 0, 0] == 0

    def test_shifts_material_ids_by_the_material_map(self, geometry_file, target_grid):
        """Expects every imported ID mapped through ``material_id_map``,
        so it names the material the importing model expects."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP + 10
        ) as r:
            r.read_data()
        assert g.solid[0, 0, 0] == 10
        assert g.solid[2, 2, 2] == 36

    def test_get_data_returns_without_writing(self, geometry_file, target_grid):
        """Expects the array back and the grid untouched — the caller uses this
        when it needs to rebuild the rigid and ID arrays itself."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            data = r.get_data()
        assert data.shape == (3, 3, 3)
        assert not np.any(g.solid)

    def test_get_data_reports_the_matching_local_start(self, geometry_file, target_grid):
        """The fallback voxel builder must place the first returned cell at
        the reader's assignment start (which can differ on an MPI rank)."""
        g = target_grid()
        start = np.array([2, 3, 4], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(3, 3, 3)), g, start, ID_MAP) as r:
            r.get_data()
            np.testing.assert_array_equal(r.get_local_data_start(), start)

    def test_get_data_applies_the_material_map(self, geometry_file, target_grid):
        """Expects the remapped values — ``get_data`` applies
        ``material_id_map``, unlike the old ``num_existing_materials``
        offset that only ``read_data`` applied.

        With an identity map, file values and returned values match."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            data = r.get_data()
        assert data[0, 0, 0] == 0  # identity: file 0 → numID 0

    def test_unsigned_file_is_mapped_to_wide_signed_ids(self, geometry_file, target_grid):
        """Unsigned file indices remain valid indices, while the returned
        global IDs have room for large catalogues and the -1 sentinel."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3), data_dtype=np.uint16), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            data = r.get_data()
        assert data.dtype == np.int32

    def test_compact_file_is_mapped_to_wide_signed_ids(self, geometry_file, target_grid):
        """Compact on-disk indices must not constrain target global IDs."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3), data_dtype=np.int16), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert r.get_data().dtype == np.int32


class TestReadRigidAndId:
    def test_reads_rigid_e(self, geometry_file, target_grid):
        """Expects all 12 components written into the grid's ``rigidE``."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            r.read_rigidE()
        assert np.all(g.rigidE[:, :3, :3, :3] == 1)

    def test_reads_rigid_h(self, geometry_file, target_grid):
        """Expects all 6 components written into ``rigidH``."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            r.read_rigidH()
        assert np.all(g.rigidH[:, :3, :3, :3] == 2)

    def test_rigid_arrays_are_not_material_shifted(self, geometry_file, target_grid):
        """Expects the mapping *not* applied — the rigid arrays hold flags, not
        material indices."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP + 10
        ) as r:
            r.read_rigidE()
        assert np.all(g.rigidE[:, :3, :3, :3] == 1)

    def test_reads_id_with_the_inclusive_bound(self, geometry_file, target_grid):
        """Expects ``(nx+1)`` nodes per axis, because ``ID`` is node-centred —
        the reader asks for a read slice with ``upper_bound_exclusive=False``."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            r.read_ID()
        assert np.all(g.ID[:, :4, :4, :4] == 3)

    def test_id_is_material_shifted(self, geometry_file, target_grid):
        """Expects the mapping applied here, unlike the rigid arrays — ``ID``
        does hold material indices."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP + 10
        ) as r:
            r.read_ID()
        assert np.all(g.ID[:, :4, :4, :4] == 13)

    def test_rigid_and_id_land_at_the_requested_start(self, geometry_file, target_grid):
        """Expects the same offsetting as ``read_data``, so all four arrays
        describe the same region."""
        g = target_grid()
        start = np.array([2, 2, 2], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(3, 3, 3)), g, start, ID_MAP) as r:
            r.read_rigidE()
        assert np.all(g.rigidE[:, 2:5, 2:5, 2:5] == 1)
        assert not np.any(g.rigidE[:, :2, :2, :2])

    def test_a_full_import_writes_all_four_arrays(self, geometry_file, target_grid):
        """Expects the complete fast path to reconstruct the geometry without
        re-running the voxel builder."""
        g = target_grid()
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), g, np.zeros(3, np.int32), ID_MAP
        ) as r:
            r.read_data()
            r.read_ID()
            r.read_rigidE()
            r.read_rigidH()
        assert g.solid[2, 2, 2] == 26
        assert g.ID[0, 0, 0, 0] == 3
        assert g.rigidE[0, 0, 0, 0] == 1
        assert g.rigidH[0, 0, 0, 0] == 2


class TestClippedPlacement:
    """A placement whose bounding box leaves the model domain.

    The toolboxes instruct exactly this: STLtoVoxel/STEPtoVoxel print the
    file's origin as the suggested ``p1``, and that origin is negative for
    any model with negative coordinates. The part of the file outside the
    domain is dropped - matching the far-boundary truncation the voxel
    builder has always applied - and a file that misses the domain entirely
    is an error rather than a wrapped or shifted write.
    """

    def test_the_view_is_clipped_to_the_domain(self, geometry_file, target_grid):
        """Expects the view itself to carry the clipped extent, so every
        slice it produces stays inside the grid's arrays."""
        start = np.array([-2, -2, -2], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(4, 4, 4)), target_grid(), start, ID_MAP) as r:
            assert r.grid_view.start.tolist() == [0, 0, 0]
            assert r.grid_view.stop.tolist() == [2, 2, 2]

    def test_a_negative_start_clips_the_in_domain_part(self, geometry_file, target_grid):
        """A 4-wide file placed two cells below the domain keeps the value
        identity of its in-domain part: file slice 2 lands at grid cell 0,
        and nothing is written beyond the object's true extent."""
        g = target_grid()
        start = np.array([-2, -2, -2], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(4, 4, 4)), g, start, ID_MAP) as r:
            r.read_data()
        # /data holds arange, so file cell (2, 2, 2) carries 26.
        assert g.solid[0, 0, 0] == 42  # file cell (2, 2, 2)
        assert g.solid[1, 1, 1] == 63  # file cell (3, 3, 3)
        assert int(np.count_nonzero(g.solid)) == 2 * 2 * 2
        assert g.solid[2, 0, 0] == 0
        assert g.solid[7, 7, 7] == 0

    def test_a_negative_start_clips_the_id_array(self, geometry_file, target_grid):
        """The node-centred ``ID`` planes follow the same clip: the file's
        planes 2, 3 and 4 land on grid planes 0, 1 and 2 (the trailing edge
        of the last placed cell), and plane 3 keeps its existing value."""
        g = target_grid()
        start = np.array([-2, -2, -2], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(4, 4, 4)), g, start, ID_MAP) as r:
            r.read_ID()
        assert np.all(g.ID[:, :3, :3, :3] == 3)
        assert np.all(g.ID[:, 3, :, :] == 0)

    def test_rigid_arrays_follow_the_same_clip(self, geometry_file, target_grid):
        g = target_grid()
        start = np.array([-2, -2, -2], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(4, 4, 4)), g, start, ID_MAP) as r:
            r.read_rigidE()
        assert np.all(g.rigidE[:, :2, :2, :2] == 1)
        assert not np.any(g.rigidE[:, 2, :, :])

    def test_get_data_and_local_start_are_clipped_together(self, geometry_file, target_grid):
        """The voxel-builder contract: the first returned cell must pair with
        the reported start, so ``build_voxels_from_array`` places the file's
        in-domain slices at the domain edge rather than the origin face."""
        g = target_grid()
        start = np.array([-2, -2, -2], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(4, 4, 4)), g, start, ID_MAP) as r:
            data = r.get_data()
            np.testing.assert_array_equal(r.get_local_data_start(), [0, 0, 0])
        assert data.shape == (2, 2, 2)
        assert data[0, 0, 0] == 42  # file cell (2, 2, 2), identity map

    def test_an_overhanging_file_clips_at_the_far_boundary(self, geometry_file, target_grid):
        """A file whose stop runs past the domain used to crash on the shape
        mismatch between the full file read and the wrapped write region."""
        g = target_grid()
        start = np.array([6, 0, 0], dtype=np.int32)
        with ReadGeometryObject(geometry_file(shape=(4, 4, 4)), g, start, ID_MAP) as r:
            r.read_data()
        assert g.solid[7, 0, 0] == 16  # file cell (1, 0, 0)
        assert g.solid[6, 3, 3] == 15  # file cell (0, 3, 3)
        # Only the two in-domain x-slices of the file are written, with
        # their own values (/data is arange over the 4x4x4 block).
        np.testing.assert_array_equal(g.solid[6:8, 0:4, 0:4], np.arange(2 * 4 * 4).reshape(2, 4, 4))
        assert g.solid[5, 0, 0] == 0

    def test_a_file_that_misses_the_domain_raises(self, geometry_file, target_grid):
        """A file entirely below the domain used to write its cells into the
        middle of the grid through numpy's negative-start wrap-around."""
        g = target_grid()
        start = np.array([-6, 0, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="lies entirely outside"):
            with ReadGeometryObject(geometry_file(shape=(4, 4, 4)), g, start, ID_MAP) as r:
                r.read_data()
        assert not np.any(g.solid)

    def test_a_2d_invariant_placement_outside_the_axis_raises(self, geometry_file, target_grid):
        """In 2D the invariant axis carries the object's broadcast rows; a
        placement below it has no in-domain row and must raise instead of
        writing nothing."""
        g = target_grid()
        start = np.array([0, 0, -1], dtype=np.int32)
        with pytest.raises(ValueError, match="does not"):
            with ReadGeometryObject(
                geometry_file(shape=(3, 3, 4)),
                g,
                start,
                ID_MAP,
                invariant_axis=2,
                target_invariant_size=1,
            ) as r:
                r.read_data()
        assert not np.any(g.solid)


class TestRoundTrip:
    def test_an_exported_object_reads_back_unchanged(
        self, make_view_grid, null_pbar, target_grid, outputs_config
    ):
        """Expects a full write-then-read cycle to reproduce the source
        geometry.

        This is the property the whole pair exists for: build an expensive
        geometry once, export it, and get exactly the same cells back in a
        later model."""
        from gprMax.geometry_outputs.geometry_objects import GeometryObject

        source = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        source.solid[...] = 0
        source.ID[...] = 0
        source.solid[1, 2, 3] = 1
        source.ID[:, 1, 2, 3] = 1
        obj = GeometryObject(source, 0, 0, 0, 4, 4, 4, "roundtrip")
        obj.write_hdf5("t", null_pbar)

        target = target_grid()
        with ReadGeometryObject(obj.filename_hdf5, target, np.zeros(3, np.int32), ID_MAP) as r:
            assert r.has_valid_discritisation()
            r.read_data()
            r.read_ID()
        assert target.solid[1, 2, 3] == 1
        assert target.ID[0, 1, 2, 3] == 1

    def test_the_round_trip_respects_a_material_offset(
        self, make_view_grid, null_pbar, target_grid
    ):
        """Expects every imported material shifted, so the object's materials
        sit after the host model's own."""
        from gprMax.geometry_outputs.geometry_objects import GeometryObject

        source = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        source.solid[...] = 0
        source.ID[...] = 0
        source.solid[1, 1, 1] = 1
        source.ID[:, 1, 1, 1] = 1
        obj = GeometryObject(source, 0, 0, 0, 4, 4, 4, "offset")
        obj.write_hdf5("t", null_pbar)

        target = target_grid()
        with ReadGeometryObject(obj.filename_hdf5, target, np.zeros(3, np.int32), ID_MAP + 5) as r:
            r.read_data()
        assert target.solid[1, 1, 1] == 6
        assert target.solid[0, 0, 0] == 5


class TestMpiPaths:
    @pytest.fixture
    def overlapping_grid(self, make_mpi_grid):
        arrays = {
            "solid": np.zeros((8, 8, 8), dtype=np.uint32),
            "ID": np.zeros((6, 9, 9, 9), dtype=np.uint32),
            "rigidE": np.zeros((12, 8, 8, 8), dtype=np.int8),
            "rigidH": np.zeros((6, 8, 8, 8), dtype=np.int8),
        }
        return make_mpi_grid(
            size=(8, 8, 8), negative_halo_offset=(0, 0, 0), origin=(0, 0, 0), arrays=arrays
        )

    @pytest.fixture
    def as_mpi_grid(self, overlapping_grid):
        """Mark the fake grid as distributed through the grid capability flag."""
        overlapping_grid.is_distributed = True
        return overlapping_grid

    def test_an_overlapping_rank_gets_an_mpi_grid_view(self, geometry_file, as_mpi_grid):
        """Expects ``MPIGridView`` when this rank's domain intersects the
        object's bounding box."""
        from gprMax.geometry_outputs.grid_view import MPIGridView

        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), as_mpi_grid, np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert isinstance(r.grid_view, MPIGridView)

    def test_a_non_overlapping_rank_gets_no_view(self, geometry_file, as_mpi_grid):
        """Expects ``grid_view = None`` when this rank owns none of the object.

        The rank still calls ``comm.Split(MPI.UNDEFINED)`` first: the ranks
        that *do* overlap will split their communicator inside ``MPIGridView``,
        and an unmatched collective would hang every one of them."""
        as_mpi_grid.local_bounds_overlap_grid = lambda start, stop: False
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), as_mpi_grid, np.zeros(3, np.int32), ID_MAP
        ) as r:
            assert r.grid_view is None

    @pytest.fixture
    def viewless_reader(self, geometry_file, as_mpi_grid):
        as_mpi_grid.local_bounds_overlap_grid = lambda start, stop: False
        with ReadGeometryObject(
            geometry_file(shape=(3, 3, 3)), as_mpi_grid, np.zeros(3, np.int32), ID_MAP
        ) as r:
            yield r

    def test_validation_passes_trivially_without_a_view(self, viewless_reader):
        """Expects ``True`` — a rank with nothing to read cannot disagree about
        the discretisation, and returning ``False`` would abort the run for
        everyone."""
        assert viewless_reader.has_valid_discritisation()

    @pytest.mark.parametrize("method", ["read_data", "read_rigidE", "read_rigidH", "read_ID"])
    def test_every_read_short_circuits_without_a_view(self, viewless_reader, method):
        """Expects each reader to return immediately rather than raise.
        (4 parameter sets)"""
        assert getattr(viewless_reader, method)() is None

    def test_get_data_returns_none_without_a_view(self, viewless_reader):
        """Expects ``None`` rather than an empty array, so the caller can tell
        "nothing for this rank" from "an empty object"."""
        assert viewless_reader.get_data() is None


pytestmark = pytest.mark.unit
