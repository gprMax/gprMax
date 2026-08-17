"""``GeometryObject`` — exporting a model's raw arrays for later reuse.

Not a picture: a working copy. This writer dumps ``solid``, ``rigidE``,
``rigidH`` and ``ID`` into a plain ``.h5``, alongside a versioned
``_materials.json`` database. The HDF5 ``/material_keys`` dataset maps compact
array indices to stable database keys. The point is that an expensive antenna
geometry can be built once and then read straight back into later models with
``#geometry_objects_read`` without executing generated input commands.

**Material IDs are compacted.** ``write_hdf5`` calls ``initialise_materials()``
with filtering on, so the exported arrays are renumbered from zero over just
the materials present. An exported object therefore carries no reference to
materials that were only used elsewhere in the source model.

**``rigidE`` has 12 components and ``rigidH`` 6.** The byte-size arithmetic
folds them into a single factor of 18, which is worth naming because ``18``
appearing alone in a size calculation is otherwise unexplained.
"""

import json

import numpy as np
import pytest

from gprMax._version import __version__
from gprMax.geometry_outputs.geometry_objects import GeometryObject, MPIGeometryObject
from gprMax.materials import DispersiveMaterial

from .conftest import DL, DL_ANISO


@pytest.fixture
def make_geometry_object(make_view_grid):
    """Factory for a ``GeometryObject`` over a real grid."""

    def _make(start=(0, 0, 0), stop=(4, 4, 4), filename="geoobj", grid=None, **grid_kwargs):
        g = grid if grid is not None else make_view_grid(**grid_kwargs)
        return GeometryObject(g, *start, *stop, filename)

    return _make


class TestConstruction:
    def test_builds_a_grid_view_from_the_extents(self, make_geometry_object):
        """Expects the six coordinates handed to a ``GridView``; unlike the
        view exporters there is no stride argument at all."""
        obj = make_geometry_object(start=(1, 2, 3), stop=(5, 6, 7), nx=8, ny=8, nz=8)
        assert obj.grid_view.start.tolist() == [1, 2, 3]
        assert obj.grid_view.step.tolist() == [1, 1, 1]

    def test_the_hdf5_filename_takes_the_h5_suffix(self, make_geometry_object):
        """Expects ``<name>.h5`` for the array file."""
        assert make_geometry_object(filename="antenna").filename_hdf5.name == "antenna.h5"

    def test_the_materials_filename_is_suffixed_and_json(self, make_geometry_object):
        """Expects a schema-versioned JSON database beside the HDF5 file."""
        obj = make_geometry_object(filename="antenna")
        assert obj.filename_materials.name == "antenna_materials.json"

    def test_generated_database_name_is_safe_for_hash_commands(self, make_geometry_object):
        """Spaces in an API output basename do not make the pair unreadable."""

        obj = make_geometry_object(filename="antenna geometry")
        assert obj.filename_hdf5.name == "antenna geometry.h5"
        assert obj.filename_materials.name == "antenna_geometry_materials.json"

    def test_files_land_beside_the_input_file(self, make_geometry_object, outputs_config):
        """Expects the input file's directory, not the output directory —
        geometry objects are inputs to later runs."""
        obj = make_geometry_object()
        assert obj.filename_hdf5.parent == outputs_config.sim_config.input_file_path.parent

    def test_grid_is_reached_through_the_view(self, make_geometry_object, make_view_grid):
        """Expects the usual forwarding property."""
        g = make_view_grid()
        assert make_geometry_object(grid=g).grid is g


class TestSizeArithmetic:
    def test_solid_size_is_one_uint32_per_cell(self, make_geometry_object):
        """Expects ``nx·ny·nz · 4`` bytes."""
        obj = make_geometry_object(stop=(4, 4, 4), nx=8, ny=8, nz=8)
        assert obj.solidsize == 64 * 4

    def test_rigid_size_covers_both_arrays(self, make_geometry_object):
        """Expects ``18 · nx·ny·nz · 1`` bytes.

        The 18 is ``rigidE``'s 12 components plus ``rigidH``'s 6 — the two are
        written together and sized together."""
        obj = make_geometry_object(stop=(4, 4, 4), nx=8, ny=8, nz=8)
        assert obj.rigidsize == 18 * 64 * 1

    def test_id_size_uses_the_node_count(self, make_geometry_object):
        """Expects ``6 · (nx+1)(ny+1)(nz+1) · 4`` bytes — ``ID`` is
        node-centred, so it has one more entry per axis than ``solid``."""
        obj = make_geometry_object(stop=(4, 4, 4), nx=8, ny=8, nz=8)
        assert obj.IDsize == 6 * 125 * 4

    def test_total_is_the_sum_of_the_three(self, make_geometry_object):
        """Expects ``datawritesize`` to size the progress bar from everything
        that will be written."""
        obj = make_geometry_object(stop=(4, 4, 4), nx=8, ny=8, nz=8)
        assert obj.datawritesize == obj.solidsize + obj.rigidsize + obj.IDsize

    def test_sizes_are_floats(self, make_geometry_object):
        """Expects floats, since ``tqdm`` scales them into human units."""
        obj = make_geometry_object()
        assert isinstance(obj.solidsize, float)

    def test_sizes_track_the_view_not_the_grid(self, make_geometry_object):
        """Expects a partial view to report its own extent, so a small export
        from a large model does not claim the whole model's bytes."""
        small = make_geometry_object(stop=(2, 2, 2), nx=16, ny=16, nz=16)
        large = make_geometry_object(stop=(4, 4, 4), nx=16, ny=16, nz=16)
        assert large.solidsize == 8 * small.solidsize


class TestWriteMetadata:
    @pytest.fixture
    def written(self, make_geometry_object, tmp_path, null_pbar, make_view_grid):
        g = make_view_grid(nx=8, ny=8, nz=8, dl=DL_ANISO, materials=3)
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("A geometry object", null_pbar)
        return obj

    def test_records_the_gprmax_version(self, written, read_h5):
        """Expects the writing version stamped at the root."""
        attrs, _ = read_h5(written.filename_hdf5)
        assert attrs["gprMax"] == __version__

    def test_records_the_title(self, written, read_h5):
        """Expects the model title as given."""
        attrs, _ = read_h5(written.filename_hdf5)
        assert attrs["Title"] == "A geometry object"

    def test_records_the_discretisation(self, written, read_h5):
        """Expects ``dx_dy_dz`` per axis.

        The reader checks this against the importing model's own spacing and
        refuses to build if they differ — a geometry object is a fixed lattice
        of cells, not a scalable shape."""
        attrs, _ = read_h5(written.filename_hdf5)
        assert attrs["dx_dy_dz"] == pytest.approx(list(DL_ANISO))


class TestWriteHdf5Arrays:
    @pytest.fixture
    def written(self, make_geometry_object, null_pbar, make_view_grid):
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        return obj

    def test_writes_all_four_arrays(self, written, read_h5):
        """Expects ``/data``, ``/rigidE``, ``/rigidH`` and ``/ID`` — everything
        needed to rebuild the geometry without re-running the build step."""
        _, data = read_h5(written.filename_hdf5)
        assert set(data) == {"data", "rigidE", "rigidH", "ID", "material_keys"}

    def test_data_is_cell_shaped(self, written, read_h5):
        """Expects ``(nx, ny, nz)`` for the solid array."""
        _, data = read_h5(written.filename_hdf5)
        assert data["data"].shape == (4, 4, 4)

    def test_data_is_int16(self, written, read_h5):
        """Expects a *signed* type, because ``-1`` means "background, build
        nothing here" — an unsigned array could not express that."""
        _, data = read_h5(written.filename_hdf5)
        assert data["data"].dtype == np.int16

    def test_id_is_node_shaped_with_six_components(self, written, read_h5):
        """Expects ``(6, nx+1, ny+1, nz+1)``."""
        _, data = read_h5(written.filename_hdf5)
        assert data["ID"].shape == (6, 5, 5, 5)

    def test_rigid_arrays_keep_their_component_counts(self, written, read_h5):
        """Expects 12 components for ``rigidE`` and 6 for ``rigidH``, matching
        the 18 in the byte arithmetic."""
        _, data = read_h5(written.filename_hdf5)
        assert data["rigidE"].shape == (12, 4, 4, 4)
        assert data["rigidH"].shape == (6, 4, 4, 4)

    def test_arrays_are_not_transposed(self, written, read_h5):
        """Expects plain ``(x, y, z)`` ordering — this is a raw HDF5 file, not
        VTKHDF, so none of the ZYX reordering applies."""
        _, data = read_h5(written.filename_hdf5)
        assert data["data"].shape == tuple(written.grid_view.size)

    def test_material_ids_are_compacted(
        self, make_geometry_object, null_pbar, make_view_grid, read_h5
    ):
        """Expects renumbering from zero over the materials actually present.

        A view containing only material 2 exports it as 0, so the file's
        indices line up with its own materials list."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        g.solid[...] = 2
        g.ID[...] = 2
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        _, data = read_h5(obj.filename_hdf5)
        assert set(np.unique(data["data"])) == {0}

    def test_progress_is_reported_in_three_steps(
        self, make_geometry_object, null_pbar, make_view_grid
    ):
        """Expects one update after the solid array, one after both rigid
        arrays, and one after ``ID``."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        assert null_pbar.updates == [obj.solidsize, obj.rigidsize, obj.IDsize]

    def test_reported_bytes_total_the_declared_size(
        self, make_geometry_object, null_pbar, make_view_grid
    ):
        """Expects the progress total to match ``datawritesize``."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        assert null_pbar.total == obj.datawritesize


class TestMaterialsFile:
    @staticmethod
    def _entries(obj):
        return json.loads(obj.filename_materials.read_text())["materials"]

    def test_writes_one_entry_per_material(self, make_geometry_object, null_pbar, make_view_grid):
        """Expects one JSON entry for each material in the compacted list."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        g.ID[...] = 1
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        assert len(self._entries(obj)) == 1

    def test_writes_versioned_database_schema(
        self, make_geometry_object, null_pbar, make_view_grid
    ):
        """Expects the public material-database schema rather than executable input."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=2)
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        document = json.loads(obj.filename_materials.read_text())
        assert document["schema"] == "gprMax-material-database"
        assert document["schema_version"] == 1

    def test_the_constitutive_parameters_are_written(
        self, make_geometry_object, null_pbar, make_view_grid
    ):
        """Expects permittivity, conductivity, permeability and magnetic loss
        in that order, followed by the name."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=2)
        # ``ID`` and ``solid`` both initialise to 1 (free space in a full
        # model); these grids define only material 0.
        g.ID[...] = 0
        g.solid[...] = 0
        g.materials[0].er, g.materials[0].se = 4.5, 0.01
        g.materials[0].mr, g.materials[0].sm = 1.5, 0.02
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        entry = next(iter(self._entries(obj).values()))
        assert entry["base"] == {
            "relative_permittivity": 4.5,
            "electric_conductivity_s_per_m": 0.01,
            "relative_permeability": 1.5,
            "magnetic_conductivity_s_per_m": 0.02,
        }

    @pytest.mark.parametrize(
        "model",
        ["debye", "drude"],
    )
    def test_dispersive_materials_get_a_second_line(
        self,
        make_geometry_object,
        null_pbar,
        make_view_grid,
        make_dispersive,
        model,
    ):
        """Expects the dispersion model and pole to survive serialisation."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=2)
        # ``ID`` and ``solid`` both initialise to 1 (free space in a full
        # model); these grids define only material 0.
        g.ID[...] = 0
        g.solid[...] = 0
        disp = make_dispersive(ID="soil", numID=0, model=model, poles=[(2.0, 1e-9, 0.5)])
        g.materials = [disp]
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        entry = next(iter(self._entries(obj).values()))
        assert entry["model"] == model
        assert len(entry["poles"]) == 1

    def test_lorentz_dispersion_writes_three_values_per_pole(
        self, make_geometry_object, null_pbar, make_view_grid, make_dispersive
    ):
        """Expects ``deltaer``, ``tau`` and ``alpha`` for a Lorentz pole —
        one more than Debye needs.

        """
        g = make_view_grid(nx=8, ny=8, nz=8, materials=2)
        # ``ID`` and ``solid`` both initialise to 1 (free space in a full
        # model); these grids define only material 0.
        g.ID[...] = 0
        g.solid[...] = 0
        disp = make_dispersive(ID="soil", numID=0, model="lorentz", poles=[(2.0, 1e-9, 0.5)])
        g.materials = [disp]
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        entry = next(iter(self._entries(obj).values()))
        assert entry["model"] == "lorentz"
        assert set(entry["poles"][0]) == {
            "relative_permittivity_difference",
            "resonance_frequency_hz",
            "damping_coefficient_per_s",
        }

    def test_the_material_name_ends_each_dispersion_line(
        self, make_geometry_object, null_pbar, make_view_grid, make_dispersive
    ):
        """Expects the material ID appended, so the command binds to the right
        material when re-parsed."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=2)
        # ``ID`` and ``solid`` both initialise to 1 (free space in a full
        # model); these grids define only material 0.
        g.ID[...] = 0
        g.solid[...] = 0
        disp = make_dispersive(ID="soil", numID=0, model="debye", poles=[(2.0, 1e-9, 0.0)])
        g.materials = [disp]
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        entry = next(iter(self._entries(obj).values()))
        assert entry["metadata"]["original_id"] == "soil"

    def test_non_dispersive_materials_get_no_second_line(
        self, make_geometry_object, null_pbar, make_view_grid
    ):
        """Expects a plain material to produce exactly one line."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=2)
        # ``ID`` and ``solid`` both initialise to 1 (free space in a full
        # model); these grids define only material 0.
        g.ID[...] = 0
        g.solid[...] = 0
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        entry = next(iter(self._entries(obj).values()))
        assert entry["model"] == "constant"
        assert "poles" not in entry

    def test_materials_are_written_in_compacted_order(
        self, make_geometry_object, null_pbar, make_view_grid
    ):
        """Expects the file's line order to match the index the arrays use, so
        line *n* describes material *n*."""
        g = make_view_grid(nx=8, ny=8, nz=8, materials=3)
        obj = make_geometry_object(grid=g, stop=(4, 4, 4))
        obj.write_hdf5("t", null_pbar)
        names = [entry["metadata"]["original_id"] for entry in self._entries(obj).values()]
        assert names == [m.ID for m in obj.grid_view.materials]


class TestMpiVariant:
    def test_extends_the_serial_writer(self):
        """Expects only the grid-view type and the write method to be
        overridden."""
        assert issubclass(MPIGeometryObject, GeometryObject)
        overrides = {n for n in MPIGeometryObject.__dict__ if not n.startswith("__")}
        assert overrides <= {
            "GRID_VIEW_TYPE",
            "_merge_negative_rigid_halos",
            "write_hdf5",
        }

    def test_uses_an_mpi_grid_view(self, make_mpi_grid, make_materials):
        """Expects ``MPIGridView``, so each rank exports its own share."""
        from gprMax.geometry_outputs.grid_view import MPIGridView

        grid = make_mpi_grid(size=(8, 8, 8), negative_halo_offset=(0, 0, 0))
        grid.materials = make_materials(2)
        obj = MPIGeometryObject(grid, 0, 0, 0, 8, 8, 8, "mpi")
        assert isinstance(obj.grid_view, MPIGridView)

    def test_size_arithmetic_is_inherited(self, make_mpi_grid, make_materials):
        """Expects the byte counts to be computed by the base constructor,
        unchanged."""
        grid = make_mpi_grid(size=(8, 8, 8), negative_halo_offset=(0, 0, 0))
        grid.materials = make_materials(2)
        obj = MPIGeometryObject(grid, 0, 0, 0, 8, 8, 8, "mpi")
        assert obj.solidsize == 512 * 4

    def test_rigid_halo_merge_is_part_of_parallel_export(self):
        """The parallel writer must reconcile redundant rigid markers.

        Actual multi-rank round trips are exercised by the MPI integration
        suite. Keep a small source-level guard here because ordinary pytest
        installations commonly use an h5py build without parallel HDF5.
        """
        import inspect

        source = inspect.getsource(MPIGeometryObject.write_hdf5)
        assert "_merge_negative_rigid_halos(rigidE" in source
        assert "_merge_negative_rigid_halos(rigidH" in source

    def test_the_parallel_write_needs_parallel_hdf5(self):
        """Expects ``MPIGeometryObject.write_hdf5`` to open with
        ``driver="mpio"``.

        That is unavailable here — ``h5py.get_config().mpi`` is ``False`` — so
        the write path itself is out of reach in this environment. Recorded
        explicitly rather than left as an unexplained coverage hole."""
        import inspect

        source = inspect.getsource(MPIGeometryObject.write_hdf5)
        assert 'driver="mpio"' in source


pytestmark = pytest.mark.unit
