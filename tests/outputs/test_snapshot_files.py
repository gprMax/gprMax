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

"""Writing snapshots to disk — both formats, round-tripped through ``tmp_path``.

Every test here writes a real file and reads it back with ``h5py``. Nothing is
mocked, because the on-disk layout *is* the contract: dataset names, shapes,
dtypes and attributes are what ParaView and every downstream analysis script
depend on, and a mock-based test would confirm only that a method was called.

Two formats share one dispatcher:

- ``.h5`` — a flat HDF5 file with one root dataset per requested component
- ``.vtkhdf`` — VTK ImageData, written through
  ``gprMax/vtkhdf_filehandlers/``, with cell data under ``VTKHDF/CellData``
  and the geometry carried in root attributes

``vtkhdf_filehandlers`` belongs to a later PR; here it is exercised
transitively, which is the point — these tests establish what the current
layout is so that PR 11 can refactor the writer with something to check
against.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from gprMax._version import __version__
from gprMax.snapshots import Snapshot, save_snapshots

from .conftest import DL, DL_ANISO, DT, FIELDS

ALL_OUTPUTS = {name: True for name in FIELDS}
NO_OUTPUTS = {name: False for name in FIELDS}


@pytest.fixture
def written_snapshot(make_view_grid, tmp_path, null_pbar):
    """Build a snapshot, store its fields, write it, and hand back the path."""

    def _write(
        fileext=".h5",
        outputs=None,
        start=(0, 0, 0),
        stop=(4, 4, 4),
        step=(1, 1, 1),
        time=10,
        dl=DL,
        name="snap",
        grid=None,
    ):
        g = grid if grid is not None else make_view_grid(nx=8, ny=8, nz=8, dl=dl)
        snap = Snapshot(
            *start,
            *stop,
            *step,
            time,
            str(tmp_path / name),
            fileext,
            dict(ALL_OUTPUTS if outputs is None else outputs),
            g,
        )
        snap.initialise_snapfields()
        snap.store()
        snap.write_file(null_pbar)
        return snap

    return _write


class TestWriteFileDispatch:
    def test_h5_extension_produces_an_hdf5_file(self, written_snapshot):
        """Expects ``.h5`` to route to ``write_hdf5``, leaving a readable
        file."""
        snap = written_snapshot(fileext=".h5")
        assert snap.filename.exists()
        with h5py.File(snap.filename, "r") as f:
            assert "Ex" in f

    def test_vtkhdf_extension_produces_a_vtk_file(self, written_snapshot):
        """Expects ``.vtkhdf`` to route to ``write_vtk``, producing the VTKHDF
        group structure rather than root datasets."""
        snap = written_snapshot(fileext=".vtkhdf")
        with h5py.File(snap.filename, "r") as f:
            assert "VTKHDF" in f
            assert "Ex" not in f

    def test_the_two_formats_write_different_layouts(self, written_snapshot, read_h5):
        """Expects the same field data to land at different paths in the two
        formats — root-level in HDF5, under ``VTKHDF/CellData`` in VTK."""
        h5 = written_snapshot(fileext=".h5", name="a")
        vtk = written_snapshot(fileext=".vtkhdf", name="b")
        _, h5_data = read_h5(h5.filename)
        _, vtk_data = read_h5(vtk.filename)
        assert "Ex" in h5_data
        assert "VTKHDF/CellData/Ex" in vtk_data


class TestHdf5Attributes:
    def test_records_the_gprmax_version(self, written_snapshot, read_h5):
        """Expects the writing version stamped at the root, so a file can be
        traced to the build that produced it."""
        attrs, _ = read_h5(written_snapshot().filename)
        assert attrs["gprMax"] == __version__

    def test_records_the_cell_counts(self, written_snapshot, read_h5):
        """Expects ``nx_ny_nz`` to be the *view's* size, not the grid's."""
        attrs, _ = read_h5(written_snapshot(start=(0, 0, 0), stop=(4, 5, 6)).filename)
        assert list(attrs["nx_ny_nz"]) == [4, 5, 6]

    def test_records_the_physical_spacing(self, written_snapshot, read_h5):
        """Expects ``dx_dy_dz == step * grid.dl`` — the physical size of one
        snapshot cell, which for a strided snapshot is larger than one grid
        cell."""
        attrs, _ = read_h5(
            written_snapshot(start=(0, 0, 0), stop=(8, 8, 8), step=(2, 2, 2)).filename
        )
        assert attrs["dx_dy_dz"] == pytest.approx([2 * DL, 2 * DL, 2 * DL])

    def test_spacing_follows_an_anisotropic_grid(self, written_snapshot, read_h5):
        """Expects each axis to take its own discretisation."""
        attrs, _ = read_h5(written_snapshot(dl=DL_ANISO).filename)
        assert attrs["dx_dy_dz"] == pytest.approx(list(DL_ANISO))

    def test_records_the_simulation_time_not_the_iteration(self, written_snapshot, read_h5):
        """Expects ``time == iteration * dt`` in seconds. The constructor takes
        an iteration index; the file carries physical time."""
        attrs, _ = read_h5(written_snapshot(time=10).filename)
        assert attrs["time"] == pytest.approx(10 * DT)

    def test_time_zero_is_written(self, written_snapshot, read_h5):
        """Expects a snapshot at iteration 0 to record ``0.0`` rather than
        omitting the attribute."""
        attrs, _ = read_h5(written_snapshot(time=0).filename)
        assert attrs["time"] == pytest.approx(0.0)


class TestHdf5Datasets:
    def test_requested_components_become_root_datasets(self, written_snapshot, read_h5):
        """Expects one dataset per requested component, named for it."""
        _, data = read_h5(written_snapshot().filename)
        assert set(data) == set(FIELDS)

    def test_unrequested_components_are_absent(self, written_snapshot, read_h5):
        """Expects the dummy ``(1,1,1)`` placeholder arrays *not* to be
        written — they exist only to satisfy the Cython signature."""
        _, data = read_h5(written_snapshot(outputs={**NO_OUTPUTS, "Ez": True}).filename)
        assert set(data) == {"Ez"}

    @pytest.mark.parametrize("name", FIELDS)
    def test_each_component_can_be_written_alone(self, written_snapshot, read_h5, name):
        """Expects the six flags to be independent at the file level.
        (6 parameter sets)"""
        _, data = read_h5(written_snapshot(outputs={**NO_OUTPUTS, name: True}).filename)
        assert set(data) == {name}

    def test_dataset_shape_is_the_view_size(self, written_snapshot, read_h5):
        """Expects ``(nx, ny, nz)`` of the view."""
        _, data = read_h5(written_snapshot(start=(0, 0, 0), stop=(4, 5, 6)).filename)
        assert data["Ex"].shape == (4, 5, 6)

    def test_dataset_dtype_matches_the_configured_precision(self, written_snapshot, read_h5):
        """Expects ``float64`` under the double-precision fixture."""
        _, data = read_h5(written_snapshot().filename)
        assert data["Ex"].dtype == np.float64

    def test_values_match_the_in_memory_snapfields(self, written_snapshot, read_h5):
        """Expects the file to carry exactly what ``store()`` computed — no
        scaling, reordering or truncation on the way out."""
        snap = written_snapshot()
        _, data = read_h5(snap.filename)
        for name in FIELDS:
            assert data[name] == pytest.approx(snap.snapfields[name])

    def test_a_strided_snapshot_writes_the_reduced_shape(self, written_snapshot, read_h5):
        """Expects the output to be the number of *snapshot* cells."""
        _, data = read_h5(
            written_snapshot(start=(0, 0, 0), stop=(8, 8, 8), step=(2, 2, 2)).filename
        )
        assert data["Ex"].shape == (4, 4, 4)


class TestVtkLayout:
    def test_cell_data_is_named_for_the_component(self, written_snapshot, read_h5):
        """Expects each requested component under ``VTKHDF/CellData``."""
        _, data = read_h5(written_snapshot(fileext=".vtkhdf").filename)
        assert {f"VTKHDF/CellData/{name}" for name in FIELDS} <= set(data)

    def test_unrequested_components_are_absent(self, written_snapshot, read_h5):
        """Expects the same flag filtering as the HDF5 writer."""
        _, data = read_h5(
            written_snapshot(fileext=".vtkhdf", outputs={**NO_OUTPUTS, "Hy": True}).filename
        )
        assert [k for k in data if k.startswith("VTKHDF/CellData")] == ["VTKHDF/CellData/Hy"]

    def test_declares_the_image_data_type(self, written_snapshot):
        """Expects ``Type == b"ImageData"`` on the ``VTKHDF`` group — the
        marker that tells VTK how to read the file."""
        snap = written_snapshot(fileext=".vtkhdf")
        with h5py.File(snap.filename, "r") as f:
            assert f["VTKHDF"].attrs["Type"] == b"ImageData"

    def test_whole_extent_spans_the_view(self, written_snapshot):
        """Expects ``[0, nx, 0, ny, 0, nz]`` — the extent is relative to the
        snapshot's own origin, not to the grid."""
        snap = written_snapshot(fileext=".vtkhdf", start=(2, 2, 2), stop=(6, 6, 6))
        with h5py.File(snap.filename, "r") as f:
            assert list(f["VTKHDF"].attrs["WholeExtent"]) == [0, 4, 0, 4, 0, 4]

    def test_origin_is_the_physical_start_of_the_window(self, written_snapshot):
        """Expects ``start * grid.dl`` in metres, so a snapshot of a sub-region
        lands in the right place when overlaid on the full model."""
        snap = written_snapshot(fileext=".vtkhdf", start=(2, 3, 4), stop=(6, 7, 8))
        with h5py.File(snap.filename, "r") as f:
            assert f["VTKHDF"].attrs["Origin"] == pytest.approx([2 * DL, 3 * DL, 4 * DL])

    def test_origin_follows_an_anisotropic_grid(self, written_snapshot):
        """Expects each axis scaled by its own discretisation."""
        snap = written_snapshot(fileext=".vtkhdf", start=(2, 2, 2), stop=(6, 6, 6), dl=DL_ANISO)
        with h5py.File(snap.filename, "r") as f:
            assert f["VTKHDF"].attrs["Origin"] == pytest.approx([2 * d for d in DL_ANISO])

    def test_spacing_is_the_physical_cell_size(self, written_snapshot):
        """Expects ``step * grid.dl``, matching the HDF5 writer's
        ``dx_dy_dz``."""
        snap = written_snapshot(fileext=".vtkhdf", start=(0, 0, 0), stop=(8, 8, 8), step=(2, 2, 2))
        with h5py.File(snap.filename, "r") as f:
            assert f["VTKHDF"].attrs["Spacing"] == pytest.approx([2 * DL] * 3)

    def test_direction_defaults_to_the_identity(self, written_snapshot):
        """Expects an axis-aligned lattice — gprMax never rotates a snapshot."""
        snap = written_snapshot(fileext=".vtkhdf")
        with h5py.File(snap.filename, "r") as f:
            assert list(f["VTKHDF"].attrs["Direction"]) == [1, 0, 0, 0, 1, 0, 0, 0, 1]

    def test_values_are_written_in_zyx_order(self, written_snapshot, read_h5):
        """Expects the array **transposed** relative to ``snapfields``.

        The VTKHDF specification stores datasets ZYX-major, so the writer
        transposes on the way out. This is the one place in the PR where the
        on-disk array is not element-for-element what was computed, and reading
        such a file back without transposing gives a model reflected through
        its main diagonal — plausible-looking and completely wrong."""
        snap = written_snapshot(fileext=".vtkhdf")
        _, data = read_h5(snap.filename)
        assert data["VTKHDF/CellData/Ex"] == pytest.approx(snap.snapfields["Ex"].T)

    def test_the_transpose_is_visible_in_the_dataset_shape(self, written_snapshot, read_h5):
        """Expects ``(nz, ny, nx)`` on disk for an ``(nx, ny, nz)`` snapshot —
        the cheapest way to notice the reordering."""
        snap = written_snapshot(fileext=".vtkhdf", start=(0, 0, 0), stop=(2, 3, 4))
        _, data = read_h5(snap.filename)
        assert snap.snapfields["Ex"].shape == (2, 3, 4)
        assert data["VTKHDF/CellData/Ex"].shape == (4, 3, 2)

    def test_the_hdf5_writer_does_not_transpose(self, written_snapshot, read_h5):
        """Expects the plain ``.h5`` format to store ``(nx, ny, nz)`` as
        computed — the two formats genuinely differ here, and a script reading
        both must account for it."""
        snap = written_snapshot(fileext=".h5", start=(0, 0, 0), stop=(2, 3, 4))
        _, data = read_h5(snap.filename)
        assert data["Ex"].shape == (2, 3, 4)
        assert data["Ex"] == pytest.approx(snap.snapfields["Ex"])


class TestProgressReporting:
    def test_bytes_are_reported_per_component(self, make_view_grid, tmp_path, null_pbar):
        """Expects one ``update`` call per written component, each carrying
        that array's byte count."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        snap = Snapshot(
            0, 0, 0, 4, 4, 4, 1, 1, 1, 5, str(tmp_path / "s"), ".h5", dict(ALL_OUTPUTS), g
        )
        snap.initialise_snapfields()
        snap.store()
        snap.write_file(null_pbar)
        assert len(null_pbar.updates) == 6

    def test_reported_bytes_total_nbytes(self, make_view_grid, tmp_path, null_pbar):
        """Expects the progress total to match the ``nbytes`` the bar was sized
        with — otherwise the bar finishes short or overruns."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        snap = Snapshot(
            0, 0, 0, 4, 4, 4, 1, 1, 1, 5, str(tmp_path / "s"), ".h5", dict(ALL_OUTPUTS), g
        )
        snap.initialise_snapfields()
        snap.store()
        snap.write_file(null_pbar)
        assert null_pbar.total == snap.nbytes

    def test_unrequested_components_are_not_reported(self, make_view_grid, tmp_path, null_pbar):
        """Expects no progress update for a component that is not written."""
        g = make_view_grid(nx=8, ny=8, nz=8)
        outputs = {**NO_OUTPUTS, "Ex": True}
        snap = Snapshot(0, 0, 0, 4, 4, 4, 1, 1, 1, 5, str(tmp_path / "s"), ".h5", outputs, g)
        snap.initialise_snapfields()
        snap.store()
        snap.write_file(null_pbar)
        assert len(null_pbar.updates) == 1


class TestSaveSnapshots:
    """The orchestrator: make a directory, relocate each file, write it."""

    @pytest.fixture
    def snapshots(self, make_view_grid, tmp_path):
        def _make(count=2, fileext=".h5"):
            g = make_view_grid(nx=8, ny=8, nz=8)
            built = []
            for i in range(count):
                snap = Snapshot(
                    0, 0, 0, 4, 4, 4, 1, 1, 1, i, f"snap{i}", fileext, dict(ALL_OUTPUTS), g
                )
                snap.initialise_snapfields()
                snap.store()
                built.append(snap)
            return built

        return _make

    def test_creates_the_snapshot_directory(self, snapshots, outputs_config, tmp_path):
        """Expects the directory from ``set_snapshots_dir()`` to be created if
        absent."""
        target = outputs_config.model_config.set_snapshots_dir()
        assert not target.exists()
        save_snapshots(snapshots(1))
        assert target.is_dir()

    def test_tolerates_an_existing_directory(self, snapshots, outputs_config):
        """Expects ``mkdir(exist_ok=True)`` semantics, so a second model in the
        same run does not fail."""
        outputs_config.model_config.set_snapshots_dir().mkdir()
        save_snapshots(snapshots(1))

    def test_relocates_each_file_into_that_directory(self, snapshots, outputs_config):
        """Expects the snapshot directory to be prepended to each filename —
        the snapshot is constructed with a bare name and only learns its
        directory here."""
        target = outputs_config.model_config.set_snapshots_dir()
        snaps = snapshots(2)
        save_snapshots(snaps)
        assert snaps[0].filename == target / "snap0.h5"

    def test_writes_every_snapshot(self, snapshots, outputs_config):
        """Expects one file per snapshot in the list."""
        snaps = snapshots(3)
        save_snapshots(snaps)
        assert all(s.filename.exists() for s in snaps)

    def test_the_written_files_are_readable(self, snapshots, read_h5):
        """Expects a complete, valid file rather than a truncated one — a
        writer that never closes its handle could leave the last one short."""
        snaps = snapshots(2)
        save_snapshots(snaps)
        _, data = read_h5(snaps[1].filename)
        assert set(data) == set(FIELDS)

    def test_an_empty_list_writes_nothing(self, outputs_config):
        """Expects the directory to be created but left empty — a model with no
        snapshots must not fail here."""
        save_snapshots([])
        assert list(outputs_config.model_config.set_snapshots_dir().iterdir()) == []

    def test_logs_the_directory(self, snapshots, caplog, outputs_config):
        """Expects the resolved path in the log, since it is the only place the
        user learns where the files went."""
        import logging

        with caplog.at_level(logging.INFO, logger="gprMax.snapshots"):
            save_snapshots(snapshots(1))
        assert "Snapshot directory:" in caplog.text

    def test_vtk_snapshots_are_saved_too(self, snapshots, outputs_config):
        """Expects the orchestrator to be format-agnostic — it defers to
        ``write_file``."""
        snaps = snapshots(1, fileext=".vtkhdf")
        save_snapshots(snaps)
        assert snaps[0].filename.exists()
        assert snaps[0].filename.suffix == ".vtkhdf"


pytestmark = pytest.mark.unit
