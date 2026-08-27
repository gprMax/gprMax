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

"""``Snapshot`` — freezing the fields inside a window at one iteration.

A snapshot is a ``GridView`` plus six output flags plus a time. Construction
and property access are thin delegation; the interesting part is ``store()``,
which drives a real OpenMP Cython kernel.

**Why the kernel averages, and why 4 versus 2.** In a Yee lattice the six field
components do not live at the same place. An electric component sits on a cell
*edge*, offset from the cell centre along the two axes that are not its own; a
magnetic component sits on a cell *face*, offset along only its own axis.
Writing them out unaveraged would give six pictures of six slightly different
places. So the kernel interpolates each component onto the cell centre:

    Exsnap[i,j,k] = (Ex[i,j,k] + Ex[i,j+1,k] + Ex[i,j,k+1] + Ex[i,j+1,k+1]) / 4
    Hxsnap[i,j,k] = (Hx[i,j,k] + Hx[i+1,j,k]) / 2

Four neighbours for E, two for H — and which axes are stepped differs per
component. That is the whole Yee convention in six lines, and it is why
``GridView.get_Ex()`` fetches one extra node in every direction.

The tests below pin the ratio, the axes, and the dependence on the extra node,
by filling the grid with fields whose values vary along exactly one axis at a
time.
"""

from pathlib import Path

import numpy as np
import pytest

from gprMax.snapshots import Snapshot

from .conftest import DL, DL_ANISO, DT, FIELDS

ALL_OUTPUTS = {name: True for name in FIELDS}
NO_OUTPUTS = {name: False for name in FIELDS}


@pytest.fixture
def make_snapshot(make_view_grid):
    """Factory for a ``Snapshot`` over a real grid.

    ``initialise_snapfields()`` is called by default because ``store()``
    ``KeyError``s without it; pass ``initialise=False`` to inspect the bare
    constructor state.
    """

    def _make(
        start=(0, 0, 0),
        stop=(4, 4, 4),
        step=(1, 1, 1),
        time=10,
        filename="snap1",
        fileext=".h5",
        outputs=None,
        grid=None,
        initialise=True,
        **grid_kwargs,
    ):
        g = grid if grid is not None else make_view_grid(**grid_kwargs)
        snap = Snapshot(
            *start,
            *stop,
            *step,
            time,
            filename,
            fileext,
            dict(ALL_OUTPUTS if outputs is None else outputs),
            g,
        )
        if initialise:
            snap.initialise_snapfields()
        return snap

    return _make


class TestClassSurface:
    def test_six_allowable_outputs(self):
        """Expects exactly the six field components — snapshots have no current
        outputs, unlike receivers."""
        assert set(Snapshot.allowableoutputs) == set(FIELDS)

    def test_two_file_extensions(self):
        """Expects ``.vtkhdf`` and ``.h5``, the two formats ``write_file``
        dispatches on."""
        assert Snapshot.fileexts == [".vtkhdf", ".h5"]

    def test_max_dimensions_start_at_zero(self):
        """Expects the shared GPU sizing attributes to begin cleared.

        These are *class* attributes mutated by ``htod_snapshot_array``, so the
        suite resets them between tests — see the conftest."""
        assert (Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max) == (0, 0, 0)

    def test_default_threads_per_block(self):
        """Expects ``(1, 1, 1)`` until a GPU run overrides it."""
        assert Snapshot.tpb == (1, 1, 1)

    def test_the_grid_view_type_is_the_serial_one(self, make_snapshot):
        """Expects a plain ``GridView``; ``MPISnapshot`` overrides this to
        return ``MPIGridView``."""
        from gprMax.geometry_outputs.grid_view import GridView

        assert make_snapshot().GRID_VIEW_TYPE is GridView


class TestConstruction:
    def test_builds_a_grid_view_from_the_extents(self, make_snapshot):
        """Expects the nine coordinate arguments to be handed straight to a
        ``GridView`` — the snapshot itself stores no coordinates."""
        snap = make_snapshot(start=(1, 2, 3), stop=(5, 6, 7))
        assert snap.grid_view.start.tolist() == [1, 2, 3]
        assert snap.grid_view.stop.tolist() == [5, 6, 7]

    def test_stores_the_iteration(self, make_snapshot):
        """Expects ``time`` to be the iteration index, kept verbatim."""
        assert make_snapshot(time=42).time == 42

    def test_stores_the_output_flags(self, make_snapshot):
        """Expects the outputs dict as given, so a caller can request a subset."""
        outputs = {**NO_OUTPUTS, "Ez": True}
        assert make_snapshot(outputs=outputs).outputs == outputs

    def test_stores_the_file_extension(self, make_snapshot):
        """Expects ``fileext`` kept separately from the filename — it is what
        ``write_file`` dispatches on."""
        assert make_snapshot(fileext=".vtkhdf").fileext == ".vtkhdf"

    def test_byte_count_starts_at_zero(self, make_snapshot):
        """Expects ``nbytes`` to be zero before ``initialise_snapfields``
        accumulates it."""
        assert make_snapshot(initialise=False).nbytes == 0

    def test_snapfields_starts_empty(self, make_snapshot):
        """Expects no arrays until explicitly initialised."""
        assert make_snapshot(initialise=False).snapfields == {}

    def test_no_validation_is_performed(self, make_snapshot):
        """Expects construction to accept an inverted extent without
        complaint — all geometry checking lives in the user-object layer, which
        PR 6 covers, not here."""
        snap = make_snapshot(start=(4, 4, 4), stop=(2, 2, 2), initialise=False)
        assert snap.grid_view.size.tolist() == [-2, -2, -2]

    def test_grid_is_reached_through_the_view(self, make_snapshot, make_view_grid):
        """Expects ``snap.grid`` to be a property forwarding to
        ``grid_view.grid`` rather than a stored reference."""
        g = make_view_grid()
        assert make_snapshot(grid=g).grid is g


class TestFilename:
    def test_the_extension_is_applied(self, make_snapshot):
        """Expects ``snap1`` plus ``.h5``."""
        assert make_snapshot(filename="snap1", fileext=".h5").filename == Path("snap1.h5")

    def test_an_existing_suffix_is_replaced_not_appended(self, make_snapshot):
        """Expects ``with_suffix`` semantics: ``snap.vtkhdf`` with ``.h5``
        becomes ``snap.h5``, **not** ``snap.vtkhdf.h5``.

        Worth pinning — a user filename containing a dot is silently truncated
        at that dot."""
        snap = make_snapshot(filename="snap.vtkhdf", fileext=".h5")
        assert snap.filename == Path("snap.h5")

    def test_a_dotted_name_is_preserved(self, make_snapshot):
        """Unknown dotted basename components are preserved."""
        snap = make_snapshot(filename="run.2.field", fileext=".h5")
        assert snap.filename == Path("run.2.field.h5")

    def test_directory_components_are_preserved(self, make_snapshot):
        """Expects only the final path component to be re-suffixed."""
        snap = make_snapshot(filename="out.d/snap", fileext=".h5")
        assert snap.filename == Path("out.d/snap.h5")

    def test_the_result_is_a_path(self, make_snapshot):
        """Expects a ``Path``, since ``save_snapshots`` later joins a directory
        onto it."""
        assert isinstance(make_snapshot().filename, Path)


class TestDelegatedProperties:
    @pytest.mark.parametrize(
        "name", ["xs", "ys", "zs", "xf", "yf", "zf", "nx", "ny", "nz", "dx", "dy", "dz"]
    )
    def test_each_property_forwards_to_the_grid_view(self, make_snapshot, name):
        """Expects all twelve accessors to return the view's value, so the
        snapshot holds no duplicate coordinate state. (12 parameter sets)"""
        snap = make_snapshot(
            start=(1, 2, 3), stop=(13, 14, 15), step=(1, 2, 3), nx=16, ny=16, nz=16
        )
        assert getattr(snap, name) == getattr(snap.grid_view, name)

    def test_sizes_follow_the_ceiling_rule(self, make_snapshot):
        """Expects ``nx == ceil((xf-xs)/dx)``: a 0-to-10 window at step 3 is
        four cells, inherited straight from ``GridView``."""
        snap = make_snapshot(
            start=(0, 0, 0), stop=(10, 10, 10), step=(3, 3, 3), nx=12, ny=12, nz=12
        )
        assert (snap.nx, snap.ny, snap.nz) == (4, 4, 4)

    def test_axes_are_independent(self, make_snapshot):
        """Expects three different steps to give three different sizes."""
        snap = make_snapshot(
            start=(0, 0, 0), stop=(12, 12, 12), step=(1, 2, 3), nx=12, ny=12, nz=12
        )
        assert (snap.nx, snap.ny, snap.nz) == (12, 6, 4)


class TestInitialiseSnapfields:
    def test_requested_outputs_get_full_sized_arrays(self, make_snapshot):
        """Expects one array of the view's shape per requested component."""
        snap = make_snapshot(start=(0, 0, 0), stop=(4, 5, 6))
        assert snap.snapfields["Ex"].shape == (4, 5, 6)

    def test_unrequested_outputs_get_a_dummy_array(self, make_snapshot):
        """Expects a ``(1, 1, 1)`` placeholder rather than no entry at all.

        The Cython kernel takes all twelve arrays positionally whatever the
        flags say, so every key must exist — but there is no reason to allocate
        a full volume for a component nobody asked for."""
        snap = make_snapshot(outputs={**NO_OUTPUTS, "Ex": True})
        assert snap.snapfields["Ey"].shape == (1, 1, 1)

    def test_all_six_keys_are_present_regardless(self, make_snapshot):
        """Expects every component to have an entry even when none were
        requested."""
        snap = make_snapshot(outputs=NO_OUTPUTS)
        assert set(snap.snapfields) == set(FIELDS)

    def test_arrays_start_zeroed(self, make_snapshot):
        """Expects a clean buffer — ``store()`` writes every cell, but a
        partially-requested snapshot leaves the dummies untouched."""
        assert not np.any(make_snapshot().snapfields["Ex"])

    def test_uses_the_configured_float_dtype(self, make_snapshot):
        """Expects ``float64`` under the double-precision fixture, matching the
        grid's own field arrays — a mismatch would be rejected by the kernel's
        fused-type memoryview."""
        assert make_snapshot().snapfields["Hz"].dtype == np.float64

    def test_nbytes_counts_only_requested_outputs(self, make_snapshot):
        """Expects the dummy arrays to be excluded, so the progress bar totals
        what will actually be written."""
        snap = make_snapshot(start=(0, 0, 0), stop=(4, 4, 4), outputs={**NO_OUTPUTS, "Ex": True})
        assert snap.nbytes == 64 * 8

    def test_nbytes_sums_across_components(self, make_snapshot):
        """Expects six requested components to total six times one."""
        one = make_snapshot(stop=(4, 4, 4), outputs={**NO_OUTPUTS, "Ex": True})
        six = make_snapshot(stop=(4, 4, 4))
        assert six.nbytes == 6 * one.nbytes

    def test_calling_twice_double_counts(self, make_snapshot):
        """Expects ``nbytes`` to accumulate rather than reset — the method is
        written to be called exactly once, and callers do."""
        snap = make_snapshot(stop=(4, 4, 4))
        before = snap.nbytes
        snap.initialise_snapfields()
        assert snap.nbytes == 2 * before


class TestStoreAveraging:
    """The real Cython kernel, against fields varying on one axis at a time."""

    @staticmethod
    def _uniform_grid(make_view_grid, **kwargs):
        g = make_view_grid(nx=4, ny=4, nz=4, fill=False, **kwargs)
        for name in FIELDS:
            getattr(g, name)[...] = 0.0
        return g

    def test_a_constant_field_is_unchanged(self, make_snapshot, make_view_grid):
        """Expects averaging four equal values to give that value — the
        simplest possible check that the divisor is right."""
        g = self._uniform_grid(make_view_grid)
        g.Ex[...] = 7.0
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Ex"] == pytest.approx(np.full((4, 4, 4), 7.0))

    def test_ex_averages_over_y_and_z(self, make_snapshot, make_view_grid):
        """Expects ``Ex`` to average its two *transverse* axes. A field varying
        only along x must therefore pass through untouched."""
        g = self._uniform_grid(make_view_grid)
        g.Ex[...] = np.arange(5).reshape(5, 1, 1)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Ex"][:, 0, 0] == pytest.approx([0, 1, 2, 3])

    def test_ex_smooths_a_ramp_along_y(self, make_snapshot, make_view_grid):
        """Expects the midpoint of adjacent y samples: a ramp ``0,1,2,3,4``
        along y averages to ``0.5, 1.5, 2.5, 3.5``."""
        g = self._uniform_grid(make_view_grid)
        g.Ex[...] = np.arange(5).reshape(1, 5, 1)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Ex"][0, :, 0] == pytest.approx([0.5, 1.5, 2.5, 3.5])

    def test_ey_averages_over_x_and_z(self, make_snapshot, make_view_grid):
        """Expects ``Ey`` to leave a y-varying field alone, by symmetry with
        ``Ex``."""
        g = self._uniform_grid(make_view_grid)
        g.Ey[...] = np.arange(5).reshape(1, 5, 1)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Ey"][0, :, 0] == pytest.approx([0, 1, 2, 3])

    def test_ez_averages_over_x_and_y(self, make_snapshot, make_view_grid):
        """Expects ``Ez`` to leave a z-varying field alone."""
        g = self._uniform_grid(make_view_grid)
        g.Ez[...] = np.arange(5).reshape(1, 1, 5)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Ez"][0, 0, :] == pytest.approx([0, 1, 2, 3])

    def test_hx_averages_over_x_only(self, make_snapshot, make_view_grid):
        """Expects a magnetic component to be averaged along its *own* axis —
        the opposite convention from the electric ones."""
        g = self._uniform_grid(make_view_grid)
        g.Hx[...] = np.arange(5).reshape(5, 1, 1)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Hx"][:, 0, 0] == pytest.approx([0.5, 1.5, 2.5, 3.5])

    def test_hx_ignores_the_transverse_axes(self, make_snapshot, make_view_grid):
        """Expects a y-varying ``Hx`` to pass through unchanged."""
        g = self._uniform_grid(make_view_grid)
        g.Hx[...] = np.arange(5).reshape(1, 5, 1)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Hx"][0, :, 0] == pytest.approx([0, 1, 2, 3])

    def test_hy_averages_over_y_only(self, make_snapshot, make_view_grid):
        """Expects the y analogue of the ``Hx`` case."""
        g = self._uniform_grid(make_view_grid)
        g.Hy[...] = np.arange(5).reshape(1, 5, 1)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Hy"][0, :, 0] == pytest.approx([0.5, 1.5, 2.5, 3.5])

    def test_hz_averages_over_z_only(self, make_snapshot, make_view_grid):
        """Expects the z analogue."""
        g = self._uniform_grid(make_view_grid)
        g.Hz[...] = np.arange(5).reshape(1, 1, 5)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Hz"][0, 0, :] == pytest.approx([0.5, 1.5, 2.5, 3.5])

    def test_electric_averaging_uses_four_neighbours(self, make_snapshot, make_view_grid):
        """Expects a single unit spike to be spread over four cells at ¼ each —
        the defining signature of the electric stencil."""
        g = self._uniform_grid(make_view_grid)
        g.Ex[1, 1, 1] = 1.0
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        touched = snap.snapfields["Ex"]
        assert np.count_nonzero(touched) == 4
        assert touched[touched != 0] == pytest.approx(np.full(4, 0.25))

    def test_magnetic_averaging_uses_two_neighbours(self, make_snapshot, make_view_grid):
        """Expects a spike to be spread over two cells at ½ each."""
        g = self._uniform_grid(make_view_grid)
        g.Hx[1, 1, 1] = 1.0
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        touched = snap.snapfields["Hx"]
        assert np.count_nonzero(touched) == 2
        assert touched[touched != 0] == pytest.approx(np.full(2, 0.5))

    def test_the_extra_node_is_read(self, make_snapshot, make_view_grid):
        """Expects the last output cell to depend on the node *past* the view's
        stop bound.

        This is why ``get_Ex`` fetches with ``upper_bound_exclusive=False``.
        Placing a value only at index 4 of a 0-to-4 view still changes the
        result at output index 3."""
        g = self._uniform_grid(make_view_grid)
        g.Ex[0, 4, 4] = 8.0
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Ex"][0, 3, 3] == pytest.approx(2.0)


class TestStoreFlags:
    def test_an_unrequested_component_is_not_written(self, make_snapshot, make_view_grid):
        """Expects the dummy array to stay zeroed when the flag is false — the
        kernel skips the whole assignment."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        snap = make_snapshot(grid=g, stop=(4, 4, 4), outputs={**NO_OUTPUTS, "Ex": True})
        snap.store()
        assert not np.any(snap.snapfields["Ey"])

    def test_a_requested_component_is_written(self, make_snapshot, make_view_grid):
        """Expects the ramp-filled grid to produce non-zero output."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        snap = make_snapshot(grid=g, stop=(4, 4, 4), outputs={**NO_OUTPUTS, "Ex": True})
        snap.store()
        assert np.any(snap.snapfields["Ex"])

    @pytest.mark.parametrize("name", FIELDS)
    def test_each_component_can_be_requested_alone(self, make_snapshot, make_view_grid, name):
        """Expects the six flags to be independent, so requesting one leaves
        the other five untouched. (6 parameter sets)"""
        g = make_view_grid(nx=4, ny=4, nz=4)
        snap = make_snapshot(grid=g, stop=(4, 4, 4), outputs={**NO_OUTPUTS, name: True})
        snap.store()
        assert np.any(snap.snapfields[name])
        for other in FIELDS:
            if other != name:
                assert not np.any(snap.snapfields[other])

    def test_store_returns_none(self, make_snapshot):
        """Expects in-place population of ``snapfields`` rather than a return
        value."""
        assert make_snapshot().store() is None

    def test_store_is_repeatable(self, make_snapshot, make_view_grid):
        """Expects a second call on unchanged fields to give the same answer —
        the kernel assigns rather than accumulates."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        first = snap.snapfields["Ex"].copy()
        snap.store()
        assert snap.snapfields["Ex"] == pytest.approx(first)

    def test_a_later_store_picks_up_new_field_values(self, make_snapshot, make_view_grid):
        """Expects the slices to be re-fetched each call, so a snapshot object
        reused across iterations sees current data."""
        g = make_view_grid(nx=4, ny=4, nz=4, fill=False)
        g.Ex[...] = 0.0
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        g.Ex[...] = 5.0
        snap.store()
        assert snap.snapfields["Ex"] == pytest.approx(np.full((4, 4, 4), 5.0))


class TestStoreWithStride:
    def test_a_strided_snapshot_samples_rather_than_averages_the_gap(
        self, make_snapshot, make_view_grid
    ):
        """Expects a step-2 view to produce half-sized output, with each cell
        still averaged from its own immediate neighbours rather than over the
        skipped cells."""
        g = make_view_grid(nx=8, ny=8, nz=8, fill=False)
        for name in FIELDS:
            getattr(g, name)[...] = 0.0
        g.Hx[...] = np.arange(9).reshape(9, 1, 1)
        snap = make_snapshot(grid=g, stop=(8, 8, 8), step=(2, 2, 2))
        snap.store()
        assert snap.snapfields["Hx"].shape == (4, 4, 4)
        assert snap.snapfields["Hx"][:, 0, 0] == pytest.approx([1.0, 3.0, 5.0, 7.0])

    def test_output_shape_follows_the_ceiling_rule(self, make_snapshot, make_view_grid):
        """Expects a non-dividing strided view to keep its partial final
        cell."""
        g = make_view_grid(nx=12, ny=12, nz=12)
        snap = make_snapshot(grid=g, start=(0, 0, 0), stop=(10, 10, 10), step=(3, 3, 3))
        snap.store()
        assert snap.snapfields["Ex"].shape == (4, 4, 4)

    def test_an_offset_window_reads_the_right_region(self, make_snapshot, make_view_grid):
        """Expects a window starting at 2 to average cells 2 and 3, not 0
        and 1."""
        g = make_view_grid(nx=8, ny=8, nz=8, fill=False)
        for name in FIELDS:
            getattr(g, name)[...] = 0.0
        g.Hx[...] = np.arange(9).reshape(9, 1, 1)
        snap = make_snapshot(grid=g, start=(2, 0, 0), stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Hx"][:, 0, 0] == pytest.approx([2.5, 3.5])


class TestSinglePrecision:
    def test_store_works_with_float32_arrays(self, monkeypatch, make_view_grid, make_snapshot):
        """Expects the kernel's fused ``float_or_double`` type to bind to the
        single-precision specialisation when the run is configured for it.

        Both the grid arrays and the snapshot buffers must agree; the shared
        ``config`` key guarantees they do."""
        from gprMax import config

        monkeypatch.setitem(config.sim_config.dtypes, "float_or_double", np.float32)
        g = make_view_grid(nx=4, ny=4, nz=4, fill=False)
        for name in FIELDS:
            getattr(g, name)[...] = 2.0
        snap = make_snapshot(grid=g, stop=(4, 4, 4))
        snap.store()
        assert snap.snapfields["Ex"].dtype == np.float32
        assert snap.snapfields["Ex"] == pytest.approx(np.full((4, 4, 4), 2.0))


pytestmark = pytest.mark.unit
