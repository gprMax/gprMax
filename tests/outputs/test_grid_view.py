"""``GridView`` — the rectangular window every exporter looks through.

A ``GridView`` is a start, a stop and a stride. Given those it answers two
questions: what shape is this region, and hand me that slice of an array. Every
snapshot, every geometry view and every geometry object holds one and delegates
all its coordinate arithmetic to it, which makes this the single highest-traffic
class in the PR.

Three things govern the assertions below.

**Size is a ceiling, not a floor.** ``size = ceil((stop - start) / step)``. A
view from 0 to 10 with step 3 spans *four* cells, not three: the final partial
cell is kept. Substituting integer division would silently shorten every
exported array by one in every axis on any non-dividing view.

**There are two slice families, and they mean different things.**
``getter_slice``/``setter_slice`` index the *grid's* arrays in grid
coordinates. ``get_output_slice``/``get_read_slice`` index the *view's* own
output buffer, always starting at zero. In the serial class the members of each
pair are literally the same function; only ``MPIGridView`` makes them diverge,
which is precisely why the equivalences are asserted here — they are the
baseline the MPI overrides are measured against.

**``upper_bound_exclusive=False`` fetches one extra step.** Node-centred arrays
(``ID``) and the six field arrays are read this way, because a Yee cell has one
more node than cell along each axis and the snapshot kernel averages across
that extra node. Cell-centred arrays (``solid``, ``rigidE``, ``rigidH``) are
read exclusively. Getting this backwards does not crash; it shifts every
exported field half a cell.
"""

import numpy as np
import pytest

from gprMax.geometry_outputs.grid_view import GridView

from .conftest import DL_ANISO, nonzero_set


class TestConstruction:
    def test_stores_start_stop_and_step(self, make_view_grid):
        """Expects the nine coordinate arguments to become three int32
        triples."""
        view = GridView(make_view_grid(), 1, 2, 3, 5, 6, 7, 1, 1, 1)
        assert view.start.tolist() == [1, 2, 3]
        assert view.stop.tolist() == [5, 6, 7]
        assert view.step.tolist() == [1, 1, 1]

    def test_step_defaults_to_one(self, make_view_grid):
        """Expects an unstrided view when no step is given — the common case
        for a geometry view of the whole domain."""
        view = GridView(make_view_grid(), 0, 0, 0, 4, 4, 4)
        assert view.step.tolist() == [1, 1, 1]

    def test_coordinate_arrays_are_int32(self, make_grid_view):
        """Expects ``int32`` throughout: these feed HDF5 extent attributes that
        downstream readers type-check."""
        view = make_grid_view()
        assert view.start.dtype == np.int32
        assert view.stop.dtype == np.int32
        assert view.step.dtype == np.int32
        assert view.size.dtype == np.int32

    def test_holds_the_grid_by_reference(self, make_view_grid):
        """Expects ``view.grid`` to be the same object — the view slices the
        grid's live arrays, so a copy would silently detach every setter."""
        g = make_view_grid()
        assert GridView(g, 0, 0, 0, 4, 4, 4).grid is g

    def test_the_id_cache_starts_empty(self, make_grid_view):
        """Expects ``_ID`` to be ``None`` until first requested — the slice is
        built lazily and then cached."""
        assert make_grid_view()._ID is None

    def test_logs_its_creation_at_debug(self, make_view_grid, caplog):
        """Expects a debug record naming the grid and all four coordinate
        triples, which is the only trace a view leaves in a normal run."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="gprMax.geometry_outputs.grid_view"):
            GridView(make_view_grid(), 0, 0, 0, 4, 4, 4)
        assert "Created GridView for grid 'main_grid'" in caplog.text


class TestSizeArithmetic:
    def test_unit_step_size_is_the_extent(self, make_grid_view):
        """Expects ``stop - start`` when the step is one."""
        view = make_grid_view(start=(1, 2, 3), stop=(7, 8, 9))
        assert view.size.tolist() == [6, 6, 6]

    def test_exact_division_gives_the_quotient(self, make_grid_view):
        """Expects ``(0, 8)`` step 2 to give four cells."""
        view = make_grid_view(start=(0, 0, 0), stop=(8, 8, 8), step=(2, 2, 2))
        assert view.size.tolist() == [4, 4, 4]

    def test_non_dividing_extent_rounds_up(self, make_grid_view):
        """Expects ``ceil(10/3) == 4``, not ``10 // 3 == 3``.

        This is the assertion that distinguishes the actual implementation from
        the one most readers assume. The tenth cell is a partial step and it is
        still counted."""
        view = make_grid_view(
            start=(0, 0, 0), stop=(10, 10, 10), step=(3, 3, 3), nx=12, ny=12, nz=12
        )
        assert view.size.tolist() == [4, 4, 4]

    @pytest.mark.parametrize(
        "stop,step,expected",
        [
            (7, 3, 3),
            (8, 3, 3),
            (9, 3, 3),
            (10, 3, 4),
            (5, 2, 3),
            (6, 2, 3),
            (7, 2, 4),
        ],
    )
    def test_ceiling_boundaries(self, make_grid_view, stop, step, expected):
        """Expects the size to tick over exactly one element past each
        multiple of the step. (7 parameter sets)"""
        view = make_grid_view(
            start=(0, 0, 0),
            stop=(stop, stop, stop),
            step=(step, step, step),
            nx=16,
            ny=16,
            nz=16,
        )
        assert view.size[0] == expected

    def test_axes_are_sized_independently(self, make_grid_view):
        """Expects three different steps to give three different sizes — an
        axis mix-up cannot survive this."""
        view = make_grid_view(
            start=(0, 0, 0), stop=(12, 12, 12), step=(1, 2, 3), nx=12, ny=12, nz=12
        )
        assert view.size.tolist() == [12, 6, 4]

    def test_a_zero_width_axis_gives_zero_cells(self, make_grid_view):
        """Expects ``start == stop`` to produce an empty axis rather than one
        cell."""
        view = make_grid_view(start=(3, 0, 0), stop=(3, 4, 4))
        assert view.size[0] == 0

    def test_offset_start_does_not_change_the_count(self, make_grid_view):
        """Expects size to depend on the extent, not on where it begins."""
        a = make_grid_view(start=(0, 0, 0), stop=(4, 4, 4))
        b = make_grid_view(start=(2, 2, 2), stop=(6, 6, 6))
        assert a.size.tolist() == b.size.tolist()


class TestCoordinateProperties:
    @pytest.mark.parametrize(
        "name,index,source",
        [
            ("xs", 0, "start"),
            ("ys", 1, "start"),
            ("zs", 2, "start"),
            ("xf", 0, "stop"),
            ("yf", 1, "stop"),
            ("zf", 2, "stop"),
            ("dx", 0, "step"),
            ("dy", 1, "step"),
            ("dz", 2, "step"),
            ("nx", 0, "size"),
            ("ny", 1, "size"),
            ("nz", 2, "size"),
        ],
    )
    def test_each_property_reads_its_own_axis(self, make_grid_view, name, index, source):
        """Expects the twelve scalar accessors to index the right element of
        the right triple. Anisotropic steps make a wrong axis impossible to
        miss. (12 parameter sets)"""
        view = make_grid_view(
            start=(1, 2, 3), stop=(13, 14, 15), step=(1, 2, 3), nx=16, ny=16, nz=16
        )
        assert getattr(view, name) == getattr(view, source)[index]

    def test_the_three_size_properties_are_distinct(self, make_grid_view):
        """Expects ``nx``, ``ny`` and ``nz`` to disagree on an anisotropic
        view — a sanity check on the parametrised test above."""
        view = make_grid_view(
            start=(0, 0, 0), stop=(12, 12, 12), step=(1, 2, 3), nx=12, ny=12, nz=12
        )
        assert len({view.nx, view.ny, view.nz}) == 3


class TestGetterSlice:
    def test_exclusive_upper_bound_stops_at_stop(self, make_grid_view):
        """Expects ``slice(start, stop, step)`` — the default, used for
        cell-centred arrays."""
        view = make_grid_view(start=(1, 0, 0), stop=(5, 4, 4))
        assert view.getter_slice(0) == slice(1, 5, 1)

    def test_inclusive_upper_bound_adds_one_step(self, make_grid_view):
        """Expects ``slice(start, stop + step, step)`` — one extra sample, for
        node-centred arrays and the field arrays."""
        view = make_grid_view(start=(1, 0, 0), stop=(5, 4, 4))
        assert view.getter_slice(0, upper_bound_exclusive=False) == slice(1, 6, 1)

    def test_the_extra_sample_is_a_whole_step(self, make_grid_view):
        """Expects a strided view to extend by its own step, not by one cell —
        so a step-3 view reaches ``stop + 3``."""
        view = make_grid_view(start=(0, 0, 0), stop=(9, 9, 9), step=(3, 3, 3), nx=12, ny=12, nz=12)
        assert view.getter_slice(0, upper_bound_exclusive=False) == slice(0, 12, 3)

    @pytest.mark.parametrize("dimension", [0, 1, 2])
    def test_each_dimension_uses_its_own_coordinates(self, make_grid_view, dimension):
        """Expects the slice for axis ``d`` to be built from ``start[d]``,
        ``stop[d]`` and ``step[d]``. (3 parameter sets)"""
        view = make_grid_view(
            start=(1, 2, 3), stop=(13, 14, 15), step=(1, 2, 3), nx=16, ny=16, nz=16
        )
        expected = slice(view.start[dimension], view.stop[dimension], view.step[dimension])
        assert view.getter_slice(dimension) == expected


class TestSetterSliceMatchesGetterSlice:
    """In the serial class the two are the same function.

    ``MPIGridView`` overrides ``setter_slice`` with halo-aware logic, so these
    equivalences are the contrast the MPI tests are measured against. Asserting
    them here means a future divergence in the *base* class shows up as a
    failure rather than as a silent behaviour change.
    """

    @pytest.mark.parametrize("dimension", [0, 1, 2])
    @pytest.mark.parametrize("exclusive", [True, False])
    def test_setter_slice_delegates_to_getter_slice(self, make_grid_view, dimension, exclusive):
        """Expects identical slices from both, on every axis and both bound
        conventions. (6 parameter sets)"""
        view = make_grid_view(start=(1, 2, 3), stop=(7, 8, 9))
        assert view.setter_slice(dimension, exclusive) == view.getter_slice(dimension, exclusive)

    @pytest.mark.parametrize("dimension", [0, 1, 2])
    def test_read_slice_delegates_to_output_slice(self, make_grid_view, dimension):
        """Expects ``get_read_slice`` and ``get_output_slice`` to coincide: a
        single process has no separate read and write partitioning.
        (3 parameter sets)"""
        view = make_grid_view()
        assert view.get_read_slice(dimension) == view.get_output_slice(dimension)

    def test_3d_read_slice_matches_3d_output_slice(self, make_grid_view):
        """Expects the tuple forms to agree as well."""
        view = make_grid_view()
        assert view.get_3d_read_slice() == view.get_3d_output_slice()


class TestOutputSlice:
    def test_starts_at_zero(self, make_grid_view):
        """Expects output slices to index the view's own buffer, so they begin
        at zero however far into the grid the view sits."""
        view = make_grid_view(start=(3, 3, 3), stop=(7, 7, 7))
        assert view.get_output_slice(0) == slice(0, 4)

    def test_length_is_the_view_size(self, make_grid_view):
        """Expects the slice to span exactly ``size[dimension]``."""
        view = make_grid_view(start=(0, 0, 0), stop=(9, 9, 9), step=(3, 3, 3), nx=12, ny=12, nz=12)
        assert view.get_output_slice(0) == slice(0, 3)

    def test_inclusive_bound_adds_exactly_one(self, make_grid_view):
        """Expects ``size + 1``, not ``size + step`` — output slices index a
        dense buffer, so the extra node is one element regardless of stride."""
        view = make_grid_view(start=(0, 0, 0), stop=(9, 9, 9), step=(3, 3, 3), nx=12, ny=12, nz=12)
        assert view.get_output_slice(0, upper_bound_exclusive=False) == slice(0, 4)

    def test_3d_form_returns_one_slice_per_axis(self, make_grid_view):
        """Expects a three-tuple in x, y, z order."""
        view = make_grid_view(start=(0, 0, 0), stop=(4, 6, 8))
        assert view.get_3d_output_slice() == (slice(0, 4), slice(0, 6), slice(0, 8))


class TestArraySlicing:
    def test_slices_the_last_three_dimensions(self, make_grid_view):
        """Expects a leading component axis to pass through untouched, so a
        ``(6, nx, ny, nz)`` array keeps all six components."""
        view = make_grid_view(start=(0, 0, 0), stop=(4, 4, 4))
        array = np.zeros((6, 8, 8, 8))
        assert view.get_array_slice(array).shape == (6, 4, 4, 4)

    def test_extracts_the_requested_region(self, make_grid_view, make_view_grid):
        """Expects the values from exactly the named cells — the ramp fill
        makes every cell distinguishable."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        g.solid[...] = np.arange(64).reshape(4, 4, 4)
        view = make_grid_view(grid=g, start=(1, 1, 1), stop=(3, 3, 3))
        assert view.get_array_slice(g.solid).tolist() == [
            [[21, 22], [25, 26]],
            [[37, 38], [41, 42]],
        ]

    def test_stride_takes_every_nth_cell(self, make_grid_view, make_view_grid):
        """Expects a step of two to select alternate cells, not the first
        half."""
        g = make_view_grid(nx=4, ny=1, nz=1)
        g.solid[:, 0, 0] = [10, 20, 30, 40]
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 1, 1), step=(2, 1, 1))
        assert view.get_array_slice(g.solid)[:, 0, 0].tolist() == [10, 30]

    def test_result_is_contiguous(self, make_grid_view):
        """Expects ``np.ascontiguousarray`` to have been applied — the arrays
        go straight into HDF5 and typed memoryviews, both of which require it."""
        view = make_grid_view(start=(0, 0, 0), stop=(8, 8, 8), step=(2, 2, 2))
        assert view.get_array_slice(view.grid.solid).flags["C_CONTIGUOUS"]

    def test_result_is_a_copy_not_a_view(self, make_grid_view, make_view_grid):
        """Expects mutations of the slice not to reach the grid: strided
        slicing plus ``ascontiguousarray`` necessarily copies, and callers rely
        on being able to remap material IDs without corrupting the model."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4), step=(2, 2, 2))
        sliced = view.get_array_slice(g.solid)
        sliced[...] = 99
        assert not np.any(g.solid == 99)

    def test_set_array_slice_writes_back(self, make_grid_view, make_view_grid):
        """Expects the setter to reach the grid's own array, and only the cells
        the view covers."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        g.solid[...] = 0
        view = make_grid_view(grid=g, start=(1, 1, 1), stop=(3, 3, 3))
        view.set_array_slice(g.solid, np.full((2, 2, 2), 7, dtype=g.solid.dtype))
        assert nonzero_set(g.solid) == {(i, j, k) for i in (1, 2) for j in (1, 2) for k in (1, 2)}

    def test_set_array_slice_leaves_the_rest_alone(self, make_grid_view, make_view_grid):
        """Expects cells outside the view to keep their previous values."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        g.solid[...] = 3
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(2, 2, 2))
        view.set_array_slice(g.solid, np.zeros((2, 2, 2), dtype=g.solid.dtype))
        assert g.solid[3, 3, 3] == 3

    def test_get_and_set_round_trip(self, make_grid_view, make_view_grid):
        """Expects reading a region and writing it straight back to be a
        no-op."""
        g = make_view_grid(nx=6, ny=6, nz=6)
        g.solid[...] = np.arange(216).reshape(6, 6, 6) % 3
        view = make_grid_view(grid=g, start=(1, 1, 1), stop=(5, 5, 5))
        before = g.solid.copy()
        view.set_array_slice(g.solid, view.get_array_slice(g.solid))
        assert np.array_equal(g.solid, before)


class TestTypedArrayAccessors:
    """Which arrays are fetched with which bound convention."""

    @pytest.mark.parametrize("name", ["solid", "rigidE", "rigidH"])
    def test_cell_centred_arrays_use_the_exclusive_bound(self, make_grid_view, name):
        """Expects ``solid``, ``rigidE`` and ``rigidH`` to span exactly the
        view's size in each spatial axis — they hold one value per *cell*.
        (3 parameter sets)"""
        view = make_grid_view(start=(0, 0, 0), stop=(4, 4, 4))
        result = getattr(view, f"get_{name}")()
        assert result.shape[-3:] == (4, 4, 4)

    @pytest.mark.parametrize("name", ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
    def test_field_arrays_fetch_one_extra_node(self, make_grid_view, name):
        """Expects ``size + 1`` in every axis. The snapshot kernel averages
        neighbouring samples to bring the six staggered components onto a
        common point, so it needs the node past the edge. (6 parameter sets)"""
        view = make_grid_view(start=(0, 0, 0), stop=(4, 4, 4))
        assert getattr(view, f"get_{name}")().shape == (5, 5, 5)

    def test_id_fetches_one_extra_node(self, make_grid_view):
        """Expects ``(6, size+1, size+1, size+1)`` — ``ID`` is node-centred and
        carries all six components."""
        view = make_grid_view(start=(0, 0, 0), stop=(4, 4, 4))
        assert view.get_ID().shape == (6, 5, 5, 5)

    def test_rigid_arrays_keep_their_leading_axes(self, make_grid_view):
        """Expects ``rigidE`` to keep 12 components and ``rigidH`` 6 — the
        leading axis is not spatial and must not be sliced."""
        view = make_grid_view(start=(0, 0, 0), stop=(4, 4, 4))
        assert view.get_rigidE().shape[0] == 12
        assert view.get_rigidH().shape[0] == 6

    @pytest.mark.parametrize(
        "name,setter,dtype",
        [
            ("solid", "set_solid", np.uint32),
            ("rigidE", "set_rigidE", np.int8),
            ("rigidH", "set_rigidH", np.int8),
        ],
    )
    def test_setters_write_back_to_the_grid(
        self, make_grid_view, make_view_grid, name, setter, dtype
    ):
        """Expects each typed setter to land in the grid array of the same
        name. (3 parameter sets)"""
        g = make_view_grid(nx=4, ny=4, nz=4)
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4))
        target = getattr(g, name)
        getattr(view, setter)(np.full(target.shape, 5, dtype=dtype))
        assert np.all(target == 5)

    def test_set_id_uses_the_inclusive_bound(self, make_grid_view, make_view_grid):
        """Expects the ``ID`` setter to cover ``size + 1`` nodes, matching its
        getter — a mismatched pair would raise on assignment."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4))
        view.set_ID(np.full((6, 5, 5, 5), 2, dtype=np.uint32))
        assert np.all(g.ID[:, :5, :5, :5] == 2)


class TestIdCaching:
    def test_the_first_call_populates_the_cache(self, make_grid_view):
        """Expects ``_ID`` to be filled on first access."""
        view = make_grid_view()
        view.get_ID()
        assert view._ID is not None

    def test_repeat_calls_return_the_cached_array(self, make_grid_view):
        """Expects the same object back, so ``initialise_materials`` and a user
        call do not each rebuild the slice."""
        view = make_grid_view()
        assert view.get_ID() is view.get_ID()

    def test_force_refresh_rebuilds(self, make_grid_view):
        """Expects a new array when asked, so geometry built after the first
        access is picked up."""
        view = make_grid_view()
        first = view.get_ID()
        assert view.get_ID(force_refresh=True) is not first

    def test_the_cache_is_stale_after_the_grid_changes(self, make_grid_view, make_view_grid):
        """Expects the cached copy *not* to track later grid mutations.

        ``get_array_slice`` copies, so this is inherent rather than a defect —
        but it means anything reading ``ID`` after geometry changes must pass
        ``force_refresh``, which is exactly what ``initialise_materials``
        does."""
        g = make_view_grid(nx=4, ny=4, nz=4)
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4))
        cached = view.get_ID()
        g.ID[...] = 2
        assert np.array_equal(view.get_ID(), cached)
        assert np.all(view.get_ID(force_refresh=True) == 2)


class TestInitialiseMaterials:
    def test_unfiltered_takes_every_material(self, make_grid_view):
        """Expects all of the grid's materials, whether or not they appear in
        the view — what ``GeometryViewLines`` asks for."""
        view = make_grid_view(materials=3)
        view.initialise_materials(filter_materials=False)
        assert len(view.materials) == 3

    def test_filtered_keeps_only_what_the_view_contains(self, make_grid_view, make_view_grid):
        """Expects a view over free-space-only cells to report one material,
        even though the grid defines three."""
        g = make_view_grid(nx=4, ny=4, nz=4, materials=3)
        g.ID[...] = 1
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4))
        view.initialise_materials(filter_materials=True)
        assert len(view.materials) == 1

    def test_filtering_reads_the_id_array_afresh(self, make_grid_view, make_view_grid):
        """Expects ``force_refresh`` on the internal ``get_ID`` call, so
        geometry written after an earlier ``get_ID()`` is still seen."""
        g = make_view_grid(nx=4, ny=4, nz=4, materials=3)
        g.ID[...] = 1
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4))
        view.get_ID()
        g.ID[...] = 2
        view.initialise_materials(filter_materials=True)
        assert [m.numID for m in view.materials] == [2]

    def test_materials_are_sorted(self, make_grid_view):
        """Expects ascending order, so the exported material table is stable
        between runs regardless of definition order."""
        view = make_grid_view(materials=4)
        view.initialise_materials(filter_materials=False)
        numids = [m.numID for m in view.materials]
        assert numids == sorted(numids)

    def test_builds_a_dense_index_map(self, make_grid_view, make_view_grid):
        """Expects the map to renumber sparse grid IDs onto ``0..n-1``, which
        is what makes an exported file self-contained."""
        g = make_view_grid(nx=4, ny=4, nz=4, materials=3)
        g.ID[...] = 2
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4))
        view.initialise_materials(filter_materials=True)
        assert view.map_to_view_materials(np.array([2], dtype=np.uint32)).tolist() == [0]

    def test_map_preserves_dtype(self, make_grid_view):
        """Expects ``uint32`` in, ``uint32`` out — ``np.vectorize`` would
        otherwise widen to int64 and break the HDF5 layout."""
        view = make_grid_view(materials=3)
        view.initialise_materials(filter_materials=False)
        result = view.map_to_view_materials(np.zeros((2, 2), dtype=np.uint32))
        assert result.dtype == np.uint32

    def test_map_preserves_shape(self, make_grid_view):
        """Expects an elementwise mapping, not a flattening."""
        view = make_grid_view(materials=3)
        view.initialise_materials(filter_materials=False)
        result = view.map_to_view_materials(np.zeros((2, 3, 4), dtype=np.uint32))
        assert result.shape == (2, 3, 4)

    def test_unfiltered_map_is_the_identity_for_contiguous_ids(self, make_grid_view):
        """Expects no renumbering when the grid's own IDs are already dense —
        materials 0, 1, 2 map to 0, 1, 2."""
        view = make_grid_view(materials=3)
        view.initialise_materials(filter_materials=False)
        ids = np.array([0, 1, 2], dtype=np.uint32)
        assert view.map_to_view_materials(ids).tolist() == [0, 1, 2]

    def test_an_unmapped_id_raises(self, make_grid_view, make_view_grid):
        """Expects ``KeyError`` for a material outside the filtered set — the
        map is a plain dict lookup, so an ID the view never saw is a hard
        error rather than a silently wrong colour."""
        g = make_view_grid(nx=4, ny=4, nz=4, materials=3)
        g.ID[...] = 1
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4))
        view.initialise_materials(filter_materials=True)
        with pytest.raises(KeyError):
            view.map_to_view_materials(np.array([2], dtype=np.uint32))


class TestAnisotropicViews:
    def test_each_axis_slices_independently(self, make_grid_view, make_view_grid):
        """Expects three different steps to produce three different lengths in
        the sliced result."""
        g = make_view_grid(nx=12, ny=12, nz=12)
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(12, 12, 12), step=(1, 2, 3))
        assert view.get_solid().shape == (12, 6, 4)

    def test_field_slices_add_one_node_per_axis(self, make_grid_view, make_view_grid):
        """Expects ``(13, 7, 5)`` for the same view — one extra sample on each
        axis regardless of that axis's stride."""
        g = make_view_grid(nx=12, ny=12, nz=12)
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(12, 12, 12), step=(1, 2, 3))
        assert view.get_Ex().shape == (13, 7, 5)

    def test_spacing_does_not_affect_shapes(self, make_grid_view):
        """Expects an anisotropic *physical* discretisation to leave every
        shape unchanged: ``GridView`` counts cells and never consults ``dl``."""
        uniform = make_grid_view(stop=(4, 4, 4))
        aniso = make_grid_view(stop=(4, 4, 4), dl=DL_ANISO)
        assert uniform.get_solid().shape == aniso.get_solid().shape


pytestmark = pytest.mark.unit
