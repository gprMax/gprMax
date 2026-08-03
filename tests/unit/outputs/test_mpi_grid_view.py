"""``MPIGridView`` — the same window, split across ranks.

Under MPI each rank owns a slab of the domain plus a *halo*: a border of cells
mirroring its neighbours' edges, needed so the field update can reach one cell
past the rank boundary. A geometry view or snapshot spanning the whole domain
therefore has to be trimmed on each rank to the part that rank actually owns,
without double-counting halo cells and without breaking the view's stride.

That trimming is what this class adds, and it is pure numpy:

- ``global_*`` records the view as the user asked for it, in global coordinates
- ``has_negative_neighbour`` / ``has_positive_neighbour`` mark which faces abut
  another rank rather than the true domain edge
- ``start`` and ``stop`` are pulled back inside the local grid, *staying aligned
  to the step* — that modulo is the fiddly part
- ``offset`` says where this rank's block belongs inside the global output

**Why these tests work at one rank.** ``MPIGridView.__init__`` asserts
``isinstance(comm, MPI.Intracomm)``, so a mock communicator is rejected
outright. The fixtures hand it a genuine ``MPI.COMM_SELF`` and fake only the
*grid* — and the clamping arithmetic depends on ``negative_halo_offset`` and
``grid.size``, not on how many ranks exist. Setting a halo offset makes a
one-rank view behave exactly as a mid-domain rank would, so every branch below
is reachable. What one rank cannot show is cross-rank agreement; those tests
assert the local arithmetic and the collective call contract instead.
"""

import numpy as np
import pytest
from mpi4py import MPI

from gprMax.geometry_outputs.grid_view import GridView, MPIGridView


@pytest.fixture
def make_mpi_view(make_mpi_grid):
    """Factory for an ``MPIGridView`` over a faked MPI grid.

    ``negative_halo_offset`` and the grid ``size`` between them decide which
    faces count as abutting a neighbour, so those are the knobs tests turn.
    """

    def _make(
        start=(0, 0, 0),
        stop=(12, 12, 12),
        step=(2, 2, 2),
        size=(10, 10, 10),
        negative_halo_offset=(2, 2, 2),
        origin=(100, 100, 100),
        grid=None,
        **grid_kwargs,
    ):
        g = grid if grid is not None else make_mpi_grid(
            size=size, negative_halo_offset=negative_halo_offset, origin=origin,
            **grid_kwargs,
        )
        return MPIGridView(g, *start, *stop, *step)

    return _make


@pytest.fixture
def interior_view(make_mpi_view):
    """A view with neighbours on both sides — the interesting case.

    Grid 10 cells wide with a 2-cell negative halo; the view asks for 0..12
    step 2. Both faces overrun the local grid, so both clamps fire.
    """
    return make_mpi_view()


@pytest.fixture
def edge_view(make_mpi_view):
    """A view with no neighbours at all — the degenerate single-rank case.

    With a zero halo and a view exactly matching the grid, neither clamp fires
    and the class must behave identically to the serial ``GridView``.
    """
    return make_mpi_view(
        start=(0, 0, 0), stop=(10, 10, 10), step=(1, 1, 1),
        negative_halo_offset=(0, 0, 0),
    )


class TestConstruction:
    def test_extends_the_serial_grid_view(self):
        """Expects ``MPIGridView`` to inherit the whole serial surface."""
        assert issubclass(MPIGridView, GridView)

    def test_creates_a_cartesian_communicator(self, interior_view):
        """Expects ``comm`` to be a real ``Cartcomm``, built from the range of
        MPI grid coordinates the view spans."""
        assert isinstance(interior_view.comm, MPI.Cartcomm)

    def test_the_communicator_is_a_new_one(self, interior_view, make_mpi_grid):
        """Expects a fresh sub-communicator rather than the grid's own, so
        collectives over a view do not involve ranks outside it."""
        assert interior_view.comm != MPI.COMM_SELF

    def test_requires_a_real_intracomm(self, make_mpi_grid):
        """Expects the ``assert isinstance(comm, MPI.Intracomm)`` guard to
        reject a stand-in communicator.

        This is why every fixture here supplies a genuine ``MPI.COMM_SELF``:
        the class cannot be tested with a mock."""

        class NotAComm:
            def Split(self):
                return object()

        grid = make_mpi_grid(comm=NotAComm())
        with pytest.raises(AssertionError):
            MPIGridView(grid, 0, 0, 0, 4, 4, 4)

    def test_logs_its_creation_at_debug(self, make_mpi_view, caplog):
        """Expects a debug record carrying both the global and the local
        coordinate triples — the only way to see the clamping in a real run."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="gprMax.geometry_outputs.grid_view"):
            make_mpi_view()
        assert "Created MPIGridView for grid 'mpi_grid'" in caplog.text


class TestGlobalCoordinates:
    def test_global_start_is_the_local_start_mapped_out(self, interior_view):
        """Expects ``local_to_global_coordinate(start)`` — with the fixture's
        origin of 100, a local 0 becomes a global 100."""
        assert interior_view.global_start.tolist() == [100, 100, 100]

    def test_global_stop_is_the_local_stop_mapped_out(self, interior_view):
        """Expects the requested upper bound in global coordinates, before any
        clamping: 12 local becomes 112 global."""
        assert interior_view.global_stop.tolist() == [112, 112, 112]

    def test_global_size_is_the_unclamped_size(self, interior_view):
        """Expects ``ceil((12 - 0)/2) == 6`` — the size of the view *as
        requested*, captured before the local clamp shrinks ``size``.

        This is the shape of the collective output dataset, so it must reflect
        the whole view rather than this rank's share of it."""
        assert interior_view.global_size.tolist() == [6, 6, 6]

    def test_global_size_exceeds_local_size_when_clamped(self, interior_view):
        """Expects the local block to be strictly smaller once both halo faces
        are trimmed — six global cells, four local."""
        assert interior_view.size.tolist() == [4, 4, 4]
        assert np.all(interior_view.global_size > interior_view.size)

    def test_they_agree_when_nothing_is_clamped(self, edge_view):
        """Expects global and local sizes to coincide for a rank owning the
        whole view."""
        assert edge_view.global_size.tolist() == edge_view.size.tolist()

    @pytest.mark.parametrize("name,index", [("gx", 0), ("gy", 1), ("gz", 2)])
    def test_global_size_properties(self, interior_view, name, index):
        """Expects ``gx``/``gy``/``gz`` to index ``global_size``, mirroring
        ``nx``/``ny``/``nz`` on the local size. (3 parameter sets)"""
        assert getattr(interior_view, name) == interior_view.global_size[index]


class TestNeighbourDetection:
    def test_a_start_inside_the_halo_means_a_negative_neighbour(self, interior_view):
        """Expects ``start < negative_halo_offset`` to flag the low face: a
        view beginning at 0 with a 2-cell halo is asking for cells that belong
        to the rank below."""
        assert interior_view.has_negative_neighbour.tolist() == [True, True, True]

    def test_a_start_outside_the_halo_means_no_negative_neighbour(self, make_mpi_view):
        """Expects the flag to clear once the view begins at or past the halo
        boundary."""
        view = make_mpi_view(start=(2, 2, 2), stop=(10, 10, 10), step=(2, 2, 2))
        assert view.has_negative_neighbour.tolist() == [False, False, False]

    def test_a_stop_past_the_grid_means_a_positive_neighbour(self, interior_view):
        """Expects ``stop > grid.size`` to flag the high face."""
        assert interior_view.has_positive_neighbour.tolist() == [True, True, True]

    def test_a_stop_within_the_grid_means_no_positive_neighbour(self, make_mpi_view):
        """Expects the flag to clear for a view ending inside the local grid."""
        view = make_mpi_view(start=(2, 2, 2), stop=(8, 8, 8), step=(2, 2, 2))
        assert view.has_positive_neighbour.tolist() == [False, False, False]

    def test_the_two_faces_are_detected_independently(self, make_mpi_view):
        """Expects a view abutting a neighbour on one side only to set exactly
        one flag."""
        view = make_mpi_view(start=(0, 0, 0), stop=(8, 8, 8), step=(2, 2, 2))
        assert view.has_negative_neighbour.tolist() == [True, True, True]
        assert view.has_positive_neighbour.tolist() == [False, False, False]

    def test_axes_are_detected_independently(self, make_mpi_view):
        """Expects a per-axis decision, so a view can abut a neighbour in x and
        the domain edge in z."""
        view = make_mpi_view(
            start=(0, 4, 4), stop=(12, 8, 8), step=(2, 2, 2)
        )
        assert view.has_negative_neighbour.tolist() == [True, False, False]


class TestClamping:
    def test_start_is_pulled_out_of_the_negative_halo(self, interior_view):
        """Expects ``start`` to move from 0 to 2, the first cell this rank
        actually owns."""
        assert interior_view.start.tolist() == [2, 2, 2]

    def test_the_clamped_start_stays_aligned_to_the_step(self, make_mpi_view):
        """Expects ``halo + ((start - halo) % step)``, which keeps the sample
        points on the same lattice the user asked for.

        With ``start=1``, ``halo=2``, ``step=3``: ``(1-2) % 3 == 2``, so the
        clamped start is 4 — not 2. Snapping naively to the halo boundary would
        shift every exported sample by one cell."""
        view = make_mpi_view(
            start=(1, 1, 1), stop=(16, 16, 16), step=(3, 3, 3),
            size=(12, 12, 12), negative_halo_offset=(2, 2, 2),
        )
        assert view.start.tolist() == [4, 4, 4]

    def test_stop_is_pulled_back_into_the_local_grid(self, interior_view):
        """Expects ``stop`` to move from 12 to 10, the local grid size."""
        assert interior_view.stop.tolist() == [10, 10, 10]

    def test_the_clamped_stop_stays_aligned_to_the_step(self, make_mpi_view):
        """Expects ``grid.size + ((stop - grid.size) % step)``, which can land
        *past* the grid size to preserve the stride.

        With ``stop=16``, ``size=12``, ``step=3``: ``(16-12) % 3 == 1``, so the
        clamped stop is 13."""
        view = make_mpi_view(
            start=(1, 1, 1), stop=(16, 16, 16), step=(3, 3, 3),
            size=(12, 12, 12), negative_halo_offset=(2, 2, 2),
        )
        assert view.stop.tolist() == [13, 13, 13]

    def test_nothing_is_clamped_at_the_domain_edge(self, edge_view):
        """Expects a view matching the grid exactly to keep its own bounds."""
        assert edge_view.start.tolist() == [0, 0, 0]
        assert edge_view.stop.tolist() == [10, 10, 10]

    def test_size_is_recomputed_after_clamping(self, interior_view):
        """Expects ``ceil((10 - 2)/2) == 4`` — the size attribute is
        overwritten once the bounds settle."""
        assert interior_view.size.tolist() == [4, 4, 4]

    def test_clamping_is_per_axis(self, make_mpi_view):
        """Expects each axis to be clamped against its own halo and grid
        extent."""
        view = make_mpi_view(
            start=(0, 3, 3), stop=(12, 9, 9), step=(1, 1, 1),
            size=(10, 10, 10), negative_halo_offset=(2, 0, 0),
        )
        assert view.start.tolist() == [2, 3, 3]
        assert view.stop.tolist() == [10, 9, 9]


class TestOffset:
    def test_offset_places_the_local_block_in_the_global_output(self, interior_view):
        """Expects ``(global(start) - global_start) // step``: the clamped
        start is two cells in, at a step of two, so this rank's block begins at
        index 1 of the global dataset."""
        assert interior_view.offset.tolist() == [1, 1, 1]

    def test_offset_is_zero_when_nothing_was_clamped(self, edge_view):
        """Expects a rank owning the whole view to write from index 0."""
        assert edge_view.offset.tolist() == [0, 0, 0]

    def test_offset_is_measured_in_output_cells_not_grid_cells(self, make_mpi_view):
        """Expects the division by ``step``, so a stride-3 view moved three
        grid cells shifts by one output cell."""
        view = make_mpi_view(
            start=(0, 0, 0), stop=(15, 15, 15), step=(3, 3, 3),
            size=(12, 12, 12), negative_halo_offset=(3, 3, 3),
        )
        assert view.start.tolist() == [3, 3, 3]
        assert view.offset.tolist() == [1, 1, 1]

    def test_offset_plus_size_fits_inside_the_global_size(self, interior_view):
        """Expects this rank's block to lie wholly within the global dataset —
        an offset or size overshoot would corrupt a neighbour's data."""
        assert np.all(interior_view.offset + interior_view.size <= interior_view.global_size)


class TestGetterSliceOverride:
    """The extra node is suppressed when a neighbour will supply it."""

    def test_exclusive_bound_behaves_as_in_the_base_class(self, interior_view):
        """Expects ``slice(start, stop, step)`` — unchanged from serial."""
        assert interior_view.getter_slice(0) == slice(2, 10, 2)

    def test_inclusive_bound_is_suppressed_with_a_positive_neighbour(self, interior_view):
        """Expects *no* extra step: the node past this rank's edge belongs to
        the neighbour, which will contribute it. Taking it here would
        double-count."""
        assert interior_view.getter_slice(0, upper_bound_exclusive=False) == slice(2, 10, 2)

    def test_inclusive_bound_applies_at_the_domain_edge(self, edge_view):
        """Expects the extra step to be taken when there is no neighbour —
        matching the serial class exactly."""
        assert edge_view.getter_slice(0, upper_bound_exclusive=False) == slice(0, 11, 1)

    def test_the_decision_is_per_axis(self, make_mpi_view):
        """Expects the extra node on an axis at the domain edge and not on one
        abutting a neighbour, within the same view."""
        view = make_mpi_view(
            start=(0, 0, 0), stop=(12, 8, 8), step=(2, 2, 2),
            size=(10, 10, 10), negative_halo_offset=(2, 2, 2),
        )
        assert view.getter_slice(0, upper_bound_exclusive=False).stop == 10
        assert view.getter_slice(1, upper_bound_exclusive=False).stop == 10


class TestSetterSliceOverride:
    """Reading back extends *downward* into the negative halo."""

    def test_start_reaches_back_one_step_with_a_negative_neighbour(self, interior_view):
        """Expects ``start - step``: when writing data in, this rank must fill
        its own halo cell too, so the neighbour's edge value is present locally
        for the next field update."""
        assert interior_view.setter_slice(0) == slice(0, 10, 2)

    def test_start_is_unchanged_at_the_domain_edge(self, edge_view):
        """Expects no reach-back where there is no neighbour below."""
        assert edge_view.setter_slice(0) == slice(0, 10, 1)

    def test_the_inclusive_bound_always_extends(self, interior_view):
        """Expects ``stop + step`` regardless of the positive neighbour — the
        setter, unlike the getter, does not suppress the extra node."""
        assert interior_view.setter_slice(0, upper_bound_exclusive=False) == slice(0, 12, 2)

    def test_setter_and_getter_slices_diverge(self, interior_view):
        """Expects the two to differ once halos are in play.

        In the serial class ``setter_slice`` is literally ``getter_slice``.
        This is the assertion that shows the override is doing something."""
        assert interior_view.setter_slice(0) != interior_view.getter_slice(0)


class TestOutputSliceOverride:
    def test_starts_at_the_rank_offset(self, interior_view):
        """Expects ``slice(offset, offset + size)`` rather than starting at
        zero — each rank writes into its own region of a shared dataset."""
        assert interior_view.get_output_slice(0) == slice(1, 5)

    def test_the_extra_node_is_suppressed_with_a_positive_neighbour(self, interior_view):
        """Expects no ``+1``, matching ``getter_slice``: the two must agree or
        the write would be shape-mismatched."""
        assert interior_view.get_output_slice(0, upper_bound_exclusive=False) == slice(1, 5)

    def test_the_extra_node_applies_at_the_domain_edge(self, edge_view):
        """Expects ``size + 1`` where no neighbour contributes it."""
        assert edge_view.get_output_slice(0, upper_bound_exclusive=False) == slice(0, 11)

    def test_output_slice_length_matches_the_getter_slice_length(self, interior_view):
        """Expects the destination window and the source data to be the same
        length — the invariant that makes ``dset[out] = get_solid()`` valid."""
        for exclusive in (True, False):
            out = interior_view.get_output_slice(0, exclusive)
            got = interior_view.getter_slice(0, exclusive)
            assert out.stop - out.start == len(range(got.start, got.stop, got.step))


class TestReadSliceOverride:
    def test_reaches_back_with_a_negative_neighbour(self, interior_view):
        """Expects ``offset - 1`` and one extra element, so the halo cell is
        read back along with the owned block."""
        assert interior_view.get_read_slice(0) == slice(0, 5)

    def test_is_unchanged_at_the_domain_edge(self, edge_view):
        """Expects a plain ``slice(0, size)`` where there is no halo to fill."""
        assert edge_view.get_read_slice(0) == slice(0, 10)

    def test_the_inclusive_bound_always_extends(self, interior_view):
        """Expects ``+1`` for the inclusive bound regardless of the positive
        neighbour, mirroring ``setter_slice``."""
        assert interior_view.get_read_slice(0, upper_bound_exclusive=False) == slice(0, 6)

    def test_read_slice_length_matches_the_setter_slice_length(self, interior_view):
        """Expects the source window and the destination to agree, the mirror
        of the output/getter invariant."""
        for exclusive in (True, False):
            read = interior_view.get_read_slice(0, exclusive)
            setter = interior_view.setter_slice(0, exclusive)
            assert read.stop - read.start == len(
                range(setter.start, setter.stop, setter.step)
            )

    def test_read_and_output_slices_diverge(self, interior_view):
        """Expects the two to differ — in the serial class they are the same
        function, so this is the override's fingerprint."""
        assert interior_view.get_read_slice(0) != interior_view.get_output_slice(0)


class TestDegeneratesToTheSerialClass:
    """With no halo and no overrun, MPI and serial must agree exactly."""

    @pytest.mark.parametrize("dimension", [0, 1, 2])
    def test_getter_slices_match(self, edge_view, make_grid_view, dimension):
        """Expects identical getter slices on both bound conventions.
        (3 parameter sets)"""
        serial = make_grid_view(start=(0, 0, 0), stop=(10, 10, 10), nx=10, ny=10, nz=10)
        for exclusive in (True, False):
            assert edge_view.getter_slice(dimension, exclusive) == serial.getter_slice(
                dimension, exclusive
            )

    @pytest.mark.parametrize("dimension", [0, 1, 2])
    def test_setter_slices_match(self, edge_view, make_grid_view, dimension):
        """Expects identical setter slices. (3 parameter sets)"""
        serial = make_grid_view(start=(0, 0, 0), stop=(10, 10, 10), nx=10, ny=10, nz=10)
        assert edge_view.setter_slice(dimension) == serial.setter_slice(dimension)

    @pytest.mark.parametrize("dimension", [0, 1, 2])
    def test_output_slices_match(self, edge_view, make_grid_view, dimension):
        """Expects identical output slices. (3 parameter sets)"""
        serial = make_grid_view(start=(0, 0, 0), stop=(10, 10, 10), nx=10, ny=10, nz=10)
        assert edge_view.get_output_slice(dimension) == serial.get_output_slice(dimension)

    def test_sizes_match(self, edge_view, make_grid_view):
        """Expects the same cell counts from both classes."""
        serial = make_grid_view(start=(0, 0, 0), stop=(10, 10, 10), nx=10, ny=10, nz=10)
        assert edge_view.size.tolist() == serial.size.tolist()


class TestMaterialsAcrossRanks:
    """``initialise_materials`` is collective; at one rank it must still work."""

    @pytest.fixture
    def material_view(self, make_mpi_grid, make_materials):
        grid = make_mpi_grid(
            size=(4, 4, 4),
            negative_halo_offset=(0, 0, 0),
            arrays={"ID": np.ones((6, 5, 5, 5), dtype=np.uint32)},
        )
        grid.materials = make_materials(3)
        return MPIGridView(grid, 0, 0, 0, 4, 4, 4)

    def test_the_coordinator_collects_every_material(self, material_view):
        """Expects rank 0 to end up holding the deduplicated union — with one
        rank that is simply its own list."""
        material_view.initialise_materials(filter_materials=False)
        assert len(material_view.materials) == 3

    def test_filtering_keeps_only_what_this_rank_sees(self, material_view):
        """Expects an all-free-space block to report one material even though
        the grid defines three."""
        material_view.initialise_materials(filter_materials=True)
        assert len(material_view.materials) == 1

    def test_builds_a_working_map(self, material_view):
        """Expects the local-to-global map to renumber the single visible
        material to index 0."""
        material_view.initialise_materials(filter_materials=True)
        assert material_view.map_to_view_materials(np.array([1], dtype=np.uint32)).tolist() == [0]

    def test_the_map_preserves_dtype(self, material_view):
        """Expects ``uint32`` to survive the ``np.vectorize`` round trip, as in
        the serial class."""
        material_view.initialise_materials(filter_materials=False)
        result = material_view.map_to_view_materials(np.zeros((2, 2), dtype=np.uint32))
        assert result.dtype == np.uint32

    def test_materials_are_deduplicated_and_sorted(self, material_view):
        """Expects ``np.unique`` ordering on the gathered union, so every rank
        agrees on the global numbering."""
        material_view.initialise_materials(filter_materials=False)
        numids = [m.numID for m in material_view.materials]
        assert numids == sorted(set(numids))
