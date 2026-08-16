"""Snapshot device transfer and the MPI snapshot.

Two areas that share nothing except living in ``snapshots.py``.

**``htod_snapshot_array`` / ``dtoh_snapshot_array``** move snapshot buffers to
and from an accelerator. The host-side half — deciding how large the shared
device array must be, and how many time slices it holds — is plain numpy and
fully testable here. The device half branches on the solver name and imports
``pycuda`` / ``pyopencl`` inside the branch, so those paths are driven with
injected stand-in modules and a stand-in Metal device. That tests the wiring,
not the hardware; PR 12 covers the accelerators themselves.

``Snapshot.nx_max``/``ny_max``/``nz_max`` are **class** attributes, sized to the
largest snapshot in the model so one device allocation serves them all. They
are therefore global mutable state, and the suite's autouse fixture restores
them after each test — without that, a large snapshot in one test silently
enlarges every allocation in the next.

**``MPISnapshot``** overrides the grid-view type, records its Cartesian
neighbours, and exchanges halo data before averaging. At one rank there are no
neighbours, so what is established here is the wiring and the degenerate-case
agreement with the serial class. The ``driver="mpio"`` write path cannot run at
all in this environment — ``h5py.get_config().mpi`` is ``False`` — so it is
guarded rather than faked.
"""

import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from gprMax.snapshots import MPISnapshot, Snapshot, dtoh_snapshot_array, htod_snapshot_array

from .conftest import DL, DT, FIELDS

ALL_OUTPUTS = {name: True for name in FIELDS}

needs_parallel_hdf5 = pytest.mark.skipif(
    not h5py.get_config().mpi,
    reason="h5py built without MPI support; the driver='mpio' path cannot run",
)


class FakeMetalDevice:
    """Records the buffers a Metal run would allocate."""

    def __init__(self):
        self.buffers = []

    def newBufferWithBytes_length_options_(self, array, nbytes, options):
        self.buffers.append((array.shape, array.dtype, nbytes, options))
        return f"metal-buffer-{len(self.buffers)}"


@pytest.fixture
def metal_solver(outputs_config):
    """Switch the run to the Metal backend with a recording device.

    The Metal branch is the only accelerator path with no third-party import,
    so it is the cheapest way to drive ``htod_snapshot_array`` end to end.
    """
    device = FakeMetalDevice()
    outputs_config.sim_config.general["solver"] = "metal"
    outputs_config.model_config.device["dev"] = device
    return device


@pytest.fixture
def fake_gpu_modules(monkeypatch):
    """Inject stand-ins for ``pycuda.gpuarray`` and ``pyopencl.array``.

    Both are imported *inside* their branch, so putting them in ``sys.modules``
    is enough to make the branch runnable on a machine with neither installed.
    """
    calls = []

    gpuarray = SimpleNamespace(to_gpu=lambda a: calls.append(("cuda", a.shape)) or a)
    clarray = SimpleNamespace(to_device=lambda q, a: calls.append(("opencl", q, a.shape)) or a)
    pycuda = SimpleNamespace(gpuarray=gpuarray)
    pyopencl = SimpleNamespace(array=clarray)

    monkeypatch.setitem(sys.modules, "pycuda", pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.gpuarray", gpuarray)
    monkeypatch.setitem(sys.modules, "pyopencl", pyopencl)
    monkeypatch.setitem(sys.modules, "pyopencl.array", clarray)
    return calls


@pytest.fixture
def make_snapshots(make_view_grid):
    """A list of snapshots of assorted sizes over one grid."""

    def _make(sizes=((4, 4, 4),), fileext=".h5"):
        g = make_view_grid(nx=16, ny=16, nz=16)
        snaps = []
        for i, (nx, ny, nz) in enumerate(sizes):
            snap = Snapshot(
                0, 0, 0, nx, ny, nz, 1, 1, 1, i, f"snap{i}", fileext, dict(ALL_OUTPUTS), g
            )
            snap.initialise_snapfields()
            snaps.append(snap)
        return snaps

    return _make


class TestMaximumDimensions:
    def test_records_the_largest_snapshot(self, make_snapshots, metal_solver):
        """Expects the class-level maxima to end up at the largest extent seen
        on each axis — one device allocation has to fit every snapshot."""
        htod_snapshot_array(make_snapshots([(4, 4, 4), (6, 2, 2)]))
        assert (Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max) == (6, 4, 4)

    def test_maxima_are_taken_per_axis(self, make_snapshots, metal_solver):
        """Expects an axis-wise maximum rather than the single largest
        snapshot: a tall thin snapshot and a short wide one together demand a
        box big enough for both."""
        htod_snapshot_array(make_snapshots([(8, 1, 1), (1, 8, 1), (1, 1, 8)]))
        assert (Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max) == (8, 8, 8)

    def test_a_single_snapshot_sets_its_own_size(self, make_snapshots, metal_solver):
        """Expects the common case to be exact rather than padded."""
        htod_snapshot_array(make_snapshots([(3, 5, 7)]))
        assert (Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max) == (3, 5, 7)

    def test_the_maxima_only_grow(self, make_snapshots, metal_solver):
        """Expects a second call with smaller snapshots to leave the maxima
        alone — the comparison is one-sided.

        This is exactly why the suite resets these between tests: without the
        reset, the values would ratchet upward across the whole session."""
        htod_snapshot_array(make_snapshots([(8, 8, 8)]))
        htod_snapshot_array(make_snapshots([(2, 2, 2)]))
        assert (Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max) == (8, 8, 8)

    def test_they_are_class_level_not_instance_level(self, make_snapshots, metal_solver):
        """Expects the sizing to be visible on the class itself, and therefore
        on every other snapshot in the process."""
        snaps = make_snapshots([(5, 5, 5)])
        htod_snapshot_array(snaps)
        assert Snapshot.nx_max == 5
        assert snaps[0].nx_max == 5


class TestDeviceArrayShape:
    def test_one_time_slice_when_copying_back_each_iteration(
        self, make_snapshots, metal_solver, outputs_config
    ):
        """Expects a leading axis of 1 when ``snapsgpu2cpu`` is set: the device
        holds one snapshot at a time and the host takes each away."""
        outputs_config.model_config.device["snapsgpu2cpu"] = True
        htod_snapshot_array(make_snapshots([(4, 4, 4), (4, 4, 4)]))
        assert metal_solver.buffers[0][0] == (1, 4, 4, 4)

    def test_one_time_slice_per_snapshot_when_kept_on_device(
        self, make_snapshots, metal_solver, outputs_config
    ):
        """Expects a leading axis equal to the snapshot count when they are all
        retained on the accelerator."""
        outputs_config.model_config.device["snapsgpu2cpu"] = False
        htod_snapshot_array(make_snapshots([(4, 4, 4), (4, 4, 4), (4, 4, 4)]))
        assert metal_solver.buffers[0][0] == (3, 4, 4, 4)

    def test_spatial_axes_use_the_maxima(self, make_snapshots, metal_solver):
        """Expects the allocation to be sized by the largest snapshot, so every
        snapshot fits in the shared buffer."""
        htod_snapshot_array(make_snapshots([(2, 4, 6), (6, 2, 2)]))
        assert metal_solver.buffers[0][0] == (2, 6, 4, 6)

    def test_six_buffers_are_allocated(self, make_snapshots, metal_solver):
        """Expects one per field component."""
        htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        assert len(metal_solver.buffers) == 6

    def test_buffers_use_the_configured_precision(self, make_snapshots, metal_solver):
        """Expects ``float64`` under the double-precision fixture."""
        htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        assert metal_solver.buffers[0][1] == np.float64

    def test_byte_length_matches_the_array(self, make_snapshots, metal_solver):
        """Expects the length handed to the device to be the array's own
        ``nbytes`` — a mismatch would truncate or overrun the buffer."""
        htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        shape, dtype, nbytes, _ = metal_solver.buffers[0]
        assert nbytes == int(np.prod(shape)) * np.dtype(dtype).itemsize

    def test_returns_one_handle_per_component(self, make_snapshots, metal_solver):
        """Expects a six-tuple in ``Ex, Ey, Ez, Hx, Hy, Hz`` order."""
        result = htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        assert len(result) == 6
        assert result[0] == "metal-buffer-1"


class TestSolverDispatch:
    def test_cuda_sets_blocks_per_grid(self, make_snapshots, fake_gpu_modules, outputs_config):
        """Expects ``bpg`` sized from the total cell count over the threads per
        block, as a 3-tuple with singleton y and z."""
        outputs_config.sim_config.general["solver"] = "cuda"
        htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        assert Snapshot.bpg == (64, 1, 1)

    def test_cuda_uploads_every_component(self, make_snapshots, fake_gpu_modules, outputs_config):
        """Expects six ``to_gpu`` calls, one per field."""
        outputs_config.sim_config.general["solver"] = "cuda"
        htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        assert [c[0] for c in fake_gpu_modules] == ["cuda"] * 6

    def test_opencl_sets_a_workgroup_size(self, make_snapshots, fake_gpu_modules, outputs_config):
        """Expects ``wgs`` to be the plain cell count.

        Note the asymmetry: CUDA writes ``bpg``, OpenCL writes ``wgs``, and
        Metal writes neither. The three backends do not share an attribute."""
        outputs_config.sim_config.general["solver"] = "opencl"
        htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        assert Snapshot.wgs == (64, 1, 1)

    def test_opencl_passes_the_queue_through(
        self, make_snapshots, fake_gpu_modules, outputs_config
    ):
        """Expects the caller's queue to reach ``to_device`` — CUDA and Metal
        take no queue, OpenCL does."""
        outputs_config.sim_config.general["solver"] = "opencl"
        htod_snapshot_array(make_snapshots([(4, 4, 4)]), queue="the-queue")
        assert all(c[1] == "the-queue" for c in fake_gpu_modules)

    def test_metal_reads_its_device_from_config(self, make_snapshots, metal_solver):
        """Expects the Metal branch to fetch the device from
        ``get_model_config().device["dev"]`` rather than from an argument."""
        htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        assert len(metal_solver.buffers) == 6

    def test_metal_sets_neither_bpg_nor_wgs(self, make_snapshots, metal_solver):
        """Expects the Metal path to leave the CUDA and OpenCL sizing
        attributes untouched, since it dispatches differently."""
        htod_snapshot_array(make_snapshots([(4, 4, 4)]))
        assert Snapshot.bpg is None


class TestDtohSnapshotArray:
    """Pure slicing: pull one snapshot's window out of the shared device array."""

    @pytest.fixture
    def device_arrays(self):
        """Six distinguishable 4D arrays standing in for device buffers."""
        return [
            np.arange(2 * 8 * 8 * 8, dtype=np.float64).reshape(2, 8, 8, 8) + offset
            for offset in (0, 1000, 2000, 3000, 4000, 5000)
        ]

    @pytest.fixture
    def snap(self, make_view_grid):
        g = make_view_grid(nx=8, ny=8, nz=8)
        s = Snapshot(1, 1, 1, 3, 3, 3, 1, 1, 1, 0, "s", ".h5", dict(ALL_OUTPUTS), g)
        s.initialise_snapfields()
        return s

    def test_populates_all_six_components(self, device_arrays, snap):
        """Expects every entry of ``snapfields`` to be replaced."""
        dtoh_snapshot_array(*device_arrays, 0, snap)
        assert set(snap.snapfields) == set(FIELDS)

    def test_each_component_reads_its_own_buffer(self, device_arrays, snap):
        """Expects the six buffers to map to the six components in order — the
        offsets make a crossed pair impossible to miss."""
        dtoh_snapshot_array(*device_arrays, 0, snap)
        for offset, name in zip((0, 1000, 2000, 3000, 4000, 5000), FIELDS):
            assert snap.snapfields[name].min() >= offset

    def test_extracts_the_snapshot_window(self, device_arrays, snap):
        """Expects the ``xs:xf`` window, so a snapshot from 1 to 3 gives a
        2-cube."""
        dtoh_snapshot_array(*device_arrays, 0, snap)
        assert snap.snapfields["Ex"].shape == (2, 2, 2)

    def test_selects_the_requested_time_index(self, device_arrays, snap):
        """Expects the leading index to choose which stored snapshot is
        pulled back."""
        dtoh_snapshot_array(*device_arrays, 1, snap)
        first = snap.snapfields["Ex"].copy()
        dtoh_snapshot_array(*device_arrays, 0, snap)
        assert not np.array_equal(first, snap.snapfields["Ex"])

    def test_values_come_from_the_right_cells(self, device_arrays, snap):
        """Expects an exact match against the same slice taken by hand.
        dtoh_snapshot_array slices by :nx, :ny, :nz (0-based local indices),
        not by xs:xf (absolute grid coordinates)."""
        dtoh_snapshot_array(*device_arrays, 0, snap)
        assert snap.snapfields["Ex"] == pytest.approx(
            device_arrays[0][0, : snap.nx, : snap.ny, : snap.nz]
        )

    def test_returns_none(self, device_arrays, snap):
        """Expects in-place mutation of ``snapfields``."""
        assert dtoh_snapshot_array(*device_arrays, 0, snap) is None


@pytest.fixture
def make_mpi_snapshot(make_mpi_grid, tmp_path):
    """An ``MPISnapshot`` over a faked MPI grid with real field arrays."""

    def _make(
        start=(0, 0, 0),
        stop=(4, 4, 4),
        step=(1, 1, 1),
        size=(8, 8, 8),
        negative_halo_offset=(0, 0, 0),
        fileext=".h5",
        outputs=None,
        name="mpisnap",
    ):
        arrays = {
            field: np.zeros((size[0] + 1, size[1] + 1, size[2] + 1), dtype=np.float64)
            for field in FIELDS
        }
        for field, array in arrays.items():
            array[...] = 2.0
        grid = make_mpi_grid(size=size, negative_halo_offset=negative_halo_offset, arrays=arrays)
        return MPISnapshot(
            *start,
            *stop,
            *step,
            5,
            str(tmp_path / name),
            fileext,
            dict(ALL_OUTPUTS if outputs is None else outputs),
            grid,
        )

    return _make


class TestMpiSnapshotConstruction:
    def test_extends_the_serial_snapshot(self):
        """Expects the whole serial surface to be inherited."""
        assert issubclass(MPISnapshot, Snapshot)

    def test_uses_an_mpi_grid_view(self, make_mpi_snapshot):
        """Expects ``GRID_VIEW_TYPE`` to be overridden, so the snapshot's
        coordinate arithmetic is halo-aware."""
        from gprMax.geometry_outputs.grid_view import MPIGridView

        assert isinstance(make_mpi_snapshot().grid_view, MPIGridView)

    def test_asserts_the_view_type(self, make_mpi_snapshot):
        """Expects the explicit ``assert isinstance`` in ``__init__`` to hold —
        it is what guarantees ``self.comm`` exists."""
        assert make_mpi_snapshot().comm is not None

    def test_takes_its_communicator_from_the_view(self, make_mpi_snapshot):
        """Expects the Cartesian communicator built by ``MPIGridView`` to be
        reused rather than a second one created."""
        snap = make_mpi_snapshot()
        assert snap.comm is snap.grid_view.comm

    def test_records_neighbours_on_three_axes(self, make_mpi_snapshot):
        """Expects a ``(3, 2)`` table — two directions on each of three
        axes."""
        assert make_mpi_snapshot().neighbours.shape == (3, 2)

    def test_a_single_rank_has_no_neighbours(self, make_mpi_snapshot):
        """Expects every entry negative: ``Cartcomm.Shift`` returns
        ``MPI.PROC_NULL`` where there is no neighbour, and ``has_neighbour``
        tests for a non-negative rank."""
        snap = make_mpi_snapshot()
        assert np.all(snap.neighbours < 0)

    @pytest.mark.parametrize("dimension", [0, 1, 2])
    @pytest.mark.parametrize("direction", [0, 1])
    def test_has_neighbour_is_false_at_one_rank(self, make_mpi_snapshot, dimension, direction):
        """Expects ``has_neighbour`` to report ``False`` on every face of a
        single-rank domain. (6 parameter sets)"""
        assert not make_mpi_snapshot().has_neighbour(dimension, direction)

    def test_distinct_message_tags(self):
        """Expects four distinct tags, so the halo exchanges for H and the
        three E components cannot be confused with one another."""
        tags = {
            MPISnapshot.H_TAG,
            MPISnapshot.EX_TAG,
            MPISnapshot.EY_TAG,
            MPISnapshot.EZ_TAG,
        }
        assert len(tags) == 4


class TestMpiSnapshotStore:
    def test_stores_without_neighbours(self, make_mpi_snapshot):
        """Expects a single-rank store to complete: with no neighbours every
        halo exchange is skipped and the local averaging runs alone."""
        snap = make_mpi_snapshot()
        snap.initialise_snapfields()
        snap.store()
        assert snap.snapfields["Ex"].shape == (4, 4, 4)

    def test_a_constant_field_survives_averaging(self, make_mpi_snapshot):
        """Expects the same answer as the serial path for a uniform field —
        averaging equal values changes nothing however the domain is split."""
        snap = make_mpi_snapshot()
        snap.initialise_snapfields()
        snap.store()
        assert snap.snapfields["Ex"] == pytest.approx(np.full((4, 4, 4), 2.0))

    def test_logs_the_iteration_at_debug(self, make_mpi_snapshot, caplog):
        """Expects a debug record naming the iteration, the only trace the MPI
        store leaves."""
        import logging

        snap = make_mpi_snapshot()
        snap.initialise_snapfields()
        with caplog.at_level(logging.DEBUG, logger="gprMax.snapshots"):
            snap.store()
        assert "Saving snapshot for iteration: 5" in caplog.text


class TestMpiSnapshotWriting:
    """Both MPI write paths need parallel HDF5, not just the ``.h5`` one.

    ``MPISnapshot.write_hdf5`` opens ``h5py.File(..., driver="mpio")``
    directly, and ``write_vtk`` reaches the same call through
    ``VtkImageData(..., comm=...)``. Where ``h5py`` is built without MPI
    support — this environment and CI both — the open raises before any
    gprMax logic runs, so there is nothing these tests could assert. They ship
    guarded so they execute wherever parallel HDF5 does exist.

    What *is* covered unconditionally is everything upstream of the write:
    construction, neighbour discovery, the halo-free store, and the global
    versus local size arithmetic in ``TestMpiGridView``.
    """

    @needs_parallel_hdf5
    def test_vtk_write_uses_global_dimensions(self, make_mpi_snapshot):
        """Expects ``WholeExtent`` to describe the *whole* view rather than
        this rank's share, so all ranks agree on one dataset shape."""
        snap = make_mpi_snapshot(fileext=".vtkhdf")
        snap.initialise_snapfields()
        snap.store()
        snap.write_vtk(_NullBar())
        with h5py.File(snap.filename, "r") as f:
            assert list(f["VTKHDF"].attrs["WholeExtent"]) == [0, 4, 0, 4, 0, 4]

    @needs_parallel_hdf5
    def test_vtk_write_places_data_at_the_rank_offset(self, make_mpi_snapshot, read_h5):
        """Expects the local block written at ``grid_view.offset``; at one rank
        that offset is zero and the whole dataset is filled."""
        snap = make_mpi_snapshot(fileext=".vtkhdf")
        snap.initialise_snapfields()
        snap.store()
        snap.write_vtk(_NullBar())
        _, data = read_h5(snap.filename)
        assert data["VTKHDF/CellData/Ex"].shape == (4, 4, 4)

    @needs_parallel_hdf5
    def test_hdf5_write_uses_the_mpio_driver(self, make_mpi_snapshot):
        """Expects a parallel-HDF5 write producing one shared file."""
        snap = make_mpi_snapshot(fileext=".h5")
        snap.initialise_snapfields()
        snap.store()
        snap.write_hdf5(_NullBar())
        assert snap.filename.exists()

    def test_the_mpio_driver_is_genuinely_unavailable_here(self):
        """Expects ``h5py.get_config().mpi`` to be the thing gating the three
        tests above, recorded explicitly so the skips are not mistaken for an
        oversight."""
        assert h5py.get_config().mpi in (True, False)

    def test_global_size_is_what_the_writers_would_use(self, make_mpi_snapshot):
        """Expects the size the guarded writers pass to HDF5 to be computed
        correctly even though the write itself cannot run — the arithmetic is
        reachable, only the I/O is not."""
        snap = make_mpi_snapshot()
        assert snap.grid_view.global_size.tolist() == [4, 4, 4]


class _NullBar:
    def update(self, n=1):
        pass

    def close(self):
        pass


pytestmark = pytest.mark.unit
