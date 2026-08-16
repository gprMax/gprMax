"""Shared fixtures for the output-writing test suite.

Three source areas share this directory — ``snapshots.py``,
``fields_outputs.py`` and the whole ``geometry_outputs/`` package — because
they share one abstraction. ``Snapshot``, every ``GeometryView`` and
``GeometryObject`` each *hold* a ``GridView`` and delegate all their
coordinate arithmetic to it. Splitting them across directories would mean
building the ``GridView`` fixture in one suite and importing it into another.

The configuration surface is wider than the PML suite's, and every key below
is read somewhere in these files:

- ``sim_config.dtypes["float_or_double"]`` — snapshot field allocation
- ``sim_config.general["progressbars"]`` — the ``tqdm`` bars in the two
  ``save_*`` orchestrators
- ``sim_config.general["solver"]`` — dispatch in ``htod_snapshot_array``
- ``sim_config.input_file_path`` — ``GeometryObject`` filename derivation
- ``get_model_config().output_file_path`` and ``.appendmodelnumber`` —
  ``GeometryView.set_filename``
- ``get_model_config().ompthreads`` — the snapshot Cython kernel
- ``get_model_config().device`` — the GPU branches of ``htod_snapshot_array``
- ``get_model_config().set_snapshots_dir()`` — ``save_snapshots``

Two traps are worth stating once here.

**``GridView.size`` uses ``ceil``, not floor.** A view from 0 to 10 with step
3 has four cells, not three; the final partial cell is kept. Several fixtures
below deliberately use non-dividing extents so that a change to floor
division would be caught rather than accommodated.

**``Snapshot.nx_max``/``ny_max``/``nz_max`` are mutable *class* attributes**,
written by ``htod_snapshot_array``. Without the autouse reset below, one test
leaks into the next and the failure surfaces somewhere unrelated.
"""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.constants import c as C_LIGHT
from scipy.constants import epsilon_0, mu_0

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material

# Uniform spatial discretisation shared by most tests.
DL = 0.001

# Anisotropic spacing, so a test reading the wrong axis cannot pass by luck.
DL_ANISO = (0.001, 0.002, 0.004)

# A fixed time step, used for the snapshot ``time`` attribute.
DT = 1e-12

# The six field components, in the order every writer iterates them.
FIELDS = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


@pytest.fixture(autouse=True)
def outputs_config(monkeypatch, tmp_path, request):
    """Patch ``gprMax.config`` for the output modules.

    ``output_file_path`` and ``input_file_path`` point into ``tmp_path`` so
    that any test which lets a filename be derived writes somewhere harmless.
    """
    if request.node.get_closest_marker("unit") is None:
        return

    from gprMax import config

    snapshot_dir = tmp_path / "snapshots"

    model_cfg = SimpleNamespace(
        mode="3D",
        ompthreads=1,
        appendmodelnumber="",
        output_file_path=tmp_path / "model",
        output_file_path_ext=tmp_path / "model.h5",
        device={"snapsgpu2cpu": False, "dev": None},
        set_snapshots_dir=lambda: snapshot_dir,
        materials={
            "maxpoles": 0,
            "dispersivedtype": np.complex128,
            "dispersiveCdtype": None,
            "drudelorentz": None,
            "crealfunc": None,
        },
        numdispersion={
            "highestfreqthres": 40,
            "maxnumericaldisp": 2,
            "mingridsampling": 3,
        },
    )
    sim_cfg = SimpleNamespace(
        general={"solver": "cpu", "precision": "double", "subgrid": False, "progressbars": False},
        dtypes={"float_or_double": np.float64, "complex": np.complex128},
        em_consts={
            "c": C_LIGHT,
            "e0": epsilon_0,
            "m0": mu_0,
            "z0": np.sqrt(mu_0 / epsilon_0),
        },
        input_file_path=tmp_path / "model.in",
        args=SimpleNamespace(autotranslate=False, geometry_only=False),
        geometry_fixed=False,
        number_of_models=1,
        current_model=0,
        model_end=1,
        study=None,
    )

    monkeypatch.setattr(config, "sim_config", sim_cfg)
    monkeypatch.setattr(config, "get_model_config", lambda: model_cfg)

    return SimpleNamespace(sim_config=sim_cfg, model_config=model_cfg)


@pytest.fixture(autouse=True)
def reset_snapshot_class_state(request):
    """Restore ``Snapshot``'s mutable class attributes after every test.

    ``htod_snapshot_array`` writes ``Snapshot.nx_max`` and friends on the
    *class*, not the instance, so they persist for the rest of the session.
    """
    if request.node.get_closest_marker("unit") is None:
        yield
        return

    from gprMax.snapshots import Snapshot

    saved = (Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max, Snapshot.bpg)
    yield
    Snapshot.nx_max, Snapshot.ny_max, Snapshot.nz_max, Snapshot.bpg = saved


def nonzero_set(arr):
    """Set of index tuples at which ``arr`` is nonzero.

    The idiom carried over from the geometry-primitives, fractals and grid
    suites: every "which cells were written" assertion compares one of these
    against an expected set.
    """
    return set(map(tuple, np.argwhere(np.asarray(arr))))


def ramp(shape, dtype=np.float64):
    """An array whose value is its own flat index.

    Every element is distinct, so an assertion about *which* cells were read
    cannot be satisfied by the wrong ones holding the same value.
    """
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


@pytest.fixture
def make_materials():
    """Factory for a short, ordered list of ``Material`` objects.

    ``GridView.initialise_materials`` sorts this list and builds a
    ``numID -> index`` map from it, so the ``numID`` values must match the
    positions the ``ID`` array will reference.
    """

    def _make(count=3):
        names = ["pec", "free_space", "sand", "water", "concrete", "clay"]
        materials = []
        for i in range(count):
            m = Material(i, names[i % len(names)])
            m.er = 1.0 + i
            m.se = 0.0
            m.mr = 1.0
            m.sm = 0.0
            materials.append(m)
        return materials

    return _make


@pytest.fixture
def make_view_grid(make_materials):
    """Factory for a real ``FDTDGrid`` suitable for viewing and exporting.

    A genuine grid rather than a stub: ``GridView`` slices its ``solid``,
    ``ID``, ``rigidE``, ``rigidH`` and six field arrays directly, and the
    snapshot kernel reads the field arrays through typed memoryviews that a
    ``SimpleNamespace`` could not satisfy.

    Field arrays are filled with a distinct-per-cell ramp by default so that
    slicing assertions can name exactly which cells were read.
    """

    def _make(nx=8, ny=8, nz=8, dl=DL, dt=DT, materials=3, fill=True, name="main_grid"):
        g = FDTDGrid()
        g.name = name
        g.size = np.array([nx, ny, nz], dtype=np.int64)
        if np.isscalar(dl):
            g.dl = np.array([dl, dl, dl], dtype=np.float64)
        else:
            g.dl = np.array(dl, dtype=np.float64)
        g.dt = dt
        g.iterations = 5
        g.materials = make_materials(materials)
        g.initialise_geometry_arrays()
        g.initialise_field_arrays()
        g.initialise_std_update_coeff_arrays()
        if fill:
            for name_ in FIELDS:
                arr = getattr(g, name_)
                arr[...] = ramp(arr.shape, arr.dtype)
        return g

    return _make


@pytest.fixture
def make_grid_view(make_view_grid):
    """Factory for a ``GridView`` over a freshly built grid.

    Defaults to a step of one over the whole domain; pass ``step`` to
    exercise strided views, including deliberately non-dividing ones.
    """
    from gprMax.geometry_outputs.grid_view import GridView

    def _make(start=(0, 0, 0), stop=None, step=(1, 1, 1), grid=None, **grid_kwargs):
        g = grid if grid is not None else make_view_grid(**grid_kwargs)
        if stop is None:
            stop = tuple(int(v) for v in g.size)
        return GridView(g, *start, *stop, *step)

    return _make


class FakeMPIGrid:
    """A grid stand-in carrying a **real** MPI communicator.

    ``MPIGridView.__init__`` asserts ``isinstance(comm, MPI.Intracomm)``, so a
    mock communicator is rejected outright. Handing it a genuine
    ``MPI.COMM_SELF`` while faking the grid's own methods lets the halo
    clamping and offset arithmetic — which depend on ``negative_halo_offset``
    and ``size``, not on rank count — be driven exactly.
    """

    def __init__(self, comm, size, negative_halo_offset, origin, grid_coord, arrays=None):
        self.name = "mpi_grid"
        self.comm = comm
        self.size = np.array(size, dtype=np.int32)
        self.global_size = np.array(size, dtype=np.int32)
        self.negative_halo_offset = np.array(negative_halo_offset, dtype=np.int32)
        self._origin = np.array(origin, dtype=np.int32)
        self._grid_coord = np.array(grid_coord, dtype=np.int32)
        self.dl = np.array([DL, DL, DL], dtype=np.float64)
        self.dt = DT
        self.materials = []
        for name, array in (arrays or {}).items():
            setattr(self, name, array)

    def local_to_global_coordinate(self, coord):
        return np.asarray(coord) + self._origin

    def get_grid_coord_from_coordinate(self, coord):
        return np.full(3, self._grid_coord[0], dtype=np.int32)

    def local_bounds_overlap_grid(self, start, stop):
        return bool(np.all(np.asarray(stop) > 0) and np.all(np.asarray(start) < self.size))


@pytest.fixture
def make_mpi_grid():
    """Factory for :class:`FakeMPIGrid` on a real single-rank communicator."""
    from mpi4py import MPI

    def _make(
        size=(10, 10, 10),
        negative_halo_offset=(2, 2, 2),
        origin=(100, 100, 100),
        grid_coord=(0, 0, 0),
        comm=None,
        arrays=None,
    ):
        return FakeMPIGrid(
            comm if comm is not None else MPI.COMM_SELF,
            size,
            negative_halo_offset,
            origin,
            grid_coord,
            arrays,
        )

    return _make


@pytest.fixture
def read_h5():
    """Open a written HDF5 file and hand back its attributes and datasets.

    Returns ``(attrs, datasets)`` where both are plain dicts, so round-trip
    assertions read as data comparisons rather than h5py mechanics. Groups are
    flattened to ``"parent/child"`` keys.
    """

    def _read(path):
        attrs = {}
        datasets = {}

        def visit(name, obj):
            for key, value in obj.attrs.items():
                attrs[f"{name}/{key}"] = value
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[()]

        with h5py.File(Path(path), "r") as f:
            attrs.update(dict(f.attrs))
            f.visititems(visit)
        return attrs, datasets

    return _read


@pytest.fixture
def make_rx():
    """Factory for a receiver with an explicit ID and output time series.

    ``Rx.__init__`` only *annotates* ``self.ID``, never assigns it, so a
    receiver that has not been named raises ``AttributeError`` the moment
    ``write_hd5_data`` sorts the list. Every receiver built here is named.
    """
    from gprMax.receivers import Rx

    def _make(ID="rx1", position=(1, 2, 3), outputs=("Ex", "Ey"), iterations=5):
        rx = Rx()
        rx.ID = ID
        rx.xcoord, rx.ycoord, rx.zcoord = position
        rx.xcoordorigin, rx.ycoordorigin, rx.zcoordorigin = position
        rx.outputs = {name: np.zeros(iterations, dtype=np.float64) for name in outputs}
        return rx

    return _make


@pytest.fixture
def make_tl():
    """Factory for a transmission-line stand-in.

    ``fields_outputs`` reads eleven attributes off a transmission line and
    calls none of its methods, so a namespace is enough here.
    """

    def _make(position=(2, 2, 2), resistance=50.0, dl=DL, iterations=5, antpos=1):
        return SimpleNamespace(
            ID="tl1",
            xcoord=position[0],
            ycoord=position[1],
            zcoord=position[2],
            coord=position,
            polarisation="x",
            start=0.0,
            stop=iterations * DT,
            waveformID="wf",
            waveformvalues_wholedt=np.zeros(iterations + 1),
            waveformvalues_halfdt=np.zeros(iterations + 1),
            resistance=resistance,
            dl=dl,
            antpos=antpos,
            Vinc=np.arange(iterations, dtype=np.float64),
            Iinc=np.arange(iterations, dtype=np.float64) * 2,
            Vtotal=np.zeros(iterations, dtype=np.float64),
            Itotal=np.zeros(iterations, dtype=np.float64),
            voltage=np.arange(10, dtype=np.float64) * 3,
            current=np.arange(10, dtype=np.float64) * 5,
        )

    return _make


@pytest.fixture
def null_pbar():
    """A progress-bar stand-in that records the byte counts pushed to it.

    Every writer takes a ``tqdm`` and calls ``update(n=...)`` as it goes;
    capturing those totals lets the tests check the ``nbytes`` bookkeeping
    without constructing a real bar.
    """

    class NullPbar:
        def __init__(self):
            self.updates = []

        def update(self, n=1):
            self.updates.append(n)

        @property
        def total(self):
            return sum(self.updates)

        def close(self):
            pass

    return NullPbar()
