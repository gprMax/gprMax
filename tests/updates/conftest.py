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

"""Shared fixtures for the update-dispatcher test suite.

``cpu_updates.py`` reads the owning grid's pole count and dispersive dtype,
plus the model's threading, mode, and configured precision:

- ``config.get_model_config().ompthreads`` — passed straight to every kernel;
- ``grid.maxpoles`` — selects the plain or
  dispersive branch in ``update_electric_a`` and gates ``update_electric_b``;
- ``grid.dispersivedtype`` and
  ``config.sim_config.dtypes["complex"]`` — compared against each other to
  choose a ``real`` or ``complex`` kernel;
- ``config.sim_config.general["precision"]`` — chooses ``float`` or ``double``.

``updates.py`` reads no configuration at all.

**The patching trap.** ``cpu_updates`` binds the Cython kernels as module
globals at import::

    from gprMax.cython.fields_updates_normal import update_electric as update_electric_cpu

so a test that wants to intercept a kernel must patch
``gprMax.updates.cpu_updates.update_electric_cpu``. Patching
``gprMax.cython.fields_updates_normal.update_electric`` has no effect — the
name was already resolved. The same applies to ``update_magnetic_cpu``,
``store_outputs_cpu`` and ``timer``.

Two grid fixtures, for two different jobs. ``make_wiring_grid`` is a
``SimpleNamespace`` of recorders and is what most tests want: the questions
being asked are "which kernel, with which arguments, in which order", and none
of those need real arrays. ``make_kernel_grid`` builds a genuine ``FDTDGrid``
for the handful of tests that run a real kernel. It is deliberately tiny —
the upstream sketch in ``tests/updates/`` uses a 100³ grid and pays about
15 seconds for 25 tests, which does not scale to a suite this size.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material

# A small grid. Every kernel-level assertion here is about which cells were
# touched, not about physics, so the smallest grid that still has an interior
# is the right size.
NX, NY, NZ = 4, 5, 6

DL = 0.001
DT = 1e-12


@pytest.fixture(autouse=True)
def updates_config(monkeypatch, request):
    """Patch ``gprMax.config`` for the update modules.

    Double precision throughout, one OpenMP thread, and no dispersive
    materials — the last of these means ``update_electric_a`` takes its plain
    branch unless a test says otherwise, which is the common case in
    production too.
    """
    if request.node.get_closest_marker("unit") is None:
        return

    from gprMax import config

    model_cfg = SimpleNamespace(
        mode="3D",
        ompthreads=1,
        materials={
            "maxpoles": 0,
            "dispersivedtype": None,
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
        general={
            "solver": "cpu",
            "precision": "double",
            "subgrid": False,
            "progressbars": False,
        },
        dtypes={"float_or_double": np.float64, "complex": np.complex128},
        current_model=0,
        model_end=1,
    )

    monkeypatch.setattr(config, "sim_config", sim_cfg)
    monkeypatch.setattr(config, "get_model_config", lambda: model_cfg)

    return SimpleNamespace(sim_config=sim_cfg, model_config=model_cfg)


class Recorder:
    """Callable that records every call made to it.

    Used for kernels, sources and PML slabs alike. ``calls`` holds
    ``(args, kwargs)`` pairs; ``args_of(n)`` is a convenience for the common
    "what did the nth call receive" assertion.
    """

    def __init__(self, name="recorder", result=None):
        self.name = name
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    @property
    def call_count(self):
        return len(self.calls)

    def args_of(self, index=0):
        return self.calls[index][0]

    def kwargs_of(self, index=0):
        return self.calls[index][1]


@pytest.fixture
def recorder():
    """Factory for :class:`Recorder`."""

    def _make(name="recorder", result=None):
        return Recorder(name=name, result=result)

    return _make


class SourceRecorder:
    """Stand-in for a source object.

    Sources are only ever asked to update themselves, so a recorder per
    method is enough. ``log`` is shared across every source built by one
    factory call, which is how ordering across the concatenated source lists
    is asserted.
    """

    def __init__(self, ID, log, dispersive=False):
        self.ID = ID
        self.dispersive = dispersive
        self._log = log
        self.electric_calls = []
        self.magnetic_calls = []
        self.plane_wave_calls = []

    def update_electric(self, *args):
        self._log.append(f"E:{self.ID}")
        self.electric_calls.append(args)

    def update_magnetic(self, *args):
        self._log.append(f"H:{self.ID}")
        self.magnetic_calls.append(args)

    def update_plane_wave_electric(self, *args, **kwargs):
        self._log.append(f"PWE:{self.ID}")
        self.plane_wave_calls.append((args, kwargs))

    def update_plane_wave_electric_dispersive(self, *args, **kwargs):
        self._log.append(f"PWEd:{self.ID}")
        self.plane_wave_calls.append((args, kwargs))

    def update_plane_wave_magnetic(self, *args, **kwargs):
        self._log.append(f"PWH:{self.ID}")
        self.plane_wave_calls.append((args, kwargs))


class PmlRecorder:
    """Stand-in for a PML slab; both update methods take no arguments."""

    def __init__(self, ID, log):
        self.ID = ID
        self._log = log

    def update_electric(self):
        self._log.append(f"pmlE:{self.ID}")

    def update_magnetic(self):
        self._log.append(f"pmlH:{self.ID}")


class SnapshotRecorder:
    """Stand-in for a snapshot; ``store()`` takes no arguments."""

    def __init__(self, iteration, log):
        self.iteration = iteration
        self._log = log
        self.store_count = 0

    def store(self):
        self.store_count += 1
        self._log.append(f"snap:{self.iteration}")


@pytest.fixture
def make_wiring_grid(updates_config):
    """Factory for a recorder-backed grid stand-in.

    Supplies every attribute ``CPUUpdates`` reads. The field and coefficient
    arrays are unique sentinel objects rather than numpy arrays, so an
    assertion that a kernel received ``grid.Ex`` is an identity check and
    cannot pass by coincidence.
    """

    def _make(
        nx=NX,
        ny=NY,
        nz=NZ,
        voltagesources=(),
        transmissionlines=(),
        hertziandipoles=(),
        magneticdipoles=(),
        magneticfrillsources=(),
        discreteplanewaves=(),
        pml_slabs=(),
        snapshots=(),
        log=None,
        **extra,
    ):
        if log is None:
            log = []

        ns = SimpleNamespace(
            nx=nx,
            ny=ny,
            nz=nz,
            # Sentinels: distinct objects, identity-comparable.
            ID=f"ID-{id(log)}",
            Ex="Ex",
            Ey="Ey",
            Ez="Ez",
            Hx="Hx",
            Hy="Hy",
            Hz="Hz",
            Tx="Tx",
            Ty="Ty",
            Tz="Tz",
            updatecoeffsE="updatecoeffsE",
            updatecoeffsH="updatecoeffsH",
            updatecoeffsdispersive="updatecoeffsdispersive",
            maxpoles=updates_config.model_config.materials["maxpoles"],
            dispersivedtype=updates_config.model_config.materials["dispersivedtype"],
            voltagesources=list(voltagesources),
            transmissionlines=list(transmissionlines),
            hertziandipoles=list(hertziandipoles),
            magneticdipoles=list(magneticdipoles),
            magneticfrillsources=list(magneticfrillsources),
            discreteplanewaves=list(discreteplanewaves),
            pmls={"slabs": list(pml_slabs)},
            snapshots=list(snapshots),
            rxs=[],
            ntff_monitors=[],
            log=log,
        )
        for key, value in extra.items():
            setattr(ns, key, value)
        return ns

    return _make


@pytest.fixture
def make_source():
    """Factory for :class:`SourceRecorder` sharing one ordering log."""

    def _make(ID, log, dispersive=False):
        return SourceRecorder(ID, log, dispersive=dispersive)

    return _make


@pytest.fixture
def make_pml_slab():
    """Factory for :class:`PmlRecorder`."""

    def _make(ID, log):
        return PmlRecorder(ID, log)

    return _make


@pytest.fixture
def make_snapshot():
    """Factory for :class:`SnapshotRecorder`."""

    def _make(iteration, log):
        return SnapshotRecorder(iteration, log)

    return _make


@pytest.fixture
def make_kernel_grid():
    """Factory for a real ``FDTDGrid`` sized for the compiled kernels.

    Only the tests that actually run ``update_electric`` / ``update_magnetic``
    need this. The arrays must be C-contiguous and match
    ``dtypes["float_or_double"]`` exactly or Cython raises a buffer dtype
    mismatch, which is itself worth one test.

    ``ID`` initialises to 1 in ``FDTDGrid``, so the material list needs at
    least two entries for that index to be valid.
    """

    def _make(nx=NX, ny=NY, nz=NZ, dl=DL, dt=DT):
        g = FDTDGrid()
        g.nx, g.ny, g.nz = nx, ny, nz
        g.dx = g.dy = g.dz = dl
        g.dt = dt
        g.materials = [Material(0, "pec"), Material(1, "free_space")]
        g.initialise_geometry_arrays()
        g.initialise_field_arrays()
        g.initialise_std_update_coeff_arrays()
        return g

    return _make


@pytest.fixture
def ramped_grid(make_kernel_grid):
    """A real grid whose fields vary in space, so no curl cancels.

    Filling a field uniformly is the obvious thing to do and is useless: the
    update kernels take *differences* of neighbouring samples, so a constant
    field produces a zero curl and the updated component stays at zero. Every
    "which cells were written" assertion would then see an empty array.

    Each component gets a linear ramp with a distinct multiplier. The ramps
    differ per component, so the two terms of a curl cannot cancel, and every
    cell inside the updated region ends up non-zero.

    Only the *source* field is filled. ``fill="E"`` ramps the electric
    components and leaves the magnetic ones at zero, so after
    ``update_magnetic`` a simple ``H != 0`` mask is exactly the set of cells
    the kernel wrote. ``fill="H"`` is the mirror image, for the electric
    update.

    All five update coefficients are set to 1 for the one non-PEC material, so
    the arithmetic is a plain sum of differences.
    """

    def _make(fill="E", nx=NX, ny=NY, nz=NZ):
        grid = make_kernel_grid(nx=nx, ny=ny, nz=nz)
        shape = grid.Ex.shape
        size = int(np.prod(shape))
        names = ("Ex", "Ey", "Ez") if fill == "E" else ("Hx", "Hy", "Hz")
        for index, name in enumerate(names, start=1):
            ramp = np.arange(size, dtype=np.float64).reshape(shape) * index + index
            getattr(grid, name)[:] = ramp
        grid.updatecoeffsE[1] = 1.0
        grid.updatecoeffsH[1] = 1.0
        return grid

    return _make


@pytest.fixture
def nonzero_set():
    """Set of index tuples at which an array is non-zero.

    Reused verbatim from the geometry-primitives, fractals, grid and outputs
    suites — the shared idiom for "which cells did this touch".
    """

    def _nonzero(arr):
        return set(zip(*np.nonzero(arr)))

    return _nonzero
