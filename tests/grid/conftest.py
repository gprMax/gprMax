"""Shared fixtures for the grid test suite.

``FDTDGrid`` is unusual among the classes tested so far: it constructs with
no global configuration at all. ``__init__`` only allocates small bookkeeping
containers and calls ``set_pml_thickness(10)``.

Configuration is read later, and only by three groups of methods:

- the array initialisers, which need
  ``config.sim_config.dtypes["float_or_double"]`` and, for the dispersive
  arrays, ``config.get_model_config().materials``;
- ``calculate_dt``, which needs ``config.sim_config.em_consts["c"]`` and
  ``config.get_model_config().mode``;
- ``build()`` / ``dispersion_analysis``, which additionally read
  ``ompthreads``, ``numdispersion`` and ``sim_config.general``.

``config.sim_config`` is ``None`` until a real run initialises it, so the
autouse fixture below is mandatory for anything beyond bare construction.
``config.c`` is a plain module-level constant and needs no patching.

All spatial tests use a uniform discretisation of ``DL`` (1 mm) unless they
are specifically exercising anisotropic spacing, so cell index ``i`` maps to
coordinate ``i * DL``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c as C_LIGHT
from scipy.constants import epsilon_0, mu_0

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material

# Uniform spatial discretisation shared by most tests.
DL = 0.001

# A deliberately anisotropic discretisation. The three values are distinct
# and not multiples of each other, so a test that reads the wrong axis of
# ``dl`` cannot accidentally still pass.
DL_ANISO = (0.001, 0.002, 0.004)


@pytest.fixture(autouse=True)
def grid_config(monkeypatch, request):
    """Patch ``gprMax.config`` for the grid modules.

    Double precision arrays, a single OpenMP thread, 3D mode, no dispersive
    poles, and the stock numerical-dispersion thresholds.
    """
    if request.node.get_closest_marker("unit") is None:
        return

    from gprMax import config

    model_cfg = SimpleNamespace(
        mode="3D",
        ompthreads=1,
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
        args=SimpleNamespace(autotranslate=False),
        current_model=0,
        model_end=1,
    )

    monkeypatch.setattr(config, "sim_config", sim_cfg)
    monkeypatch.setattr(config, "get_model_config", lambda: model_cfg)

    return SimpleNamespace(sim_config=sim_cfg, model_config=model_cfg)


def nonzero_set(arr):
    """Set of index tuples at which ``arr`` is nonzero.

    The idiom carried over from the geometry-primitives and fractals suites:
    every "which cells were written" assertion compares one of these against
    an expected set.
    """
    return set(map(tuple, np.argwhere(np.asarray(arr))))


@pytest.fixture
def make_grid():
    """Factory for a real ``FDTDGrid``.

    Unlike the geometry-primitives and fractals suites, which had to stub the
    grid, here the grid *is* the class under test, so a genuine instance is
    constructed. Geometry and field arrays are allocated by default because
    most tests need them; pass ``arrays=False`` to inspect the bare
    constructor state.
    """

    def _make(nx=8, ny=8, nz=8, dl=DL, arrays=True, pml_thickness=None):
        g = FDTDGrid()
        g.size = np.array([nx, ny, nz], dtype=np.int64)
        if np.isscalar(dl):
            g.dl = np.array([dl, dl, dl], dtype=np.float64)
        else:
            g.dl = np.array(dl, dtype=np.float64)
        if pml_thickness is not None:
            g.set_pml_thickness(pml_thickness)
        # Upstream now requires 'free_space' material for geometry arrays.
        g.materials = [Material(0, "pec"), Material(1, "free_space")]
        if arrays:
            g.initialise_geometry_arrays()
            g.initialise_field_arrays()
        return g

    return _make
