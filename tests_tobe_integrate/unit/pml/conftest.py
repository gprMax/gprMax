"""Shared fixtures for the PML test suite.

``pml.py`` reads a strikingly small slice of global configuration — four
keys, and that is the lot:

- ``config.sim_config.em_consts["z0"]`` in ``CFS.calculate_sigmamax``;
- ``config.sim_config.em_consts["e0"]`` throughout
  ``PML.calculate_update_coeffs``;
- ``config.sim_config.dtypes["float_or_double"]`` in every array allocation;
- ``config.get_model_config().ompthreads``, passed straight through to the
  Cython update kernels.

``config.sim_config`` is ``None`` until a real run initialises it, so the
autouse fixture below is mandatory for anything past ``CFSParameter()``.

One construction detail governs every fixture here. ``PML.__init__`` calls
``check_kappamin()``, which sums ``kappa.min`` over the grid's CFS list and
rejects a total below one — so a grid whose ``pmls["cfs"]`` is empty cannot
build a PML at all. ``FDTDGrid.build()`` installs a default ``CFS()`` before
constructing any slab; ``make_pml_grid`` does the same.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0
from scipy.constants import c as C_LIGHT

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import Material
from gprMax.pml import CFS, PML

# Uniform spatial discretisation shared by most tests.
DL = 0.001

# A deliberately anisotropic discretisation. The three values are distinct
# and not multiples of each other, so a test that reads the wrong axis of
# ``dl`` when picking ``PML.d`` cannot accidentally still pass.
DL_ANISO = (0.001, 0.002, 0.004)

# A fixed time step. Real models derive this from the CFL limit, but every
# coefficient formula here is linear in ``dt``, so a round number keeps the
# hand-computed expectations readable.
DT = 1e-12

# Maps ``PML.boundaryIDs`` to the ``PML.directions`` entry each slab uses.
# Absorption increases *away* from the domain, so the low-side slabs point
# in the negative direction.
ID_TO_DIRECTION = {
    "x0": "xminus",
    "y0": "yminus",
    "z0": "zminus",
    "xmax": "xplus",
    "ymax": "yplus",
    "zmax": "zplus",
}


@pytest.fixture(autouse=True)
def pml_config(monkeypatch):
    """Patch ``gprMax.config`` for the PML modules.

    Double precision arrays and a single OpenMP thread. The electromagnetic
    constants are the real ones from ``scipy.constants`` so that the
    closed-form coefficient assertions are checking gprMax's algebra rather
    than a made-up value of ``e0``.
    """
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


@pytest.fixture
def make_cfs():
    """Factory for a ``CFS`` with any of its three parameters overridden.

    Each keyword takes a dict of ``CFSParameter`` attributes to set, e.g.
    ``make_cfs(kappa={"max": 5, "scalingprofile": "linear"})``. Anything not
    named keeps the stock default, which matters: the defaults switch
    ``alpha`` and ``kappa`` off entirely, so a test that wants those terms to
    contribute has to say so.
    """

    def _make(alpha=None, kappa=None, sigma=None):
        cfs = CFS()
        for parameter, overrides in (
            (cfs.alpha, alpha),
            (cfs.kappa, kappa),
            (cfs.sigma, sigma),
        ):
            for name, value in (overrides or {}).items():
                setattr(parameter, name, value)
        return cfs

    return _make


@pytest.fixture
def make_pml_grid():
    """Factory for a real ``FDTDGrid`` configured to carry PML slabs.

    A genuine grid rather than a stub, because ``PML`` reads ``dx``/``dy``/
    ``dz`` and ``dt`` off it and the update methods pass its field arrays
    straight into Cython. ``cfs=None`` installs a single default ``CFS()``;
    pass an explicit list for multi-pole tests, or ``[]`` to exercise the
    ``check_kappamin`` rejection.
    """

    def _make(nx=10, ny=10, nz=10, dl=DL, dt=DT, cfs=None, formulation="HORIPML", arrays=True):
        g = FDTDGrid()
        g.size = np.array([nx, ny, nz], dtype=np.int64)
        if np.isscalar(dl):
            g.dl = np.array([dl, dl, dl], dtype=np.float64)
        else:
            g.dl = np.array(dl, dtype=np.float64)
        g.dt = dt
        g.pmls["formulation"] = formulation
        g.pmls["cfs"] = [CFS()] if cfs is None else cfs
        if arrays:
            # A free-space material so ``initialise_std_update_coeff_arrays``
            # produces a (1, 5) array rather than the degenerate (0, 5) an
            # empty list would give — the update kernels index it through
            # ``ID``, which initialises to 1.
            g.materials = [Material(0, "pec"), Material(1, "free_space")]
            g.initialise_geometry_arrays()
            g.initialise_field_arrays()
            g.initialise_std_update_coeff_arrays()
        return g

    return _make


@pytest.fixture
def make_pml(make_pml_grid):
    """Factory for a constructed ``PML`` slab.

    ``thickness`` sizes the slab along its own normal; the other two axes span
    the whole face, as ``FDTDGrid._construct_pml`` arranges in production.
    Returns the ``PML``; reach its grid through ``pml.G``.
    """

    def _make(pml_id="x0", thickness=4, grid=None, **grid_kwargs):
        g = grid if grid is not None else make_pml_grid(**grid_kwargs)
        nx, ny, nz = (int(v) for v in g.size)
        extents = {
            "x0": (0, thickness, 0, ny + 1, 0, nz + 1),
            "xmax": (nx - thickness, nx, 0, ny + 1, 0, nz + 1),
            "y0": (0, nx + 1, 0, thickness, 0, nz + 1),
            "ymax": (0, nx + 1, ny - thickness, ny, 0, nz + 1),
            "z0": (0, nx + 1, 0, ny + 1, 0, thickness),
            "zmax": (0, nx + 1, 0, ny + 1, nz - thickness, nz),
        }[pml_id]
        xs, xf, ys, yf, zs, zf = extents
        return PML(g, pml_id, ID_TO_DIRECTION[pml_id], xs, xf, ys, yf, zs, zf)

    return _make
