"""Shared fixtures for the subgrid test suite.

A Huygens subgrid is defined almost entirely by arithmetic on one number,
``ratio``. The fixtures here keep that arithmetic explicit so the tests can
assert against hand-computed values rather than against the code's own
derivation.

Sizes are chosen small but *consistent*: the main grid must be large enough to
contain the subgrid's Inner Surface plus the ``is_os_sep`` margin, and the
precursor slices reach one main cell outside the IS (``i0 - 1``), so the IS
cannot sit on the domain boundary. The spatial interpolation is a
``RectBivariateSpline`` of degree ``interpolation``, which needs at least
``degree + 1`` samples per axis, so the working region cannot be a single cell
either.

With the defaults below:

    s_is_os_sep      = is_os_sep * ratio          = 1 * 3 =  3
    d_to_pml         = s_is_os_sep + pml_sep      = 3 + 2 =  5
    n_boundary_cells = d_to_pml + pml_thickness   = 5 + 2 =  7
    nwx              = (i1 - i0) * ratio          = 6 * 3 = 18
    nx               = 2 * n_boundary_cells + nwx = 14 + 18 = 32
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c as C_LIGHT
from scipy.constants import epsilon_0, mu_0

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.subgrids.precursor_nodes import PrecursorNodes, PrecursorNodesFiltered
from gprMax.subgrids.subgrid_hsg import SubGridHSG as SubGridHSGGrid

DL = 0.001

# Main-grid indices of the subgrid's Inner Surface.
IS_LOWER = (10, 10, 10)
IS_UPPER = (16, 16, 16)

# Main grid size, comfortably containing the IS plus its OS margin.
MAIN_CELLS = 30


@pytest.fixture(autouse=True)
def subgrid_config(monkeypatch, request):
    """Patch ``gprMax.config`` for the subgrid modules.

    ``SubGridBaseGrid`` inherits ``FDTDGrid``, so the same three config
    surfaces are needed: array dtype, OpenMP thread count (passed straight
    into the HSG Cython kernels) and the model mode.
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
        general={"solver": "cpu", "precision": "double", "subgrid": True, "progressbars": False},
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
    """Set of index tuples at which ``arr`` is nonzero."""
    return set(map(tuple, np.argwhere(np.asarray(arr))))


@pytest.fixture
def subgrid_kwargs():
    """The eight keyword arguments ``SubGridBaseGrid.__init__`` requires.

    Returned fresh each time so a test can drop or override exactly one.
    """
    return {
        "ratio": 3,
        "id": "test_subgrid",
        "filter": True,
        "is_os_sep": 1,
        "pml_separation": 2,
        "subgrid_pml_thickness": 2,
        "interpolation": 1,
    }


@pytest.fixture
def make_subgrid(subgrid_kwargs, make_material):
    """Factory for a ``SubGridHSG`` grid with its sizes filled in.

    The grid class alone only computes the boundary-cell counts; the working
    region and total size are set by the *user object* during ``setup()``.
    This factory does the same by hand so grid-level tests do not need the
    whole command layer.

    With ``arrays=True`` the update-coefficient arrays are filled with ones.
    They must be non-zero for the HSG kernels to have any visible effect —
    the kernels multiply the incoming precursor value by a coefficient looked
    up through the ``ID`` array — and they must have at least as many rows as
    the highest material ID, hence the materials list.
    """

    def _make(is_lower=IS_LOWER, is_upper=IS_UPPER, arrays=False, **overrides):
        sg = SubGridHSGGrid(**{**subgrid_kwargs, **overrides})

        sg.i0, sg.j0, sg.k0 = is_lower
        sg.i1, sg.j1, sg.k1 = is_upper

        sg.nwx = (sg.i1 - sg.i0) * sg.ratio
        sg.nwy = (sg.j1 - sg.j0) * sg.ratio
        sg.nwz = (sg.k1 - sg.k0) * sg.ratio

        sg.nx = 2 * sg.n_boundary_cells_x + sg.nwx
        sg.ny = 2 * sg.n_boundary_cells_y + sg.nwy
        sg.nz = 2 * sg.n_boundary_cells_z + sg.nwz

        sg.dl = np.array([DL / sg.ratio] * 3)

        if arrays:
            sg.materials = [
                make_material(ID="pec", numID=0),
                make_material(ID="free_space", numID=1),
            ]
            sg.initialise_geometry_arrays()
            sg.initialise_field_arrays()
            sg.initialise_std_update_coeff_arrays()
            sg.updatecoeffsE[:] = 1.0
            sg.updatecoeffsH[:] = 1.0

        return sg

    return _make


@pytest.fixture
def make_main_grid(make_material):
    """A main ``FDTDGrid`` large enough to host the subgrid."""

    def _make(cells=MAIN_CELLS):
        g = FDTDGrid()
        g.size = np.array([cells, cells, cells], dtype=np.int64)
        g.dl = np.array([DL, DL, DL])
        g.materials = [
            make_material(ID="pec", numID=0),
            make_material(ID="free_space", numID=1),
        ]
        g.initialise_geometry_arrays()
        g.initialise_field_arrays()
        g.initialise_std_update_coeff_arrays()
        g.updatecoeffsE[:] = 1.0
        g.updatecoeffsH[:] = 1.0
        g.calculate_dt()
        return g

    return _make


@pytest.fixture
def coupled_grids(make_main_grid, make_subgrid):
    """A main grid and an HSG subgrid wired together, plus their precursors.

    This is the fixture the IS/OS tests depend on, so
    ``test_subgrid_hsg.py::TestCoupledGridsFixture`` asserts its consistency
    directly before any behavioural test relies on it.

    Returns a namespace with ``main``, ``sub`` and ``precursors``.
    """

    def _make(filtered=False, **overrides):
        main = make_main_grid()
        sub = make_subgrid(arrays=True, **overrides)
        sub.parent_grid = main
        sub.dt = main.dt / sub.ratio

        cls = PrecursorNodesFiltered if filtered else PrecursorNodes
        precursors = cls(main, sub)

        return SimpleNamespace(main=main, sub=sub, precursors=precursors)

    return _make


@pytest.fixture
def spy_updater():
    """Records the order in which a ``SubgridUpdater``'s steps are called.

    ``hsg_1`` / ``hsg_2`` are pure choreography — the interesting property is
    the *sequence* of sub-steps, not their numerical effect, so every step is
    replaced by a recorder.

    The three collaborators share method names — both the updater and the
    precursors have ``update_magnetic`` — so each recorded call is prefixed
    with the object it belongs to. Without that the counts silently conflate
    two different steps.
    """

    def _make(updater, precursors, subgrid):
        calls = []

        def record(target, name, prefix):
            setattr(target, name, lambda *a, **k: calls.append(f"{prefix}{name}"))

        for name in (
            "store_outputs",
            "update_electric_a",
            "update_electric_b",
            "update_electric_pml",
            "update_electric_sources",
            "update_magnetic",
            "update_magnetic_pml",
            "update_magnetic_sources",
        ):
            record(updater, name, "")

        for name in (
            "update_electric",
            "update_magnetic",
            "interpolate_magnetic_in_time",
            "interpolate_electric_in_time",
            "calc_exact_magnetic_in_time",
            "calc_exact_electric_in_time",
        ):
            record(precursors, name, "precursors.")

        for name in (
            "update_electric_is",
            "update_magnetic_is",
            "update_electric_os",
            "update_magnetic_os",
        ):
            record(subgrid, name, "sub.")

        return calls

    return _make
