"""Auto-applied fixtures for the user-objects test suite.

The user-object classes under ``gprMax/user_objects/`` read pieces of
global config and grid state at ``build()`` time. Per-test we monkeypatch
``gprMax.config.sim_config`` to a predictable cpu-solver/double-precision
environment, and supply tiny ``SimpleNamespace`` stubs in place of the
real ``FDTDGrid`` and ``Model`` for ``build()`` smoke-tests.

These tests never construct a real ``FDTDGrid`` — the goal is to exercise
the constructor → attribute mirroring contract and the ``build()``
validation branches, not to drive an FDTD simulation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0


@pytest.fixture(autouse=True)
def user_object_config(monkeypatch, tmp_path):
    """Patch ``gprMax.config`` so ``build()`` calls run in isolation.

    Defaults to cpu solver, double-precision floats, free-space EM
    constants, and a ``model_config`` whose mode/output-path attribute
    assignments are no-ops. Subgrid disabled by default. Tests can
    override any value via ``monkeypatch.setattr`` in the test body.
    """
    from gprMax import config

    model_cfg = SimpleNamespace(
        mode="3D",
        requested_2d_mode=None,
        ompthreads=1,
        materials={"maxpoles": 0},
        set_output_file_path=MagicMock(),
    )

    sim_cfg = SimpleNamespace(
        general={"solver": "cpu", "precision": "double", "subgrid": False},
        dtypes={"float_or_double": np.float64},
        em_consts={
            "c": c,
            "e0": epsilon_0,
            "m0": mu_0,
            "z0": float(np.sqrt(mu_0 / epsilon_0)),
        },
        args=SimpleNamespace(autotranslate=False),
        input_file_path=tmp_path / "fake_input.in",
    )

    monkeypatch.setattr(config, "sim_config", sim_cfg)
    monkeypatch.setattr(config, "get_model_config", lambda: model_cfg)
    # Also attach get_model_config to sim_cfg itself (some paths access it directly).
    sim_cfg.get_model_config = lambda: model_cfg
    monkeypatch.setattr(config, "c", c, raising=False)
    monkeypatch.setattr(config, "e0", epsilon_0, raising=False)
    monkeypatch.setattr(config, "m0", mu_0, raising=False)

    return SimpleNamespace(sim_config=sim_cfg, model_config=model_cfg)


def make_material(numID, ID, er=1.0, se=0.0, mr=1.0, sm=0.0, averagable=True):
    """Build a stub ``Material``-shaped object.

    The user-object ``build()`` methods only read a handful of attributes
    on each grid material (``ID``, ``numID``, ``averagable``, ``er``,
    ``se``, ``mr``, ``sm``) — a ``SimpleNamespace`` is enough.
    """
    return SimpleNamespace(
        numID=numID,
        ID=ID,
        er=er,
        se=se,
        mr=mr,
        sm=sm,
        averagable=averagable,
    )


def make_waveform(wid):
    """Build a stub waveform with only the ``.ID`` attribute set."""
    return SimpleNamespace(ID=wid)


@pytest.fixture
def stub_grid():
    """Minimal grid stub for ``GridUserObject.build()`` calls.

    Carries just enough state for the validation branches in the
    user-object ``build()`` methods to run: discretisation, time window,
    a free-space material, an empty waveform list, and PML thickness
    defaults. Append to ``materials``/``waveforms`` in the test body to
    pre-populate the grid.
    """
    grid = SimpleNamespace()
    grid.dx = grid.dy = grid.dz = 0.001
    grid.dl = np.array([0.001, 0.001, 0.001])
    grid.dt = 1.927e-12
    grid.timewindow = 1e-9
    grid.iterations = 100
    grid.nx = grid.ny = grid.nz = 50
    grid.size = np.array([50, 50, 50])
    grid.averagevolumeobjects = True
    grid.materials = [
        make_material(0, "pec", er=1.0, se=float("inf"), averagable=False),
        make_material(1, "free_space"),
    ]
    grid.waveforms = []
    grid.mixingmodels = []
    grid.discreteplanewaves = []
    grid.pmls = {
        "formulation": "HORIPML",
        "global_formulation_set": False,
        "thickness": {"x0": 10, "y0": 10, "z0": 10, "xmax": 10, "ymax": 10, "zmax": 10},
        "cfs": [],
        "profiles": {},
    }
    grid.add_source = MagicMock()
    grid.add_receiver = MagicMock()
    grid.set_pml_thickness = MagicMock()
    grid.calculate_dt = MagicMock()
    grid.within_bounds = MagicMock()
    return grid


@pytest.fixture
def stub_model(stub_grid):
    """Minimal model stub for ``ModelUserObject.build()`` calls."""
    model = SimpleNamespace()
    model.G = stub_grid
    model.nx = model.ny = model.nz = stub_grid.nx
    model.cells = model.nx * model.ny * model.nz
    model.dl = stub_grid.dl
    model.dx = stub_grid.dx
    model.dy = stub_grid.dy
    model.dz = stub_grid.dz
    model.dt = stub_grid.dt
    model.dt_mod = 1.0
    model.timewindow = stub_grid.timewindow
    model.iterations = stub_grid.iterations
    model.title = ""
    model.srcsteps = np.zeros(3, dtype=np.int32)
    model.rxsteps = np.zeros(3, dtype=np.int32)
    model.set_size = MagicMock()
    model.add_snapshot = MagicMock()
    model.add_geometry_view_voxels = MagicMock()
    model.add_geometry_view_lines = MagicMock()
    model.add_geometry_object = MagicMock()
    return model
