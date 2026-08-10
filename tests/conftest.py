"""Shared pytest configuration for the gprMax test suite."""

import os
from types import SimpleNamespace

import pytest

from gprMax.materials import DispersiveMaterial, Material
from gprMax.waveforms import Waveform

# Some Linux MPI/OFI installations otherwise try to select an unavailable
# network provider while pytest imports gprMax. An explicit user setting is
# always preserved.
os.environ.setdefault("FI_PROVIDER", "shm")


def pytest_configure(config):
    """Ensure a minimal sim_config exists before any test imports.
    Prevents AttributeError: 'NoneType' object has no attribute 'get_model_config'
    in shared-folder tests that construct FDTDGrid before patching."""
    from types import SimpleNamespace
    import numpy as np
    import gprMax.config as cfg
    if cfg.sim_config is None:
        cfg.sim_config = SimpleNamespace(
            general={"solver": "cpu", "precision": "double", "subgrid": False, "progressbars": False},
            dtypes={"float_or_double": np.float64, "complex": np.complex128},
            current_model=0, model_end=1,
        )
        cfg.sim_config.get_model_config = lambda mc=SimpleNamespace(
            mode="3D", ompthreads=1,
            materials={"maxpoles": 0, "dispersivedtype": None, "dispersiveCdtype": None, "drudelorentz": None, "crealfunc": None},
        ): mc


def pytest_addoption(parser):
    """Add command-line options shared by hardware tests."""

    group = parser.getgroup("gprMax")
    group.addoption(
        "--gpu-device",
        action="store",
        type=int,
        default=int(os.environ.get("GPRMAX_TEST_GPU", "0")),
        metavar="INDEX",
        help="GPU device index used by tests marked 'gpu' (default: 0)",
    )


# Upstream test directories that run real gprMax models — too slow for CI.
_SLOW_DIRS = {
    "cmds_multiuse", "cmds_geometry", "cmds_singleuse",
    "fdfd_eigenmode_solver", "geometry_objects", "ntff", "ports",
    "toolboxes", "user_inputs",
}
# Upstream test files in shared folders that run real models (>1s each).
_SLOW_FILES = {
    "grid/test_grid.py", "grid/test_pml_pec_termination.py",
    "grid/test_te_mode_boundaries.py", "grid/test_te_reduced_kernels.py",
    "grid/test_restart_stepping_bounds_check.py",
    "subgrids/test_grid.py", "subgrids/test_subgrid_em_correctness.py",
    "subgrids/test_subgrid_inf_rejected.py",
    "subgrids/test_subgrid_magnetic_frill.py",
    "subgrids/test_subgrid_target_mainrx_steps.py",
    "subgrids/test_user_objects.py",
    "updates/test_h_boundary_symmetry.py",
    "updates/test_symmetry_boundary_solve_loop.py",
    "outputs/test_fields_outputs.py",
    "fractals/test_fractal_te_invariance.py",
    "fractals/test_grass_te_invariance.py",
    "fractals/test_surface_water_te.py",
    "materials/test_magnetic_averaging_mode.py",
    "materials/test_material_averaging.py",
    "materials/test_pec_h_components.py",
    "pml/test_internal_pml_slab.py",
    "utilities/test_logging.py",
    "test_precision_args.py",
    "materials/test_array_based_averaging_gate.py",
    "materials/test_complex_permittivity.py",
    "materials/test_debye_averaging.py",
    "materials/test_dispersive_infinite_conductivity.py",
    "materials/test_dispersive_mode2d.py",
    "materials/test_dispersive_averaging.py",
    "materials/test_dispersive_averaging_references.py",
    "outputs/test_geometry_view_fine_boundary_edges.py",
    "outputs/test_geometry_view_inf.py",
    "outputs/test_snapshot_inf.py",
    "outputs/test_snapshot_memory_freed_after_write.py",
}

def pytest_collection_modifyitems(items):
    """Mark upstream model-running tests as slow so they can be skipped."""
    for item in items:
        parts = item.nodeid.split("::")[0].replace("\\", "/").split("/")
        if len(parts) > 1 and parts[0] == "tests":
            if parts[1] in _SLOW_DIRS:
                item.add_marker(pytest.mark.slow)
            elif "/".join(parts[1:]) in _SLOW_FILES:
                item.add_marker(pytest.mark.slow)


@pytest.fixture
def gpu_device(request):
    """Return the selected CUDA device, or skip when it is unavailable."""

    device = request.config.getoption("--gpu-device")
    if device < 0:
        pytest.fail("--gpu-device must be a non-negative integer")

    try:
        import pycuda.driver as cuda

        cuda.init()
        device_count = cuda.Device.count()
    except Exception as exc:
        pytest.skip(f"CUDA hardware is unavailable: {exc}")
    if device >= device_count:
        pytest.skip(
            f"CUDA device {device} was requested but only {device_count} device(s) were found"
        )
    return device


# ---------------------------------------------------------------------------
# Shared factory fixtures for unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def make_waveform():
    """Factory for Waveform instances with sensible defaults.

    Usage:
        w = make_waveform("gaussian", freq=1e9, amp=2.0)
    """

    def _make(wave_type, freq=1e9, amp=1.0):
        w = Waveform()
        w.type = wave_type
        w.freq = freq
        w.amp = amp
        return w

    return _make


@pytest.fixture
def make_material():
    """Factory for non-dispersive Material instances.

    The constitutive parameters default to free space; override what each
    test cares about.
    """

    def _make(ID="test", numID=0, er=1.0, se=0.0, mr=1.0, sm=0.0):
        m = Material(numID, ID)
        m.er = er
        m.se = se
        m.mr = mr
        m.sm = sm
        return m

    return _make


@pytest.fixture
def make_dispersive():
    """Factory for DispersiveMaterial instances.

    ``poles`` is a sequence of ``(deltaer, tau, alpha)`` triples. Their
    meaning depends on ``model``:
        debye   — tau is the relaxation time; alpha is unused
        lorentz — tau is the pole frequency; alpha is the damping coeff
        drude   — tau is the pole frequency; alpha is the inverse relax time
    """

    def _make(ID="disp", numID=0, model="debye", er=1.0, se=0.0, poles=()):
        m = DispersiveMaterial(numID, ID)
        m.type = model
        m.er = er
        m.se = se
        m.poles = len(poles)
        for deltaer, tau, alpha in poles:
            m.deltaer.append(deltaer)
            m.tau.append(tau)
            m.alpha.append(alpha)
        return m

    return _make


@pytest.fixture
def fake_grid():
    """Factory for a minimal FDTDGrid stand-in.

    Materials tests use the minimal attribute set (``dt``, ``dx``, ``dy``,
    ``dz``, ``materials``). Sources / receivers tests pass extra kwargs for
    the richer attributes those methods read (``iterations``, ``timewindow``,
    ``waveforms``, per-type source lists, ``IDlookup``, ``ID``, ``rxs``).
    A ``SimpleNamespace`` is enough at the unit level.
    """

    def _make(
        dt=1e-12,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        materials=None,
        iterations=10,
        timewindow=None,
        waveforms=None,
        voltagesources=None,
        hertziandipoles=None,
        magneticdipoles=None,
        transmissionlines=None,
        IDlookup=None,
        ID=None,
        rxs=None,
        ntff_monitors=None,
        magneticfrillsources=None,
        pmls=None,
        mode="3D",
        **extra,
    ):
        if timewindow is None:
            timewindow = iterations * dt
        if IDlookup is None:
            IDlookup = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
        if pmls is None:
            pmls = {"formulation": "HORIPML"}
        ns = SimpleNamespace(
            dt=dt,
            dx=dx,
            dy=dy,
            dz=dz,
            materials=materials if materials is not None else [],
            iterations=iterations,
            timewindow=timewindow,
            waveforms=waveforms if waveforms is not None else [],
            voltagesources=voltagesources if voltagesources is not None else [],
            hertziandipoles=hertziandipoles if hertziandipoles is not None else [],
            magneticdipoles=magneticdipoles if magneticdipoles is not None else [],
            transmissionlines=transmissionlines if transmissionlines is not None else [],
            IDlookup=IDlookup,
            ID=ID,
            rxs=rxs if rxs is not None else [],
            ntff_monitors=ntff_monitors if ntff_monitors is not None else [],
            magneticfrillsources=magneticfrillsources if magneticfrillsources is not None else [],
            pmls=pmls,
            mode=mode,
        )
        for key, val in extra.items():
            setattr(ns, key, val)
        return ns

    return _make


@pytest.fixture
def fake_model_config():
    """Factory for a minimal ModelConfig stand-in.

    Returns a ``SimpleNamespace`` with the attributes that upstream code
    reads from ``config.get_model_config()`` at runtime. Tests that call
    real ``gprMax`` code through a ``fake_grid`` should also patch
    ``gprMax.config.get_model_config`` to return the result of this
    fixture.
    """

    def _make(debye_averaging=True, requested_2d_mode=None, **extra):
        ns = SimpleNamespace(
            debye_averaging=debye_averaging,
            requested_2d_mode=requested_2d_mode,
        )
        for key, val in extra.items():
            setattr(ns, key, val)
        return ns

    return _make


class _ConstantWaveform:
    """Test double for ``gprMax.waveforms.Waveform``."""

    def __init__(self, ID="wf", freq=1e9, value=1.0, wave_type="gaussian"):
        self.ID = ID
        self.freq = freq
        self.type = wave_type
        self._value = value

    def calculate_value(self, time, dt):
        return self._value if time >= 0 else 0.0


@pytest.fixture
def make_constant_waveform():
    """Returns the ``_ConstantWaveform`` test double."""

    def _make(ID="wf", freq=1e9, value=1.0, wave_type="gaussian"):
        return _ConstantWaveform(ID=ID, freq=freq, value=value, wave_type=wave_type)

    return _make


@pytest.fixture(autouse=True)
def _reset_gprmax_logger(caplog):
    """Ensure caplog works even after upstream tests pollute the gprMax logger."""
    import logging
    logger = logging.getLogger("gprMax")
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO, logger="gprMax")
