"""Shared pytest configuration for the gprMax test suite."""

import os
import sys
from types import SimpleNamespace

# Some Linux MPI/OFI installations otherwise try to select an unavailable
# network provider while pytest imports gprMax. An explicit user setting is
# always preserved.
if sys.platform.startswith("linux"):
    os.environ.setdefault("FI_PROVIDER", "shm")

import pytest

from gprMax.materials import DispersiveMaterial, Material
from gprMax.waveforms import Waveform


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
    group.addoption(
        "--opencl-device",
        action="store",
        type=int,
        default=int(os.environ.get("GPRMAX_TEST_OPENCL", "0")),
        metavar="INDEX",
        help="OpenCL device index used by tests marked 'gpu' (default: 0)",
    )


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


@pytest.fixture
def opencl_device(request):
    """Return the selected OpenCL device, or skip when it is unavailable."""

    device = request.config.getoption("--opencl-device")
    if device < 0:
        pytest.fail("--opencl-device must be a non-negative integer")

    try:
        import pyopencl as cl

        devices = [item for platform in cl.get_platforms() for item in platform.get_devices()]
    except Exception as exc:
        pytest.skip(f"OpenCL hardware is unavailable: {exc}")
    if device >= len(devices):
        pytest.skip(
            f"OpenCL device {device} was requested but only " f"{len(devices)} device(s) were found"
        )
    return device


# ---------------------------------------------------------------------------
# Shared factory fixtures for unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def make_waveform():
    """Factory for Waveform instances with sensible defaults."""

    def _make(wave_type, freq=1e9, amp=1.0):
        waveform = Waveform()
        waveform.type = wave_type
        waveform.freq = freq
        waveform.amp = amp
        return waveform

    return _make


@pytest.fixture
def make_material():
    """Factory for non-dispersive Material instances."""

    def _make(ID="test", numID=0, er=1.0, se=0.0, mr=1.0, sm=0.0):
        material = Material(numID, ID)
        material.er = er
        material.se = se
        material.mr = mr
        material.sm = sm
        return material

    return _make


@pytest.fixture
def make_dispersive():
    """Factory for dispersive Material instances.

    ``poles`` is a sequence of ``(deltaer, tau, alpha)`` triples. Their
    meaning depends on ``model``:

    * Debye: ``tau`` is the relaxation time and ``alpha`` is unused.
    * Lorentz: ``tau`` is the pole frequency and ``alpha`` is damping.
    * Drude: ``tau`` is the pole frequency and ``alpha`` is inverse relaxation time.
    """

    def _make(ID="disp", numID=0, model="debye", er=1.0, se=0.0, poles=()):
        material = DispersiveMaterial(numID, ID)
        material.type = model
        material.er = er
        material.se = se
        material.poles = len(poles)
        for deltaer, tau, alpha in poles:
            material.deltaer.append(deltaer)
            material.tau.append(tau)
            material.alpha.append(alpha)
        return material

    return _make


@pytest.fixture
def fake_grid():
    """Factory for a minimal FDTDGrid stand-in used by isolated unit tests."""

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
        grid = SimpleNamespace(
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
            magneticfrillsources=(magneticfrillsources if magneticfrillsources is not None else []),
            pmls=pmls,
            mode=mode,
        )
        for key, value in extra.items():
            setattr(grid, key, value)
        return grid

    return _make


@pytest.fixture
def fake_model_config():
    """Factory for a minimal ModelConfig stand-in."""

    def _make(debye_averaging=True, requested_2d_mode=None, **extra):
        model_config = SimpleNamespace(
            debye_averaging=debye_averaging,
            requested_2d_mode=requested_2d_mode,
        )
        for key, value in extra.items():
            setattr(model_config, key, value)
        return model_config

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
    """Return a factory for constant waveform test doubles."""

    def _make(ID="wf", freq=1e9, value=1.0, wave_type="gaussian"):
        return _ConstantWaveform(ID=ID, freq=freq, value=value, wave_type=wave_type)

    return _make
