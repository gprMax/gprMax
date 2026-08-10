"""Shared pytest configuration for the gprMax test suite."""

import os

import pytest

# Some Linux MPI/OFI installations otherwise try to select an unavailable
# network provider while pytest imports gprMax. An explicit user setting is
# always preserved.
os.environ.setdefault("FI_PROVIDER", "shm")


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
