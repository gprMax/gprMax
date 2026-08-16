"""Backend-neutral tests for device-resident NTFF DFT collection."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gprMax.cuda_opencl.knl_ntff import (
    build_equivalent_current_time_kernel_source,
    build_ntff_kernel_source,
    build_time_domain_ntff_kernel_source,
)
from gprMax.ntff.device import (
    CUDAKSIRCollector,
    MetalKSIRCollector,
    MetalTimeDomainKSIRCollector,
    OpenCLKSIRCollector,
    OpenCLTimeDomainKSIRCollector,
    _DeviceKSIRCollector,
)
from gprMax.ntff.frequency_domain import KSIRFrequencyDomainMonitor
from gprMax.ntff.surfaces import build_component_surface


class _HostEmulatedDeviceCollector(_DeviceKSIRCollector):
    """Execute the device kernel contract with NumPy for deterministic tests."""

    def _allocate(self, record):
        for name in ("inside_real", "inside_imag", "outside_real", "outside_imag"):
            record.device[name] = np.zeros(record.total, dtype=record.monitor.real_dtype)

    def _accumulate(self, record, field, multiplier):
        values = np.asarray(field).ravel()
        inside = values[record.inside_index]
        outside = values[record.outside_index]
        shape = record.shape
        record.device["inside_real"] += (
            (multiplier.real[:, np.newaxis] * inside[np.newaxis, :]).reshape(shape).ravel()
        )
        record.device["inside_imag"] += (
            (multiplier.imag[:, np.newaxis] * inside[np.newaxis, :]).reshape(shape).ravel()
        )
        record.device["outside_real"] += (
            (multiplier.real[:, np.newaxis] * outside[np.newaxis, :]).reshape(shape).ravel()
        )
        record.device["outside_imag"] += (
            (multiplier.imag[:, np.newaxis] * outside[np.newaxis, :]).reshape(shape).ravel()
        )

    def _download(self, record):
        return tuple(
            record.device[name]
            for name in ("inside_real", "inside_imag", "outside_real", "outside_imag")
        )


def test_frequency_collector_rejects_int32_work_item_overflow():
    int32_max = int(np.iinfo(np.int32).max)
    surface = SimpleNamespace(
        npatches=int32_max // 2 + 1,
        faces=[
            SimpleNamespace(
                inside_flat_indices=np.asarray([0]),
                outside_flat_indices=np.asarray([1]),
            )
        ],
    )
    monitor = SimpleNamespace(
        device_sampling_multiplier=None,
        frequencies=SimpleNamespace(size=2),
        surfaces={"Ex": surface},
    )
    updates = SimpleNamespace(grid=SimpleNamespace())

    with pytest.raises(ValueError, match="exceeds device int32 indexing"):
        _HostEmulatedDeviceCollector(updates, monitors=[monitor], configure=False)


def _material():
    return SimpleNamespace(
        numID=1,
        ID="free_space",
        er=1.0,
        mr=1.0,
        se=0.0,
        sm=0.0,
        poles=0,
    )


def _monitor(name, surfaces, real_dtype, complex_dtype, iterations, dt):
    return KSIRFrequencyDomainMonitor(
        name,
        surfaces,
        [2 / (iterations * dt), 5 / (iterations * dt)],
        [40.0, 90.0],
        [20.0, 120.0],
        dt,
        iterations,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )


@pytest.mark.parametrize(
    "real_dtype,complex_dtype,rtol",
    [(np.dtype("f4"), np.dtype("c8"), 2e-6), (np.dtype("f8"), np.dtype("c16"), 2e-13)],
)
def test_device_contract_matches_cpu_flat_index_collection(real_dtype, complex_dtype, rtol):
    shape = (9, 10, 8)
    surfaces = {
        component: build_component_surface(
            component, (2, 2, 2), (5, 6, 5), (0.03, 0.04, 0.05), shape
        )
        for component in ("Ex", "Hx")
    }
    iterations = 12
    dt = 1e-11
    cpu = _monitor("cpu", surfaces, real_dtype, complex_dtype, iterations, dt)
    device = _monitor("device", surfaces, real_dtype, complex_dtype, iterations, dt)

    ids = np.ones((6,) + shape, dtype=np.uint32)
    lookup = {"Ex": 0, "Hx": 3}
    cpu.validate_materials(ids, lookup)
    cpu.configure_background([_material()])

    grid = SimpleNamespace(
        ntff_monitors=[device],
        ID=ids,
        IDlookup=lookup,
        materials=[_material()],
    )
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        setattr(grid, f"{component}_dev", np.zeros(shape, dtype=real_dtype))
    collector = _HostEmulatedDeviceCollector(SimpleNamespace(grid=grid))

    indices = np.indices(shape)
    zeros = np.zeros(shape, dtype=real_dtype)
    for iteration in range(iterations):
        ex = np.asarray(
            (iteration + 1) * (indices[0] - 0.3 * indices[1]),
            dtype=real_dtype,
        )
        hx = np.asarray(
            (iteration + 0.5) * (indices[2] + 0.2 * indices[1]),
            dtype=real_dtype,
        )
        grid.Ex_dev[...] = ex
        grid.Hx_dev[...] = hx
        cpu.observe_electric(iteration, ex, zeros, zeros)
        collector.observe_electric(iteration)
        cpu.observe_magnetic(iteration, hx, zeros, zeros)
        collector.observe_magnetic(iteration)

    cpu.finalise()
    collector.finalise()
    for component in ("Ex", "Hx"):
        assert device.surface_data[component].field.dtype == complex_dtype
        assert device.result.range_normalized_fields[component].dtype == complex_dtype
        assert_allclose(
            device.surface_data[component].field,
            cpu.surface_data[component].field,
            rtol=rtol,
        )
        assert_allclose(
            device.surface_data[component].normal_derivative,
            cpu.surface_data[component].normal_derivative,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    "real_dtype,complex_dtype,rtol",
    [(np.dtype("f4"), np.dtype("c8"), 2e-6), (np.dtype("f8"), np.dtype("c16"), 2e-13)],
)
def test_device_incident_plane_wave_dft_matches_cpu(real_dtype, complex_dtype, rtol):
    shape = (8, 8, 8)
    surface = build_component_surface("Ex", (2, 2, 2), (5, 5, 5), (0.01,) * 3, shape)
    iterations = 16
    dt = 1e-11
    cpu = _monitor("cpu", {"Ex": surface}, real_dtype, complex_dtype, iterations, dt)
    device = _monitor("device", {"Ex": surface}, real_dtype, complex_dtype, iterations, dt)

    def plane_wave():
        fields = np.zeros((3, 1), dtype=real_dtype)
        return SimpleNamespace(
            m=np.zeros(3, dtype=np.int32),
            origin=np.zeros(3, dtype=np.int32),
            axial=0,
            E_fields=fields,
            E_fields_dev=fields.copy(),
            corners=np.asarray((1, 1, 1, 6, 6, 6), dtype=np.int32),
            waveformID="pulse",
            materialID="free_space",
            actual_angles=np.asarray((90.0, 0.0), dtype=real_dtype),
            psi=0.0,
            start=0.0,
            stop=1.0,
        )

    cpu_plane_wave = plane_wave()
    device_plane_wave = plane_wave()
    cpu.associate_plane_wave(cpu_plane_wave, (0.01,) * 3, 0)
    device.associate_plane_wave(device_plane_wave, (0.01,) * 3, 0)

    ids = np.ones((6,) + shape, dtype=np.uint32)
    lookup = {"Ex": 0}
    cpu.validate_materials(ids, lookup)
    cpu.configure_background([_material()])
    grid = SimpleNamespace(
        ntff_monitors=[device],
        ID=ids,
        IDlookup=lookup,
        materials=[_material()],
        Ex_dev=np.zeros(shape, dtype=real_dtype),
    )
    collector = _HostEmulatedDeviceCollector(SimpleNamespace(grid=grid))
    zeros = np.zeros(shape, dtype=real_dtype)

    for iteration in range(iterations):
        time = iteration * dt
        sample = np.asarray(
            (
                np.sin(2 * np.pi * cpu.frequencies[0] * time),
                0.25 * np.cos(2 * np.pi * cpu.frequencies[1] * time),
                0.1 * (iteration + 1),
            ),
            dtype=real_dtype,
        )
        cpu_plane_wave.E_fields[:, 0] = sample
        device_plane_wave.E_fields_dev[:, 0] = sample
        cpu.observe_electric(iteration, zeros, zeros, zeros)
        collector.observe_electric(iteration)

    cpu.finalise()
    collector.finalise()

    assert device._incident_next_iteration == iterations
    assert_allclose(
        device.result.incident_electric,
        cpu.result.incident_electric,
        rtol=rtol,
        atol=rtol * np.max(np.abs(cpu.result.incident_electric)),
    )


@pytest.mark.parametrize(
    "backend,c_real,marker",
    [
        ("cuda", "float", "blockIdx.x"),
        ("opencl", "double", "cl_khr_fp64"),
        ("metal", "float", "thread_position_in_grid"),
    ],
)
def test_backend_kernel_sources_use_configured_real_type(backend, c_real, marker):
    source = build_ntff_kernel_source(backend, c_real)
    assert marker in source
    assert f"const {c_real}*" in source
    assert "inside_real[i] +=" in source
    assert "complex" not in source.lower()


@pytest.mark.parametrize(
    "backend,c_real,marker",
    [
        ("cuda", "float", "blockIdx.x"),
        ("opencl", "double", "cl_khr_fp64"),
        ("metal", "float", "thread_position_in_grid"),
    ],
)
def test_equivalent_current_time_kernel_sources_are_backend_complete(
    backend, c_real, marker
):
    source = build_equivalent_current_time_kernel_source(c_real, backend)

    assert marker in source
    assert "gather_equivalent_current_time" in source
    assert "deposit_equivalent_current_time" in source
    assert "current[patch * 3]" in source
    assert "inverse_dt" in source


@pytest.mark.parametrize(
    "backend,c_real,marker",
    [
        ("cuda", "float", "ksir_atomic_add"),
        ("opencl", "double", "get_global_id(0)"),
        ("metal", "float", "thread_position_in_grid"),
    ],
)
def test_time_domain_kernel_uses_configured_real_type(backend, c_real, marker):
    source = build_time_domain_ntff_kernel_source(c_real, backend=backend)

    assert "gather_time_domain_ntff" in source
    assert "deposit_time_domain_ntff" in source
    assert c_real in source
    assert marker in source
    assert "time_origin_steps[point]" in source
    if backend != "cuda":
        assert "ksir_atomic_add" not in source
        assert "for (int patch = 0; patch < neffective_patches; patch++)" in source


class _FakeArray:
    def __init__(self, name):
        self.gpudata = f"ptr:{name}"
        self.data = f"buffer:{name}"
        self.base_data = self.data
        self.offset = 0
        self.dtype = np.dtype("f8")
        self.set_value = None
        self.set_input_contiguous = None

    def set(self, value, **kwargs):
        self.set_input_contiguous = np.asarray(value).flags.c_contiguous
        self.set_value = np.asarray(value).copy()


def _dispatch_record():
    arrays = {
        name: _FakeArray(name)
        for name in (
            "inside_index",
            "outside_index",
            "multiplier_real",
            "multiplier_imag",
            "inside_real",
            "inside_imag",
            "outside_real",
            "outside_imag",
        )
    }
    return SimpleNamespace(total=6, npatches=3, device=arrays)


def test_cuda_dispatch_uses_split_configured_real_buffers():
    collector = CUDAKSIRCollector.__new__(CUDAKSIRCollector)
    collector.real_dtype = np.dtype("f4")
    calls = []
    collector.kernel = lambda *args, **kwargs: calls.append((args, kwargs))
    record = _dispatch_record()
    field = _FakeArray("field")
    multiplier = np.asarray([1 + 2j, 3 + 4j], dtype="c8")

    collector._accumulate(record, field, multiplier)

    args, kwargs = calls[0]
    assert args[6] == "ptr:field"
    assert args[7:11] == (
        "ptr:inside_real",
        "ptr:inside_imag",
        "ptr:outside_real",
        "ptr:outside_imag",
    )
    assert kwargs["block"] == (128, 1, 1)
    assert record.device["multiplier_real"].set_value.dtype == np.dtype("f4")
    assert_allclose(record.device["multiplier_imag"].set_value, [2, 4])


def test_opencl_dispatch_uses_split_configured_real_buffers():
    collector = OpenCLKSIRCollector.__new__(OpenCLKSIRCollector)
    collector.real_dtype = np.dtype("f8")
    collector.queue = object()
    calls = []
    collector.kernel = lambda *args: calls.append(args)
    record = _dispatch_record()
    field = _FakeArray("field")
    field.offset = 16
    multiplier = np.asarray([1 + 2j, 3 + 4j], dtype="c16")

    collector._accumulate(record, field, multiplier)

    args = calls[0]
    assert args[0] is collector.queue
    assert args[5] == "buffer:inside_index"
    assert args[9] == 2
    assert args[10] == "buffer:field"
    assert args[11:15] == (
        "buffer:inside_real",
        "buffer:inside_imag",
        "buffer:outside_real",
        "buffer:outside_imag",
    )
    assert record.device["multiplier_real"].set_value.dtype == np.dtype("f8")
    assert record.device["multiplier_real"].set_input_contiguous
    assert record.device["multiplier_imag"].set_input_contiguous


class _FakeMetalEncoder:
    def __init__(self):
        self.buffers = {}
        self.scalar_bytes = {}
        self.dispatched = None

    def setComputePipelineState_(self, pipeline):
        self.pipeline = pipeline

    def setBuffer_offset_atIndex_(self, buffer, offset, index):
        self.buffers[index] = buffer

    def setBytes_length_atIndex_(self, value, length, index):
        self.scalar_bytes[index] = (bytes(value), length)

    def dispatchThreads_threadsPerThreadgroup_(self, threads, group):
        self.dispatched = (threads, group)

    def endEncoding(self):
        pass


class _FakeMetalCommand:
    def __init__(self):
        self.encoder = _FakeMetalEncoder()

    def computeCommandEncoder(self):
        return self.encoder

    def commit(self):
        pass

    def waitUntilCompleted(self):
        pass


def test_metal_dispatch_preserves_kernel_buffer_order():
    collector = MetalKSIRCollector.__new__(MetalKSIRCollector)
    collector.real_dtype = np.dtype("f4")
    collector.pipeline = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: 64)
    command = _FakeMetalCommand()
    collector.queue = SimpleNamespace(commandBuffer=lambda: command)
    collector.metal = SimpleNamespace(MTLSizeMake=lambda x, y, z: (x, y, z))
    collector._buffer = lambda values: ("temporary", np.asarray(values).copy())
    record = _dispatch_record()
    field = object()

    collector._accumulate(record, field, np.asarray([1 + 2j, 3 + 4j], dtype="c8"))

    buffers = command.encoder.buffers
    assert set(buffers) == set(range(11))
    assert buffers[2] is record.device["inside_index"]
    assert buffers[3] is record.device["outside_index"]
    assert buffers[6] is field
    assert buffers[7] is record.device["inside_real"]
    assert buffers[10] is record.device["outside_imag"]
    assert command.encoder.dispatched == ((6, 1, 1), (6, 1, 1))


def _time_dispatch_record():
    arrays = {
        name: _FakeArray(name)
        for name in (
            "normal_derivative_weight",
            "field_weight",
            "time_derivative_weight",
            "source_patch_index",
            "integer_delay",
            "fractional_delay",
            "time_origin_steps",
            "output",
        )
    }
    arrays["surface"] = [_FakeArray(f"surface{item}") for item in range(3)]
    arrays["normal_derivative"] = [_FakeArray(f"normal{item}") for item in range(3)]
    return SimpleNamespace(
        npoints=2,
        neffective_patches=3,
        output_length=11,
        device=arrays,
    )


def test_opencl_time_deposit_dispatch_is_point_owned():
    collector = OpenCLTimeDomainKSIRCollector.__new__(OpenCLTimeDomainKSIRCollector)
    collector.queue = object()
    collector.real_scalar = np.float32
    calls = []
    collector.deposit_kernel = lambda *args: calls.append(args)
    record = _time_dispatch_record()

    collector._deposit(record, 4, (2, 3, 4), (-1.0, 1.0, 0.0))

    args = calls[0]
    assert args[1] == (record.npoints,)
    assert args[3] == record.npoints
    assert args[4] == record.neffective_patches
    assert args[7] == "buffer:surface1"
    assert args[-1] == "buffer:output"


def test_metal_time_deposit_dispatch_preserves_argument_contract():
    collector = MetalTimeDomainKSIRCollector.__new__(MetalTimeDomainKSIRCollector)
    collector.real_scalar = np.float32
    collector.deposit_pipeline = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: 64)
    command = _FakeMetalCommand()
    collector.queue = SimpleNamespace(commandBuffer=lambda: command)
    collector.metal = SimpleNamespace(MTLSizeMake=lambda x, y, z: (x, y, z))
    record = _time_dispatch_record()

    collector._deposit(record, 4, (2, 3, 4), (-1.0, 1.0, 0.0))

    encoder = command.encoder
    assert set(encoder.scalar_bytes) == set(range(4)) | set(range(9, 12))
    assert encoder.buffers[4] is record.device["surface"][1]
    assert encoder.buffers[12] is record.device["normal_derivative_weight"]
    assert encoder.buffers[19] is record.device["output"]
    assert encoder.dispatched == ((record.npoints, 1, 1), (2, 1, 1))
