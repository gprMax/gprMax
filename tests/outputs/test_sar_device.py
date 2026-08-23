"""Backend-neutral tests for device-resident SAR DFT collection."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.cuda_opencl.knl_sar import build_sar_kernel_source
from gprMax.sar_device import _DeviceSARCollector


class _Monitor:
    def __init__(self, frequencies, indices, dt):
        self.frequencies = np.asarray(frequencies)
        self.edge_flat_indices = indices
        self.real_dtype = np.dtype("f8")
        self.complex_dtype = np.dtype("c16")
        self.collection_backend = "cpu"
        self._dt = dt
        self._iteration = 0
        self.loaded = {}

    def device_sampling_multiplier(self, iteration):
        assert iteration == self._iteration
        self._iteration += 1
        return self._dt * np.exp(-2j * np.pi * self.frequencies * iteration * self._dt)

    def load_device_component_dfts(self, component, values):
        self.loaded[component] = np.asarray(values).copy()


class _HostDeviceSARCollector(_DeviceSARCollector):
    backend = "test"

    def _allocate(self, record):
        record.device["multiplier"] = None
        for component in record.components:
            shape = (record.nfrequencies, component.nedges)
            component.device["values"] = np.zeros(shape, dtype=np.complex128)

    def _upload_multiplier(self, record, multiplier):
        record.device["multiplier"] = multiplier

    def _accumulate(self, record, component, field):
        samples = np.asarray(field).ravel()[component.indices]
        component.device["values"] += (
            record.device["multiplier"][:, np.newaxis] * samples[np.newaxis, :]
        )

    def _download(self, record, component):
        values = component.device["values"]
        return values.real.ravel(), values.imag.ravel()


def test_device_sar_contract_matches_direct_sparse_dft():
    shape = (5, 4, 6)
    indices = {
        "Ex": np.asarray((0, 7, 19, 47), dtype=np.int64),
        "Ey": np.asarray((2, 13, 29), dtype=np.int64),
        "Ez": np.asarray((1, 31, 57, 83, 101), dtype=np.int64),
    }
    frequencies = np.asarray((1.0e8, 3.0e8))
    dt = 2.0e-11
    monitor = _Monitor(frequencies, indices, dt)
    grid = SimpleNamespace(sar_monitors=[monitor])
    for component in indices:
        setattr(grid, f"{component}_dev", np.zeros(shape))
    collector = _HostDeviceSARCollector(SimpleNamespace(grid=grid))
    expected = {
        component: np.zeros((frequencies.size, values.size), dtype=np.complex128)
        for component, values in indices.items()
    }

    coordinates = np.indices(shape)
    for iteration in range(9):
        fields = {
            "Ex": (iteration + 1) * (coordinates[0] + 0.2 * coordinates[2]),
            "Ey": (iteration + 0.5) * (coordinates[1] - 0.1 * coordinates[0]),
            "Ez": (iteration + 2) * (coordinates[2] + 0.3 * coordinates[1]),
        }
        multiplier = dt * np.exp(-2j * np.pi * frequencies * iteration * dt)
        for component, field in fields.items():
            setattr(grid, f"{component}_dev", field)
            expected[component] += (
                multiplier[:, np.newaxis] * field.ravel()[indices[component]][np.newaxis, :]
            )
        collector.observe_electric(iteration)

    collector.finalise()
    assert monitor.collection_backend == "test_device"
    for component in indices:
        np.testing.assert_allclose(monitor.loaded[component], expected[component])


def test_device_sar_allocates_only_monitor_active_components():
    monitor = _Monitor(
        frequencies=(1e9,),
        indices={"Ez": np.asarray((1, 7, 13), dtype=np.int64)},
        dt=1e-12,
    )
    grid = SimpleNamespace(sar_monitors=[monitor], Ez_dev=np.zeros((3, 3, 3)))

    collector = _HostDeviceSARCollector(SimpleNamespace(grid=grid))

    assert tuple(component.component for component in collector.records[0].components) == ("Ez",)


@pytest.mark.parametrize(
    "backend,c_real,marker",
    (
        ("cuda", "float", "blockIdx.x"),
        ("opencl", "double", "cl_khr_fp64"),
        ("metal", "float", "thread_position_in_grid"),
    ),
)
def test_sar_kernel_sources_cover_all_accelerator_backends(backend, c_real, marker):
    source = build_sar_kernel_source(backend, c_real)

    assert marker in source
    assert "accumulate_sar" in source
    assert "edge_index[edge]" in source
    assert "output_real[i] +=" in source
    assert c_real in source


def _sar_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=1.5e-9))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(2))
    scene.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    scene.add(
        gprMax.Box(
            p1=(0.004, 0.004, 0.004),
            p2=(0.020, 0.020, 0.020),
            material_id="tissue",
            tag="target",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.VoltageSource(
            p1=(0.012, 0.012, 0.012),
            polarisation="z",
            resistance=50,
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.SAR(
            frequencies=(0.75e9, 1e9, 1.25e9),
            waveform_id="pulse",
            tags="target",
            id="target_sar",
            spectrum_limit="nyquist",
            averaging_masses=(0.001,),
        )
    )
    return scene


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_device_sar_matches_cpu(tmp_path, request, backend, precision):
    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}
    cpu_path = tmp_path / f"cpu_sar_{precision}"
    device_path = tmp_path / f"{backend}_sar_{precision}"
    gprMax.run(
        scenes=[_sar_scene()],
        n=1,
        outputfile=cpu_path,
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[_sar_scene()],
        n=1,
        outputfile=device_path,
        hide_progress_bars=True,
        gpu_precision=precision,
        **device_options,
    )

    tolerance = 5e-4 if precision == "single" else 5e-7
    paths = (
        "sar",
        "absorbed_power_density",
        "source_spectrum",
        "normalisation_scale",
        "spatial_average/1g/sar",
        "spatial_average/1g/peak_sar",
    )
    with h5py.File(str(cpu_path) + ".h5", "r") as cpu, h5py.File(
        str(device_path) + ".h5", "r"
    ) as device:
        cpu_group = cpu["sar/target_sar"]
        device_group = device["sar/target_sar"]
        assert device_group.attrs["CollectionBackend"] == f"{backend}_device"
        for path in paths:
            reference = cpu_group[path][...]
            scale = max(float(np.nanmax(np.abs(reference), initial=0.0)), 1e-18)
            np.testing.assert_allclose(
                device_group[path][...],
                reference,
                rtol=tolerance,
                atol=tolerance * scale,
                equal_nan=True,
            )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
def test_device_sar_state_is_reset_for_geometry_fixed_runs(tmp_path, request, backend):
    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}
    output = tmp_path / f"{backend}_sar_reuse"
    gprMax.run(
        scenes=[_sar_scene()],
        n=2,
        geometry_fixed=True,
        outputfile=output,
        hide_progress_bars=True,
        **device_options,
    )

    results = []
    for index in (1, 2):
        with h5py.File(f"{output}{index}.h5", "r") as data:
            results.append(data["sar/target_sar/sar"][...])
    np.testing.assert_array_equal(results[0], results[1])
