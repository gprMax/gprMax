"""Device-kernel and full-solve parity for virtual eigenmode waveguides."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax
from gprMax.cuda_opencl import knl_eigenmode, knl_virtual_waveguide
from tests.test_virtual_waveguide_integration import _uniform_waveguide_scene


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
def test_eigenmode_device_templates_have_complete_substitutions(backend):
    substitutions = {
        "CUDA_IDX": "int i = 0;" if backend == "cuda" else "",
        "METAL_DFT_PARAMETERS": (
            "int NF = parameters.NF; int NM = parameters.NM;" if backend == "metal" else ""
        ),
        "REAL": "double",
        "NX_FIELDS": 12,
        "NY_FIELDS": 13,
        "NZ_FIELDS": 14,
    }
    specifications = (
        knl_eigenmode.update_eigenmode_magnetic,
        knl_eigenmode.update_eigenmode_electric,
        knl_eigenmode.accumulate_eigenmode_dft,
        knl_virtual_waveguide.couple_magnetic,
        knl_virtual_waveguide.clear_rear_magnetic,
        knl_virtual_waveguide.couple_electric,
        knl_virtual_waveguide.clear_rear_electric,
    )
    for specification in specifications:
        arguments = specification[f"args_{backend}"].substitute({"REAL": "double"})
        body = specification["func"].substitute(substitutions)
        assert "$" not in arguments
        assert "$" not in body


def _port_results(path, port_number=1):
    with h5py.File(Path(path).with_suffix(".h5")) as output:
        port = output[f"eigenmode_ports/port{port_number}"]
        return {name: port[name][...] for name in ("incident", "outgoing", "S")}


def _broadband_virtual_waveguide_scene():
    """Three-anchor source exercising all device modal-profile bases."""

    scene = _uniform_waveguide_scene(normal_axis=0, direction="+")
    objects = scene.grid_objects

    time_window = next(
        obj for obj in scene.single_use_objects if isinstance(obj, gprMax.TimeWindow)
    )
    time_window.time = 1.2e-9

    band = next(obj for obj in objects if isinstance(obj, gprMax.EigenmodeBand))
    band.kwargs.update(fmin=20e9, fmax=24e9, points=17)

    port = next(obj for obj in objects if isinstance(obj, gprMax.EigenmodePort))
    port.kwargs["anchors"] = (16.9e9, 22e9, 27e9)
    excitation = next(obj for obj in objects if isinstance(obj, gprMax.EigenmodeExcitation))
    excitation.kwargs["waveform"] = "auto"
    return scene


def _active_and_passive_virtual_waveguide_scene():
    """Finite guide segment joined to active and passive auxiliary guides."""

    scene = _uniform_waveguide_scene(normal_axis=0, direction="+")
    scene.add(
        gprMax.EigenmodePort(
            port=2,
            p1=(0.04, 0.001, 0.001),
            p2=(0.04, 0.009, 0.011),
            direction="-",
            modes=(1,),
            anchors=(22e9,),
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.VirtualWaveguide(
            port=2,
            length_cells=18,
            pml_cells=8,
            source_clearance_cells=4,
        )
    )
    return scene


def _direct_eigenmode_source_scene():
    """Conventional in-domain TF/SF source without an auxiliary guide."""

    scene = _uniform_waveguide_scene(normal_axis=0, direction="+")
    scene.grid_objects = [
        obj for obj in scene.grid_objects if not isinstance(obj, gprMax.VirtualWaveguide)
    ]
    return scene


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("normal_axis", range(3))
@pytest.mark.parametrize("direction", ["-", "+"])
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
def test_device_virtual_waveguide_matches_cpu(tmp_path, request, backend, normal_axis, direction):
    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}

    suffix = f"{'xyz'[normal_axis]}{'minus' if direction == '-' else 'plus'}"
    cpu_path = tmp_path / f"cpu_virtual_{suffix}"
    device_path = tmp_path / f"{backend}_virtual_{suffix}"
    gprMax.run(
        scenes=[_uniform_waveguide_scene(normal_axis, direction)],
        outputfile=cpu_path,
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
    )
    gprMax.run(
        scenes=[_uniform_waveguide_scene(normal_axis, direction)],
        outputfile=device_path,
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
        **device_options,
    )

    cpu = _port_results(cpu_path)
    device = _port_results(device_path)
    incident_scale = max(float(np.max(np.abs(cpu["incident"]))), 1e-20)
    for name in cpu:
        assert np.isfinite(device[name]).all()
        absolute_tolerance = 5e-5 if name == "S" else 5e-5 * incident_scale
        assert_allclose(
            device[name],
            cpu[name],
            rtol=5e-4,
            atol=absolute_tolerance,
        )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
def test_device_virtual_waveguide_broadband_matches_cpu(tmp_path, request, backend):
    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}

    cpu_path = tmp_path / "cpu_virtual_broadband"
    device_path = tmp_path / f"{backend}_virtual_broadband"
    gprMax.run(
        scenes=[_broadband_virtual_waveguide_scene()],
        outputfile=cpu_path,
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
    )
    gprMax.run(
        scenes=[_broadband_virtual_waveguide_scene()],
        outputfile=device_path,
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
        **device_options,
    )

    cpu = _port_results(cpu_path)
    device = _port_results(device_path)
    incident_scale = max(float(np.max(np.abs(cpu["incident"]))), 1e-20)
    for name in cpu:
        absolute_tolerance = 5e-5 if name == "S" else 5e-5 * incident_scale
        assert_allclose(device[name], cpu[name], rtol=5e-4, atol=absolute_tolerance)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
def test_device_passive_virtual_waveguide_matches_cpu(tmp_path, request, backend):
    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}

    cpu_path = tmp_path / "cpu_virtual_passive"
    device_path = tmp_path / f"{backend}_virtual_passive"
    run_options = {
        "hide_progress_bars": True,
        "log_level": 30,
    }
    gprMax.run(
        scenes=[_active_and_passive_virtual_waveguide_scene()],
        outputfile=cpu_path,
        cpu_precision="double",
        **run_options,
    )
    gprMax.run(
        scenes=[_active_and_passive_virtual_waveguide_scene()],
        outputfile=device_path,
        gpu_precision="double",
        **device_options,
        **run_options,
    )

    for port_number in (1, 2):
        cpu = _port_results(cpu_path, port_number)
        device = _port_results(device_path, port_number)
        incident_scale = max(float(np.max(np.abs(cpu["incident"]))), 1e-20)
        for name in cpu:
            absolute_tolerance = 5e-5 if name == "S" else 5e-5 * incident_scale
            assert_allclose(device[name], cpu[name], rtol=5e-4, atol=absolute_tolerance)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
def test_device_direct_eigenmode_source_matches_cpu(tmp_path, request, backend):
    if backend == "cuda":
        device_options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        device_options = {"opencl": [request.getfixturevalue("opencl_device")]}

    cpu_path = tmp_path / "cpu_direct_eigenmode"
    device_path = tmp_path / f"{backend}_direct_eigenmode"
    run_options = {"hide_progress_bars": True, "log_level": 30}
    gprMax.run(
        scenes=[_direct_eigenmode_source_scene()],
        outputfile=cpu_path,
        cpu_precision="double",
        **run_options,
    )
    gprMax.run(
        scenes=[_direct_eigenmode_source_scene()],
        outputfile=device_path,
        gpu_precision="double",
        **device_options,
        **run_options,
    )

    cpu = _port_results(cpu_path)
    device = _port_results(device_path)
    incident_scale = max(float(np.max(np.abs(cpu["incident"]))), 1e-20)
    for name in cpu:
        absolute_tolerance = 5e-5 if name == "S" else 5e-5 * incident_scale
        assert_allclose(device[name], cpu[name], rtol=5e-4, atol=absolute_tolerance)
