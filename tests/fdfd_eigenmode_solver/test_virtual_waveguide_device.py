# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Device-kernel and full-solve parity for virtual eigenmode waveguides."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax
import gprMax.config as config
from gprMax.cuda_opencl import knl_eigenmode, knl_virtual_waveguide
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.updates.metal_updates import MetalUpdates
from gprMax.updates.opencl_updates import OpenCLUpdates
from gprMax.virtual_waveguide import VirtualWaveguide as RuntimeVirtualWaveguide
from tests.test_virtual_waveguide_integration import _uniform_waveguide_scene

try:
    import Metal

    HAS_METAL = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:
    HAS_METAL = False


def _device_options(request, backend):
    if backend == "cuda":
        return {"gpu": [request.getfixturevalue("gpu_device")]}, "double"
    if backend == "opencl":
        return {"opencl": [request.getfixturevalue("opencl_device")]}, "double"
    if not HAS_METAL:
        pytest.skip("No Apple Metal device/PyObjC available")
    return {"metal": True}, "single"


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


@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("direction_sign", [-1, 1])
@pytest.mark.parametrize("normal_axis", range(3))
def test_rear_clear_dispatch_covers_only_the_aperture_prism(
    normal_axis, direction_sign, magnetic
):
    size = np.array([101, 82, 73])
    guide = RuntimeVirtualWaveguide.__new__(RuntimeVirtualWaveguide)
    guide.main_grid = SimpleNamespace(size=size)
    guide.normal_axis = normal_axis
    guide.direction_sign = direction_sign
    guide.plane_index = 29 if direction_sign > 0 else int(size[normal_axis]) - 23
    guide.nu = 7
    guide.nv = 5

    if direction_sign < 0:
        normal_points = (
            int(size[normal_axis]) - guide.plane_index + int(magnetic)
        )
    else:
        normal_points = guide.plane_index
    expected = normal_points * (guide.nu + 1) * (guide.nv + 1)

    assert guide._rear_clear_points(magnetic=magnetic) == expected
    assert expected < int(np.prod(size + 1))


def _validation_guide(size, nu, nv):
    guide = RuntimeVirtualWaveguide.__new__(RuntimeVirtualWaveguide)
    guide.main_grid = SimpleNamespace(
        size=np.asarray(size),
        materials=[SimpleNamespace(numID=0, ID="free_space", poles=0)],
    )
    guide.port = SimpleNamespace(invariant_axis=None)
    guide.spec = SimpleNamespace(
        length_cells=6,
        pml_cells=2,
        source_clearance_cells=1,
    )
    guide.normal_axis = 2
    guide.direction_sign = 1
    guide.plane_index = 1
    guide.nu = nu
    guide.nv = nv
    guide.mpi = False
    guide._adjacent_component_ids = lambda: (np.zeros(1), np.zeros(1))
    guide._adjacent_solids = lambda: (np.zeros(1), np.zeros(1))
    guide._component_cross_section = lambda: np.zeros(1, dtype=np.uint32)
    return guide


def test_device_validation_allows_wide_main_field_addresses(monkeypatch):
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(general={"solver": "metal"}))
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D"))
    guide = _validation_guide(size=(50_000, 50_000, 2), nu=4, nv=5)

    assert int(np.prod(np.asarray(guide.main_grid.size, dtype=object) + 1)) > np.iinfo(
        np.int32
    ).max
    guide._validate()


def test_device_validation_still_rejects_oversized_compact_dispatch(monkeypatch):
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(general={"solver": "metal"}))
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D"))
    guide = _validation_guide(size=(60_000, 60_000, 2), nu=50_000, nv=50_000)

    with pytest.raises(ValueError, match="device indexing exceeds"):
        guide._validate()


@pytest.mark.parametrize(
    "specification,field_kind",
    [
        (knl_virtual_waveguide.clear_rear_magnetic, "H"),
        (knl_virtual_waveguide.clear_rear_electric, "E"),
    ],
)
def test_rear_clear_kernels_use_compact_coordinates_and_wide_main_addresses(
    specification, field_kind
):
    body = specification["func"].template

    assert "size_t uv_plane" in body
    assert "size_t main_index = IDX3D_FIELDS" in body
    assert "$NY_FIELDS * $NZ_FIELDS" not in body
    assert f"main_{field_kind}x[i]" not in body
    for backend in ("cuda", "opencl", "metal"):
        arguments = specification[f"args_{backend}"].template
        assert f"main_{field_kind}x" in arguments


def test_aperture_coupling_keeps_main_grid_offsets_pointer_sized():
    magnetic = knl_virtual_waveguide.couple_magnetic["func"].template
    electric = knl_virtual_waveguide.couple_electric["func"].template

    assert "size_t main_index" in magnetic
    assert "int main_index" not in magnetic
    assert "size_t midx" in electric
    assert "int aidx = 0, midx" not in electric


@pytest.mark.parametrize("updates", [CUDAUpdates, OpenCLUpdates, MetalUpdates])
def test_accelerator_rear_clear_dispatch_does_not_use_the_main_field_size(updates):
    magnetic = inspect.getsource(updates._update_virtual_waveguide_magnetic)
    electric = inspect.getsource(updates._update_virtual_waveguide_electric)

    assert "_rear_clear_points(magnetic=True)" in magnetic
    assert "_rear_clear_points(magnetic=False)" in electric
    assert "self.grid.Ex.size" not in magnetic + electric


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
@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
def test_device_virtual_waveguide_matches_cpu(tmp_path, request, backend, normal_axis, direction):
    device_options, device_precision = _device_options(request, backend)

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
        gpu_precision=device_precision,
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
@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
def test_device_virtual_waveguide_broadband_matches_cpu(tmp_path, request, backend):
    device_options, device_precision = _device_options(request, backend)

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
        gpu_precision=device_precision,
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
@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
def test_device_passive_virtual_waveguide_matches_cpu(tmp_path, request, backend):
    device_options, device_precision = _device_options(request, backend)

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
        gpu_precision=device_precision,
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
@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
def test_device_direct_eigenmode_source_matches_cpu(tmp_path, request, backend):
    device_options, device_precision = _device_options(request, backend)

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
        gpu_precision=device_precision,
        **device_options,
        **run_options,
    )

    cpu = _port_results(cpu_path)
    device = _port_results(device_path)
    incident_scale = max(float(np.max(np.abs(cpu["incident"]))), 1e-20)
    for name in cpu:
        absolute_tolerance = 5e-5 if name == "S" else 5e-5 * incident_scale
        assert_allclose(device[name], cpu[name], rtol=5e-4, atol=absolute_tolerance)
