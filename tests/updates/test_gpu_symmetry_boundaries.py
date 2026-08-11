"""Hardware-independent coverage of accelerator symmetry boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax import config
from gprMax.cuda_opencl.knl_symmetry_boundaries import (
    dispersive_substitutions,
    nondispersive_substitutions,
    update_electric_pmc,
    update_electric_pmc_dispersive,
    update_electric_pmc_dispersive_b,
)
from gprMax.model import Model
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.updates.metal_updates import MetalUpdates
from gprMax.updates.opencl_updates import OpenCLUpdates
from gprMax.user_objects.cmds_multiuse import SymmetryBoundary


@pytest.mark.parametrize(
    "backend,marker",
    [
        ("cuda", "__global__ void update_electric_pmc"),
        ("opencl", "__global float* Ex"),
        ("metal", "thread_position_in_grid"),
    ],
)
def test_pmc_kernel_templates_cover_faces_and_edges(backend, marker):
    declaration = update_electric_pmc[f"args_{backend}"].substitute(REAL="float")
    substitutions = dict(
        REAL="float",
        CUDA_IDX=(
            "int i = blockIdx.x * blockDim.x + threadIdx.x;"
            if backend == "cuda"
            else ""
        ),
        NX_FIELDS=5,
        NY_FIELDS=6,
        NZ_FIELDS=7,
        NX_ID=5,
        NY_ID=6,
        NZ_ID=7,
    )
    substitutions.update(nondispersive_substitutions())
    body = update_electric_pmc["func"].substitute(substitutions)

    source = declaration + body
    assert marker in source
    assert "ex_on_pmc" in source
    assert "ey_on_pmc" in source
    assert "ez_on_pmc" in source
    assert "PMC_X0" in source and "PMC_XMAX" in source
    assert "PMC_Y0" in source and "PMC_YMAX" in source
    assert "PMC_Z0" in source and "PMC_ZMAX" in source


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
def test_dispersive_pmc_templates_include_two_phase_ade_update(backend):
    arguments = {"REAL": "float", "COMPLEX": "float"}
    substitutions = {
        "REAL": "float",
        "CUDA_IDX": (
            "int i = blockIdx.x * blockDim.x + threadIdx.x;"
            if backend == "cuda"
            else ""
        ),
        "NX_FIELDS": 5,
        "NY_FIELDS": 6,
        "NZ_FIELDS": 7,
        "NX_ID": 5,
        "NY_ID": 6,
        "NZ_ID": 7,
        "NX_T": 5,
        "NY_T": 6,
        "NZ_T": 7,
    }
    phase_a = dict(substitutions)
    phase_a.update(dispersive_substitutions("float"))
    source_a = update_electric_pmc_dispersive[f"args_{backend}"].substitute(
        arguments
    ) + update_electric_pmc_dispersive["func"].substitute(phase_a)
    source_b = update_electric_pmc_dispersive_b[f"args_{backend}"].substitute(
        arguments
    ) + update_electric_pmc_dispersive_b["func"].substitute(substitutions)

    assert "MAXPOLES" in source_a and "MAXPOLES" in source_b
    assert "GPRMAX_CREAL" in source_a
    assert "GPRMAX_CADD" in source_a
    assert "GPRMAX_CSUB" in source_b
    assert "$DISP_" not in source_a and "$PHI_" not in source_a


def _command_grid():
    return SimpleNamespace(
        symmetry_boundaries={},
        pmls={
            "thickness": {
                face: 10
                for face in ("x0", "xmax", "y0", "ymax", "z0", "zmax")
            }
        },
    )


@pytest.mark.parametrize("solver", ["cuda", "opencl", "metal"])
@pytest.mark.parametrize("boundary_type", ["pec", "pmc"])
def test_nondispersive_gpu_symmetry_command_is_accepted(
    monkeypatch, solver, boundary_type
):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(general={"solver": solver}, mpi=False),
    )
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(materials={"maxpoles": 0}, mode="3D"),
    )
    grid = _command_grid()

    SymmetryBoundary(face="x0", type=boundary_type).build(grid)

    assert grid.symmetry_boundaries == {"x0": boundary_type}
    assert grid.pmls["thickness"]["x0"] == 0


@pytest.mark.parametrize("solver", ["cuda", "opencl", "metal"])
def test_dispersive_gpu_pmc_is_accepted_after_material_resolution(
    monkeypatch, solver
):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(general={"solver": solver}, mpi=False),
    )
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(materials={"maxpoles": 1}, mode="3D"),
    )

    grid = _command_grid()
    SymmetryBoundary(face="x0", type="pmc").build(grid)

    Model.__new__(Model)._check_accelerator_symmetry_boundaries([grid])
    assert grid.symmetry_boundaries == {"x0": "pmc"}


class _FakeCUDAArray:
    def __init__(self, name):
        self.gpudata = name


def _dispatch_grid():
    arrays = {
        name: _FakeCUDAArray(name)
        for name in ("ID_dev", "Ex_dev", "Ey_dev", "Ez_dev", "Hx_dev", "Hy_dev", "Hz_dev")
    }
    return SimpleNamespace(
        nx=4,
        ny=5,
        nz=6,
        symmetry_boundaries={"x0": "pmc", "ymax": "pmc"},
        tpb=(64, 1, 1),
        bpg=(2, 1, 1),
        **arrays,
    )


def test_cuda_pmc_dispatch_uses_canonical_face_flag_order(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(materials={"maxpoles": 0}),
    )
    updates = CUDAUpdates.__new__(CUDAUpdates)
    updates.grid = _dispatch_grid()
    calls = []
    updates.update_electric_pmc_dev = lambda *args, **kwargs: calls.append(
        (args, kwargs)
    )

    updates.update_symmetry_boundaries_electric()

    args, kwargs = calls[0]
    assert args[:3] == (4, 5, 6)
    assert args[3:9] == (1, 0, 0, 1, 0, 0)
    assert args[9:16] == (
        "ID_dev",
        "Ex_dev",
        "Ey_dev",
        "Ez_dev",
        "Hx_dev",
        "Hy_dev",
        "Hz_dev",
    )
    assert kwargs == {"block": (64, 1, 1), "grid": (2, 1, 1)}


def test_opencl_pmc_dispatch_uses_device_arrays(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(materials={"maxpoles": 0}),
    )
    updates = OpenCLUpdates.__new__(OpenCLUpdates)
    updates.grid = _dispatch_grid()
    calls = []
    updates.update_electric_pmc_dev = lambda *args: calls.append(args)

    updates.update_symmetry_boundaries_electric()

    args = calls[0]
    assert args[3:9] == (1, 0, 0, 1, 0, 0)
    assert args[9].gpudata == "ID_dev"
    assert args[15].gpudata == "Hz_dev"


def test_opencl_pmc_kernel_builder_substitutes_real_type(monkeypatch):
    updates = OpenCLUpdates.__new__(OpenCLUpdates)
    updates.grid = SimpleNamespace(
        nx=4,
        ny=5,
        nz=6,
        ID=np.zeros((6, 5, 6, 7), dtype=np.uint32),
    )
    updates.ctx = object()
    updates.knl_common = "common"
    captured = {}

    def fake_elementwise_kernel(ctx, arguments, operation, name, **kwargs):
        captured.update(arguments=arguments, operation=operation, name=name, kwargs=kwargs)
        return object()

    updates.elwiseknl = fake_elementwise_kernel
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            dtypes={"C_float_or_double": "float"},
            devices={"compiler_opts": []},
        ),
    )
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(materials={"maxpoles": 0}),
    )

    updates._set_symmetry_boundary_knl()

    assert "$REAL" not in captured["arguments"]
    assert "$REAL" not in captured["operation"]
    assert "float" in captured["arguments"]
    assert "float" in captured["operation"]
    assert captured["name"] == "update_electric_pmc"


class _FakeMetalEncoder:
    def __init__(self):
        self.pipeline = None
        self.scalars = {}
        self.buffers = {}
        self.dispatch = None
        self.ended = False

    def setComputePipelineState_(self, pipeline):
        self.pipeline = pipeline

    def setBytes_length_atIndex_(self, value, length, index):
        self.scalars[index] = (value, length)

    def setBuffer_offset_atIndex_(self, buffer, offset, index):
        self.buffers[index] = (buffer, offset)

    def dispatchThreads_threadsPerThreadgroup_(self, threads, group):
        self.dispatch = (threads, group)

    def endEncoding(self):
        self.ended = True


class _FakeMetalCommand:
    def __init__(self, encoder):
        self.encoder = encoder
        self.committed = False
        self.waited = False

    def computeCommandEncoder(self):
        return self.encoder

    def commit(self):
        self.committed = True

    def waitUntilCompleted(self):
        self.waited = True


def test_metal_pmc_dispatch_uses_expected_scalar_and_buffer_slots(monkeypatch):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(materials={"maxpoles": 0}),
    )
    updates = MetalUpdates.__new__(MetalUpdates)
    updates.grid = _dispatch_grid()
    updates.grid.tptg = "threads"
    encoder = _FakeMetalEncoder()
    command = _FakeMetalCommand(encoder)
    updates.cmdqueue = SimpleNamespace(commandBuffer=lambda: command)
    updates.pso_electric_pmc = SimpleNamespace(
        maxTotalThreadsPerThreadgroup=lambda: 128
    )
    updates.metal = SimpleNamespace(
        MTLSizeMake=lambda x, y, z: (x, y, z)
    )

    updates.update_symmetry_boundaries_electric()

    scalar_values = tuple(
        np.frombuffer(encoder.scalars[index][0], dtype=np.int32)[0]
        for index in range(9)
    )
    assert scalar_values == (4, 5, 6, 1, 0, 0, 1, 0, 0)
    assert tuple(encoder.buffers) == tuple(range(9, 16))
    assert encoder.buffers[9] == (updates.grid.ID_dev, 0)
    assert encoder.buffers[15] == (updates.grid.Hz_dev, 0)
    assert encoder.pipeline is updates.pso_electric_pmc
    assert encoder.dispatch == ("threads", (128, 1, 1))
    assert encoder.ended and command.committed and command.waited
