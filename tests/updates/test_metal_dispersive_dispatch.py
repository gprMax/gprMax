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

"""Regression tests for full dispersive-material (Debye/Lorentz/Drude)
support on the Metal backend (gprMax/updates/metal_updates.py).

Previously, Metal completely ignored dispersive materials:
update_electric_a's maxpoles>0 branch just duplicated the non-dispersive
update with a "TODO" comment, and update_electric_b was a bare `pass` -
users got a silently non-dispersive result for any dispersive material.
The user asked to finish this properly, using the CUDA implementation
(gprMax/updates/cuda_updates.py) as the trusted physics reference -
dispersive_update_a/_b there are assumed correct, and the Metal kernels
(knl_fields_updates.update_electric_dispersive_A/_B) already have
complete args_metal templates with the identical phi/T-array Phase A/B
formulas, so this is a wiring job, not new physics.

Fixed by:
1. _set_field_knls(): building both dispersive kernels (mirroring
   CUDAUpdates._set_field_knls()'s equivalent block) and eagerly
   uploading Tx/Ty/Tz/updatecoeffsdispersive via
   self.grid.htod_dispersive_arrays(self.dev).
2. update_electric_a()'s maxpoles>0 branch: a real Metal dispatch of
   update_electric_dispersive_A (15 args: NX,NY,NZ,MAXPOLES,
   updatecoeffsdispersive,Tx,Ty,Tz,ID,Ex,Ey,Ez,Hx,Hy,Hz).
3. update_electric_b(): a real Metal dispatch of
   update_electric_dispersive_B (12 args - same as A but no H
   components, since phase B only re-updates T from post-update E).
4. Metal Shading Language has no native complex type (unlike CUDA's
   pycuda::complex<T> or OpenCL's pyopencl-complex.h) - added a small
   custom `gprMaxComplex` struct (+/-/* operators, .real()) to
   knl_common_metal.tmpl for the Lorentz/Drude (complex-pole) case, and
   pointed config.py's C_complex at it instead of the nonexistent
   "metal::complex<float>".
5. Removed the now-obsolete stopgap warning in solvers.py's
   create_solver() (see test_metal_dispersive_warning.py).

Real Apple Metal hardware/PyObjC isn't available in this environment, so
these tests exercise the dispatch methods directly against a
MetalUpdates instance built with __new__ (bypassing __init__) and fake
Metal API stand-ins, checking the exact buffer-index-to-kernel-argument
contract. The MSL source itself (including the gprMaxComplex struct)
cannot be compile-verified without real hardware - flagged explicitly,
not silently assumed correct.
"""
import numpy as np

from gprMax import config
from gprMax.updates.metal_updates import MetalUpdates


class _FakeBuffer:
    def __init__(self, data=b""):
        self.data = bytes(data)


class _FakeLib:
    def newFunctionWithName_(self, name):
        return f"function:{name}"


class _FakePSO:
    def __init__(self, max_threads=64):
        self._max_threads = max_threads

    def maxTotalThreadsPerThreadgroup(self):
        return self._max_threads


class _FakeDevice:
    def newBufferWithBytes_length_options_(self, data, length, options):
        return _FakeBuffer(data)

    def newLibraryWithSource_options_error_(self, source, opts, error):
        return _FakeLib(), None

    def newComputePipelineStateWithFunction_error_(self, func, error):
        return (_FakePSO(), None)


class _FakeEncoder:
    def __init__(self):
        self.pso = None
        self.bytes_args = {}
        self.buffers = {}
        self.dispatched_group_size = None
        self.ended = False

    def setComputePipelineState_(self, pso):
        self.pso = pso

    def setBytes_length_atIndex_(self, data, length, index):
        self.bytes_args[index] = np.frombuffer(data, dtype=np.int32)[0]

    def setBuffer_offset_atIndex_(self, buf, offset, index):
        self.buffers[index] = buf

    def dispatchThreads_threadsPerThreadgroup_(self, grid_size, group_size):
        self.dispatched_group_size = group_size

    def endEncoding(self):
        self.ended = True


class _FakeCommandBuffer:
    def __init__(self):
        self.encoder = _FakeEncoder()

    def computeCommandEncoder(self):
        return self.encoder

    def commit(self):
        pass

    def waitUntilCompleted(self):
        pass


class _FakeCommandQueue:
    def __init__(self):
        self.buffers_created = []

    def commandBuffer(self):
        buf = _FakeCommandBuffer()
        self.buffers_created.append(buf)
        return buf


class _FakeMetalModule:
    def MTLSizeMake(self, x, y, z):
        return (x, y, z)


def _make_updates(monkeypatch, maxpoles):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: type("_MC", (), {"materials": {"maxpoles": maxpoles}})(),
    )

    updates = MetalUpdates.__new__(MetalUpdates)
    updates.cmdqueue = _FakeCommandQueue()
    updates.metal = _FakeMetalModule()
    updates.psoE = "psoE"
    updates.pso_dispersive_a = _FakePSO(max_threads=32)
    updates.pso_dispersive_b = _FakePSO(max_threads=16)

    class _Grid:
        nx, ny, nz = 10, 12, 14
        tptg = "tptg"
        tgs = "field-sized-tgs"
        ID_dev = _FakeBuffer(b"id")
        Ex_dev = _FakeBuffer(b"ex")
        Ey_dev = _FakeBuffer(b"ey")
        Ez_dev = _FakeBuffer(b"ez")
        Hx_dev = _FakeBuffer(b"hx")
        Hy_dev = _FakeBuffer(b"hy")
        Hz_dev = _FakeBuffer(b"hz")
        updatecoeffsdispersive_dev = _FakeBuffer(b"dispcoeffs")
        Tx_dev = _FakeBuffer(b"tx")
        Ty_dev = _FakeBuffer(b"ty")
        Tz_dev = _FakeBuffer(b"tz")

    updates.grid = _Grid()
    updates.grid.maxpoles = maxpoles
    return updates


def test_update_electric_a_dispersive_dispatches_with_correct_15_arg_contract(monkeypatch):
    updates = _make_updates(monkeypatch, maxpoles=2)

    updates.update_electric_a()

    assert len(updates.cmdqueue.buffers_created) == 1
    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.pso is updates.pso_dispersive_a
    assert encoder.ended

    assert encoder.bytes_args[0] == updates.grid.nx
    assert encoder.bytes_args[1] == updates.grid.ny
    assert encoder.bytes_args[2] == updates.grid.nz
    assert encoder.bytes_args[3] == 2

    assert set(encoder.buffers.keys()) == set(range(4, 15))
    assert encoder.buffers[4] is updates.grid.updatecoeffsdispersive_dev
    assert encoder.buffers[5] is updates.grid.Tx_dev
    assert encoder.buffers[6] is updates.grid.Ty_dev
    assert encoder.buffers[7] is updates.grid.Tz_dev
    assert encoder.buffers[8] is updates.grid.ID_dev
    assert encoder.buffers[9] is updates.grid.Ex_dev
    assert encoder.buffers[10] is updates.grid.Ey_dev
    assert encoder.buffers[11] is updates.grid.Ez_dev
    assert encoder.buffers[12] is updates.grid.Hx_dev
    assert encoder.buffers[13] is updates.grid.Hy_dev
    assert encoder.buffers[14] is updates.grid.Hz_dev

    # Must use pso_dispersive_a's OWN threadgroup limit (32), not the
    # shared, differently-sized self.grid.tgs (which is sized for psoE).
    assert encoder.dispatched_group_size == (32, 1, 1)


def test_update_electric_b_dispersive_dispatches_with_correct_12_arg_contract(monkeypatch):
    updates = _make_updates(monkeypatch, maxpoles=3)

    updates.update_electric_b()

    assert len(updates.cmdqueue.buffers_created) == 1
    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.pso is updates.pso_dispersive_b
    assert encoder.ended

    assert encoder.bytes_args[0] == updates.grid.nx
    assert encoder.bytes_args[1] == updates.grid.ny
    assert encoder.bytes_args[2] == updates.grid.nz
    assert encoder.bytes_args[3] == 3

    # Phase B has no H components - only 12 args (0-11), not 15.
    assert set(encoder.buffers.keys()) == set(range(4, 12))
    assert encoder.buffers[4] is updates.grid.updatecoeffsdispersive_dev
    assert encoder.buffers[5] is updates.grid.Tx_dev
    assert encoder.buffers[6] is updates.grid.Ty_dev
    assert encoder.buffers[7] is updates.grid.Tz_dev
    assert encoder.buffers[8] is updates.grid.ID_dev
    assert encoder.buffers[9] is updates.grid.Ex_dev
    assert encoder.buffers[10] is updates.grid.Ey_dev
    assert encoder.buffers[11] is updates.grid.Ez_dev

    # Must use pso_dispersive_b's OWN threadgroup limit (16), independent
    # of pso_dispersive_a's (32) or the shared field-sized tgs.
    assert encoder.dispatched_group_size == (16, 1, 1)


def test_update_electric_b_no_op_when_no_dispersive_materials(monkeypatch):
    updates = _make_updates(monkeypatch, maxpoles=0)

    updates.update_electric_b()

    assert updates.cmdqueue.buffers_created == []


def test_update_electric_a_non_dispersive_path_unaffected(monkeypatch):
    """maxpoles == 0 must still take the plain psoE dispatch, never touching
    pso_dispersive_a."""
    updates = _make_updates(monkeypatch, maxpoles=0)

    updates.update_electric_a()

    encoder = updates.cmdqueue.buffers_created[0].encoder
    assert encoder.pso == "psoE"


def test_set_field_knls_builds_dispersive_kernels_and_uploads_arrays_when_maxpoles_gt_0(
    monkeypatch,
):
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: type(
            "_MC",
            (),
            {"materials": {"maxpoles": 1, "crealfunc": "", "dispersiveCdtype": "float"}},
        )(),
    )
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.dtypes = {"C_float_or_double": "float"}

    updates = MetalUpdates.__new__(MetalUpdates)
    updates.dev = _FakeDevice()
    updates.opts = None
    updates.metal = _FakeMetalModule()
    updates.knl_common = ""
    updates.subs_func = {
        "REAL": "float",
        "CUDA_IDX": "",
        "NX_FIELDS": 5,
        "NY_FIELDS": 5,
        "NZ_FIELDS": 5,
        "NX_ID": 4,
        "NY_ID": 4,
        "NZ_ID": 4,
    }
    updates.subs_name_args = {"REAL": "float", "COMPLEX": "float"}

    htod_calls = []

    class _Grid:
        nx = ny = nz = 4
        maxpoles = 1
        crealfunc = ""
        pmls = {"slabs": []}
        rxs = []
        voltagesources = hertziandipoles = magneticdipoles = []
        snapshots = []
        Tx = np.zeros((1, 2, 2, 2))

        def set_threads_per_thread_group(self):
            pass

        def set_thread_group_size(self, pso):
            pass

        def htod_dispersive_arrays(self, dev):
            htod_calls.append(dev)

        def htod_geometry_arrays(self, dev):
            pass

        def htod_field_arrays(self, dev):
            pass

        def htod_material_arrays(self, dev):
            pass

    updates.grid = _Grid()

    updates._set_field_knls()

    assert updates.pso_dispersive_a is not None
    assert updates.pso_dispersive_b is not None
    assert htod_calls == [updates.dev]
