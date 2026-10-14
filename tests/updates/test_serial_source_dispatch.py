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

"""CPU-only checks of serial source dispatch and shared kernel structure.

The real host methods run against recording transports, not GPU drivers.
These tests protect launch sizes, argument mapping and source-family order;
they do not compile device code or validate numerical results on hardware.
"""

import re
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.cuda_opencl import knl_source_updates, knl_transmission_line
from gprMax.updates.cuda_updates import CUDAUpdates
from gprMax.updates.metal_updates import MetalUpdates
from gprMax.updates.opencl_updates import OpenCLUpdates

pytestmark = pytest.mark.unit

BACKENDS = {"cuda": CUDAUpdates, "opencl": OpenCLUpdates, "metal": MetalUpdates}
FAMILIES = (
    ("voltage_source", "voltagesources", "voltage", "E"),
    ("hertzian_dipole", "hertziandipoles", "hertzian", "E"),
    ("magnetic_dipole", "magneticdipoles", "magnetic", "H"),
)


class _MetalEncoder:
    def __init__(self, calls):
        self.calls = calls
        self.arguments = {}
        self.record = None

    def setComputePipelineState_(self, pipeline):
        self.pipeline = pipeline

    def setBuffer_offset_atIndex_(self, buffer, offset, index):
        assert offset == 0
        self.arguments[index] = buffer

    def setBytes_length_atIndex_(self, data, length, index):
        assert len(data) == length
        self.arguments[index] = SimpleNamespace(data=bytes(data))

    def dispatchThreads_threadsPerThreadgroup_(self, grid_size, group_size):
        assert set(self.arguments) == set(range(len(self.arguments)))
        self.record = SimpleNamespace(
            name=self.pipeline.name,
            args=tuple(self.arguments[index] for index in range(len(self.arguments))),
            kwargs={"grid": grid_size, "group": group_size},
            ended=False,
            committed=False,
            waited=False,
        )
        self.calls.append(self.record)

    def endEncoding(self):
        self.record.ended = True


class _MetalQueue:
    def __init__(self, calls):
        self.calls = calls

    def commandBuffer(self):
        encoder = _MetalEncoder(self.calls)
        return SimpleNamespace(
            computeCommandEncoder=lambda: encoder,
            commit=lambda: setattr(encoder.record, "committed", True),
            waitUntilCompleted=lambda: setattr(encoder.record, "waited", True),
        )

    def submit(self, command):
        command.commit()


def _record_kernel(calls, name):
    def kernel(*args, **kwargs):
        calls.append(SimpleNamespace(name=name, args=args, kwargs=kwargs))

    return kernel


def _make_updates(backend, count):
    """Give each family distinct buffers and a different nonzero source count."""

    updates = BACKENDS[backend].__new__(BACKENDS[backend])
    calls = []
    grid = SimpleNamespace(dx=0.001, dy=0.002, dz=0.003, iteration=7)
    for multiplier, attribute in enumerate(
        (
            "voltagesources",
            "transmissionlines",
            "hertziandipoles",
            "magneticdipoles",
            "magneticfrillsources",
        ),
        start=1,
    ):
        setattr(grid, attribute, [object() for _ in range(multiplier * count)])
    for name in ("ID", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        setattr(grid, f"{name}_dev", SimpleNamespace(gpudata=object()))
    updates.grid = grid
    for _, _, prefix, _ in FAMILIES:
        for array in ("srcinfo1", "srcinfo2", "srcwaves"):
            setattr(updates, f"{array}_{prefix}_dev", SimpleNamespace(gpudata=object()))
    for name in (
        "tl_info",
        "tl_resistance",
        "tl_waveform_whole",
        "tl_voltage",
        "tl_current",
        "tl_abcv0",
        "tl_abcv1",
        "frill_term_counts",
        "frill_term_info",
        "frill_term_params",
        "frill_params",
        "frill_state",
        "frill_waveform",
        "frill_Vinc",
        "frill_Vtotal",
        "frill_Itot",
    ):
        setattr(updates, f"{name}_dev", SimpleNamespace(gpudata=object()))
    updates.tl_line_coefficient = np.float64(0.5)
    updates.tl_abc_coefficient = np.float64(-0.25)
    # Deliberately nonserial sizes: the electric TL dispatch must no longer
    # borrow the per-line magnetic launch configuration.
    updates.tl_tpb = (32, 1, 1)
    updates.tl_bpg = (3, 1, 1)
    updates.frill_tpb = (32, 1, 1)
    updates.frill_bpg = (2, 1, 1)
    updates.nrxcurrent = 0

    if backend == "metal":
        updates.dev = SimpleNamespace(
            newBufferWithBytes_length_options_=lambda data, length, options: SimpleNamespace(
                data=bytes(data)
            )
        )
        updates.cmdqueue = _MetalQueue(calls)
        updates.metal = SimpleNamespace(MTLSizeMake=lambda x, y, z: (x, y, z))
    for name in (
        "voltage_source",
        "hertzian_dipole",
        "magnetic_dipole",
        "transmission_line_electric",
        "magnetic_frill_source",
    ):
        if backend == "metal":
            attribute = "magnetic_frill" if name == "magnetic_frill_source" else name
            setattr(
                updates,
                f"pso_{attribute}",
                SimpleNamespace(name=name, maxTotalThreadsPerThreadgroup=lambda: 64),
            )
        else:
            setattr(updates, f"update_{name}_dev", _record_kernel(calls, name))
    return updates, calls


def _scalar(call, index, dtype, backend):
    value = call.args[index]
    if backend == "metal":
        return np.frombuffer(value.data, dtype=dtype).item()
    assert np.asarray(value).dtype == np.dtype(dtype)
    return value


def _assert_serial_launch(call, backend):
    if backend == "cuda":
        assert call.kwargs["block"] == (1, 1, 1)
        assert call.kwargs["grid"] == (1, 1, 1)
    elif backend == "opencl":
        assert call.kwargs["range"] == slice(0, 1)
    else:
        assert call.kwargs["grid"] == (1, 1, 1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("count", [1, 17])
def test_source_dispatch_keeps_full_counts_and_cpu_family_order(backend, count):
    updates, calls = _make_updates(backend, count)
    # Avoid the optional first-iteration Metal diagnostics; only dispatch is
    # under test, and these sentinel buffers do not contain field data.
    iteration = 7
    updates.update_electric_sources(iteration)
    updates.update_magnetic_sources(iteration)

    assert [call.name for call in calls] == [
        "voltage_source",
        "transmission_line_electric",
        "hertzian_dipole",
        "magnetic_dipole",
        "magnetic_frill_source",
    ]
    by_name = {call.name: call for call in calls}
    for name, attribute, prefix, component in FAMILIES:
        call = by_name[name]
        _assert_serial_launch(call, backend)
        assert _scalar(call, 0, np.int32, backend) == len(getattr(updates.grid, attribute))
        assert _scalar(call, 1, np.int32, backend) == iteration
        for index, axis in enumerate("xyz", start=2):
            assert _scalar(call, index, np.float64, backend) == getattr(updates.grid, f"d{axis}")
        expected = [
            getattr(updates, f"{array}_{prefix}_dev")
            for array in ("srcinfo1", "srcinfo2", "srcwaves")
        ]
        expected += [updates.grid.ID_dev]
        expected += [getattr(updates.grid, f"{component}{axis}_dev") for axis in "xyz"]
        if backend == "cuda":
            expected = [buffer.gpudata for buffer in expected]
        assert len(call.args) == 12
        assert all(actual is buffer for actual, buffer in zip(call.args[5:], expected))

    tl_call = by_name["transmission_line_electric"]
    _assert_serial_launch(tl_call, backend)
    assert _scalar(tl_call, 0, np.int32, backend) == len(updates.grid.transmissionlines)
    assert _scalar(tl_call, 1, np.int32, backend) == iteration
    assert _scalar(by_name["magnetic_frill_source"], 0, np.int32, backend) == len(
        updates.grid.magneticfrillsources
    )
    if backend == "metal":
        assert all(call.ended and call.committed and not call.waited for call in calls)


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_source_lists_do_not_launch(backend):
    updates, calls = _make_updates(backend, count=0)
    updates.update_electric_sources(iteration=7)
    updates.update_magnetic_sources(iteration=7)
    assert calls == []


@pytest.mark.parametrize(
    "template,count_name",
    [
        (knl_source_updates.update_voltage_source, "NVOLTSRC"),
        (knl_source_updates.update_hertzian_dipole, "NHERTZDIPOLE"),
        (knl_source_updates.update_magnetic_dipole, "NMAGDIPOLE"),
        (knl_transmission_line.update_transmission_line_electric, "NTL"),
    ],
    ids=["voltage", "hertzian", "magnetic", "transmission-line-electric"],
)
def test_shared_source_body_uses_lane_zero_and_ascending_local_source_index(template, count_name):
    # Inspect the shared operation without CUDA's index declaration. OpenCL
    # supplies its own outer i loop and Metal supplies i as a kernel argument.
    body = template["func"].safe_substitute(CUDA_IDX="", REAL="double")
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.DOTALL)
    loop = re.search(
        r"if\s*\(\s*i\s*==\s*0\s*\)\s*\{\s*"
        r"for\s*\(\s*int\s+(?P<index>\w+)\s*=\s*0\s*;\s*"
        rf"(?P=index)\s*<\s*{count_name}\s*;\s*"
        r"(?:\+\+\s*(?P=index)|(?P=index)\s*\+\+)\s*\)",
        body,
    )
    assert loop is not None
    assert loop.group("index") != "i", "Do not reuse OpenCL's outer work-item index"
    # All per-source indexing, including the voltage activity-tail offset,
    # must use the local index rather than accidentally reusing lane zero.
    assert len(re.findall(r"\bi\b", body)) == 1
