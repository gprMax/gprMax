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

"""PML launches use both staggered history volumes, not the full grid ID array."""

import re
import sys
from importlib import import_module
from itertools import product
from math import prod
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gprMax.pml import CFS, PML, OpenCLPML

pytestmark = pytest.mark.unit
HISTORIES = ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
COEFFICIENTS = ("ERA", "ERB", "ERE", "ERF", "HRA", "HRB", "HRE", "HRF")


@pytest.fixture
def fake_opencl_upload(monkeypatch):
    """Keep these tests independent of pyopencl and available hardware."""

    uploads = []
    package = ModuleType("pyopencl")
    package.__path__ = []
    arrays = ModuleType("pyopencl.array")

    def to_device(queue, host):
        device = SimpleNamespace(shape=tuple(host.shape), queue=queue, host=host)
        uploads.append((queue, host, device))
        return device

    arrays.to_device = to_device
    package.array = arrays
    monkeypatch.setitem(sys.modules, "pyopencl", package)
    monkeypatch.setitem(sys.modules, "pyopencl.array", arrays)
    return uploads


@pytest.fixture
def make_opencl_pml(make_pml_grid, fake_opencl_upload):
    def make(direction="xminus", formulation="HORIPML", order=1, transpose=False, kind="native"):
        axis = "xyz".index(direction[0])
        transverse = [index for index in range(3) if index != axis]
        lengths = (9, 5) if transpose else (5, 9)
        domain = [20, 20, 20]
        for index, length in zip(transverse, lengths):
            domain[index] = length + 6
        grid = make_pml_grid(
            nx=domain[0],
            ny=domain[1],
            nz=domain[2],
            arrays=False,
            dl=(0.001, 0.002, 0.004),
            formulation=formulation,
            cfs=[CFS() for _ in range(order)],
        )
        if order == 2 and formulation == "HORIPML":
            grid.pmls["cfs"][1].alpha.max = 20.0
        lower = [0, 0, 0]
        upper = [value + 1 for value in domain]
        if kind == "internal":
            lower[axis], upper[axis] = 5, 9
            for index, length in zip(transverse, lengths):
                lower[index], upper[index] = 2, 2 + length
        elif direction.endswith("minus"):
            lower[axis], upper[axis] = 0, 4
        else:
            lower[axis], upper[axis] = domain[axis] - 4, domain[axis]
        bounds = tuple(value for pair in zip(lower, upper) for value in pair)
        slab = OpenCLPML(
            grid,
            "test_slab",
            direction,
            *bounds,
            internal=kind != "native",
            formulation=formulation,
        )
        slab.calculate_update_coeffs(er=1, mr=1)
        for component in ("ID", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            setattr(grid, f"{component}_dev", object())
        slab.set_queue(object())
        slab.htod_field_arrays()
        return slab, bounds

    return make


def _record_kernels(slab):
    calls = []

    def kernel(family):
        def invoke(*args, **kwargs):
            calls.append((family, args, kwargs))
            return SimpleNamespace(wait=lambda: calls.append((family, "wait")))

        return invoke

    slab.update_electric_dev = kernel("E")
    slab.update_magnetic_dev = kernel("H")
    return calls


@pytest.mark.parametrize("direction", PML.directions)
@pytest.mark.parametrize("formulation", PML.formulations)
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("transpose", [False, True], ids=["short-long", "long-short"])
@pytest.mark.parametrize("kind", ["native", "internal", "boundary-replacement"])
def test_opencl_pml_uses_each_fields_larger_spatial_history(
    make_opencl_pml, fake_opencl_upload, direction, formulation, order, transpose, kind
):
    slab, bounds = make_opencl_pml(direction, formulation, order, transpose, kind)
    calls = _record_kernels(slab)
    slab.update_electric()
    slab.update_magnetic()

    assert len(fake_opencl_upload) == 12
    assert all(queue is slab.queue for queue, _, _ in fake_opencl_upload)
    assert all(
        host is getattr(slab, name)
        for (_, host, _), name in zip(fake_opencl_upload, COEFFICIENTS + HISTORIES)
    )
    assert len(calls) == 4
    assert calls[1] == ("E", "wait")
    assert calls[3] == ("H", "wait")

    electric_bounds = list(bounds)
    if kind == "internal":
        endpoint = 2 * "xyz".index(direction[0]) + int(direction.endswith("plus"))
        electric_bounds[endpoint] += 1 if direction.endswith("plus") else -1

    ranges = []
    for call_index, family in ((0, "E"), (2, "H")):
        _, args, kwargs = calls[call_index]
        histories = [getattr(slab, f"{family}Phi{index}") for index in (1, 2)]
        volumes = [prod(history.shape[1:]) for history in histories]
        assert volumes[0] != volumes[1]  # both aspect ratios must exercise a real max
        expected_range = slice(0, max(volumes))
        assert kwargs == {"range": expected_range}
        cache_name = "_electric_update_range" if family == "E" else "_magnetic_update_range"
        assert kwargs["range"] is getattr(slab, cache_name)
        assert type(kwargs["range"].stop) is int
        if order == 2:
            assert expected_range.stop != max(history.size for history in histories)
        ranges.append(expected_range.stop)

        assert len(args) == 28
        assert all(isinstance(value, np.int32) for value in args[:13])
        assert tuple(args[:6]) == tuple(electric_bounds if family == "E" else bounds)
        assert tuple(args[6:12]) == histories[0].shape[1:] + histories[1].shape[1:]
        assert args[12] == slab.thickness + int(family == "E" and kind == "internal")
        expected_buffers = [
            getattr(slab.G, f"{name}_dev") for name in ("ID", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        ] + [
            getattr(slab, f"{family}{name}_dev")
            for name in ("Phi1", "Phi2", "RA", "RB", "RE", "RF")
        ]
        assert all(actual is expected for actual, expected in zip(args[13:26], expected_buffers))
        assert isinstance(args[26], np.float64)
        assert args[26] == (0.001, 0.002, 0.004)["xyz".index(direction[0])]
        assert isinstance(args[27], np.int32)
        assert args[27] == slab.pmc_mask == 0
    assert ranges[0] != ranges[1]  # E's padded normal dimension is one larger


def test_opencl_pml_caches_ranges_until_each_history_upload(make_opencl_pml, fake_opencl_upload):
    slab, _ = make_opencl_pml(order=2)
    original = (slab._electric_update_range, slab._magnetic_update_range)
    calls = _record_kernels(slab)

    class UnreadableHostShape:
        @property
        def shape(self):
            raise AssertionError("the timestep must use cached ranges, not host shapes")

    for name in HISTORIES:
        setattr(slab, name, UnreadableHostShape())
    slab.update_electric()
    slab.update_magnetic()
    assert calls[0][2]["range"] is original[0]
    assert calls[2][2]["range"] is original[1]

    shapes = ((2, 3, 5, 7), (2, 3, 7, 6), (2, 2, 7, 6), (2, 2, 5, 7))
    for name, shape in zip(HISTORIES, shapes):
        setattr(slab, name, np.zeros(shape))
    slab.set_queue(object())
    slab.htod_field_arrays()
    assert len(fake_opencl_upload) == 24
    assert all(queue is slab.queue for queue, _, _ in fake_opencl_upload[12:])
    assert slab._electric_update_range == slice(0, 126)
    assert slab._magnetic_update_range == slice(0, 84)
    assert slab._electric_update_range is not original[0]
    assert slab._magnetic_update_range is not original[1]
    calls.clear()
    slab.update_electric()
    slab.update_magnetic()
    assert calls[0][2] == {"range": slice(0, 126)}
    assert calls[2][2] == {"range": slice(0, 84)}


def test_opencl_pml_range_product_is_python_integer_without_allocation(make_opencl_pml):
    """Host arithmetic only: this does not assert support for an enormous GPU grid."""

    slab, _ = make_opencl_pml(order=2)
    side = 2**22
    shapes = (
        (2, side, side, side),
        (2, side, side, side + 1),
        (2, side // 2, side, side + 1),
        (2, side // 2, side, side),
    )
    for name, shape in zip(HISTORIES, shapes):
        setattr(slab, name, SimpleNamespace(shape=shape))
    slab.htod_field_arrays()
    expected_electric = side * side * (side + 1)
    expected_magnetic = (side // 2) * side * (side + 1)
    assert expected_magnetic > np.iinfo(np.int64).max
    assert type(slab._electric_update_range.stop) is int
    assert type(slab._magnetic_update_range.stop) is int
    assert slab._electric_update_range == slice(0, expected_electric)
    assert slab._magnetic_update_range == slice(0, expected_magnetic)


@pytest.mark.parametrize("family", ["electric", "magnetic"])
@pytest.mark.parametrize("formulation", PML.formulations)
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("direction", PML.directions)
def test_all_opencl_pml_templates_guard_each_spatial_history(
    make_opencl_pml, family, formulation, order, direction
):
    """All 48 templates map one work item to both Phi pitches and all CFS terms."""

    module = import_module(f"gprMax.cuda_opencl.knl_pml_updates_{family}_{formulation}")
    body = getattr(module, f"order{order}_{direction}")["func"].template
    compact = re.sub(r"\s+", "", body)
    for axis in "xyz":
        assert f"intn{axis}={axis}f-{axis}s;" in compact
    for index in (1, 2):
        assert f"size_tphi{index}_plane=(size_t)NY_PHI{index}*(size_t)NZ_PHI{index};" in compact
        assert f"size_tphi{index}_volume=(size_t)NX_PHI{index}*phi{index}_plane;" in compact
        assert f"size_trem{index}=(size_t)i%phi{index}_volume;" in compact
        assert f"intp{index}=(int)((size_t)i/phi{index}_volume);" in compact
        assert f"inti{index}=(int)(rem{index}/phi{index}_plane);" in compact
        assert f"size_tjk{index}=rem{index}%phi{index}_plane;" in compact
        assert f"intj{index}=(int)(jk{index}/(size_t)NZ_PHI{index});" in compact
        assert f"intk{index}=(int)(jk{index}%(size_t)NZ_PHI{index});" in compact
        components = [axis for axis in "xyz" if axis != direction[0]]
        padded = components[2 - index] if family == "electric" else components[index - 1]
        extents = {
            axis: f"n{axis}" + (f"+((PMC_MASK>>{'xyz'.index(axis)})&1)" if axis == padded else "")
            for axis in "xyz"
        }
        assert (
            f"if(p{index}==0&&i{index}<{extents['x']}&&"
            f"j{index}<{extents['y']}&&k{index}<{extents['z']})"
        ) in compact
        for axis, coordinate in zip("xyz", "ijk"):
            if axis == direction[0] and direction.endswith("minus"):
                offset = (
                    f"{coordinate}{index}" if family == "electric" else f"({coordinate}{index}+1)"
                )
                expression = f"{axis}f-{offset}"
            else:
                expression = f"{coordinate}{index}+{axis}s"
            assert f"{coordinate}{coordinate}={expression};" in compact
        for term in range(order):
            assert f"PHI{index}[IDX4D_PHI{index}({term},i{index},j{index},k{index})]=" in compact

    for transpose in (False, True):
        slab, bounds = make_opencl_pml(direction, formulation, order, transpose, kind="internal")
        if family == "electric":
            bounds = slab._electric_update_bounds()
        counts = tuple(bounds[2 * axis + 1] - bounds[2 * axis] for axis in range(3))
        expected = set(product(*(range(value) for value in counts)))
        prefix = "E" if family == "electric" else "H"
        shapes = [getattr(slab, f"{prefix}Phi{index}").shape[1:] for index in (1, 2)]
        stop = max(prod(shape) for shape in shapes)
        for shape in shapes:
            # Independently enumerate the guarded flat-to-pitched mapping.
            plane, volume = shape[1] * shape[2], prod(shape)
            visited = []
            for linear in range(stop):
                term, remainder = divmod(linear, volume)
                i, jk = divmod(remainder, plane)
                j, k = divmod(jk, shape[2])
                if term == 0 and all(index < bound for index, bound in zip((i, j, k), counts)):
                    visited.append((i, j, k))
            assert set(visited) == expected
            assert len(visited) == len(expected)  # no repeats on the smaller Phi
            assert stop // volume >= 1  # every omitted tail item has p != 0
