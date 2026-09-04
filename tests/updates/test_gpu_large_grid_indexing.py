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

"""Source-generation regressions for accelerator grids above 2^31 entries."""

from string import Template

import pytest
from jinja2 import Environment, PackageLoader

from gprMax.cuda_opencl import (
    knl_fields_updates,
    knl_magnetic_frill_source,
    knl_pml_updates_electric_HORIPML,
    knl_pml_updates_electric_MRIPML,
    knl_pml_updates_magnetic_HORIPML,
    knl_pml_updates_magnetic_MRIPML,
    knl_rational_network,
    knl_snapshots,
    knl_symmetry_boundaries,
)
from gprMax.updates.cuda_updates import CUDA_THREAD_INDEX


def _render_common(backend):
    env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))
    return env.get_template(f"knl_common_{backend}.tmpl").render(
        REAL="float",
        DRUDELORENTZ=False,
        NY_FIELDS=50_001,
        NZ_FIELDS=50_003,
        NY_RXS=50_005,
        NZ_RXS=50_007,
        NX_ID=6,
        NY_ID=50_009,
        NZ_ID=50_021,
        NX_SNAPS=6,
        NY_SNAPS=50_023,
        NZ_SNAPS=50_029,
        NX_T=6,
        NY_T=50_033,
        NZ_T=50_047,
    )


def test_cuda_global_thread_index_uses_pointer_sized_arithmetic():
    assert CUDA_THREAD_INDEX.startswith("size_t i =")
    assert "(size_t)blockIdx.x" in CUDA_THREAD_INDEX
    assert "(size_t)blockDim.x" in CUDA_THREAD_INDEX
    assert "(size_t)threadIdx.x" in CUDA_THREAD_INDEX


@pytest.mark.parametrize("backend", ["cuda", "opencl", "metal"])
@pytest.mark.parametrize(
    "macro",
    [
        "IDX3D_FIELDS",
        "IDX3D_RXS",
        "IDX4D_ID",
        "IDX4D_SNAPS",
        "IDX4D_T",
        "IDX4D_PHI1",
        "IDX4D_PHI2",
    ],
)
def test_flattened_accelerator_indices_use_pointer_sized_arithmetic(
    backend, macro
):
    source = _render_common(backend)
    definition = next(
        line for line in source.splitlines() if line.startswith(f"#define {macro}(")
    )

    assert "(size_t)" in definition


@pytest.mark.parametrize(
    "kernel",
    [
        knl_fields_updates.update_electric,
        knl_fields_updates.update_magnetic,
        knl_fields_updates.update_electric_dispersive_A,
        knl_fields_updates.update_electric_dispersive_B,
        knl_symmetry_boundaries.update_electric_pmc,
        knl_symmetry_boundaries.update_electric_pmc_dispersive_b,
    ],
)
def test_field_coordinate_products_use_pointer_sized_arithmetic(kernel):
    source = kernel["func"].template

    assert "size_t field_plane" in source
    assert "size_t ID_plane" in source
    assert "size_t ID_offset" in source


@pytest.mark.parametrize(
    "kernel",
    [
        knl_fields_updates.update_electric_dispersive_A,
        knl_fields_updates.update_electric_dispersive_B,
    ],
)
def test_dispersive_coordinate_products_use_pointer_sized_arithmetic(kernel):
    source = kernel["func"].template

    assert "size_t T_plane" in source
    assert "size_t T_offset" in source


def test_sparse_source_field_offsets_remain_pointer_sized():
    rational = knl_rational_network.update_rational_network["func"].template
    frill = knl_magnetic_frill_source.update_magnetic_frill_source["func"].template

    assert rational.count("size_t field_index = IDX3D_FIELDS") == 1
    assert frill.count("size_t field_index = IDX3D_FIELDS") == 2
    assert "int field_index = IDX3D_FIELDS" not in rational + frill


def test_dispersive_symmetry_state_offsets_remain_pointer_sized():
    phase_a = knl_symmetry_boundaries._DISPERSION_SNIPPET.template
    phase_b = knl_symmetry_boundaries.update_electric_pmc_dispersive_b["func"].template

    assert phase_a.count("size_t state = IDX4D_T") == 1
    assert phase_b.count("size_t state = IDX4D_T") == 3
    assert "int state = IDX4D_T" not in phase_a + phase_b


def test_snapshot_coordinate_products_use_pointer_sized_arithmetic():
    source = knl_snapshots.store_snapshot["func"].template

    assert "size_t snapshot_plane" in source
    assert "size_t snapshot_volume" in source
    assert "size_t rem_snaps" in source
    assert "$NX_SNAPS * $NY_SNAPS * $NZ_SNAPS" not in source


@pytest.mark.parametrize(
    "module",
    [
        knl_pml_updates_electric_HORIPML,
        knl_pml_updates_electric_MRIPML,
        knl_pml_updates_magnetic_HORIPML,
        knl_pml_updates_magnetic_MRIPML,
    ],
)
def test_pml_coordinate_products_use_pointer_sized_arithmetic(module):
    sources = [
        value["func"].template
        for value in vars(module).values()
        if isinstance(value, dict) and isinstance(value.get("func"), Template)
    ]

    assert len(sources) == 12
    for source in sources:
        assert "size_t phi1_plane" in source
        assert "size_t phi1_volume" in source
        assert "size_t phi2_plane" in source
        assert "size_t phi2_volume" in source
        assert "NX_PHI1 * NY_PHI1 * NZ_PHI1" not in source
        assert "NX_PHI2 * NY_PHI2 * NZ_PHI2" not in source
