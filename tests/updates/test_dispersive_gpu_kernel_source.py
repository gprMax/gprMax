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

"""Source-generation regressions for complex dispersive GPU updates."""

import pytest
from jinja2 import Environment, PackageLoader

from gprMax.config import SimulationConfig
from gprMax.cuda_opencl import knl_fields_updates


def _render_common(backend, real, drudelorentz):
    env = Environment(loader=PackageLoader("gprMax", "cuda_opencl"))
    return env.get_template(f"knl_common_{backend}.tmpl").render(
        REAL=real,
        DRUDELORENTZ=drudelorentz,
    )


@pytest.mark.parametrize(
    ("real", "complex_type", "prefix"),
    [
        ("float", "cfloat_t", "cfloat"),
        ("double", "cdouble_t", "cdouble"),
    ],
)
def test_opencl_complex_kernel_uses_pyopencl_struct_arithmetic(
    real, complex_type, prefix
):
    source = _render_common("opencl", real, drudelorentz=True)

    assert f"#define GPRMAX_CMUL(a, b) {prefix}_mul((a), (b))" in source
    assert f"#define GPRMAX_CRMUL(a, b) {prefix}_mulr((a), (b))" in source
    assert f"#define GPRMAX_CADD(a, b) {prefix}_add((a), (b))" in source
    assert f"#define GPRMAX_CSUB(a, b) {prefix}_sub((a), (b))" in source
    assert f"#define GPRMAX_CREAL(a) {prefix}_real(a)" in source

    args = knl_fields_updates.update_electric_dispersive_A[
        "args_opencl"
    ].substitute(REAL=real, COMPLEX=complex_type)
    assert f"__global const {complex_type}* restrict updatecoeffsdispersive" in args

    if real == "double":
        assert "#define PYOPENCL_DEFINE_CDOUBLE" in source


def test_opencl_debye_kernel_keeps_scalar_arithmetic():
    source = _render_common("opencl", "float", drudelorentz=False)

    assert "#define GPRMAX_CMUL(a, b) ((a) * (b))" in source
    assert "#define GPRMAX_CADD(a, b) ((a) + (b))" in source
    assert "#define GPRMAX_CREAL(a) (a)" in source
    assert "cfloat_mul" not in source


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_cpp_gpu_backends_extract_real_part_after_complete_product(backend):
    source = _render_common(backend, "float", drudelorentz=True)

    assert "#define GPRMAX_CMUL(a, b) ((a) * (b))" in source
    assert "#define GPRMAX_CREAL(a) ((a).real())" in source


def test_shared_gpu_kernel_uses_backend_complex_operations_in_both_phases():
    phase_a = knl_fields_updates.update_electric_dispersive_A["func"].template
    phase_b = knl_fields_updates.update_electric_dispersive_B["func"].template

    assert phase_a.count("GPRMAX_CREAL(GPRMAX_CMUL(") == 3
    assert phase_a.count("GPRMAX_CADD(") == 3
    assert phase_a.count("GPRMAX_CRMUL(") == 3
    assert phase_b.count("GPRMAX_CSUB(") == 3
    assert phase_b.count("GPRMAX_CRMUL(") == 3


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("single", "cfloat_t"), ("double", "cdouble_t")],
)
def test_opencl_complex_dtype_names_match_pyopencl_header(precision, expected):
    sim_config = SimulationConfig.__new__(SimulationConfig)
    sim_config.general = {"solver": "opencl", "precision": precision}

    sim_config._set_precision()

    assert sim_config.dtypes["C_complex"] == expected
