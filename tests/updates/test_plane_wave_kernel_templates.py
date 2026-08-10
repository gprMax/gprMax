"""Backend-neutral plane-wave and OpenCL NTFF kernel-source regressions."""

from gprMax.cuda_opencl import knl_planewave_updates, knl_tfsf_injection
from gprMax.cuda_opencl.knl_ntff import build_ntff_kernel_source


def test_tfsf_kernels_use_backend_index_substitution():
    kernels = (
        knl_tfsf_injection.STANDARD_H_KERNELS
        + knl_tfsf_injection.STANDARD_E_KERNELS
        + knl_tfsf_injection.AXIAL_H_KERNELS
        + knl_tfsf_injection.AXIAL_E_KERNELS
    )

    assert len(kernels) == 24
    for kernel in kernels:
        body = kernel["func"].template
        assert "$TFSF_IDX" in body
        assert "blockIdx.x" not in body


def test_axial_opencl_kernels_receive_material_coefficients():
    names = (
        "update_1d_magnetic_axial_source",
        "update_1d_magnetic_axial_source_pml",
        "update_1d_magnetic_axial_inject",
        "update_1d_magnetic_axial_main",
        "update_1d_magnetic_axial_main_pml_end",
        "update_1d_magnetic_axial_main_pml_start",
        "update_1d_electric_axial_source",
        "update_1d_electric_axial_source_pml",
        "update_1d_electric_axial_inject",
        "update_1d_electric_axial_main",
        "update_1d_electric_axial_main_pml_end",
        "update_1d_electric_axial_main_pml_start",
    )

    for name in names:
        arguments = getattr(knl_planewave_updates, name)["args_opencl"].template
        assert "matH" in arguments
        assert "matE" in arguments


def test_opencl_ntff_source_accepts_offset_field_views():
    opencl = build_ntff_kernel_source("opencl", "float")
    cuda = build_ntff_kernel_source("cuda", "float")

    assert "int field_offset" in opencl
    assert "field[field_offset + inside_index[patch]]" in opencl
    assert "field[0 + inside_index[patch]]" in cuda
