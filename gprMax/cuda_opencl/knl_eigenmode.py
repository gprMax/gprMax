# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Shared device kernels for eigenmode TF/SF injection and modal DFTs."""

from string import Template


def _arguments(name, fields, backend):
    qualifiers = {
        "cuda": ("__global__ void", "const $REAL* __restrict__", "$REAL*", ""),
        "opencl": ("", "__global const $REAL* restrict", "__global $REAL*", ""),
        "metal": (
            "kernel void",
            "device const $REAL*",
            "device $REAL*",
            ", uint i [[thread_position_in_grid]]",
        ),
    }
    prefix, const_real, real, suffix = qualifiers[backend]
    scalar = "device const int&" if backend == "metal" else "int"
    scalar_real = "device const $REAL&" if backend == "metal" else "$REAL"
    id_pointer = (
        "device const uint*"
        if backend == "metal"
        else (
            "const unsigned int* __restrict__"
            if backend == "cuda"
            else "__global const uint* restrict"
        )
    )
    arguments = [
        f"{scalar} NPOINTS",
        f"{scalar} normal_axis",
        f"{scalar} direction_sign",
        f"{scalar} u0",
        f"{scalar} v0",
        f"{scalar} u1",
        f"{scalar} v1",
        f"{scalar} plane_index",
        f"{scalar} basis",
        f"{scalar_real} envelope",
        f"{const_real} profile",
        f"{const_real} material_coeffs",
        f"{id_pointer} ID",
    ]
    arguments.extend(f"{real} {field}" for field in fields)
    if backend == "cuda":
        return Template(f"{prefix} {name}(" + ",\n            ".join(arguments) + ")")
    if backend == "metal":
        return Template(f"{prefix} {name}(" + ",\n            ".join(arguments) + suffix + ")")
    return Template(",\n            ".join(arguments))


update_eigenmode_magnetic = {
    "args_cuda": _arguments("update_eigenmode_magnetic", ("Hx", "Hy", "Hz"), "cuda"),
    "args_opencl": _arguments("update_eigenmode_magnetic", ("Hx", "Hy", "Hz"), "opencl"),
    "args_metal": _arguments("update_eigenmode_magnetic", ("Hx", "Hy", "Hz"), "metal"),
    "func": Template(
        """
    $CUDA_IDX
    if (i < NPOINTS) {
        int nv = v1 - v0;
        int stride_v = nv + 1;
        int local_u = i / stride_v;
        int local_v = i - local_u * stride_v;
        int nu = u1 - u0;
        int profile_plane = (nu + 1) * (nv + 1);
        int profile_offset = basis * 3 * profile_plane;
        int u = u0 + local_u;
        int v = v0 + local_v;
        int x, y, z, target, material;
        $REAL coefficient;

        if (normal_axis == 0) {
            target = plane_index - (direction_sign > 0 ? 1 : 0);
            x = target; y = u; z = v;
            if (local_u <= nu && local_v < nv) {
                material = ID[IDX4D_ID(4,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,1)];
                Hy[IDX3D_FIELDS(x,y,z)] -= direction_sign * coefficient * envelope
                    * profile[profile_offset + 2 * profile_plane + local_u * stride_v + local_v];
            }
            if (local_u < nu && local_v <= nv) {
                material = ID[IDX4D_ID(5,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,1)];
                Hz[IDX3D_FIELDS(x,y,z)] += direction_sign * coefficient * envelope
                    * profile[profile_offset + profile_plane + local_u * stride_v + local_v];
            }
        }
        else if (normal_axis == 1) {
            target = plane_index - (direction_sign > 0 ? 1 : 0);
            x = u; y = target; z = v;
            if (local_u <= nu && local_v < nv) {
                material = ID[IDX4D_ID(3,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,2)];
                Hx[IDX3D_FIELDS(x,y,z)] += direction_sign * coefficient * envelope
                    * profile[profile_offset + 2 * profile_plane + local_u * stride_v + local_v];
            }
            if (local_u < nu && local_v <= nv) {
                material = ID[IDX4D_ID(5,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,2)];
                Hz[IDX3D_FIELDS(x,y,z)] -= direction_sign * coefficient * envelope
                    * profile[profile_offset + local_u * stride_v + local_v];
            }
        }
        else {
            target = plane_index - (direction_sign > 0 ? 1 : 0);
            x = u; y = v; z = target;
            if (local_u < nu && local_v <= nv) {
                material = ID[IDX4D_ID(4,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,3)];
                Hy[IDX3D_FIELDS(x,y,z)] += direction_sign * coefficient * envelope
                    * profile[profile_offset + local_u * stride_v + local_v];
            }
            if (local_u <= nu && local_v < nv) {
                material = ID[IDX4D_ID(3,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,3)];
                Hx[IDX3D_FIELDS(x,y,z)] -= direction_sign * coefficient * envelope
                    * profile[profile_offset + profile_plane + local_u * stride_v + local_v];
            }
        }
    }
"""
    ),
}


update_eigenmode_electric = {
    "args_cuda": _arguments("update_eigenmode_electric", ("Ex", "Ey", "Ez"), "cuda"),
    "args_opencl": _arguments("update_eigenmode_electric", ("Ex", "Ey", "Ez"), "opencl"),
    "args_metal": _arguments("update_eigenmode_electric", ("Ex", "Ey", "Ez"), "metal"),
    "func": Template(
        """
    $CUDA_IDX
    if (i < NPOINTS) {
        int nv = v1 - v0;
        int stride_v = nv + 1;
        int local_u = i / stride_v;
        int local_v = i - local_u * stride_v;
        int nu = u1 - u0;
        int profile_plane = (nu + 1) * (nv + 1);
        int profile_offset = basis * 3 * profile_plane;
        int u = u0 + local_u;
        int v = v0 + local_v;
        int x, y, z, material;
        $REAL coefficient;

        if (normal_axis == 0) {
            x = plane_index; y = u; z = v;
            if (local_u <= nu && local_v < nv) {
                material = ID[IDX4D_ID(2,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,1)];
                Ez[IDX3D_FIELDS(x,y,z)] -= coefficient * envelope
                    * profile[profile_offset + profile_plane + local_u * stride_v + local_v];
            }
            if (local_u < nu && local_v <= nv) {
                material = ID[IDX4D_ID(1,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,1)];
                Ey[IDX3D_FIELDS(x,y,z)] += coefficient * envelope
                    * profile[profile_offset + 2 * profile_plane + local_u * stride_v + local_v];
            }
        }
        else if (normal_axis == 1) {
            x = u; y = plane_index; z = v;
            if (local_u <= nu && local_v < nv) {
                material = ID[IDX4D_ID(2,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,2)];
                Ez[IDX3D_FIELDS(x,y,z)] += coefficient * envelope
                    * profile[profile_offset + local_u * stride_v + local_v];
            }
            if (local_u < nu && local_v <= nv) {
                material = ID[IDX4D_ID(0,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,2)];
                Ex[IDX3D_FIELDS(x,y,z)] -= coefficient * envelope
                    * profile[profile_offset + 2 * profile_plane + local_u * stride_v + local_v];
            }
        }
        else {
            x = u; y = v; z = plane_index;
            if (local_u <= nu && local_v < nv) {
                material = ID[IDX4D_ID(1,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,3)];
                Ey[IDX3D_FIELDS(x,y,z)] -= coefficient * envelope
                    * profile[profile_offset + local_u * stride_v + local_v];
            }
            if (local_u < nu && local_v <= nv) {
                material = ID[IDX4D_ID(0,x,y,z)];
                coefficient = material_coeffs[IDX2D_MAT(material,3)];
                Ex[IDX3D_FIELDS(x,y,z)] += coefficient * envelope
                    * profile[profile_offset + profile_plane + local_u * stride_v + local_v];
            }
        }
    }
"""
    ),
}


accumulate_eigenmode_dft = {
    "args_cuda": Template(
        """
        __global__ void accumulate_eigenmode_dft(
            int NF, int NM, int normal_axis, int direction_sign,
            int magnetic_side, int u0, int v0, int u1, int v1,
            int plane_index, $REAL dt, $REAL measure, int handedness,
            $REAL* electric_phase_real, $REAL* electric_phase_imag,
            $REAL* magnetic_phase_real, $REAL* magnetic_phase_imag,
            const $REAL* __restrict__ phase_step_real,
            const $REAL* __restrict__ phase_step_imag,
            const $REAL* __restrict__ conj_eu_real,
            const $REAL* __restrict__ conj_eu_imag,
            const $REAL* __restrict__ conj_ev_real,
            const $REAL* __restrict__ conj_ev_imag,
            const $REAL* __restrict__ conj_hu_real,
            const $REAL* __restrict__ conj_hu_imag,
            const $REAL* __restrict__ conj_hv_real,
            const $REAL* __restrict__ conj_hv_imag,
            $REAL* electric_dft_real, $REAL* electric_dft_imag,
            $REAL* magnetic_dft_real, $REAL* magnetic_dft_imag,
            const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey,
            const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx,
            const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz)
        """
    ),
    "args_opencl": Template(
        """
            int NF, int NM, int normal_axis, int direction_sign,
            int magnetic_side, int u0, int v0, int u1, int v1,
            int plane_index, $REAL dt, $REAL measure, int handedness,
            __global $REAL* electric_phase_real,
            __global $REAL* electric_phase_imag,
            __global $REAL* magnetic_phase_real,
            __global $REAL* magnetic_phase_imag,
            __global const $REAL* restrict phase_step_real,
            __global const $REAL* restrict phase_step_imag,
            __global const $REAL* restrict conj_eu_real,
            __global const $REAL* restrict conj_eu_imag,
            __global const $REAL* restrict conj_ev_real,
            __global const $REAL* restrict conj_ev_imag,
            __global const $REAL* restrict conj_hu_real,
            __global const $REAL* restrict conj_hu_imag,
            __global const $REAL* restrict conj_hv_real,
            __global const $REAL* restrict conj_hv_imag,
            __global $REAL* electric_dft_real,
            __global $REAL* electric_dft_imag,
            __global $REAL* magnetic_dft_real,
            __global $REAL* magnetic_dft_imag,
            __global const $REAL* restrict Ex,
            __global const $REAL* restrict Ey,
            __global const $REAL* restrict Ez,
            __global const $REAL* restrict Hx,
            __global const $REAL* restrict Hy,
            __global const $REAL* restrict Hz
        """
    ),
    "args_metal": Template(
        """
        struct EigenmodeDFTParameters {
            int NF;
            int NM;
            int normal_axis;
            int direction_sign;
            int magnetic_side;
            int u0;
            int v0;
            int u1;
            int v1;
            int plane_index;
            $REAL dt;
            $REAL measure;
            int handedness;
        };

        kernel void accumulate_eigenmode_dft(
            constant EigenmodeDFTParameters& parameters,
            device $REAL* electric_phase_real, device $REAL* electric_phase_imag,
            device $REAL* magnetic_phase_real, device $REAL* magnetic_phase_imag,
            device const $REAL* phase_step_real, device const $REAL* phase_step_imag,
            device const $REAL* conj_eu_real, device const $REAL* conj_eu_imag,
            device const $REAL* conj_ev_real, device const $REAL* conj_ev_imag,
            device const $REAL* conj_hu_real, device const $REAL* conj_hu_imag,
            device const $REAL* conj_hv_real, device const $REAL* conj_hv_imag,
            device $REAL* electric_dft_real, device $REAL* electric_dft_imag,
            device $REAL* magnetic_dft_real, device $REAL* magnetic_dft_imag,
            device const $REAL* Ex, device const $REAL* Ey, device const $REAL* Ez,
            device const $REAL* Hx, device const $REAL* Hy, device const $REAL* Hz,
            uint i [[thread_position_in_grid]])
        """
    ),
    "func": Template(
        """
    $CUDA_IDX
    $METAL_DFT_PARAMETERS
    if (i < NF) {
        int nu = u1 - u0;
        int nv = v1 - v0;
        int hplane = direction_sign * magnetic_side > 0 ? plane_index : plane_index - 1;
        $REAL factor = ($REAL)0.5 * handedness * measure * dt;
        for (int mode = 0; mode < NM; mode++) {
            $REAL esr=0, esi=0, msr=0, msi=0;
            for (int u=0; u<nu; u++) {
                for (int v=0; v<nv; v++) {
                    $REAL eu,ev,hu,hv;
                    if(normal_axis==0){eu=($REAL)0.5*(Ey[IDX3D_FIELDS(plane_index,u0+u,v0+v)]+Ey[IDX3D_FIELDS(plane_index,u0+u,v0+v+1)]);ev=($REAL)0.5*(Ez[IDX3D_FIELDS(plane_index,u0+u,v0+v)]+Ez[IDX3D_FIELDS(plane_index,u0+u+1,v0+v)]);hu=($REAL)0.5*(Hy[IDX3D_FIELDS(hplane,u0+u,v0+v)]+Hy[IDX3D_FIELDS(hplane,u0+u+1,v0+v)]);hv=($REAL)0.5*(Hz[IDX3D_FIELDS(hplane,u0+u,v0+v)]+Hz[IDX3D_FIELDS(hplane,u0+u,v0+v+1)]);}
                    else if(normal_axis==1){eu=($REAL)0.5*(Ex[IDX3D_FIELDS(u0+u,plane_index,v0+v)]+Ex[IDX3D_FIELDS(u0+u,plane_index,v0+v+1)]);ev=($REAL)0.5*(Ez[IDX3D_FIELDS(u0+u,plane_index,v0+v)]+Ez[IDX3D_FIELDS(u0+u+1,plane_index,v0+v)]);hu=($REAL)0.5*(Hx[IDX3D_FIELDS(u0+u,hplane,v0+v)]+Hx[IDX3D_FIELDS(u0+u+1,hplane,v0+v)]);hv=($REAL)0.5*(Hz[IDX3D_FIELDS(u0+u,hplane,v0+v)]+Hz[IDX3D_FIELDS(u0+u,hplane,v0+v+1)]);}
                    else{eu=($REAL)0.5*(Ex[IDX3D_FIELDS(u0+u,v0+v,plane_index)]+Ex[IDX3D_FIELDS(u0+u,v0+v+1,plane_index)]);ev=($REAL)0.5*(Ey[IDX3D_FIELDS(u0+u,v0+v,plane_index)]+Ey[IDX3D_FIELDS(u0+u+1,v0+v,plane_index)]);hu=($REAL)0.5*(Hx[IDX3D_FIELDS(u0+u,v0+v,hplane)]+Hx[IDX3D_FIELDS(u0+u+1,v0+v,hplane)]);hv=($REAL)0.5*(Hy[IDX3D_FIELDS(u0+u,v0+v,hplane)]+Hy[IDX3D_FIELDS(u0+u,v0+v+1,hplane)]);}
                    int p=((i*NM+mode)*nu+u)*nv+v;
                    esr += eu*conj_hv_real[p]-ev*conj_hu_real[p];
                    esi += eu*conj_hv_imag[p]-ev*conj_hu_imag[p];
                    msr += direction_sign*(conj_eu_real[p]*hv-conj_ev_real[p]*hu);
                    msi += direction_sign*(conj_eu_imag[p]*hv-conj_ev_imag[p]*hu);
                }
            }
            int out=i*NM+mode;
            electric_dft_real[out]+=factor*(electric_phase_real[i]*esr-electric_phase_imag[i]*esi);
            electric_dft_imag[out]+=factor*(electric_phase_real[i]*esi+electric_phase_imag[i]*esr);
            magnetic_dft_real[out]+=factor*(magnetic_phase_real[i]*msr-magnetic_phase_imag[i]*msi);
            magnetic_dft_imag[out]+=factor*(magnetic_phase_real[i]*msi+magnetic_phase_imag[i]*msr);
        }
        $REAL er=electric_phase_real[i],ei=electric_phase_imag[i];
        electric_phase_real[i]=er*phase_step_real[i]-ei*phase_step_imag[i];
        electric_phase_imag[i]=er*phase_step_imag[i]+ei*phase_step_real[i];
        $REAL mr=magnetic_phase_real[i],mi=magnetic_phase_imag[i];
        magnetic_phase_real[i]=mr*phase_step_real[i]-mi*phase_step_imag[i];
        magnetic_phase_imag[i]=mr*phase_step_imag[i]+mi*phase_step_real[i];
    }
"""
    ),
}
