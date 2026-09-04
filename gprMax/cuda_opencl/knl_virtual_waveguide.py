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

"""Device-resident aperture coupling for auxiliary virtual waveguides."""

from string import Template


def _args(name, backend, *, electric, field_kind="H"):
    prefix = {"cuda": "__global__ void", "opencl": "", "metal": "kernel void"}[backend]
    integer = "device const int&" if backend == "metal" else "int"
    real_const = {
        "cuda": "const $REAL* __restrict__",
        "opencl": "__global const $REAL* restrict",
        "metal": "device const $REAL*",
    }[backend]
    real = {"cuda": "$REAL*", "opencl": "__global $REAL*", "metal": "device $REAL*"}[backend]
    uint_const = {
        "cuda": "const unsigned int* __restrict__",
        "opencl": "__global const uint* restrict",
        "metal": "device const uint*",
    }[backend]
    values = [
        f"{integer} NPOINTS",
        f"{integer} normal_axis",
        f"{integer} direction_sign",
        f"{integer} u0",
        f"{integer} v0",
        f"{integer} u1",
        f"{integer} v1",
        f"{integer} plane_index",
        f"{integer} aux_nx",
        f"{integer} aux_ny",
        f"{integer} aux_nz",
    ]
    if electric:
        values.extend((f"{real_const} aux_coeffs", f"{uint_const} aux_ID"))
        names = (
            "main_Ex",
            "main_Ey",
            "main_Ez",
            "main_Hx",
            "main_Hy",
            "main_Hz",
            "aux_Ex",
            "aux_Ey",
            "aux_Ez",
            "aux_Hx",
            "aux_Hy",
            "aux_Hz",
        )
    else:
        names = tuple(
            f"{owner}_{field_kind}{axis}"
            for owner in ("main", "aux")
            for axis in "xyz"
        )
    values.extend(f"{real} {value}" for value in names)
    suffix = ", uint i [[thread_position_in_grid]]" if backend == "metal" else ""
    if backend == "opencl":
        return Template(",\n            ".join(values))
    return Template(f"{prefix} {name}(" + ",\n            ".join(values) + suffix + ")")


couple_magnetic = {
    "args_cuda": _args("couple_virtual_waveguide_magnetic", "cuda", electric=False),
    "args_opencl": _args("couple_virtual_waveguide_magnetic", "opencl", electric=False),
    "args_metal": _args("couple_virtual_waveguide_magnetic", "metal", electric=False),
    "func": Template(
        """
    $CUDA_IDX
    if (i < NPOINTS) {
        int nu = u1 - u0;
        int nv = v1 - v0;
        int stride_v = nv + 1;
        int u = i / stride_v;
        int v = i - u * stride_v;
        int aperture = direction_sign < 0 ? 0 :
            (normal_axis == 0 ? aux_nx : (normal_axis == 1 ? aux_ny : aux_nz));
        int aux_index;
        size_t main_index;
        if (u < nu && v < nv) {
            if (normal_axis == 0) {
                aux_index = aperture * (aux_ny + 1) * (aux_nz + 1) + u * (aux_nz + 1) + v;
                main_index = IDX3D_FIELDS(plane_index,u0+u,v0+v);
                aux_Hx[aux_index] = main_Hx[main_index];
            }
            else if (normal_axis == 1) {
                aux_index = u * (aux_ny + 1) * (aux_nz + 1) + aperture * (aux_nz + 1) + v;
                main_index = IDX3D_FIELDS(u0+u,plane_index,v0+v);
                aux_Hy[aux_index] = main_Hy[main_index];
            }
            else {
                aux_index = u * (aux_ny + 1) * (aux_nz + 1) + v * (aux_nz + 1) + aperture;
                main_index = IDX3D_FIELDS(u0+u,v0+v,plane_index);
                aux_Hz[aux_index] = main_Hz[main_index];
            }
        }
    }
"""
    ),
}


clear_rear_magnetic = {
    "args_cuda": _args("clear_virtual_waveguide_rear_magnetic", "cuda", electric=False),
    "args_opencl": _args("clear_virtual_waveguide_rear_magnetic", "opencl", electric=False),
    "args_metal": _args("clear_virtual_waveguide_rear_magnetic", "metal", electric=False),
    "func": Template(
        """
    $CUDA_IDX
    if (i < NPOINTS) {
        int nu = u1 - u0;
        int nv = v1 - v0;
        size_t uv_plane = (size_t)(nu + 1) * (size_t)(nv + 1);
        int normal_offset = (int)((size_t)i / uv_plane);
        size_t uv = (size_t)i % uv_plane;
        int u = (int)(uv / (size_t)(nv + 1));
        int v = (int)(uv % (size_t)(nv + 1));
        int normal_cells = normal_axis == 0 ? $NX_FIELDS - 1 :
            (normal_axis == 1 ? $NY_FIELDS - 1 : $NZ_FIELDS - 1);
        int normal = (direction_sign < 0 ? plane_index : 0) + normal_offset;
        int x = normal_axis == 0 ? normal : u0 + u;
        int y = normal_axis == 1 ? normal : (normal_axis == 0 ? u0 + u : v0 + v);
        int z = normal_axis == 2 ? normal : v0 + v;
        size_t main_index = IDX3D_FIELDS(x,y,z);

        bool clear_normal = u < nu && v < nv &&
            (direction_sign > 0 || normal > plane_index);
        bool clear_tangential = direction_sign > 0 || normal < normal_cells;
        bool clear_u = clear_tangential && u <= nu && v < nv;
        bool clear_v = clear_tangential && u < nu && v <= nv;

        if (normal_axis == 0) {
            if (clear_normal) main_Hx[main_index] = 0;
            if (clear_u) main_Hy[main_index] = 0;
            if (clear_v) main_Hz[main_index] = 0;
        }
        else if (normal_axis == 1) {
            if (clear_normal) main_Hy[main_index] = 0;
            if (clear_u) main_Hx[main_index] = 0;
            if (clear_v) main_Hz[main_index] = 0;
        }
        else {
            if (clear_normal) main_Hz[main_index] = 0;
            if (clear_u) main_Hx[main_index] = 0;
            if (clear_v) main_Hy[main_index] = 0;
        }
    }
"""
    ),
}


couple_electric = {
    "args_cuda": _args("couple_virtual_waveguide_electric", "cuda", electric=True),
    "args_opencl": _args("couple_virtual_waveguide_electric", "opencl", electric=True),
    "args_metal": _args("couple_virtual_waveguide_electric", "metal", electric=True),
    "func": Template(
        """
    $CUDA_IDX
    if (i < NPOINTS) {
        int nu = u1 - u0;
        int nv = v1 - v0;
        int stride_v = nv + 1;
        int u = i / stride_v;
        int v = i - u * stride_v;
        int aperture = direction_sign < 0 ? 0 :
            (normal_axis == 0 ? aux_nx : (normal_axis == 1 ? aux_ny : aux_nz));
        int inside = direction_sign < 0 ? 0 : aperture - 1;
        int aidx = 0, material;
        size_t midx = 0;
        $REAL cross_field;

        if (normal_axis == 0) {
            if (u < nu && v > 0 && v < nv) {
                aidx = aperture*(aux_ny+1)*(aux_nz+1)+u*(aux_nz+1)+v;
                material = aux_ID[1*(aux_nx+1)*(aux_ny+1)*(aux_nz+1)+aidx];
                cross_field = direction_sign < 0
                    ? aux_Hz[u*(aux_nz+1)+v] - main_Hz[IDX3D_FIELDS(plane_index-1,u0+u,v0+v)]
                    : main_Hz[IDX3D_FIELDS(plane_index,u0+u,v0+v)] - aux_Hz[inside*(aux_ny+1)*(aux_nz+1)+u*(aux_nz+1)+v];
                aux_Ey[aidx] = aux_coeffs[IDX2D_MAT(material,0)]*aux_Ey[aidx]
                    + aux_coeffs[IDX2D_MAT(material,3)]*(aux_Hx[aidx]-aux_Hx[aidx-1])
                    - aux_coeffs[IDX2D_MAT(material,1)]*cross_field;
            }
            if (u > 0 && u < nu && v < nv) {
                aidx = aperture*(aux_ny+1)*(aux_nz+1)+u*(aux_nz+1)+v;
                material = aux_ID[2*(aux_nx+1)*(aux_ny+1)*(aux_nz+1)+aidx];
                cross_field = direction_sign < 0
                    ? aux_Hy[u*(aux_nz+1)+v] - main_Hy[IDX3D_FIELDS(plane_index-1,u0+u,v0+v)]
                    : main_Hy[IDX3D_FIELDS(plane_index,u0+u,v0+v)] - aux_Hy[inside*(aux_ny+1)*(aux_nz+1)+u*(aux_nz+1)+v];
                aux_Ez[aidx] = aux_coeffs[IDX2D_MAT(material,0)]*aux_Ez[aidx]
                    + aux_coeffs[IDX2D_MAT(material,1)]*cross_field
                    - aux_coeffs[IDX2D_MAT(material,2)]*(aux_Hx[aidx]-aux_Hx[aidx-(aux_nz+1)]);
            }
            if (u < nu && v <= nv) {
                aidx=aperture*(aux_ny+1)*(aux_nz+1)+u*(aux_nz+1)+v;
                main_Ey[IDX3D_FIELDS(plane_index,u0+u,v0+v)]=aux_Ey[aidx];
            }
            if (u <= nu && v < nv) {
                aidx=aperture*(aux_ny+1)*(aux_nz+1)+u*(aux_nz+1)+v;
                main_Ez[IDX3D_FIELDS(plane_index,u0+u,v0+v)]=aux_Ez[aidx];
            }
            aidx=(direction_sign<0?0:inside)*(aux_ny+1)*(aux_nz+1)+u*(aux_nz+1)+v;
            midx=IDX3D_FIELDS(direction_sign<0?plane_index:plane_index-1,u0+u,v0+v);
            main_Ex[midx]=aux_Ex[aidx];
        }
        else if (normal_axis == 1) {
            if (u < nu && v > 0 && v < nv) {
                aidx=u*(aux_ny+1)*(aux_nz+1)+aperture*(aux_nz+1)+v;
                material=aux_ID[aidx];
                cross_field=direction_sign<0
                    ? aux_Hz[u*(aux_ny+1)*(aux_nz+1)+v]-main_Hz[IDX3D_FIELDS(u0+u,plane_index-1,v0+v)]
                    : main_Hz[IDX3D_FIELDS(u0+u,plane_index,v0+v)]-aux_Hz[u*(aux_ny+1)*(aux_nz+1)+inside*(aux_nz+1)+v];
                aux_Ex[aidx]=aux_coeffs[IDX2D_MAT(material,0)]*aux_Ex[aidx]
                    +aux_coeffs[IDX2D_MAT(material,2)]*cross_field
                    -aux_coeffs[IDX2D_MAT(material,3)]*(aux_Hy[aidx]-aux_Hy[aidx-1]);
            }
            if (u > 0 && u < nu && v < nv) {
                aidx=u*(aux_ny+1)*(aux_nz+1)+aperture*(aux_nz+1)+v;
                material=aux_ID[2*(aux_nx+1)*(aux_ny+1)*(aux_nz+1)+aidx];
                cross_field=direction_sign<0
                    ? aux_Hx[u*(aux_ny+1)*(aux_nz+1)+v]-main_Hx[IDX3D_FIELDS(u0+u,plane_index-1,v0+v)]
                    : main_Hx[IDX3D_FIELDS(u0+u,plane_index,v0+v)]-aux_Hx[u*(aux_ny+1)*(aux_nz+1)+inside*(aux_nz+1)+v];
                aux_Ez[aidx]=aux_coeffs[IDX2D_MAT(material,0)]*aux_Ez[aidx]
                    +aux_coeffs[IDX2D_MAT(material,1)]*(aux_Hy[aidx]-aux_Hy[aidx-(aux_ny+1)*(aux_nz+1)])
                    -aux_coeffs[IDX2D_MAT(material,2)]*cross_field;
            }
            if(u<nu&&v<=nv){aidx=u*(aux_ny+1)*(aux_nz+1)+aperture*(aux_nz+1)+v;main_Ex[IDX3D_FIELDS(u0+u,plane_index,v0+v)]=aux_Ex[aidx];}
            if(u<=nu&&v<nv){aidx=u*(aux_ny+1)*(aux_nz+1)+aperture*(aux_nz+1)+v;main_Ez[IDX3D_FIELDS(u0+u,plane_index,v0+v)]=aux_Ez[aidx];}
            aidx=u*(aux_ny+1)*(aux_nz+1)+(direction_sign<0?0:inside)*(aux_nz+1)+v;
            main_Ey[IDX3D_FIELDS(u0+u,direction_sign<0?plane_index:plane_index-1,v0+v)]=aux_Ey[aidx];
        }
        else {
            if(u<nu&&v>0&&v<nv){
                aidx=u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)+aperture;material=aux_ID[aidx];
                cross_field=direction_sign<0?aux_Hy[u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)]-main_Hy[IDX3D_FIELDS(u0+u,v0+v,plane_index-1)]:main_Hy[IDX3D_FIELDS(u0+u,v0+v,plane_index)]-aux_Hy[u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)+inside];
                aux_Ex[aidx]=aux_coeffs[IDX2D_MAT(material,0)]*aux_Ex[aidx]+aux_coeffs[IDX2D_MAT(material,2)]*(aux_Hz[aidx]-aux_Hz[aidx-(aux_nz+1)])-aux_coeffs[IDX2D_MAT(material,3)]*cross_field;
            }
            if(u>0&&u<nu&&v<nv){
                aidx=u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)+aperture;material=aux_ID[(aux_nx+1)*(aux_ny+1)*(aux_nz+1)+aidx];
                cross_field=direction_sign<0?aux_Hx[u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)]-main_Hx[IDX3D_FIELDS(u0+u,v0+v,plane_index-1)]:main_Hx[IDX3D_FIELDS(u0+u,v0+v,plane_index)]-aux_Hx[u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)+inside];
                aux_Ey[aidx]=aux_coeffs[IDX2D_MAT(material,0)]*aux_Ey[aidx]+aux_coeffs[IDX2D_MAT(material,3)]*cross_field-aux_coeffs[IDX2D_MAT(material,1)]*(aux_Hz[aidx]-aux_Hz[aidx-(aux_ny+1)*(aux_nz+1)]);
            }
            if(u<nu&&v<=nv){aidx=u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)+aperture;main_Ex[IDX3D_FIELDS(u0+u,v0+v,plane_index)]=aux_Ex[aidx];}
            if(u<=nu&&v<nv){aidx=u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)+aperture;main_Ey[IDX3D_FIELDS(u0+u,v0+v,plane_index)]=aux_Ey[aidx];}
            aidx=u*(aux_ny+1)*(aux_nz+1)+v*(aux_nz+1)+(direction_sign<0?0:inside);
            main_Ez[IDX3D_FIELDS(u0+u,v0+v,direction_sign<0?plane_index:plane_index-1)]=aux_Ez[aidx];
        }
    }
"""
    ),
}


clear_rear_electric = {
    "args_cuda": _args(
        "clear_virtual_waveguide_rear_electric",
        "cuda",
        electric=False,
        field_kind="E",
    ),
    "args_opencl": _args(
        "clear_virtual_waveguide_rear_electric",
        "opencl",
        electric=False,
        field_kind="E",
    ),
    "args_metal": _args(
        "clear_virtual_waveguide_rear_electric",
        "metal",
        electric=False,
        field_kind="E",
    ),
    "func": Template(
        """
    $CUDA_IDX
    if (i < NPOINTS) {
        int nu = u1 - u0;
        int nv = v1 - v0;
        size_t uv_plane = (size_t)(nu + 1) * (size_t)(nv + 1);
        int normal_offset = (int)((size_t)i / uv_plane);
        size_t uv = (size_t)i % uv_plane;
        int u = (int)(uv / (size_t)(nv + 1));
        int v = (int)(uv % (size_t)(nv + 1));
        int normal = (direction_sign < 0 ? plane_index + 1 : 0) + normal_offset;
        int x = normal_axis == 0 ? normal : u0 + u;
        int y = normal_axis == 1 ? normal : (normal_axis == 0 ? u0 + u : v0 + v);
        int z = normal_axis == 2 ? normal : v0 + v;
        size_t main_index = IDX3D_FIELDS(x,y,z);

        bool clear_normal = u <= nu && v <= nv &&
            (direction_sign < 0 || normal < plane_index - 1);
        bool clear_u = u < nu && v <= nv;
        bool clear_v = u <= nu && v < nv;

        if (normal_axis == 0) {
            if (clear_normal) main_Ex[main_index] = 0;
            if (clear_u) main_Ey[main_index] = 0;
            if (clear_v) main_Ez[main_index] = 0;
        }
        else if (normal_axis == 1) {
            if (clear_normal) main_Ey[main_index] = 0;
            if (clear_u) main_Ex[main_index] = 0;
            if (clear_v) main_Ez[main_index] = 0;
        }
        else {
            if (clear_normal) main_Ez[main_index] = 0;
            if (clear_u) main_Ex[main_index] = 0;
            if (clear_v) main_Ey[main_index] = 0;
        }
    }
"""
    ),
}
