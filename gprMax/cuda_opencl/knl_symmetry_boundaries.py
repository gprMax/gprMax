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

"""CUDA, OpenCL, and Metal PMC ghost-image boundary kernels."""

from string import Template


update_electric_pmc = {
    "args_cuda": Template(
        """
__global__ void update_electric_pmc(
    int NX, int NY, int NZ,
    int PMC_X0, int PMC_XMAX,
    int PMC_Y0, int PMC_YMAX,
    int PMC_Z0, int PMC_ZMAX,
    const unsigned int* __restrict__ ID,
    $REAL* Ex, $REAL* Ey, $REAL* Ez,
    const $REAL* __restrict__ Hx,
    const $REAL* __restrict__ Hy,
    const $REAL* __restrict__ Hz)
"""
    ),
    "args_opencl": Template(
        """
    int NX, int NY, int NZ,
    int PMC_X0, int PMC_XMAX,
    int PMC_Y0, int PMC_YMAX,
    int PMC_Z0, int PMC_ZMAX,
    __global const unsigned int* restrict ID,
    __global $REAL* Ex, __global $REAL* Ey, __global $REAL* Ez,
    __global const $REAL* restrict Hx,
    __global const $REAL* restrict Hy,
    __global const $REAL* restrict Hz
"""
    ),
    "args_metal": Template(
        """
kernel void update_electric_pmc(
    device const int& NX, device const int& NY, device const int& NZ,
    device const int& PMC_X0, device const int& PMC_XMAX,
    device const int& PMC_Y0, device const int& PMC_YMAX,
    device const int& PMC_Z0, device const int& PMC_ZMAX,
    device const uint* ID,
    device $REAL* Ex, device $REAL* Ey, device $REAL* Ez,
    device const $REAL* Hx,
    device const $REAL* Hy,
    device const $REAL* Hz,
    uint i [[thread_position_in_grid]])
"""
    ),
    "func": Template(
        r"""
    // One work-item owns one Yee array index. A tangential E component on
    // one or two PMC planes is updated once, so edges receive one self term
    // plus the doubled ghost contribution from each adjoining PMC plane.
    $CUDA_IDX

    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;

    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) %
        ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) %
        ($NY_ID * $NZ_ID)) % $NZ_ID;

    int ex_on_pmc = (y == 0 && PMC_Y0) || (y == NY && PMC_YMAX)
        || (z == 0 && PMC_Z0) || (z == NZ && PMC_ZMAX);
    if (x >= 0 && x < NX && y >= 0 && y <= NY && z >= 0 && z <= NZ
            && ex_on_pmc) {
        int material = ID[IDX4D_ID(0,x_ID,y_ID,z_ID)];
        $DISP_EX
        $REAL dHz_dy = ($REAL)0;
        $REAL dHy_dz = ($REAL)0;
        if (y == 0) {
            if (PMC_Y0) dHz_dy = ($REAL)2 * Hz[IDX3D_FIELDS(x,0,z)];
        } else if (y == NY) {
            if (PMC_YMAX) dHz_dy = -($REAL)2 * Hz[IDX3D_FIELDS(x,NY-1,z)];
        } else {
            dHz_dy = Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x,y-1,z)];
        }
        if (z == 0) {
            if (PMC_Z0) dHy_dz = ($REAL)2 * Hy[IDX3D_FIELDS(x,y,0)];
        } else if (z == NZ) {
            if (PMC_ZMAX) dHy_dz = -($REAL)2 * Hy[IDX3D_FIELDS(x,y,NZ-1)];
        } else {
            dHy_dz = Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x,y,z-1)];
        }
        Ex[IDX3D_FIELDS(x,y,z)] =
            updatecoeffsE[IDX2D_MAT(material,0)] * Ex[IDX3D_FIELDS(x,y,z)]
            + updatecoeffsE[IDX2D_MAT(material,2)] * dHz_dy
            - updatecoeffsE[IDX2D_MAT(material,3)] * dHy_dz $PHI_EX;
    }

    int ey_on_pmc = (x == 0 && PMC_X0) || (x == NX && PMC_XMAX)
        || (z == 0 && PMC_Z0) || (z == NZ && PMC_ZMAX);
    if (x >= 0 && x <= NX && y >= 0 && y < NY && z >= 0 && z <= NZ
            && ey_on_pmc) {
        int material = ID[IDX4D_ID(1,x_ID,y_ID,z_ID)];
        $DISP_EY
        $REAL dHx_dz = ($REAL)0;
        $REAL dHz_dx = ($REAL)0;
        if (z == 0) {
            if (PMC_Z0) dHx_dz = ($REAL)2 * Hx[IDX3D_FIELDS(x,y,0)];
        } else if (z == NZ) {
            if (PMC_ZMAX) dHx_dz = -($REAL)2 * Hx[IDX3D_FIELDS(x,y,NZ-1)];
        } else {
            dHx_dz = Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y,z-1)];
        }
        if (x == 0) {
            if (PMC_X0) dHz_dx = ($REAL)2 * Hz[IDX3D_FIELDS(0,y,z)];
        } else if (x == NX) {
            if (PMC_XMAX) dHz_dx = -($REAL)2 * Hz[IDX3D_FIELDS(NX-1,y,z)];
        } else {
            dHz_dx = Hz[IDX3D_FIELDS(x,y,z)] - Hz[IDX3D_FIELDS(x-1,y,z)];
        }
        Ey[IDX3D_FIELDS(x,y,z)] =
            updatecoeffsE[IDX2D_MAT(material,0)] * Ey[IDX3D_FIELDS(x,y,z)]
            + updatecoeffsE[IDX2D_MAT(material,3)] * dHx_dz
            - updatecoeffsE[IDX2D_MAT(material,1)] * dHz_dx $PHI_EY;
    }

    int ez_on_pmc = (x == 0 && PMC_X0) || (x == NX && PMC_XMAX)
        || (y == 0 && PMC_Y0) || (y == NY && PMC_YMAX);
    if (x >= 0 && x <= NX && y >= 0 && y <= NY && z >= 0 && z < NZ
            && ez_on_pmc) {
        int material = ID[IDX4D_ID(2,x_ID,y_ID,z_ID)];
        $DISP_EZ
        $REAL dHy_dx = ($REAL)0;
        $REAL dHx_dy = ($REAL)0;
        if (x == 0) {
            if (PMC_X0) dHy_dx = ($REAL)2 * Hy[IDX3D_FIELDS(0,y,z)];
        } else if (x == NX) {
            if (PMC_XMAX) dHy_dx = -($REAL)2 * Hy[IDX3D_FIELDS(NX-1,y,z)];
        } else {
            dHy_dx = Hy[IDX3D_FIELDS(x,y,z)] - Hy[IDX3D_FIELDS(x-1,y,z)];
        }
        if (y == 0) {
            if (PMC_Y0) dHx_dy = ($REAL)2 * Hx[IDX3D_FIELDS(x,0,z)];
        } else if (y == NY) {
            if (PMC_YMAX) dHx_dy = -($REAL)2 * Hx[IDX3D_FIELDS(x,NY-1,z)];
        } else {
            dHx_dy = Hx[IDX3D_FIELDS(x,y,z)] - Hx[IDX3D_FIELDS(x,y-1,z)];
        }
        Ez[IDX3D_FIELDS(x,y,z)] =
            updatecoeffsE[IDX2D_MAT(material,0)] * Ez[IDX3D_FIELDS(x,y,z)]
            + updatecoeffsE[IDX2D_MAT(material,1)] * dHy_dx
            - updatecoeffsE[IDX2D_MAT(material,2)] * dHx_dy $PHI_EZ;
    }
"""
    ),
}


update_electric_pmc_dispersive = {
    "args_cuda": Template(
        """
__global__ void update_electric_pmc_dispersive(
    int NX, int NY, int NZ, int MAXPOLES,
    int PMC_X0, int PMC_XMAX,
    int PMC_Y0, int PMC_YMAX,
    int PMC_Z0, int PMC_ZMAX,
    const $COMPLEX* __restrict__ updatecoeffsdispersive,
    $COMPLEX* __restrict__ Tx,
    $COMPLEX* __restrict__ Ty,
    $COMPLEX* __restrict__ Tz,
    const unsigned int* __restrict__ ID,
    $REAL* Ex, $REAL* Ey, $REAL* Ez,
    const $REAL* __restrict__ Hx,
    const $REAL* __restrict__ Hy,
    const $REAL* __restrict__ Hz)
"""
    ),
    "args_opencl": Template(
        """
    int NX, int NY, int NZ, int MAXPOLES,
    int PMC_X0, int PMC_XMAX,
    int PMC_Y0, int PMC_YMAX,
    int PMC_Z0, int PMC_ZMAX,
    __global const unsigned int* restrict ID,
    __global $REAL* Ex, __global $REAL* Ey, __global $REAL* Ez,
    __global const $REAL* restrict Hx,
    __global const $REAL* restrict Hy,
    __global const $REAL* restrict Hz,
    __global const $COMPLEX* restrict updatecoeffsdispersive,
    __global $COMPLEX* restrict Tx,
    __global $COMPLEX* restrict Ty,
    __global $COMPLEX* restrict Tz
"""
    ),
    "args_metal": Template(
        """
kernel void update_electric_pmc_dispersive(
    device const int& NX, device const int& NY, device const int& NZ,
    device const int& MAXPOLES,
    device const int& PMC_X0, device const int& PMC_XMAX,
    device const int& PMC_Y0, device const int& PMC_YMAX,
    device const int& PMC_Z0, device const int& PMC_ZMAX,
    device const $COMPLEX* updatecoeffsdispersive,
    device $COMPLEX* Tx, device $COMPLEX* Ty, device $COMPLEX* Tz,
    device const uint* ID,
    device $REAL* Ex, device $REAL* Ey, device $REAL* Ez,
    device const $REAL* Hx, device const $REAL* Hy, device const $REAL* Hz,
    uint i [[thread_position_in_grid]])
"""
    ),
    "func": update_electric_pmc["func"],
}


_DISPERSION_SNIPPET = Template(
    """
        $REAL phi = ($REAL)0;
        for (int pole = 0; pole < MAXPOLES; pole++) {
            int state = IDX4D_T(pole,x,y,z);
            phi += GPRMAX_CREAL(GPRMAX_CMUL(
                updatecoeffsdispersive[IDX2D_MATDISP(material,pole*3)],
                $T[state]));
            $T[state] = GPRMAX_CADD(
                GPRMAX_CMUL(
                    updatecoeffsdispersive[IDX2D_MATDISP(material,1+pole*3)],
                    $T[state]),
                GPRMAX_CRMUL(
                    updatecoeffsdispersive[IDX2D_MATDISP(material,2+pole*3)],
                    $E[IDX3D_FIELDS(x,y,z)]));
        }
"""
)


update_electric_pmc_dispersive_b = {
    "args_cuda": Template(
        """
__global__ void update_electric_pmc_dispersive_b(
    int NX, int NY, int NZ, int MAXPOLES,
    int PMC_X0, int PMC_XMAX,
    int PMC_Y0, int PMC_YMAX,
    int PMC_Z0, int PMC_ZMAX,
    const $COMPLEX* __restrict__ updatecoeffsdispersive,
    $COMPLEX* __restrict__ Tx,
    $COMPLEX* __restrict__ Ty,
    $COMPLEX* __restrict__ Tz,
    const unsigned int* __restrict__ ID,
    const $REAL* __restrict__ Ex,
    const $REAL* __restrict__ Ey,
    const $REAL* __restrict__ Ez)
"""
    ),
    "args_opencl": Template(
        """
    int NX, int NY, int NZ, int MAXPOLES,
    int PMC_X0, int PMC_XMAX,
    int PMC_Y0, int PMC_YMAX,
    int PMC_Z0, int PMC_ZMAX,
    __global const unsigned int* restrict ID,
    __global const $REAL* restrict Ex,
    __global const $REAL* restrict Ey,
    __global const $REAL* restrict Ez,
    __global const $COMPLEX* restrict updatecoeffsdispersive,
    __global $COMPLEX* restrict Tx,
    __global $COMPLEX* restrict Ty,
    __global $COMPLEX* restrict Tz
"""
    ),
    "args_metal": Template(
        """
kernel void update_electric_pmc_dispersive_b(
    device const int& NX, device const int& NY, device const int& NZ,
    device const int& MAXPOLES,
    device const int& PMC_X0, device const int& PMC_XMAX,
    device const int& PMC_Y0, device const int& PMC_YMAX,
    device const int& PMC_Z0, device const int& PMC_ZMAX,
    device const $COMPLEX* updatecoeffsdispersive,
    device $COMPLEX* Tx, device $COMPLEX* Ty, device $COMPLEX* Tz,
    device const uint* ID,
    device const $REAL* Ex, device const $REAL* Ey, device const $REAL* Ez,
    uint i [[thread_position_in_grid]])
"""
    ),
    "func": Template(
        r"""
    $CUDA_IDX
    int x = i / ($NY_FIELDS * $NZ_FIELDS);
    int y = (i % ($NY_FIELDS * $NZ_FIELDS)) / $NZ_FIELDS;
    int z = (i % ($NY_FIELDS * $NZ_FIELDS)) % $NZ_FIELDS;
    int x_ID = (i % ($NX_ID * $NY_ID * $NZ_ID)) / ($NY_ID * $NZ_ID);
    int y_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) %
        ($NY_ID * $NZ_ID)) / $NZ_ID;
    int z_ID = ((i % ($NX_ID * $NY_ID * $NZ_ID)) %
        ($NY_ID * $NZ_ID)) % $NZ_ID;

    int ex_on_pmc = (y == 0 && PMC_Y0) || (y == NY && PMC_YMAX)
        || (z == 0 && PMC_Z0) || (z == NZ && PMC_ZMAX);
    if (x >= 0 && x < NX && y >= 0 && y <= NY && z >= 0 && z <= NZ
            && ex_on_pmc) {
        int material = ID[IDX4D_ID(0,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            int state = IDX4D_T(pole,x,y,z);
            Tx[state] = GPRMAX_CSUB(Tx[state], GPRMAX_CRMUL(
                updatecoeffsdispersive[IDX2D_MATDISP(material,2+pole*3)],
                Ex[IDX3D_FIELDS(x,y,z)]));
        }
    }
    int ey_on_pmc = (x == 0 && PMC_X0) || (x == NX && PMC_XMAX)
        || (z == 0 && PMC_Z0) || (z == NZ && PMC_ZMAX);
    if (x >= 0 && x <= NX && y >= 0 && y < NY && z >= 0 && z <= NZ
            && ey_on_pmc) {
        int material = ID[IDX4D_ID(1,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            int state = IDX4D_T(pole,x,y,z);
            Ty[state] = GPRMAX_CSUB(Ty[state], GPRMAX_CRMUL(
                updatecoeffsdispersive[IDX2D_MATDISP(material,2+pole*3)],
                Ey[IDX3D_FIELDS(x,y,z)]));
        }
    }
    int ez_on_pmc = (x == 0 && PMC_X0) || (x == NX && PMC_XMAX)
        || (y == 0 && PMC_Y0) || (y == NY && PMC_YMAX);
    if (x >= 0 && x <= NX && y >= 0 && y <= NY && z >= 0 && z < NZ
            && ez_on_pmc) {
        int material = ID[IDX4D_ID(2,x_ID,y_ID,z_ID)];
        for (int pole = 0; pole < MAXPOLES; pole++) {
            int state = IDX4D_T(pole,x,y,z);
            Tz[state] = GPRMAX_CSUB(Tz[state], GPRMAX_CRMUL(
                updatecoeffsdispersive[IDX2D_MATDISP(material,2+pole*3)],
                Ez[IDX3D_FIELDS(x,y,z)]));
        }
    }
"""
    ),
}


def nondispersive_substitutions():
    """Placeholders used by the shared PMC body without ADE state."""

    return {
        "DISP_EX": "",
        "DISP_EY": "",
        "DISP_EZ": "",
        "PHI_EX": "",
        "PHI_EY": "",
        "PHI_EZ": "",
    }


def dispersive_substitutions(real):
    """Placeholders used by the shared PMC body with ADE state."""

    return {
        "DISP_EX": _DISPERSION_SNIPPET.substitute(REAL=real, T="Tx", E="Ex"),
        "DISP_EY": _DISPERSION_SNIPPET.substitute(REAL=real, T="Ty", E="Ey"),
        "DISP_EZ": _DISPERSION_SNIPPET.substitute(REAL=real, T="Tz", E="Ez"),
        "PHI_EX": "- updatecoeffsE[IDX2D_MAT(material,4)] * phi",
        "PHI_EY": "- updatecoeffsE[IDX2D_MAT(material,4)] * phi",
        "PHI_EZ": "- updatecoeffsE[IDX2D_MAT(material,4)] * phi",
    }
