# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom

# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""Device-side precursor node operations for the HSG subgrid.

Ports the parts of subgrids/precursor_nodes.py that run every timestep.
Three kernels cover all of it:

    gather_taps      slice extraction, the 3-tap stabilising filter and the
                     transverse blend, collapsed into one weighted sum
    bilinear_interp  main grid - subgrid spatial interpolation, replacing
                     scipy.interpolate.RectBivariateSpline
    time_blend       linear weighting between the previous and current main
                     grid timesteps

gather_taps subsumes both PrecursorNodes and PrecursorNodesFiltered: the
filtered class applies 0.25/0.5/0.25 across three adjacent nodes and the
unfiltered one takes a single node, which is the same weighted sum with
different constants. Setting unused weights to zero covers the plain case.
"""

from string import Template

# GATHER + WEIGHTED SUM
#
# Reads up to four parallel slices out of a 3D field array and combines them
# with scalar weights. The slices are described by their flat base offsets and
# two strides, so one kernel serves every face orientation without branching:
#
#   normal x - stride_a = NY*NZ apart between slices, in-plane (NZ, 1)
#   normal y - in-plane (NY*NZ, 1)
#   normal z - in-plane (NY*NZ, NZ)
#
# Weights encode the operation:
#   filtered H : 0.25*c1, 0.5*c1 + 0.25*c2, 0.25*c1 + 0.5*c2, 0.25*c2
#   filtered E : 0.25, 0.5, 0.25, 0
#   plain H    : c1, c2, 0, 0
#   plain E    : 1, 0, 0, 0

gather_taps = {
    "args_cuda": Template(
        """
        __global__ void gather_taps(
            const int n_a,
            const int n_b,
            const long base0,
            const long base1,
            const long base2,
            const long base3,
            const long stride_a,
            const long stride_b,
            const $REAL w0,
            const $REAL w1,
            const $REAL w2,
            const $REAL w3,
            const $REAL* __restrict__ src,
            $REAL* __restrict__ dst
        )
        """
    ),
    "args_opencl": Template(
        """
            const int n_a,
            const int n_b,
            const long base0,
            const long base1,
            const long base2,
            const long base3,
            const long stride_a,
            const long stride_b,
            const $REAL w0,
            const $REAL w1,
            const $REAL w2,
            const $REAL w3,
            __global const $REAL* restrict src,
            __global $REAL* restrict dst
        """
    ),
    "func": Template(
        """
    // One thread per output element of a 2D face slice.
    //
    // Thread mapping: i in [0, n_a*n_b) -> (a, b) with
    //     a = i / n_b,  b = i % n_b
    //
    // dst is a compact [n_a, n_b] array; src is the full 3D field, addressed
    // through a per-slice base offset plus the two in-plane strides.

    $CUDA_IDX

    if (i >= n_a * n_b) return;

    int a = i / n_b;
    int b = i % n_b;
    long off = (long)a * stride_a + (long)b * stride_b;

    dst[i] = w0 * src[base0 + off]
           + w1 * src[base1 + off]
           + w2 * src[base2 + off]
           + w3 * src[base3 + off];
    """
    ),
}


# BILINEAR INTERPOLATION, MAIN GRID - SUBGRID
#
# Replaces interpolate_to_sub_grid(). RectBivariateSpline with kx=ky=1 is
# exactly piecewise bilinear on a rectilinear grid, and FITPACK clamps to the
# boundary value outside the source range rather than extrapolating. Both
# behaviours are reproduced by clamping the index to [0, n-2] AND the weight
# to [0, 1] when the index/weight pairs are built on the host:
#
#   p = clip(floor((tgt - src[0]) / h), 0, n - 2)
#   w = clip((tgt - src[p]) / h, 0.0, 1.0)
#
# create_interpolated_coords() runs once at setup, so p/wx/q/wz are computed
# and uploaded once and reused for the whole run.

bilinear_interp = {
    "args_cuda": Template(
        """
        __global__ void bilinear_interp(
            const int N_a,
            const int N_b,
            const int n_b,
            const int* __restrict__ p,
            const $REAL* __restrict__ wx,
            const int* __restrict__ q,
            const $REAL* __restrict__ wz,
            const $REAL* __restrict__ src,
            $REAL* __restrict__ dst
        )
        """
    ),
    "args_opencl": Template(
        """
            const int N_a,
            const int N_b,
            const int n_b,
            __global const int* restrict p,
            __global const $REAL* restrict wx,
            __global const int* restrict q,
            __global const $REAL* restrict wz,
            __global const $REAL* restrict src,
            __global $REAL* restrict dst
        """
    ),
    "func": Template(
        """
    // One thread per output element. Four reads, three multiply-adds.
    //
    //   src : [n_a, n_b] coarse values
    //   dst : [N_a, N_b] interpolated values
    //   p, wx : per-output-row source index and fractional weight
    //   q, wz : per-output-column source index and fractional weight

    $CUDA_IDX

    if (i >= N_a * N_b) return;

    int a = i / N_b;
    int b = i % N_b;

    int P = p[a];
    int Q = q[b];
    $REAL WX = wx[a];
    $REAL WZ = wz[b];

    long r0 = (long)P * n_b + Q;
    long r1 = r0 + n_b;

    $REAL f00 = src[r0];
    $REAL f10 = src[r1];
    $REAL f01 = src[r0 + 1];
    $REAL f11 = src[r1 + 1];

    dst[i] = (1 - WX) * (1 - WZ) * f00
           + WX * (1 - WZ) * f10
           + (1 - WX) * WZ * f01
           + WX * WZ * f11;
    """
    ),
}


# TIME WEIGHTING
#
# Ports weight_pre_and_current_fields(). The subgrid runs `ratio` substeps per
# main grid step, so the incident field at substep m is a linear blend of the
# main grid values at the previous and current timesteps:
#
#   c1 = (ratio - m) / ratio,  c2 = m / ratio

time_blend = {
    "args_cuda": Template(
        """
        __global__ void time_blend(
            const int n,
            const $REAL c1,
            const $REAL c2,
            const $REAL* __restrict__ f_0,
            const $REAL* __restrict__ f_1,
            $REAL* __restrict__ out
        )
        """
    ),
    "args_opencl": Template(
        """
            const int n,
            const $REAL c1,
            const $REAL c2,
            __global const $REAL* restrict f_0,
            __global const $REAL* restrict f_1,
            __global $REAL* restrict out
        """
    ),
    "func": Template(
        """
    // Flat elementwise blend over a precursor slice.

    $CUDA_IDX

    if (i >= n) return;

    out[i] = c1 * f_0[i] + c2 * f_1[i];
    """
    ),
}
