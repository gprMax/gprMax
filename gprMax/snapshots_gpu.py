# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
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

from string import Template

kernel_template_store_snapshot = Template("""

// Macros for converting subscripts to linear index:
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
#define INDEX4D_SNAPS(p, i, j, k) (p)*($NX_SNAPS)*($NY_SNAPS)*($NZ_SNAPS)+(i)*($NY_SNAPS)*($NZ_SNAPS)+(j)*($NZ_SNAPS)+(k)

////////////////////
// Store snapshot //
////////////////////

__global__ void store_snapshot(int p, int xs, int xf, int ys, int yf, int zs, int zf, int dx, int dy, int dz,
                                const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey,
                                const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx,
                                const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz,
                                $REAL *snapEx, $REAL *snapEy, $REAL *snapEz,
                                $REAL *snapHx, $REAL *snapHy, $REAL *snapHz) {

    //  This function stores field values for a snapshot.
    //
    //  Args:
    //      p: Snapshot number
    //      xs, xf, ys, yf, xs, xf: Start and finish cell coordinates for snapshot
    //      dx, dy, dz: Sampling interval in cell coordinates for snapshot
    //      E, H: Access to field component arrays
    //      snapEx, snapEy, snapEz, snapHx, snapHy, snapHz: Access to arrays to store snapshots

    // Obtain the linear index corresponding to the current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the linear index to subscripts for 4D SNAPS array
    int i = (idx % ($NX_SNAPS * $NY_SNAPS * $NZ_SNAPS)) / ($NY_SNAPS * $NZ_SNAPS);
    int j = ((idx % ($NX_SNAPS * $NY_SNAPS * $NZ_SNAPS)) % ($NY_SNAPS * $NZ_SNAPS)) / $NZ_SNAPS;
    int k = ((idx % ($NX_SNAPS * $NY_SNAPS * $NZ_SNAPS)) % ($NY_SNAPS * $NZ_SNAPS)) % $NZ_SNAPS;

    // Subscripts for field arrays
    int ii, jj, kk;

    if (i >= xs && i < xf && j >= ys && j < yf && k >= zs && k < zf) {

        // Increment subscripts for field array to account for spatial sampling of snapshot
        ii = (xs + i) * dx;
        jj = (ys + j) * dy;
        kk = (zs + k) * dz;

        // The electric field component value at a point comes from an average of
        // the 4 electric field component values in that cell
        snapEx[INDEX4D_SNAPS(p,i,j,k)] = (Ex[INDEX3D_FIELDS(ii,jj,kk)] + Ex[INDEX3D_FIELDS(ii,jj+1,kk)] + Ex[INDEX3D_FIELDS(ii,jj,kk+1)] + Ex[INDEX3D_FIELDS(ii,jj+1,kk+1)]) / 4;
        snapEy[INDEX4D_SNAPS(p,i,j,k)] = (Ey[INDEX3D_FIELDS(ii,jj,kk)] + Ey[INDEX3D_FIELDS(ii+1,jj,kk)] + Ey[INDEX3D_FIELDS(ii,jj,kk+1)] + Ey[INDEX3D_FIELDS(ii+1,jj,kk+1)]) / 4;
        snapEz[INDEX4D_SNAPS(p,i,j,k)] = (Ez[INDEX3D_FIELDS(ii,jj,kk)] + Ez[INDEX3D_FIELDS(ii+1,jj,kk)] + Ez[INDEX3D_FIELDS(ii,jj+1,kk)] + Ez[INDEX3D_FIELDS(ii+1,jj+1,kk)]) / 4;

        // The magnetic field component value at a point comes from average of
        // 2 magnetic field component values in that cell and the following cell
        snapHx[INDEX4D_SNAPS(p,i,j,k)] = (Hx[INDEX3D_FIELDS(ii,jj,kk)] + Hx[INDEX3D_FIELDS(ii+1,jj,kk)]) / 2;
        snapHy[INDEX4D_SNAPS(p,i,j,k)] = (Hy[INDEX3D_FIELDS(ii,jj,kk)] + Hy[INDEX3D_FIELDS(ii,jj+1,kk)]) / 2;
        snapHz[INDEX4D_SNAPS(p,i,j,k)] = (Hz[INDEX3D_FIELDS(ii,jj,kk)] + Hz[INDEX3D_FIELDS(ii,jj,kk+1)]) / 2;
    }
}

""")
