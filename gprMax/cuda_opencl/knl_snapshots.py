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

from string import Template

store_snapshot = {
    "args_cuda": Template(
        """
                                __global__ void store_snapshot(int p,
                                                    int xs,
                                                    int ys,
                                                    int zs,
                                                    int nx,
                                                    int ny,
                                                    int nz,
                                                    int dx,
                                                    int dy,
                                                    int dz,
                                                    int sx,
                                                    int sy,
                                                    int sz,
                                                    const $REAL* __restrict__ Ex,
                                                    const $REAL* __restrict__ Ey,
                                                    const $REAL* __restrict__ Ez,
                                                    const $REAL* __restrict__ Hx,
                                                    const $REAL* __restrict__ Hy,
                                                    const $REAL* __restrict__ Hz,
                                                    $REAL *snapEx,
                                                    $REAL *snapEy,
                                                    $REAL *snapEz,
                                                    $REAL *snapHx,
                                                    $REAL *snapHy,
                                                    $REAL *snapHz)
                                """
    ),
    "args_opencl": Template(
        """
                                    int p,
                                    int xs,
                                    int ys,
                                    int zs,
                                    int nx,
                                    int ny,
                                    int nz,
                                    int dx,
                                    int dy,
                                    int dz,
                                    int sx,
                                    int sy,
                                    int sz,
                                    __global const $REAL* restrict Ex,
                                    __global const $REAL* restrict Ey,
                                    __global const $REAL* restrict Ez,
                                    __global const $REAL* restrict Hx,
                                    __global const $REAL* restrict Hy,
                                    __global const $REAL* restrict Hz,
                                    __global $REAL* restrict snapEx,
                                    __global $REAL* restrict snapEy,
                                    __global $REAL* restrict snapEz,
                                    __global $REAL* restrict snapHx,
                                    __global $REAL* restrict snapHy,
                                    __global $REAL* restrict snapHz
                                """
    ),
    "args_metal": Template(
        """
                                        kernel void store_snapshot(device const int& p,
                                                    device const int& xs,
                                                    device const int& ys,
                                                    device const int& zs,
                                                    device const int& nx,
                                                    device const int& ny,
                                                    device const int& nz,
                                                    device const int& dx,
                                                    device const int& dy,
                                                    device const int& dz,
                                                    device const int& sx,
                                                    device const int& sy,
                                                    device const int& sz,
                                                    device const $REAL* Ex,
                                                    device const $REAL* Ey,
                                                    device const $REAL* Ez,   
                                                    device const $REAL* Hx,   
                                                    device const $REAL* Hy,   
                                                    device const $REAL* Hz,    
                                                    device $REAL* snapEx,   
                                                    device $REAL* snapEy,    
                                                    device $REAL* snapEz,  
                                                    device $REAL* snapHx,    
                                                    device $REAL* snapHy,   
                                                    device $REAL* snapHz,    
                                                    uint i [[thread_position_in_grid]])
                                        """
    ),
    "func": Template(
        """
    // Stores field values for a snapshot.
    //
    //  Args:
    //      p: Snapshot number.
    //      xs, ys, zs: Start cell coordinates for snapshot (in the full
    //          field-array coordinate space).
    //      nx, ny, nz: Number of samples for this snapshot (already
    //          computed on the host as ceil((finish-start)/step), i.e.
    //          the snapshot's own local, 0-based output size - NOT the
    //          absolute finish coordinate. The snaps-array thread index
    //          (x, y, z below) is itself local/0-based (0..NX_SNAPS-1
    //          etc, where NX_SNAPS is sized to the *largest* requested
    //          snapshot, since multiple different-sized snapshots can
    //          share one buffer) - comparing it against nx/ny/nz (this
    //          snapshot's own local size) is the correct bounds check.
    //          Comparing it against an *absolute* finish coordinate
    //          instead (the previous, buggy version of this kernel) only
    //          happened to work for a snapshot starting exactly at the
    //          grid origin (xs=ys=zs=0) - any other snapshot position
    //          silently produced a truncated, wrongly-offset output.
    //      dx, dy, dz: Sampling interval in cell coordinates for snapshot.
    //      sx, sy, sz: Neighbour-offset strides along x, y, z (1 = genuine
    //          averaging with the +1 neighbour; 0 = no genuine neighbour
    //          exists along that axis, so both terms of any pair on that
    //          axis collapse to the same index - used for a 2D TE-mode
    //          model's invariant axis, which has only one real field
    //          value flanked by forced-zero boundary padding, not a
    //          second genuine value to average against. 1 on every axis
    //          for 3D mode and 2D TM mode, matching the original formula
    //          exactly - see gprMax.snapshots._snapshot_axis_strides().
    //      E, H: Access to field component arrays.
    //      snapEx, snapEy, snapEz, snapHx, snapHy, snapHz: Access to arrays to store snapshots.

    $CUDA_IDX

    // Convert the linear index to subscripts for 4D SNAPS array
    int rem_snaps = i % ($NX_SNAPS * $NY_SNAPS * $NZ_SNAPS);
    int x = rem_snaps / ($NY_SNAPS * $NZ_SNAPS);
    int yz_snaps = rem_snaps % ($NY_SNAPS * $NZ_SNAPS);
    int y = yz_snaps / $NZ_SNAPS;
    int z = yz_snaps % $NZ_SNAPS;

    // Subscripts for field arrays
    int xx, yy, zz;

    if (x < nx && y < ny && z < nz) {

        // Increment subscripts for field array to account for spatial sampling of snapshot
        xx = xs + x * dx;
        yy = ys + y * dy;
        zz = zs + z * dz;

        // The electric field component value at a point comes from an average of
        // the 4 electric field component values at the corners of the sampled
        // snapshot cell. For a strided snapshot, those corners are separated by
        // dx/dy/dz full-grid cells, matching the CPU's strided GridView.
        snapEx[IDX4D_SNAPS(p,x,y,z)] = (Ex[IDX3D_FIELDS(xx,yy,zz)] +
                                        Ex[IDX3D_FIELDS(xx,yy+sy*dy,zz)] +
                                        Ex[IDX3D_FIELDS(xx,yy,zz+sz*dz)] +
                                        Ex[IDX3D_FIELDS(xx,yy+sy*dy,zz+sz*dz)]) * ($REAL)0.25;
        snapEy[IDX4D_SNAPS(p,x,y,z)] = (Ey[IDX3D_FIELDS(xx,yy,zz)] +
                                        Ey[IDX3D_FIELDS(xx+sx*dx,yy,zz)] +
                                        Ey[IDX3D_FIELDS(xx,yy,zz+sz*dz)] +
                                        Ey[IDX3D_FIELDS(xx+sx*dx,yy,zz+sz*dz)]) * ($REAL)0.25;
        snapEz[IDX4D_SNAPS(p,x,y,z)] = (Ez[IDX3D_FIELDS(xx,yy,zz)] +
                                        Ez[IDX3D_FIELDS(xx+sx*dx,yy,zz)] +
                                        Ez[IDX3D_FIELDS(xx,yy+sy*dy,zz)] +
                                        Ez[IDX3D_FIELDS(xx+sx*dx,yy+sy*dy,zz)]) * ($REAL)0.25;

        // The magnetic field component value at a point comes from average of
        // 2 magnetic field component values on opposing sampled-cell faces.
        snapHx[IDX4D_SNAPS(p,x,y,z)] = (Hx[IDX3D_FIELDS(xx,yy,zz)] +
                                        Hx[IDX3D_FIELDS(xx+sx*dx,yy,zz)]) * ($REAL)0.5;
        snapHy[IDX4D_SNAPS(p,x,y,z)] = (Hy[IDX3D_FIELDS(xx,yy,zz)] +
                                        Hy[IDX3D_FIELDS(xx,yy+sy*dy,zz)]) * ($REAL)0.5;
        snapHz[IDX4D_SNAPS(p,x,y,z)] = (Hz[IDX3D_FIELDS(xx,yy,zz)] +
                                        Hz[IDX3D_FIELDS(xx,yy,zz+sz*dz)]) * ($REAL)0.5;

    }
"""
    ),
}
