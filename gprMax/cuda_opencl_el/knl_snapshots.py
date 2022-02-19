# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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


store_snapshot = Template("""
    // Stores field values for a snapshot.
    //
    //  Args:
    //      p: Snapshot number.
    //      xs, xf, ys, yf, xs, xf: Start and finish cell coordinates for snapshot.
    //      dx, dy, dz: Sampling interval in cell coordinates for snapshot.
    //      E, H: Access to field component arrays.
    //      snapEx, snapEy, snapEz, snapHx, snapHy, snapHz: Access to arrays to store snapshots.


    // Convert the linear index to subscripts for 4D SNAPS array
    int x = (i % ($NX_SNAPS * $NY_SNAPS * $NZ_SNAPS)) / ($NY_SNAPS * $NZ_SNAPS);
    int y = ((i % ($NX_SNAPS * $NY_SNAPS * $NZ_SNAPS)) % ($NY_SNAPS * $NZ_SNAPS)) / $NZ_SNAPS;
    int z = ((i % ($NX_SNAPS * $NY_SNAPS * $NZ_SNAPS)) % ($NY_SNAPS * $NZ_SNAPS)) % $NZ_SNAPS;

    // Subscripts for field arrays
    int xx, yy, zz;

    if (x >= xs && x < xf && y >= ys && y < yf && z >= zs && z < zf) {

        // Increment subscripts for field array to account for spatial sampling of snapshot
        xx = (xs + x) * dx;
        yy = (ys + y) * dy;
        zz = (zs + z) * dz;

        // The electric field component value at a point comes from an average of
        // the 4 electric field component values in that cell
        snapEx[IDX4D_SNAPS(p,x,y,z)] = (Ex[IDX3D_FIELDS(xx,yy,zz)] + 
                                        Ex[IDX3D_FIELDS(xx,yy+1,zz)] + 
                                        Ex[IDX3D_FIELDS(xx,yy,zz+1)] + 
                                        Ex[IDX3D_FIELDS(xx,yy+1,zz+1)]) / 4;
        snapEy[IDX4D_SNAPS(p,x,y,z)] = (Ey[IDX3D_FIELDS(xx,yy,zz)] + 
                                        Ey[IDX3D_FIELDS(xx+1,yy,zz)] + 
                                        Ey[IDX3D_FIELDS(xx,yy,zz+1)] + 
                                        Ey[IDX3D_FIELDS(xx+1,yy,zz+1)]) / 4;
        snapEz[IDX4D_SNAPS(p,x,y,z)] = (Ez[IDX3D_FIELDS(xx,yy,zz)] + 
                                        Ez[IDX3D_FIELDS(xx+1,yy,zz)] + 
                                        Ez[IDX3D_FIELDS(xx,yy+1,zz)] + 
                                        Ez[IDX3D_FIELDS(xx+1,yy+1,zz)]) / 4;

        // The magnetic field component value at a point comes from average of
        // 2 magnetic field component values in that cell and the following cell
        snapHx[IDX4D_SNAPS(p,x,y,z)] = (Hx[IDX3D_FIELDS(xx,yy,zz)] + 
                                        Hx[IDX3D_FIELDS(xx+1,yy,zz)]) / 2;
        snapHy[IDX4D_SNAPS(p,x,y,z)] = (Hy[IDX3D_FIELDS(xx,yy,zz)] + 
                                        Hy[IDX3D_FIELDS(xx,yy+1,zz)]) / 2;
        snapHz[IDX4D_SNAPS(p,x,y,z)] = (Hz[IDX3D_FIELDS(xx,yy,zz)] + 
                                        Hz[IDX3D_FIELDS(xx,yy,zz+1)]) / 2;
    }
""")