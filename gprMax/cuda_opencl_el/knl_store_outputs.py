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


store_outputs = Template("""
    // Stores field component values for every receiver in the model.
    //
    // Args: 
    //    NRX: total number of receivers in the model.
    //    rxs: array to store field components for receivers - rows 
    //          are field components; columns are iterations; pages are receiver.

    if (i < NRX) {
        int x, y, z;
        x = rxcoords[IDX2D_RXCOORDS(i,0)];
        y = rxcoords[IDX2D_RXCOORDS(i,1)];
        z = rxcoords[IDX2D_RXCOORDS(i,2)];
        rxs[IDX3D_RXS(0,iteration,i)] = Ex[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(1,iteration,i)] = Ey[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(2,iteration,i)] = Ez[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(3,iteration,i)] = Hx[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(4,iteration,i)] = Hy[IDX3D_FIELDS(x,y,z)];
        rxs[IDX3D_RXS(5,iteration,i)] = Hz[IDX3D_FIELDS(x,y,z)];
    }
""")