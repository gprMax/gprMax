# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
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


cpdef pml_average_er_mr(
    str dir,
    int s,
    G
):
    """Calculates average permittivity and permeability for building PML 
        (based on underlying material er and mr from solid array).

    Args:
        dir: string identifier for direction of PML.
        s: int for starting cell of PML.
        G: FDTDGrid class describing a grid in a model.
    """

    sumer = 0  # Sum of relative permittivities in PML slab
    summr = 0  # Sum of relative permeabilities in PML slab

    if dir == 'x':
        for j in range(G.ny):
            for k in range(G.nz):
                numID = G.solid[s, j, k]
                material = [x for x in G.materials if x.numID == numID]
                material = material[0]
                sumer += material.er
                summr += material.mr
        averageer = sumer / (G.ny * G.nz)
        averagemr = summr / (G.ny * G.nz)

    elif dir == 'y':
        for i in range(G.nx):
            for k in range(G.nz):
                numID = G.solid[i, s, k]
                material = [x for x in G.materials if x.numID == numID]
                material = material[0]
                sumer += material.er
                summr += material.mr
        averageer = sumer / (G.nx * G.nz)
        averagemr = summr / (G.nx * G.nz)

    elif dir == 'z':
        for i in range(G.nx):
            for j in range(G.ny):
                numID = G.solid[i, j, s]
                material = [x for x in G.materials if x.numID == numID]
                material = material[0]
                sumer += material.er
                summr += material.mr
        averageer = sumer / (G.nx * G.ny)
        averagemr = summr / (G.nx * G.ny)

    return averageer, averagemr
