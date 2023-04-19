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

import numpy as np
cimport numpy as np

from gprMax.materials import Material
from gprMax.yee_cell_setget_rigid_ext cimport get_rigid_Ex
from gprMax.yee_cell_setget_rigid_ext cimport get_rigid_Ey
from gprMax.yee_cell_setget_rigid_ext cimport get_rigid_Ez
from gprMax.yee_cell_setget_rigid_ext cimport get_rigid_Hx
from gprMax.yee_cell_setget_rigid_ext cimport get_rigid_Hy
from gprMax.yee_cell_setget_rigid_ext cimport get_rigid_Hz


cpdef void create_electric_average(int i, int j, int k, int numID1, int numID2, int numID3, int numID4, int componentID, G):
    """This function creates a new material by averaging the dielectric properties of the surrounding cells.

    Args:
        i, j, k (int): Cell coordinates.
        numID1, numID2, numID3, numID4 (int): Numeric IDs for materials in surrounding cells.
        componentID (int): Numeric ID for electric field component.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    # Make an ID composed of the names of the four materials that will be averaged
    requiredID = G.materials[numID1].ID + '+' + G.materials[numID2].ID + '+' + G.materials[numID3].ID + '+' + G.materials[numID4].ID

    # Check if this material already exists
    tmp = requiredID.split('+')
    material = [x for x in G.materials if
             x.ID.count(tmp[0]) == requiredID.count(tmp[0]) and
             x.ID.count(tmp[1]) == requiredID.count(tmp[1]) and
             x.ID.count(tmp[2]) == requiredID.count(tmp[2]) and
             x.ID.count(tmp[3]) == requiredID.count(tmp[3])]

    if material:
        G.ID[componentID, i, j, k] = material[0].numID
    else:
        # Create new material
        newNumID = len(G.materials)
        m = Material(newNumID, requiredID)
        m.type = 'dielectric-smoothed'
        # Create averaged constituents for material
        m.er = np.mean((G.materials[numID1].er, G.materials[numID2].er, G.materials[numID3].er, G.materials[numID4].er), axis=0)
        m.se = np.mean((G.materials[numID1].se, G.materials[numID2].se, G.materials[numID3].se, G.materials[numID4].se), axis=0)
        m.mr = np.mean((G.materials[numID1].mr, G.materials[numID2].mr, G.materials[numID3].mr, G.materials[numID4].mr), axis=0)
        m.sm = np.mean((G.materials[numID1].sm, G.materials[numID2].sm, G.materials[numID3].sm, G.materials[numID4].sm), axis=0)

        # Append the new material object to the materials list
        G.materials.append(m)

        G.ID[componentID, i, j, k] = newNumID


cpdef void create_magnetic_average(int i, int j, int k, int numID1, int numID2, int componentID, G):
    """This function creates a new material by averaging the dielectric properties of the surrounding cells.

    Args:
        i, j, k (int): Cell coordinates.
        numID1, numID2 (int): Numeric IDs for materials in surrounding cells.
        componentID (int): Numeric ID for electric field component.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    # Make an ID composed of the names of the two materials that will be averaged
    requiredID = G.materials[numID1].ID + '+' + G.materials[numID2].ID

    # Check if this material already exists
    tmp = requiredID.split('+')
    material = [x for x in G.materials if
                (x.ID.count(tmp[0]) == requiredID.count(tmp[0]) and
                 x.ID.count(tmp[1]) == requiredID.count(tmp[1])) or
                (x.ID.count(tmp[0]) % 2 == 0 and x.ID.count(tmp[1]) % 2 == 0)]

    if material:
        G.ID[componentID, i, j, k] = material[0].numID
    else:
        # Create new material
        newNumID = len(G.materials)
        m = Material(newNumID, requiredID)
        m.type = 'dielectric-smoothed'
        # Create averaged constituents for material
        m.er = np.mean((G.materials[numID1].er, G.materials[numID2].er), axis=0)
        m.se = np.mean((G.materials[numID1].se, G.materials[numID2].se), axis=0)
        m.mr = np.mean((G.materials[numID1].mr, G.materials[numID2].mr), axis=0)
        m.sm = np.mean((G.materials[numID1].sm, G.materials[numID2].sm), axis=0)

        # Append the new material object to the materials list
        G.materials.append(m)

        G.ID[componentID, i, j, k] = newNumID


cpdef void build_electric_components(np.uint32_t[:, :, ::1] solid, np.int8_t[:, :, :, ::1] rigidE, np.uint32_t[:, :, :, ::1] ID, G):
    """This function builds the electric field components in the ID array.

    Args:
        solid, rigid, ID (memoryviews): Access to solid, rigid and ID arrays
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    cdef Py_ssize_t i, j, k
    cdef int numID1, numID2, numID3, numID4, componentID

    # Ex component
    componentID = G.IDlookup['Ex']
    for i in range(0, G.nx):
        for j in range(1, G.ny):
            for k in range(1, G.nz):

                # If rigid is True do not average
                if get_rigid_Ex(i, j, k, rigidE):
                    pass
                else:
                    numID1 = solid[i, j, k]
                    numID2 = solid[i, j - 1, k]
                    numID3 = solid[i, j - 1, k - 1]
                    numID4 = solid[i, j, k - 1]

                    # If all values are the same no need to average
                    if numID1 == numID2 and numID1 == numID3 and numID1 == numID4:
                        ID[componentID, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_electric_average(i, j, k, numID1, numID2, numID3, numID4, componentID, G)

    # Ey component
    componentID = G.IDlookup['Ey']
    for i in range(1, G.nx):
        for j in range(0, G.ny):
            for k in range(1, G.nz):

                # If rigid is True do not average
                if get_rigid_Ey(i, j, k, rigidE):
                    pass
                else:
                    numID1 = solid[i, j, k]
                    numID2 = solid[i - 1, j, k]
                    numID3 = solid[i - 1, j, k - 1]
                    numID4 = solid[i, j, k - 1]

                    # If all values are the same no need to average
                    if numID1 == numID2 and numID1 == numID3 and numID1 == numID4:
                        ID[componentID, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_electric_average(i, j, k, numID1, numID2, numID3, numID4, componentID, G)

    # Ez component
    componentID = G.IDlookup['Ez']
    for i in range(1, G.nx):
        for j in range(1, G.ny):
            for k in range(0, G.nz):

                # If rigid is True do not average
                if get_rigid_Ez(i, j, k, rigidE):
                    pass
                else:
                    numID1 = solid[i, j, k]
                    numID2 = solid[i - 1, j, k]
                    numID3 = solid[i - 1, j - 1, k]
                    numID4 = solid[i, j - 1, k]

                    # If all values are the same no need to average
                    if numID1 == numID2 and numID1 == numID3 and numID1 == numID4:
                        ID[componentID, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_electric_average(i, j, k, numID1, numID2, numID3, numID4, componentID, G)


cpdef void build_magnetic_components(np.uint32_t[:, :, ::1] solid, np.int8_t[:, :, :, ::1] rigidH, np.uint32_t[:, :, :, ::1] ID, G):
    """This function builds the magnetic field components in the ID array.

    Args:
        solid, rigid, ID (memoryviews): Access to solid, rigid and ID arrays
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    cdef Py_ssize_t i, j, k
    cdef int numID1, numID2, componentID

    # Hx component
    componentID = G.IDlookup['Hx']
    for i in range(1, G.nx):
        for j in range(0, G.ny):
            for k in range(0, G.nz):

                # If rigid is True do not average
                if get_rigid_Hx(i, j, k, rigidH):
                    pass
                else:
                    numID1 = solid[i, j, k]
                    numID2 = solid[i - 1, j, k]

                    # If all values are the same no need to average
                    if numID1 == numID2:
                        ID[componentID, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_magnetic_average(i, j, k, numID1, numID2, componentID, G)

    # Hy component
    componentID = G.IDlookup['Hy']
    for i in range(0, G.nx):
        for j in range(1, G.ny):
            for k in range(0, G.nz):

                # If rigid is True do not average
                if get_rigid_Hy(i, j, k, rigidH):
                    pass
                else:
                    numID1 = solid[i, j, k]
                    numID2 = solid[i, j - 1, k]

                    # If all values are the same no need to average
                    if numID1 == numID2:
                        ID[4, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_magnetic_average(i, j, k, numID1, numID2, componentID, G)

    # Hz component
    componentID = G.IDlookup['Hz']
    for i in range(0, G.nx):
        for j in range(0, G.ny):
            for k in range(1, G.nz):

                # If rigid is True do not average
                if get_rigid_Hz(i, j, k, rigidH):
                    pass
                else:
                    numID1 = solid[i, j, k]
                    numID2 = solid[i, j, k - 1]

                    # If all values are the same no need to average
                    if numID1 == numID2:
                        ID[5, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_magnetic_average(i, j, k, numID1, numID2, componentID, G)
