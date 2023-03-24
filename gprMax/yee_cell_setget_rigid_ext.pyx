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

# Get and set functions for the rigid electric component array. The rigid array is 4D with the 1st dimension holding
# the 12 electric edge components of a cell - Ex1, Ex2, Ex3, Ex4, Ey1, Ey2, Ey3, Ey4, Ez1, Ez2, Ez3, Ez4
cdef bint get_rigid_Ex(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE):
    cdef bint result
    result = False
    if rigidE[0, i, j, k]:
        result = True
    if j != 0:
        if rigidE[1, i, j - 1, k]:
            result = True
    if k != 0:
        if rigidE[3, i, j, k - 1]:
            result = True
    if j != 0 and k != 0:
        if rigidE[2, i, j - 1, k - 1]:
            result = True
    return result

cdef bint get_rigid_Ey(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE):
    cdef bint result
    result = False
    if rigidE[4, i, j, k]:
        result = True
    if i != 0:
        if rigidE[7, i - 1, j, k]:
            result = True
    if k != 0:
        if rigidE[5, i, j, k - 1]:
            result = True
    if i != 0 and k != 0:
        if rigidE[6, i - 1, j, k - 1]:
            result = True
    return result

cdef bint get_rigid_Ez(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE):
    cdef bint result
    result = False
    if rigidE[8, i, j, k]:
        result = True
    if i != 0:
        if rigidE[9, i - 1, j, k]:
            result = True
    if j != 0:
        if rigidE[11, i, j - 1, k]:
            result = True
    if i != 0 and j != 0:
        if rigidE[10, i - 1, j - 1, k]:
            result = True
    return result

cdef void set_rigid_Ex(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE):
    rigidE[0, i, j, k] = True
    if j != 0:
        rigidE[1, i, j - 1, k] = True
    if k != 0:
        rigidE[3, i, j, k - 1] = True
    if j != 0 and k != 0:
        rigidE[2, i, j - 1, k - 1] = True

cdef void set_rigid_Ey(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE):
    rigidE[4, i, j, k] = True
    if i != 0:
        rigidE[7, i - 1, j, k] = True
    if k != 0:
        rigidE[5, i, j, k - 1] = True
    if i != 0 and k != 0:
        rigidE[6, i - 1, j, k - 1] = True

cdef void set_rigid_Ez(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE):
    rigidE[8, i, j, k] = True
    if i != 0:
        rigidE[9, i - 1, j, k] = True
    if j != 0:
        rigidE[11, i, j - 1, k] = True
    if i != 0 and j != 0:
        rigidE[10, i - 1, j - 1, k] = True

cdef void set_rigid_E(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE):
    rigidE[:, i, j, k] = True

cdef void unset_rigid_E(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE):
    rigidE[:, i, j, k] = False

# Get and set functions for the rigid magnetic component array. The rigid array is 4D with the 1st dimension holding
# the 6 magnetic edge components - Hx1, Hx2, Hy1, Hy2, Hz1, Hz2
cdef bint get_rigid_Hx(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH):
    cdef bint result
    result = False
    if rigidH[0, i, j, k]:
        result = True
    if i != 0:
        if rigidH[1, i - 1, j, k]:
            result = True
    return result

cdef bint get_rigid_Hy(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH):
    cdef bint result
    result = False
    if rigidH[2, i, j, k]:
        result = True
    if j != 0:
        if rigidH[3, i, j - 1, k]:
            result = True
    return result

cdef bint get_rigid_Hz(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH):
    cdef bint result
    result = False
    if rigidH[4, i, j, k]:
        result = True
    if k != 0:
        if rigidH[5, i, j, k - 1]:
            result = True
    return result

cdef void set_rigid_Hx(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH):
    rigidH[0, i, j, k] = True
    if i != 0:
        rigidH[1, i - 1, j, k] = True

cdef void set_rigid_Hy(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH):
    rigidH[2, i, j, k] = True
    if j != 0:
        rigidH[3, i, j - 1, k] = True

cdef void set_rigid_Hz(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH):
    rigidH[4, i, j, k] = True
    if k != 0:
        rigidH[5, i, j, k - 1] = True

cdef void set_rigid_H(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH):
    rigidH[:, i, j, k] = True

cdef void unset_rigid_H(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH):
    rigidH[:, i, j, k] = False

