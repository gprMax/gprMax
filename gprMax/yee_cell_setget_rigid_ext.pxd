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
cdef bint get_rigid_Ex(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE)
cdef bint get_rigid_Ey(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE)
cdef bint get_rigid_Ez(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE)
cdef void set_rigid_Ex(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE)
cdef void set_rigid_Ey(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE)
cdef void set_rigid_Ez(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE)
cdef void set_rigid_E(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE)
cdef void unset_rigid_E(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidE)

# Get and set functions for the rigid magnetic component array. The rigid array is 4D with the 1st dimension holding
# the 6 magnetic edge components - Hx1, Hx2, Hy1, Hy2, Hz1, Hz2
cdef bint get_rigid_Hx(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH)
cdef bint get_rigid_Hy(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH)
cdef bint get_rigid_Hz(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH)
cdef void set_rigid_Hx(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH)
cdef void set_rigid_Hy(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH)
cdef void set_rigid_Hz(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH)
cdef void set_rigid_H(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH)
cdef void unset_rigid_H(int i, int j, int k, np.int8_t[:, :, :, ::1] rigidH)


