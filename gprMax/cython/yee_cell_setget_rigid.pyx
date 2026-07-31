# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley, 
#                          and Nathan Mannall
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


# Get and set functions for the rigid electric component array. The rigid array
# is 4D with the 1st dimension holding the 12 electric edge components of a
# cell - Ex1, Ex2, Ex3, Ex4, Ey1, Ey2, Ey3, Ey4, Ez1, Ez2, Ez3, Ez4
cdef bint get_rigid_Ex(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidE
) nogil:
    # j, k may run to rigidE.shape[2]/[3] (Ex's dependency-axis far wall),
    # one past the last valid cell index - guard the "own corner" reads
    # (which use the raw j/k) the same way the "neighbour corner" reads
    # already guard j-1/k-1.
    cdef bint result
    result = False
    if j < rigidE.shape[2] and k < rigidE.shape[3]:
        if rigidE[0, i, j, k]:
            result = True
    if j != 0 and k < rigidE.shape[3]:
        if rigidE[1, i, j - 1, k]:
            result = True
    if k != 0 and j < rigidE.shape[2]:
        if rigidE[3, i, j, k - 1]:
            result = True
    if j != 0 and k != 0:
        if rigidE[2, i, j - 1, k - 1]:
            result = True
    return result


cdef bint get_rigid_Ey(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidE
) nogil:
    # i, k may run to rigidE.shape[1]/[3] (Ey's dependency-axis far wall).
    cdef bint result
    result = False
    if i < rigidE.shape[1] and k < rigidE.shape[3]:
        if rigidE[4, i, j, k]:
            result = True
    if i != 0 and k < rigidE.shape[3]:
        if rigidE[7, i - 1, j, k]:
            result = True
    if k != 0 and i < rigidE.shape[1]:
        if rigidE[5, i, j, k - 1]:
            result = True
    if i != 0 and k != 0:
        if rigidE[6, i - 1, j, k - 1]:
            result = True
    return result


cdef bint get_rigid_Ez(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidE
) nogil:
    # i, j may run to rigidE.shape[1]/[2] (Ez's dependency-axis far wall).
    cdef bint result
    result = False
    if i < rigidE.shape[1] and j < rigidE.shape[2]:
        if rigidE[8, i, j, k]:
            result = True
    if i != 0 and j < rigidE.shape[2]:
        if rigidE[9, i - 1, j, k]:
            result = True
    if j != 0 and i < rigidE.shape[1]:
        if rigidE[11, i, j - 1, k]:
            result = True
    if i != 0 and j != 0:
        if rigidE[10, i - 1, j - 1, k]:
            result = True
    return result


cdef void set_rigid_Ex(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidE
) nogil:
    # j, k may run to rigidE.shape[2]/[3] (Ex's dependency-axis far wall,
    # e.g. an x-directed #edge lying exactly on the y=ny or z=nz domain
    # boundary) - guard the "own corner" writes (which use the raw j/k)
    # the same way get_rigid_Ex already guards its "own corner" reads,
    # mirroring it term-for-term. Without this, boundscheck=False (set
    # globally in setup.py) means an out-of-range write here silently
    # corrupts adjacent memory instead of raising an IndexError.
    if j < rigidE.shape[2] and k < rigidE.shape[3]:
        rigidE[0, i, j, k] = True
    if j != 0 and k < rigidE.shape[3]:
        rigidE[1, i, j - 1, k] = True
    if k != 0 and j < rigidE.shape[2]:
        rigidE[3, i, j, k - 1] = True
    if j != 0 and k != 0:
        rigidE[2, i, j - 1, k - 1] = True


cdef void set_rigid_Ey(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidE
) nogil:
    # i, k may run to rigidE.shape[1]/[3] (Ey's dependency-axis far wall)
    # - see set_rigid_Ex's comment; mirrors get_rigid_Ey term-for-term.
    if i < rigidE.shape[1] and k < rigidE.shape[3]:
        rigidE[4, i, j, k] = True
    if i != 0 and k < rigidE.shape[3]:
        rigidE[7, i - 1, j, k] = True
    if k != 0 and i < rigidE.shape[1]:
        rigidE[5, i, j, k - 1] = True
    if i != 0 and k != 0:
        rigidE[6, i - 1, j, k - 1] = True


cdef void set_rigid_Ez(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidE
) nogil:
    # i, j may run to rigidE.shape[1]/[2] (Ez's dependency-axis far wall)
    # - see set_rigid_Ex's comment; mirrors get_rigid_Ez term-for-term.
    if i < rigidE.shape[1] and j < rigidE.shape[2]:
        rigidE[8, i, j, k] = True
    if i != 0 and j < rigidE.shape[2]:
        rigidE[9, i - 1, j, k] = True
    if j != 0 and i < rigidE.shape[1]:
        rigidE[11, i, j - 1, k] = True
    if i != 0 and j != 0:
        rigidE[10, i - 1, j - 1, k] = True


cdef void set_rigid_E(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidE
) nogil:
    rigidE[:, i, j, k] = True


cdef void unset_rigid_E(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidE
) nogil:
    rigidE[:, i, j, k] = False

# Get and set functions for the rigid magnetic component array. The rigid array
# is 4D with the 1st dimension holding the 6 magnetic edge components - Hx1,
# Hx2, Hy1, Hy2, Hz1, Hz2
cdef bint get_rigid_Hx(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidH
) nogil:
    # i may run to rigidH.shape[1] (Hx's dependency-axis far wall). j, k
    # (transverse to Hx's own axis) are only ever iterated safely within
    # bounds by the one existing caller today (yee_cell_build.pyx's
    # material-averaging loops), but a #magnetic_edge lying on the y=ny
    # or z=nz domain boundary calls the corresponding set_rigid_Hx with
    # j/k at exactly that wall - guarded here too for symmetry with that
    # write path and to avoid a latent trap for any future caller.
    cdef bint result
    result = False
    if j >= rigidH.shape[2] or k >= rigidH.shape[3]:
        return result
    if i < rigidH.shape[1]:
        if rigidH[0, i, j, k]:
            result = True
    if i != 0:
        if rigidH[1, i - 1, j, k]:
            result = True
    return result


cdef bint get_rigid_Hy(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidH
) nogil:
    # j may run to rigidH.shape[2] (Hy's dependency-axis far wall) - see
    # get_rigid_Hx's comment for why i, k (transverse) are also guarded.
    cdef bint result
    result = False
    if i >= rigidH.shape[1] or k >= rigidH.shape[3]:
        return result
    if j < rigidH.shape[2]:
        if rigidH[2, i, j, k]:
            result = True
    if j != 0:
        if rigidH[3, i, j - 1, k]:
            result = True
    return result


cdef bint get_rigid_Hz(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidH
) nogil:
    # k may run to rigidH.shape[3] (Hz's dependency-axis far wall) - see
    # get_rigid_Hx's comment for why i, j (transverse) are also guarded.
    cdef bint result
    result = False
    if i >= rigidH.shape[1] or j >= rigidH.shape[2]:
        return result
    if k < rigidH.shape[3]:
        if rigidH[4, i, j, k]:
            result = True
    if k != 0:
        if rigidH[5, i, j, k - 1]:
            result = True
    return result


cdef void set_rigid_Hx(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidH
) nogil:
    # Self-consistent single-position marker, mirroring set_rigid_Ex's own
    # own-position/neighbour-offset shape (just one dependency axis instead
    # of two): a single call here only ever satisfies get_rigid_Hx queried
    # at this SAME (i,j,k) - it never leaks into i-1 or i+1. Callers that
    # need to mark BOTH of a cell's two true H faces (build_voxel/build_box)
    # call this twice, once at each face position.
    #
    # j, k (transverse to Hx's own axis) are used raw with no neighbour
    # offset at all, so - unlike i - they need a plain upper-bound guard,
    # not a get_rigid_Ex-style split across multiple terms: a
    # #magnetic_edge lying exactly on the y=ny or z=nz domain boundary
    # passes j/k == that wall directly (unlike i, which the edge-building
    # loop in geometry_primitives.pyx never lets reach nx, matching
    # get_rigid_Hx's own i < rigidH.shape[1] guard below).
    if j >= rigidH.shape[2] or k >= rigidH.shape[3]:
        return
    if i < rigidH.shape[1]:
        rigidH[0, i, j, k] = True
    if i != 0:
        rigidH[1, i - 1, j, k] = True


cdef void set_rigid_Hy(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidH
) nogil:
    # i, k (transverse to Hy's own axis) - see set_rigid_Hx's comment.
    if i >= rigidH.shape[1] or k >= rigidH.shape[3]:
        return
    if j < rigidH.shape[2]:
        rigidH[2, i, j, k] = True
    if j != 0:
        rigidH[3, i, j - 1, k] = True


cdef void set_rigid_Hz(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidH
) nogil:
    # i, j (transverse to Hz's own axis) - see set_rigid_Hx's comment.
    if i >= rigidH.shape[1] or j >= rigidH.shape[2]:
        return
    if k < rigidH.shape[3]:
        rigidH[4, i, j, k] = True
    if k != 0:
        rigidH[5, i, j, k - 1] = True


cdef void set_rigid_H(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidH
) nogil:
    rigidH[:, i, j, k] = True


cdef void unset_rigid_H(
    int i,
    int j,
    int k,
    np.int8_t[:, :, :, ::1] rigidH
) nogil:
    rigidH[:, i, j, k] = False
