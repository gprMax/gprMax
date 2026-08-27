# cython: cdivision=True
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

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.config cimport float_or_double


cpdef void update_electric(
    int nx,
    int ny,
    int nz,
    int mode2d,
    int nthreads,
    float_or_double[:, ::1] updatecoeffsE,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates the electric field components (for standard materials).

    Args:
        nx, ny, nz: ints for grid size in cells.
        mode2d: int explicitly identifying the active 2D reduction (if
                any), NOT inferred from nx/ny/nz - a domain 2 cells thick
                on one axis is also a valid (if unusual) plain 3D size, and
                relying on size alone would misfire for it, as well as for
                any future reduced-domain feature that also happens to
                produce a small axis. Encoding: -1 = 3D (no reduction);
                0/1/2 = 2D TM,
                invariant axis x/y/z (only the one E component along that
                axis is live); 3/4/5 = 2D TE, invariant axis x/y/z (only
                the two E components tangential to that axis are live, and
                only at the interior reference layer, index 1, on it - the
                third (own-axis) component and the two outer wall layers,
                index 0 and nx/ny/nz on that axis, are forced pec by
                tex()/tey()/tez() and never read by anything, so skipping
                them here is a pure performance win, not an approximation;
                verified bit-exact against the pre-existing 3D fallback
                path this replaces for TE, which already produced the same
                result at greater cost).
        nthreads: int for number of threads to use.
        updatecoeffs, ID, E, H: memoryviews to access update coefficients,
                                ID and field component arrays
    """

    cdef Py_ssize_t i, j, k
    cdef int materialEx, materialEy, materialEz

    # 2D TM, invariant x - Ex component only
    if mode2d == 0:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    materialEx = ID[0, i, j, k]
                    Ex[i, j, k] = (updatecoeffsE[materialEx, 0] * Ex[i, j, k] +
                                   updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) -
                                   updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]))

    # 2D TM, invariant y - Ey component only
    elif mode2d == 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    materialEy = ID[1, i, j, k]
                    Ey[i, j, k] = (updatecoeffsE[materialEy, 0] * Ey[i, j, k] +
                                   updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) -
                                   updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]))

    # 2D TM, invariant z - Ez component only
    elif mode2d == 2:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    materialEz = ID[2, i, j, k]
                    Ez[i, j, k] = (updatecoeffsE[materialEz, 0] * Ez[i, j, k] +
                                   updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) -
                                   updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]))

    # 2D TE, invariant x - Ey, Ez components (Ex forced dead)
    elif mode2d == 3:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    materialEy = ID[1, i, j, k]
                    Ey[i, j, k] = (updatecoeffsE[materialEy, 0] * Ey[i, j, k] +
                                   updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) -
                                   updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]))
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    materialEz = ID[2, i, j, k]
                    Ez[i, j, k] = (updatecoeffsE[materialEz, 0] * Ez[i, j, k] +
                                   updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) -
                                   updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]))

    # 2D TE, invariant y - Ex, Ez components (Ey forced dead)
    elif mode2d == 4:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    materialEx = ID[0, i, j, k]
                    Ex[i, j, k] = (updatecoeffsE[materialEx, 0] * Ex[i, j, k] +
                                   updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) -
                                   updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]))
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    materialEz = ID[2, i, j, k]
                    Ez[i, j, k] = (updatecoeffsE[materialEz, 0] * Ez[i, j, k] +
                                   updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) -
                                   updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]))

    # 2D TE, invariant z - Ex, Ey components (Ez forced dead)
    elif mode2d == 5:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    materialEx = ID[0, i, j, k]
                    Ex[i, j, k] = (updatecoeffsE[materialEx, 0] * Ex[i, j, k] +
                                   updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) -
                                   updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]))
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    materialEy = ID[1, i, j, k]
                    Ey[i, j, k] = (updatecoeffsE[materialEy, 0] * Ey[i, j, k] +
                                   updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) -
                                   updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]))

    # 3D
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    materialEx = ID[0, i, j, k]
                    materialEy = ID[1, i, j, k]
                    materialEz = ID[2, i, j, k]
                    Ex[i, j, k] = (updatecoeffsE[materialEx, 0] * Ex[i, j, k] +
                                   updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) -
                                   updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]))
                    Ey[i, j, k] = (updatecoeffsE[materialEy, 0] * Ey[i, j, k] +
                                   updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) -
                                   updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]))
                    Ez[i, j, k] = (updatecoeffsE[materialEz, 0] * Ez[i, j, k] +
                                   updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) -
                                   updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]))

        # Ex components at i = 0
        for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(1, nz):
                materialEx = ID[0, 0, j, k]
                Ex[0, j, k] = (updatecoeffsE[materialEx, 0] * Ex[0, j, k] +
                               updatecoeffsE[materialEx, 2] * (Hz[0, j, k] - Hz[0, j - 1, k]) -
                               updatecoeffsE[materialEx, 3] * (Hy[0, j, k] - Hy[0, j, k - 1]))

        # Ey components at j = 0
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(1, nz):
                materialEy = ID[1, i, 0, k]
                Ey[i, 0, k] = (updatecoeffsE[materialEy, 0] * Ey[i, 0, k] +
                               updatecoeffsE[materialEy, 3] * (Hx[i, 0, k] - Hx[i, 0, k - 1]) -
                               updatecoeffsE[materialEy, 1] * (Hz[i, 0, k] - Hz[i - 1, 0, k]))

        # Ez components at k = 0
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                materialEz = ID[2, i, j, 0]
                Ez[i, j, 0] = (updatecoeffsE[materialEz, 0] * Ez[i, j, 0] +
                               updatecoeffsE[materialEz, 1] * (Hy[i, j, 0] - Hy[i - 1, j, 0]) -
                               updatecoeffsE[materialEz, 2] * (Hx[i, j, 0] - Hx[i, j - 1, 0]))


cpdef void update_magnetic(
    int nx,
    int ny,
    int nz,
    int mode2d,
    int nthreads,
    float_or_double[:, ::1] updatecoeffsH,
    np.uint32_t[:, :, :, ::1] ID,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz
):
    """Updates the magnetic field components.

    Args:
        nx, ny, nz: ints for grid size in cells.
        mode2d: int explicitly identifying the active 2D reduction (if
                any) - see update_electric() for the full encoding and the
                reasoning for not inferring this from nx/ny/nz. For TM,
                the two H components tangential to the invariant axis are
                live (the own-axis H component is dead). For TE, only the
                one H component along the invariant axis is live, and only
                at the interior reference layer.
        nthreads: int for number of threads to use.
        updatecoeffs, ID, E, H: memoryviews to access update coefficients,
                                ID and field component arrays
    """

    cdef Py_ssize_t i, j, k
    cdef int materialHx, materialHy, materialHz

    # 2D TM, invariant x - Hy, Hz components (Hx forced dead)
    if mode2d == 0:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    materialHy = ID[4, i, j, k]
                    Hy[i, j, k] = (updatecoeffsH[materialHy, 0] * Hy[i, j, k] -
                                   updatecoeffsH[materialHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) +
                                   updatecoeffsH[materialHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k]))
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    materialHz = ID[5, i, j, k]
                    Hz[i, j, k] = (updatecoeffsH[materialHz, 0] * Hz[i, j, k] -
                                   updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) +
                                   updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k]))

    # 2D TM, invariant y - Hx, Hz components (Hy forced dead)
    elif mode2d == 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(0, nz):
                    materialHx = ID[3, i, j, k]
                    Hx[i, j, k] = (updatecoeffsH[materialHx, 0] * Hx[i, j, k] -
                                   updatecoeffsH[materialHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) +
                                   updatecoeffsH[materialHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k]))
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    materialHz = ID[5, i, j, k]
                    Hz[i, j, k] = (updatecoeffsH[materialHz, 0] * Hz[i, j, k] -
                                   updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) +
                                   updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k]))

    # 2D TM, invariant z - Hx, Hy components (Hz forced dead)
    elif mode2d == 2:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(0, nz):
                    materialHx = ID[3, i, j, k]
                    Hx[i, j, k] = (updatecoeffsH[materialHx, 0] * Hx[i, j, k] -
                                   updatecoeffsH[materialHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) +
                                   updatecoeffsH[materialHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k]))
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    materialHy = ID[4, i, j, k]
                    Hy[i, j, k] = (updatecoeffsH[materialHy, 0] * Hy[i, j, k] -
                                   updatecoeffsH[materialHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) +
                                   updatecoeffsH[materialHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k]))

    # 2D TE, invariant x - Hx only
    elif mode2d == 3:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(0, nz):
                    materialHx = ID[3, i, j, k]
                    Hx[i, j, k] = (updatecoeffsH[materialHx, 0] * Hx[i, j, k] -
                                   updatecoeffsH[materialHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) +
                                   updatecoeffsH[materialHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k]))

    # 2D TE, invariant y - Hy only
    elif mode2d == 4:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    materialHy = ID[4, i, j, k]
                    Hy[i, j, k] = (updatecoeffsH[materialHy, 0] * Hy[i, j, k] -
                                   updatecoeffsH[materialHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) +
                                   updatecoeffsH[materialHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k]))

    # 2D TE, invariant z - Hz only
    elif mode2d == 5:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    materialHz = ID[5, i, j, k]
                    Hz[i, j, k] = (updatecoeffsH[materialHz, 0] * Hz[i, j, k] -
                                   updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) +
                                   updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k]))

    # 3D
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(0, nz):
                    materialHx = ID[3, i + 1, j, k]
                    materialHy = ID[4, i, j + 1, k]
                    materialHz = ID[5, i, j, k + 1]
                    Hx[i + 1, j, k] = (updatecoeffsH[materialHx, 0] * Hx[i + 1, j, k] -
                                       updatecoeffsH[materialHx, 2] * (Ez[i + 1, j + 1, k] - Ez[i + 1, j, k]) +
                                       updatecoeffsH[materialHx, 3] * (Ey[i + 1, j, k + 1] - Ey[i + 1, j, k]))
                    Hy[i, j + 1, k] = (updatecoeffsH[materialHy, 0] * Hy[i, j + 1, k] -
                                       updatecoeffsH[materialHy, 3] * (Ex[i, j + 1, k + 1] - Ex[i, j + 1, k]) +
                                       updatecoeffsH[materialHy, 1] * (Ez[i + 1, j + 1, k] - Ez[i, j + 1, k]))
                    Hz[i, j, k + 1] = (updatecoeffsH[materialHz, 0] * Hz[i, j, k + 1] -
                                       updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k + 1] - Ey[i, j, k + 1]) +
                                       updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k + 1] - Ex[i, j, k + 1]))

        # The main loops above update the upper own-axis walls (i=nx for Hx,
        # j=ny for Hy, and k=nz for Hz) but not the corresponding lower walls.
        # Updating both sides is required by the PMC ghost-node formulation
        # and restores structural symmetry with the accelerator kernels.
        for j in prange(0, ny, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(0, nz):
                materialHx = ID[3, 0, j, k]
                Hx[0, j, k] = (updatecoeffsH[materialHx, 0] * Hx[0, j, k] -
                               updatecoeffsH[materialHx, 2] * (Ez[0, j + 1, k] - Ez[0, j, k]) +
                               updatecoeffsH[materialHx, 3] * (Ey[0, j, k + 1] - Ey[0, j, k]))

        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(0, nz):
                materialHy = ID[4, i, 0, k]
                Hy[i, 0, k] = (updatecoeffsH[materialHy, 0] * Hy[i, 0, k] -
                               updatecoeffsH[materialHy, 3] * (Ex[i, 0, k + 1] - Ex[i, 0, k]) +
                               updatecoeffsH[materialHy, 1] * (Ez[i + 1, 0, k] - Ez[i, 0, k]))

        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                materialHz = ID[5, i, j, 0]
                Hz[i, j, 0] = (updatecoeffsH[materialHz, 0] * Hz[i, j, 0] -
                               updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, 0] - Ey[i, j, 0]) +
                               updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, 0] - Ex[i, j, 0]))
