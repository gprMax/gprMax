import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.constants cimport floattype_t
from gprMax.constants cimport complextype_t
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as mu0

cpdef void save_fields_fluxes(
    floattype_t[:, :, ::1] Ex,
    floattype_t[:, :, ::1] Ey,
    floattype_t[:, :, ::1] Ez,
    floattype_t[:, :, ::1] Hx,
    floattype_t[:, :, ::1] Hy,
    floattype_t[:, :, ::1] Hz,
    floattype_t[::1] omega,
    complextype_t[:, :, :, :, ::1] E_omega,
    complextype_t[:, :, :, :, ::1] H_omega,
    int x_begin,
    int y_begin,
    int z_begin,
    int Nx,
    int Ny,
    int Nz,
    int Nf,
    int bottom_x,
    int bottom_y,
    int bottom_z,
    floattype_t dt,
    int nthreads,
    int iteration
):

    cdef Py_ssize_t i, j, k, ii, jj, kk, f
    cdef floattype_t const= dt/np.sqrt(2*np.pi)
    cdef floattype_t arg_exp = iteration*dt
    cdef complextype_t fact
    cdef floattype_t e=np.exp(1) 

    for f in prange(0, Nf, nogil = True, schedule= 'static', num_threads= nthreads):
        fact = e**(1j * arg_exp * omega[f]) * const
        for i in range(x_begin, x_begin + Nx):
            ii = bottom_x + i
            for j in range(y_begin, y_begin + Ny):
                jj = bottom_y + j
                for k in range(z_begin, z_begin + Nz):
                    kk = bottom_z + k
                    E_omega[f, i, j, k, 0] += Ex[ii, jj, kk] * fact
                    E_omega[f, i, j, k, 1] += Ey[ii, jj, kk] * fact
                    E_omega[f, i, j, k, 2] += Ez[ii, jj, kk] * fact

                    H_omega[f, i, j, k, 0] += Hx[ii, jj, kk] * fact
                    H_omega[f, i, j, k, 1] += Hy[ii, jj, kk] * fact
                    H_omega[f, i, j, k, 2] += Hz[ii, jj, kk] * fact