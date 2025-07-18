# Copyright (C) 2025: Quandela
#                 Authors: Quentin David
#
# This file is added to the modified code of gprMax allowing for cylindrical coordinate.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU GenRAl Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU GenRAl Public License for more details.
#
# You should have received a copy of the GNU GenRAl Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.
#
#To get information as to why the PMLs are updated the way they are, please
# check https://repository.mines.edu/server/api/core/bitstreams/2cbef3b2-38af-4c25-bb9c-55bdd3abec74/content

import numpy as np
cimport numpy as np
from cython.parallel import prange
from scipy.conftest import num_parallel_threads

from gprMax.constants cimport floattype_t
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as mu0

################## Update of the PMLs in the r direction ###############################

cpdef void get_Constant_lists( #OK pour les fomules des constantes
                    int rs,
                    int rf,
                    int nz,
                    float dr,
                    float dt,
                    int nthreads,
                    floattype_t[:, :, ::1] sigma,
                    floattype_t[:, :, ::1] alpha,
                    floattype_t[:, :, ::1] kappa,
                    floattype_t[:, :, ::1] return_omega,
                    floattype_t[:, :, ::1] return_Ksi_list,
                    floattype_t[:, :, ::1] return_Lambda_list,
                    floattype_t[:, :, ::1] return_Psi_list,
                    floattype_t[:, :, ::1] return_Theta_list,
                    floattype_t[:, :, ::1] return_alpha,
                    floattype_t[:, :, 1:] return_R_list,
):
    """
    This function computes the Ksi_list and Lambda_list used in the PML updates. Only updated once !
    
    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis
        dr (float): spatial discretization
        dt (float): timestep
        nthreads (int): number of threads to use
        sigma, alpha, kappa (memoryview): PML lists
        return_omega (memoryview): return list with omega values
        return_Ksi_list (memoryview): return list with Ksi_list values.
        return_Lambda_list (memoryview): return list with Lambda_list values. 
        return_Psi_list (memoryview): return list with Psi_list values. 
        return_Theta_list (memoryview): return list with Theta_list values. 
        return_alpha (memoryview): return list with exp(-alpha * dt / e0) values. 
        return_R_list (memoryview): return list with R_list values. 

    """
    cdef Py_ssize_t i, k, ii
    cdef int nr

    nr = rf - rs

    for k in prange(0, nz, nogil= True, schedule= 'static', num_threads = nthreads):
        for i in range(nr):
            arg = alpha[i,0,k]*kappa[i,0,k] + sigma[i,0,k]
            alpha_term = alpha[i, 0, k] * dt / e0
            sh = np.sinh(alpha_term/2)
            exp = np.exp(-alpha_term)

            return_omega[i, 0, k] = sigma[i, 0, k] * dr * (1 - exp) / alpha[i, 0, k] #OK
            return_Ksi_list[i,0,k] = sigma[i,0,k] * dr * sh / e0 #OK
            return_Lambda_list[i,0,k] = sigma[i,0,k] * (1 - np.exp(-arg*dt/(kappa[i,0,k]*e0))) / arg #OK
            return_Psi_list[i,0,k] = sigma[i,0,k] * (1-exp) / alpha[i,0,k] #OK
            return_Theta_list[i,0,k] = sigma[i,0,k] / e0 * sh #OK
            return_alpha[i,0,k] = exp #OK
            if i == 0:
                return_R_list[i,0,k] = 0
            else:
                return_R_list[i,0,k] = return_R_list[i-1,0,k] + sigma[i,0,k] * dr / e0 #OK

cpdef void update_XQEphi_( #OK
        int rs,
        int rf,
        int nz,
        int nthreads,
        floattype_t[:, :, ::1] EPhi,
        floattype_t[:, :, :, ::1] XQEphi_, #XQEphi_[i,j,k] donne la matrice XQEphi_ au point (i,j,k)
        floattype_t[:, :, ::1] Omega_term_list,
        floattype_t[:, :, ::1] alpha_term_list,
):
    """
    This function updates XQEphi_ from time n to time n+1
    
    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis 
        nthreads (int): number of threads to use
        EPhi, Omega_term_list, alpha_term_list (memoryview): lists required for the update. EPhi_ is taken at time n+1
        XQEphi_: list to be updated
    """
    cdef Py_ssize_t i, k, ii, iii
    cdef int nr

    nr = rf-rs

    for k in prange(0, nz, nogil = True, num_threads= nthreads, schedule= 'static'):
        for i in range(0, nr):
            ii = rs + i
            XQEphi_[i,0,k] = Omega_term_list[:,0,k] * EPhi[ii,0,k] + XQEphi_[i, 0, k] * alpha_term_list[:, 0, k]

cpdef void update_XQEzs( #OK
        int rs,
        int rf,
        int nz,
        int nthreads,
        floattype_t[:, :, ::1] Ezs,
        floattype_t[:, :, :, ::1] XQEzs, #XQEzs[i,j,k] donne la matrice XQEzs au point (i,j,k)
        floattype_t[:, :, ::1] Ksi_term_list,
        floattype_t[:, :, ::1] alpha_term_list,
):
    """
    This function updates XQEphi_ from time n to time n+1

    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis 
        nthreads (int): number of threads to use
        Ezs, Ksi_term_list, alpha_term_list (memoryview): lists required for the update. EPhi_ is taken at time n+1
        XQEzs: list to be updated
    """
    cdef Py_ssize_t i, k, ii, iii
    cdef int nr

    nr = rf-rs

    for k in prange(0, nz, nogil = True, num_threads= nthreads, schedule= 'static'):
        for i in range(0, nr):
            ii = rs + i
            XQEzs[i,0,k] = Ksi_term_list[:,0,k] * Ezs[ii,0,k] + XQEzs[i, 0, k] * alpha_term_list[:, 0, k]

cpdef void update_XQHphi_( #OK
        int rs,
        int rf,
        int nz,
        int nthreads,
        floattype_t[:, :, ::1] Hphi,
        floattype_t[:, :, :, ::1] XQHphi_, #XQHphi[i,j,k] donne la matrice XQEzs au point (i,j,k)
        floattype_t[:, :, ::1] Omega_term_list,
        floattype_t[:, :, ::1] alpha_term_list,
):
    """
    This function updates XQHphi_ from time n-1/2 to time n+1/2

    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis 
        nthreads (int): number of threads to use
        HPhi, Omega_term_list (memoryview): lists required for the update. EPhi_ is taken at time n+1
        XQHphi_: list to be updated
    """
    cdef Py_ssize_t i, k, ii
    cdef int nr

    nr = rf-rs

    for k in prange(0, nz, nogil = True, num_threads= nthreads, schedule= 'static'):
        for i in range(0, nr):
            ii = rs + i
            XQHphi_[i,0,k] = Omega_term_list[:,0,k] * Hphi[ii,0,k] + XQHphi_[i, 0, k] * alpha_term_list[:, 0, k]

cpdef void update_XQHzs( #OK
        int rs,
        int rf,
        int nz,
        int nthreads,
        floattype_t[:, :, ::1] Hzs,
        floattype_t[:, :, :, ::1] XQHzs, #XQHzs[i,j,k] donne la matrice XQHzs au point (i,j,k)
        floattype_t[:, :, ::1] Ksi_term_list,
        floattype_t[:, :, ::1] alpha_term_list,
):
    """
    This function updates XQHzs from time n-1/2 to time n+1/2

    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis 
        nthreads (int): number of threads to use
        EPhi, Omega_term_list (memoryview): lists required for the update. EPhi_ is taken at time n+1
        XQEphi_: list to be updated
    """
    cdef Py_ssize_t i, k, ii
    cdef int nr

    nr = rf-rs

    for k in prange(0, nz, nogil = True, num_threads= nthreads, schedule= 'static'):
        for i in range(0, nr):
            ii = rs + i
            XQHzs[i,0,k] = Ksi_term_list[:,0,k] * Hzs[ii,0,k] + XQHzs[i, 0, k] * alpha_term_list[:, 0, k]



cpdef void E_update_r_slab(
                        int rs,
                        int rf,
                        int m,
                        int nz,
                        int nz_tot,
                        float dr,
                        float dz,
                        float dt,
                        int nthreads,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Er,
                        floattype_t[:, :, ::1] Ers,
                        floattype_t[:, :, ::1] QErs,
                        floattype_t[:, :, ::1] Ephi,
                        floattype_t[:, :, ::1] QEphi,
                        floattype_t[:, :, ::1] Ephi_,
                        floattype_t[:, :, :, ::1] XQEphi_,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] QEz,
                        floattype_t[:, :, ::1] Ezs,
                        floattype_t[:, :, :, ::1] XQEzs, #when called, the list is at the step n-1
                        floattype_t[:, :, ::1] Hr,
                        floattype_t[:, :, ::1] Hrs,
                        floattype_t[:, :, ::1] QHrs,
                        floattype_t[:, :, ::1] Hphi,
                        floattype_t[:, :, ::1] QHphi,
                        floattype_t[:, :, ::1] Hphi_,
                        floattype_t[:, :, ::1] QHphi_,
                        floattype_t[:, :, ::1] XQHphi_, #when called, the list is at step n-3/2
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, ::1] QHz,
                        floattype_t[:, :, ::1] Hzs,
                        floattype_t[:, :, ::1] XQHzs,
                        floattype_t[:, :, ::1] alpha,
                        floattype_t[:, :, ::1] sigma,
                        floattype_t[:, :, ::1] kappa,
                        floattype_t[:, :, ::1] b,
                        floattype_t[:, :, ::1] Omega_term_list,
                        floattype_t[:, :, ::1] alpha_term_list,
                        floattype_t[:, :, ::1] Ksi_term_list,
                        floattype_t[:, :, ::1] Lambda_term_list,
                        floattype_t[:, :, ::1] R_term_list,
                        floattype_t[:, :, ::1] Psi_term_list,
                        floattype_t[:, :, ::1] Theta_term_list,
                        floattype_t[:, :, ::1] Alphaexp_term_list,

                ):
    """
    
    This function updates all the E fields inside the PML.
    
    Args:
        rs, rf, zs, zf (int): locations of the fields to be updated
        m (int): the argument in e^(i*m*phi) to ensure the symmetry
        dr, dz (float): spatial discretization (no need for dphi as we use the symmetry)
        dt (float): timestep in s
        nz_tot (int): number of cells along the z axis for the whole domain
        nthreads (int): number of threads to use
        alpha, sigma, kappa, b (memoryviews): PML parameters
        Er, Ephi, Ez, Hr, Hphi, Hz (memoryviews): fields in time domain
        Ers, Ephi_, Ezs, Hrs, Hphi_, Hzs (memoryviews): fields used for PML updates

    """
    cdef Py_ssize_t i, k, ii, kk
    cdef int nr
    cdef floattype_t alpha_term, exp, sigma_term, kappa_term, denominateur_kappa_sigma, Theta_term

    nr = rf-rs

    for i in prange(0, nr, nogil= True, schedule='static', num_threads=nthreads):
        ii = i + rs
        for k in range(0,nz):

            sigma_term = sigma[i,0,k] / (2*e0)
            kappa_term = kappa[i,0,k] / dt
            denominateur_kappa_sigma = (sigma_term + kappa_term)

            #Er,Qers,Ers-> Utiliser liste Psi
            if k == 0: #Boundary conditions
                Er[ii,0,k] += (1j * m * Hz[ii,0,k] /(dr * (ii-0.5)) - (Hphi[ii,0,k] - 0)/dz)/e0 #OK

            else:
                Er[ii,0,k] += (1j * m * Hz[ii,0,k] /(dr * (ii-0.5)) - (Hphi[ii,0,k] - Hphi[ii,0,k-1])/dz)/e0 #OK
            QErs[i,0,k] = QErs[i,0,k]*alpha_term_list[i,0,k] + Psi_term_list[i,0,k] * Er[ii,0,k] #OK
            Ers[i,0,k] = kappa[i,0,k] * Er[ii,0,k] + QErs[i,0,k] #OK

            #Ephi, QEphi
            if k ==0: #Hrs[ii, 0, kk - 1] = 0 because proportionate to Hr
                Ephi[ii, 0, k] = (((kappa_term - sigma_term) * Ephi[ii, 0, k] + QEphi[i, 0, k] * (
                            1 + alpha_term_list[i, 0, k]) + (Hrs[i, 0, k] - 0) / (dz * e0) - (
                                    Hz[ii, 0, k] - Hz[ii - 1, 0, k]) / (dr * e0))
                                   / (denominateur_kappa_sigma - Theta_term_list[i, 0, k]))  #OK
            else:
                Ephi[ii,0,k] = (((kappa_term - sigma_term) * Ephi[ii,0,k] + QEphi[i,0,k] * (1 + alpha_term_list[i,0,k]) +
                                (Hrs[i,0,k] - Hrs[i,0,k-1])/(dz*e0) - (Hz[ii,0,k] - Hz[ii-1,0,k])/(dr * e0))
                                / (denominateur_kappa_sigma - Theta_term_list[i,0,k])) #OK

            QEphi[i,0,k] = Theta_term_list[i,0,k] * Ephi[ii,0,k] + QEphi[i,0,k] * alpha_term_list[i,0,k]

    # We leave the first for statement to update XQEphi_ and XQEzs
    update_XQEphi_(rs, rf, nz, nthreads, Ephi, XQEphi_, Omega_term_list, alpha_term_list)
    update_XQEzs(rs, rf, nz, nthreads, Ezs, XQEzs, Ksi_term_list, alpha_term_list)

    for i in prange(0, nr, nogil= True, schedule='static', num_threads=nthreads):
        ii = i + rs
        for k in range(0,nz):
            # Ephi_
            Ephi_[ii,0,k] = (b[i,0,k]*Ephi[ii,0,k] + #OK
                              np.sum(XQEphi_[i,0,k])) #This sum is in fact QEphi_

            # Ezs
            b_term = b[i,0,k] / dt
            R_term = R_term_list[i,0,k] / 2
            denominateur_R_b = b_term + R_term
            arg = (alpha[i,0,k] * kappa[i,0,k] + sigma[i,0,k]) * dt / (kappa[i,0,k] * e0)
            QEzs_n = np.sum(XQEzs[i,0,k])

            #I decided to take the derivative of Hphi_ as Hphi_(i+1) - Hphi_(i), which is different from the paper
            # as it is easier to process Hphi_(rmax +  1), which is null because of boundary conditions, than Hphi_(i0-1), which has to be computed
            if i == nr-1:
                Ezs[i, 0, k] = ((b_term - R_term) * Ezs[i, 0, k] + QEzs_n + np.sum(
                    XQEzs[i, 0, k] * alpha_term_list[:, 0, k])  #OK
                                + (0 - Hphi_[i, 0, k]) / (dr * e0) -
                                1j * m * Hrs[i, 0, k] / e0) / (denominateur_R_b - np.sum(Ksi_term_list[:, 0, k]))
            else:
                Ezs[i, 0, k] = ((b_term - R_term) * Ezs[i,0,k] + QEzs_n + np.sum(XQEzs[i,0,k] * alpha_term_list[:,0,k]) #OK
                                  + (Hphi_[i+1,0,k] - Hphi_[i,0,k])/(dr * e0) -
                                  1j * m * Hrs[i,0,k] / e0 )/(denominateur_R_b - np.sum(Ksi_term_list[:,0,k]))

            #QEz
            QEz[i,0,k] = Lambda_term_list[i,0,k] * Ezs[i,0,k] + QEz[i,0,k]*np.exp(-arg) #OK

            #Ez
            Ez[i,0,k] = (Ezs[i,0,k] - QEz[i,0,k])/kappa[i,0,k] #OK





cpdef void H_update_r_slab(
                        int rs,
                        int rf,
                        int m,
                        int nz,
                        int nz_tot,
                        float dr,
                        float dz,
                        float dt,
                        int nthreads,
                        np.uint32_t[:, :, :, ::1] ID,
                        floattype_t[:, :, ::1] Er,
                        floattype_t[:, :, ::1] Ers,
                        floattype_t[:, :, ::1] QErs,
                        floattype_t[:, :, ::1] Ephi,
                        floattype_t[:, :, ::1] QEphi,
                        floattype_t[:, :, ::1] Ephi_,
                        floattype_t[:, :, :, ::1] XQEphi_,
                        floattype_t[:, :, ::1] Ez,
                        floattype_t[:, :, ::1] QEz,
                        floattype_t[:, :, ::1] Ezs,
                        floattype_t[:, :, :, ::1] XQEzs, #when called, the list is at the step n-1
                        floattype_t[:, :, ::1] Hr,
                        floattype_t[:, :, ::1] Hrs,
                        floattype_t[:, :, ::1] QHrs,
                        floattype_t[:, :, ::1] Hphi,
                        floattype_t[:, :, ::1] QHphi,
                        floattype_t[:, :, ::1] Hphi_,
                        floattype_t[:, :, ::1] QHphi_,
                        floattype_t[:, :, ::1] XQHphi_, #when called, the list is at step n-3/2
                        floattype_t[:, :, ::1] Hz,
                        floattype_t[:, :, ::1] QHz,
                        floattype_t[:, :, ::1] Hzs,
                        floattype_t[:, :, ::1] XQHzs,
                        floattype_t[:, :, ::1] alpha,
                        floattype_t[:, :, ::1] sigma,
                        floattype_t[:, :, ::1] kappa,
                        floattype_t[:, :, ::1] b,
                        floattype_t[:, :, ::1] Omega_term_list,
                        floattype_t[:, :, ::1] alpha_term_list,
                        floattype_t[:, :, ::1] Ksi_term_list,
                        floattype_t[:, :, ::1] Lambda_term_list,
                        floattype_t[:, :, ::1] R_term_list,
                        floattype_t[:, :, ::1] Psi_term_list,
                        floattype_t[:, :, ::1] Theta_term_list,
                        floattype_t[:, :, ::1] Alphaexp_term_list,

                ):
    """

    This function updates all the H fields inside the PML.

    Args:
        rs, rf, zs, zf (int): locations of the fields to be updated
        m (int): the argument in e^(i*m*phi) to ensure the symmetry
        dr, dz (float): spatial discretization (no need for dphi as we use the symmetry)
        dt (float): timestep in s
        nz_tot (int): number of cells along the z axis for the whole domain
        nthreads (int): number of threads to use
        alpha, sigma, kappa, b (memoryviews): PML parameters
        Er, Ephi, Ez, Hr, Hphi, Hz (memoryviews): fields in time domain
        Ers, Ephi_, Ezs, Hrs, Hphi_, Hzs (memoryviews): fields used for PML updates

    """
    cdef Py_ssize_t i, k, ii, kk
    cdef int nr, nz
    cdef floattype_t alpha_term, exp, sigma_term, kappa_term, denominateur_kappa_sigma, Theta_term

    nr = rf-rs

    for i in prange(0, nr, nogil= True, schedule='static', num_threads=nthreads):
        ii = i + rs
        for k in range(0,nz):
            #Hr, QHrs, Hrs
            if k ==0:
                Hr[ii,0,k] += (Ephi[ii,0,k] - 0) / (dz * mu0) - 1j * m * Ez[ii,0,k] / ((ii-1) * dr * mu0) #OK
            else:
                Hr[ii,0,k] += (Ephi[ii,0,k] - Ephi[ii,0,k-1]) / (dz * mu0) - 1j * m * Ez[ii,0,k] / ((ii-1) * dr * mu0) #OK
            QHrs[i,0,k] = Psi_term_list[i,0,k] * Hr[ii,0,k] + QHrs[i,0,k] * alpha_term_list[i,0,k] #OK
            Hrs[i,0,k] = kappa[i,0,k] * Hr[ii,0,k] + QHrs[i,0,k] #OK

            #Hphi
            sigma_term = sigma[i, 0, k] / (2 * e0)
            kappa_term = kappa[i, 0, k] / dt

            if k == nz-1: #Ers is proportionate to Er
                if i == nr-1:
                    Hphi[i, 0, k] = (((kappa_term - sigma_term) * Hphi[ii, 0, k] + (1 + alpha_term_list[i, 0, k]) *
                                      QHphi[i, 0, k]
                                      + (0 - Ez[ii, 0, k]) / (dr * mu0) - (
                                              0 - Ers[i, 0, k]) / (dz * mu0)) /
                                     (sigma_term + kappa_term - Theta_term_list[i, 0, k]))
                else:
                    Hphi[i, 0, k] = (((kappa_term - sigma_term) * Hphi[ii, 0, k] + (1 + alpha_term_list[i, 0, k]) *
                                      QHphi[i, 0, k]
                                      + (Ez[ii + 1, 0, k] - Ez[ii, 0, k]) / (dr * mu0) - (
                                                  0 - Ers[i, 0, k]) / (dz * mu0)) /
                                     (sigma_term + kappa_term - Theta_term_list[i, 0, k]))
            else:
                if i == nr-1:
                    Hphi[i, 0, k] = (((kappa_term - sigma_term) * Hphi[ii, 0, k] + (1 + alpha_term_list[i, 0, k]) *
                                      QHphi[i, 0, k]
                                      + (0 - Ez[ii, 0, k]) / (dr * mu0) - (
                                                  Ers[i, 0, k + 1] - Ers[i, 0, k]) / (dz * mu0)) /
                                     (sigma_term + kappa_term - Theta_term_list[i, 0, k]))
                else:
                    Hphi[i,0,k] = (((kappa_term - sigma_term) * Hphi[ii,0,k] + (1 + alpha_term_list[i,0,k]) * QHphi[i,0,k]
                                   + (Ez[ii+1,0,k] - Ez[ii,0,k])/ (dr * mu0) - (Ers[i,0,k+1] - Ers[i,0,k])/(dz * mu0))/
                                   (sigma_term + kappa_term - Theta_term_list[i,0,k])) #OK

            #QHphi
            QHphi[i,0,k] = Theta_term_list[i,0,k] * Hphi[ii,0,k] + QHphi[i,0,k] * alpha_term_list[i,0,k]

    # We leave the for statement to update XQHphi_ and XQHzs
    update_XQHphi_(rs, rf, nz, nthreads, Hphi, XQHphi_, Omega_term_list, alpha_term_list)
    update_XQHzs(rs, rf, nz, nthreads, Hzs, XQHzs, Ksi_term_list, alpha_term_list)
    for i in prange(0, nr, nogil= True, schedule='static', num_threads=nthreads):
        ii = i + rs
        for k in range(0,nz):

            #Hphi_
            Hphi_[i,0,k] = (b[i,0,k] * Hphi[ii,0,k] #OK
                            + np.sum(XQHphi_[i,0,k])) #QHphi_

            #Hzs
            b_term = b[i,0,k] / dt
            R_term = R_term_list[i,0,k] / 2
            QHzs = np.sum(XQHzs)

            if i == nr-1: #Ephi_ is proportionate to Ephi at this point (at different times)
                Hzs[i, 0, k] = ((b_term - R_term) * Hzs[i, 0, k] + (
                            np.sum(XQHzs[i, 0, k] * alpha_term_list[:, 0, k]) + QHzs) / mu0
                                + Ers[i, 0, k] * 1j * m / e0 - (0 - Ephi_[i, 0, k]) / (dr * e0)) / (
                                           b_term + R_term
                                           - np.sum(Ksi_term_list[:, 0, k]) / mu0)
            else:
                Hzs[i,0,k] = ((b_term - R_term) * Hzs[i,0,k] + (np.sum(XQHzs[i,0,k] * alpha_term_list[:,0,k]) + QHzs)/mu0
                              + Ers[i,0,k] * 1j * m / e0 - (Ephi_[i+1,0,k] - Ephi_[i,0,k])/(dr * e0)) / (b_term + R_term
                                -np.sum(Ksi_term_list[:,0,k])/mu0) #OK

            #QHz
            QHz[i,0,k] = (Lambda_term_list[i,0,k] * Hzs[i,0,k] +
                          QHz[i,0,k] * np.exp(-(alpha[i,0,k] * kappa[i,0,k] + sigma[i,0,k])*dt/(kappa[i,0,k] * e0)))

            #Hz
            Hz[i,0,k] = (Hzs[i,0,k] - QHz[i,0,k])/kappa[i,0,k]

########################################################################################

################## Update of the PMLs in the z direction ###############################

#For this part, it appears that the PML formulation for the z component is in fact the same as in cartesian

cpdef void E_update_z_minus_slab(
        int rs, #The PML will go from r=0 to r=rs
        int zs,
        int zf,
        float dr,
        float dz,
        float dt,
        int m,
        int nthreads,
        floattype_t[:, :, ::1] Er,
        floattype_t[:, :, ::1] Ephi,
        floattype_t[:, :, ::1] Ez,
        floattype_t[:, :, ::1] Hr,
        floattype_t[:, :, ::1] Hphi,
        floattype_t[:, :, ::1] Hz,
        floattype_t[:, :, ::1] kappa, #Not the same list as for the r update
        floattype_t[:, :, ::1] sigma, #Not the same list as for the r update
        floattype_t[:, :, ::1] alpha, #Not the same list as for the r update
        floattype_t[:, :, ::1] JHphi,
        floattype_t[:, :, ::1] JHr,
        floattype_t[:, :, ::1] QHphi,
        floattype_t[:, :, ::1] QHr,
        floattype_t[:, :, ::1] QJHphi,
        floattype_t[:, :, ::1] QJHr,
):

    cdef Py_ssize_t i, k, kk
    cdef int nz, nr

    nz = zf - zs
    nr = rs

    for i in prange(0, nr, nogil= True, schedule= 'static', num_threads= nthreads):
        for k in range(0,nz):
            kk = k + zs
        #Ez update is not impacted by the PML
        Ez[i,0,kk] += (((i + 0.5 + 1)*Hphi[i+1,0,kk] - (i + 0.5)*Hphi[i,0,kk])/((i + 0.5)*dr)  - 1j * m * Hr[i,0,kk])/e0

        #Er update
        Er[i,0,kk] += (1j * Hz[i,0,kk] / ((i + 0.5) * dr) - (Hz[i,0,kk+1] - Hz[i,0,kk])/ dz - JHphi[i,0,k])/e0

        #QHphi
        QHphi[i, 0, k] += (Hphi[i, 0, kk + 1] - Hphi[i, 0, kk]) * (
                    alpha[i, 0, k] * (kappa[i, 0, k] + 1) + sigma[i, 0, k]) * dt / e0

        #QJHr
        QJHr[i, 0, k] += (kappa[i, 0, k] * alpha[i, 0, k] + sigma[i, 0, k]) * dt * JHr[i, 0, k] / e0

        #JHr
        dHphi = (Hphi[i, 0, k + 1] - Hphi[i, 0, k]) / dz
        JHr[i, 0, k] = (QHphi[i, 0, k] + QJHr[i, 0, k]) / kappa[i, 0, k] + dHphi * (1 + 1 / kappa[i, 0, k])

        #Ephi update
        Ephi[i,0,kk] += ((Hr[i,0,kk+1] - Hr[i,0,kk])/dz - (Hz[i+1,0,kk] - Hz[i,0,kk])/dr + JHr[i,0,k])


        #