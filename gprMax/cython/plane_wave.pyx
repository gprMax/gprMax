# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley, and
#                          Adittya Pal
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
cimport cython
from libc.math cimport floor, ceil, round, sqrt, tan, cos, sin, atan2, abs, pow, exp, M_PI
from libc.stdio cimport FILE, fopen, fwrite, fclose
from libc.string cimport strcmp

from cython.parallel import prange
from gprMax.config cimport float_or_double

@cython.cdivision(True)
cpdef double[:, ::1] getProjections(
    double psi, 
    int[:] m
):
    """Gets the projection vectors to source magnetic fields of plane wave. 

    Args:
        psi: float for angle describing polatan value of required phi angle
                (which would be approximated to a rational number).
        m: int array to store integer mappings, m_x, m_y, m_z which determine 
            the rational angles, for assignment of the correct element to 3D 
            FDTD grid from 1D representation, last element stores 
            max(m_x, m_y, m_z).
    
    Returns:
        projections: float array to store projections for sourcing magnetic 
                        field and the sourcing vector.
    """

    cdef double phi, theta, cos_phi, sin_phi, cos_psi, sin_psi, cos_theta, sin_theta
    cdef double[:, ::1] projections = np.zeros((2, 3), order='C')

    if m[0] == 0:
        phi = M_PI / 2
    else:
        phi = atan2(m[1], m[0])

    if m[2] == 0:
        theta = M_PI / 2
    else:
        theta = atan2(sqrt(m[0] * m[0] + m[1] * m[1]), m[2])

    cos_phi = cos(phi)
    sin_phi = sin(phi)
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    cos_psi = cos(psi)
    sin_psi = sin(psi)

    # Magnetic field projection vector
    projections[0, 0] =  sin_psi * sin_phi + cos_psi * cos_theta * cos_phi
    projections[0, 1] = -sin_psi * cos_phi + cos_psi * cos_theta * sin_phi
    projections[0, 2] = -cos_psi * sin_theta

    # Direction cosines
    projections[1, 0] = sin_theta * cos_phi
    projections[1, 1] = sin_theta * sin_phi
    projections[1, 2] = cos_theta

    return projections


@cython.cdivision(True)
cdef int[:] getPhi(
    int[:, :] integers, 
    double required_ratio, 
    double tolerance
):
    """Gets rational angle approximation to phi within the requested tolerance 
        level using Farey Fractions to determine a rational number closest to 
        the real number. 
    
    Args:
        integers: int array to determine the value of m_x and m_y.
        required_ratio: float of tan value of the required phi angle 
                        (which would be approximated to a rational number).
        tolerance: float for acceptable deviation in the tan value of the 
                    rational angle from phi.
    
    Returns:
        integers: int array of sequence of the two integers [m_x, m_y].
    """

    if(abs(integers[2, 0]/<double>integers[2, 1]-required_ratio)<tolerance):
        return integers[2, :]
    while True:
        integers[1, 0] = integers[0, 0]+integers[2, 0]
        integers[1, 1] = integers[0, 1]+integers[2, 1]
        ratio = integers[1, 0]/<double>integers[1, 1]
        if (abs(ratio - required_ratio) <= tolerance):
            return integers[1, :]
        elif(ratio >= required_ratio):
            integers[2, :] = integers[1, :]
        else:
            integers[0, :] = integers[1, :]


@cython.cdivision(True)
cdef inline double getTanValue(
    int[:] integers, 
    double[:] dr
):
    """Returns tan value of the angle approximated to theta given three integers.
    
    Args:
        integers: int array of three integers for the rational angle 
                    approximation.
        dr: double array containing the separation between grid points along 
            the three axes [dx, dy, dz].
    
    Returns:
        _tanValue: double of tan value of the rational angle corresponding to 
                    integers m_x, m_y, m_z.
    """

    if(integers[2]==0):   #if rational angle==90 degrees
        return 99999.0       #return a large number to avoid division by zero error
    else:
        return sqrt((integers[0]/dr[0])*(integers[0]/dr[0]) + (integers[1]/dr[1])*(integers[1]/dr[1])
                   )/(integers[2]/dr[2])


@cython.cdivision(True)
cdef int[:, :] get_mZ(
    int m_x, 
    int m_y, 
    double theta, 
    double[:] Delta_r
):
    """Gets arrays to perform a binary search to determine a rational number, 
        m_z, closest to real number, m_z, to get desired tan Theta value. 
    
    Args:
        m_x and m_y: ints approximating rational angle to tan value of phi.
        theta: float of polar angle of incident plane wave (radians) to be 
                approximated to a rational angle.
        Delta_r: float array of projections of  propagation vector along 
                    different coordinate axes.
    
    Returns:
        _integers: int array of 2D sequence of three integers [m_x, m_y, m_z] 
                    to perform a binary search to determine value of m_z within 
                    given limits.
    """

    cdef double m_z = 0
    m_z = sqrt((m_x/Delta_r[0])*(m_x/Delta_r[0]) + (m_y/Delta_r[1])*(m_y/Delta_r[1]))/(tan(theta)/Delta_r[2]) #get an estimate of the m_z value 
    return np.array([[m_x, m_y, floor(m_z)],
                     [m_x, m_y, round(m_z)],
                     [m_x, m_y, ceil(m_z)]], dtype=np.int32, order='C')   #set up the integer array to search for an appropriate m_z
    

@cython.cdivision(True)
cdef int[:] getTheta(
    int m_x, 
    int m_y, 
    double theta, 
    double Delta_theta, 
    double[:] Delta_r
):
    """Gets rational angle approximation to theta within requested tolerance 
        level using Binary Search to determine a rational number closest to 
        real number. 
    
    Args:
        m_x and m_y: ints approximating rational angle to tan value of phi. 
        theta: float of polar angle of incident plane wave (radians) to be 
                approximated to a rational angle.
        Delta_theta: float of permissible error in rational angle approximation 
                        to theta (radians).
        Delta_r: float array of projections of propagation vector along 
                    different coordinate axes.
    
    Returns:
        integers: int array of sequence of three integers [m_x, m_y, m_z].
    """

    cdef Py_ssize_t i, j = 0
    cdef double tan_theta = 0.0
    cdef int[:, :] integers = get_mZ(m_x, m_y, theta, Delta_r)   #set up the integer array to search for an appropriate m_z
    while True:                                                  #if tan value of m_z greater than permitted tolerance 
        tan_theta = getTanValue(integers[1, :], Delta_r)
        if(abs(tan_theta - tan(theta)) <= Delta_theta / (cos(theta) * cos(theta))):
            break
        for i in range(3):
            for j in range(3):
                integers[i, j] = 2*integers[i, j]                 #expand the serach space by multiplying 2 to both the numerator and the denominator
        while(integers[0, 2]<integers[2, 2]-1):                   #while there are integers to search for in the denominator
            integers[1, 2] = (integers[0, 2]+integers[2, 2])/2    #get the mid integer between the upper and lower limits of denominator
            tan_theta = getTanValue(integers[1, :], Delta_r)
            if(tan_theta < tan(theta)):                           #if m_z results in a smaller tan value, make the denominator smaller
                integers[2, 2] = integers[1, 2]                   #decrease m_z, serach in the lower half of the sample space
            elif(tan_theta > tan(theta)):                         #if m_z results in a larger tan value, make the denominator larger
                integers[0, 2] = integers[1, 2]                   #increase m_z, serach in the upper half of the sample space
    
    return integers[1, :]


@cython.cdivision(True)
cpdef int[:, ::1] getIntegerForAngles(
    double phi, 
    double Delta_phi, 
    double theta, 
    double Delta_theta, 
    double[:] Delta_r
):
    """Gets [m_x, m_y, m_z] to determine rational angles given phi and theta 
        along with the permissible tolerances.
    
    Args:
        phi: float of  azimuthal angle of incident plane wave (degrees) to be 
                approximated to a rational angle.
        Delta_phi: float of permissible error in rational angle approximation 
                    to phi (degrees).
        theta: float of polar angle of incident plane wave (degrees) to be 
                approximated to a rational angle.
        Delta_theta: float of permissible error in rational angle approximation 
                        to theta (degrees).
        Delta_r: float of projections of propagation vector along different 
                    coordinate axes.
    
    Returns:
        quadrants[0, :]: int array specifying direction of propagation of plane 
                            wave along the three coordinate axes.
        quadrants[1, :]: int array of three integers [m_x, m_y, m_z].
    """

    cdef double required_ratio_phi, tolerance_phi = 0.0
    cdef int m_x, m_y, m_z = 0
    cdef int[:, ::1] quadrants = np.ones((2, 3), dtype=np.int32)
    if(theta>=90):
        quadrants[0, 2] = -1
        theta = 180-theta
    if(phi>=90 and phi<180):
        quadrants[0, 0] = -1
        phi = 180-phi
    elif(phi>=180 and phi<270):
        quadrants[0, 0] = -1
        quadrants[0, 1] = -1
        phi = phi-180
    elif(phi>=270 and phi<360):
        quadrants[0, 1] = -1
        phi = 360-phi
           
    
    if(0 <= phi < 90):                 #handle the case of phi=90 degrees separately
        required_ratio_phi = tan(M_PI/180*phi) * Delta_r[1] / Delta_r[0]   #to avoid division by zero exception
        tolerance_phi = M_PI/180*Delta_phi / (cos(M_PI/180*phi)*cos(M_PI/180*phi)) * Delta_r[1] / Delta_r[0]  #get the persissible error in tan phi
        m_y, m_x = getPhi(np.array([[floor(required_ratio_phi), 1],
                             [required_ratio_phi, 1],
                             [ceil(required_ratio_phi), 1]], dtype=np.int32, order='C')
                          , required_ratio_phi, tolerance_phi)         #get the integers [m_x, m_y] for rational angle approximation to phi
    else:
        m_x = 0                       
        m_y = 1
    if(theta < 90):
        m_x, m_y, m_z = getTheta(m_x, m_y, M_PI/180*theta, M_PI/180*Delta_theta, Delta_r)   #get the integers [m_x, m_y, m_z] for rational angle approximation to theta
    else:                               #handle the case of theta=90 degrees separately
        m_z = 0                         #to avoid division by zero exception
    quadrants[1, 0] = m_x
    quadrants[1, 1] = m_y
    quadrants[1, 2] = m_z
    return quadrants


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void applyTFSFMagnetic(
    int nthreads, 
    float_or_double[:, :, ::1] Hx, 
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz, 
    float_or_double[:, ::1] E_fields, 
    float_or_double[:] updatecoeffsH,
    int[:] m, 
    int[:] corners
):
    """Implements total field-scattered field formulation for magnetic field on 
        the edge of the TF/SF region of the TFSF Box.
    
    Args:
        nthreads: int of number of threads to parallelize for loops.
        Hx, Hy, Hz: double array to store magnetic fields for grid cells over 
                    the TFSF box at particular indices.
        E_fields: double array to store electric fields of 1D representation of 
                    plane wave in a direction along which the wave propagates.
        updatecoeffsH: float of coefficients of fields in TFSF assignment 
                        equation for the magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine rational 
            angles for assignment of correct element to 3D FDTD grid from 1D 
            representation, last element stores max(m_x, m_y, m_z).
        corners: int array of coordinates of corners of TF/SF field boundaries.    
    """

    cdef Py_ssize_t i, j, k = 0

    # Precompute index values
    cdef int index = 0
    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef int x_start = corners[0]
    cdef int y_start = corners[1]
    cdef int z_start = corners[2]
    cdef int x_stop = corners[3]
    cdef int y_stop = corners[4]
    cdef int z_stop = corners[5]

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]

    cdef float_or_double coef_H_xy = updatecoeffsH[2]
    cdef float_or_double coef_H_xz = updatecoeffsH[3]
    cdef float_or_double coef_H_yz = updatecoeffsH[3]
    cdef float_or_double coef_H_yx = updatecoeffsH[1]
    cdef float_or_double coef_H_zx = updatecoeffsH[1]    
    cdef float_or_double coef_H_zy = updatecoeffsH[2]

    #**** constant x faces -- scattered-field nodes ****
    i = x_start
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hy at firstX-1/2 by subtracting Ez_inc
            index = m_x * i + m_y * j + m_z * k
            Hy[i-1, j, k] -= coef_H_yx * E_z[index] 

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at firstX-1/2 by adding Ey_inc
            index = m_x * i + m_y * j + m_z * k
            Hz[i-1, j, k] += coef_H_zx * E_y[index]

    i = x_stop
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hy at lastX+1/2 by adding Ez_inc
            index = m_x * i + m_y * j + m_z * k
            Hy[i, j, k] += coef_H_yx * E_z[index]    

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at lastX+1/2 by subtractinging Ey_inc
            index = m_x * i + m_y * j + m_z * k
            Hz[i, j, k] -= coef_H_zx * E_y[index]            

    #**** constant y faces -- scattered-field nodes ****
    j = y_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hx at firstY-1/2 by adding Ez_inc
            index = m_x * i + m_y * j + m_z * k
            Hx[i, j-1, k] += coef_H_xy * E_z[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at firstY-1/2 by subtracting Ex_inc
            index = m_x * i + m_y * j + m_z * k
            Hz[i, j-1, k] -= coef_H_zy * E_x[index]

    j = y_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Hx at lastY+1/2 by subtracting Ez_inc
            index = m_x * i + m_y * j + m_z * k
            Hx[i, j, k] -= coef_H_xy * E_z[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Hz at lastY-1/2 by adding Ex_inc
            index = m_x * i + m_y * j + m_z * k
            Hz[i, j, k] += coef_H_zy * E_x[index]

    #**** constant z faces -- scattered-field nodes ****
    k = z_start
    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Hy at firstZ-1/2 by adding Ex_inc
            index = m_x * i + m_y * j + m_z * k
            Hy[i, j, k-1] += coef_H_yz * E_x[index]

    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Hx at firstZ-1/2 by subtracting Ey_inc
            index = m_x * i + m_y * j + m_z * k
            Hx[i, j, k-1] -= coef_H_xz * E_y[index]

    k = z_stop
    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Hy at firstZ-1/2 by subtracting Ex_inc
            index = m_x * i + m_y * j + m_z * k
            Hy[i, j, k] -= coef_H_yz * E_x[index]

    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Hx at lastZ+1/2 by adding Ey_inc
            index = m_x * i + m_y * j + m_z * k
            Hx[i, j, k] += coef_H_xz * E_y[index]

    
cdef void applyTFSFElectric(
    int nthreads, 
    float_or_double[:, :, ::1] Ex, 
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez, 
    float_or_double[:, ::1] H_fields, 
    float_or_double[:] updatecoeffsE,
    int[:] m, 
    int[:] corners
):
    
    """Implements total field-scattered field formulation for electric field on 
        edge of the TF/SF region of the TFSF Box.
    
    Args:
        nthreads: int for number of threads to parallelize the for loops.
        Ex, Ey, Ez: double array for magnetic fields for grid cells over TFSF 
                    box at particular indices.
        H_fields: double array to store electric fields of 1D representation of 
                    plane wave in direction along which wave propagates.
        updatecoeffsE: float of coefficients of fields in TFSF assignment 
                        equation for magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine rational 
            angles for assignment of correct element to 3D FDTD grid from 1D
            representation, last element stores max(m_x, m_y, m_z).
        corners: int array for coordinates of corners of TF/SF field boundaries.    
    """

    cdef Py_ssize_t i, j, k = 0

    # Precompute index values
    cdef int index = 0
    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    cdef int x_start = corners[0]
    cdef int y_start = corners[1]
    cdef int z_start = corners[2]
    cdef int x_stop = corners[3]
    cdef int y_stop = corners[4]
    cdef int z_stop = corners[5]

    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    cdef float_or_double coef_E_xy = updatecoeffsE[2]
    cdef float_or_double coef_E_xz = updatecoeffsE[3]
    cdef float_or_double coef_E_yz = updatecoeffsE[3]
    cdef float_or_double coef_E_yx = updatecoeffsE[1]
    cdef float_or_double coef_E_zx = updatecoeffsE[1]    
    cdef float_or_double coef_E_zy = updatecoeffsE[2]

    #**** constant x faces -- total-field nodes ****/
    i = x_start
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at firstX face by subtracting Hy_inc
            index = m_x * (i-1) + m_y * j + m_z * k
            Ez[i, j, k] -= coef_E_zx * H_y[index]

    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ey at firstX face by adding Hz_inc
            index = m_x * (i-1) + m_y * j + m_z * k
            Ey[i, j, k] += coef_E_yx * H_z[index]

    i = x_stop
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at lastX face by adding Hy_inc
            index = m_x * i + m_y * j + m_z * k
            Ez[i, j, k] += coef_E_zx * H_y[index]

    i = x_stop
    for j in prange(y_start, y_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ey at lastX face by subtracting Hz_inc
            index = m_x * i + m_y * j + m_z * k
            Ey[i, j, k] -= coef_E_yx * H_z[index]

    #**** constant y faces -- total-field nodes ****/
    j = y_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at firstY face by adding Hx_inc
            index = m_x * i + m_y * (j-1) + m_z * k
            Ez[i, j, k] += coef_E_zy * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ex at firstY face by subtracting Hz_inc
            index = m_x * i + m_y * (j-1) + m_z * k
            Ex[i, j, k] -= coef_E_xy * H_z[index]

    j = y_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop):
            #correct Ez at lastY face by subtracting Hx_inc
            index = m_x * i + m_y * j + m_z * k
            Ez[i, j, k] -= coef_E_zy * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(z_start, z_stop+1):
            #correct Ex at lastY face by adding Hz_inc
            index = m_x * i + m_y * j + m_z * k
            Ex[i, j, k] += coef_E_xy * H_z[index]

    #**** constant z faces -- total-field nodes ****/
    k = z_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Ey at firstZ face by subtracting Hx_inc
            index = m_x * i + m_y * j + m_z * (k-1)
            Ey[i, j, k] -= coef_E_yz * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Ex at firstZ face by adding Hy_inc
            index = m_x * i + m_y * j + m_z * (k-1)
            Ex[i, j, k] += coef_E_xz * H_y[index]

    k = z_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop):
            #correct Ey at lastZ face by adding Hx_inc
            index = m_x * i + m_y * j + m_z * k
            Ey[i, j, k] += coef_E_yz * H_x[index]

    for i in prange(x_start, x_stop, nogil=True, schedule='static', num_threads=nthreads):
        for j in range(y_start, y_stop+1):
            #correct Ex at lastZ face by subtracting Hy_inc
            index = m_x * i + m_y * j + m_z * k
            Ex[i, j, k] -= coef_E_xz * H_y[index]
        

cdef void initializeMagneticFields(
    int[:] m,
    float_or_double[:, ::1] H_fields, 
    double[:] projections, 
    float_or_double[:, ::1] waveformvalues_wholedt,
    bint precompute,
    int iteration,
    double dt,
    double ds,
    double c,
    double start, 
    double stop, 
    double freq,
    char* wavetype
):
    """Initialises first few grid points of source waveform.
        
    Args:
        m: int array of integer mappings, m_x, m_y, m_z which determine rational 
            angles for assignment of correct element to 3D FDTD grid from 1D 
            representation, last element stores max(m_x, m_y, m_z).
        H_fields: double array for electric fields of 1D representation of plane 
                    wave in a direction along which the wave propagates.
        projections: float array for projections of magnetic fields along 
                        different cartesian axes.
        waveformvalues_wholedt: double array stores precomputed waveforms at 
                                each timestep to initialize magnetic fields.
        precompute: boolean to store whether fields values have been precomputed 
                    or should be computed on the fly.
        iterations: int stores number of iterations in the simulation.
        dt: float of timestep for the simulation.
        ds: float of projection vector for sourcing the plane wave.
        c: float of speed of light in the medium.
        start: float of start time at which source is placed in the TFSF grid.
        stop: float of stop time at which source is removed from TFSF grid.
        freq: float of frequency of introduced wave which determines grid points 
                per wavelength for wave source.
        wavetype: string of type of waveform whose magnitude should be returned.
    """

    cdef Py_ssize_t r = 0
    cdef double time_x, time_y, time_z = 0.0
    if (precompute == True):
        for r in range(m[3]):      #loop to assign the source values of magnetic field to the first few gridpoints
            H_fields[0, r] = projections[0] * waveformvalues_wholedt[0, r]
            H_fields[1, r] = projections[1] * waveformvalues_wholedt[1, r]
            H_fields[2, r] = projections[2] * waveformvalues_wholedt[2, r]
    else:
        for r in range(m[3]):      #loop to assign the source values of magnetic field to the first few gridpoints
            time_x = dt * iteration - (r+ (m[1]+m[2])*0.5) * ds/c
            time_y = dt * iteration - (r+ (m[2]+m[0])*0.5) * ds/c
            time_z = dt * iteration - (r+ (m[0]+m[1])*0.5) * ds/c
            if (dt * iteration >= start and dt * iteration <= stop):
            # Set the time of the waveform evaluation to account for any delay in the start
                H_fields[0, r] = projections[0] * getSource(time_x-start, freq, wavetype, dt)
                H_fields[1, r] = projections[1] * getSource(time_y-start, freq, wavetype, dt)
                H_fields[2, r] = projections[2] * getSource(time_z-start, freq, wavetype, dt)
    

@cython.cdivision(True)
cdef void updateMagneticFields(
    int n, 
    float_or_double[:, ::1] H_fields, 
    float_or_double[:, ::1] E_fields, 
    float_or_double[:] updatecoeffsH, 
    int[:] m
):
    """Updates magnetic fields for next time step using Equation 8 of 
        DOI: 10.1109/LAWP.2009.2016851.
        
    Args:
        n: int for spatial length of DPW array to update each length grid 
            cell.
        H_fields: double array of magnetic fields of DPW until temporal 
                    index time.
        E_fields: double array of electric fields of DPW until temporal 
                    index time.
        updatecoeffsH: double array of coefficients of fields in update 
                        equation for magnetic field.
        m: int array of integer mappings, m_x, m_y, m_z which determine 
            rational angles for assignment of correct element to 3D FDTD 
            grid from 1D representation, last element stores 
            max(m_x, m_y, m_z).
    """ 

    cdef Py_ssize_t j = 0

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]
    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    cdef float_or_double coef_H_xt = updatecoeffsH[0]
    cdef float_or_double coef_H_xy = updatecoeffsH[2]
    cdef float_or_double coef_H_xz = updatecoeffsH[3]

    cdef float_or_double coef_H_yt = updatecoeffsH[0]
    cdef float_or_double coef_H_yz = updatecoeffsH[3]
    cdef float_or_double coef_H_yx = updatecoeffsH[1]

    cdef float_or_double coef_H_zt = updatecoeffsH[0]
    cdef float_or_double coef_H_zx = updatecoeffsH[1]    
    cdef float_or_double coef_H_zy = updatecoeffsH[2]

    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    for j in range(m[3], n-m[3]):  #loop to update the magnetic field at each spatial index
        H_x[j] = coef_H_xt * H_x[j] + coef_H_xz * ( E_y[j+m_z] - E_y[j] ) - coef_H_xy * ( E_z[j+m_y] - E_z[j] )     #equation 8 of Tan, Potter paper
        H_y[j] = coef_H_yt * H_y[j] + coef_H_yx * ( E_z[j+m_x] - E_z[j] ) - coef_H_yz * ( E_x[j+m_z] - E_x[j] )     #equation 8 of Tan, Potter paper
        H_z[j] = coef_H_zt * H_z[j] + coef_H_zy * ( E_x[j+m_y] - E_x[j] ) - coef_H_zx * ( E_y[j+m_x] - E_y[j] )     #equation 8 of Tan, Potter paper
    

@cython.cdivision(True)
cdef void updateElectricFields(
    int n, 
    float_or_double[:, ::1] H_fields, 
    float_or_double[:, ::1] E_fields, 
    float_or_double[:] updatecoeffsE, 
    int[:] m
):
    """Updates electric fields for next time step using Equation 9 of 
        DOI: 10.1109/LAWP.2009.2016851.
        
    Args:
        n: int for spatial length of DPW array to update each length grid 
            cell.
        H_fields: double array of magnetic fields of DPW until temporal 
                    index time.
        E_fields: double array of electric fields of DPW until temporal 
                    index time.
        updatecoeffsE: double array of coefficients of fields in update 
                        equation for electric field.
        m: int array of integer mappings, m_x, m_y, m_z which determine 
            rational angles for assignment of correct element to 3D FDTD 
            grid from 1D representation, last element stores 
            max(m_x, m_y, m_z).
    """

    cdef Py_ssize_t j = 0

    cdef float_or_double[:] E_x = E_fields[0, :]
    cdef float_or_double[:] E_y = E_fields[1, :]
    cdef float_or_double[:] E_z = E_fields[2, :]
    cdef float_or_double[:] H_x = H_fields[0, :]
    cdef float_or_double[:] H_y = H_fields[1, :]
    cdef float_or_double[:] H_z = H_fields[2, :]

    cdef float_or_double coef_E_xt = updatecoeffsE[0]
    cdef float_or_double coef_E_xy = updatecoeffsE[2]
    cdef float_or_double coef_E_xz = updatecoeffsE[3]

    cdef float_or_double coef_E_yt = updatecoeffsE[0]
    cdef float_or_double coef_E_yz = updatecoeffsE[3]
    cdef float_or_double coef_E_yx = updatecoeffsE[1]

    cdef float_or_double coef_E_zt = updatecoeffsE[0]
    cdef float_or_double coef_E_zx = updatecoeffsE[1]    
    cdef float_or_double coef_E_zy = updatecoeffsE[2]

    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]

    for j in range(m[3], n-m[3]):   #loop to update the electric field at each spatial index 
        E_x[j] = coef_E_xt * E_x[j] + coef_E_xz * ( H_z[j] - H_z[j-m_y] ) - coef_E_xy * ( H_y[j] - H_y[j-m_z] )  #equation 9 of Tan, Potter paper
        E_y[j] = coef_E_yt * E_y[j] + coef_E_yx * ( H_x[j] - H_x[j-m_z] ) - coef_E_yz * ( H_z[j] - H_z[j-m_x] )  #equation 9 of Tan, Potter paper
        E_z[j] = coef_E_zt * E_z[j] + coef_E_zy * ( H_y[j] - H_y[j-m_x] ) - coef_E_zx * ( H_x[j] - H_x[j-m_y] )  #equation 9 of Tan, Potter paper
    

@cython.cdivision(True)
cpdef double getSource(
    double time, 
    double freq, 
    char* wavetype, 
    double dt
):
    """Gets magnitude of source field in direction perpendicular to propagation 
        of plane wave.

    Args:
        time: float of time at which magnitude of source is calculated.
        freq: float of frequency of introduced wave which determines grid points 
                per wavelength for wave source.
        wavetype: string of type of waveform whose magnitude should be returned.
        dt: double of time upto which wave should exist in a impulse delta pulse.

    Returns:
        sourceMagnitude: double of magnitude of source for requested indices at 
                            current time.
    """

    # Waveforms
    if (strcmp(wavetype, "gaussian") == 0):
        return exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))

    elif (strcmp(wavetype, "gaussiandot") == 0 or strcmp(wavetype, "gaussianprime") == 0):
        return -4.0 * M_PI * M_PI * freq * (time * freq - 1.0
                ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))
    
    elif (strcmp(wavetype, "gaussiandotnorm") == 0):
        return -2.0 * M_PI * (time * freq - 1.0
                ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0))) * exp(0.5)

    elif (strcmp(wavetype, "gaussiandotdot") == 0 or strcmp(wavetype, "gaussiandoubleprime") == 0):
        return (2.0 * M_PI * freq) * (2.0 * M_PI * freq) * (2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)) - 1.0
                ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))
    
    elif (strcmp(wavetype, "gaussiandotdotnorm") == 0):
        return (2.0 * (M_PI *(time * freq - 1.0)) * (M_PI * (time * freq - 1.0)) - 1.0
                ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))
    
    elif (strcmp(wavetype, "ricker") == 0):
        return (1.0 - 2.0 * (M_PI *(time * freq - 1.0)) * (M_PI * (time * freq - 1.0))
                ) * exp(-2.0 * (M_PI * (time * freq - 1.0)) * (M_PI * (time * freq - 1.0)))  # define a Ricker wave source
        
    elif (strcmp(wavetype, "sine") == 0):
        if (time * freq <= 1):
            return sin(2.0 * M_PI * freq * time)
        else:
            return 0.0

    elif (strcmp(wavetype, "contsine") == 0):
        return min(0.25 * time* freq, 1) * sin(2 * M_PI * time* freq)

    elif (strcmp(wavetype, "impulse") == 0):
        if (time < dt):                         # time < dt condition required to do impulsive magnetic dipole
            return 1.0
        else:
            return 0.0


@cython.cdivision(True)
cpdef void calculate1DWaveformValues(
    float_or_double[:, :, ::1] waveformvalues_wholedt, 
    int iterations, 
    int[:] m, 
    double dt, 
    double ds, 
    double c,
    double start, 
    double stop, 
    double freq, 
    char* wavetype
):
    """Precomputes source waveform values so that initialization is faster, 
        if requested.

    Args:
        waveformvalues_wholedt: double array of precomputed waveforms at each 
                                    timestep to initialize magnetic fields.
        iterations: int of number of iterations in simulation.
        m: int array of integer mappings, m_x, m_y, m_z which determine rational 
            angles for assignment of correct element to 3D FDTD grid from 1D 
            representation, last element stores max(m_x, m_y, m_z).
        dt: float of timestep for the simulation.
        ds: float of projection vector for sourcing the plane wave.
        c: float of speed of light in the medium.
        start: float of start time at which source is placed in the TFSF grid.
        stop: float of stop time at which source is removed from TFSF grid.
        freq: float of frequency of introduced wave which determines grid points 
                per wavelength for wave source.
        wavetype: string of type of waveform whose magnitude should be returned.
    """
    
    cdef double time_x, time_y, time_z = 0.0
    cdef Py_ssize_t iteration, r = 0
    
    for iteration in range(iterations):            
        for r in range(m[3]):
            time_x = dt * iteration - (r+ (m[1]+m[2])*0.5) * ds/c
            time_y = dt * iteration - (r+ (m[2]+m[0])*0.5) * ds/c
            time_z = dt * iteration - (r+ (m[0]+m[1])*0.5) * ds/c
            if (dt * iteration >= start and dt * iteration <= stop):
            # Set the time of the waveform evaluation to account for any delay in the start
                waveformvalues_wholedt[iteration, 0, r] = getSource(time_x-start, freq, wavetype, dt)
                waveformvalues_wholedt[iteration, 1, r] = getSource(time_y-start, freq, wavetype, dt)
                waveformvalues_wholedt[iteration, 2, r] = getSource(time_z-start, freq, wavetype, dt)


cpdef void updatePlaneWave(    
    int n, 
    int nthreads,
    float_or_double[:, ::1] H_fields, 
    float_or_double[:, ::1] E_fields, 
    float_or_double[:] updatecoeffsE,
    float_or_double[:] updatecoeffsH,
    float_or_double[:, :, ::1] Ex,
    float_or_double[:, :, ::1] Ey,
    float_or_double[:, :, ::1] Ez,
    float_or_double[:, :, ::1] Hx,
    float_or_double[:, :, ::1] Hy,
    float_or_double[:, :, ::1] Hz, 
    double[:] projections,
    float_or_double[:, ::1] waveformvalues_wholedt,
    int[:] m,
    int[:] corners,
    bint precompute,
    int iteration,
    double dt,
    double ds,
    double c,
    double start, 
    double stop, 
    double freq,
    char* wavetype
):
    initializeMagneticFields(m, H_fields, projections, waveformvalues_wholedt, precompute, iteration, dt, ds, c, start, stop, freq, wavetype)
    updateMagneticFields(n, H_fields, E_fields, updatecoeffsH, m)
    applyTFSFMagnetic(nthreads, Hx, Hy, Hz, E_fields, updatecoeffsH, m, corners)
    applyTFSFElectric(nthreads, Ex, Ey, Ez, H_fields, updatecoeffsE, m, corners)
    updateElectricFields(n, H_fields, E_fields, updatecoeffsE, m)


@cython.cdivision(True)
cdef void takeSnapshot3D(double[:, :, ::1] field, char* filename):
    """Writes fields of plane wave simulation at a particular time step.
    
    Args:
        fields: double array of fields for grid cells over TFSF box at 
                particular indices of TFSF box at particular time step.
        filename: string of file location where fields are to be written.
        
    """

    cdef FILE *fptr = fopen(filename, "wb")
    fwrite(&field[0, 0, 0], sizeof(double), field.size, fptr)
    fclose(fptr)

