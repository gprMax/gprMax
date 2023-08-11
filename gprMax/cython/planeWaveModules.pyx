import numpy as np
cimport cython
from libc.math cimport floor, ceil, round, sqrt, tan, cos, sin, atan2, abs, pow, exp, M_PI
from libc.stdio cimport FILE, fopen, fwrite, fclose
from cython.parallel import prange

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def getProjections(double psi, int[:] m):
    cdef double phi, theta, cos_phi, sin_phi, cos_psi, sin_psi, cos_theta, sin_theta
    cdef double[:, :] projections = np.zeros((2, 3), dtype=np.float64, order='C')

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
@cython.wraparound(False)
@cython.boundscheck(False)
cdef int[:] getPhi(int[:, :] integers, double required_ratio, double tolerance):
    '''
    Method to get the rational angle approximation to phi within the requested tolerance level using
    Farey Fractions to determine a rational number closest to the real number. 
    __________________________
    
    Input parameters:
    --------------------------
        required_ratio, float : tan value of the required phi angle (which would be approximated to a rational number)
        tolerance, float      : acceptable deviation in the tan value of the rational angle from phi
    __________________________
    
    Returns:
    --------------------------
        integers, int array : sequence of the two integers [m_x, m_y]
    '''
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
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double getTanValue(int[:] integers, double[:] dr):
    '''
    Method to return the tan value of the angle approximated to theta given the three integers.
    __________________________
    
    Input parameters:
    --------------------------
        integers, int array : three integers for the rational angle approximation
        dr, double array    : array containing the separation between grid points along the three axes [dx, dy, dz]
    __________________________
    
    Returns:
    --------------------------
        _tanValue, double : tan value of the rationa angle cprrenponding to the integers m_x, m_y, m_z
    '''
    if(integers[2]==0):   #if rational angle==90 degrees
        return 99999.0       #return a large number to avoid division by zero error
    else:
        return sqrt((integers[0]/dr[0])*(integers[0]/dr[0]) + (integers[1]/dr[1])*(integers[1]/dr[1])
                   )/(integers[2]/dr[2])

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef int[:, :] get_mZ(int m_x, int m_y, double theta, double[:] Delta_r):
    cdef double m_z = 0
    m_z = sqrt((m_x/Delta_r[0])*(m_x/Delta_r[0]) + (m_y/Delta_r[1])*(m_y/Delta_r[1]))/(tan(theta)/Delta_r[2]) #get an estimate of the m_z value 
    return np.array([[m_x, m_y, floor(m_z)],
                     [m_x, m_y, round(m_z)],
                     [m_x, m_y, ceil(m_z)]], dtype=np.int32, order='C')   #set up the integer array to search for an appropriate m_z
    

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef int[:] getTheta(int m_x, int m_y, double theta, double Delta_theta, double[:] Delta_r):
    '''
    Method to get the rational angle approximation to theta within the requested tolerance level using
    Binary Search to determine a rational number closest to the real number. 
    __________________________
    
    Input parameters:
    --------------------------
        m_x and m_y, int   : integers approximating the rational angle to the tan value of phi 
        theta, float       : polar angle of the incident plane wave (in radians) to be approximated to a rational angle
        Delta_theta, float : permissible error in the rational angle approximation to theta (in radians)
        Delta_r, float     : projections of the propagation vector along the different coordinate axes
    __________________________
    
    Returns:
    --------------------------
        integers, int array : sequence of the three integers [m_x, m_y, m_z]
    '''
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
@cython.wraparound(False)
@cython.boundscheck(False)
def getIntegerForAngles(double phi, double Delta_phi, double theta, double Delta_theta, double[:] Delta_r):
    '''
    Method to get [m_x, m_y, m_z] to determine the rational angles given phi and theta along with the permissible tolerances 
    __________________________
    
    Input parameters:
    --------------------------
        phi, float         : azimuthal angle of the incident plane wave (in degrees) to be approximated to a rational angle
        Delta_phi, float   : permissible error in the rational angle approximation to phi (in degrees)
        theta, float       : polar angle of the incident plane wave (in degrees) to be approximated to a rational angle
        Delta_theta, float : permissible error in the rational angle approximation to theta (in degrees)
        Delta_r, float     : projections of the propagation vector along the different coordinate axes
    __________________________
    
    Returns:
    --------------------------
        integers, int array : sequence of the three integers [m_x, m_y, m_z]
    '''
    cdef double required_ratio_phi, tolerance_phi = 0.0
    cdef int m_x, m_y, m_z = 0
    quadrants = np.ones(3, dtype=np.int32)
    if(theta>=90):
        quadrants[2] = -1
        theta = 180-theta
    if(phi>=90 and phi<180):
        quadrants[0] = -1
        phi = 180-phi
    elif(phi>=180 and phi<270):
        quadrants[0] = -1
        quadrants[1] = -1
        phi = phi-180
    elif(phi>=270 and phi<360):
        quadrants[1] = -1
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
    return quadrants, np.array([m_x, m_y, m_z, max(m_x, m_y, m_z)], dtype=np.int32)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:, :, :, :, ::1] updateHFields(double[:] coefficients, int n_x, int n_y, int n_z, 
                                        double[:, :, :, :, ::1] fields, int waveID):
    cdef Py_ssize_t i, j, k = 0
    
    cdef double c0 = coefficients[0]
    cdef double c1 = coefficients[1]
    cdef double c2 = coefficients[2]
    cdef double c3 = coefficients[3]
    cdef double c4 = coefficients[4]
    cdef double c5 = coefficients[5]
    cdef double c6 = coefficients[6]
    cdef double c7 = coefficients[7]
    cdef double c8 = coefficients[8]
    
    cdef double[:, :, ::1] e_x = fields[waveID, 0, :, :, :]
    cdef double[:, :, ::1] e_y = fields[waveID, 1, :, :, :]
    cdef double[:, :, ::1] e_z = fields[waveID, 2, :, :, :]
    cdef double[:, :, ::1] h_x = fields[waveID, 3, :, :, :]
    cdef double[:, :, ::1] h_y = fields[waveID, 4, :, :, :]
    cdef double[:, :, ::1] h_z = fields[waveID, 5, :, :, :]
    
    for i in prange(n_x, nogil=True, schedule='static'):
        for j in range(n_y):
            for k in range(n_z):
                h_x[i, j, k] = c0 * h_x[i, j, k] + c1 * (e_y[i, j, k+1] - e_y[i, j, k]
                                               ) - c2 * (e_z[i, j+1, k] - e_z[i, j, k])
                h_y[i, j, k] = c3 * h_y[i, j, k] + c4 * (e_z[i+1, j, k] - e_z[i, j, k]
                                               ) - c5 * (e_x[i, j, k+1] - e_x[i, j, k])
                h_z[i, j, k] = c6 * h_z[i, j, k] + c7 * (e_x[i, j+1, k] - e_x[i, j, k]
                                               ) - c8 * (e_y[i+1, j, k] - e_y[i, j, k])
    
    return fields

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:, :, :, :, ::1] updateEFields(double[:] coefficients, int n_x, int n_y, int n_z, 
                                        double[:, :, :, :, ::1] fields, int waveID):
    cdef Py_ssize_t i, j, k = 0
    
    cdef double c0 = coefficients[0]
    cdef double c1 = coefficients[1]
    cdef double c2 = coefficients[2]
    cdef double c3 = coefficients[3]
    cdef double c4 = coefficients[4]
    cdef double c5 = coefficients[5]
    cdef double c6 = coefficients[6]
    cdef double c7 = coefficients[7]
    cdef double c8 = coefficients[8]
    
    cdef double[:, :, ::1] e_x = fields[waveID, 0, :, :, :]
    cdef double[:, :, ::1] e_y = fields[waveID, 1, :, :, :]
    cdef double[:, :, ::1] e_z = fields[waveID, 2, :, :, :]
    cdef double[:, :, ::1] h_x = fields[waveID, 3, :, :, :]
    cdef double[:, :, ::1] h_y = fields[waveID, 4, :, :, :]
    cdef double[:, :, ::1] h_z = fields[waveID, 5, :, :, :]
    
    for i in prange(n_x, nogil=True, schedule='static'):
        for j in range(1, n_y):
            for k in range(1, n_z):
                e_x[i, j, k] = c0 * e_x[i, j, k] + c1 * (h_z[i, j, k] - h_z[i, j-1, k]
                                               ) - c2 * (h_y[i, j, k] - h_y[i, j, k-1])
    for i in prange(1, n_x, nogil=True, schedule='static'):
        for j in range(n_y):
            for k in range(1, n_z):
                e_y[i, j, k] = c3 * e_y[i, j, k] + c4 * (h_x[i, j, k] - h_x[i, j, k-1]
                                               ) - c5 * (h_z[i, j, k] - h_z[i-1, j, k])
    for i in prange(1, n_x, nogil=True, schedule='static'):
        for j in range(1, n_y):
            for k in range(n_z):
                e_z[i, j, k] = c6 * e_z[i, j, k] + c7 * (h_y[i, j, k] - h_y[i-1, j, k]
                                               ) - c8 * (h_x[i, j, k] - h_x[i, j-1, k])
    return fields

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:, :, :, ::1] implementABC(double[:, :, :, ::1] face_fields, double abccoef, 
                                  int n_x, int n_y, int n_z, double[:, :, :, :, ::1] fields, int waveID):
    cdef Py_ssize_t i, j, k = 0
    
    cdef double[:, :, ::1] e_x = fields[waveID, 0, :, :, :]
    cdef double[:, :, ::1] e_y = fields[waveID, 1, :, :, :]
    cdef double[:, :, ::1] e_z = fields[waveID, 2, :, :, :]
    
    #implement ABC at x0
    for j in prange(n_y-1, nogil=True, schedule='static'):
        for k in range(n_z):
            e_y[0, j, k] = face_fields[waveID, 0, j, k] + abccoef*(e_y[1, j, k] - e_y[0, j, k])
            face_fields[waveID, 0, j, k] = e_y[1, j, k]
    for j in prange(n_y, nogil=True, schedule='static'):
        for k in range(n_z-1):
            e_z[0, j, k] = face_fields[waveID, 1, j, k] + abccoef*(e_z[1, j, k] - e_z[0, j, k])
            face_fields[waveID, 1, j, k] = e_z[1, j ,k]
    
    #implement ABC at x1
    for j in prange(n_y-1, nogil=True, schedule='static'):
        for k in range(n_z):
            e_y[n_x, j, k] = face_fields[waveID, 2, j, k] + abccoef*(e_y[n_x-1, j, k] - e_y[n_x, j, k])
            face_fields[waveID, 2, j, k] = e_y[n_x-1, j, k]
    for j in prange(n_y, nogil=True, schedule='static'):
        for k in range(n_z-1):
            e_z[n_x, j, k] = face_fields[waveID, 3, j, k] + abccoef*(e_z[n_x-1, j, k] - e_z[n_x, j, k])
            face_fields[waveID, 3, j, k] = e_z[n_x-1, j, k]
    
    #implement ABC at y0
    for i in prange(n_x-1, nogil=True, schedule='static'):
        for k in range(n_z):
            e_x[i, 0, k] = face_fields[waveID, 4, i, k] + abccoef*(e_x[i, 1, k] - e_x[i, 0, k])
            face_fields[waveID, 4, i, k] = e_x[i, 1, k]
    for i in prange(n_x, nogil=True, schedule='static'):
        for k in range(n_z-1):
            e_z[i, 0, k] = face_fields[waveID, 5, i, k] + abccoef*(e_z[i, 1, k] - e_z[i, 0, k])
            face_fields[waveID, 5, i, k] = e_z[i, 1, k]
    
    #implement ABC at y1
    for i in prange(n_x-1, nogil=True, schedule='static'):
        for k in range(n_z):
            e_x[i, n_y, k] = face_fields[waveID, 6, i, k] + abccoef*(e_x[i, n_y-1, k] - e_x[i, n_y, k])
            face_fields[waveID, 6, i, k] = e_x[i, n_y-1, k]
    for i in prange(n_x, nogil=True, schedule='static'):
        for k in range(n_z-1):
            e_z[i, n_y, k] = face_fields[waveID, 7, i, k] + abccoef*(e_z[i, n_y-1, k] - e_z[i, n_y, k])
            face_fields[waveID, 7, i, k] = e_z[i, n_y-1, k]
    
    #implement ABC at z0
    for i in prange(n_x-1, nogil=True, schedule='static'):
        for j in range(n_y):
            e_x[i, j, 0] = face_fields[waveID, 8, i, j] + abccoef*(e_x[i, j, 1] - e_x[i, j, 0])
            face_fields[waveID, 8, i, j] = e_x[i, j, 1]
    for i in prange(n_x, nogil=True, schedule='static'):
        for j in range(n_y-1):
            e_y[i, j, 0] = face_fields[waveID, 9, i, j] + abccoef*(e_y[i, j, 1] - e_y[i, j, 0])
            face_fields[waveID, 9, i, j] = e_y[i, j, 1]
    
    #implement ABC at z1
    for i in prange(n_x-1, nogil=True, schedule='static'):
        for j in range(n_y):
            e_x[i, j, n_z] = face_fields[waveID, 10, i, j] + abccoef*(e_x[i, j, n_z-1] - e_x[i, j, n_z])
            face_fields[waveID, 10, i, j] = e_x[i, j, n_z-1]
    for i in prange(n_x, nogil=True, schedule='static'):
        for j in range(n_y-1):
            e_y[i, j, n_z] = face_fields[waveID, 11, i, j] + abccoef*(e_y[i, j, n_z-1] - e_y[i, j, n_z])
            face_fields[waveID, 11, i, j] = e_y[i, j, n_z-1]

    return face_fields

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:, :, :, :, ::1] applyTFSFMagnetic(double[:] coefficients, double[:, :] e1D, int[:] m,
                                          double[:, :, :, :, ::1]fields, int[:, :] corners, int waveID):

    cdef Py_ssize_t i, j, k = 0
    cdef int x_start = corners[0, 0]
    cdef int y_start = corners[0, 1]
    cdef int z_start = corners[0, 2]
    cdef int x_stop = corners[1, 0]
    cdef int y_stop = corners[1, 1]
    cdef int z_stop = corners[1, 2]
    # Precompute index values
    cdef int index = 0
    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]
    
    cdef double[:, :, ::1] h_x = fields[waveID, 3, :, :, :]
    cdef double[:, :, ::1] h_y = fields[waveID, 4, :, :, :]
    cdef double[:, :, ::1] h_z = fields[waveID, 5, :, :, :]
    
    #**** constant x faces -- scattered-field nodes ****
    i = x_start
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static'):
        for k in range(z_start, z_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Hy at firstX-1/2 by subtracting Ez_inc
            h_y[i-1, j, k] -= coefficients[5] * e1D[2, index] 
    
    for j in prange(y_start, y_stop, nogil=True, schedule='static'):
        for k in range(z_start, z_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Hz at firstX-1/2 by adding Ey_inc
            h_z[i-1, j, k] += coefficients[8] * e1D[1, index]

    i = x_stop
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static'):
        for k in range(z_start, z_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Hy at lastX+1/2 by adding Ez_inc
            h_y[i, j, k] += coefficients[5] * e1D[2, index]    
    
    for j in prange(y_start, y_stop, nogil=True, schedule='static'):
        for k in range(z_start, z_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Hz at lastX+1/2 by subtractinging Ey_inc
            h_z[i, j, k] -= coefficients[8] * e1D[1, index]            
    
    #**** constant y faces -- scattered-field nodes ****
    j = y_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static'):
        for k in range(z_start, z_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Hx at firstY-1/2 by adding Ez_inc
            h_x[i, j-1, k] += coefficients[2] * e1D[2, index]
    
    for i in prange(x_start, x_stop, nogil=True, schedule='static'):
        for k in range(z_start, z_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Hz at firstY-1/2 by subtracting Ex_inc
            h_z[i, j-1, k] -= coefficients[7] * e1D[0, index]

    j = y_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static'):
        for k in range(z_start, z_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Hx at lastY+1/2 by subtracting Ez_inc
            h_x[i, j, k] -= coefficients[2] * e1D[2, index]
    
    for i in prange(x_start, x_stop, nogil=True, schedule='static'):
        for k in range(z_start, z_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Hz at lastY-1/2 by adding Ex_inc
            h_z[i, j, k] += coefficients[7] * e1D[0,  index]

    #**** constant z faces -- scattered-field nodes ****
    k = z_start
    for i in prange(x_start, x_stop, nogil=True, schedule='static'):
        for j in range(y_start, y_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Hy at firstZ-1/2 by adding Ex_inc
            h_y[i, j, k-1] += coefficients[5] * e1D[0,  index]
    
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static'):
        for j in range(y_start, y_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Hx at firstZ-1/2 by subtracting Ey_inc
            h_x[i, j, k-1] -= coefficients[1] * e1D[1, index]

    k = z_stop
    for i in prange(x_start, x_stop, nogil=True, schedule='static'):
        for j in range(y_start, y_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Hy at firstZ-1/2 by subtracting Ex_inc
            h_y[i, j, k] -= coefficients[5] * e1D[0, index]
    
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static'):
        for j in range(y_start, y_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Hx at lastZ+1/2 by adding Ey_inc
            h_x[i, j, k] += coefficients[1] * e1D[1,  index]
    return fields

@cython.wraparound(False)
@cython.boundscheck(False)    
cdef double[:, :, :, :, ::1] applyTFSFElectric(double[:] coefficients, double[:, :] h1D, int[:] m,
                                          double[:, :, :, :, ::1] fields, int[:, :] corners, int waveID):
    
    cdef Py_ssize_t i, j, k = 0
    cdef int x_start = corners[0, 0]
    cdef int y_start = corners[0, 1]
    cdef int z_start = corners[0, 2]
    cdef int x_stop = corners[1, 0]
    cdef int y_stop = corners[1, 1]
    cdef int z_stop = corners[1, 2]
    # Precompute index values
    cdef int index = 0
    cdef int m_x = m[0]
    cdef int m_y = m[1]
    cdef int m_z = m[2]
    
    cdef double[:, :, ::1] e_x = fields[waveID, 0, :, :, :]
    cdef double[:, :, ::1] e_y = fields[waveID, 1, :, :, :]
    cdef double[:, :, ::1] e_z = fields[waveID, 2, :, :, :]
    
    #**** constant x faces -- total-field nodes ****/
    i = x_start
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static'):
        for k in range(z_start, z_stop):
            index = m_x * (i-1) + m_y * j + m_z * k
            #correct Ez at firstX face by subtracting Hy_inc
            e_z[i, j, k] -= coefficients[7] * h1D[1, index]
    
    for j in prange(y_start, y_stop, nogil=True, schedule='static'):
        for k in range(z_start, z_stop+1):
            index = m_x * (i-1) + m_y * j + m_z * k
            #correct Ey at firstX face by adding Hz_inc
            e_y[i, j, k] += coefficients[4] * h1D[2, index]
    
    i = x_stop
    for j in prange(y_start, y_stop+1, nogil=True, schedule='static'):
        for k in range(z_start, z_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Ez at lastX face by adding Hy_inc
            e_z[i, j, k] += coefficients[7] * h1D[1, index]
    
    for j in prange(y_start, y_stop, nogil=True, schedule='static'):
        for k in range(z_start, z_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Ey at lastX face by subtracting Hz_inc
            e_y[i, j, k] -= coefficients[4] * h1D[2, index]

    #**** constant y faces -- total-field nodes ****/
    j = y_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static'):
        for k in range(z_start, z_stop):
            index = m_x * i + m_y * (j-1) + m_z * k
            #correct Ez at firstY face by adding Hx_inc
            e_z[i, j, k] += coefficients[8] * h1D[0, index]
    
    for i in prange(x_start, x_stop, nogil=True, schedule='static'):
        for k in range(z_start, z_stop+1):
            index = m_x * i + m_y * (j-1) + m_z * k
            #correct Ex at firstY face by subtracting Hz_inc
            e_x[i, j, k] -= coefficients[1] * h1D[2,  index]
    
    j = y_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static'):
        for k in range(z_start, z_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Ez at lastY face by subtracting Hx_inc
            e_z[i, j, k] -= coefficients[8] * h1D[0,  index]
    
    for i in prange(x_start, x_stop, nogil=True, schedule='static'):
        for k in range(z_start, z_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Ex at lastY face by adding Hz_inc
            e_x[i, j, k] += coefficients[1] * h1D[2,  index]

    #**** constant z faces -- total-field nodes ****/
    k = z_start
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static'):
        for j in range(y_start, y_stop):
            index = m_x * i + m_y * j + m_z * (k-1)
            #correct Ey at firstZ face by subtracting Hx_inc
            e_y[i, j, k] -= coefficients[4] * h1D[0,  index]
    
    for i in prange(x_start, x_stop, nogil=True, schedule='static'):
        for j in range(y_start, y_stop+1):
            index = m_x * i + m_y * j + m_z * (k-1)
            #correct Ex at firstZ face by adding Hy_inc
            e_x[i, j, k] += coefficients[2] * h1D[1, index]
    
    k = z_stop
    for i in prange(x_start, x_stop+1, nogil=True, schedule='static'):
        for j in range(y_start, y_stop):
            index = m_x * i + m_y * j + m_z * k
            #correct Ey at lastZ face by adding Hx_inc
            e_y[i, j, k] += coefficients[4] * h1D[0, index]
    
    for i in prange(x_start, x_stop, nogil=True, schedule='static'):
        for j in range(y_start, y_stop+1):
            index = m_x * i + m_y * j + m_z * k
            #correct Ex at lastZ face by subtracting Hy_inc
            e_x[i, j, k] -= coefficients[2] * h1D[1, index]
    return fields

'''
    Method to update the magnetic fields for the next time step using
    Equation 8 of DOI: 10.1109/LAWP.2009.2016851
        __________________________

        Input parameters:
        --------------------------
            n , int                      : stores the spatial length of the DPW array so that each length grid cell is updated when the method updateMagneticFields() is called
            H_coefficients, double array : stores the coefficients of the fields in the update equation for the magnetic field
            H_fields, double array       : stores the magnetic fields of the DPW till temporal index time
            E_fields, double array       : stores the electric fields of the DPW till temporal index time
            time, int                    : time index storing the current axis number which would be updated for the H_fields
        __________________________

        Returns:
        --------------------------
            H_fields, double array       : magnetic field array with the axis entry for the current time added

'''
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:, ::1] updateMagneticFields(int n, double[:] H_coefficients, double[:, ::1] H_fields, 
                                         double[:, ::1] E_fields, int dimensions, int[:] m):
    
    cdef Py_ssize_t i, j = 0
    cdef int dim_mod1, dim_mod2 = 0
    for i in prange(dimensions, nogil=True, schedule='static'):  #loop to update each component of the magnetic field
        dim_mod1 = (i+1) % dimensions
        dim_mod2 = (i+2) % dimensions
        for j in range(m[dimensions], n-m[dimensions]):  #loop to update the magnetic field at each spatial index
            H_fields[i, j] = H_coefficients[3*i] * H_fields[i, j] + H_coefficients[3*i+1] * (
                    E_fields[dim_mod1, j+m[dim_mod2]] - E_fields[dim_mod1, j]) - H_coefficients[3*i+2] * (
                    E_fields[dim_mod2, j+m[dim_mod1]] - E_fields[dim_mod2, j])     #equation 8 of Tan, Potter paper
    return H_fields

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:, ::1] updateElectricFields(int n, double[:] E_coefficients, double[:, ::1] H_fields, 
                                         double[:, ::1] E_fields, int dimensions, int[:] m):
    '''
    Method to update the electric fields for the next time step using
    Equation 9 of DOI: 10.1109/LAWP.2009.2016851
        __________________________

        Input parameters:
        --------------------------
            n , int                      : stores the spatial length of the DPW array so that each length grid cell is updated when the method updateMagneticFields() is called
            E_coefficients, double array : stores the coefficients of the fields in the update equation for the electric field
            H_fields, double array       : stores the magnetic fields of the DPW till temporal index time
            E_fields, double array       : stores the electric fields of the DPW till temporal index time
            time, int                    : time index storing the current axis number which would be updated for the H_fields
        __________________________

        Returns:
        --------------------------
            E_fields, double array       : electric field array with the axis entry for the current time added

    '''
    cdef Py_ssize_t i, j = 0
    cdef int dim_mod1, dim_mod2 = 0
    for i in prange(dimensions, nogil=True, schedule='static'):  #loop to update each component of the electric field
        dim_mod1 = (i+1) % dimensions
        dim_mod2 = (i+2) % dimensions
        for j in range(m[dimensions], n-m[dimensions]):   #loop to update the electric field at each spatial index 
            E_fields[i, j] = E_coefficients[3*i] * E_fields[i, j] + E_coefficients[3*i+1] * (
                    H_fields[dim_mod2, j] - H_fields[dim_mod2, j-m[dim_mod1]]) - E_coefficients[3*i+2] * ( 
                    H_fields[dim_mod1, j] - H_fields[dim_mod1, j-m[dim_mod2]])  #equation 9 of Tan, Potter paper
    return E_fields

@cython.cdivision(True)
cdef inline double getSource(double time, double ppw):
    '''
    Method to get the magnitude of the source field in the direction perpendicular to the propagation of the plane wave
        __________________________

        Input parameters:
        --------------------------
            time, float : time at which the magnitude of the source is calculated
            ppw, int    : points per wavelength for the wave source

        __________________________

        Returns:
        --------------------------
            sourceMagnitude, double array : magnitude of the source for the requested indices at the current time
    '''
    return (1.0-2.0*(M_PI*(time/ppw - 1.0))*(M_PI*(time/ppw - 1.0))
           )*exp(-(M_PI*(time/ppw - 1.0))*(M_PI*(time/ppw - 1.0)))  # define a Ricker wave source

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double[:, :, :, :, ::1] superImposeFields(planeWaves, int n_x, int n_y, int n_z, int dimensions,
                                               double[:, :, :, :, ::1] fields, int iterate):
    cdef Py_ssize_t i, j, k, l, n = 0
    cdef int x_start, y_start, z_start
    for n in range(iterate):
        x_start = 0
        y_start = 0
        z_start = 0
        if(planeWaves[n].directions[0]==-1):
            x_start = n_x
        if(planeWaves[n].directions[1]==-1):
            y_start = n_y
        if(planeWaves[n].directions[2]==-1):
            z_start = n_z
        for i in range(n_x+1):
            for j in range(n_y+1):
                for k in range(n_z+1):
                    for l in range(dimensions):
                        fields[iterate, l, i, j, k] += fields[n, l, abs(x_start-i), abs(y_start-j), abs(z_start-k)]
                    
    return fields
'''
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void takeSnapshot3D(double[:, :, ::1] field, char* filename):
    cdef FILE *fptr = fopen(filename, "wb")
    cdef Py_ssize_t n_x = field.shape[0]
    cdef double[:, ::1] slice_view
    cdef Py_ssize_t i = 0
    if fptr is NULL:
        raise ValueError("Failed to open the file.")
    else:
        for i in range(n_x):
            slice_view = field[i, ...]
            fwrite(&slice_view[0, 0], sizeof(double), slice_view.size, fptr)
    fclose(fptr)
'''
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void takeSnapshot3D(double[:, :, ::1] field, char* filename):
    cdef FILE *fptr = fopen(filename, "wb")
    fwrite(&field[0, 0, 0], sizeof(double), field.size, fptr)
    fclose(fptr)
    
def getGridFields(planeWave, double[:] C, double[:] D, int snapshot, 
                  int n_x, int n_y, int n_z, double[:, :, :, :, ::1] fields, int[:, ::1] corners,
                  int time_duration, double[:, :, :, ::1] face_fields,
                  double abccoef, double dt, int noOfWaves, double c, double ppw, int dimensions):
    
    cdef double real_time = 0.0   # initialize the real time to zero
    # Py_ssize_t is the proper C type for Python array indices
    cdef Py_ssize_t i, j, r = 0      #E and H are the one-dimensional waves
    for i in range(time_duration):
        for j in range(noOfWaves):
            for k in range(dimensions):
                for l in range(planeWave[j].m[dimensions]):                        #loop to assign the source values of magnetic field to the first few gridpoints
                    planeWave[j].H_fields[k, l] = planeWave[j].projections[k]*getSource(real_time - (l+(
                                planeWave[j].m[(k+1)%dimensions]+planeWave[j].m[(k+2)%dimensions]
                                                        )*0.5)*planeWave[j].ds/c, ppw)
                
            planeWave[j].H_fields = updateMagneticFields(planeWave[j].length, D, planeWave[j].H_fields, 
                                                planeWave[j].E_fields, dimensions, planeWave[j].m)  #update the magnetic fields in the rest of the DPW grid
        
            fields = updateHFields(D, n_x, n_y, n_z, fields, j)
            fields = applyTFSFMagnetic(D, planeWave[j].E_fields, planeWave[j].m, fields, corners, j)
            fields = applyTFSFElectric(C, planeWave[j].H_fields, planeWave[j].m, fields, corners, j)
            fields = updateEFields(C, n_x, n_y, n_z, fields, j)    
                  
            planeWave[j].E_fields = updateElectricFields(planeWave[j].length, C, planeWave[j].H_fields, 
                                                planeWave[j].E_fields, dimensions, planeWave[j].m)   #update the electric fields in the rest of the DPW grid

            face_fields = implementABC(face_fields, abccoef, n_x, n_y, n_z, fields, j)
        
        real_time += dt  #take a half time step because the magnetic field and electric field grids are staggered in time
          
        if (i % snapshot == 0):
            fields = superImposeFields(planeWave, n_x, n_y, n_z, dimensions, np.asarray(fields), noOfWaves)
            takeSnapshot3D(fields[noOfWaves, 0, :, :, :], f'./snapshots/electric_2{i}.dat'.encode('UTF-8'))
            fields[noOfWaves, :, :, :, :] = 0

