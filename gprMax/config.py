# Copyright (C) 2015-2019: The University of Edinburgh
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
from scipy.constants import c
from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0

from gprMax.utilities import get_host_info

# Impedance of free space (Ohms)
z0 = np.sqrt(m0 / e0)

# Setting whether messages and progress bars are printed
messages = True
progressbars = messages

# Store information about host machine
hostinfo = get_host_info()

# Store information about any GPUs as a list of GPU objects
gpus = None

# Copy snapshot data from GPU to CPU during simulation
# N.B. This will happen if the requested snapshots are too large to fit
# on the memory of the GPU. If True this will slow performance significantly
snapsgpu2cpu = False

# Numerical dispersion analysis parameters
# Threshold (dB) down from maximum power (0dB) of main frequency used
# to calculate highest frequency for numerical dispersion analysis
# Maximum allowable percentage physical phase-velocity phase error
# Minimum grid sampling of smallest wavelength for physical wave propagation
numdispersion = {'highestfreqthres': 40, 'maxnumericaldisp': 2, 'mingridsampling': 3}

# Simulation mode - 2D TMx, 2D TMy, 2D TMz, or 3D
mode = None

# Data types
#   Solid and ID arrays use 32-bit integers (0 to 4294967295)
#   Rigid arrays use 8-bit integers (the smallest available type to store true/false)
#   Fractal and dispersive coefficient arrays use complex numbers (complextype)
#                    which are represented as two floats
#   Main field arrays use floats (floattype) and complex numbers (complextype)

# Precision of floattype and complextype: single or double
precision = 'single'

if precision == 'single':
    # For numpy arrays
    floattype = np.float32
    complextype = np.complex64
    # For C (CUDA) arrays
    cudafloattype = 'float'
    cudacomplextype = 'pycuda::complex<float>'

elif precision == 'double':
    # For numpy arrays
    floattype = np.float64
    complextype = np.complex128
    # For C (CUDA) arrays
    cudafloattype = 'double'
    cudacomplextype = 'pycuda::complex<double>'
