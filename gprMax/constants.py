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

# physical constants - try to pull from scipy if available, otherwise use hard-
# coded values so that the library can be imported in minimal environments used
# by some of our unit tests.
try:
    from scipy.constants import c, mu_0 as m0, epsilon_0 as e0
except ImportError:
    # values taken from the CODATA 2018 recommended values
    c = 299792458.0  # speed of light in vacuum (m/s)
    m0 = 4e-7 * np.pi  # vacuum permeability (H/m)
    e0 = 1.0 / (m0 * c**2)  # vacuum permittivity (F/m)

# Impedance of free space (Ohms)
z0 = np.sqrt(m0 / e0)

# Data types:
#   Solid and ID arrays use 32-bit integers (0 to 4294967295)
#   Rigid arrays use 8-bit integers (the smallest available type to store true/false)
#   Fractal and dispersive coefficient arrays use complex numbers (complextype)
#                    which are represented as two floats
#   Main field arrays use floats (floattype) and complex numbers (complextype)

# Single precision
# For numpy arrays
floattype = np.float32
complextype = np.complex64

# For C (CUDA) arrays
cudafloattype = 'float'
cudacomplextype = 'pycuda::complex<float>'

# Double precision
# For numpy arrays
# floattype = np.float64
# complextype = np.complex128

# For C (CUDA) arrays
# cudafloattype = 'double'
# cudacomplextype = 'pycuda::complex<double>'
