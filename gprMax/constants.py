# Copyright (C) 2015-2016: The University of Edinburgh
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

# Data types:
#   Solid and ID arrays use 32-bit integers (0 to 4294967295)
#   Rigid arrays use 8-bit integers (the smallest available type to store true/false)
#   Fractal and dispersive coefficient arrays use complex numbers (complextype) which are represented as two floats
#   Main field arrays use floats (floattype) and complex numbers (complextype)

floattype = np.float32
complextype = np.complex64

# Speed of light in vacuum (m/s)
c = 2.9979245e8

# Permittivity of free space (F/m)
e0 = 8.854187e-12

# Permeability of free space (H/m)
m0 = 1.256637e-6

# Impedance of free space (Ohms)
z0 = 376.7303134