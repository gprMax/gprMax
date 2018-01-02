# Copyright (C) 2015-2018: The University of Edinburgh
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

from gprMax.constants import floattype
from gprMax.constants import complextype
from gprMax.fractals_generate import generate_fractal2D
from gprMax.fractals_generate import generate_fractal3D
from gprMax.utilities import round_value

np.seterr(divide='raise')


class FractalSurface(object):
    """Fractal surfaces."""

    surfaceIDs = ['xminus', 'xplus', 'yminus', 'yplus', 'zminus', 'zplus']

    def __init__(self, xs, xf, ys, yf, zs, zf, dimension):
        """
        Args:
            xs, xf, ys, yf, zs, zf (float): Extent of the fractal surface (one pair of
                            coordinates must be equal to correctly define a surface).
            dimension (float): Fractal dimension that controls the fractal distribution.
        """

        self.ID = None
        self.surfaceID = None
        self.xs = xs
        self.xf = xf
        self.ys = ys
        self.yf = yf
        self.zs = zs
        self.zf = zf
        self.nx = xf - xs
        self.ny = yf - ys
        self.nz = zf - zs
        self.seed = None
        self.dimension = dimension
        # Constant related to fractal dimension from: http://dx.doi.org/10.1017/CBO9781139174695
        self.b = -(2 * self.dimension - 7) / 2
        self.weighting = np.array([1, 1], dtype=np.float64)
        self.fractalrange = (0, 0)
        self.filldepth = 0
        self.grass = []

    def generate_fractal_surface(self, G):
        """Generate a 2D array with a fractal distribution.
            
        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if self.xs == self.xf:
            surfacedims = (self.ny, self.nz)
        elif self.ys == self.yf:
            surfacedims = (self.nx, self.nz)
        elif self.zs == self.zf:
            surfacedims = (self.nx, self.ny)

        self.fractalsurface = np.zeros(surfacedims, dtype=complextype)

        # Positional vector at centre of array, scaled by weighting
        v1 = np.array([self.weighting[0] * (surfacedims[0]) / 2, self.weighting[1] * (surfacedims[1]) / 2])

        # 2D array of random numbers to be convolved with the fractal function
        R = np.random.RandomState(self.seed)
        A = R.randn(surfacedims[0], surfacedims[1])

        # 2D FFT
        A = np.fft.fftn(A)
        # Shift the zero frequency component to the centre of the array
        A = np.fft.fftshift(A)

        # Generate fractal
        generate_fractal2D(surfacedims[0], surfacedims[1], G.nthreads, self.b, self.weighting, v1, A, self.fractalsurface)

        # Shift the zero frequency component to start of the array
        self.fractalsurface = np.fft.ifftshift(self.fractalsurface)
        # Take the real part (numerical errors can give rise to an imaginary part) of the IFFT
        self.fractalsurface = np.real(np.fft.ifftn(self.fractalsurface))
        # Scale the fractal volume according to requested range
        fractalmin = np.amin(self.fractalsurface)
        fractalmax = np.amax(self.fractalsurface)
        fractalrange = fractalmax - fractalmin
        self.fractalsurface = self.fractalsurface * ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange) \
                + self.fractalrange[0] - ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange) * fractalmin


class FractalVolume(object):
    """Fractal volumes."""

    def __init__(self, xs, xf, ys, yf, zs, zf, dimension):
        """
        Args:
            xs, xf, ys, yf, zs, zf (float): Extent of the fractal volume.
            dimension (float): Fractal dimension that controls the fractal distribution.
        """

        self.ID = None
        self.operatingonID = None
        self.xs = xs
        self.xf = xf
        self.ys = ys
        self.yf = yf
        self.zs = zs
        self.zf = zf
        self.nx = xf - xs
        self.ny = yf - ys
        self.nz = zf - zs
        self.averaging = False
        self.seed = None
        self.dimension = dimension
        # Constant related to fractal dimension from: http://dx.doi.org/10.1017/CBO9781139174695
        self.b = -(2 * self.dimension - 7) / 2
        self.weighting = np.array([1, 1, 1], dtype=np.float64)
        self.nbins = 0
        self.fractalsurfaces = []

    def generate_fractal_volume(self, G):
        """Generate a 3D volume with a fractal distribution.
            
        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        # Scale filter according to size of fractal volume
        if self.nx == 1:
            filterscaling = np.amin(np.array([self.ny, self.nz])) / np.array([self.ny, self.nz])
            filterscaling = np.insert(filterscaling, 0, 1)
        elif self.ny == 1:
            filterscaling = np.amin(np.array([self.nx, self.nz])) / np.array([self.nx, self.nz])
            filterscaling = np.insert(filterscaling, 1, 1)
        elif self.nz == 1:
            filterscaling = np.amin(np.array([self.nx, self.ny])) / np.array([self.nx, self.ny])
            filterscaling = np.insert(filterscaling, 2, 1)
        else:
            filterscaling = np.amin(np.array([self.nx, self.ny, self.nz])) / np.array([self.nx, self.ny, self.nz])

        # Adjust weighting to account for filter scaling
        self.weighting = np.multiply(self.weighting, filterscaling)

        self.fractalvolume = np.zeros((self.nx, self.ny, self.nz), dtype=complextype)

        # Positional vector at centre of array, scaled by weighting
        v1 = np.array([self.weighting[0] * self.nx / 2, self.weighting[1] * self.ny / 2, self.weighting[2] * self.nz / 2])

        # 3D array of random numbers to be convolved with the fractal function
        R = np.random.RandomState(self.seed)
        A = R.randn(self.nx, self.ny, self.nz)

        # 3D FFT
        A = np.fft.fftn(A)
        # Shift the zero frequency component to the centre of the array
        A = np.fft.fftshift(A)

        # Generate fractal
        generate_fractal3D(self.nx, self.ny, self.nz, G.nthreads, self.b, self.weighting, v1, A, self.fractalvolume)

        # Shift the zero frequency component to the start of the array
        self.fractalvolume = np.fft.ifftshift(self.fractalvolume)
        # Take the real part (numerical errors can give rise to an imaginary part) of the IFFT
        self.fractalvolume = np.real(np.fft.ifftn(self.fractalvolume))
        # Bin fractal values
        bins = np.linspace(np.amin(self.fractalvolume), np.amax(self.fractalvolume), self.nbins)
        for j in range(self.ny):
            for k in range(self.nz):
                self.fractalvolume[:, j, k] = np.digitize(self.fractalvolume[:, j, k], bins, right=True)

    def generate_volume_mask(self):
        """
        Generate a 3D volume to use as a mask for adding rough surfaces, water and grass/roots.
        Zero signifies the mask is not set, one signifies the mask is set."""

        self.mask = np.zeros((self.nx, self.ny, self.nz), dtype=np.int8)
        maskxs = self.originalxs - self.xs
        maskxf = (self.originalxf - self.originalxs) + maskxs
        maskys = self.originalys - self.ys
        maskyf = (self.originalyf - self.originalys) + maskys
        maskzs = self.originalzs - self.zs
        maskzf = (self.originalzf - self.originalzs) + maskzs
        self.mask[maskxs:maskxf, maskys:maskyf, maskzs:maskzf] = 1


class Grass(object):
    """Geometry information for blades of grass."""

    def __init__(self, numblades):
        """
        Args:
            numblades (int): Number of blades of grass.
        """

        self.numblades = numblades
        self.geometryparams = np.zeros((self.numblades, 6), dtype=floattype)
        self.seed = None

        # Randomly defined parameters that will be used to calculate geometry
        self.R1 = np.random.RandomState(self.seed)
        self.R2 = np.random.RandomState(self.seed)
        self.R3 = np.random.RandomState(self.seed)
        self.R4 = np.random.RandomState(self.seed)
        self.R5 = np.random.RandomState(self.seed)
        self.R6 = np.random.RandomState(self.seed)

        for i in range(self.numblades):
            self.geometryparams[i, 0] = 10 + 20 * self.R1.random_sample()
            self.geometryparams[i, 1] = 10 + 20 * self.R2.random_sample()
            self.geometryparams[i, 2] = self.R3.choice([-1, 1])
            self.geometryparams[i, 3] = self.R4.choice([-1, 1])

    def calculate_blade_geometry(self, blade, height):
        """Calculates the x and y coordinates for a given height of grass blade.

        Args:
            blade (int): Numeric ID of grass blade.
            height (float): Height of grass blade.

        Returns:
            x, y (float): x and y coordinates of grass blade.
        """

        x = self.geometryparams[blade, 2] * (height / self.geometryparams[blade, 0]) * (height / self.geometryparams[blade, 0])
        y = self.geometryparams[blade, 3] * (height / self.geometryparams[blade, 1]) * (height / self.geometryparams[blade, 1])
        x = round_value(x)
        y = round_value(y)

        return x, y

    def calculate_root_geometry(self, root, depth):
        """Calculates the x and y coordinates for a given depth of grass root.

        Args:
            root (int): Numeric ID of grass root.
            depth (float): Depth of grass root.

        Returns:
            x, y (float): x and y coordinates of grass root.
        """

        self.geometryparams[root, 4] += -1 + 2 * self.R5.random_sample()
        self.geometryparams[root, 5] += -1 + 2 * self.R6.random_sample()
        x = round(self.geometryparams[root, 4])
        y = round(self.geometryparams[root, 5])

        return x, y
