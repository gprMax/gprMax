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
from scipy import fftpack

from gprMax.constants import floattype
from gprMax.fractals_generate_ext import generate_fractal2D
from gprMax.fractals_generate_ext import generate_fractal3D
from gprMax.utilities import round_value

np.seterr(divide='raise')


class FractalSurface(object):
    """Fractal surfaces."""

    surfaceIDs = ['xminus', 'xplus', 'yminus', 'yplus', 'zminus', 'zplus']

    def __init__(self, xs, xf, ys, yf, zs, zf, dimension, seed):
        """
        Args:
            xs, xf, ys, yf, zs, zf (float): Extent of the fractal surface (one pair of
                            coordinates must be equal to correctly define a surface).
            dimension (float): Fractal dimension that controls the fractal distribution.
            seed (int): Seed value for random number generator.
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
        self.seed = seed
        self.dimension = dimension # Fractal dimension from: http://dx.doi.org/10.1017/CBO9781139174695
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

        self.fractalsurface = np.zeros(surfacedims, dtype=np.complex128)

        # Positional vector at centre of array, scaled by weighting
        v1 = np.array([self.weighting[0] * (surfacedims[0]) / 2, self.weighting[1] * (surfacedims[1]) / 2])

        # 2D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)
        A = rng.standard_normal(size=(surfacedims[0], surfacedims[1]))

        # 2D FFT
        A = fftpack.fftn(A)
        # Shift the zero frequency component to the centre of the array
        A = fftpack.fftshift(A)

        # Generate fractal
        generate_fractal2D(surfacedims[0], surfacedims[1], G.nthreads, self.dimension, self.weighting, v1, A, self.fractalsurface)

        # Shift the zero frequency component to start of the array
        self.fractalsurface = fftpack.ifftshift(self.fractalsurface)
        # Set DC component of FFT to zero
        self.fractalsurface[0, 0] = 0
        # Take the real part (numerical errors can give rise to an imaginary part)
        #  of the IFFT, and convert type to floattype. N.B calculation of fractals
        # must always be carried out at double precision, i.e. float64, complex128
        self.fractalsurface = np.real(fftpack.ifftn(self.fractalsurface)).astype(floattype, copy=False)
        # Scale the fractal volume according to requested range
        fractalmin = np.amin(self.fractalsurface)
        fractalmax = np.amax(self.fractalsurface)
        fractalrange = fractalmax - fractalmin
        self.fractalsurface = (self.fractalsurface * ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange) 
                               + self.fractalrange[0] - ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange) 
                               * fractalmin)


class FractalVolume(object):
    """Fractal volumes."""

    def __init__(self, xs, xf, ys, yf, zs, zf, dimension, seed):
        """
        Args:
            xs, xf, ys, yf, zs, zf (float): Extent of the fractal volume.
            dimension (float): Fractal dimension that controls the fractal distribution.
            seed (int): Seed value for random number generator.
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
        self.seed = seed
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

        self.fractalvolume = np.zeros((self.nx, self.ny, self.nz), dtype=np.complex128)

        # Positional vector at centre of array, scaled by weighting
        v1 = np.array([self.weighting[0] * self.nx / 2, self.weighting[1] * self.ny / 2, self.weighting[2] * self.nz / 2])

        # 3D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)
        A = rng.standard_normal(size=(self.nx, self.ny, self.nz))

        # 3D FFT
        A = fftpack.fftn(A)
        # Shift the zero frequency component to the centre of the array
        A = fftpack.fftshift(A)

        # Generate fractal
        generate_fractal3D(self.nx, self.ny, self.nz, G.nthreads, self.dimension, self.weighting, v1, A, self.fractalvolume)

        # Shift the zero frequency component to the start of the array
        self.fractalvolume = fftpack.ifftshift(self.fractalvolume)
        # Set DC component of FFT to zero
        self.fractalvolume[0, 0, 0] = 0
        # Take the real part (numerical errors can give rise to an imaginary part) 
        # of the IFFT, and convert type to floattype. N.B calculation of fractals
        # must always be carried out at double precision, i.e. float64, complex128
        self.fractalvolume = np.real(fftpack.ifftn(self.fractalvolume)).astype(floattype, copy=False)

        # Bin fractal values
        bins = np.linspace(np.amin(self.fractalvolume), np.amax(self.fractalvolume), self.nbins)
        for j in range(self.ny):
            for k in range(self.nz):
                self.fractalvolume[:, j, k] = np.digitize(self.fractalvolume[:, j, k], bins, right=True)

    def generate_volume_mask(self):
        """Generate a 3D volume to use as a mask for adding rough surfaces, 
            water and grass/roots. Zero signifies the mask is not set, one 
            signifies the mask is set.
        """

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

    def __init__(self, numblades, seed):
        """
        Args:
            numblades (int): Number of blades of grass.
            seed (int): Seed value for random number generator.
        """

        self.numblades = numblades
        self.geometryparams = np.zeros((self.numblades, 6), dtype=floattype)
        self.seed = seed

    def set_geometry_parameters(self):
        """Sets randomly defined parameters that will be used to calculate 
            blade and root geometries.
        """

        self.R1 = np.random.default_rng(seed=self.seed)
        self.R2 = np.random.default_rng(seed=self.seed)
        self.R3 = np.random.default_rng(seed=self.seed)
        self.R4 = np.random.default_rng(seed=self.seed)
        self.R5 = np.random.default_rng(seed=self.seed)
        self.R6 = np.random.default_rng(seed=self.seed)

        for i in range(self.numblades):
            self.geometryparams[i, 0] = 10 + 20 * self.R1.random()
            self.geometryparams[i, 1] = 10 + 20 * self.R2.random()
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

        self.geometryparams[root, 4] += -1 + 2 * self.R5.random()
        self.geometryparams[root, 5] += -1 + 2 * self.R6.random()
        x = round(self.geometryparams[root, 4])
        y = round(self.geometryparams[root, 5])

        return x, y
