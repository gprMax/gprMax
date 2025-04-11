# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from mpi4py_fft import PFFT, DistArray, newDistArray
from mpi4py_fft.pencil import Subcomm
from scipy import fftpack

import gprMax.config as config
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.output_controllers.grid_view import MPIGridView
from gprMax.utilities.mpi import Dim, Dir, get_relative_neighbour

from .cython.fractals_generate import generate_fractal2D, generate_fractal3D
from .utilities.utilities import round_value

np.seterr(divide="raise")


class FractalSurface:
    """Fractal surfaces."""

    surfaceIDs = ["xminus", "xplus", "yminus", "yplus", "zminus", "zplus"]

    def __init__(self, xs, xf, ys, yf, zs, zf, dimension, seed):
        """
        Args:
            xs, xf, ys, yf, zs, zf: floats for the extent of the fractal surface
                                        (one pair of coordinates must be equal
                                        to correctly define a surface).
            dimension: float for the fractal dimension that controls the fractal
                        distribution.
            seed: int for seed value for random number generator.
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
        self.dtype = np.dtype(np.complex128)
        self.seed = seed
        self.dimension = (
            dimension  # Fractal dimension from: http://dx.doi.org/10.1017/CBO9781139174695
        )
        self.weighting = np.array([1, 1], dtype=np.float64)
        self.fractalrange = (0, 0)
        self.filldepth = 0
        self.grass = []

    def get_surface_dims(self):
        """Gets the dimensions of the fractal surface based on surface plane."""

        if self.xs == self.xf:
            surfacedims = (self.ny, self.nz)
        elif self.ys == self.yf:
            surfacedims = (self.nx, self.nz)
        elif self.zs == self.zf:
            surfacedims = (self.nx, self.ny)

        return surfacedims

    def generate_fractal_surface(self):
        """Generate a 2D array with a fractal distribution."""

        surfacedims = self.get_surface_dims()

        self.fractalsurface = np.zeros(surfacedims, dtype=self.dtype)

        # Positional vector at centre of array, scaled by weighting
        v1 = np.array(
            [
                self.weighting[0] * (surfacedims[0]) / 2,
                self.weighting[1] * (surfacedims[1]) / 2,
            ]
        )

        # 2D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)
        A = rng.standard_normal(size=(surfacedims[0], surfacedims[1]))

        # 2D FFT
        A = fftpack.fftn(A)
        # Shift the zero frequency component to the centre of the array
        A = fftpack.fftshift(A)

        # Generate fractal
        generate_fractal2D(
            surfacedims[0],
            surfacedims[1],
            config.get_model_config().ompthreads,
            self.dimension,
            self.weighting,
            v1,
            A,
            self.fractalsurface,
        )
        # Shift the zero frequency component to start of the array
        self.fractalsurface = fftpack.ifftshift(self.fractalsurface)
        # Set DC component of FFT to zero
        self.fractalsurface[0, 0] = 0
        # Take the real part (numerical errors can give rise to an imaginary part)
        #  of the IFFT, and convert type to floattype. N.B calculation of fractals
        # must always be carried out at double precision, i.e. float64, complex128
        self.fractalsurface = np.real(fftpack.ifftn(self.fractalsurface)).astype(
            config.sim_config.dtypes["float_or_double"], copy=False
        )
        # Scale the fractal volume according to requested range
        fractalmin = np.amin(self.fractalsurface)
        fractalmax = np.amax(self.fractalsurface)
        fractalrange = fractalmax - fractalmin
        self.fractalsurface = (
            self.fractalsurface * ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange)
            + self.fractalrange[0]
            - ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange) * fractalmin
        )


class FractalVolume:
    """Fractal volumes."""

    def __init__(
        self,
        xs: int,
        xf: int,
        ys: int,
        yf: int,
        zs: int,
        zf: int,
        dimension: float,
        seed: Optional[int],
    ):
        """
        Args:
            xs, xf, ys, yf, zs, zf: floats for the extent of the fractal volume.
            dimension: float for the fractal dimension that controls the fractal
                        distribution.
            seed: int for seed value for random number generator.
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
        self.originalxs = xs
        self.originalxf = xf
        self.originalys = ys
        self.originalyf = yf
        self.originalzs = zs
        self.originalzf = zf
        self.averaging = False
        self.dtype = np.dtype(np.complex128)
        self.seed = seed
        self.dimension = (
            dimension  # Fractal dimension from: http://dx.doi.org/10.1017/CBO9781139174695
        )
        self.weighting = np.array([1, 1, 1], dtype=np.float64)
        self.nbins = 0
        self.fractalsurfaces = []

    def generate_fractal_volume(self) -> bool:
        """Generate a 3D volume with a fractal distribution."""

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
            filterscaling = np.amin(np.array([self.nx, self.ny, self.nz])) / np.array(
                [self.nx, self.ny, self.nz]
            )

        # Adjust weighting to account for filter scaling
        self.weighting = np.multiply(self.weighting, filterscaling)

        self.fractalvolume = np.zeros((self.nx, self.ny, self.nz), dtype=self.dtype)

        # Positional vector at centre of array, scaled by weighting
        v1 = np.array(
            [
                self.weighting[0] * self.nx / 2,
                self.weighting[1] * self.ny / 2,
                self.weighting[2] * self.nz / 2,
            ]
        )

        # 3D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)
        A = rng.standard_normal(size=(self.nx, self.ny, self.nz))

        # 3D FFT
        A = fftpack.fftn(A)

        # Generate fractal
        generate_fractal3D(
            self.nx,
            self.ny,
            self.nz,
            0,
            0,
            0,
            self.nx,
            self.ny,
            self.nz,
            config.get_model_config().ompthreads,
            self.dimension,
            self.weighting,
            v1,
            A,
            self.fractalvolume,
        )

        # Set DC component of FFT to zero
        self.fractalvolume[0, 0, 0] = 0
        # Take the real part (numerical errors can give rise to an imaginary part)
        # of the IFFT, and convert type to floattype. N.B calculation of fractals
        # must always be carried out at double precision, i.e. float64, complex128
        self.fractalvolume = np.real(fftpack.ifftn(self.fractalvolume)).astype(
            config.sim_config.dtypes["float_or_double"], copy=False
        )

        # Bin fractal values
        bins = np.linspace(np.amin(self.fractalvolume), np.amax(self.fractalvolume), self.nbins)
        for j in range(self.ny):
            for k in range(self.nz):
                self.fractalvolume[:, j, k] = np.digitize(
                    self.fractalvolume[:, j, k], bins, right=True
                )

        return True

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


class MPIFractalVolume(FractalVolume):
    def __init__(
        self,
        xs: int,
        xf: int,
        ys: int,
        yf: int,
        zs: int,
        zf: int,
        dimension: float,
        seed: Optional[int],
        comm: MPI.Cartcomm,
        ux: int,
        uy: int,
        uz: int,
    ):
        super().__init__(xs, xf, ys, yf, zs, zf, dimension, seed)
        self.comm = comm
        self.upper_bound = np.array([ux, uy, uz])

    def generate_fractal_volume(self) -> bool:
        """Generate a 3D volume with a fractal distribution."""

        self.start = np.array([self.xs, self.ys, self.zs], dtype=np.int32)
        self.stop = np.array([self.xf, self.yf, self.zf], dtype=np.int32)
        self.size = self.stop - self.start

        if any(self.stop < 0):
            self.comm.Split(MPI.UNDEFINED)
            return False
        else:
            comm = self.comm.Split()
            assert isinstance(comm, MPI.Intracomm)
            min_coord = np.array(self.comm.coords, dtype=np.int32)
            max_coord = min_coord + 1
            comm.Allreduce(MPI.IN_PLACE, min_coord, MPI.MIN)
            comm.Allreduce(MPI.IN_PLACE, max_coord, MPI.MAX)
            self.comm = comm.Create_cart((max_coord - min_coord).tolist())

        # Scale filter according to size of fractal volume
        sorted_size = np.sort(self.size)
        min_size = sorted_size[1] if sorted_size[0] == 1 else sorted_size[0]
        filterscaling = np.where(self.size == 1, 1, min_size / self.size)

        # Adjust weighting to account for filter scaling
        self.weighting = np.multiply(self.weighting, filterscaling)

        # Positional vector at centre of array, scaled by weighting
        v1 = self.weighting * self.size / 2

        # 3D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)

        if all([dim > 1 for dim in self.comm.dims]):
            raise ValueError(
                "Fractal volume must be positioned such that its MPI decomposition is 1 in at least"
                f" 1 dimension. Current decompostion is: {self.comm.dims}"
            )

        subcomm = Subcomm(self.comm)
        A = DistArray(self.size, subcomm, dtype=self.dtype)
        fft = PFFT(
            None,
            axes=tuple(np.argsort(self.comm.dims)[::-1]),
            darray=A,
            collapse=False,
            backend="fftw",
        )

        # Decomposition of A may be different to the MPIGrid
        substart = np.array(A.substart)
        shape = np.array(A.shape)

        for index in np.ndindex(*A.global_shape):
            index = np.array(index)
            if any(index < substart) or any(index >= substart + shape):
                rng.standard_normal()
            else:
                index -= substart
                A[index[0], index[1], index[2]] = rng.standard_normal()

        A_hat = newDistArray(fft)
        assert isinstance(A_hat, DistArray)

        # 3D FFT
        fft.forward(A, A_hat, normalize=False)
        A_hat_out = np.zeros_like(A_hat)

        # Generate fractal
        generate_fractal3D(
            A_hat.shape[0],
            A_hat.shape[1],
            A_hat.shape[2],
            A_hat.substart[0],
            A_hat.substart[1],
            A_hat.substart[2],
            A_hat.global_shape[0],
            A_hat.global_shape[1],
            A_hat.global_shape[2],
            config.get_model_config().ompthreads,
            self.dimension,
            self.weighting,
            v1,
            A_hat,
            A_hat_out,
        )

        # Set DC component of FFT to zero
        if all(substart == 0):
            A_hat_out[0, 0, 0] = 0

        # Take the real part (numerical errors can give rise to an imaginary part)
        # of the IFFT, and convert type to floattype. N.B calculation of fractals
        # must always be carried out at double precision, i.e. float64, complex128
        Aj = np.real(fft.backward(A_hat_out, normalize=True)).astype(
            config.sim_config.dtypes["float_or_double"], copy=False
        )

        min_value = np.array(np.amin(Aj), dtype=config.sim_config.dtypes["float_or_double"])
        max_value = np.array(np.amax(Aj), dtype=config.sim_config.dtypes["float_or_double"])

        self.comm.Allreduce(MPI.IN_PLACE, min_value, MPI.MIN)
        self.comm.Allreduce(MPI.IN_PLACE, max_value, MPI.MAX)

        fractalvolume_initial = np.zeros_like(Aj)

        # Bin fractal values
        bins = np.linspace(min_value, max_value, self.nbins)
        for j in range(shape[1]):
            for k in range(shape[2]):
                fractalvolume_initial[:, j, k] = np.digitize(Aj[:, j, k], bins, right=True)

        # Negative means send to negative neighbour
        # Positive means receive from negative neighbour
        negative_offset = np.where(self.start >= 0, 0, self.start + substart)

        # Negative means send to positive neighbour
        # Positive means receive from positive neighbour
        positive_offset = self.upper_bound - (self.start + substart + shape)

        print(
            f"start: {self.start}, substart: {substart}, shape: {shape}, upper_bound = {self.upper_bound}"
        )
        print(f"negative_offset: {negative_offset}, positive_offset: {positive_offset}")

        self.fractalvolume = np.zeros(
            self.upper_bound - np.where(self.start < 0, 0, self.start),
            dtype=config.sim_config.dtypes["float_or_double"],
        )

        requests: List[MPI.Request] = []

        static_dimension = Dim(A.alignment)
        dims = [dim for dim in Dim if dim != static_dimension]
        dims = (dims[0], dims[1])

        negative_offset = (negative_offset[dims[0]], negative_offset[dims[1]])
        positive_offset = (positive_offset[dims[0]], positive_offset[dims[1]])

        negative_spacing = (max(negative_offset[0], 0), max(negative_offset[1], 0))
        positive_spacing = (max(positive_offset[0], 0), max(positive_offset[1], 0))

        slices = [slice(None)] * 3
        slices[dims[0]], slices[dims[1]] = self.create_slices(
            negative_spacing, positive_spacing, None, None
        )

        negative_spacing = (abs(min(negative_offset[0], 0)), abs(min(negative_offset[1], 0)))
        positive_spacing = (abs(min(positive_offset[0], 0)), abs(min(positive_offset[1], 0)))

        initial_slices = [slice(None)] * 3
        initial_slices[dims[0]], initial_slices[dims[1]] = self.create_slices(
            negative_spacing, positive_spacing, None, None
        )

        self.fractalvolume[slices[0], slices[1], slices[2]] = fractalvolume_initial[
            initial_slices[0], initial_slices[1], initial_slices[2]
        ]

        # Negative means send to negative neighbour
        # Positive means receive from negative neighbour
        negative_offset = np.where(self.start >= 0, 0, self.start + substart)

        # Negative means send to positive neighbour
        # Positive means receive from positive neighbour
        positive_offset = self.upper_bound - (self.start + substart + shape)

        sections = [
            (Dir.NEG, Dir.NONE),
            (Dir.POS, Dir.NONE),
            (Dir.NONE, Dir.NEG),
            (Dir.NONE, Dir.POS),
            (Dir.NEG, Dir.NEG),
            (Dir.NEG, Dir.POS),
            (Dir.POS, Dir.NEG),
            (Dir.POS, Dir.POS),
        ]

        for section in sections:
            dirs = np.full(3, Dir.NONE)
            dirs[dims[0]] = section[0]
            dirs[dims[1]] = section[1]

            self.check_send(
                fractalvolume_initial,
                negative_offset,
                positive_offset,
                dirs,
            )

            request = self.check_receive(
                self.fractalvolume,
                negative_offset,
                positive_offset,
                dirs,
            )

            if request is not None:
                requests.append(request)

        if len(requests) > 0:
            requests[0].Waitall(requests)

        self.nx = self.fractalvolume.shape[0]
        self.ny = self.fractalvolume.shape[1]
        self.nz = self.fractalvolume.shape[2]

        self.xs = max(0, self.xs)
        self.ys = max(0, self.ys)
        self.zs = max(0, self.zs)

        return True

    def check_send(
        self,
        array: npt.NDArray[np.float32],
        negative_offset: npt.NDArray[np.int32],
        positive_offset: npt.NDArray[np.int32],
        dirs: npt.NDArray[np.int32],
    ):
        if all(
            np.logical_or(
                dirs == Dir.NONE,
                np.where(dirs == Dir.NEG, negative_offset < 0, positive_offset < 0),
            )
        ):
            negative_spacing = np.where(
                dirs == Dir.NONE,
                np.maximum(-negative_offset, 0),
                np.abs(negative_offset),
            )

            positive_spacing = np.where(
                dirs == Dir.NONE,
                np.maximum(-positive_offset, 0),
                np.abs(positive_offset),
            )

            rank = get_relative_neighbour(self.comm, dirs)

            shape = np.array(array.shape, dtype=np.int32)
            mpi_type = self.create_mpi_type(
                shape, negative_spacing, positive_spacing, dirs, sending=True
            )

            self.comm.Isend([array, mpi_type], rank)

    def check_receive(
        self,
        array,
        negative_offset: npt.NDArray[np.int32],
        positive_offset: npt.NDArray[np.int32],
        dirs: npt.NDArray[np.int32],
    ) -> Optional[MPI.Request]:
        if all(
            np.logical_or(
                dirs == Dir.NONE,
                np.where(dirs == Dir.NEG, negative_offset > 0, positive_offset > 0),
            )
        ):
            negative_spacing = np.where(
                dirs == Dir.NONE,
                np.maximum(negative_offset, 0),
                np.abs(negative_offset),
            )

            positive_spacing = np.where(
                dirs == Dir.NONE,
                np.maximum(positive_offset, 0),
                np.abs(positive_offset),
            )

            rank = get_relative_neighbour(self.comm, dirs)

            shape = np.array(array.shape, dtype=np.int32)
            mpi_type = self.create_mpi_type(shape, negative_spacing, positive_spacing, dirs)

            return self.comm.Irecv([array, mpi_type], rank)
        else:
            return None

    def create_slices(
        self,
        negative_offset: Tuple[int, int],
        positive_offset: Tuple[int, int],
        dir1: Optional[Dir],
        dir2: Optional[Dir],
        sending: bool = False,
    ) -> Tuple[slice, slice]:
        n1, n2 = negative_offset
        p1, p2 = positive_offset

        if dir1 == Dir.NEG:
            slice1 = slice(n1 + sending)
        elif dir1 == Dir.POS:
            slice1 = slice(-p1 - sending, None)
        elif p1 != 0:
            slice1 = slice(n1, -p1)
        else:
            slice1 = slice(n1, None)

        if dir2 == Dir.NEG:
            slice2 = slice(n2 + sending)
        elif dir2 == Dir.POS:
            slice2 = slice(-p2 - sending, None)
        elif p2 != 0:
            slice2 = slice(n2, -p2)
        else:
            slice2 = slice(n2, None)

        return slice1, slice2

    def create_mpi_type(
        self,
        shape: npt.NDArray[np.int32],
        negative_offset: npt.NDArray[np.int32],
        positive_offset: npt.NDArray[np.int32],
        dirs: npt.NDArray[np.int32],
        sending: bool = False,
    ) -> MPI.Datatype:
        starts = np.select(
            [dirs == Dir.NEG, dirs == Dir.POS],
            [0, shape - positive_offset - sending],
            default=negative_offset,
        ).tolist()

        subshape = np.select(
            [dirs == Dir.NEG, dirs == Dir.POS],
            [negative_offset + sending, positive_offset + sending],
            default=shape - negative_offset - positive_offset,
        ).tolist()

        mpi_type = MPI.FLOAT.Create_subarray(shape.tolist(), subshape, starts)
        mpi_type.Commit()
        return mpi_type


class Grass:
    """Geometry information for blades of grass."""

    def __init__(self, numblades, seed):
        """
        Args:
            numblades: int for the number of blades of grass.
            seed: int for seed value for random number generator.
        """

        self.numblades = numblades
        self.geometryparams = np.zeros(
            (self.numblades, 6), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.seed = seed
        self.set_geometry_parameters()

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
            blade: int for the numeric ID of grass blade.
            height: float for the height of grass blade.

        Returns:
            x, y: floats for the x and y coordinates of grass blade.
        """

        x = (
            self.geometryparams[blade, 2]
            * (height / self.geometryparams[blade, 0])
            * (height / self.geometryparams[blade, 0])
        )
        y = (
            self.geometryparams[blade, 3]
            * (height / self.geometryparams[blade, 1])
            * (height / self.geometryparams[blade, 1])
        )
        x = round_value(x)
        y = round_value(y)

        return x, y

    def calculate_root_geometry(self, root, depth):
        """Calculates the x and y coordinates for a given depth of grass root.

        Args:
            root: int for the umeric ID of grass root.
            depth: float for the depth of grass root.

        Returns:
            x, y: floats for the x and y coordinates of grass root.
        """

        self.geometryparams[root, 4] += -1 + 2 * self.R5.random()
        self.geometryparams[root, 5] += -1 + 2 * self.R6.random()
        x = round(self.geometryparams[root, 4])
        y = round(self.geometryparams[root, 5])

        return x, y
