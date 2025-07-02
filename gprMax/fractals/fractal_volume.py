# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from mpi4py_fft import PFFT, DistArray, newDistArray
from mpi4py_fft.pencil import Subcomm
from scipy import fftpack

from gprMax import config
from gprMax.cython.fractals_generate import generate_fractal3D
from gprMax.fractals.fractal_surface import FractalSurface
from gprMax.fractals.mpi_utilities import calculate_starts_and_subshape, create_mpi_type
from gprMax.materials import ListMaterial, PeplinskiSoil, RangeMaterial
from gprMax.utilities.mpi import Dim, Dir, get_relative_neighbour
from gprMax.utilities.utilities import round_value

logger = logging.getLogger(__name__)
np.seterr(divide="raise")


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
        self.start = np.array([xs, ys, zs], dtype=np.int32)
        self.stop = np.array([xf, yf, zf], dtype=np.int32)
        self.original_start = self.start.copy()
        self.original_stop = self.stop.copy()
        self.averaging = False
        self.dtype = np.dtype(np.complex128)
        self.seed = seed
        self.dimension = (
            dimension  # Fractal dimension from: http://dx.doi.org/10.1017/CBO9781139174695
        )
        self.weighting = np.array([1, 1, 1], dtype=np.float64)
        self.nbins = 0
        self.mixingmodel: Optional[Union[PeplinskiSoil, RangeMaterial, ListMaterial]] = None
        self.fractalsurfaces: List[FractalSurface] = []

    @property
    def xs(self) -> int:
        return self.start[0]

    @xs.setter
    def xs(self, value: int):
        self.start[0] = value

    @property
    def ys(self) -> int:
        return self.start[1]

    @ys.setter
    def ys(self, value: int):
        self.start[1] = value

    @property
    def zs(self) -> int:
        return self.start[2]

    @zs.setter
    def zs(self, value: int):
        self.start[2] = value

    @property
    def xf(self) -> int:
        return self.stop[0]

    @xf.setter
    def xf(self, value: int):
        self.stop[0] = value

    @property
    def yf(self) -> int:
        return self.stop[1]

    @yf.setter
    def yf(self, value: int):
        self.stop[1] = value

    @property
    def zf(self) -> int:
        return self.stop[2]

    @zf.setter
    def zf(self, value: int):
        self.stop[2] = value

    @property
    def size(self) -> npt.NDArray[np.int32]:
        return self.stop - self.start

    @property
    def nx(self) -> int:
        return self.xf - self.xs

    @property
    def ny(self) -> int:
        return self.yf - self.ys

    @property
    def nz(self) -> int:
        return self.zf - self.zs

    @property
    def originalxs(self) -> int:
        return self.original_start[0]

    @originalxs.setter
    def originalxs(self, value: int):
        self.original_start[0] = value

    @property
    def originalys(self) -> int:
        return self.original_start[1]

    @originalys.setter
    def originalys(self, value: int):
        self.original_start[1] = value

    @property
    def originalzs(self) -> int:
        return self.original_start[2]

    @originalzs.setter
    def originalzs(self, value: int):
        self.original_start[2] = value

    @property
    def originalxf(self) -> int:
        return self.original_stop[0]

    @originalxf.setter
    def originalxf(self, value: int):
        self.original_stop[0] = value

    @property
    def originalyf(self) -> int:
        return self.original_stop[1]

    @originalyf.setter
    def originalyf(self, value: int):
        self.original_stop[1] = value

    @property
    def originalzf(self) -> int:
        return self.original_stop[2]

    @originalzf.setter
    def originalzf(self, value: int):
        self.original_stop[2] = value

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
        upper_bound: npt.NDArray[np.int32],
    ):
        super().__init__(xs, xf, ys, yf, zs, zf, dimension, seed)
        self.comm = comm
        self.upper_bound = upper_bound

        # Limit the original start and stop to within the local bounds
        self.original_start = np.maximum(self.original_start, 0)
        self.original_stop = np.minimum(self.original_stop, self.upper_bound)

        # Ensure original_stop is not less than original_start
        self.original_stop = np.where(
            self.original_stop < self.original_start, self.original_start, self.original_stop
        )

    def generate_fractal_volume(self) -> bool:
        """Generate a 3D volume with a fractal distribution."""

        # Exit early if this rank does not contain the Fractal Volume
        # The size of a fractal volume can increase if a Fractal Surface
        # is attached. Hence the check needs to happen here once that
        # has happened.
        if any(self.stop <= 0) or any(self.start >= self.upper_bound):
            self.comm.Split(MPI.UNDEFINED)
            return False
        else:
            # Create new cartsesian communicator for the Fractal Volume
            comm = self.comm.Split()
            assert isinstance(comm, MPI.Intracomm)
            min_coord = np.array(self.comm.coords, dtype=np.int32)
            max_coord = min_coord + 1
            comm.Allreduce(MPI.IN_PLACE, min_coord, MPI.MIN)
            comm.Allreduce(MPI.IN_PLACE, max_coord, MPI.MAX)
            self.comm = comm.Create_cart((max_coord - min_coord).tolist())

        # Check domain decomosition is valid for the Fractal Volume
        if all([dim > 1 for dim in self.comm.dims]):
            raise ValueError(
                "Fractal volume must be positioned such that its MPI decomposition is 1 in at least"
                f" 1 dimension. Current decompostion is: {self.comm.dims}"
            )

        # Scale filter according to size of fractal volume
        sorted_size = np.sort(self.size)
        min_size = sorted_size[1] if sorted_size[0] == 1 else sorted_size[0]
        filterscaling = np.where(self.size == 1, 1, min_size / self.size)

        # Adjust weighting to account for filter scaling
        self.weighting = np.multiply(self.weighting, filterscaling)

        # Positional vector at centre of array, scaled by weighting
        v1 = self.weighting * self.size / 2

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
        A_shape = np.array(A.shape)
        A_substart = np.array(A.substart)
        static_dimension = Dim(A.alignment)

        # 3D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)

        # We need to generate random numbers for the whole domain in the
        # correct order (and throw away ones we don't need) to ensure
        # reproducibility when running with MPI domain decomposition
        cells_per_row = A.global_shape[Dim.Z]
        cells_per_plane = A.global_shape[Dim.Y] * cells_per_row

        # Skip forward in the x dimension
        planes_to_skip = A_substart[Dim.X]
        rng.standard_normal(size=planes_to_skip * cells_per_plane)

        for plane in range(A_shape[Dim.X]):
            # Skip forward in the y dimension
            rows_to_skip = A_substart[Dim.Y]
            rng.standard_normal(size=rows_to_skip * cells_per_row)

            for row in range(A_shape[Dim.Y]):
                # Skip forward in the z dimension
                columns_to_skip = A_substart[Dim.Z]
                rng.standard_normal(size=columns_to_skip)

                # Generate column of numbers in the z dimension
                A[plane, row, :] = rng.standard_normal(size=A_shape[Dim.Z])

                # Skip rest of the z dimension
                columns_to_skip = A.global_shape[Dim.Z] - columns_to_skip - A_shape[Dim.Z]
                rng.standard_normal(size=columns_to_skip)

            # Skip rest of the y dimension
            rows_to_skip = A.global_shape[Dim.Y] - rows_to_skip - A_shape[Dim.Y]
            rng.standard_normal(size=rows_to_skip * cells_per_row)

        A_hat = newDistArray(fft)
        assert isinstance(A_hat, DistArray)

        # 3D FFT
        fft.forward(A, A_hat, normalize=False)

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
            A_hat,
        )

        # Set DC component of FFT to zero
        if all(A_substart == 0):
            A_hat[0, 0, 0] = 0

        # Inverse 3D FFT transform
        fft.backward(A_hat, A, normalize=True)

        # Take the real part (numerical errors can give rise to an imaginary part)
        # of the IFFT, and convert type to floattype. N.B calculation of fractals
        # must always be carried out at double precision, i.e. float64, complex128
        A = np.real(A).astype(config.sim_config.dtypes["float_or_double"], copy=False)

        # Allreduce to get min and max values in the fractal volume
        min_value = np.array(np.amin(A), dtype=config.sim_config.dtypes["float_or_double"])
        max_value = np.array(np.amax(A), dtype=config.sim_config.dtypes["float_or_double"])
        self.comm.Allreduce(MPI.IN_PLACE, min_value, MPI.MIN)
        self.comm.Allreduce(MPI.IN_PLACE, max_value, MPI.MAX)

        # Bin fractal values
        bins = np.linspace(min_value, max_value, self.nbins)
        for j in range(A_shape[1]):
            for k in range(A_shape[2]):
                A[:, j, k] = np.digitize(A[:, j, k], bins, right=True)

        # Distribute A (DistArray) to match the MPIGrid decomposition
        local_shape = np.minimum(self.stop, self.upper_bound) - np.maximum(self.start, 0)
        self.fractalvolume = np.zeros(
            local_shape,
            dtype=config.sim_config.dtypes["float_or_double"],
        )

        # Negative means send to negative neighbour
        # Positive means receive from negative neighbour
        negative_offset = np.where(self.start >= 0, A_substart, self.start + A_substart)

        # Negative means send to positive neighbour
        # Positive means receive from positive neighbour
        positive_offset = np.minimum(self.stop, self.upper_bound) - (
            self.start + A_substart + A_shape
        )

        dirs = np.full(3, Dir.NONE)

        starts, subshape = calculate_starts_and_subshape(
            A_shape, -negative_offset, -positive_offset, dirs, sending=True
        )
        ends = starts + subshape
        A_local = A[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]]

        starts, subshape = calculate_starts_and_subshape(
            local_shape, negative_offset, positive_offset, dirs
        )
        ends = starts + subshape
        self.fractalvolume[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]] = A_local

        requests: List[MPI.Request] = []

        # Need to check neighbours in each direction (2D plane)
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

        # Dimensions of the 2D plane
        dims = [dim for dim in Dim if dim != static_dimension]

        for section in sections:
            dirs[dims[0]] = section[0]
            dirs[dims[1]] = section[1]
            rank = get_relative_neighbour(self.comm, dirs)

            # Skip if no neighbour
            if rank == -1:
                continue

            # Check if any data to send
            if all(
                np.select(
                    [dirs == Dir.NEG, dirs == Dir.POS],
                    [negative_offset <= 0, positive_offset <= 0],
                    dirs == Dir.NONE,
                )
            ):
                mpi_type = create_mpi_type(
                    A_shape, -negative_offset, -positive_offset, dirs, sending=True
                )

                logger.debug(f"Sending fractal volume to rank {rank}, MPI type={mpi_type.decode()}")
                self.comm.Isend([A, mpi_type], rank)

            # Check if any data to receive
            if all(
                np.select(
                    [dirs == Dir.NEG, dirs == Dir.POS],
                    [negative_offset > 0, positive_offset > 0],
                    dirs == Dir.NONE,
                )
            ):
                mpi_type = create_mpi_type(local_shape, negative_offset, positive_offset, dirs)

                logger.debug(
                    f"Receiving fractal volume from rank {rank}, MPI type={mpi_type.decode()}"
                )
                request = self.comm.Irecv([self.fractalvolume, mpi_type], rank)
                requests.append(request)

        if len(requests) > 0:
            requests[0].Waitall(requests)

        # Update start and stop to local bounds
        self.start = np.maximum(self.start, 0)
        self.stop = np.minimum(self.stop, self.upper_bound)

        logger.debug(
            f"Generated fractal volume: start={self.start}, stop={self.stop}, size={self.size}"
        )

        return True
