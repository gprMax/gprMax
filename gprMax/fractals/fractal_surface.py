import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from mpi4py_fft import PFFT, DistArray, newDistArray
from mpi4py_fft.pencil import Subcomm
from scipy import fftpack

from gprMax import config
from gprMax.cython.fractals_generate import generate_fractal2D
from gprMax.fractals.mpi_utilities import calculate_starts_and_subshape, create_mpi_type
from gprMax.utilities.mpi import Dim, Dir, get_relative_neighbour

logger = logging.getLogger(__name__)
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
        self.start = np.array([xs, ys, zs], dtype=np.int32)
        self.stop = np.array([xf, yf, zf], dtype=np.int32)
        self.dtype = np.dtype(np.complex128)
        self.seed = seed
        self.dimension = (
            dimension  # Fractal dimension from: http://dx.doi.org/10.1017/CBO9781139174695
        )
        self.weighting = np.array([1, 1], dtype=np.float64)
        self.fractalrange = (0, 0)
        self.filldepth = 0
        self.grass = []

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

        # Generate fractal
        generate_fractal2D(
            surfacedims[0],
            surfacedims[1],
            0,
            0,
            surfacedims[0],
            surfacedims[1],
            config.get_model_config().ompthreads,
            self.dimension,
            self.weighting,
            v1,
            A,
            self.fractalsurface,
        )

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


class MPIFractalSurface(FractalSurface):
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

    def generate_fractal_surface(self):
        """Generate a 2D array with a fractal distribution."""

        if self.xs == self.xf:
            color = self.xs
            static_dimension = Dim.X
            dims = [Dim.Y, Dim.Z]
        elif self.ys == self.yf:
            color = self.ys
            static_dimension = Dim.Y
            dims = [Dim.X, Dim.Z]
        elif self.zs == self.zf:
            color = self.zs
            static_dimension = Dim.Z
            dims = [Dim.X, Dim.Y]

        # Exit early if this rank does not contain the Fractal Surface
        if (
            any(self.stop[dims] <= 0)
            or any(self.start[dims] >= self.upper_bound[dims])
            or self.fractalrange[1] <= 0
            or self.fractalrange[0] >= self.upper_bound[static_dimension]
        ):
            self.comm.Split(MPI.UNDEFINED)
            # Update start and stop to local bounds
            self.start = np.maximum(self.start, 0)
            self.start = np.minimum(self.start, self.upper_bound)
            self.stop = np.maximum(self.stop, 0)
            self.stop = np.minimum(self.stop, self.upper_bound)
            return
        else:
            # Create new cartsesian communicator for the Fractal Surface
            comm = self.comm.Split(color=color)
            assert isinstance(comm, MPI.Intracomm)
            min_coord = np.array(self.comm.coords, dtype=np.int32)[dims]
            max_coord = min_coord + 1
            comm.Allreduce(MPI.IN_PLACE, min_coord, MPI.MIN)
            comm.Allreduce(MPI.IN_PLACE, max_coord, MPI.MAX)
            cart_dims = (max_coord - min_coord).tolist()
            self.comm = comm.Create_cart(cart_dims)

        # Check domain decomosition is valid for the Fractal Volume
        if all([dim > 1 for dim in self.comm.dims]):
            raise ValueError(
                "Fractal surface must be positioned such that its MPI decomposition is 1 in at least"
                f" 1 dimension. Current decompostion is: {self.comm.dims}"
            )

        surfacedims = self.get_surface_dims()

        # Positional vector at centre of array, scaled by weighting
        v1 = self.weighting * surfacedims / 2

        subcomm = Subcomm(self.comm)

        A = DistArray(self.size[dims], subcomm, dtype=self.dtype)

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

        # 3D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)

        for index in np.ndindex(*A.global_shape):
            index = np.array(index)
            if any(index < A_substart) or any(index >= A_substart + A_shape):
                rng.standard_normal()
            else:
                index -= A_substart
                A[index[0], index[1]] = rng.standard_normal()

        A_hat = newDistArray(fft)
        assert isinstance(A_hat, DistArray)

        # 2D FFT
        fft.forward(A, A_hat, normalize=False)

        # Generate fractal
        generate_fractal2D(
            A_hat.shape[0],
            A_hat.shape[1],
            A_hat.substart[0],
            A_hat.substart[1],
            A_hat.global_shape[0],
            A_hat.global_shape[1],
            config.get_model_config().ompthreads,
            self.dimension,
            self.weighting,
            v1,
            A_hat,
            A_hat,
        )

        # Set DC component of FFT to zero
        if all(A_substart == 0):
            A_hat[0, 0] = 0

        # Inverse 2D FFT transform
        fft.backward(A_hat, A, normalize=True)

        # Take the real part (numerical errors can give rise to an imaginary part)
        #  of the IFFT, and convert type to floattype. N.B calculation of fractals
        # must always be carried out at double precision, i.e. float64, complex128
        A = np.real(A).astype(config.sim_config.dtypes["float_or_double"], copy=False)

        # Allreduce to get min and max values in the fractal surface
        min_value = np.array(np.amin(A), dtype=config.sim_config.dtypes["float_or_double"])
        max_value = np.array(np.amax(A), dtype=config.sim_config.dtypes["float_or_double"])
        self.comm.Allreduce(MPI.IN_PLACE, min_value, MPI.MIN)
        self.comm.Allreduce(MPI.IN_PLACE, max_value, MPI.MAX)

        # Scale the fractal volume according to requested range
        fractalrange = max_value - min_value
        A = (
            A * ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange)
            + self.fractalrange[0]
            - ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange) * min_value
        )

        # Distribute A (DistArray) to match the MPIGrid decomposition
        local_shape = (np.minimum(self.stop, self.upper_bound) - np.maximum(self.start, 0))[dims]
        self.fractalsurface = np.zeros(
            local_shape,
            dtype=config.sim_config.dtypes["float_or_double"],
        )

        # Negative means send to negative neighbour
        # Positive means receive from negative neighbour
        negative_offset = np.where(self.start[dims] >= 0, A_substart, self.start[dims] + A_substart)

        # Negative means send to positive neighbour
        # Positive means receive from positive neighbour
        positive_offset = np.minimum(self.stop, self.upper_bound)[dims] - (
            self.start[dims] + A_substart + A_shape
        )

        dirs = np.full(2, Dir.NONE)

        starts, subshape = calculate_starts_and_subshape(
            A_shape, -negative_offset, -positive_offset, dirs, sending=True
        )
        ends = starts + subshape
        A_local = A[starts[0] : ends[0], starts[1] : ends[1]]

        starts, subshape = calculate_starts_and_subshape(
            local_shape, negative_offset, positive_offset, dirs
        )
        ends = starts + subshape
        self.fractalsurface[starts[0] : ends[0], starts[1] : ends[1]] = A_local

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

        for section in sections:
            dirs[0] = section[0]
            dirs[1] = section[1]
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

                logger.debug(
                    f"Sending fractal surface to rank {rank}, MPI type={mpi_type.decode()}"
                )
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
                    f"Receiving fractal surface from rank {rank}, MPI type={mpi_type.decode()}"
                )
                request = self.comm.Irecv([self.fractalsurface, mpi_type], rank)
                requests.append(request)

        if len(requests) > 0:
            requests[0].Waitall(requests)

        # Update start and stop to local bounds
        self.start = np.maximum(self.start, 0)
        self.start = np.minimum(self.start, self.upper_bound)
        self.stop = np.maximum(self.stop, 0)
        self.stop = np.minimum(self.stop, self.upper_bound)

        logger.debug(
            f"Generated fractal surface: start={self.start}, stop={self.stop}, size={self.size}, fractalrange={self.fractalrange}"
        )
