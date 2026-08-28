# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import fftpack

from gprMax import config
from gprMax.cython.fractals_generate import generate_fractal2D
from gprMax.fractals.grass import Grass
from gprMax.grid.axes import Dim, Dir

if TYPE_CHECKING:
    from mpi4py import MPI

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
        self.fractalrange: Tuple[int, int] = (0, 0)
        self.filldepth = 0
        self.grass: List[Grass] = []

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

    def _te_invariant_inplane_index(self, invariant_axis: int) -> int:
        """Maps a global axis index (0/1/2 for x/y/z) to its position (0 or
        1) within this surface's 2D in-plane dims, as returned by
        get_surface_dims(). Only valid for axes other than the surface's own
        normal axis.
        """

        if self.xs == self.xf:
            dims_axes = (1, 2)
        elif self.ys == self.yf:
            dims_axes = (0, 2)
        else:
            dims_axes = (0, 1)

        return dims_axes.index(invariant_axis)

    def generate_fractal_surface(self) -> bool:
        """Generate a 2D array with a fractal distribution."""

        # In 2D TE mode the invariant axis is 2 cells thick. If this surface
        # spans both cells along that axis (i.e. the invariant axis is one
        # of the surface's 2 in-plane dims, not its normal axis - normal ==
        # invariant is rejected earlier, at #add_surface_roughness build
        # time), generate a single 1-cell-thick shadow surface and copy it
        # to both cells, for the same invariance/reproducibility reasons as
        # FractalVolume.
        mode = config.get_model_config().mode
        if mode.startswith("2D TE"):
            invariant_axis = "xyz".index(mode[-1])
            if self.size[invariant_axis] == 2:
                in_plane_index = self._te_invariant_inplane_index(invariant_axis)
                return self._generate_fractal_surface_te(invariant_axis, in_plane_index)

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
        self.fractalsurface = np.ascontiguousarray(
            np.real(fftpack.ifftn(self.fractalsurface)),
            dtype=config.sim_config.dtypes["float_or_double"],
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

        return True

    def _generate_fractal_surface_te(self, invariant_axis: int, in_plane_index: int) -> bool:
        """Generate a fractal surface for a 2D TE-mode rough surface that is
        2 cells thick along the invariant axis, by generating a single
        1-cell-thick shadow surface (same seed/dimension/weighting/range,
        reusing the existing, unmodified generation code) and broadcasting
        it to both cells.

        Args:
            invariant_axis: 0, 1 or 2 for x, y or z - the axis on which this
                surface is 2 cells thick.
            in_plane_index: 0 or 1 - position of invariant_axis within this
                surface's 2D in-plane dims (see get_surface_dims()).
        """

        shadow_stop = self.stop.copy()
        shadow_stop[invariant_axis] = self.start[invariant_axis] + 1

        shadow = FractalSurface(
            self.start[0],
            shadow_stop[0],
            self.start[1],
            shadow_stop[1],
            self.start[2],
            shadow_stop[2],
            self.dimension,
            self.seed,
        )
        shadow.weighting = self.weighting.copy()
        shadow.fractalrange = self.fractalrange
        shadow.generate_fractal_surface()

        self.fractalsurface = np.zeros(self.get_surface_dims(), dtype=shadow.fractalsurface.dtype)
        layer = np.take(shadow.fractalsurface, 0, axis=in_plane_index)
        for i in range(self.size[invariant_axis]):
            indexer = [slice(None), slice(None)]
            indexer[in_plane_index] = i
            self.fractalsurface[tuple(indexer)] = layer

        return True


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
        upper_bound: npt.NDArray[np.int32],
    ):
        super().__init__(xs, xf, ys, yf, zs, zf, dimension, seed)
        self.comm = comm
        self.upper_bound = upper_bound

    def generate_fractal_surface(self) -> bool:
        """Generate a 2D array with a fractal distribution."""

        from gprMax.fractals.mpi_utilities import calculate_starts_and_subshape, create_mpi_type
        from gprMax.mpi_support import require_mpi
        from gprMax.utilities.mpi import get_relative_neighbour

        MPI = require_mpi("distributed fractal-surface generation")

        # Import from mpi4py_fft
        # This is an optional dependency so only import if required
        from mpi4py_fft import PFFT, DistArray, newDistArray
        from mpi4py_fft.pencil import Subcomm

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
            return False
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
                "Fractal surface must be positioned such that its MPI decomposition is 1 in at"
                f" least 1 dimension. Current decompostion is: {self.comm.dims}"
            )

        # Check domain decomosition is valid for the Fractal Volume
        if len(self.grass) > 0 and self.comm.size > 1:
            raise ValueError(
                "Grass cannot currently be split across multiple MPI rank. Either change the MPI "
                " decomposition such that the grass object is contained within a single MPI rank,"
                " or divide the grass object into multiple sections. Current decompostion is:"
                f" {self.comm.dims}"
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

        # 2D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)

        # We need to generate random numbers for the whole domain in the
        # correct order (and throw away ones we don't need) to ensure
        # reproducibility when running with MPI domain decomposition

        # We use the following terms:
        # x - number of rows
        # y - number of cells
        cells_per_row = A.global_shape[Dim.Y]

        skip_to_next_row = cells_per_row - A_shape[Dim.Y]

        # Skip to the start of the fractal surface
        rng.standard_normal(size=A_substart[Dim.X] * cells_per_row)
        rng.standard_normal(size=A_substart[Dim.Y])

        # Generate numbers for the first row
        A[0, :] = rng.standard_normal(size=A_shape[Dim.Y])

        # Generate numbers for the remaining rows
        for row in range(1, A_shape[Dim.X]):
            rng.standard_normal(size=skip_to_next_row)
            A[row, :] = rng.standard_normal(size=A_shape[Dim.Y])

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
        A = np.ascontiguousarray(
            np.real(A), dtype=config.sim_config.dtypes["float_or_double"]
        )

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
                    A_shape,
                    -negative_offset,
                    -positive_offset,
                    dirs,
                    A.dtype,
                    sending=True,
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
                mpi_type = create_mpi_type(
                    local_shape,
                    negative_offset,
                    positive_offset,
                    dirs,
                    self.fractalsurface.dtype,
                )

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

        return True
