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

from typing import Tuple

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from gprMax.utilities.mpi import Dir


def calculate_starts_and_subshape(
    shape: npt.NDArray[np.int32],
    negative_offset: npt.NDArray[np.int32],
    positive_offset: npt.NDArray[np.int32],
    dirs: npt.NDArray[np.int32],
    sending: bool = False,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    negative_offset = np.where(
        dirs == Dir.NONE,
        np.maximum(negative_offset, 0),
        np.abs(negative_offset),
    )

    positive_offset = np.where(
        dirs == Dir.NONE,
        np.maximum(positive_offset, 0),
        np.abs(positive_offset),
    )

    starts = np.select(
        [dirs == Dir.NEG, dirs == Dir.POS],
        [0, shape - positive_offset - sending],
        default=negative_offset,
    )

    subshape = np.select(
        [dirs == Dir.NEG, dirs == Dir.POS],
        [negative_offset + sending, positive_offset + sending],
        default=shape - negative_offset - positive_offset,
    )

    return starts, subshape


def create_mpi_type(
    shape: npt.NDArray[np.int32],
    negative_offset: npt.NDArray[np.int32],
    positive_offset: npt.NDArray[np.int32],
    dirs: npt.NDArray[np.int32],
    sending: bool = False,
) -> MPI.Datatype:
    starts, subshape = calculate_starts_and_subshape(
        shape, negative_offset, positive_offset, dirs, sending
    )
    mpi_type = MPI.FLOAT.Create_subarray(shape.tolist(), subshape.tolist(), starts.tolist())
    mpi_type.Commit()
    return mpi_type
