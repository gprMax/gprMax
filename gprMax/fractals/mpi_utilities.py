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
