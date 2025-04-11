from enum import IntEnum, unique
from typing import Union

import numpy as np
import numpy.typing as npt
from mpi4py import MPI


@unique
class Dim(IntEnum):
    X = 0
    Y = 1
    Z = 2


@unique
class Dir(IntEnum):
    NONE = -1
    NEG = 0
    POS = 1


def get_neighbours(comm: MPI.Cartcomm) -> npt.NDArray[np.int32]:
    neighbours = np.full((3, 2), -1, dtype=np.int32)
    neighbours[Dim.X] = comm.Shift(direction=Dim.X, disp=1)
    neighbours[Dim.Y] = comm.Shift(direction=Dim.Y, disp=1)
    neighbours[Dim.Z] = comm.Shift(direction=Dim.Z, disp=1)

    return neighbours


def get_neighbour(comm: MPI.Cartcomm, dim: Dim, dir: Dir) -> int:
    neighbours = comm.Shift(direction=dim, disp=1)
    return neighbours[dir]


def get_relative_neighbour(
    comm: MPI.Cartcomm, dirs: npt.NDArray[np.int32], disp: Union[int, npt.NDArray[np.int32]] = 1
) -> int:
    offset = np.zeros(3)
    offset = np.select([dirs == Dir.NEG, dirs == Dir.POS], [-disp, disp], default=0)

    coord = comm.coords + offset

    if any(coord < 0) or any(coord >= comm.dims):
        return -1

    return comm.Get_cart_rank(coord.tolist())
