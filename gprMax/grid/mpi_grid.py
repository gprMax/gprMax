from typing import Optional

from mpi4py import MPI

from gprMax.grid.fdtd_grid import FDTDGrid


class MPIGrid(FDTDGrid):
    xmin: int
    ymin: int
    zmin: int
    xmax: int
    ymax: int
    zmax: int

    def __init__(self, mpi_tasks_x: int, mpi_tasks_y: int, mpi_tasks_z: int, comm: Optional[MPI.Intracomm] = None):
        super().__init__()

        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        if mpi_tasks_x * mpi_tasks_y * mpi_tasks_z > self.size:
            # TODO: Raise expection - insufficient MPI tasks to create the grid as requested
            pass

        self.mpi_tasks_x = mpi_tasks_x
        self.mpi_tasks_y = mpi_tasks_y
        self.mpi_tasks_z = mpi_tasks_z

        self.rank = self.comm.rank
        self.size = self.comm.size

    def initialise_field_arrays(self):
        super().initialise_field_arrays()

        self.local_grid_size_x = self.nx // self.mpi_tasks_x
        self.local_grid_size_y = self.ny // self.mpi_tasks_y
        self.local_grid_size_z = self.nz // self.mpi_tasks_z

        self.xmin = (self.rank % self.nx) * self.local_grid_size_x
        self.ymin = ((self.mpi_tasks_x * self.rank) % self.ny) * self.local_grid_size_y
        self.zmin = ((self.mpi_tasks_y * self.mpi_tasks_x * self.rank) % self.nz) * self.local_grid_size_z
        self.xmax = self.xmin + self.local_grid_size_x
        self.ymax = self.ymin + self.local_grid_size_y
        self.zmax = self.zmin + self.local_grid_size_z
