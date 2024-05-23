import logging
from re import L
from typing import Optional

import numpy as np
from mpi4py import MPI

from gprMax import config
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.model import Model

logger = logging.getLogger(__name__)


class MPIModel(Model):
    def __init__(self, comm: Optional[MPI.Intracomm] = None):
        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        self.rank = self.comm.Get_rank()

        self.G = self._create_grid()

        return super().__init__()

    def is_coordinator(self):
        return self.rank == 0

    def build(self):
        self.build_geometry()
        return
        return super().build()

    def build_geometry(self):
        if self.is_coordinator():
            self._check_for_dispersive_materials([self.G])
            self._check_memory_requirements([self.G])
        self._broadcast_model()

        self.G.global_size = np.array([self.gnx, self.gny, self.gnz], dtype=int)

        self.G.build()
        return
        self.G.dispersion_analysis(self.iterations)

    def _broadcast_model(self):
        self.gnx = self.comm.bcast(self.gnx)
        self.gny = self.comm.bcast(self.gny)
        self.gnz = self.comm.bcast(self.gnz)

        self.comm.Bcast(self.dl)
        self.dt = self.comm.bcast(self.dt)

        self.iterations = self.comm.bcast(self.iterations)

        self.srcsteps = self.comm.bcast(self.srcsteps)
        self.rxsteps = self.comm.bcast(self.rxsteps)

    def _output_geometry(self):
        if self.is_coordinator():
            logger.info("Geometry views and geometry objects are not currently supported with MPI.")

    def _create_grid(self) -> MPIGrid:
        cart_comm = MPI.COMM_WORLD.Create_cart(config.sim_config.mpi)
        return MPIGrid(cart_comm)
