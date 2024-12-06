import logging
from typing import Optional

import numpy as np
from mpi4py import MPI

from gprMax import config
from gprMax.fields_outputs import write_hdf5_outputfile
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.model import Model
from gprMax.snapshots import save_snapshots

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

    @property
    def nx(self) -> float:
        return self.G.global_size[0]

    @nx.setter
    def nx(self, value: float):
        self.G.global_size[0] = value

    @property
    def ny(self) -> float:
        return self.G.global_size[1]

    @ny.setter
    def ny(self, value: float):
        self.G.global_size[1] = value

    @property
    def nz(self) -> float:
        return self.G.global_size[2]

    @nz.setter
    def nz(self, value: float):
        self.G.global_size[2] = value

    def is_coordinator(self):
        return self.rank == 0

    def build_geometry(self):
        self._broadcast_model()

        super().build_geometry()

        self._filter_geometry_objects()

    def _broadcast_model(self):
        self.title = self.comm.bcast(self.title)

        self.nx = self.comm.bcast(self.nx)
        self.ny = self.comm.bcast(self.ny)
        self.nz = self.comm.bcast(self.nz)

        self.comm.Bcast(self.dl)
        self.dt = self.comm.bcast(self.dt)

        self.iterations = self.comm.bcast(self.iterations)

        self.srcsteps = self.comm.bcast(self.srcsteps)
        self.rxsteps = self.comm.bcast(self.rxsteps)

        model_config = config.get_model_config()
        model_config.materials["maxpoles"] = self.comm.bcast(model_config.materials["maxpoles"])
        model_config.ompthreads = self.comm.bcast(model_config.ompthreads)

    def _filter_geometry_objects(self):
        objects = self.comm.bcast(self.geometryobjects)
        self.geometryobjects = []

        for go in objects:
            start = np.array([go.xs, go.ys, go.zs], dtype=np.intc)
            stop = np.array([go.xf, go.yf, go.zf], dtype=np.intc)
            if self.G.global_bounds_overlap_local_grid(start, stop):
                comm = self.comm.Split()
                assert isinstance(comm, MPI.Intracomm)
                start_grid_coord = self.G.get_grid_coord_from_coordinate(start)
                stop_grid_coord = self.G.get_grid_coord_from_coordinate(stop) + 1
                go.comm = comm.Create_cart((stop_grid_coord - start_grid_coord).tolist())

                go.global_size = np.array([go.nx, go.ny, go.nz], dtype=np.intc)
                start, stop, offset = self.G.limit_global_bounds_to_within_local_grid(start, stop)
                go.size = stop - start
                go.start = start
                go.stop = stop
                go.offset = offset
                self.geometryobjects.append(go)
            else:
                self.comm.Split(MPI.UNDEFINED)

    def write_output_data(self):
        """Writes output data, i.e. field data for receivers and snapshots to
        file(s).
        """
        # Write any snapshots to file for each grid
        if self.G.snapshots:
            save_snapshots(self.G.snapshots)

        self.G.gather_grid_objects()

        # Write output data to file if they are any receivers in any grids
        if self.is_coordinator() and (self.G.rxs or self.G.transmissionlines):
            self.G.size = self.G.global_size
            write_hdf5_outputfile(config.get_model_config().output_file_path_ext, self.title, self)

    def _create_grid(self) -> MPIGrid:
        cart_comm = MPI.COMM_WORLD.Create_cart(config.sim_config.mpi)
        return MPIGrid(cart_comm)
