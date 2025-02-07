import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from gprMax import config
from gprMax.fields_outputs import write_hdf5_outputfile
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.model import Model
from gprMax.output_controllers.geometry_objects import MPIGeometryObject
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

    def set_size(self, size: npt.NDArray[np.int32]):
        super().set_size(size)

        self.G.calculate_local_extents()

    def add_geometry_object(
        self,
        grid: MPIGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        basefilename: str,
    ) -> Optional[MPIGeometryObject]:
        comm = grid.create_sub_communicator(start, stop)

        if comm is None:
            return None
        else:
            geometry_object = MPIGeometryObject(
                grid, start[0], start[1], start[2], stop[0], stop[1], stop[2], basefilename, comm
            )
            self.geometryobjects.append(geometry_object)
            return geometry_object

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
