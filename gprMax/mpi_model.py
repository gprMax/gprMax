import logging
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from gprMax import config
from gprMax.fields_outputs import write_hdf5_outputfile
from gprMax.geometry_outputs.geometry_objects import MPIGeometryObject
from gprMax.geometry_outputs.geometry_view_lines import MPIGeometryViewLines
from gprMax.geometry_outputs.geometry_view_voxels import MPIGeometryViewVoxels
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.model import Model
from gprMax.snapshots import MPISnapshot, Snapshot, save_snapshots

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
        """Add a geometry object to the model.

        Args:
            grid: Grid to create a geometry object for
            start: Lower extent of the geometry object (x, y, z)
            stop: Upper extent of the geometry object (x, y, z)
            basefilename: Output filename of the geometry object

        Returns:
            geometry_object: The new geometry object or None if no
                geometry object was created.
        """
        if grid.local_bounds_overlap_grid(start, stop):
            geometry_object = MPIGeometryObject(
                grid, start[0], start[1], start[2], stop[0], stop[1], stop[2], basefilename
            )
            self.geometryobjects.append(geometry_object)
            return geometry_object
        else:
            # The MPIGridView created by the MPIGeometryObject will
            # create a new communicator using MPI_Split. Calling this
            # here prevents deadlock if not all ranks create the new
            # MPIGeometryObject.
            grid.comm.Split(MPI.UNDEFINED)
            return None

    def add_geometry_view_voxels(
        self,
        grid: MPIGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        dl: npt.NDArray[np.int32],
        filename: str,
    ) -> Optional[MPIGeometryViewVoxels]:
        """Add a voxel geometry view to the model.

        Args:
            grid: Grid to create a geometry view for.
            start: Lower extent of the geometry view (x, y, z).
            stop: Upper extent of the geometry view (x, y, z).
            dl: Discritisation of the geometry view (x, y, z).
            filename: Output filename of the geometry view.

        Returns:
            geometry_view: The new geometry view or None if no geometry
                view was created.
        """
        if grid.local_bounds_overlap_grid(start, stop):
            geometry_view = MPIGeometryViewVoxels(
                start[0],
                start[1],
                start[2],
                stop[0],
                stop[1],
                stop[2],
                dl[0],
                dl[1],
                dl[2],
                filename,
                grid,
            )
            self.geometryviews.append(geometry_view)
            return geometry_view
        else:
            # The MPIGridView created by MPIGeometryViewVoxels will
            # create a new communicator using MPI_Split. Calling this
            # here prevents deadlock if not all ranks create the new
            # MPIGeometryViewVoxels.
            grid.comm.Split(MPI.UNDEFINED)
            return None

    def add_geometry_view_lines(
        self,
        grid: MPIGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        filename: str,
    ) -> Optional[MPIGeometryViewLines]:
        """Add a lines geometry view to the model.

        Args:
            grid: Grid to create a geometry view for.
            start: Lower extent of the geometry view (x, y, z).
            stop: Upper extent of the geometry view (x, y, z).
            filename: Output filename of the geometry view.

        Returns:
            geometry_view: The new geometry view or None if no geometry
                view was created.
        """
        if grid.local_bounds_overlap_grid(start, stop):
            geometry_view = MPIGeometryViewLines(
                start[0],
                start[1],
                start[2],
                stop[0],
                stop[1],
                stop[2],
                filename,
                grid,
            )
            self.geometryviews.append(geometry_view)
            return geometry_view
        else:
            # The MPIGridView created by MPIGeometryViewLines will
            # create a new communicator using MPI_Split. Calling this
            # here prevents deadlock if not all ranks create the new
            # MPIGeometryViewLines.
            grid.comm.Split(MPI.UNDEFINED)
            return None

    def add_snapshot(
        self,
        grid: MPIGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        dl: npt.NDArray[np.int32],
        time: int,
        filename: str,
        fileext: str,
        outputs: Dict[str, bool],
    ) -> Optional[MPISnapshot]:
        """Add a snapshot to the provided grid.

        Args:
            grid: Grid to create a snapshot for.
            start: Lower extent of the snapshot (x, y, z).
            stop: Upper extent of the snapshot (x, y, z).
            dl: Discritisation of the snapshot (x, y, z).
            time: Iteration number to take the snapshot on
            filename: Output filename of the snapshot.
            fileext: File extension of the snapshot.
            outputs: Fields to use in the snapshot.

        Returns:
            snapshot: The new snapshot or None if no snapshot was
                created.
        """
        if grid.local_bounds_overlap_grid(start, stop):
            snapshot = MPISnapshot(
                start[0],
                start[1],
                start[2],
                stop[0],
                stop[1],
                stop[2],
                dl[0],
                dl[1],
                dl[2],
                time,
                filename,
                fileext,
                outputs,
                grid,
            )
            # TODO: Move snapshots into the Model
            grid.snapshots.append(snapshot)
            return snapshot
        else:
            # The MPIGridView created by MPISnapshot will create a new
            # communicator using MPI_Split. Calling this here prevents
            # deadlock if not all ranks create the new MPISnapshot.
            grid.comm.Split(MPI.UNDEFINED)
            return None

    def write_output_data(self):
        """Writes output data, i.e. field data for receivers and snapshots to
        file(s).
        """
        # Write any snapshots to file for each grid
        if self.G.snapshots:
            save_snapshots(self.G.snapshots)

        # TODO: Output sources and receivers using parallel I/O
        self.G.gather_grid_objects()

        # Write output data to file if they are any receivers in any grids
        if self.is_coordinator() and (self.G.rxs or self.G.transmissionlines):
            self.G.size = self.G.global_size
            write_hdf5_outputfile(config.get_model_config().output_file_path_ext, self.title, self)

    def _create_grid(self) -> MPIGrid:
        cart_comm = MPI.COMM_WORLD.Create_cart(config.sim_config.mpi)
        return MPIGrid(cart_comm)
