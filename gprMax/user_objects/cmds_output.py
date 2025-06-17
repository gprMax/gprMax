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

import logging
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.model import Model
from gprMax.snapshots import Snapshot as SnapshotUser
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.user_objects.user_objects import OutputUserObject
from gprMax.utilities.utilities import round_int

logger = logging.getLogger(__name__)


class Snapshot(OutputUserObject):
    """Obtains information about the electromagnetic fields within a volume
        of the model at a given time instant.

    Attributes:
        p1: tuple required to specify lower left (x,y,z) coordinates of volume
                of snapshot in metres.
        p2: tuple required to specify upper right (x,y,z) coordinates of volume
                of snapshot in metres.
        dl: tuple require to specify spatial discretisation of the snapshot
                in metres.
        filename: string required for name of the file to store snapshot.
        time/iterations: either a float for time or an int for iterations
                            must be specified for point in time at which the
                            snapshot will be taken.
        fileext: optional string to indicate type for snapshot file, either
                            '.vti' (default) or '.h5'
        outputs: optional list of outputs for receiver. It can be any
                    selection from Ex, Ey, Ez, Hx, Hy, or Hz.
    """

    @property
    def order(self):
        return 9

    @property
    def hash(self):
        return "#snapshot"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        dl: Tuple[float, float, float],
        filename: str,
        fileext: Optional[str] = None,
        iterations: Optional[int] = None,
        time: Optional[float] = None,
        outputs: Optional[List[str]] = None,
    ):
        super().__init__(
            p1=p1,
            p2=p2,
            dl=dl,
            filename=filename,
            fileext=fileext,
            iterations=iterations,
            time=time,
            outputs=outputs,
        )
        self.lower_bound = p1
        self.upper_bound = p2
        self.dl = dl
        self.filename = filename
        self.file_extension = fileext
        self.iterations = iterations
        self.time = time
        self.outputs = outputs

    def _calculate_upper_bound(
        self, start: npt.NDArray[np.int32], step: npt.NDArray[np.int32], size: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int32]:
        return start + step * np.ceil(size / step)

    def build(self, model: Model, grid: FDTDGrid):
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f"{self.params_str()} do not add snapshots to subgrids.")

        uip = self._create_uip(grid)
        discretised_lower_bound, discretised_upper_bound = uip.check_output_object_bounds(
            self.lower_bound, self.upper_bound, self.params_str()
        )
        discretised_dl = uip.discretise_static_point(self.dl)

        snapshot_size = discretised_upper_bound - discretised_lower_bound

        # If p2 does not line up with the set discretisation, the actual
        # maximum element accessed in the grid will be this upper bound.
        upper_bound = self._calculate_upper_bound(
            discretised_lower_bound, discretised_dl, snapshot_size
        )

        # Each coordinate may need a different method to correct p2.
        # Therefore, this check needs to be repeated after each
        # correction has been applied.
        while any(discretised_upper_bound < upper_bound):
            try:
                uip.point_within_bounds(
                    upper_bound, f"[{upper_bound[0]}, {upper_bound[1]}, {upper_bound[2]}]"
                )
                upper_bound_within_grid = True
            except ValueError:
                upper_bound_within_grid = False

            # Ideally extend p2 up to the correct upper bound. This will
            # not change the snapshot output.
            if upper_bound_within_grid:
                discretised_upper_bound = upper_bound
                upper_bound_continuous = discretised_upper_bound * grid.dl
                logger.warning(
                    f"{self.params_str()} upper bound not aligned with discretisation. Updating 'p2'"
                    f" to {upper_bound_continuous}"
                )
            # If the snapshot size cannot be increased, the
            # discretisation may need reducing. E.g. for snapshots of 2D
            # models.
            elif any(discretised_dl > snapshot_size):
                discretised_dl = np.where(
                    discretised_dl > snapshot_size, snapshot_size, discretised_dl
                )
                upper_bound = self._calculate_upper_bound(
                    discretised_lower_bound, discretised_dl, snapshot_size
                )
                dl_continuous = discretised_dl * grid.dl
                logger.warning(
                    f"{self.params_str()} current bounds and discretisation would go outside"
                    f" domain. As discretisation is larger than the snapshot size in at least one"
                    f" dimension, limiting 'dl' to {dl_continuous}"
                )
            # Otherwise, limit p2 to the discretisation step below the
            # current snapshot size. This will reduce the size of the
            # snapshot by 1 in the effected dimension(s), but avoid out
            # of memory access.
            else:
                discretised_upper_bound = np.where(
                    discretised_upper_bound < upper_bound,
                    upper_bound - discretised_dl,
                    discretised_upper_bound,
                )
                snapshot_size = discretised_upper_bound - discretised_lower_bound
                upper_bound = self._calculate_upper_bound(
                    discretised_lower_bound, discretised_dl, snapshot_size
                )
                upper_bound_continuous = discretised_upper_bound * grid.dl
                logger.warning(
                    f"{self.params_str()} current bounds and discretisation would go outside"
                    f" domain. Limiting 'p2' to {upper_bound_continuous}"
                )

                # Raise error to prevent an infinite loop. This is here
                # as a precaution, it shouldn't be needed.
                if any(discretised_upper_bound < upper_bound):
                    raise ValueError(f"{self.params_str()} invalid snapshot.")

        if any(discretised_dl < 0):
            raise ValueError(f"{self.params_str()} the step size should not be less than zero.")
        if any(discretised_dl < 1):
            raise ValueError(
                f"{self.params_str()} the step size should not be less than the spatial discretisation."
            )

        if self.iterations is not None and self.time is not None:
            logger.warning(
                f"{self.params_str()} Time and iterations were both specified, using 'iterations'"
            )

        # If number of iterations given
        if self.iterations is not None:
            if self.iterations <= 0 or self.iterations > grid.iterations:
                raise ValueError(f"{self.params_str()} time value is not valid.")

        # If time value given
        elif self.time is not None:
            if self.time > 0:
                self.iterations = round_int((self.time / grid.dt)) + 1
            else:
                raise ValueError(f"{self.params_str()} time value must be greater than zero.")

        # No iteration or time value given
        else:
            raise ValueError(f"{self} specify a time or number of iterations")

        if self.file_extension is None:
            self.file_extension = SnapshotUser.fileexts[0]
        elif self.file_extension not in SnapshotUser.fileexts:
            raise ValueError(
                f"'{self.file_extension}' is not a valid format for a snapshot file."
                f" Valid options are: {' '.join(SnapshotUser.fileexts)}."
            )

        # TODO: Allow VTKHDF files when they are implemented
        if isinstance(grid, MPIGrid) and self.file_extension != ".h5":
            raise ValueError(
                f"{self.params_str()} currently only '.h5' snapshots are compatible with MPI."
            )

        if self.outputs is None:
            outputs = dict.fromkeys(SnapshotUser.allowableoutputs, True)
        else:
            outputs = dict.fromkeys(SnapshotUser.allowableoutputs, False)
            # Check and set output names
            for output in self.outputs:
                if output not in SnapshotUser.allowableoutputs.keys():
                    raise ValueError(
                        f"{self.params_str()} contains an output type that is not"
                        " allowable. Allowable outputs in current context are "
                        f"{', '.join(SnapshotUser.allowableoutputs.keys())}."
                    )
                else:
                    outputs[output] = True

        snapshot = model.add_snapshot(
            grid,
            discretised_lower_bound,
            discretised_upper_bound,
            discretised_dl,
            self.iterations,
            self.filename,
            self.file_extension,
            outputs,
        )

        if snapshot is not None:
            p1 = uip.round_to_grid_static_point(self.lower_bound)
            p2 = uip.round_to_grid_static_point(self.upper_bound)
            dl = uip.round_to_grid_static_point(self.dl)

            logger.info(
                f"{self.grid_name(grid)}Snapshot from"
                f" {p1[0]:g}m, {p1[1]:g}m, {p1[2]:g}m, to"
                f" {p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m, discretisation"
                f" {dl[0]:g}m, {dl[1]:g}m, {dl[2]:g}m, at"
                f" {snapshot.time * grid.dt:g} secs with field outputs"
                f" {', '.join([k for k, v in outputs.items() if v])} "
                f" and filename {snapshot.filename}{snapshot.fileext}"
                " will be created."
            )


class GeometryView(OutputUserObject):
    """Outputs to file(s) information about the geometry (mesh) of model.

    The geometry information is saved in Visual Toolkit (VTK) formats.

    Attributes:
        p1: tuple required for lower left (x,y,z) coordinates of volume of
                geometry view in metres.
        p2: tuple required for upper right (x,y,z) coordinates of volume of
                geometry view in metres.
        dl: tuple required for spatial discretisation of geometry view in metres.
        output_tuple: string required for per-cell 'n' (normal) or per-cell-edge
                        'f' (fine) geometry views.
        filename: string required for filename where geometry view will be
                    stored in the same directory as input file.
    """

    @property
    def order(self):
        return 17

    @property
    def hash(self):
        return "#geometry_view"

    def __init__(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        dl: Tuple[float, float, float],
        output_type: str,
        filename: str,
    ):
        super().__init__(p1=p1, p2=p2, dl=dl, filename=filename, output_type=output_type)
        self.lower_bound = p1
        self.upper_bound = p2
        self.dl = dl
        self.filename = filename
        self.output_type = output_type

    def build(self, model: Model, grid: FDTDGrid):
        uip = self._create_uip(grid)
        discretised_lower_bound, discretised_upper_bound = uip.check_output_object_bounds(
            self.lower_bound, self.upper_bound, self.params_str()
        )
        discretised_dl = uip.discretise_static_point(self.dl)

        if any(discretised_dl < 0):
            raise ValueError(f"{self.params_str()} the step size should not be less than zero.")
        if any(discretised_dl > grid.size):
            raise ValueError(
                f"{self.params_str()} the step size should be less than the domain size."
            )
        if any(discretised_dl < 1):
            raise ValueError(
                f"{self.params_str()} the step size should not be less than the spatial"
                " discretisation."
            )
        if self.output_type == "f" and any(discretised_dl != 1):
            raise ValueError(
                f"{self.params_str()} requires the spatial discretisation for the geometry view to"
                " be the same as the model for geometry view of type f (fine)."
            )

        if self.output_type == "n":
            g = model.add_geometry_view_voxels(
                grid,
                discretised_lower_bound,
                discretised_upper_bound,
                discretised_dl,
                self.filename,
            )
        elif self.output_type == "f":
            g = model.add_geometry_view_lines(
                grid,
                discretised_lower_bound,
                discretised_upper_bound,
                self.filename,
            )
        else:
            raise ValueError(
                f"{self.params_str()} requires type to be either n (normal) or f (fine)."
            )

        if g is not None:
            p1 = uip.round_to_grid_static_point(self.lower_bound)
            p2 = uip.round_to_grid_static_point(self.upper_bound)
            dl = discretised_dl * grid.dl

            logger.info(
                f"{self.grid_name(grid)}Geometry view from"
                f" {p1[0]:g}m, {p1[1]:g}m, {p1[2]:g}m,"
                f" to {p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m,"
                f" discretisation {dl[0]:g}m, {dl[1]:g}m, {dl[2]:g}m,"
                f" with filename base {g.filenamebase} created."
            )


class GeometryObjectsWrite(OutputUserObject):
    """Writes geometry generated in a model to file which can be imported into
        other models.

    Attributes:
        p1: tuple required for lower left (x,y,z) coordinates of volume of
                output in metres.
        p2: tuple required for upper right (x,y,z) coordinates of volume of
                output in metres.
        filename: string required for filename where output will be stored in
                    the same directory as input file.
    """

    @property
    def order(self):
        return 18

    @property
    def hash(self):
        return "#geometry_objects_write"

    def __init__(
        self, p1: Tuple[float, float, float], p2: Tuple[float, float, float], filename: str
    ):
        super().__init__(p1=p1, p2=p2, filename=filename)
        self.lower_bound = p1
        self.upper_bound = p2
        self.basefilename = filename

    def build(self, model: Model, grid: FDTDGrid):
        if isinstance(grid, SubGridBaseGrid):
            raise ValueError(f"{self.params_str()} do not add geometry objects to subgrids.")

        uip = self._create_uip(grid)

        discretised_lower_bound, discretised_upper_bound = uip.check_output_object_bounds(
            self.lower_bound, self.upper_bound, self.params_str()
        )

        g = model.add_geometry_object(
            grid, discretised_lower_bound, discretised_upper_bound, self.basefilename
        )

        if g is not None:
            p1 = uip.round_to_grid_static_point(self.lower_bound)
            p2 = uip.round_to_grid_static_point(self.upper_bound)

            logger.info(
                f"Geometry objects in the volume from {p1[0]:g}m,"
                f" {p1[1]:g}m, {p1[2]:g}m, to {p2[0]:g}m, {p2[1]:g}m,"
                f" {p2[2]:g}m, will be written to {g.filename_hdf5},"
                f" with materials written to {g.filename_materials}"
            )
