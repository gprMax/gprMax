# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import datetime
import logging
import sys
from typing import Dict, List, Optional, Sequence

import humanize
import numpy as np
import numpy.typing as npt
import psutil
from colorama import Fore, Style, init

from gprMax.grid.cuda_grid import CUDAGrid
from gprMax.grid.opencl_grid import OpenCLGrid
from gprMax.grid.metal_grid import MetalGrid
from gprMax.output_controllers.geometry_objects import GeometryObject
from gprMax.output_controllers.geometry_view_lines import GeometryViewLines
from gprMax.output_controllers.geometry_view_voxels import GeometryViewVoxels
from gprMax.output_controllers.geometry_views import GeometryView, save_geometry_views
from gprMax.subgrids.grid import SubGridBaseGrid

init()

from tqdm import tqdm

import gprMax.config as config
from gprMax.fields_outputs import write_hdf5_outputfile
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.snapshots import Snapshot, save_snapshots
from gprMax.utilities.host_info import mem_check_build_all, mem_check_run_all, set_omp_threads
from gprMax.utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


class Model:
    """Builds and runs (solves) a model."""

    def __init__(self):
        self.title = ""

        self.dt_mod = 1.0  # Time step stability factor

        self.iteration = 0  # Current iteration number

        self.G = self._create_grid()
        self.subgrids: List[SubGridBaseGrid] = []

        self.geometryviews: List[GeometryView] = []
        self.geometryobjects: List[GeometryObject] = []

        # Monitor memory usage
        self.p = None

        # Set number of OpenMP threads to physical threads at this point to be
        # used with threaded model building methods, e.g. fractals. Can be
        # changed by the user via #num_threads command in input file or via API
        # later for use with CPU solver.
        config.get_model_config().ompthreads = set_omp_threads(config.get_model_config().ompthreads)

    @property
    def nx(self) -> int:
        return self.G.nx

    @nx.setter
    def nx(self, value: int):
        self.G.nx = value

    @property
    def ny(self) -> int:
        return self.G.ny

    @ny.setter
    def ny(self, value: int):
        self.G.ny = value

    @property
    def nz(self) -> int:
        return self.G.nz

    @nz.setter
    def nz(self, value: int):
        self.G.nz = value

    @property
    def dx(self) -> float:
        return self.G.dl[0]

    @dx.setter
    def dx(self, value: float):
        self.G.dl[0] = value

    @property
    def dy(self) -> float:
        return self.G.dl[0]

    @dy.setter
    def dy(self, value: float):
        self.G.dl[1] = value

    @property
    def dz(self) -> float:
        return self.G.dl[0]

    @dz.setter
    def dz(self, value: float):
        self.G.dl[2] = value

    @property
    def dl(self) -> npt.NDArray[np.float64]:
        return self.G.dl

    @dl.setter
    def dl(self, value: npt.NDArray[np.float64]):
        self.G.dl = value

    @property
    def dt(self) -> float:
        return self.G.dt

    @dt.setter
    def dt(self, value: float):
        self.G.dt = value

    @property
    def iterations(self) -> int:
        return self.G.iterations

    @iterations.setter
    def iterations(self, value: int):
        self.G.iterations = value

    @property
    def timewindow(self) -> float:
        return self.G.timewindow

    @timewindow.setter
    def timewindow(self, value: float):
        self.G.timewindow = value

    @property
    def srcsteps(self) -> npt.NDArray[np.int32]:
        return self.G.srcsteps

    @srcsteps.setter
    def srcsteps(self, value: npt.NDArray[np.int32]):
        self.G.srcsteps = value

    @property
    def rxsteps(self) -> npt.NDArray[np.int32]:
        return self.G.rxsteps

    @rxsteps.setter
    def rxsteps(self, value: npt.NDArray[np.int32]):
        self.G.rxsteps = value

    def _create_grid(self) -> FDTDGrid:
        """Create grid object according to solver.

        Returns:
            grid: FDTDGrid class describing a grid in a model.
        """
        if config.sim_config.general["solver"] == "cpu":
            grid = FDTDGrid()
        elif config.sim_config.general["solver"] == "cuda":
            grid = CUDAGrid()
        elif config.sim_config.general["solver"] == "opencl":
            grid = OpenCLGrid()
        elif config.sim_config.general["solver"] == "metal":
            grid = MetalGrid()

        return grid

    def set_size(self, size: npt.NDArray[np.int32]):
        """Set size of the model.

        Args:
            size: Array to set the size (3 dimensional).
        """
        self.nx, self.ny, self.nz = size

    def add_geometry_object(
        self,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        basefilename: str,
    ) -> Optional[GeometryObject]:
        """Add a geometry object to the model.

        Args:
            grid: Grid to create a geometry object for.
            start: Lower extent of the geometry object (x, y, z).
            stop: Upper extent of the geometry object (x, y, z).
            basefilename: Output filename of the geometry object.

        Returns:
            geometry_object: The created geometry object.
        """
        geometry_object = GeometryObject(
            grid, start[0], start[1], start[2], stop[0], stop[1], stop[2], basefilename
        )
        self.geometryobjects.append(geometry_object)
        return geometry_object

    def add_geometry_view_voxels(
        self,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        dl: npt.NDArray[np.int32],
        filename: str,
    ) -> Optional[GeometryViewVoxels]:
        """Add a voxel geometry view to the model.

        Args:
            grid: Grid to create a geometry view for.
            start: Lower extent of the geometry view (x, y, z).
            stop: Upper extent of the geometry view (x, y, z).
            dl: Discritisation of the geometry view (x, y, z).
            filename: Output filename of the geometry view.

        Returns:
            geometry_view: The created geometry view.
        """
        geometry_view = GeometryViewVoxels(
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

    def add_geometry_view_lines(
        self,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        filename: str,
    ) -> Optional[GeometryViewLines]:
        """Add a lines geometry view to the model.

        Args:
            grid: Grid to create a geometry view for.
            start: Lower extent of the geometry view (x, y, z).
            stop: Upper extent of the geometry view (x, y, z).
            filename: Output filename of the geometry view.

        Returns:
            geometry_view: The created geometry view.
        """
        geometry_view = GeometryViewLines(
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

    def add_snapshot(
        self,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        dl: npt.NDArray[np.int32],
        time: int,
        filename: str,
        fileext: str,
        outputs: Dict[str, bool],
    ) -> Optional[Snapshot]:
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
            snapshot: The created snapshot.
        """
        snapshot = Snapshot(
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

    def build(self):
        """Builds the Yee cells for a model."""

        # Monitor memory usage
        self.p = psutil.Process()

        # Normal model reading/building process; bypassed if geometry information to be reused
        if config.get_model_config().reuse_geometry():
            self.reuse_geometry()
        else:
            self.build_geometry()

        logger.info(
            f"Output directory: {config.get_model_config().output_file_path.parent.resolve()}\n"
        )

        self.G.update_sources_and_recievers()

        self._output_geometry()

    def _output_geometry(self):
        # Write files for any geometry views and geometry object outputs
        if (
            not self.geometryviews
            and not self.geometryobjects
            and config.sim_config.args.geometry_only
        ):
            logger.warning(
                "Geometry only run specified, but no geometry views or geometry objects found."
            )
            return

        save_geometry_views(self.geometryviews)

        if self.geometryobjects:
            logger.info("")
            for i, go in enumerate(self.geometryobjects):
                pbar = tqdm(
                    total=go.datawritesize,
                    unit="byte",
                    unit_scale=True,
                    desc=f"Writing geometry object file {i + 1}/{len(self.geometryobjects)}, "
                    + f"{go.filename_hdf5.name}",
                    ncols=get_terminal_width() - 1,
                    file=sys.stdout,
                    disable=not config.sim_config.general["progressbars"],
                )
                go.write_hdf5(self.title, pbar)
                pbar.close()
            logger.info("")

    def build_geometry(self):
        logger.info(config.get_model_config().inputfilestr)

        # Print info on any subgrids
        for subgrid in self.subgrids:
            subgrid.print_info()

        # Combine available grids
        grids = [self.G] + self.subgrids

        self._check_for_dispersive_materials(grids)
        self._check_memory_requirements(grids)

        for grid in grids:
            grid.build()
            grid.dispersion_analysis(self.iterations)

    def _check_for_dispersive_materials(self, grids: Sequence[FDTDGrid]):
        # Check for dispersive materials (and specific type)
        if config.get_model_config().materials["maxpoles"] != 0:
            # TODO: This sets materials["drudelorentz"] based only the
            # last grid/subgrid. Is that correct?
            for grid in grids:
                config.get_model_config().materials["drudelorentz"] = any(
                    [m for m in grid.materials if "drude" in m.type or "lorentz" in m.type]
                )

            # Set data type if any dispersive materials (must be done before memory checks)
            config.get_model_config().set_dispersive_material_types()

    def _check_memory_requirements(self, grids: Sequence[FDTDGrid]):
        # Check memory requirements to build model/scene (different to memory
        # requirements to run model when FractalVolumes/FractalSurfaces are
        # used as these can require significant additional memory)
        total_mem_build, mem_strs_build = mem_check_build_all(grids)

        # Check memory requirements to run model
        total_mem_run, mem_strs_run = mem_check_run_all(grids)

        if total_mem_build > total_mem_run:
            logger.info(
                f'Memory required (estimated): {" + ".join(mem_strs_build)} + '
                f"~{humanize.naturalsize(config.get_model_config().mem_overhead)} "
                f"overhead = {humanize.naturalsize(total_mem_build)}\n"
            )
        else:
            logger.info(
                f'Memory required (estimated): {" + ".join(mem_strs_run)} + '
                f"~{humanize.naturalsize(config.get_model_config().mem_overhead)} "
                f"overhead = {humanize.naturalsize(total_mem_run)}\n"
            )

    def reuse_geometry(self):
        s = (
            f"\n--- Model {config.get_model_config().appendmodelnumber}/{config.sim_config.model_end}, "
            f"input file (not re-processed, i.e. geometry fixed): "
            f"{config.sim_config.input_file_path}"
        )
        config.get_model_config().inputfilestr = (
            Fore.GREEN + f"{s} {'-' * (get_terminal_width() - 1 - len(s))}\n\n" + Style.RESET_ALL
        )
        logger.basic(config.get_model_config().inputfilestr)
        self.iteration = 0  # Reset current iteration number
        for grid in [self.G] + self.subgrids:
            grid.reset_fields()

    def write_output_data(self):
        """Writes output data, i.e. field data for receivers and snapshots to
        file(s).
        """

        # Write output data to file if they are any receivers in any grids
        sg_rxs = [True for sg in self.subgrids if sg.rxs]
        sg_tls = [True for sg in self.subgrids if sg.transmissionlines]
        if self.G.rxs or sg_rxs or self.G.transmissionlines or sg_tls:
            write_hdf5_outputfile(config.get_model_config().output_file_path_ext, self.title, self)

        # Write any snapshots to file for each grid
        for grid in [self.G] + self.subgrids:
            if grid.snapshots:
                save_snapshots(grid.snapshots)

    def solve(self, solver):
        """Solve using FDTD method.

        Args:
            solver: solver object.
        """

        # Print information about and check OpenMP threads
        if config.sim_config.general["solver"] == "cpu":
            logger.basic(
                f"Model {config.sim_config.current_model + 1}/{config.sim_config.model_end} "
                f"on {config.sim_config.hostinfo['hostname']} "
                f"with OpenMP backend using {config.get_model_config().ompthreads} thread(s)"
            )
            if config.get_model_config().ompthreads > config.sim_config.hostinfo["physicalcores"]:
                logger.warning(
                    f"You have specified more threads ({config.get_model_config().ompthreads}) "
                    f"than available physical CPU cores ({config.sim_config.hostinfo['physicalcores']}). "
                    f"This may lead to degraded performance."
                )
        elif config.sim_config.general["solver"] in ["cuda", "opencl", "metal"]:
            if config.sim_config.general["solver"] == "opencl":
                solvername = "OpenCL"
                platformname = (
                    " ".join(config.get_model_config().device["dev"].platform.name.split())
                    + " with "
                )
                devicename = (
                    f'Device {config.get_model_config().device["deviceID"]}: '
                    f'{" ".join(config.get_model_config().device["dev"].name.split())}'
                )
            elif config.sim_config.general["solver"] == "cuda":
                solvername = "CUDA"
                platformname = ""
                devicename = (
                    f'Device {config.get_model_config().device["deviceID"]}: '
                    f'{" ".join(config.get_model_config().device["dev"].name().split())}'
                )
            else:  # Metal
                solvername = "Apple Metal"
                platformname = ""
                devicename = (
                    f'Device {config.get_model_config().device["deviceID"]}: '
                    f'{" ".join(config.get_model_config().device["dev"].name().split())}'
                )
            logger.basic(
                f"\nModel {config.sim_config.current_model + 1}/{config.sim_config.model_end} "
                f"solving on {config.sim_config.hostinfo['hostname']} "
                f"with {solvername} backend using {platformname}{devicename}"
            )

        # Prepare iterator
        if config.sim_config.general["progressbars"]:
            iterator = tqdm(
                range(self.iterations),
                desc="|--->",
                ncols=get_terminal_width() - 1,
                file=sys.stdout,
                disable=not config.sim_config.general["progressbars"],
            )
        else:
            iterator = range(self.iterations)

        # Run solver
        solver.solve(iterator)

        # Write output data, i.e. field data for receivers and snapshots to file(s)
        self.write_output_data()

        # Print information about memory usage and solving time for a model
        # Add a string on device (GPU) memory usage if applicable
        mem_str = ""
        if config.sim_config.general["solver"] == "cuda":
            mem_str = f" host + ~{humanize.naturalsize(solver.memused)} device"
        elif config.sim_config.general["solver"] == "opencl":
            mem_str = f" host + unknown for device"

        logger.info(
            f"Memory used (estimated): "
            + f"~{humanize.naturalsize(self.p.memory_full_info().uss)}{mem_str}"
        )
        logger.info(
            f"Time taken: "
            + f"{humanize.precisedelta(datetime.timedelta(seconds=solver.solvetime), format='%0.4f')}\n"
        )
