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
from typing import List

import humanize
import numpy as np
import psutil
from colorama import Fore, Style, init

from gprMax.grid.cuda_grid import CUDAGrid
from gprMax.grid.opencl_grid import OpenCLGrid
from gprMax.subgrids.grid import SubGridBaseGrid

init()

from tqdm import tqdm

import gprMax.config as config

from .fields_outputs import write_hdf5_outputfile
from .geometry_outputs import save_geometry_views
from .grid.fdtd_grid import FDTDGrid
from .snapshots import save_snapshots
from .utilities.host_info import set_omp_threads
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


class Model:
    """Builds and runs (solves) a model."""

    def __init__(self):
        self.title = ""

        self.gnx = 0
        self.gny = 0
        self.gnz = 0

        self.dt_mod = 1.0  # Time step stability factor

        self.iteration = 0  # Current iteration number
        self.iterations = 0  # Total number of iterations
        self.timewindow = 0.0

        self.G = self._create_grid()
        self.subgrids: List[SubGridBaseGrid] = []

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
    def dl(self) -> np.ndarray:
        return self.G.dl

    @dl.setter
    def dl(self, value: np.ndarray):
        self.G.dl = value

    @property
    def dt(self) -> float:
        return self.G.dt

    @dt.setter
    def dt(self, value: float):
        self.G.dt = value

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

        return grid

    def build(self):
        """Builds the Yee cells for a model."""

        G = self.G

        # Monitor memory usage
        self.p = psutil.Process()

        # Normal model reading/building process; bypassed if geometry information to be reused
        self.reuse_geometry() if config.get_model_config().reuse_geometry() else self.build_geometry()

        logger.info(
            f"\nOutput directory: {config.get_model_config().output_file_path.parent.resolve()}"
        )

        # Adjust position of simple sources and receivers if required
        model_num = config.sim_config.current_model
        G.update_simple_source_positions(step=model_num)
        G.update_receiver_positions(step=model_num)

        # Write files for any geometry views and geometry object outputs
        gvs = G.geometryviews + [gv for sg in G.subgrids for gv in sg.geometryviews]
        if not gvs and not G.geometryobjectswrite and config.sim_config.args.geometry_only:
            logger.exception("\nNo geometry views or geometry objects found.")
            raise ValueError
        save_geometry_views(gvs)

        if G.geometryobjectswrite:
            logger.info("")
            for i, go in enumerate(G.geometryobjectswrite):
                pbar = tqdm(
                    total=go.datawritesize,
                    unit="byte",
                    unit_scale=True,
                    desc=f"Writing geometry object file {i + 1}/{len(G.geometryobjectswrite)}, "
                    + f"{go.filename_hdf5.name}",
                    ncols=get_terminal_width() - 1,
                    file=sys.stdout,
                    disable=not config.sim_config.general["progressbars"],
                )
                go.write_hdf5(self.title, self.G, pbar)
                pbar.close()
            logger.info("")

    def build_geometry(self):
        logger.info(config.get_model_config().inputfilestr)
        # TODO: Make this correctly sets local nx, ny and nz when using MPI (likely use a function inside FDTDGrid/MPIGrid)
        self.G.nx = self.gnx
        self.G.ny = self.gny
        self.G.nz = self.gnz
        self.G.build()

    def reuse_geometry(self):
        s = (
            f"\n--- Model {config.get_model_config().appendmodelnumber}/{config.sim_config.model_end}, "
            f"input file (not re-processed, i.e. geometry fixed): "
            f"{config.sim_config.input_file_path}"
        )
        config.get_model_config().inputfilestr = (
            Fore.GREEN + f"{s} {'-' * (get_terminal_width() - 1 - len(s))}\n" + Style.RESET_ALL
        )
        logger.basic(config.get_model_config().inputfilestr)
        for grid in [self.G] + self.G.subgrids:
            grid.iteration = 0  # Reset current iteration number
            grid.reset_fields()

    def write_output_data(self):
        """Writes output data, i.e. field data for receivers and snapshots to
        file(s).
        """

        # Write output data to file if they are any receivers in any grids
        sg_rxs = [True for sg in self.G.subgrids if sg.rxs]
        sg_tls = [True for sg in self.G.subgrids if sg.transmissionlines]
        if self.G.rxs or sg_rxs or self.G.transmissionlines or sg_tls:
            write_hdf5_outputfile(
                config.get_model_config().output_file_path_ext, self.title, self.G
            )

        # Write any snapshots to file for each grid
        for grid in [self.G] + self.G.subgrids:
            if grid.snapshots:
                save_snapshots(grid)

    def solve(self, solver):
        """Solve using FDTD method.

        Args:
            solver: solver object.
        """

        # Print information about and check OpenMP threads
        if config.sim_config.general["solver"] == "cpu":
            logger.basic(
                f"\nModel {config.sim_config.current_model + 1}/{config.sim_config.model_end} "
                f"on {config.sim_config.hostinfo['hostname']} "
                f"with OpenMP backend using {config.get_model_config().ompthreads} thread(s)"
            )
            if config.get_model_config().ompthreads > config.sim_config.hostinfo["physicalcores"]:
                logger.warning(
                    f"You have specified more threads ({config.get_model_config().ompthreads}) "
                    f"than available physical CPU cores ({config.sim_config.hostinfo['physicalcores']}). "
                    f"This may lead to degraded performance."
                )
        elif config.sim_config.general["solver"] in ["cuda", "opencl"]:
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
            else:
                solvername = "CUDA"
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
                range(self.G.iterations),
                desc="|--->",
                ncols=get_terminal_width() - 1,
                file=sys.stdout,
                disable=not config.sim_config.general["progressbars"],
            )
        else:
            iterator = range(self.G.iterations)

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
            f"\nMemory used (estimated): "
            + f"~{humanize.naturalsize(self.p.memory_full_info().uss)}{mem_str}"
        )
        logger.info(
            f"Time taken: "
            + f"{humanize.precisedelta(datetime.timedelta(seconds=solver.solvetime), format='%0.4f')}"
        )
