# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
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
import itertools
import logging
import sys

import humanize
import numpy as np
import psutil
from colorama import Fore, Style, init

init()

from terminaltables import SingleTable
from tqdm import tqdm

import gprMax.config as config

from .cython.yee_cell_build import build_electric_components, build_magnetic_components
from .fields_outputs import write_hdf5_outputfile
from .geometry_outputs import save_geometry_views
from .grid import dispersion_analysis
from .hash_cmds_file import parse_hash_commands
from .materials import process_materials
from .pml import CFS, build_pml, print_pml_info
from .scene import Scene
from .snapshots import save_snapshots
from .utilities.host_info import mem_check_build_all, mem_check_run_all, set_omp_threads
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


class ModelBuildRun:
    """Builds and runs (solves) a model."""

    def __init__(self, G):
        self.G = G
        # Monitor memory usage
        self.p = None

        # Set number of OpenMP threads to physical threads at this point to be
        # used with threaded model building methods, e.g. fractals. Can be
        # changed by #num_threads command in input file or via API later for
        # use with CPU solver.
        config.get_model_config().ompthreads = set_omp_threads(config.get_model_config().ompthreads)

    def build(self):
        """Builds the Yee cells for a model."""

        G = self.G

        # Monitor memory usage
        self.p = psutil.Process()

        # Normal model reading/building process; bypassed if geometry information to be reused
        self.reuse_geometry() if config.get_model_config().reuse_geometry else self.build_geometry()

        logger.info(f"\nOutput directory: {config.get_model_config().output_file_path.parent.resolve()}")

        # Adjust position of simple sources and receivers if required
        if G.srcsteps[0] != 0 or G.srcsteps[1] != 0 or G.srcsteps[2] != 0:
            for source in itertools.chain(G.hertziandipoles, G.magneticdipoles):
                if config.model_num == 0:
                    if (
                        source.xcoord + G.srcsteps[0] * config.sim_config.model_end < 0
                        or source.xcoord + G.srcsteps[0] * config.sim_config.model_end > G.nx
                        or source.ycoord + G.srcsteps[1] * config.sim_config.model_end < 0
                        or source.ycoord + G.srcsteps[1] * config.sim_config.model_end > G.ny
                        or source.zcoord + G.srcsteps[2] * config.sim_config.model_end < 0
                        or source.zcoord + G.srcsteps[2] * config.sim_config.model_end > G.nz
                    ):
                        logger.exception("Source(s) will be stepped to a position outside the domain.")
                        raise ValueError
                source.xcoord = source.xcoordorigin + config.model_num * G.srcsteps[0]
                source.ycoord = source.ycoordorigin + config.model_num * G.srcsteps[1]
                source.zcoord = source.zcoordorigin + config.model_num * G.srcsteps[2]
        if G.rxsteps[0] != 0 or G.rxsteps[1] != 0 or G.rxsteps[2] != 0:
            for receiver in G.rxs:
                if config.model_num == 0:
                    if (
                        receiver.xcoord + G.rxsteps[0] * config.sim_config.model_end < 0
                        or receiver.xcoord + G.rxsteps[0] * config.sim_config.model_end > G.nx
                        or receiver.ycoord + G.rxsteps[1] * config.sim_config.model_end < 0
                        or receiver.ycoord + G.rxsteps[1] * config.sim_config.model_end > G.ny
                        or receiver.zcoord + G.rxsteps[2] * config.sim_config.model_end < 0
                        or receiver.zcoord + G.rxsteps[2] * config.sim_config.model_end > G.nz
                    ):
                        logger.exception("Receiver(s) will be stepped to a position outside the domain.")
                        raise ValueError
                receiver.xcoord = receiver.xcoordorigin + config.model_num * G.rxsteps[0]
                receiver.ycoord = receiver.ycoordorigin + config.model_num * G.rxsteps[1]
                receiver.zcoord = receiver.zcoordorigin + config.model_num * G.rxsteps[2]

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
                go.write_hdf5(G, pbar)
                pbar.close()
            logger.info("")

    def build_geometry(self):
        G = self.G

        logger.info(config.get_model_config().inputfilestr)

        # Build objects in the scene and check memory for building
        self.build_scene()

        # Print info on any subgrids
        for sg in G.subgrids:
            sg.print_info()

        # Combine available grids
        grids = [G] + G.subgrids

        # Check for dispersive materials (and specific type)
        for grid in grids:
            if config.get_model_config().materials["maxpoles"] != 0:
                config.get_model_config().materials["drudelorentz"] = any(
                    [m for m in grid.materials if "drude" in m.type or "lorentz" in m.type]
                )

        # Set data type if any dispersive materials (must be done before memory checks)
        if config.get_model_config().materials["maxpoles"] != 0:
            config.get_model_config().set_dispersive_material_types()

        # Check memory requirements to build model/scene (different to memory
        # requirements to run model when FractalVolumes/FractalSurfaces are
        # used as these can require significant additional memory)
        total_mem_build, mem_strs_build = mem_check_build_all(grids)

        # Check memory requirements to run model
        total_mem_run, mem_strs_run = mem_check_run_all(grids)

        if total_mem_build > total_mem_run:
            logger.info(
                f'\nMemory required (estimated): {" + ".join(mem_strs_build)} + '
                f"~{humanize.naturalsize(config.get_model_config().mem_overhead)} "
                f"overhead = {humanize.naturalsize(total_mem_build)}"
            )
        else:
            logger.info(
                f'\nMemory required (estimated): {" + ".join(mem_strs_run)} + '
                f"~{humanize.naturalsize(config.get_model_config().mem_overhead)} "
                f"overhead = {humanize.naturalsize(total_mem_run)}"
            )

        # Build grids
        gridbuilders = [GridBuilder(grid) for grid in grids]
        for gb in gridbuilders:
            # Set default CFS parameter for PMLs if not user provided
            if not gb.grid.pmls["cfs"]:
                gb.grid.pmls["cfs"] = [CFS()]
            logger.info(print_pml_info(gb.grid))
            if not all(value == 0 for value in gb.grid.pmls["thickness"].values()):
                gb.build_pmls()
            if gb.grid.averagevolumeobjects:
                gb.build_components()
            gb.tm_grid_update()
            gb.update_voltage_source_materials()
            gb.grid.initialise_field_arrays()
            gb.grid.initialise_std_update_coeff_arrays()
            if config.get_model_config().materials["maxpoles"] > 0:
                gb.grid.initialise_dispersive_arrays()
                gb.grid.initialise_dispersive_update_coeff_array()
            gb.build_materials()

            # Check to see if numerical dispersion might be a problem
            results = dispersion_analysis(gb.grid)
            if results["error"]:
                logger.warning(
                    f"\nNumerical dispersion analysis [{gb.grid.name}] " f"not carried out as {results['error']}"
                )
            elif results["N"] < config.get_model_config().numdispersion["mingridsampling"]:
                logger.exception(
                    f"\nNon-physical wave propagation in [{gb.grid.name}] "
                    f"detected. Material '{results['material'].ID}' "
                    f"has wavelength sampled by {results['N']} cells, "
                    f"less than required minimum for physical wave "
                    f"propagation. Maximum significant frequency "
                    f"estimated as {results['maxfreq']:g}Hz"
                )
                raise ValueError
            elif (
                results["deltavp"]
                and np.abs(results["deltavp"]) > config.get_model_config().numdispersion["maxnumericaldisp"]
            ):
                logger.warning(
                    f"\n[{gb.grid.name}] has potentially significant "
                    f"numerical dispersion. Estimated largest physical "
                    f"phase-velocity error is {results['deltavp']:.2f}% "
                    f"in material '{results['material'].ID}' whose "
                    f"wavelength sampled by {results['N']} cells. "
                    f"Maximum significant frequency estimated as "
                    f"{results['maxfreq']:g}Hz"
                )
            elif results["deltavp"]:
                logger.info(
                    f"\nNumerical dispersion analysis [{gb.grid.name}]: "
                    f"estimated largest physical phase-velocity error is "
                    f"{results['deltavp']:.2f}% in material '{results['material'].ID}' "
                    f"whose wavelength sampled by {results['N']} cells. "
                    f"Maximum significant frequency estimated as "
                    f"{results['maxfreq']:g}Hz"
                )

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

    def build_scene(self):
        # API for multiple scenes / model runs
        scene = config.get_model_config().get_scene()

        # If there is no scene, process the hash commands
        if not scene:
            scene = Scene()
            config.sim_config.scenes.append(scene)
            # Parse the input file into user objects and add them to the scene
            scene = parse_hash_commands(scene)

        # Creates the internal simulation objects
        scene.create_internal_objects(self.G)

        return scene

    def write_output_data(self):
        """Writes output data, i.e. field data for receivers and snapshots to
        file(s).
        """

        # Write output data to file if they are any receivers in any grids
        sg_rxs = [True for sg in self.G.subgrids if sg.rxs]
        sg_tls = [True for sg in self.G.subgrids if sg.transmissionlines]
        if self.G.rxs or sg_rxs or self.G.transmissionlines or sg_tls:
            write_hdf5_outputfile(config.get_model_config().output_file_path_ext, self.G)

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
                f"\nModel {config.model_num + 1}/{config.sim_config.model_end} "
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
                platformname = " on " + " ".join(config.get_model_config().device["dev"].platform.name.split())
            else:
                solvername = "CUDA"
                platformname = ""
            
            devicename = (f'[{config.get_model_config().device["deviceID"]}] '
                          f'{" ".join(config.get_model_config().device["dev"].name().split())}')
            logger.basic(
                f"\nModel {config.model_num + 1}/{config.sim_config.model_end} "
                f"on {config.sim_config.hostinfo['hostname']} "
                f"with {solvername} backend using {devicename}{platformname}"
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
        # Add a string on GPU memory usage if applicable
        mem_str = (
            f" host + ~{humanize.naturalsize(solver.memused)} GPU"
            if config.sim_config.general["solver"] == "cuda"
            else ""
        )

        logger.info(f"\nMemory used (estimated): " + f"~{humanize.naturalsize(self.p.memory_full_info().uss)}{mem_str}")
        logger.info(
            f"Time taken: " + f"{humanize.precisedelta(datetime.timedelta(seconds=solver.solvetime), format='%0.4f')}"
        )


class GridBuilder:
    def __init__(self, grid):
        self.grid = grid

    def build_pmls(self):
        pbar = tqdm(
            total=sum(1 for value in self.grid.pmls["thickness"].values() if value > 0),
            desc=f"Building PML boundaries [{self.grid.name}]",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        for pml_id, thickness in self.grid.pmls["thickness"].items():
            if thickness > 0:
                build_pml(self.grid, pml_id, thickness)
                pbar.update()
        pbar.close()

    def build_components(self):
        # Build the model, i.e. set the material properties (ID) for every edge
        # of every Yee cell
        logger.info("")
        pbar = tqdm(
            total=2,
            desc=f"Building Yee cells [{self.grid.name}]",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        build_electric_components(self.grid.solid, self.grid.rigidE, self.grid.ID, self.grid)
        pbar.update()
        build_magnetic_components(self.grid.solid, self.grid.rigidH, self.grid.ID, self.grid)
        pbar.update()
        pbar.close()

    def tm_grid_update(self):
        if config.get_model_config().mode == "2D TMx":
            self.grid.tmx()
        elif config.get_model_config().mode == "2D TMy":
            self.grid.tmy()
        elif config.get_model_config().mode == "2D TMz":
            self.grid.tmz()

    def update_voltage_source_materials(self):
        # Process any voltage sources (that have resistance) to create a new
        # material at the source location
        for voltagesource in self.grid.voltagesources:
            voltagesource.create_material(self.grid)

    def build_materials(self):
        # Process complete list of materials - calculate update coefficients,
        # store in arrays, and build text list of materials/properties
        materialsdata = process_materials(self.grid)
        materialstable = SingleTable(materialsdata)
        materialstable.outer_border = False
        materialstable.justify_columns[0] = "right"

        logger.info(f"\nMaterials [{self.grid.name}]:")
        logger.info(materialstable.table)
