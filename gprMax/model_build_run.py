# Copyright (C) 2015-2019: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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
from importlib import import_module
import itertools
import logging
import os
import psutil
import sys

from colorama import init
from colorama import Fore
from colorama import Style
init()
import cython
import numpy as np
from terminaltables import SingleTable
from tqdm import tqdm
from pathlib import Path

import gprMax.config as config
from .cuda.fields_updates import kernel_template_fields
from .cuda.snapshots import kernel_template_store_snapshot
from .cuda.source_updates import kernel_template_sources
from .cython.yee_cell_build import build_electric_components
from .cython.yee_cell_build import build_magnetic_components
from .exceptions import GeneralError
from .fields_outputs import kernel_template_store_outputs
from .fields_outputs import write_hdf5_outputfile
from .grid import FDTDGrid
from .grid import dispersion_analysis
from .hash_cmds_geometry import process_geometrycmds
from .hash_cmds_file import process_python_include_code
from .hash_cmds_file import write_processed_file
from .hash_cmds_file import check_cmd_names
from .hash_cmds_file import parse_hash_commands
from .hash_cmds_singleuse import process_singlecmds
from .hash_cmds_multiuse import process_multicmds
from .materials import Material
from .materials import process_materials
from .pml import CFS
from .pml import PML
from .pml import build_pml
from .pml import pml_information
from .receivers import initialise_rx_arrays_gpu
from .receivers import get_rx_array_gpu
from .receivers import Rx
from .scene import Scene
from .snapshots import Snapshot
from .snapshots import initialise_snapshot_array_gpu
from .snapshots import get_snapshot_array_gpu
from .solvers import create_solver
from .sources import initialise_src_arrays_gpu
from .utilities import get_terminal_width
from .utilities import human_size
from .utilities import mem_check
from .utilities import open_path_file
from .utilities import round32
from .utilities import set_omp_threads
from .utilities import timer

log = logging.getLogger(__name__)


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
        config.model_configs[self.G.model_num].ompthreads = set_omp_threads(config.model_configs[self.G.model_num].ompthreads)

    def build(self):
        """Builds the Yee cells for a model."""

        G = self.G

        # Monitor memory usage
        self.p = psutil.Process()

        # Normal model reading/building process; bypassed if geometry information to be reused
        self.reuse_geometry() if config.model_configs[G.model_num].reuse_geometry else self.build_geometry()

        log.info(f'\nOutput directory: {config.model_configs[G.model_num].output_file_path.parent.resolve()}')

        # Adjust position of simple sources and receivers if required
        if G.srcsteps[0] != 0 or G.srcsteps[1] != 0 or G.srcsteps[2] != 0:
            for source in itertools.chain(G.hertziandipoles, G.magneticdipoles):
                if config.model_configs[G.model_num] == 0:
                    if (source.xcoord + G.srcsteps[0] * config.sim_config.model_end < 0 or
                        source.xcoord + G.srcsteps[0] * config.sim_config.model_end > G.nx or
                        source.ycoord + G.srcsteps[1] * config.sim_config.model_end < 0 or
                        source.ycoord + G.srcsteps[1] * config.sim_config.model_end > G.ny or
                        source.zcoord + G.srcsteps[2] * config.sim_config.model_end < 0 or
                        source.zcoord + G.srcsteps[2] * config.sim_config.model_end > G.nz):
                        raise GeneralError('Source(s) will be stepped to a position outside the domain.')
                source.xcoord = source.xcoordorigin + G.model_num * G.srcsteps[0]
                source.ycoord = source.ycoordorigin + G.model_num * G.srcsteps[1]
                source.zcoord = source.zcoordorigin + G.model_num * G.srcsteps[2]
        if G.rxsteps[0] != 0 or G.rxsteps[1] != 0 or G.rxsteps[2] != 0:
            for receiver in G.rxs:
                if config.model_configs[G.model_num] == 0:
                    if (receiver.xcoord + G.rxsteps[0] * config.sim_config.model_end < 0 or
                        receiver.xcoord + G.rxsteps[0] * config.sim_config.model_end > G.nx or
                        receiver.ycoord + G.rxsteps[1] * config.sim_config.model_end < 0 or
                        receiver.ycoord + G.rxsteps[1] * config.sim_config.model_end > G.ny or
                        receiver.zcoord + G.rxsteps[2] * config.sim_config.model_end < 0 or
                        receiver.zcoord + G.rxsteps[2] * config.sim_config.model_end > G.nz):
                        raise GeneralError('Receiver(s) will be stepped to a position outside the domain.')
                receiver.xcoord = receiver.xcoordorigin + G.model_num * G.rxsteps[0]
                receiver.ycoord = receiver.ycoordorigin + G.model_num * G.rxsteps[1]
                receiver.zcoord = receiver.zcoordorigin + G.model_num * G.rxsteps[2]

        # Write files for any geometry views and geometry object outputs
        if not (G.geometryviews or G.geometryobjectswrite) and config.sim_config.args.geometry_only:
            log.warning(Fore.RED + f'\nNo geometry views or geometry objects found.' + Style.RESET_ALL)
        for i, geometryview in enumerate(G.geometryviews):
            log.info('')
            geometryview.set_filename()
            pbar = tqdm(total=geometryview.datawritesize, unit='byte', unit_scale=True,
                        desc=f'Writing geometry view file {i + 1}/{len(G.geometryviews)}, {geometryview.filename.name}',
                        ncols=get_terminal_width() - 1, file=sys.stdout,
                        disable=not config.sim_config.general['progressbars'])
            geometryview.write_vtk(G, pbar)
            pbar.close()
        for i, geometryobject in enumerate(G.geometryobjectswrite):
            log.info('')
            pbar = tqdm(total=geometryobject.datawritesize, unit='byte', unit_scale=True,
                        desc=f'Writing geometry object file {i + 1}/{len(G.geometryobjectswrite)}, {geometryobject.filename.name}',
                        ncols=get_terminal_width() - 1, file=sys.stdout,
                        disable=not config.sim_config.general['progressbars'])
            geometryobject.write_hdf5(G, pbar)
            pbar.close()

    def build_geometry(self):
        G = self.G

        log.info(config.model_configs[G.model_num].inputfilestr)

        scene = self.build_scene()

        # Combine available grids and check basic memory requirements
        grids = [G] + G.subgrids
        for grid in grids:
            config.model_configs[G.model_num].mem_use += grid.mem_est_basic()
        mem_check(config.model_configs[G.model_num].mem_use)

        # Set datatype for dispersive arrays if there are any dispersive materials.
        if config.model_configs[G.model_num].materials['maxpoles'] != 0:
            drudelorentz = any([m for m in G.materials if 'drude' in m.type or 'lorentz' in m.type])
            if drudelorentz:
                config.model_configs[G.model_num].materials['dispersivedtype'] = config.sim_config.dtypes['complex']
                config.model_configs[G.model_num].materials['dispersiveCdtype'] = config.sim_config.dtypes['C_complex']
            else:
                config.model_configs[G.model_num].materials['dispersivedtype'] = config.sim_config.dtypes['float_or_double']
                config.model_configs[G.model_num].materials['dispersiveCdtype'] = config.sim_config.dtypes['C_float_or_double']

            # Update estimated memory (RAM) usage
            config.model_configs[G.model_num].mem_use += G.mem_est_dispersive()
            mem_check(config.model_configs[G.model_num].mem_use)

        # Check there is sufficient memory to store any snapshots
        if G.snapshots:
            snaps_mem = 0
            for snap in G.snapshots:
                # 2 x required to account for electric and magnetic fields
                snaps_mem += int(2 * snap.datasizefield)
            config.model_configs[G.model_num].mem_use += snaps_mem
            # Check if there is sufficient memory on host
            mem_check(config.model_configs[G.model_num].mem_use)
            if config.sim_config.general['cuda']:
                mem_check_gpu_snaps(G.model_num, snaps_mem)

        log.info(f'\nMemory (RAM) required: ~{human_size(config.model_configs[G.model_num].mem_use)}')

        # Build grids
        gridbuilders = [GridBuilder(grid) for grid in grids]
        for gb in gridbuilders:
            pml_information(gb.grid)
            gb.build_pmls()
            gb.build_components()
            gb.tm_grid_update()
            gb.update_voltage_source_materials()
            gb.grid.initialise_std_update_coeff_arrays()
            if config.model_configs[G.model_num].materials['maxpoles'] != 0:
                gb.grid.initialise_dispersive_arrays()
            gb.build_materials()

        # Check to see if numerical dispersion might be a problem
        results = dispersion_analysis(G)
        if results['error']:
            log.warning(Fore.RED + f"\nNumerical dispersion analysis not carried out as {results['error']}" + Style.RESET_ALL)
        elif results['N'] < config.model_configs[G.model_num].numdispersion['mingridsampling']:
            raise GeneralError(f"Non-physical wave propagation: Material '{results['material'].ID}' has wavelength sampled by {results['N']} cells, less than required minimum for physical wave propagation. Maximum significant frequency estimated as {results['maxfreq']:g}Hz")
        elif (results['deltavp'] and np.abs(results['deltavp']) >
              config.model_configs[G.model_num].numdispersion['maxnumericaldisp']):
            log.warning(Fore.RED + f"\nPotentially significant numerical dispersion. Estimated largest physical phase-velocity error is {results['deltavp']:.2f}% in material '{results['material'].ID}' whose wavelength sampled by {results['N']} cells. Maximum significant frequency estimated as {results['maxfreq']:g}Hz" + Style.RESET_ALL)
        elif results['deltavp']:
            log.info(f"\nNumerical dispersion analysis: estimated largest physical phase-velocity error is {results['deltavp']:.2f}% in material '{results['material'].ID}' whose wavelength sampled by {results['N']} cells. Maximum significant frequency estimated as {results['maxfreq']:g}Hz")

    def reuse_geometry(self):
        # Reset iteration number
        self.G.iteration = 0
        config.model_configs[self.G.model_num].set_inputfilestr(f'\n--- Model {config.model_configs[self.G.model_num].appendmodelnumber}/{config.sim_config.model_end}, input file (not re-processed, i.e. geometry fixed): {config.sim_config.input_file_path}')
        log.info(config.model_configs[self.G.model_num].inputfilestr)
        for grid in [self.G] + self.G.subgrids:
            grid.reset_fields()

    def build_scene(self):
        # API for multiple scenes / model runs
        scene = config.model_configs[self.G.model_num].get_scene()

        # If there is no scene, process the hash commands
        if not scene:
            scene = Scene()
            # Parse the input file into user objects and add them to the scene
            scene = parse_hash_commands(scene, self.G)

        # Creates the internal simulation objects
        scene.create_internal_objects(self.G)

        return scene

    def write_output_data(self):
        """Write output data, i.e. field data for receivers and snapshots
            to file(s).
        """

        # Write an output file in HDF5 format
        write_hdf5_outputfile(config.model_configs[self.G.model_num].output_file_path_ext, self.G)

        # Write any snapshots to file
        if self.G.snapshots:
            # Create directory for snapshots
            config.model_configs[self.G.model_num].set_snapshots_file_path()
            snapshotdir = config.model_configs[self.G.model_num].snapshot_file_path
            snapshotdir.mkdir(exist_ok=True)

            log.info('')
            for i, snap in enumerate(self.G.snapshots):
                fn = snapshotdir / Path(snap.filename)
                snap.filename = fn.with_suffix('.vti')
                pbar = tqdm(total=snap.vtkdatawritesize, leave=True, unit='byte',
                            unit_scale=True, desc=f'Writing snapshot file {i + 1} of {len(self.G.snapshots)}, {snap.filename.name}', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.sim_config.general['progressbars'])
                snap.write_vtk_imagedata(pbar, self.G)
                pbar.close()
            log.info('')

    def print_resource_info(self, tsolve, memsolve):
        """Print resource information on runtime and memory usage.

        Args:
            tsolve (float): Time taken to execute solving (seconds).
            memsolve (float): Memory (RAM) used on GPU.
        """

        memGPU_str = ''
        if config.sim_config.general['cuda']:
            memGPU_str = f' host + ~{human_size(memsolve)} GPU'

        log.info(f'\nMemory (RAM) used: ~{human_size(self.p.memory_full_info().uss)}{memGPU_str}')
        log.info(f'Solving time [HH:MM:SS]: {datetime.timedelta(seconds=tsolve)}')

    def solve(self, solver):
        """Solve using FDTD method.

        Args:
            solver (Solver): solver object.

        Returns:
            tsolve (float): time taken to execute solving (seconds).
        """

        log.info(f'\nOutput file: {config.model_configs[self.G.model_num].output_file_path_ext.name}')

        # Check number of OpenMP threads
        if config.sim_config.general['cpu']:
            log.info(f'CPU (OpenMP) threads for solving: {config.model_configs[self.G.model_num].ompthreads}\n')
            if config.model_configs[self.G.model_num].ompthreads > config.sim_config.hostinfo['physicalcores']:
                log.warning(Fore.RED + f"You have specified more threads ({config.model_configs[self.G.model_num].ompthreads}) than available physical CPU cores ({config.sim_config.hostinfo['physicalcores']}). This may lead to degraded performance." + Style.RESET_ALL)
        # Print information about any GPU in use
        elif config.sim_config.general['cuda']:
            log.info(f"GPU for solving: {config.model_configs[self.G.model_num].cuda['gpu'].deviceID} - {config.model_configs[self.G.model_num].cuda['gpu'].name}\n")

        # Prepare iterator
        if config.sim_config.is_messages():
            iterator = tqdm(range(self.G.iterations), desc='Running simulation, model ' + str(self.G.model_num + 1) + '/' + str(config.sim_config.model_end), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.sim_config.general['progressbars'])
        else:
            iterator = range(self.G.iterations)

        # Run solver
        tsolve, memsolve = solver.solve(iterator)

        # Write output data, i.e. field data for receivers and snapshots to file(s)
        self.write_output_data()

        # Print resource information on runtime and memory usage
        self.print_resource_info(tsolve, memsolve)

        return tsolve


class GridBuilder:
    def __init__(self, grid):
        self.grid = grid

    def build_pmls(self):
        log.info('')
        pbar = tqdm(total=sum(1 for value in self.grid.pmlthickness.values() if value > 0),
                    desc=f'Building {self.grid.name} Grid PML boundaries',
                    ncols=get_terminal_width() - 1, file=sys.stdout,
                    disable=not config.sim_config.general['progressbars'])
        for pml_id, thickness in self.grid.pmlthickness.items():
            if thickness > 0:
                build_pml(self.grid, pml_id, thickness)
                pbar.update()
        pbar.close()

    def build_components(self):
        # Build the model, i.e. set the material properties (ID) for every edge
        # of every Yee cell
        log.info('')
        pbar = tqdm(total=2, desc=f'Building {self.grid.name} Grid',
                    ncols=get_terminal_width() - 1, file=sys.stdout,
                    disable=not config.sim_config.general['progressbars'])
        build_electric_components(self.grid.solid, self.grid.rigidE, self.grid.ID, self.grid)
        pbar.update()
        build_magnetic_components(self.grid.solid, self.grid.rigidH, self.grid.ID, self.grid)
        pbar.update()
        pbar.close()

    def tm_grid_update(self):
        if '2D TMx' == config.model_configs[self.grid.model_num].mode:
            self.grid.tmx()
        elif '2D TMy' == config.model_configs[self.grid.model_num].mode:
            self.grid.tmy()
        elif '2D TMz' == config.model_configs[self.grid.model_num].mode:
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
        materialstable.justify_columns[0] = 'right'

        log.info(f'\n{self.grid.name} Grid Materials:')
        log.info(materialstable.table)
