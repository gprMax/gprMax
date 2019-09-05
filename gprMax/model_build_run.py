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
from .input_cmds_geometry import process_geometrycmds
from .input_cmds_file import process_python_include_code
from .input_cmds_file import write_processed_file
from .input_cmds_file import check_cmd_names
from .input_cmds_file import parse_hash_commands
from .input_cmds_singleuse import process_singlecmds
from .input_cmds_multiuse import process_multicmds
from .materials import Material
from .materials import process_materials
from .pml import CFS
from .pml import PML
from .pml import build_pml
from .pml import pml_information
from .receivers import gpu_initialise_rx_arrays
from .receivers import gpu_get_rx_array
from .receivers import Rx
from .snapshots import Snapshot
from .snapshots import gpu_initialise_snapshot_array
from .snapshots import gpu_get_snapshot_array
from .sources import gpu_initialise_src_arrays
from .utilities import get_terminal_width
from .utilities import human_size
from .utilities import open_path_file
from .utilities import round32
from .utilities import timer
from .utilities import Printer
from .scene import Scene
from .solvers import create_solver



class ModelBuildRun:

    def __init__(self, G, sim_config, model_config):
        self.G = G
        self.sim_config = sim_config
        self.model_config = model_config
        self.printer = Printer(config)
        # Monitor memory usage
        self.p = None

    def build(self):
        """Runs a model - processes the input file; builds the Yee cells; calculates update coefficients; runs main FDTD loop.

        Args:
            args (dict): Namespace with command line arguments
            currentmodelrun (int): Current model run number.
            modelend (int): Number of last model to run.
            numbermodelruns (int): Total number of model runs.
            inputfile (object): File object for the input file.
            usernamespace (dict): Namespace that can be accessed by user
                    in any Python code blocks in input file.

        Returns:
            tsolve (int): Length of time (seconds) of main FDTD calculations
        """
        # Monitor memory usage
        self.p = psutil.Process()

        # Normal model reading/building process; bypassed if geometry information to be reused
        if self.model_config.reuse_geometry:
            self.reuse_geometry()
        else:
            self.build_geometry()

        G = self.G

        # Adjust position of simple sources and receivers if required
        if G.srcsteps[0] != 0 or G.srcsteps[1] != 0 or G.srcsteps[2] != 0:
            for source in itertools.chain(G.hertziandipoles, G.magneticdipoles):
                if currentmodelrun == 1:
                    if source.xcoord + G.srcsteps[0] * modelend < 0 or source.xcoord + G.srcsteps[0] * modelend > G.nx or source.ycoord + G.srcsteps[1] * modelend < 0 or source.ycoord + G.srcsteps[1] * modelend > G.ny or source.zcoord + G.srcsteps[2] * modelend < 0 or source.zcoord + G.srcsteps[2] * modelend > G.nz:
                        raise GeneralError('Source(s) will be stepped to a position outside the domain.')
                source.xcoord = source.xcoordorigin + (currentmodelrun - 1) * G.srcsteps[0]
                source.ycoord = source.ycoordorigin + (currentmodelrun - 1) * G.srcsteps[1]
                source.zcoord = source.zcoordorigin + (currentmodelrun - 1) * G.srcsteps[2]
        if G.rxsteps[0] != 0 or G.rxsteps[1] != 0 or G.rxsteps[2] != 0:
            for receiver in G.rxs:
                if currentmodelrun == 1:
                    if receiver.xcoord + G.rxsteps[0] * modelend < 0 or receiver.xcoord + G.rxsteps[0] * modelend > G.nx or receiver.ycoord + G.rxsteps[1] * modelend < 0 or receiver.ycoord + G.rxsteps[1] * modelend > G.ny or receiver.zcoord + G.rxsteps[2] * modelend < 0 or receiver.zcoord + G.rxsteps[2] * modelend > G.nz:
                        raise GeneralError('Receiver(s) will be stepped to a position outside the domain.')
                receiver.xcoord = receiver.xcoordorigin + (currentmodelrun - 1) * G.rxsteps[0]
                receiver.ycoord = receiver.ycoordorigin + (currentmodelrun - 1) * G.rxsteps[1]
                receiver.zcoord = receiver.zcoordorigin + (currentmodelrun - 1) * G.rxsteps[2]

        # Write files for any geometry views and geometry object outputs
        if not (G.geometryviews or G.geometryobjectswrite) and self.sim_config.geometry_only and config.is_messages():
            print(Fore.RED + '\nWARNING: No geometry views or geometry objects to output found.' + Style.RESET_ALL)
        if config.is_messages(): print()
        for i, geometryview in enumerate(G.geometryviews):
            geometryview.set_filename(self.model_config.appendmodelnumber)
            pbar = tqdm(total=geometryview.datawritesize, unit='byte', unit_scale=True, desc='Writing geometry view file {}/{}, {}'.format(i + 1, len(G.geometryviews), os.path.split(geometryview.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
            geometryview.write_vtk(G, pbar)
            pbar.close()
        for i, geometryobject in enumerate(G.geometryobjectswrite):
            pbar = tqdm(total=geometryobject.datawritesize, unit='byte', unit_scale=True, desc='Writing geometry object file {}/{}, {}'.format(i + 1, len(G.geometryobjectswrite), os.path.split(geometryobject.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
            geometryobject.write_hdf5(G, pbar)
            pbar.close()

        # If only writing geometry information
        if self.sim_config.geometry_only:
            tsolve = 0

    def build_geometry(self):
        model_config = self.model_config
        sim_config = self.sim_config

        G = self.G

        printer = Printer(config)
        printer.print(model_config.next_model)

        scene = self.build_scene()

        # combine available grids
        grids = [G] + G.subgrids
        gridbuilders = [GridBuilder(grid, self.printer) for grid in grids]

        for gb in gridbuilders:
            gb.printer.print(pml_information(gb.grid))
            gb.build_pmls()
            gb.build_components()
            gb.tm_grid_update()
            gb.update_voltage_source_materials()
            gb.grid.initialise_std_update_coeff_arrays()

        # Set datatype for dispersive arrays if there are any dispersive materials.
        if config.materials['maxpoles'] != 0:
            drudelorentz = any([m for m in G.materials if 'drude' in m.type or 'lorentz' in m.type])
            if drudelorentz:
                config.materials['dispersivedtype'] = config.dtypes['complex']
                config.materials['dispersiveCdtype'] = config.dtypes['C_complex']
            else:
                config.materials['dispersivedtype'] = config.dtypes['float_or_double']
                config.materials['dispersiveCdtype'] = config.dtypes['C_float_or_double']

            # Update estimated memory (RAM) usage
            G.memoryusage += int(3 * config.materials['maxpoles'] * (G.nx + 1) * (G.ny + 1) * (G.nz + 1) * np.dtype(config.materials['dispersivedtype']).itemsize)
            G.memory_check()
            printer.print('\nMemory (RAM) required - updated (dispersive): ~{}\n'.format(human_size(G.memoryusage)))

            for gb in gridbuilders:
                gb.grid.initialise_dispersive_arrays(config.materials['dispersivedtype'])

        # Check there is sufficient memory to store any snapshots
        if G.snapshots:
            snapsmemsize = 0
            for snap in G.snapshots:
                # 2 x required to account for electric and magnetic fields
                snapsmemsize += (2 * snap.datasizefield)
            G.memoryusage += int(snapsmemsize)
            G.memory_check(snapsmemsize=int(snapsmemsize))

            printer.print('\nMemory (RAM) required - updated (snapshots): ~{}\n'.format(human_size(G.memoryusage)))

        for gb in gridbuilders:
            gb.build_materials()

        # Check to see if numerical dispersion might be a problem
        results = dispersion_analysis(G)
        if results['error']:
            printer.print(Fore.RED + "\nWARNING: Numerical dispersion analysis not carried out as {}".format(results['error']) + Style.RESET_ALL)
        elif results['N'] < config.numdispersion['mingridsampling']:
            raise GeneralError("Non-physical wave propagation: Material '{}' has wavelength sampled by {} cells, less than required minimum for physical wave propagation. Maximum significant frequency estimated as {:g}Hz".format(results['material'].ID, results['N'], results['maxfreq']))
        elif results['deltavp'] and np.abs(results['deltavp']) > config.numdispersion['maxnumericaldisp']:
            printer.print(Fore.RED + "\nWARNING: Potentially significant numerical dispersion. Estimated largest physical phase-velocity error is {:.2f}% in material '{}' whose wavelength sampled by {} cells. Maximum significant frequency estimated as {:g}Hz".format(results['deltavp'], results['material'].ID, results['N'], results['maxfreq']) + Style.RESET_ALL)
        elif results['deltavp']:
            printer.print("\nNumerical dispersion analysis: estimated largest physical phase-velocity error is {:.2f}% in material '{}' whose wavelength sampled by {} cells. Maximum significant frequency estimated as {:g}Hz".format(results['deltavp'], results['material'].ID, results['N'], results['maxfreq']))


    def reuse_geometry(self):
        G = self.G
        inputfilestr = '\n--- Model {}/{}, input file (not re-processed, i.e. geometry fixed): {}'.format(self.model_config.appendmodelnumber, self.sim_config.model_end, self.sim_config.input_file_path)
        self.printer.print(Fore.GREEN + '{} {}\n'.format(self.model_config.inputfilestr, '-' * (get_terminal_width() - 1 - len(inputfilestr))) + Style.RESET_ALL)
        for grid in [G] + G.subgrids:
            grid.reset_fields()

    def build_scene(self):
        G = self.G
        # api for multiple scenes / model runs
        scene = self.model_config.get_scene()

        # if there is no scene - process the hash commands instead
        if not scene:
            scene = Scene()
            # parse the input file into user objects and add them to the scene
            scene = parse_hash_commands(self.model_config, G, scene)

        # Creates the internal simulation objects.
        scene.create_internal_objects(G)
        return scene

    def create_output_directory(self):

        if self.G.outputdirectory:
            # Check and set output directory and filename
            try:
                os.mkdir(self.G.outputdirectory)
                self.printer.print('\nCreated output directory: {}'.format(self.G.outputdirectory))
            except FileExistsError:
                pass
            # modify the output path (hack)
            self.model_config.output_file_path_ext = Path(self.G.outputdirectory, self.model_config.output_file_path_ext)


    def run_model(self, solver):

        G = self.G

        self.create_output_directory()
        self.printer.print('\nOutput file: {}\n'.format(self.model_config.output_file_path_ext))

        tsolve = self.solve(solver)

        # Write an output file in HDF5 format
        write_hdf5_outputfile(self.model_config.output_file_path_ext, G)

        # Write any snapshots to file
        if G.snapshots:
            # Create directory and construct filename from user-supplied name and model run number
            snapshotdir = self.model_config.snapshot_dir
            if not os.path.exists(snapshotdir):
                os.mkdir(snapshotdir)

            self.printer.print('')
            for i, snap in enumerate(G.snapshots):
                fn = snapshotdir / Path(self.model_config.output_file_path.stem + '_' + snap.basefilename)
                snap.filename = fn.with_suffix('.vti')
                pbar = tqdm(total=snap.vtkdatawritesize, leave=True, unit='byte', unit_scale=True, desc='Writing snapshot file {} of {}, {}'.format(i + 1, len(G.snapshots), os.path.split(snap.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
                snap.write_vtk_imagedata(pbar, G)
                pbar.close()
            self.printer.print('')

        memGPU = ''
        if config.cuda['gpus']:
            memGPU = ' host + ~{} GPU'.format(human_size(self.solver.get_memsolve()))

        self.printer.print('\nMemory (RAM) used: ~{}{}'.format(human_size(self.p.memory_full_info().uss), memGPU))
        self.printer.print('Solving time [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=tsolve)))

        return tsolve

    def solve(self, solver):
        """
        Solving using FDTD method on CPU. Parallelised using Cython (OpenMP) for
        electric and magnetic field updates, and PML updates.

        Args:
            currentmodelrun (int): Current model run number.
            modelend (int): Number of last model to run.
            G (class): Grid class instance - holds essential parameters describing the model.

        Returns:
            tsolve (float): Time taken to execute solving (seconds)
        """
        G = self.G

        if config.is_messages():
            iterator = tqdm(range(G.iterations), desc='Running simulation, model ' + str(self.model_config
                            .i + 1) + '/' + str(self.sim_config.model_end), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
        else:
            iterator = range(0, G.iterations)

        tsolve = solver.solve(iterator)

        return tsolve


class GridBuilder:
    def __init__(self, grid, printer):
        self.grid = grid
        self.printer = printer

    def build_pmls(self):

        grid = self.grid
        # build the PMLS
        pbar = tqdm(total=sum(1 for value in grid.pmlthickness.values() if value > 0), desc='Building {} Grid PML boundaries'.format(self.grid.name), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])

        for pml_id, thickness in grid.pmlthickness.items():
            build_pml(grid, pml_id, thickness)
            pbar.update()
        pbar.close()

    def build_components(self):
        # Build the model, i.e. set the material properties (ID) for every edge
        # of every Yee cell
        self.printer.print('')
        pbar = tqdm(total=2, desc='Building {} grid'.format(self.grid.name), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
        build_electric_components(self.grid.solid, self.grid.rigidE, self.grid.ID, self.grid)
        pbar.update()
        build_magnetic_components(self.grid.solid, self.grid.rigidH, self.grid.ID, self.grid)
        pbar.update()
        pbar.close()

    def tm_grid_update(self):
        if '2D TMx' == config.general['mode']:
            self.grid.tmx()
        elif '2D TMy' == config.general['mode']:
            self.grid.tmy()
        elif '2D TMz' == config.general['mode']:
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

        self.printer.print('\n{} Grid Materials:'.format(self.grid.name))
        self.printer.print(materialstable.table)
