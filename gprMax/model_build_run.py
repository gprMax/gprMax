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

import gprMax.config as config
from .cuda.fields_updates import kernel_template_fields
from .cuda.snapshots import kernel_template_store_snapshot
from .cuda.source_updates import kernel_template_sources
from .cython.yee_cell_build import build_electric_components
from .cython.yee_cell_build import build_magnetic_components
from .exceptions import GeneralError
from .fields_outputs import store_outputs
from .fields_outputs import kernel_template_store_outputs
from .fields_outputs import write_hdf5_outputfile
from .grid import FDTDGrid
from .grid import dispersion_analysis
from .input_cmds_geometry import process_geometrycmds
from .input_cmds_file import process_python_include_code
from .input_cmds_file import write_processed_file
from .input_cmds_file import check_cmd_names
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

class Printer():

    def __init__(self, sim_config):
        self.printing = sim_config.general['messages']

    def print(self, str):
        if self.printing:
            print(str)

class ModelBuildRun:

    def __init__(self, solver, sim_config, model_config):
        self.solver = solver
        self.sim_config = sim_config
        self.model_config = model_config
        self.G = solver.get_G()

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
        p = psutil.Process()

        # Normal model reading/building process; bypassed if geometry information to be reused
        if not self.sim_config.geometry_fixed:
            self.build_geometry()
        else:
            self.reuse_geometry()

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
        if not (G.geometryviews or G.geometryobjectswrite) and self.sim_config.geometry_only and config.general['messages']:
            print(Fore.RED + '\nWARNING: No geometry views or geometry objects to output found.' + Style.RESET_ALL)
        if config.general['messages']: print()
        for i, geometryview in enumerate(G.geometryviews):
            geometryview.set_filename(appendmodelnumber)
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

        printer = Printer(sim_config)
        printer.print(model_config.next_model)

        scene = self.build_scene()

        # print PML information
        printer.print(pml_information(G))

        self.build_pmls()
        self.build_components()

        # update grid for tm modes
        self.tm_grid_update()

        self.update_voltage_source_materials()

        # Initialise arrays of update coefficients to pass to update functions
        G.initialise_std_update_coeff_arrays()

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

            G.initialise_dispersive_arrays(config.materials['dispersivedtype'])

        # Check there is sufficient memory to store any snapshots
        if G.snapshots:
            snapsmemsize = 0
            for snap in G.snapshots:
                # 2 x required to account for electric and magnetic fields
                snapsmemsize += (2 * snap.datasizefield)
            G.memoryusage += int(snapsmemsize)
            G.memory_check(snapsmemsize=int(snapsmemsize))

            printer.print('\nMemory (RAM) required - updated (snapshots): ~{}\n'.format(human_size(G.memoryusage)))

        # Process complete list of materials - calculate update coefficients,
        # store in arrays, and build text list of materials/properties
        materialsdata = process_materials(G)
        materialstable = SingleTable(materialsdata)
        materialstable.outer_border = False
        materialstable.justify_columns[0] = 'right'

        printer.print('\nMaterials:')
        printer.print(materialstable.table)

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

        # set the dispersive update functions based on the model configuration
        props = self.solver.updates.adapt_dispersive_config(config)
        self.solver.updates.set_dispersive_updates(props)

    def reuse_geometry(self):
        printer = Printer(model_config)
        inputfilestr = '\n--- Model {}/{}, input file (not re-processed, i.e. geometry fixed): {}'.format(currentmodelrun, modelend, model_config.input_file_path)
        printer.print(Fore.GREEN + '{} {}\n'.format(model_config.inputfilestr, '-' * (get_terminal_width() - 1 - len(model_config.inputfilestr))) + Style.RESET_ALL)
        self.G.reset_fields()

    def tm_grid_update(self):
        if '2D TMx' == config.general['mode']:
            self.G.tmx()
        elif '2D TMy' == config.general['mode']:
            self.G.tmy()
        elif '2D TMz' == config.general['mode']:
            self.G.tmz()

    def build_scene(self):
        # api for multiple scenes / model runs
        try:
            scene = self.model_config.get_scene()
        # process using hashcommands
        except AttributeError:
            scene = Scene()
            # parse the input file into user objects and add them to the scene
            scene = parse_hash_commands(args, usernamespace, appendmodelnumber, self.G, scene)

        # Creates the internal simulation objects.
        scene.create_internal_objects(self.G)
        return scene

    def build_pmls(self):
        # build the PMLS
        pbar = tqdm(total=sum(1 for value in self.G.pmlthickness.values() if value > 0), desc='Building PML boundaries', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])

        for pml_id, thickness in self.G.pmlthickness.items():
            build_pml(self.G, pml_id, thickness)
            pbar.update()
        pbar.close()

    def build_components(self):
        # Build the model, i.e. set the material properties (ID) for every edge
        # of every Yee cell
        G = self.G
        printer = Printer(self.sim_config)
        printer.print('')
        pbar = tqdm(total=2, desc='Building main grid', ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
        build_electric_components(G.solid, G.rigidE, G.ID, G)
        pbar.update()
        build_magnetic_components(G.solid, G.rigidH, G.ID, G)
        pbar.update()
        pbar.close()

    def update_voltage_source_materials(self):
        # Process any voltage sources (that have resistance) to create a new
        # material at the source location
        for voltagesource in self.G.voltagesources:
            voltagesource.create_material(self.G)


    def run_model(self):

        # Check and set output directory and filename
        if not os.path.isdir(config.outputfilepath):
            os.mkdir(config.outputfilepath)
            printer.print('\nCreated output directory: {}'.format(config.outputfilepath))

        outputfile = os.path.join(config.outputfilepath, os.path.splitext(os.path.split(config.inputfilepath)[1])[0] + appendmodelnumber + '.out')
        printer.print('\nOutput file: {}\n'.format(outputfile))

        tsolve, memsolve = solve(currentmodelrun, modelend, G)

        # Write an output file in HDF5 format
        write_hdf5_outputfile(outputfile, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz, G)

        # Write any snapshots to file
        if G.snapshots:
            # Create directory and construct filename from user-supplied name and model run number
            snapshotdir = os.path.splitext(config.inputfilepath)[0] + '_snaps' + appendmodelnumber
            if not os.path.exists(snapshotdir):
                os.mkdir(snapshotdir)

            if config.general['messages']: print()
            for i, snap in enumerate(G.snapshots):
                snap.filename = os.path.abspath(os.path.join(snapshotdir, snap.basefilename + '.vti'))
                pbar = tqdm(total=snap.vtkdatawritesize, leave=True, unit='byte', unit_scale=True, desc='Writing snapshot file {} of {}, {}'.format(i + 1, len(G.snapshots), os.path.split(snap.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=not config.general['progressbars'])
                snap.write_vtk_imagedata(pbar, G)
                pbar.close()
            if config.general['messages']: print()

        memGPU = ''
        if config.cuda['gpus']:
            memGPU = ' host + ~{} GPU'.format(human_size(memsolve))

        printer.print('\nMemory (RAM) used: ~{}{}'.format(human_size(p.memory_full_info().uss), memGPU))
        printer.print('Solving time [HH:MM:SS]: {}'.format(datetime.timedelta(seconds=tsolve)))


        return tsolve

def solve(solver, sim_config, model_config):
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
    tsolvestart = timer()

    solver.solve()

    return tsolve
