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

from pathlib import Path

from colorama import init
from colorama import Fore
from colorama import Style
init()
import cython
import numpy as np
from scipy.constants import c
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as m0

from .utilities import get_host_info
from .utilities import get_terminal_width


# General settings for the simulation
#   inputfilepath: path to inputfile location
#   outputfilepath: path to outputfile location
#   messages: whether to print all messages as output to stdout or not
#   progressbars: whether to show progress bars on stdoout or not
#   mode: 2D TMx, 2D TMy, 2D TMz, or 3D
#   cpu, cuda, opencl: solver type
#   precision: data type for electromagnetic field output (single/double)
#   autotranslate: auto translate objects with main grid coordinates
#   to their equivalent local grid coordinate within the subgrid. If this option is off
#   users must specify sub-grid object point within the global subgrid space.
#   z0: impedance of free space (Ohms)
general = {'messages': True,
           'progressbars': True,
           'mode': '3D',
           'cpu': True,
           'cuda': False,
           'opencl': False,
           'precision': 'single',
           'autotranslate': False,
           'z0': np.sqrt(m0 / e0)}

def is_messages():
    """Return messages."""
    return general['messages']

# Store information about host machine
hostinfo = get_host_info()

# Store information for CUDA solver type
#   gpus: information about any GPUs as a list of GPU objects
#   snapsgpu2cpu: copy snapshot data from GPU to CPU during simulation
#     N.B. This will happen if the requested snapshots are too large to fit
#     on the memory of the GPU. If True this will slow performance significantly
cuda = {'gpus': None, 'snapsgpu2cpu': False}

# Data type (precision) for electromagnetic field output
#   Solid and ID arrays use 32-bit integers (0 to 4294967295)
#   Rigid arrays use 8-bit integers (the smallest available type to store true/false)
#   Fractal arrays use complex numbers
#   Dispersive coefficient arrays use either float or complex numbers
#   Main field arrays use floats

if general['precision'] == 'single':
    dtypes = {'float_or_double': np.float32,
              'complex': np.complex64,
              'cython_float_or_double': cython.float,
              'cython_complex': cython.floatcomplex,
              'C_float_or_double': 'float',
              'C_complex': 'pycuda::complex<float>'}
elif general['precision'] == 'double':
    dtypes = {'float_or_double': np.float64,
              'complex': np.complex128,
              'cython_float_or_double': cython.double,
              'cython_complex': cython.doublecomplex,
              'C_float_or_double': 'double',
              'C_complex': 'pycuda::complex<double>'}


class ModelConfig:
    """Configuration parameters for a model.
        N.B. Multiple models can exist within a simulation
    """

    def __init__(self, sim_config, i):
        """
        Args:
            sim_config (SimConfig): Simulation level configuration object.
            i (int): Current model number.
        """

        self.sim_config = sim_config
        self.i = i # Indexed from 0
        self.reuse_geometry = False

        if not sim_config.single_model:
            self.appendmodelnumber = str(self.i + 1) # Indexed from 1
        else:
            self.appendmodelnumber = ''

        # Output file path for specific model
        parts = self.sim_config.output_file_path.with_suffix('').parts
        self.output_file_path = Path(*parts[:-1], parts[-1] + self.appendmodelnumber)
        self.output_file_path_ext = self.output_file_path.with_suffix('.out')

        # Make a snapshot directory
        self.snapshot_dir = '_snaps'

        # String to print at start of each model run
        inputfilestr_f = '\n--- Model {}/{}, input file: {}'
        self.inputfilestr = inputfilestr_f.format(self.i + 1, self.sim_config.model_end, self.sim_config.input_file_path)
        self.next_model = Fore.GREEN + '{} {}\n'.format(self.inputfilestr, '-' * (get_terminal_width() - 1 - len(self.inputfilestr))) + Style.RESET_ALL

        # Numerical dispersion analysis parameters
        #   highestfreqthres: threshold (dB) down from maximum power (0dB) of main frequency used
        #     to calculate highest frequency for numerical dispersion analysis
        #   maxnumericaldisp: maximum allowable percentage physical phase-velocity phase error
        #   mingridsampling: minimum grid sampling of smallest wavelength for physical wave propagation
        self.numdispersion = {'highestfreqthres': 40,
                              'maxnumericaldisp': 2,
                              'mingridsampling': 3}

        # General information to configure materials
        #   maxpoles: Maximum number of dispersive material poles in a model
        #   dispersivedtype: Data type for dispersive materials
        #   dispersiveCdtype: Data type for dispersive materials in Cython
        self.materials = {'maxpoles': 0,
                          'dispersivedtype': None,
                          'dispersiveCdtype': None}

    def get_scene(self):
        if self.sim_config.scenes:
            return self.sim_config.scenes[self.i]
        else: return None

    def get_usernamespace(self):
        return {'c': c, # Speed of light in free space (m/s)
                'e0': e0, # Permittivity of free space (F/m)
                'm0': m0, # Permeability of free space (H/m)
                'z0': general['z0'], # Impedance of free space (Ohms)
                'number_model_runs': self.sim_config.model_end + 1,
                'current_model_run': self.i + 1,
                'inputfile': self.sim_config.input_file_path.resolve()}


class SimulationConfig:
    """Configuration parameters for a standard simulation.
        N.B. A simulation can consist of multiple models.
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Arguments from either API or CLI.
        """

        self.args = args
        self.general = general
        self.hostinfo = hostinfo
        self.cuda = cuda
        self.dtypes = dtypes

        # Subgrid parameter may not exist if user enters via CLI
        try:
            self.subgrid = args.subgrid
        except AttributeError:
            self.subgrid = False

        # Scenes parameter may not exist if user enters via CLI
        try:
            self.scenes = args.scenes
        except AttributeError:
            self.scenes = []

        # Set more complex parameters
        self.set_input_file_path()
        self.set_output_file_path()
        self.set_model_start_end()
        self.set_single_model()

    def set_single_model(self):
        if self.model_start == 0 and self.model_end == 1:
            self.single_model = True
        else:
            self.single_model = False

    def set_model_start_end(self):
        """Set range for number of models to run (internally 0 index)."""
        if self.args.task:
            # Job array feeds args.n number of single tasks
            modelstart = self.args.task - 1
            modelend = self.args.task
        elif self.args.restart:
            modelstart = self.args.restart - 1
            modelend = modelstart + self.args.n - 1
        else:
            modelstart = 0
            modelend = modelstart + self.args.n

        self.model_start = modelstart
        self.model_end = modelend

    def set_input_file_path(self):
        """If the API is in use an id for the simulation must be provided."""
        if self.args.inputfile is None:
            self.input_file_path = Path(self.args.outputfile)
        else:
            self.input_file_path = Path(self.args.inputfile)

    def set_output_file_path(self):
        """Output file path can be provided by the user. If they havent provided one
            use the inputfile file path instead."""
        try:
            self.output_file_path = Path(self.args.outputfile)
        except AttributeError:
            self.output_file_path = Path(self.input_file_path)


class SimulationConfigMPI(SimulationConfig):
    """Configuration parameters for a MPI simulation.
        N.B. A simulation can consist of multiple models.
    """

    def __init__(self, args):
        super().__init__(args)

    def set_model_start_end(self):
        # Set range for number of models to run
        self.model_start = self.args.restart if self.args.restart else 1
        self.model_end = self.modelstart + self.args.n


def create_simulation_config(args):
    """Create simulation level configuration object to hold simulation
        level parameters.

    Args:
        args (Namespace): Arguments from either API or CLI.

    Returns:
        sc (SimulationConfig): Simulation configuration object.
    """

    if not args.mpi and not args.mpi_no_spawn:
        sc = SimulationConfig(args)
    elif args.mpi:
        sc = SimulationConfigMPI(args)

    return sc


def create_model_config(sim_config, i):
    """Create model level configuration object to hold model level
        parameters.

    Args:
        sim_config (SimConfig): Simulation level configuration object.
        i (int): Current model number.
    """

    mc = ModelConfig(sim_config, i)

    return mc
