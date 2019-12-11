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

import logging
from pathlib import Path
import sys

from colorama import init
from colorama import Fore
from colorama import Style
init()
import cython
import numpy as np
from scipy.constants import c
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as m0

from .exceptions import GeneralError
from .utilities import detect_check_gpus
from .utilities import get_host_info
from .utilities import get_terminal_width

log = logging.getLogger(__name__)


# Single instance of SimConfig to hold simulation configuration parameters.
sim_config = None

# Instances of ModelConfig that hold model configuration parameters.
model_configs = []

# Each model in a simulation is given a unique number when the instance of
# ModelConfig is created
model_num = 0

def get_model_config():
    """Return ModelConfig instace for specific model."""
    return model_configs[model_num]


class ModelConfig:
    """Configuration parameters for a model.
        N.B. Multiple models can exist within a simulation
    """

    def __init__(self):

        self.mode = '3D'
        self.grids = []
        self.ompthreads = None # Number of OpenMP threads

        # Store information for CUDA solver
        #   gpu: GPU object
        #   snapsgpu2cpu: copy snapshot data from GPU to CPU during simulation
        #     N.B. This will happen if the requested snapshots are too large to fit
        #     on the memory of the GPU. If True this will slow performance significantly
        if sim_config.general['cuda']:
            gpu = sim_config.set_model_gpu()
            self.cuda = {'gpu': gpu,
                         'snapsgpu2cpu': False}

        # Total memory usage for all grids in the model. Starts with 50MB overhead.
        self.mem_overhead = 50e6
        self.mem_use = self.mem_overhead

        self.reuse_geometry = False

        # String to print at start of each model run
        s = f'\n--- Model {model_num + 1}/{sim_config.model_end}, input file: {sim_config.input_file_path}'
        self.inputfilestr = Fore.GREEN + f"{s} {'-' * (get_terminal_width() - 1 - len(s))}\n" + Style.RESET_ALL

        # Output file path and name for specific model
        self.appendmodelnumber = '' if sim_config.single_model else str(model_num + 1) # Indexed from 1
        self.set_output_file_path()

        # Specify a snapshot directory
        self.set_snapshots_dir()

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
        #   drudelorentz: True/False model contains Drude or Lorentz materials
        #   cudarealfunc: String to substitute into CUDA kernels for fields
        #                   dependent on dispersive material type
        self.materials = {'maxpoles': 0,
                          'dispersivedtype': None,
                          'dispersiveCdtype': None,
                          'drudelorentz': None,
                          'cudarealfunc': ''}

    def get_scene(self):
        if sim_config.scenes:
            return sim_config.scenes[model_num]
        else: return None

    def get_usernamespace(self):
        return {'c': c, # Speed of light in free space (m/s)
                'e0': e0, # Permittivity of free space (F/m)
                'm0': m0, # Permeability of free space (H/m)
                'z0': np.sqrt(m0 / e0), # Impedance of free space (Ohms)
                'number_model_runs': sim_config.model_end,
                'current_model_run': model_num + 1,
                'inputfile': sim_config.input_file_path.resolve()}

    def set_dispersive_material_types(self):
        """Set data type for disperive materials. Complex if Drude or Lorentz
            materials are present. Real if Debye materials.
        """
        if self.materials['drudelorentz']:
            self.materials['cudarealfunc'] = '.real()'
            self.materials['dispersivedtype'] = sim_config.dtypes['complex']
            self.materials['dispersiveCdtype'] = sim_config.dtypes['C_complex']
        else:
            self.materials['dispersivedtype'] = sim_config.dtypes['float_or_double']
            self.materials['dispersiveCdtype'] = sim_config.dtypes['C_float_or_double']

    def set_output_file_path(self, outputdir=None):
        """Output file path can be provided by the user via the API or an input file
            command. If they haven't provided one use the input file path instead.

        Args:
            outputdir (str): Output file directory given from input file command.
        """

        if not outputdir:
            try:
                self.output_file_path = Path(self.args.outputfile)
            except AttributeError:
                self.output_file_path = sim_config.input_file_path.with_suffix('')
        else:
            try:
                Path(outputdir).mkdir(exist_ok=True)
                self.output_file_path = Path(outputdir, sim_config.input_file_path.stem)
            except AttributeError:
                self.output_file_path = sim_config.input_file_path.with_suffix('')

        parts = self.output_file_path.parts
        self.output_file_path = Path(*parts[:-1], parts[-1] + self.appendmodelnumber)
        self.output_file_path_ext = self.output_file_path.with_suffix('.h5')

    def set_snapshots_dir(self):
        """Set directory to store any snapshots."""
        parts = self.output_file_path.with_suffix('').parts
        self.snapshot_dir = Path(*parts[:-1], parts[-1] + '_snaps')


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

        # General settings for the simulation
        #   inputfilepath: path to inputfile location
        #   outputfilepath: path to outputfile location
        #   messages: whether to print all messages as output to stdout or not
        #   progressbars: whether to show progress bars on stdoout or not
        #   cpu, cuda, opencl: solver type
        #   subgrid: whether the simulation uses sub-grids
        #   precision: data type for electromagnetic field output (single/double)
        self.general = {'messages': True,
                        'progressbars': True,
                        'cpu': True,
                        'cuda': False,
                        'opencl': False,
                        'subgrid': False,
                        'precision': 'single'}

        self.em_consts = {'c': c, # Speed of light in free space (m/s)
                          'e0': e0, # Permittivity of free space (F/m)
                          'm0': m0, # Permeability of free space (H/m)
                          'z0': np.sqrt(m0 / e0)} # Impedance of free space (Ohms)

        # Store information about host machine
        self.hostinfo = get_host_info()

        # Information about any Nvidia GPUs
        if self.args.gpu is not None:
            self.general['cuda'] = True
            self.general['cpu'] = False
            self.general['opencl'] = False
            # Both single and double precision are possible on GPUs, but single
            # provides best performance.
            self.general['precision'] = 'single'
            self.cuda = {'gpus': [], # gpus: list of GPU objects
                         'gpus_str': [], # gpus_str: list of strings describing GPU(s)
                         'nvcc_opts': None} # nvcc_opts: nvcc compiler options
            # Suppress nvcc warnings on Microsoft Windows
            if sys.platform == 'win32': self.cuda['nvcc_opts'] = ['-w']

            # Flatten a list of lists
            if any(isinstance(element, list) for element in self.args.gpu):
                self.args.gpu = [val for sublist in self.args.gpu for val in sublist]

            self.cuda['gpus'] = detect_check_gpus(self.args.gpu)

        # Subgrid parameter may not exist if user enters via CLI
        try:
            self.general['subgrid'] = self.args.subgrid
        except AttributeError:
            self.general['subgrid'] = False

        # Double precision should be used with subgrid for best accuracy
        if self.general['subgrid']:
            self.general['precision'] = 'double'

        if self.general['subgrid'] and self.general['cuda']:
            raise GeneralError('The CUDA-based solver cannot currently be used with models that contain sub-grids.')

        # Scenes parameter may not exist if user enters via CLI
        try:
            self.scenes = args.scenes
        except AttributeError:
            self.scenes = []

        # Set more complex parameters
        self.set_precision()
        self.get_byteorder()
        self.set_input_file_path()
        self.set_model_start_end()
        self.set_single_model()

    def is_messages(self):
        return self.general['messages']

    def set_model_gpu(self):
        """Specify single GPU object for model."""
        return self.cuda['gpus'][0]

    def set_precision(self):
        """Data type (precision) for electromagnetic field output.

            Solid and ID arrays use 32-bit integers (0 to 4294967295)
            Rigid arrays use 8-bit integers (the smallest available type to store true/false)
            Fractal arrays use complex numbers
            Dispersive coefficient arrays use either float or complex numbers
            Main field arrays use floats
        """

        if self.general['precision'] == 'single':
            self.dtypes = {'float_or_double': np.float32,
                      'complex': np.complex64,
                      'cython_float_or_double': cython.float,
                      'cython_complex': cython.floatcomplex,
                      'C_float_or_double': 'float',
                      'C_complex': 'pycuda::complex<float>',
                      'vtk_float': 'Float32'}
        elif self.general['precision'] == 'double':
            self.dtypes = {'float_or_double': np.float64,
                      'complex': np.complex128,
                      'cython_float_or_double': cython.double,
                      'cython_complex': cython.doublecomplex,
                      'C_float_or_double': 'double',
                      'C_complex': 'pycuda::complex<double>',
                      'vtk_float': 'Float64'}

    def get_byteorder(self):
        """Check the byte order of system to use for VTK files, i.e. geometry
            views and snapshots.
        """
        self.vtk_byteorder = 'LittleEndian' if sys.byteorder == 'little' else 'BigEndian'

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
        """Set input file path for CLI or API."""
        # API
        if self.args.inputfile is None:
            self.input_file_path = Path(self.args.outputfile)
        # API/CLI
        else:
            self.input_file_path = Path(self.args.inputfile)


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

    def set_gpus(self):
        """Leave list of GPU object(s) as multi-object list."""
        pass
