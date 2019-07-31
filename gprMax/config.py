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

import os

import cython
import numpy as np
from scipy.constants import c
from scipy.constants import mu_0 as m0
from scipy.constants import epsilon_0 as e0

from .utilities import get_host_info

# Impedance of free space (Ohms)
z0 = np.sqrt(m0 / e0)

# General setting for the simulation
#   inputfilepath: path to inputfile location
#   outputfilepath: path to outputfile location
#   messages: whether to print all messages as output to stdout or not
#   progressbars: whether to show progress bars on stdoout or not
#   mode: 2D TMx, 2D TMy, 2D TMz, or 3D
#   cpu, cuda, opencl: solver type
general = {'inputfilepath': '', 'outputfilepath': '', 'messages': True,
           'progressbars': True, 'mode': None, 'cpu': True, 'cuda': False, 'opencl': False}

# Store information about host machine
hostinfo = get_host_info()

# Store information for CUDA solver type
#   gpus: information about any GPUs as a list of GPU objects
#   snapsgpu2cpu: copy snapshot data from GPU to CPU during simulation
#     N.B. This will happen if the requested snapshots are too large to fit
#     on the memory of the GPU. If True this will slow performance significantly
cuda = {'gpus': None, 'snapsgpu2cpu': False}

# Numerical dispersion analysis parameters
#   highestfreqthres: threshold (dB) down from maximum power (0dB) of main frequency used
#     to calculate highest frequency for numerical dispersion analysis
#   maxnumericaldisp: maximum allowable percentage physical phase-velocity phase error
#   mingridsampling: minimum grid sampling of smallest wavelength for physical wave propagation
numdispersion = {'highestfreqthres': 40, 'maxnumericaldisp': 2, 'mingridsampling': 3}

# Materials
#   maxpoles: Maximum number of dispersive material poles in a model
materials = {'maxpoles': 0, 'dispersivedtype': None, 'dispersiveCdtype': None}

# Data types
#   Solid and ID arrays use 32-bit integers (0 to 4294967295)
#   Rigid arrays use 8-bit integers (the smallest available type to store true/false)
#   Fractal and dispersive coefficient arrays use complex numbers (complex)
#                    which are represented as two floats
#   Main field arrays use floats (float_or_double) and complex numbers (complex)
#   Precision of float_or_double and complex: single or double for numpy and C (CUDA) arrays
precision = 'single'

if precision == 'single':
    dtypes = {'float_or_double': np.float32, 'complex': np.complex64,
              'cython_float_or_double': cython.float, 'cython_complex': cython.floatcomplex,
              'C_float_or_double': 'float', 'C_complex': 'pycuda::complex<float>'}
elif precision == 'double':
    dtypes = {'float_or_double': np.float64, 'complex': np.complex128,
              'cython_float_or_double': cython.double, 'cython_complex': cython.doublecomplex,
              'C_float_or_double': 'double', 'C_complex': 'pycuda::complex<double>'}


def create_simulation_config(args):
    pass


class ModelConfig():

    def __init__(sim_config, i):
        self.sim_config = sim.sim_config
        # current model number (indexed from 0)
        self.i = i

        if not sim_config.single_model:
            # 1 indexed
            self.appendmodelnumber = str(self.i) + 1

        inputfilestr_f = '\n--- Model {}/{}, input file: {}'.format()
        self.inputfilestr_f.format(self.i + 1, self.sim_config.model_end, self.sim_config.inputfile.name)


        # Add the current model run to namespace that can be accessed by
        # user in any Python code blocks in input file
        self.usernamespace['current_model_run'] = self.i + 1


class SimulationConfig():

    def __init__(args):
        """Adapter for args into Simulation level configuration"""

        # adapt the arg properties to link smoothly with MPIRunner(), CPURunner() etc..

        # args.inputfile
        # args.n
        # args.task
        # args.restart
        # args.mpi
        # args.mpi_no_spawn
        # args.mpicomm
        # args.gpu
        # args.benchmark
        # args.geometry_only
        # args.geometry_fixed
        # args.write_processed

        self.model_start = 0
        self.model_end = 1
        self.n_models = 0
        self.inputfile = args.inputfile
        self.inputfilepath = os.path.realpath(inputfile.name)
        self.outputfilepath = os.path.dirname(os.path.abspath(self.inputfilepath))


    def set_single_model(self):
        if self.mode_start == 0 and self.model_end == 1:
            self.single_model = True
        else:
            self.single_model = False

    # for example
    def set_model_start(self):

        # standard simulation
        if not self.args.mpi and not self.args.mpi_no_spawn:
            # Set range for number of models to run
            if args.task:
              # Job array feeds args.n number of single tasks
              self.modelstart = args.task
            elif args.restart:
              self.modelstart = args.restart
            else:
              self.modelstart = 1
        # mpi
        elif args.mpi:
        # Set range for number of models to run
            self.modelstart = args.restart if args.restart else 1 # etc...

    def set_model_end():
      pass

    def set_precision():
      pass
