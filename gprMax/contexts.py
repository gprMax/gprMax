# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
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

import gprMax.config as config

from ._version import __version__, codename
from .model_build_run import ModelBuildRun
from .solvers import create_G, create_solver
from .utilities.host_info import (detect_cuda_gpus, detect_opencl,
                                  print_cuda_info, print_host_info,
                                  print_opencl_info)
from .utilities.utilities import get_terminal_width, logo, timer

logger = logging.getLogger(__name__)


class Context:
    """Standard context - models are run one after another and each model
        can exploit parallelisation using either OpenMP (CPU), CUDA (GPU), or
        OpenCL (CPU/GPU).
    """

    def __init__(self):
        self.model_range = range(config.sim_config.model_start, 
                                 config.sim_config.model_end)
        self.tsimend = None
        self.tsimstart = None        

    def run(self):
        """Run the simulation in the correct context."""
        self.tsimstart = timer()
        self.print_logo_copyright()
        print_host_info(config.sim_config.hostinfo)
        if config.sim_config.general['solver'] == 'cuda':
            print_cuda_info(config.sim_config.devices['devs'])
        elif config.sim_config.general['solver'] == 'opencl':
            print_opencl_info(config.sim_config.devices['devs'])

        # Clear list of model configs. It can be retained when gprMax is
        # called in a loop, and want to avoid this.
        config.model_configs = []

        for i in self.model_range:
            config.model_num = i
            model_config = config.ModelConfig()
            config.model_configs.append(model_config)

            # Always create a grid for the first model. The next model to run
            # only gets a new grid if the geometry is not re-used.
            if i != 0 and config.sim_config.args.geometry_fixed:
                config.get_model_config().reuse_geometry = True
            else:
                G = create_G()

            model = ModelBuildRun(G)
            model.build()
            
            if not config.sim_config.args.geometry_only:
                solver = create_solver(G)
                model.solve(solver)

        self.tsimend = timer()
        self.print_time_report()

    def print_logo_copyright(self):
        """Print gprMax logo, version, and copyright/licencing information."""
        logo_copyright = logo(__version__ + ' (' + codename + ')')
        logger.basic(logo_copyright)

    def print_time_report(self):
        """Print the total simulation time based on context."""
        s = ("\n=== Simulation completed in [HH:MM:SS]: "
             f"{datetime.timedelta(seconds=self.tsimend - self.tsimstart)}")
        logger.basic(f"{s} {'=' * (get_terminal_width() - 1 - len(s))}\n")


class MPIContext(Context):
    """Mixed mode MPI/OpenMP/CUDA context - MPI task farm is used to distribute
        models, and each model parallelised using either OpenMP (CPU), 
        CUDA (GPU), or OpenCL (CPU/GPU).
    """

    def __init__(self):
        super().__init__()
        from mpi4py import MPI

        from gprMax.mpi import MPIExecutor

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.MPIExecutor = MPIExecutor

    def _run_model(self, **work):
        """Process for running a single model.
        
        Args:
            work (dict): contains any additional information that is passed to 
                            MPI workers. By default only model number (i) is 
                            used.
        """

        # Create configuration for model
        config.model_num = work['i']
        model_config = config.ModelConfig()
        # Set GPU deviceID according to worker rank
        if config.sim_config.general['cuda']:
            model_config.cuda = {'gpu': config.sim_config.cuda['gpus'][self.rank - 1],
                                 'snapsgpu2cpu': False}
        config.model_configs = model_config

        G = create_G()
        model = ModelBuildRun(G)
        model.build()
        
        if not config.sim_config.args.geometry_only:
            solver = create_solver(G)
            model.solve(solver)

    def run(self):
        """Specialise how the models are run."""
        if self.rank == 0:
            self.tsimstart = timer()
            self.print_logo_copyright()
            self.print_host_info()
            if config.sim_config.general['cuda']:
                self.print_gpu_info()
            sys.stdout.flush()

        # Contruct MPIExecutor
        executor = self.MPIExecutor(self._run_model, comm=self.comm)

        # Check GPU resources versus number of MPI tasks
        if executor.is_master():
            if config.sim_config.general['cuda']:
                if executor.size - 1 > len(config.sim_config.cuda['gpus']):
                    logger.exception('Not enough GPU resources for number of '
                                     'MPI tasks requested. Number of MPI tasks '
                                     'should be equal to number of GPUs + 1.')
                    raise ValueError

        # Create job list
        jobs = []
        for i in self.model_range:
            jobs.append({'i': i})

        # Send the workers to their work loop
        executor.start()
        if executor.is_master():
            results = executor.submit(jobs)

        # Make the workers exit their work loop and join the main loop again
        executor.join()

        if executor.is_master():
            self.tsimend = timer()
            self.print_time_report()


class SPOTPYContext(Context):
    """Specialised context used when gprMax is coupled with SPOTPY 
        (https://github.com/thouska/spotpy). SPOTPY coupling can utilise 2 levels
        of MPI parallelism - where the top level is where SPOPTY optmisation 
        algorithms can be parallelised, and the lower level is where gprMax
        models can be parallelised using either OpenMP (CPU), CUDA (GPU), or
        OpenCL (CPU/GPU).
    """

    def __init__(self):
        super().__init__()

    def run(self, i):
        """Process for running a single model."""

        # self.print_logo_copyright()
        # self.print_host_info()
        # if config.sim_config.general['cuda']:
        #     self.print_gpu_info()
        self.tsimstart = timer()

        # Create configuration for model
        config.model_num = i
        model_config = config.ModelConfig()
        config.model_configs = model_config

        G = create_G()
        model = ModelBuildRun(G)
        model.build()
        
        if not config.sim_config.args.geometry_only:
            solver = create_solver(G)
            model.solve(solver)

        self.tsimend = timer()
        self.print_time_report()
