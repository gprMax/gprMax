# Copyright (C) 2015-2021: The University of Edinburgh
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
from .utilities.utilities import get_terminal_width, human_size, logo, timer

import os
from .random_gen import compress_pkl_dat

logger = logging.getLogger(__name__)


class Context:
    """Standard context - models are run one after another and each model
        can exploit parallelisation using either OpenMP (CPU) or CUDA (GPU).
    """

    def __init__(self):
        self.model_range = range(config.sim_config.model_start, config.sim_config.model_end)
        self.tsimend = None
        self.tsimstart = None       
        self.data_labels = None 

    def run(self):
        """Run the simulation in the correct context."""
        self.tsimstart = timer()
        self.print_logo_copyright()
        self.print_host_info()
        if config.sim_config.general['cuda']:
            self.print_gpu_info()
        
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
            self.data_labels = model.data_labels
            
            if not config.sim_config.args.geometry_only:
                solver = create_solver(G)
                model.solve(solver)

        self.tsimend = timer()
        self.print_time_report()

        if os.path.isfile(config.get_model_config().output_file_path_ext_random):
            self.display_labels(self.data_labels)
            self.save_compressed_data()

    def print_logo_copyright(self):
        """Print gprMax logo, version, and copyright/licencing information."""
        logo_copyright = logo(__version__ + ' (' + codename + ')')
        logger.basic(logo_copyright)

    def print_host_info(self):
        """Print information about the host machine."""
        hyperthreadingstr = f", {config.sim_config.hostinfo['logicalcores']} cores with Hyper-Threading" if config.sim_config.hostinfo['hyperthreading'] else ''
        logger.basic(f"\nHost: {config.sim_config.hostinfo['hostname']} | {config.sim_config.hostinfo['machineID']} | {config.sim_config.hostinfo['sockets']} x {config.sim_config.hostinfo['cpuID']} ({config.sim_config.hostinfo['physicalcores']} cores{hyperthreadingstr}) | {human_size(config.sim_config.hostinfo['ram'], a_kilobyte_is_1024_bytes=True)} RAM | {config.sim_config.hostinfo['osversion']}")

    def print_gpu_info(self):
        """Print information about any NVIDIA CUDA GPUs detected."""
        gpus_info = []
        for gpu in config.sim_config.cuda['gpus']:
            gpus_info.append(f'{gpu.deviceID} - {gpu.name}, {human_size(gpu.totalmem, a_kilobyte_is_1024_bytes=True)}')
        logger.basic(f"GPU resources: {' | '.join(gpus_info)}")

    def print_time_report(self):
        """Print the total simulation time based on context."""
        s = f"\n=== Simulation completed in [HH:MM:SS]: {datetime.timedelta(seconds=self.tsimend - self.tsimstart)}"
        logger.basic(f"{s} {'=' * (get_terminal_width() - 1 - len(s))}\n")

    def save_compressed_data(self):
        """Remove all redundant features and save compressed data file containing randomly generated parameters"""
        compress_pkl_dat(config.get_model_config().output_file_path_ext_random, str(config.get_model_config().output_file_path_random) + '_{compressed}.pkl')

    def display_labels(self, labels):
        """Display labels corresponding to the random parameters in the order they are saved"""
        from terminaltables import SingleTable
        labels_table = SingleTable(labels)
        labels_table.outer_border = False
        labels_table.justify_columns[0] = 'right'
        logger.info(f'-> Data Labels \n')
        logger.info(labels_table.table)


class MPIContext(Context):
    """Mixed mode MPI/OpenMP/CUDA context - MPI task farm is used to distribute
        models, and each model parallelised using either OpenMP (CPU)
        or CUDA (GPU).
    """

    def __init__(self):
        super().__init__()
        from mpi4py import MPI
        from gprMax.mpi import MPIExecutor

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.MPIExecutor = MPIExecutor
        self.data_labels = None

    def _run_model(self, i):
        """Process for running a single model."""

        # Create configuration for model
        config.model_num = i
        model_config = config.ModelConfig()
        # Set GPU deviceID according to worker rank
        if config.sim_config.general['cuda']:
            model_config.cuda = {'gpu': config.sim_config.cuda['gpus'][self.rank - 1],
                                 'snapsgpu2cpu': False}
        config.model_configs = model_config

        G = create_G()
        model = ModelBuildRun(G)
        model.build()
        self.data_labels = model.data_labels
        
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
                    logger.exception('Not enough GPU resources for number of MPI tasks requested. Number of MPI tasks should be equal to number of GPUs + 1.')
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

            if os.path.isfile(config.get_model_config().output_file_path_ext_random):
                self.display_labels(self.data_labels)
                self.save_compressed_data()


class SPOTPYContext(Context):
    """Specialised context used when gprMax is coupled with SPOTPY 
        (https://github.com/thouska/spotpy). SPOTPY coupling can utilise 2 levels
        of MPI parallelism - where the top level is where SPOPTY optmisation 
        algorithms can be parallelised, and the lower level is where gprMax
        models can be parallelised using either OpenMP (CPU) or CUDA (GPU).
    """

    def __init__(self):
        super().__init__()
        self.data_labels = None

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
        self.data_labels = model.data_labels
        
        if not config.sim_config.args.geometry_only:
            solver = create_solver(G)
            model.solve(solver)

        self.tsimend = timer()
        self.print_time_report()
        
        if os.path.isfile(config.get_model_config().output_file_path_ext_random):
            self.display_labels(self.data_labels)
            self.save_compressed_data()
