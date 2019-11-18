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

import gprMax.config as config


def write_simulation_config(args):
    """Write simulation level configuration parameters to config module. As
        there can only be one instance of the config module objects are always
        found via 'import gprMax.config'

    Args:
        args (Namespace): Arguments from either API or CLI.
    """

    if args.mpi or args.mpi_no_spawn:
        config.sim_config = config.SimulationConfigMPI(args)
    else:
        config.sim_config = config.SimulationConfig(args)


def write_model_config():
    """Write model level configuration parameters to config module. As there can
        only be one instance of the config module objects are always found via
        'import gprMax.config'
    """

    model_config = config.ModelConfig()
    config.model_configs.append(model_config)
