# Copyright (C) 2015-2017: The University of Edinburgh
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
import decimal as d
import sys

from colorama import init, Fore, Style
init()
import numpy as np

from gprMax.constants import c, floattype
from gprMax.exceptions import CmdInputError, GeneralError
from gprMax.utilities import get_host_info, human_size, memory_usage, round_value
from gprMax.waveforms import Waveform


def process_singlecmds(singlecmds, G):
    """Checks the validity of command parameters and creates instances of classes of parameters.

    Args:
        singlecmds (dict): Commands that can only occur once in the model.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    # Check validity of command parameters in order needed
    # messages
    cmd = '#messages'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            raise CmdInputError(cmd + ' requires exactly one parameter')
        if singlecmds[cmd].lower() == 'y':
            G.messages = True
        elif singlecmds[cmd].lower() == 'n':
            G.messages = False
        else:
            raise CmdInputError(cmd + ' requires input values of either y or n')

    # Title
    cmd = '#title'
    if singlecmds[cmd] != 'None':
        G.title = singlecmds[cmd]
        if G.messages:
            print('Model title: {}'.format(G.title))

    # Get information about host machine
    hostinfo = get_host_info()

    # Number of threads (OpenMP) to use
    cmd = '#num_threads'
    if sys.platform == 'darwin':
        os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'  # Should waiting threads consume CPU power (can drastically effect performance)
    os.environ['OMP_DYNAMIC'] = 'FALSE' # Number of threads may be adjusted by the run time environment to best utilize system resources
    os.environ['OMP_PLACES'] = 'cores' # Each place corresponds to a single core (having one or more hardware threads)
    #os.environ['OMP_PLACES'] = '{0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30}'
    os.environ['OMP_PROC_BIND'] = 'TRUE'  # Bind threads to physical cores
    # os.environ['OMP_DISPLAY_ENV'] = 'TRUE' # Prints OMP version and environment variables (useful for debug)

    # Catch bug with Windows Subsystem for Linux (https://github.com/gprMax/gprMax/issues/95)
    if 'Microsoft' in hostinfo['osversion']:
        os.environ['KMP_AFFINITY'] = 'disabled'
        del os.environ['OMP_PLACES']
        del os.environ['OMP_PROC_BIND']

    if singlecmds[cmd] != 'None':
        tmp = tuple(int(x) for x in singlecmds[cmd].split())
        if len(tmp) != 1:
            raise CmdInputError(cmd + ' requires exactly one parameter to specify the number of threads to use')
        if tmp[0] < 1:
            raise CmdInputError(cmd + ' requires the value to be an integer not less than one')
        G.nthreads = tmp[0]
        os.environ['OMP_NUM_THREADS'] = str(G.nthreads)
    elif os.environ.get('OMP_NUM_THREADS'):
        G.nthreads = int(os.environ.get('OMP_NUM_THREADS'))
    else:
        # Set number of threads to number of physical CPU cores
        G.nthreads = hostinfo['physicalcores']
        os.environ['OMP_NUM_THREADS'] = str(G.nthreads)

    if G.messages:
        print('Number of CPU (OpenMP) threads: {}'.format(G.nthreads))
    if G.nthreads > hostinfo['physicalcores']:
        print(Fore.RED + 'WARNING: You have specified more threads ({}) than available physical CPU cores ({}). This may lead to degraded performance.'.format(G.nthreads, hostinfo['physicalcores']) + Style.RESET_ALL)

    # Spatial discretisation
    cmd = '#dx_dy_dz'
    tmp = [float(x) for x in singlecmds[cmd].split()]
    if len(tmp) != 3:
        raise CmdInputError(cmd + ' requires exactly three parameters')
    if tmp[0] <= 0:
        raise CmdInputError(cmd + ' requires the x-direction spatial step to be greater than zero')
    if tmp[1] <= 0:
        raise CmdInputError(cmd + ' requires the y-direction spatial step to be greater than zero')
    if tmp[2] <= 0:
        raise CmdInputError(cmd + ' requires the z-direction spatial step to be greater than zero')
    G.dx = tmp[0]
    G.dy = tmp[1]
    G.dz = tmp[2]
    if G.messages:
        print('Spatial discretisation: {:g} x {:g} x {:g}m'.format(G.dx, G.dy, G.dz))

    # Domain
    cmd = '#domain'
    tmp = [float(x) for x in singlecmds[cmd].split()]
    if len(tmp) != 3:
        raise CmdInputError(cmd + ' requires exactly three parameters')
    G.nx = round_value(tmp[0] / G.dx)
    G.ny = round_value(tmp[1] / G.dy)
    G.nz = round_value(tmp[2] / G.dz)
    if G.nx == 0 or G.ny == 0 or G.nz == 0:
        raise CmdInputError(cmd + ' requires at least one cell in every dimension')
    if G.messages:
        print('Domain size: {:g} x {:g} x {:g}m ({:d} x {:d} x {:d} = {:g} cells)'.format(tmp[0], tmp[1], tmp[2], G.nx, G.ny, G.nz, (G.nx * G.ny * G.nz)))

    # Estimate memory (RAM) usage
    memestimate = memory_usage(G)
    if memestimate > hostinfo['ram']:
        print(Fore.RED + 'WARNING: Estimated memory (RAM) required ~{} exceeds {} detected!\n'.format(human_size(memestimate), human_size(hostinfo['ram'], a_kilobyte_is_1024_bytes=True)) + Style.RESET_ALL)
    if G.messages:
        print('Estimated memory (RAM) required: ~{}'.format(human_size(memestimate)))

    # Time step CFL limit (use either 2D or 3D) and default PML thickness
    if G.nx == 1:
        G.dt = 1 / (c * np.sqrt((1 / G.dy) * (1 / G.dy) + (1 / G.dz) * (1 / G.dz)))
        G.dimension = '2D'
        G.pmlthickness['x0'] = 0
        G.pmlthickness['xmax'] = 0
    elif G.ny == 1:
        G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dz) * (1 / G.dz)))
        G.dimension = '2D'
        G.pmlthickness['y0'] = 0
        G.pmlthickness['ymax'] = 0
    elif G.nz == 1:
        G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dy) * (1 / G.dy)))
        G.dimension = '2D'
        G.pmlthickness['z0'] = 0
        G.pmlthickness['zmax'] = 0
    else:
        G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dy) * (1 / G.dy) + (1 / G.dz) * (1 / G.dz)))
        G.dimension = '3D'

    # Round down time step to nearest float with precision one less than hardware maximum. Avoids inadvertently exceeding the CFL due to binary representation of floating point number.
    G.dt = round_value(G.dt, decimalplaces=d.getcontext().prec - 1)

    if G.messages:
        print('Time step (at {} CFL limit): {:g} secs'.format(G.dimension, G.dt))

    # Time step stability factor
    cmd = '#time_step_stability_factor'
    if singlecmds[cmd] != 'None':
        tmp = tuple(float(x) for x in singlecmds[cmd].split())
        if len(tmp) != 1:
            raise CmdInputError(cmd + ' requires exactly one parameter')
        if tmp[0] <= 0 or tmp[0] > 1:
            raise CmdInputError(cmd + ' requires the value of the time step stability factor to be between zero and one')
        G.dt = G.dt * tmp[0]
        if G.messages:
            print('Time step (modified): {:g} secs'.format(G.dt))

    # Time window
    cmd = '#time_window'
    tmp = singlecmds[cmd].split()
    if len(tmp) != 1:
        raise CmdInputError(cmd + ' requires exactly one parameter to specify the time window. Either in seconds or number of iterations.')
    tmp = tmp[0].lower()

    # If number of iterations given
    try:
        tmp = int(tmp)
        G.timewindow = (tmp - 1) * G.dt
        G.iterations = tmp
    # If real floating point value given
    except:
        tmp = float(tmp)
        if tmp > 0:
            G.timewindow = tmp
            G.iterations = round_value((tmp / G.dt)) + 1
        else:
            raise CmdInputError(cmd + ' must have a value greater than zero')
    if G.messages:
        print('Time window: {:g} secs ({} iterations)'.format(G.timewindow, G.iterations))

    # PML
    cmd = '#pml_cells'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1 and len(tmp) != 6:
            raise CmdInputError(cmd + ' requires either one or six parameters')
        if len(tmp) == 1:
            for key in G.pmlthickness.keys():
                G.pmlthickness[key] = int(tmp[0])
        else:
            G.pmlthickness['x0'] = int(tmp[0])
            G.pmlthickness['y0'] = int(tmp[1])
            G.pmlthickness['z0'] = int(tmp[2])
            G.pmlthickness['xmax'] = int(tmp[3])
            G.pmlthickness['ymax'] = int(tmp[4])
            G.pmlthickness['zmax'] = int(tmp[5])
    if 2 * G.pmlthickness['x0'] >= G.nx or 2 * G.pmlthickness['y0'] >= G.ny or 2 * G.pmlthickness['z0'] >= G.nz or 2 * G.pmlthickness['xmax'] >= G.nx or 2 * G.pmlthickness['ymax'] >= G.ny or 2 * G.pmlthickness['zmax'] >= G.nz:
        raise CmdInputError(cmd + ' has too many cells for the domain size')

    # src_steps
    cmd = '#src_steps'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 3:
            raise CmdInputError(cmd + ' requires exactly three parameters')
        G.srcsteps[0] = round_value(float(tmp[0]) / G.dx)
        G.srcsteps[1] = round_value(float(tmp[1]) / G.dy)
        G.srcsteps[2] = round_value(float(tmp[2]) / G.dz)
        if G.messages:
            print('Simple sources will step {:g}m, {:g}m, {:g}m for each model run.'.format(G.srcsteps[0] * G.dx, G.srcsteps[1] * G.dy, G.srcsteps[2] * G.dz))

    # rx_steps
    cmd = '#rx_steps'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 3:
            raise CmdInputError(cmd + ' requires exactly three parameters')
        G.rxsteps[0] = round_value(float(tmp[0]) / G.dx)
        G.rxsteps[1] = round_value(float(tmp[1]) / G.dy)
        G.rxsteps[2] = round_value(float(tmp[2]) / G.dz)
        if G.messages:
            print('All receivers will step {:g}m, {:g}m, {:g}m for each model run.'.format(G.rxsteps[0] * G.dx, G.rxsteps[1] * G.dy, G.rxsteps[2] * G.dz))

    # Excitation file for user-defined source waveforms
    cmd = '#excitation_file'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            raise CmdInputError(cmd + ' requires exactly one parameter')
        excitationfile = tmp[0]

        # See if file exists at specified path and if not try input file directory
        if not os.path.isfile(excitationfile):
            excitationfile = os.path.abspath(os.path.join(G.inputdirectory, excitationfile))

        # Get waveform names
        with open(excitationfile, 'r') as f:
            waveformIDs = f.readline().split()

        # Read all waveform values into an array
        waveformvalues = np.loadtxt(excitationfile, skiprows=1, dtype=floattype)

        for waveform in range(len(waveformIDs)):
            if any(x.ID == waveformIDs[waveform] for x in G.waveforms):
                raise CmdInputError('Waveform with ID {} already exists'.format(waveformIDs[waveform]))
            w = Waveform()
            w.ID = waveformIDs[waveform]
            w.type = 'user'
            if len(waveformvalues.shape) == 1:
                w.uservalues = waveformvalues[:]
            else:
                w.uservalues = waveformvalues[:, waveform]

            if G.messages:
                print('User waveform {} created.'.format(w.ID))

            G.waveforms.append(w)
