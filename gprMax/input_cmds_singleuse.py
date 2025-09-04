# Copyright (C) 2015-2023: The University of Edinburgh
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
import inspect
import sys

from colorama import init
from colorama import Fore
from colorama import Style
init()
import numpy as np
from scipy import interpolate

from gprMax.constants import c
from gprMax.constants import floattype
from gprMax.exceptions import CmdInputError
from gprMax.exceptions import GeneralError
from gprMax.pml import PML
from gprMax.utilities import get_host_info
from gprMax.utilities import human_size
from gprMax.utilities import round_value
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
    if singlecmds[cmd] is not None:
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
    if singlecmds[cmd] is not None:
        G.title = singlecmds[cmd]
        if G.messages:
            print('Model title: {}'.format(G.title))

    # Get information about host machine
    hostinfo = get_host_info()

    # Number of threads (OpenMP) to use
    cmd = '#num_threads'
    if sys.platform == 'darwin':
        os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'  # Should waiting threads consume CPU power (can drastically effect performance)
    os.environ['OMP_DYNAMIC'] = 'FALSE'  # Number of threads may be adjusted by the run time environment to best utilize system resources
    os.environ['OMP_PLACES'] = 'cores'  # Each place corresponds to a single core (having one or more hardware threads)
    os.environ['OMP_PROC_BIND'] = 'TRUE'  # Bind threads to physical cores
    # os.environ['OMP_DISPLAY_ENV'] = 'TRUE' # Prints OMP version and environment variables (useful for debug)

    # Catch bug with Windows Subsystem for Linux (https://github.com/Microsoft/BashOnWindows/issues/785)
    if 'Microsoft' in G.hostinfo['osversion']:
        os.environ['KMP_AFFINITY'] = 'disabled'
        del os.environ['OMP_PLACES']
        del os.environ['OMP_PROC_BIND']

    if singlecmds[cmd] is not None:
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
    if G.nthreads > G.hostinfo['physicalcores']:
        print(Fore.RED + 'WARNING: You have specified more threads ({}) than available physical CPU cores ({}). This may lead to degraded performance.'.format(G.nthreads, hostinfo['physicalcores']) + Style.RESET_ALL)

    # Print information about any GPU in use
    if G.messages:
        if G.gpu is not None:
            print('GPU solving using: {} - {}'.format(G.gpu.deviceID, G.gpu.name))

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

    # Time step CFL limit (either 2D or 3D); switch off appropriate PMLs for 2D
    if G.nx == 1:
        G.dt = 1 / (c * np.sqrt((1 / G.dy) * (1 / G.dy) + (1 / G.dz) * (1 / G.dz)))
        G.mode = '2D TMx'
        G.pmlthickness['x0'] = 0
        G.pmlthickness['xmax'] = 0
    elif G.ny == 1:
        G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dz) * (1 / G.dz)))
        G.mode = '2D TMy'
        G.pmlthickness['y0'] = 0
        G.pmlthickness['ymax'] = 0
    elif G.nz == 1:
        G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dy) * (1 / G.dy)))
        G.mode = '2D TMz'
        G.pmlthickness['z0'] = 0
        G.pmlthickness['zmax'] = 0
    else:
        G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dy) * (1 / G.dy) + (1 / G.dz) * (1 / G.dz)))
        G.mode = '3D'

    # Round down time step to nearest float with precision one less than hardware maximum.
    # Avoids inadvertently exceeding the CFL due to binary representation of floating point number.
    G.dt = round_value(G.dt, decimalplaces=d.getcontext().prec - 1)

    if G.messages:
        print('Mode: {}'.format(G.mode))
        print('Time step (at CFL limit): {:g} secs'.format(G.dt))

    # Time step stability factor
    cmd = '#time_step_stability_factor'
    if singlecmds[cmd] is not None:
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
    # The +/- 1 used in calculating the number of iterations is to account for
    # the fact that the solver (iterations) loop runs from 0 to < G.iterations
    try:
        tmp = int(tmp)
        G.timewindow = (tmp - 1) * G.dt
        G.iterations = tmp
    # If real floating point value given
    except ValueError:
        tmp = float(tmp)
        if tmp > 0:
            G.timewindow = tmp
            G.iterations = int(np.ceil(tmp / G.dt)) + 1
        else:
            raise CmdInputError(cmd + ' must have a value greater than zero')
    if G.messages:
        print('Time window: {:g} secs ({} iterations)'.format(G.timewindow, G.iterations))

    # PML cells
    cmd = '#pml_cells'
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1 and len(tmp) != 6:
            raise CmdInputError(cmd + ' requires either one or six parameter(s)')
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

    # PML formulation
    cmd = '#pml_formulation'
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            raise CmdInputError(cmd + ' requires exactly one parameter')
        if singlecmds[cmd].upper() in PML.formulations:
            G.pmlformulation = singlecmds[cmd].upper()
        else:
            raise CmdInputError(cmd + ' PML formulation is not found')

    # src_steps
    cmd = '#src_steps'
    if singlecmds[cmd] is not None:
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
    if singlecmds[cmd] is not None:
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
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1 and len(tmp) != 3:
            raise CmdInputError(cmd + ' requires either one or three parameter(s)')
        excitationfile = tmp[0]

        # Optional parameters passed directly to scipy.interpolate.interp1d
        kwargs = dict()
        if len(tmp) > 1:
            kwargs['kind'] = tmp[1]
            kwargs['fill_value'] = tmp[2]
        else:
            fullargspec = inspect.getfullargspec(interpolate.interp1d)
            kwargs = dict(zip(reversed(fullargspec.args), 
                              reversed(fullargspec.defaults)))

        # See if file exists at specified path and if not try input file directory
        if not os.path.isfile(excitationfile):
            excitationfile = os.path.abspath(os.path.join(G.inputdirectory, excitationfile))

        if G.messages:
            print('\nExcitation file: {}'.format(excitationfile))

        # Get waveform names
        with open(excitationfile, 'r') as f:
            waveformIDs = f.readline().split()

        # Read all waveform values into an array
        waveformvalues = np.loadtxt(excitationfile, skiprows=1, dtype=floattype)

        # Time array (if specified) for interpolation, otherwise use simulation time
        if waveformIDs[0].lower() == 'time':
            waveformIDs = waveformIDs[1:]
            waveformtime = waveformvalues[:, 0]
            waveformvalues = waveformvalues[:, 1:]
            timestr = 'user-defined time array'
        else:
            waveformtime = np.arange(0, G.timewindow + G.dt, G.dt)
            timestr = 'simulation time array'

        for waveform in range(len(waveformIDs)):
            if any(x.ID == waveformIDs[waveform] for x in G.waveforms):
                raise CmdInputError('Waveform with ID {} already exists'.format(waveformIDs[waveform]))
            w = Waveform()
            w.ID = waveformIDs[waveform]
            w.type = 'user'

            # Select correct column of waveform values depending on array shape
            singlewaveformvalues = waveformvalues[:] if len(waveformvalues.shape) == 1 else waveformvalues[:, waveform]

            # Truncate waveform array if it is longer than time array
            if len(singlewaveformvalues) > len(waveformtime):
                singlewaveformvalues = singlewaveformvalues[:len(waveformtime)]
            # Zero-pad end of waveform array if it is shorter than time array
            elif len(singlewaveformvalues) < len(waveformtime):
                tmp = np.zeros(len(waveformtime))
                tmp[:len(singlewaveformvalues)] = singlewaveformvalues
                singlewaveformvalues = tmp

            # Interpolate waveform values
            w.userfunc = interpolate.interp1d(waveformtime, singlewaveformvalues, **kwargs)

            if G.messages:
                print('User waveform {} created using {} and, if required, interpolation parameters (kind: {}, fill value: {}).'.format(w.ID, timestr, kwargs['kind'], kwargs['fill_value']))

            G.waveforms.append(w)

    # Set the output directory
    cmd = '#output_dir'
    if singlecmds[cmd] is not None:
        outputdir = singlecmds[cmd]
        G.outputdirectory = outputdir
