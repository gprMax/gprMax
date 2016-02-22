# Copyright (C) 2015-2016: The University of Edinburgh
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

import os, psutil
import decimal as d
import numpy as np

from gprMax.constants import c, floattype
from gprMax.exceptions import CmdInputError
from gprMax.pml import PML, CFS
from gprMax.utilities import round_value, human_size
from gprMax.waveforms import Waveform


def process_singlecmds(singlecmds, multicmds, G):
    """Checks the validity of command parameters and creates instances of classes of parameters.
        
    Args:
        singlecmds (dict): Commands that can only occur once in the model.
        multicmds (dict): Commands that can have multiple instances in the model (required to pass to process_materials_file function).
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


    # Number of processors to run on (OpenMP)
    cmd = '#num_threads'
    ompthreads = os.environ.get('OMP_NUM_THREADS')
    if singlecmds[cmd] != 'None':
        tmp = tuple(int(x) for x in singlecmds[cmd].split())
        if len(tmp) != 1:
            raise CmdInputError(cmd + ' requires exactly one parameter to specify the number of threads to use')
        if tmp[0] < 1:
            raise CmdInputError(cmd + ' requires the value to be an integer not less than one')
        G.nthreads = tmp[0]
    elif ompthreads:
        G.nthreads = int(ompthreads)
    else:
        # Set number of threads to number of physical CPU cores, i.e. avoid hyperthreading with OpenMP
        G.nthreads = psutil.cpu_count(logical=False)
    if G.messages:
            print('Number of threads: {}'.format(G.nthreads))


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
    G.nx = round_value(tmp[0]/G.dx)
    G.ny = round_value(tmp[1]/G.dy)
    G.nz = round_value(tmp[2]/G.dz)
    if G.messages:
        print('Model domain: {:g} x {:g} x {:g}m ({:d} x {:d} x {:d} = {:g} cells)'.format(tmp[0], tmp[1], tmp[2], G.nx, G.ny, G.nz, (G.nx * G.ny * G.nz)))
        mem = (((G.nx + 1) * (G.ny + 1) * (G.nz + 1) * 13 * np.dtype(floattype).itemsize + (G.nx + 1) * (G.ny + 1) * (G.nz + 1) * 18) * 1.1) + 30e6
        print('Memory (RAM) usage: ~{} required, {} available'.format(human_size(mem), human_size(psutil.virtual_memory().total)))


    # Time step CFL limit - use either 2D or 3D (default)
    cmd = '#time_step_limit_type'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            raise CmdInputError(cmd + ' requires exactly one parameter')
        if singlecmds[cmd].lower() == '2d':
            if G.nx == 1:
                G.dt = 1 / (c * np.sqrt((1 / G.dy) * (1 / G.dy) + (1 / G.dz) * (1 / G.dz)))
            elif G.ny == 1:
                G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dz) * (1 / G.dz)))
            elif G.nz == 1:
                G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dy) * (1 / G.dy)))
            else:
                raise CmdInputError(cmd + ' 2D CFL limit can only be used when one dimension of the domain is one cell')
        elif singlecmds[cmd].lower() == '3d':
            G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dy) * (1 / G.dy) + (1 / G.dz) * (1 / G.dz)))
        else:
            raise CmdInputError(cmd + ' requires input values of either 2D or 3D')
    else:
        G.dt = 1 / (c * np.sqrt((1 / G.dx) * (1 / G.dx) + (1 / G.dy) * (1 / G.dy) + (1 / G.dz) * (1 / G.dz)))

    # Round down time step to nearest float with precision one less than hardware maximum. Avoids inadvertently exceeding the CFL due to binary representation of floating point number.
    G.dt = round_value(G.dt, decimalplaces=d.getcontext().prec - 1)

    if G.messages:
        print('Time step: {:g} secs'.format(G.dt))


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
    # If real floating point value given
    if '.' in tmp or 'e' in tmp:
        if float(tmp) > 0:
            G.timewindow = float(tmp)
            G.iterations = round_value((float(tmp) / G.dt)) + 1
        else:
            raise CmdInputError(cmd + ' must have a value greater than zero')
    # If number of iterations given
    else:
        G.timewindow = (int(tmp) - 1) * G.dt
        G.iterations = int(tmp)
    if G.messages:
        print('Time window: {:g} secs ({} iterations)'.format(G.timewindow, G.iterations))


    # PML
    cmd = '#pml_cells'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1 and len(tmp) != 6:
            raise CmdInputError(cmd + ' requires either one or six parameters')
        if len(tmp) == 1:
            G.pmlthickness = (int(tmp[0]), int(tmp[0]), int(tmp[0]), int(tmp[0]), int(tmp[0]), int(tmp[0]))
        else:
            G.pmlthickness = (int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]), int(tmp[4]), int(tmp[5]))
    if 2*G.pmlthickness[0] >= G.nx or 2*G.pmlthickness[1] >= G.ny or 2*G.pmlthickness[2] >= G.nz or 2*G.pmlthickness[3] >= G.nx or 2*G.pmlthickness[4] >= G.ny or 2*G.pmlthickness[5] >= G.nz:
        raise CmdInputError(cmd + ' has too many cells for the domain size')

    
    # src_steps
    cmd = '#src_steps'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 3:
            raise CmdInputError(cmd + ' requires exactly three parameters')
        G.srcstepx = round_value(float(tmp[0])/G.dx)
        G.srcstepy = round_value(float(tmp[1])/G.dy)
        G.srcstepz = round_value(float(tmp[2])/G.dz)
        if G.messages:
            print('All sources will step {:g}m, {:g}m, {:g}m for each model run.'.format(G.srcstepx * G.dx, G.srcstepy * G.dy, G.srcstepz * G.dz))


    # rx_steps
    cmd = '#rx_steps'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 3:
            raise CmdInputError(cmd + ' requires exactly three parameters')
        G.rxstepx = round_value(float(tmp[0])/G.dx)
        G.rxstepy = round_value(float(tmp[1])/G.dy)
        G.rxstepz = round_value(float(tmp[2])/G.dz)
        if G.messages:
            print('All receivers will step {:g}m, {:g}m, {:g}m for each model run.'.format(G.rxstepx * G.dx, G.rxstepy * G.dy, G.rxstepz * G.dz))


    # Excitation file for user-defined source waveforms
    cmd = '#excitation_file'
    if singlecmds[cmd] != 'None':
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            raise CmdInputError(cmd + ' requires exactly one parameter')
        excitationfile = tmp[0]

        # See if file exists at specified path and if not try input file directory
        if not os.path.isfile(excitationfile):
            excitationfile = os.path.join(G.inputdirectory, excitationfile)

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
                w.uservalues = waveformvalues[:,waveform]
        
            if G.messages:
                print('User waveform {} created.'.format(w.ID))

            G.waveforms.append(w)




