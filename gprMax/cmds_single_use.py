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

import inspect
import os
import sys
import decimal as d

import gprMax.config as config
from .config import c
from .config import dtypes
from .config import hostinfo
from .exceptions import CmdInputError
from .waveforms import Waveform
from .utilities import round_value

import numpy as np
from scipy import interpolate
from colorama import init
from colorama import Fore
from colorama import Style
init()

floattype = dtypes['float_or_double']


class Properties:
    pass


class UserObjectSingle:

    def __init__(self, **kwargs):
        # each single command has an order to specify the order in which
        # the commands are constructed. IE. discretisation must be
        # created before the domain
        self.order = None
        self.kwargs = kwargs
        self.props = Properties()

        for k, v in kwargs.items():
            setattr(self.props, k, v)

    def __str__(self):
        pass

    def create(self, grid, uip):
        pass


class Domain(UserObjectSingle):
    """Allows you to specify the size of the model.

    :param p1: point specifying total extend in x, y, z
    :type p1: list of floats, non-optional
    """

    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 2

    def __str__(self):
        try:
            s = '#domain: {} {} {}'.format(self.kwargs['p1'][0],
                                           self.kwargs['p1'][1],
                                           self.kwargs['p1'][2])
        except KeyError:
            print('error message')

        return s

    def create(self, G, uip):

        # code to create the gprMax domain as per input_cmds_singleuse.py
        try:
            G.nx, G.ny, G.nz = uip.discretise_point(self.kwargs['p1'])
        except KeyError:
            raise CmdInputError(self.__str__ + ' Please specify a point')

        if G.nx == 0 or G.ny == 0 or G.nz == 0:
            raise CmdInputError(self.__str__ + ' requires at least one cell in every dimension')
        if config.is_messages():
            print('Domain size: {:g} x {:g} x {:g}m ({:d} x {:d} x {:d} = {:g} cells)'.format(self.kwargs['p1'][0], self.kwargs['p1'][1], self.kwargs['p1'][2], G.nx, G.ny, G.nz, (G.nx * G.ny * G.nz)))

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

        # Round down time step to nearest float with precision one less than hardware maximum. Avoids inadvertently exceeding the CFL due to binary representation of floating point number.
        # Round down time step to nearest float with precision one less than hardware maximum.
        # Avoids inadvertently exceeding the CFL due to binary representation of floating point number.
        G.dt = round_value(G.dt, decimalplaces=d.getcontext().prec - 1)

        if config.is_messages():
            print('Mode: {}'.format(G.mode))
            print('Time step (at CFL limit): {:g} secs'.format(G.dt))

        # Number of threads (OpenMP) to use
        if sys.platform == 'darwin':
            os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'  # Should waiting threads consume CPU power (can drastically effect performance)
        os.environ['OMP_DYNAMIC'] = 'FALSE'  # Number of threads may be adjusted by the run time environment to best utilize system resources
        os.environ['OMP_PLACES'] = 'cores'  # Each place corresponds to a single core (having one or more hardware threads)
        os.environ['OMP_PROC_BIND'] = 'TRUE'  # Bind threads to physical cores
        # os.environ['OMP_DISPLAY_ENV'] = 'TRUE' # Prints OMP version and environment variables (useful for debug)

        # Catch bug with Windows Subsystem for Linux (https://github.com/Microsoft/BashOnWindows/issues/785)
        if 'Microsoft' in hostinfo['osversion']:
            os.environ['KMP_AFFINITY'] = 'disabled'
            del os.environ['OMP_PLACES']
            del os.environ['OMP_PROC_BIND']

        if os.environ.get('OMP_NUM_THREADS'):
            G.nthreads = int(os.environ.get('OMP_NUM_THREADS'))
        else:
            # Set number of threads to number of physical CPU cores
            G.nthreads = hostinfo['physicalcores']
            os.environ['OMP_NUM_THREADS'] = str(G.nthreads)

        if config.is_messages():
            print('Number of CPU (OpenMP) threads: {}'.format(G.nthreads))
        if G.nthreads > hostinfo['physicalcores']:
            print(Fore.RED + 'WARNING: You have specified more threads ({}) than available physical CPU cores ({}). This may lead to degraded performance.'.format(G.nthreads, hostinfo['physicalcores']) + Style.RESET_ALL)


class Discretisation(UserObjectSingle):
    """Allows you to specify the discretization of space in the x , y and z directions respectively

    :param p1: Specify discretisation in x, y, z direction
    :type p1: list of floats, non-optional
    """
    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 1

    def __str__(self):
        try:
            s = '#dx_dy_dz: {} {} {}'.format(self.kwargs['p1'][0],
                                             self.kwargs['p1'][1],
                                             self.kwargs['p1'][2])
        except KeyError:
            print('error message')

        return s

    def create(self, G, uip):

        try:
            G.dl = np.array(self.kwargs['p1'])
            G.dx, G.dy, G.dz = self.kwargs['p1']

        except KeyError:
            raise CmdInputError('Discretisation requires a point')

        if G.dl[0] <= 0:
            raise CmdInputError('Discretisation requires the x-direction spatial step to be greater than zero')
        if G.dl[1] <= 0:
            raise CmdInputError(' Discretisation requires the y-direction spatial step to be greater than zero')
        if G.dl[2] <= 0:
            raise CmdInputError('Discretisation requires the z-direction spatial step to be greater than zero')

        if config.is_messages():
            print('Spatial discretisation: {:g} x {:g} x {:g}m'.format(*G.dl))


class TimeWindow(UserObjectSingle):
    """Allows you to specify the total required simulated time

    :param time: Required simulated time in seconds
    :type time: float, optional
    :param iterations: Required number of iterations
    :type iterations: int, optional
    """
    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 4

    def __str__(self):
        try:
            s = '#time_window: {}'.format(self.kwargs['time'])
        except KeyError:
            try:
                s = '#time_window: {}'.format(self.kwargs['iterations'])
            except KeyError:
                print('time window error')

        return s

    def create(self, G, uip):
        # If number of iterations given
        # The +/- 1 used in calculating the number of iterations is to account for
        # the fact that the solver (iterations) loop runs from 0 to < G.iterations
        try:
            iterations = int(self.kwargs['iterations'])
            G.timewindow = (iterations - 1) * G.dt
            G.iterations = iterations

        except KeyError:
            pass

        try:
            tmp = float(self.kwargs['time'])
            if tmp > 0:
                G.timewindow = tmp
                G.iterations = int(np.ceil(tmp / G.dt)) + 1
            else:
                raise CmdInputError(self.__str__() + ' must have a value greater than zero')

        except KeyError:
            pass

        if not G.timewindow:
            raise CmdInputError('TimeWindow: Specify a time or number of iterations')

        if config.is_messages():
            print('Time window: {:g} secs ({} iterations)'.format(G.timewindow, G.iterations))


class Messages(UserObjectSingle):
    """Allows you to control the amount of information displayed on screen when gprMax is run

    :param yn: Whether information should be displayed.
    :type yn: bool, optional
    """

    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 0

    def __str__(self):
        try:
            s = '#messages: {}'.format(self.kwargs['yn'])
        except KeyError:
            print('messages problem')

    def create(self, G, uip):
        try:
            yn = self.kwargs['yn']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires exactly one parameter')

        if yn.lower() == 'y':
            config.general['messages'] = True
        elif yn.lower() == 'n':
            config.general['messages'] = False
        else:
            raise CmdInputError(self.__str__() + ' requires input values of either y or n')


class Title(UserObjectSingle):
    """Allows you to include a title for your model.

    :param name: Simulation title.
    :type name: str, optional
    """

    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 5

    def create(self, G, uip):
        # Title
        try:
            title = self.kwargs['name']
            G.title = title
        except KeyError:
            pass

        if config.is_messages():
            print('Model title: {}'.format(G.title))

class NumThreads(UserObjectSingle):
    """Allows you to control how many OpenMP threads (usually the number of
    physical CPU cores available) are used when running the model.

    :param n: Number of threads.
    :type n: int, optional
    """
    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 6

    def __str__(self):
        try:
            return '#n_threads: {}'.format(self.kwargs['n'])
        except KeyError:
            return '#n_threads:'

    def create(self, G, uip):
        # Get information about host machine

        try:
            n = self.kwargs['n']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires exactly one parameter to specify the number of threads to use')

        if n < 1:
            raise CmdInputError(self.__str__() + ' requires the value to be an integer not less than one')

        G.nthreads = n
        os.environ['OMP_NUM_THREADS'] = str(G.nthreads)

        if config.is_messages():
            print('Number of CPU (OpenMP) threads: {}'.format(G.nthreads))
        if G.nthreads > hostinfo['physicalcores']:
            print(Fore.RED + 'WARNING: You have specified more threads ({}) than available physical CPU cores ({}). This may lead to degraded performance.'.format(G.nthreads, hostinfo['physicalcores']) + Style.RESET_ALL)

        # Print information about any GPU in use
        if config.is_messages():
            if G.gpu is not None:
                print('GPU solving using: {} - {}'.format(G.gpu.deviceID, G.gpu.name))


# Time step stability factor
class TimeStepStabilityFactor(UserObjectSingle):
    """Factor by which to reduce the time step from the CFL limit.

    :param f: Factor to multiple time step.
    :type f: float, optional
    """
    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 7

    def __str__(self):

        try:
            return '#time_step_stability_factor: {}'.format(self.kwargs['f'])
        except KeyError:
            return '#time_step_stability_factor:'

    def create(self, G, uip):

        try:
            f = self.kwargs['f']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires exactly one parameter')

        if f <= 0 or f > 1:
            raise CmdInputError(self.__str__() + ' requires the value of the time step stability factor to be between zero and one')
        G.dt = G.dt * f
        if config.is_messages():
            print('Time step (modified): {:g} secs'.format(G.dt))


class PMLCells(UserObjectSingle):
    """Allows you to control the number of cells (thickness) of PML that are used
    on the six sides of the model domain. Specify either single thickness or
    thickness on each side.

    :param thickness: Thickness of PML on all 6 sides.
    :type thickness: int, optional
    :param x0: Thickness of PML on left side.
    :type x0: int, optional
    :param y0: Thickness of PML on the front side.
    :type y0: int, optional
    :param z0: Thickness of PML on bottom side.
    :type z0: int, optional
    :param xmax: Thickness of PML on right side.
    :type xmax: int, optional
    :param ymax: Thickness of PML on the back side.
    :type ymax: int, optional
    :param zmax: Thickness of PML on top side.
    :type zmax: int, optional
    """
    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 8

    def create(self, G, uip):

        try:
            thickness = self.kwargs['thickness']

            for key in G.pmlthickness.keys():
                G.pmlthickness[key] = int(thickness)

        except KeyError:
            try:
                G.pmlthickness['x0'] = int(self.kwargs['x0'])
                G.pmlthickness['y0'] = int(self.kwargs['y0'])
                G.pmlthickness['z0'] = int(self.kwargs['z0'])
                G.pmlthickness['xmax'] = int(self.kwargs['xmax'])
                G.pmlthickness['ymax'] = int(self.kwargs['ymax'])
                G.pmlthickness['zmax'] = int(self.kwargs['zmax'])
            except KeyError:
                raise CmdInputError('#pml_cells: requires either one or six parameter(s)')

        if (2 * G.pmlthickness['x0'] >= G.nx or
            2 * G.pmlthickness['y0'] >= G.ny or
            2 * G.pmlthickness['z0'] >= G.nz or
            2 * G.pmlthickness['xmax'] >= G.nx or
            2 * G.pmlthickness['ymax'] >= G.ny or
            2 * G.pmlthickness['zmax'] >= G.nz):
                raise CmdInputError('#pml_thickness: has too many cells for the domain size')


class SrcSteps(UserObjectSingle):
    """Provides a simple method to allow you to move the location of all simple sources

    :param p1: increments (x,y,z) to move all simple sources
    :type p1: list, non-optional
    """

    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 9

    def create(self, G, uip):
        try:
            G.srcsteps = uip.discretise_point(self.kwargs['p1'])
        except KeyError:
            raise CmdInputError('#src_steps: requires exactly three parameters')
        # src_steps
        if config.is_messages():
            print('Simple sources will step {:g}m, {:g}m, {:g}m for each model run.'.format(G.srcsteps[0] * G.dx, G.srcsteps[1] * G.dy, G.srcsteps[2] * G.dz))


class RxSteps(UserObjectSingle):
    """Provides a simple method to allow you to move the location of all simple receivers

    :param p1: increments (x,y,z) to move all simple receivers
    :type p1: list, non-optional
    """

    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 10

    def create(self, G, uip):
        try:
            G.rxsteps = uip.discretise_point(self.kwargs['p1'])
        except KeyError:
            raise CmdInputError('#rx_steps: requires exactly three parameters')
        if config.is_messages():
            print('All receivers will step {:g}m, {:g}m, {:g}m for each model run.'.format(G.rxsteps[0] * G.dx, G.rxsteps[1] * G.dy, G.rxsteps[2] * G.dz))


class ExcitationFile(UserObjectSingle):
    """Allows you to specify an ASCII file that contains columns of amplitude
    values that specify custom waveform shapes that can be used with sources in the model.

    :param filepath: Excitation file path.
    :type filepath: str, non-optional
    :param kind:  passed to the interpolation function (scipy.interpolate.interp1d).
    :type kind: float, optional
    :param fill_value:  passed to the interpolation function (scipy.interpolate.interp1d).
    :type fill_value: float, optional
    """

    def create(self, G, uip):
    # Excitation file for user-defined source waveforms
        try:
            kwargs = dict()
            excitationfile = self.kwargs['filepath']
            kwargs['kind'] = self.kwargs['kind']
            kwargs['fill_value'] = self.kwargs['fill_value']

        except KeyError:
            try:
                excitationfile = self.kwargs['filepath']
                args, varargs, keywords, defaults = inspect.getargspec(interpolate.interp1d)
                kwargs = dict(zip(reversed(args), reversed(defaults)))
            except KeyError:
                raise CmdInputError('#excitation_file: requires either one or three parameter(s)')

            # See if file exists at specified path and if not try input file directory
            if not os.path.isfile(excitationfile):
                excitationfile = os.path.abspath(os.path.join(G.inputdirectory, excitationfile))

            if config.is_messages():
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
                    singlewaveformvalues = np.lib.pad(singlewaveformvalues, (0, len(singlewaveformvalues) - len(waveformvalues)), 'constant', constant_values=0)

                # Interpolate waveform values
                w.userfunc = interpolate.interp1d(waveformtime, singlewaveformvalues, **kwargs)

                if config.is_messages():
                    print('User waveform {} created using {} and, if required, interpolation parameters (kind: {}, fill value: {}).'.format(w.ID, timestr, kwargs['kind'], kwargs['fill_value']))

                G.waveforms.append(w)


class OutputDir(UserObjectSingle):
    """Allows you to control the directory where output file(s) will be stored.

    :param dir: File path to directory.
    :type dir: str, non-optional
    """
    def __init__(self, **kwargs):
        # dont need to define parameters in advance. Just catch errors
        # when they occur
        super().__init__(**kwargs)
        self.order = 11

    def create(self, grid, uip):
        grid.outputdirectory = self.kwargs['dir']


class NumberOfModelRuns(UserObjectSingle):
    """Number of times to run the simulation. This required when using multiple
    class:Scene instances.

    :param n: File path to directory.
    :type n: str, non-optional
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 12

    def create(self, grid, uip):
        try:
            grid.numberofmodelruns = self.kwargs['n']
        except KeyError:
            raise CmdInputError('#numberofmodelruns: requires exactly one parameter')
