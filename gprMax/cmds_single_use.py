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
import logging
from pathlib import Path

from colorama import init
from colorama import Fore
from colorama import Style
init()
import numpy as np
from scipy.constants import c
from scipy import interpolate

import gprMax.config as config
from .exceptions import CmdInputError
from .waveforms import Waveform
from .utilities import round_value

log = logging.getLogger(__name__)


class Properties:
    pass


class UserObjectSingle:
    """Object that can only occur a single time in a model."""

    def __init__(self, **kwargs):
        # Each single command has an order to specify the order in which
        # the commands are constructed, e.g. discretisation must be
        # created before the domain
        self.order = None
        self.kwargs = kwargs
        self.props = Properties()
        self.autotranslate = True

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
        super().__init__(**kwargs)
        self.order = 2

    def __str__(self):
        try:
            s = f"#domain: {self.kwargs['p1'][0]} {self.kwargs['p1'][1]} {self.kwargs['p1'][2]}"
        except KeyError:
            log.exception('error message')

        return s

    def create(self, G, uip):
        try:
            G.nx, G.ny, G.nz = uip.discretise_point(self.kwargs['p1'])
        except KeyError:
            raise CmdInputError(f"'{self.params_str()}' please specify a point")

        if G.nx == 0 or G.ny == 0 or G.nz == 0:
            raise CmdInputError(f"'{self.params_str()}' requires at least one cell in every dimension")

        log.info(f"Domain size: {self.kwargs['p1'][0]:g} x {self.kwargs['p1'][1]:g} x {self.kwargs['p1'][2]:g}m ({G.nx:d} x {G.ny:d} x {G.nz:d} = {(G.nx * G.ny * G.nz):g} cells)")

        # Calculate time step at CFL limit; switch off appropriate PMLs for 2D
        if G.nx == 1:
            G.calculate_dt()
            G.mode = '2D TMx'
            G.pmlthickness['x0'] = 0
            G.pmlthickness['xmax'] = 0
        elif G.ny == 1:
            G.calculate_dt()
            G.mode = '2D TMy'
            G.pmlthickness['y0'] = 0
            G.pmlthickness['ymax'] = 0
        elif G.nz == 1:
            G.calculate_dt()
            G.mode = '2D TMz'
            G.pmlthickness['z0'] = 0
            G.pmlthickness['zmax'] = 0
        else:
            G.calculate_dt()
            G.mode = '3D'

        log.info(f'Mode: {G.mode}')
        log.info(f'Time step (at CFL limit): {G.dt:g} secs')


class Discretisation(UserObjectSingle):
    """Allows you to specify the discretization of space in the x, y, and z
        directions respectively.

    :param p1: Specify discretisation in x, y, z direction
    :type p1: list of floats, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 1

    def __str__(self):
        try:
            s = f"#dx_dy_dz: {self.kwargs['p1'][0]} {self.kwargs['p1'][1]} {self.kwargs['p1'][2]}"
        except KeyError:
            log.exception('error message')

        return s

    def create(self, G, uip):
        try:
            G.dl = np.array(self.kwargs['p1'])
            G.dx, G.dy, G.dz = self.kwargs['p1']
        except KeyError:
            raise CmdInputError(f"'{self.params_str()}' discretisation requires a point")

        if G.dl[0] <= 0:
            raise CmdInputError(f"'{self.params_str()}' discretisation requires the x-direction spatial step to be greater than zero")
        if G.dl[1] <= 0:
            raise CmdInputError(f"'{self.params_str()}' discretisation requires the y-direction spatial step to be greater than zero")
        if G.dl[2] <= 0:
            raise CmdInputError(f"'{self.params_str()}' discretisation requires the z-direction spatial step to be greater than zero")

        log.info(f'Spatial discretisation: {G.dl[0]:g} x {G.dl[1]:g} x {G.dl[2]:g}m')


class TimeWindow(UserObjectSingle):
    """Allows you to specify the total required simulated time

    :param time: Required simulated time in seconds
    :type time: float, optional
    :param iterations: Required number of iterations
    :type iterations: int, optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 4

    def __str__(self):
        try:
            s = f"#time_window: {self.kwargs['time']}"
        except KeyError:
            try:
                s = f"#time_window: {self.kwargs['iterations']}"
            except KeyError:
                log.exception('time window error')

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
                raise CmdInputError(f"'{self.params_str()}' must have a value greater than zero")
        except KeyError:
            pass

        if not G.timewindow:
            raise CmdInputError(f"'{self.params_str()}' specify a time or number of iterations")

        log.info(f'Time window: {G.timewindow:g} secs ({G.iterations} iterations)')


class Messages(UserObjectSingle):
    """Allows you to control the amount of information displayed on screen
        when gprMax is run

    :param yn: Whether information should be displayed.
    :type yn: bool, optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 0

    def __str__(self):
        try:
            s = '#messages: {}'.format(self.kwargs['yn'])
        except KeyError:
            log.exception('messages problem')

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
        super().__init__(**kwargs)
        self.order = 5

    def create(self, G, uip):
        # Title
        try:
            title = self.kwargs['name']
            G.title = title
        except KeyError:
            pass

        log.info(f'Model title: {G.title}')


class NumThreads(UserObjectSingle):
    """Allows you to control how many OpenMP threads (usually the number of
        physical CPU cores available) are used when running the model.

    :param n: Number of threads.
    :type n: int, optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 6

    def __str__(self):
        try:
            return '#n_threads: {}'.format(self.kwargs['n'])
        except KeyError:
            return '#n_threads:'

    def create(self, G, uip):
        try:
            n = self.kwargs['n']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires exactly one parameter to specify the number of threads to use')
        if n < 1:
            raise CmdInputError(self.__str__() + ' requires the value to be an integer not less than one')

        config.model_configs[G.model_num].ompthreads = set_omp_threads(n)


class TimeStepStabilityFactor(UserObjectSingle):
    """Factor by which to reduce the time step from the CFL limit.

    :param f: Factor to multiple time step.
    :type f: float, optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 7

    def __str__(self):
        try:
            return f"#time_step_stability_factor: {self.kwargs['f']}"
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
        log.info(f'Time step (modified): {G.dt:g} secs')


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
    """Provides a simple method to allow you to move the location of all simple
        sources.

    :param p1: increments (x,y,z) to move all simple sources
    :type p1: list, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 9

    def create(self, G, uip):
        try:
            G.srcsteps = uip.discretise_point(self.kwargs['p1'])
        except KeyError:
            raise CmdInputError('#src_steps: requires exactly three parameters')

        log.info(f'Simple sources will step {G.srcsteps[0] * G.dx:g}m, \
                 {G.srcsteps[1] * G.dy:g}m, {G.srcsteps[2] * G.dz:g}m \
                 for each model run.')


class RxSteps(UserObjectSingle):
    """Provides a simple method to allow you to move the location of all simple
        receivers.

    :param p1: increments (x,y,z) to move all simple receivers
    :type p1: list, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 10

    def create(self, G, uip):
        try:
            G.rxsteps = uip.discretise_point(self.kwargs['p1'])
        except KeyError:
            raise CmdInputError('#rx_steps: requires exactly three parameters')

        log.info(f'All receivers will step {G.rxsteps[0] * G.dx:g}m, \
                 {G.rxsteps[1] * G.dy:g}m, {G.rxsteps[2] * G.dz:g}m \
                 for each model run.')


class ExcitationFile(UserObjectSingle):
    """Allows you to specify an ASCII file that contains columns of amplitude
        values that specify custom waveform shapes that can be used with sources
        in the model.

    :param filepath: Excitation file path.
    :type filepath: str, non-optional
    :param kind:  passed to the interpolation function (scipy.interpolate.interp1d).
    :type kind: float, optional
    :param fill_value:  passed to the interpolation function (scipy.interpolate.interp1d).
    :type fill_value: float, optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 11

    def create(self, G, uip):
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
        excitationfile = Path(excitationfile)
        if not excitationfile.exists():
            excitationfile = Path(config.sim_config.input_file_path.parent, excitationfile)

        log.info(f'Excitation file: {excitationfile}')

        # Get waveform names
        with open(excitationfile, 'r') as f:
            waveformIDs = f.readline().split()

        # Read all waveform values into an array
        waveformvalues = np.loadtxt(excitationfile, skiprows=1, dtype=config.sim_config.dtypes['float_or_double'])

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
                raise CmdInputError(f'Waveform with ID {waveformIDs[waveform]} already exists')
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
                singlewaveformvalues = np.lib.pad(singlewaveformvalues,
                                                  (0, len(singlewaveformvalues) -
                                                  len(waveformvalues)),
                                                  'constant', constant_values=0)

            # Interpolate waveform values
            w.userfunc = interpolate.interp1d(waveformtime, singlewaveformvalues, **kwargs)

            log.info(f"User waveform {w.ID} created using {timestr} and, if required, interpolation parameters (kind: {kwargs['kind']}, fill value: {kwargs['fill_value']}).")

            G.waveforms.append(w)


class OutputDir(UserObjectSingle):
    """Allows you to control the directory where output file(s) will be stored.

    :param dir: File path to directory.
    :type dir: str, non-optional
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 12

    def create(self, grid, uip):
        config.model_configs[grid.model_num].set_output_file_path(self.kwargs['dir'])


class NumberOfModelRuns(UserObjectSingle):
    """Number of times to run the simulation. This required when using multiple
        class:Scene instances.

    :param n: File path to directory.
    :type n: str, non-optional
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 13

    def create(self, grid, uip):
        try:
            grid.numberofmodelruns = self.kwargs['n']
        except KeyError:
            raise CmdInputError('#numberofmodelruns: requires exactly one parameter')
