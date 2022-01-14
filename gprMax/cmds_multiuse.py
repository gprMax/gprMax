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

import inspect
import logging

import numpy as np
from scipy import interpolate

import gprMax.config as config

from .cmds_geometry.cmds_geometry import (UserObjectGeometry,
                                          rotate_2point_object,
                                          rotate_polarisation)
from .geometry_outputs import GeometryObjects as GeometryObjectsUser
from .materials import DispersiveMaterial as DispersiveMaterialUser
from .materials import Material as MaterialUser
from .materials import PeplinskiSoil as PeplinskiSoilUser
from .pml import CFS, CFSParameter
from .receivers import Rx as RxUser
from .snapshots import Snapshot as SnapshotUser
from .sources import HertzianDipole as HertzianDipoleUser
from .sources import MagneticDipole as MagneticDipoleUser
from .sources import TransmissionLine as TransmissionLineUser
from .sources import VoltageSource as VoltageSourceUser
from .subgrids.base import SubGridBase
from .utilities.utilities import round_value
from .waveforms import Waveform as WaveformUser

logger = logging.getLogger(__name__)


class UserObjectMulti:
    """Object that can occur multiple times in a model."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.order = None
        self.hash = '#example'
        self.autotranslate = True
        self.dorotate = False

    def __str__(self):
        """Readable user string as per hash commands."""
        s = ''
        for _, v in self.kwargs.items():
            if isinstance(v, tuple) or isinstance(v, list):
                v = ' '.join([str(el) for el in v])
            s += str(v) + ' '

        return f'{self.hash}: {s[:-1]}'

    def create(self, grid, uip):
        """Creates object and adds it to grid."""
        pass

    def rotate(self, axis, angle, origin=None):
        """Rotates object (specialised for each object)."""
        pass

    def params_str(self):
        """Readable string of parameters given to object."""
        return self.hash + ': ' + str(self.kwargs)

    def grid_name(self, grid):
        """Returns subgrid name for use with logging info. Returns an empty
            string if the grid is the main grid.
        """

        if isinstance(grid, SubGridBase):
            return f'[{grid.name}] '  
        else:
            return ''


class Waveform(UserObjectMulti):
    """Specifies waveforms to use with sources in the model.

    Attributes:
        wave_type: string required to specify waveform type.
        amp: float to scale maximum amplitude of waveform.
        freq: float to specify centre frequency (Hz) of waveform.
        id: string required for identifier of waveform.
        user_values: optional 1D array of amplitude values to use with 
                        user waveform.
        user_time: optional 1D array of time values to use with user waveform.
        kind: optional string or int, see scipy.interpolate.interp1d - https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy-interpolate-interp1d
        fill_value: optional array or 'extrapolate', see scipy.interpolate.interp1d - https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#scipy-interpolate-interp1d
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 1
        self.hash = '#waveform'

    def create(self, grid, uip):
        try:
            wavetype = self.kwargs['wave_type'].lower()
        except KeyError:
            logger.exception(self.params_str() + (f" must have one of the " 
                             f"following types {','.join(WaveformUser.types)}."))
            raise
        if wavetype not in WaveformUser.types:
            logger.exception(self.params_str() + (f" must have one of the "
                             f"following types {','.join(WaveformUser.types)}."))
            raise ValueError

        if wavetype != 'user':
            try:
                amp = self.kwargs['amp']
                freq = self.kwargs['freq']
                ID = self.kwargs['id']
            except KeyError:
                logger.exception(self.params_str() + (' builtin waveforms '
                                 'require exactly four parameters.'))
                raise
            if freq <= 0:
                logger.exception(self.params_str() + (' requires an excitation '
                                 'frequency value of greater than zero.'))
                raise ValueError
            if any(x.ID == ID for x in grid.waveforms):
                logger.exception(self.params_str() + (f' with ID {ID} already '
                                 'exists.'))
                raise ValueError

            w = WaveformUser()
            w.ID = ID
            w.type = wavetype
            w.amp = amp
            w.freq = freq

            logger.info(self.grid_name(grid) + (f'Waveform {w.ID} of type '
                        f'{w.type} with maximum amplitude scaling {w.amp:g}, '
                        f'frequency {w.freq:g}Hz created.'))

        else:
            try:
                uservalues = self.kwargs['user_values']
                ID = self.kwargs['id']
                args, varargs, keywords, defaults = inspect.getargspec(interpolate.interp1d)
                kwargs = dict(zip(reversed(args), reversed(defaults)))
            except KeyError:
                logger.exception(self.params_str() + (' a user-defined '
                                'waveform requires at least two parameters.'))
                raise

            if 'user_time' in self.kwargs:
                waveformtime = self.kwargs['user_time']
            else:
                waveformtime = np.arange(0, grid.timewindow + grid.dt, grid.dt)

            if any(x.ID == ID for x in grid.waveforms):
                logger.exception(self.params_str() + (f' with ID {ID} already '
                                 'exists.'))
                raise ValueError

            w = WaveformUser()
            w.ID = ID
            w.type = wavetype
            w.userfunc = interpolate.interp1d(waveformtime, uservalues, **kwargs)

            logger.info(self.grid_name(grid) + (f'Waveform {w.ID} that is'
                                                'user-defined created.'))

        grid.waveforms.append(w)


class VoltageSource(UserObjectMulti):
    """Specifies a voltage source at an electric field location.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        resistance: float required for internal resistance (Ohms) of 
                        voltage source.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 2
        self.hash = '#voltage_source'

    def rotate(self, axis, angle, origin=None):
        """Sets parameters for rotation."""
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.dorotate = True

    def __dorotate(self, grid):
        """Performs rotation."""
        rot_pol_pts, self.kwargs['polarisation'] = rotate_polarisation(self.kwargs['p1'], 
                                                                       self.kwargs['polarisation'], 
                                                                       self.axis, 
                                                                       self.angle, 
                                                                       grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, 
                                       self.origin)
        self.kwargs['p1'] = tuple(rot_pts[0, :])

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            polarisation = self.kwargs['polarisation'].lower()
            resistance = self.kwargs['resistance']
            waveform_id = self.kwargs['waveform_id']
        except KeyError:
            logger.exception(self.params_str() + (' requires at least six '
                                                  'parameters.'))
            raise

        if self.dorotate:
            self.__dorotate(grid)

        # Check polarity & position parameters
        if polarisation not in ('x', 'y', 'z'):
            logger.exception(self.params_str() + (' polarisation must be '
                                                  'x, y, or z.'))
            raise ValueError
        if '2D TMx' in (config.get_model_config().mode and 
                        (polarisation == 'y' or polarisation == 'z')):
            logger.exception(self.params_str() + (' polarisation must be x in '
                                                  '2D TMx mode.'))
            raise ValueError
        elif '2D TMy' in (config.get_model_config().mode and 
                          (polarisation == 'x' or polarisation == 'z')):
            logger.exception(self.params_str() + (' polarisation must be y in '
                                                  '2D TMy mode.'))
            raise ValueError
        elif '2D TMz' in (config.get_model_config().mode and 
                          (polarisation == 'x' or polarisation == 'y')):
            logger.exception(self.params_str() + (' polarisation must be z in '
                                                  '2D TMz mode.'))
            raise ValueError

        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.params_str())
        p2 = uip.round_to_grid_static_point(p1)

        if resistance < 0:
            logger.exception(self.params_str() + (' requires a source '
                                                  'resistance of zero '
                                                  'or greater.'))
            raise ValueError

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(self.params_str() + (' there is no waveform with '
                                                  'the identifier '
                                                  f'{waveform_id}.'))
            raise ValueError

        v = VoltageSourceUser()
        v.polarisation = polarisation
        v.xcoord = xcoord
        v.ycoord = ycoord
        v.zcoord = zcoord
        v.ID = (v.__class__.__name__ + '(' + str(v.xcoord) + ',' + 
                str(v.ycoord) + ',' + str(v.zcoord) + ')')
        v.resistance = resistance
        v.waveformID = waveform_id

        try:
            start = self.kwargs['start']
            stop = self.kwargs['stop']
            # Check source start & source remove time parameters
            if start < 0:
                logger.exception(self.params_str() + (' delay of the initiation '
                                                      'of the source should not '
                                                      'be less than zero.'))
                raise ValueError
            if stop < 0:
                logger.exception(self.params_str() + (' time to remove the '
                                                      'source should not be '
                                                      'less than zero.'))
                raise ValueError
            if stop - start <= 0:
                logger.exception(self.params_str() + (' duration of the source '
                                                      'should not be zero or '
                                                      'less.'))
                raise ValueError
            v.start = start
            if stop > grid.timewindow:
                v.stop = grid.timewindow
            else:
                v.stop = stop
            startstop = (f' start time {v.start:g} secs, finish time '
                         f'{v.stop:g} secs ')
        except KeyError:
            v.start = 0
            v.stop = grid.timewindow
            startstop = ' '

        v.calculate_waveform_values(grid)

        logger.info(self.grid_name(grid) + f'Voltage source with polarity '
                    f'{v.polarisation} at {p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m, '
                    f'resistance {v.resistance:.1f} Ohms,' + startstop + 
                    f'using waveform {v.waveformID} created.')

        grid.voltagesources.append(v)


class HertzianDipole(UserObjectMulti):
    """Specifies a current density term at an electric field location.

    The simplest excitation, often referred to as an additive or soft source.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 3
        self.hash = '#hertzian_dipole'

    def rotate(self, axis, angle, origin=None):
        """Sets parameters for rotation."""
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.dorotate = True

    def __dorotate(self, grid):
        """Performs rotation."""
        rot_pol_pts, self.kwargs['polarisation'] = rotate_polarisation(self.kwargs['p1'], 
                                                                       self.kwargs['polarisation'], 
                                                                       self.axis, 
                                                                       self.angle, 
                                                                       grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.kwargs['p1'] = tuple(rot_pts[0, :])

    def create(self, grid, uip):
        try:
            polarisation = self.kwargs['polarisation'].lower()
            p1 = self.kwargs['p1']
            waveform_id = self.kwargs['waveform_id']
        except KeyError:
            logger.exception(self.params_str() + ' requires at least 3 ' + 
                             'parameters.')
            raise

        if self.dorotate:
            self.__dorotate(grid)

        # Check polarity & position parameters
        if polarisation not in ('x', 'y', 'z'):
            logger.exception(self.params_str() + (' polarisation must be '
                                                  'x, y, or z.'))
            raise ValueError
        if '2D TMx' in (config.get_model_config().mode and 
                        (polarisation == 'y' or polarisation == 'z')):
            logger.exception(self.params_str() + (' polarisation must be x in '
                                                  '2D TMx mode.'))
            raise ValueError
        elif '2D TMy' in (config.get_model_config().mode and 
                          (polarisation == 'x' or polarisation == 'z')):
            logger.exception(self.params_str() + (' polarisation must be y in '
                                                  '2D TMy mode.'))
            raise ValueError
        elif '2D TMz' in (config.get_model_config().mode and 
                          (polarisation == 'x' or polarisation == 'y')):
            logger.exception(self.params_str() + (' polarisation must be z in '
                                                  '2D TMz mode.'))
            raise ValueError

        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.params_str())
        p2 = uip.round_to_grid_static_point(p1)


        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(self.params_str() + ' there is no waveform ' + 
                             f'with the identifier {waveform_id}.')
            raise ValueError

        h = HertzianDipoleUser()
        h.polarisation = polarisation

        # Set length of dipole to grid size in polarisation direction
        if h.polarisation == 'x':
            h.dl = grid.dx
        elif h.polarisation == 'y':
            h.dl = grid.dy
        elif h.polarisation == 'z':
            h.dl = grid.dz

        h.xcoord = xcoord
        h.ycoord = ycoord
        h.zcoord = zcoord
        h.xcoordorigin = xcoord
        h.ycoordorigin = ycoord
        h.zcoordorigin = zcoord
        h.ID = (h.__class__.__name__ + '(' + str(h.xcoord) + ',' + 
                str(h.ycoord) + ',' + str(h.zcoord) + ')')
        h.waveformID = waveform_id

        try:
            # Check source start & source remove time parameters
            start = self.kwargs['start']
            stop = self.kwargs['stop']
            if start < 0:
                logger.exception(self.params_str() + (' delay of the initiation '
                                                      'of the source should not '
                                                      'be less than zero.'))
                raise ValueError
            if stop < 0:
                logger.exception(self.params_str() + (' time to remove the '
                                                      'source should not be '
                                                      'less than zero.'))
                raise ValueError
            if stop - start <= 0:
                logger.exception(self.params_str() + (' duration of the source '
                                                      'should not be zero or '
                                                      'less.'))
                raise ValueError
            h.start = start
            if stop > grid.timewindow:
                h.stop = grid.timewindow
            else:
                h.stop = stop
            startstop = (f' start time {h.start:g} secs, finish time '
                         f'{h.stop:g} secs ')
        except KeyError:
            h.start = 0
            h.stop = grid.timewindow
            startstop = ' '

        h.calculate_waveform_values(grid)

        if config.get_model_config().mode == '2D':
            logger.info(self.grid_name(grid) + f'Hertzian dipole is a line ' +
                        f'source in 2D with polarity {h.polarisation} at ' + 
                        f'{p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m,' + startstop + 
                        f'using waveform {h.waveformID} created.')
        else:
            logger.info(self.grid_name(grid) + f'Hertzian dipole with ' + 
                        f'polarity {h.polarisation} at {p2[0]:g}m, ' + 
                        f'{p2[1]:g}m, {p2[2]:g}m,' + startstop + 
                        f'using waveform {h.waveformID} created.')

        grid.hertziandipoles.append(h)


class MagneticDipole(UserObjectMulti):
    """Simulates an infinitesimal magnetic dipole. 
    
    Often referred to as an additive or soft source.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 4
        self.hash = '#magnetic_dipole'

    def rotate(self, axis, angle, origin=None):
        """Sets parameters for rotation."""
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.dorotate = True

    def __dorotate(self, grid):
        """Performs rotation."""
        rot_pol_pts, self.kwargs['polarisation'] = rotate_polarisation(self.kwargs['p1'], 
                                                                       self.kwargs['polarisation'], 
                                                                       self.axis, 
                                                                       self.angle, 
                                                                       grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.kwargs['p1'] = tuple(rot_pts[0, :])

    def create(self, grid, uip):
        try:
            polarisation = self.kwargs['polarisation'].lower()
            p1 = self.kwargs['p1']
            waveform_id = self.kwargs['waveform_id']
        except KeyError:
            logger.exception(self.params_str() + ' requires at least five ' 
                             'parameters.')
            raise

        if self.dorotate:
            self.__dorotate(grid)

        # Check polarity & position parameters
        if polarisation not in ('x', 'y', 'z'):
            logger.exception(self.params_str() + (' polarisation must be '
                                                  'x, y, or z.'))
            raise ValueError
        if '2D TMx' in (config.get_model_config().mode and 
                        (polarisation == 'y' or polarisation == 'z')):
            logger.exception(self.params_str() + (' polarisation must be x in '
                                                  '2D TMx mode.'))
            raise ValueError
        elif '2D TMy' in (config.get_model_config().mode and 
                          (polarisation == 'x' or polarisation == 'z')):
            logger.exception(self.params_str() + (' polarisation must be y in '
                                                  '2D TMy mode.'))
            raise ValueError
        elif '2D TMz' in (config.get_model_config().mode and 
                          (polarisation == 'x' or polarisation == 'y')):
            logger.exception(self.params_str() + (' polarisation must be z in '
                                                  '2D TMz mode.'))
            raise ValueError

        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.params_str())
        p2 = uip.round_to_grid_static_point(p1)


        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(self.params_str() + ' there is no waveform ' + 
                             f'with the identifier {waveform_id}.')
            raise ValueError

        m = MagneticDipoleUser()
        m.polarisation = polarisation
        m.xcoord = xcoord
        m.ycoord = ycoord
        m.zcoord = zcoord
        m.xcoordorigin = xcoord
        m.ycoordorigin = ycoord
        m.zcoordorigin = zcoord
        m.ID = (m.__class__.__name__ + '(' + str(m.xcoord) + ',' + 
                str(m.ycoord) + ',' + str(m.zcoord) + ')')
        m.waveformID = waveform_id

        try:
            # Check source start & source remove time parameters
            start = self.kwargs['start']
            stop = self.kwargs['stop']
            if start < 0:
                logger.exception(self.params_str() + (' delay of the initiation '
                                                      'of the source should not '
                                                      'be less than zero.'))
                raise ValueError
            if stop < 0:
                logger.exception(self.params_str() + (' time to remove the '
                                                      'source should not be '
                                                      'less than zero.'))
                raise ValueError
            if stop - start <= 0:
                logger.exception(self.params_str() + (' duration of the source '
                                                      'should not be zero or '
                                                      'less.'))
                raise ValueError
            m.start = start
            if stop > grid.timewindow:
                m.stop = grid.timewindow
            else:
                m.stop = stop
            startstop = (f' start time {m.start:g} secs, '
                         f'finish time {m.stop:g} secs ')
        except KeyError:
            m.start = 0
            m.stop = grid.timewindow
            startstop = ' '

        m.calculate_waveform_values(grid)

        logger.info(self.grid_name(grid) + f'Magnetic dipole with polarity ' +
                    f'{m.polarisation} at {p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m,' + 
                    startstop + f'using waveform {m.waveformID} created.')

        grid.magneticdipoles.append(m)


class TransmissionLine(UserObjectMulti):
    """Specifies a one-dimensional transmission line model at an electric 
        field location.

    Attributes:
        polarisation: string required for polarisation of the source x, y, z.
        p1: tuple required for position of source x, y, z.
        resistance: float required for internal resistance (Ohms) of 
                        voltage source.
        waveform_id: string required for identifier of waveform used with source.
        start: float optional to delay start time (secs) of source.
        stop: float optional to time (secs) to remove source.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 5
        self.hash = '#transmission_line'

    def rotate(self, axis, angle, origin=None):
        """Sets parameters for rotation."""
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.dorotate = True

    def __dorotate(self, grid):
        """Performs rotation."""
        rot_pol_pts, self.kwargs['polarisation'] = rotate_polarisation(self.kwargs['p1'], 
                                                                       self.kwargs['polarisation'], 
                                                                       self.axis, 
                                                                       self.angle, 
                                                                       grid)
        rot_pts = rotate_2point_object(rot_pol_pts, self.axis, self.angle, self.origin)
        self.kwargs['p1'] = tuple(rot_pts[0, :])

    def create(self, grid, uip):
        try:
            polarisation = self.kwargs['polarisation'].lower()
            p1 = self.kwargs['p1']
            waveform_id = self.kwargs['waveform_id']
            resistance = self.kwargs['resistance']
        except KeyError:
            logger.exception(self.params_str() + ' requires at least six '
                             'parameters.')
            raise

        if self.dorotate:
            self.__dorotate(grid)

        # Warn about using a transmission line on GPU
        if config.sim_config.general['cuda']:
            logger.exception(self.params_str() + ' cannot currently be used ' +
                             'with GPU solving. Consider using a ' + 
                             '#voltage_source instead.')
            raise ValueError

        # Check polarity & position parameters
        if polarisation not in ('x', 'y', 'z'):
            logger.exception(self.params_str() + (' polarisation must be '
                                                  'x, y, or z.'))
            raise ValueError
        if '2D TMx' in (config.get_model_config().mode and 
                        (polarisation == 'y' or polarisation == 'z')):
            logger.exception(self.params_str() + (' polarisation must be x in '
                                                  '2D TMx mode.'))
            raise ValueError
        elif '2D TMy' in (config.get_model_config().mode and 
                          (polarisation == 'x' or polarisation == 'z')):
            logger.exception(self.params_str() + (' polarisation must be y in '
                                                  '2D TMy mode.'))
            raise ValueError
        elif '2D TMz' in (config.get_model_config().mode and 
                          (polarisation == 'x' or polarisation == 'y')):
            logger.exception(self.params_str() + (' polarisation must be z in '
                                                  '2D TMz mode.'))
            raise ValueError

        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.params_str())
        p2 = uip.round_to_grid_static_point(p1)


        if resistance <= 0 or resistance >= config.sim_config.em_consts['z0']:
            logger.exception(self.params_str() + ' requires a resistance ' + 
                             'greater than zero and less than the impedance ' +
                             'of free space, i.e. 376.73 Ohms.')
            raise ValueError

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            logger.exception(self.params_str() + ' there is no waveform ' + 
                             f'with the identifier {waveform_id}.')
            raise ValueError

        t = TransmissionLineUser(grid)
        t.polarisation = polarisation
        t.xcoord = xcoord
        t.ycoord = ycoord
        t.zcoord = zcoord
        t.ID = (t.__class__.__name__ + '(' + str(t.xcoord) + ',' + 
                str(t.ycoord) + ',' + str(t.zcoord) + ')')
        t.resistance = resistance
        t.waveformID = waveform_id

        try:
            # Check source start & source remove time parameters
            start = self.kwargs['start']
            stop = self.kwargs['stop']
            if start < 0:
                logger.exception(self.params_str() + (' delay of the initiation '
                                                      'of the source should not '
                                                      'be less than zero.'))
                raise ValueError
            if stop < 0:
                logger.exception(self.params_str() + (' time to remove the '
                                                      'source should not be '
                                                      'less than zero.'))
                raise ValueError
            if stop - start <= 0:
                logger.exception(self.params_str() + (' duration of the source '
                                                      'should not be zero or '
                                                      'less.'))
                raise ValueError
            t.start = start
            if stop > grid.timewindow:
                t.stop = grid.timewindow
            else:
                t.stop = stop
            startstop = (f' start time {t.start:g} secs, finish time '
                         f'{t.stop:g} secs ')
        except KeyError:
            t.start = 0
            t.stop = grid.timewindow
            startstop = ' '

        t.calculate_waveform_values(grid)
        t.calculate_incident_V_I(grid)

        logger.info(self.grid_name(grid) + f'Transmission line with polarity ' +
                    f'{t.polarisation} at {p2[0]:g}m, {p2[1]:g}m, ' + 
                    f'{p2[2]:g}m, resistance {t.resistance:.1f} Ohms,' + 
                    startstop + f'using waveform {t.waveformID} created.')

        grid.transmissionlines.append(t)


class Rx(UserObjectMulti):
    """Specifies output points in the model. 
    
    These are locations where the values of the electric and magnetic field 
    components over the numberof iterations of the model will be saved to file.

    Attributes:
        p1: tuple required for position of receiver x, y, z.
        id: optional string used as identifier for receiver.
        outputs: optional list of outputs for receiver. It can be any
                    selection from Ex, Ey, Ez, Hx, Hy, Hz, Ix, Iy, or Iz.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 6
        self.hash = '#rx'
        self.constructor = RxUser

    def rotate(self, axis, angle, origin=None):
        """Sets parameters for rotation."""
        self.axis = axis
        self.angle = angle
        self.origin = origin
        self.dorotate = True

    def __dorotate(self, G):
        """Performs rotation."""
        new_pt = (self.kwargs['p1'][0] + G.dx, self.kwargs['p1'][1] + G.dy, 
                  self.kwargs['p1'][2] + G.dz)
        pts = np.array([self.kwargs['p1'], new_pt])
        rot_pts = rotate_2point_object(pts, self.axis, self.angle, self.origin)
        self.kwargs['p1'] = tuple(rot_pts[0, :])

        # If specific field components are specified, set to output all components
        try:
            self.kwargs['id']
            self.kwargs['outputs']
            rxargs = dict(self.kwargs)
            del rxargs['outputs']
            self.kwargs = rxargs
        except KeyError:
            pass

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
        except KeyError:
            logger.exception(self.params_str())
            raise

        if self.dorotate:
            self.__dorotate(grid)

        p = uip.check_src_rx_point(p1, self.params_str())
        p2 = uip.round_to_grid_static_point(p1)

        r = self.constructor()
        r.xcoord, r.ycoord, r.zcoord = p
        r.xcoordorigin, r.ycoordorigin, r.zcoordorigin = p

        try:
            r.ID = self.kwargs['id']
            outputs = [self.kwargs['outputs']]
        except KeyError:
            # If no ID or outputs are specified, use default
            r.ID = (r.__class__.__name__ + '(' + str(r.xcoord) + ',' + 
                    str(r.ycoord) + ',' + str(r.zcoord) + ')')
            for key in RxUser.defaultoutputs:
                r.outputs[key] = np.zeros(grid.iterations, 
                                          dtype=config.sim_config.dtypes['float_or_double'])
        else:
            outputs.sort()
            # Get allowable outputs
            allowableoutputs = RxUser.allowableoutputs_gpu if config.sim_config.general['cuda'] else RxUser.allowableoutputs
            # Check and add field output names
            for field in outputs:
                if field in allowableoutputs:
                    r.outputs[field] = np.zeros(grid.iterations, 
                                                dtype=config.sim_config.dtypes['float_or_double'])
                else:
                    logger.exception(self.params_str() + ' contains an '
                                     'output type that is not allowable. '
                                     'Allowable outputs in current context are '
                                     f'{allowableoutputs}.')
                    raise ValueError

        logger.info(self.grid_name(grid) + f"Receiver at {p2[0]:g}m, " 
                    f"{p2[1]:g}m, {p2[2]:g}m with output component(s) "
                    f"{', '.join(r.outputs)} created.")

        grid.rxs.append(r)

        return r


class RxArray(UserObjectMulti):
    """Defines multiple output points in the model.

    Attributes:
        p1: tuple required for position of first receiver x, y, z.
        p2: tuple required for position of last receiver x, y, z.
        dl: tuple required for receiver spacing dx, dy, dz.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 7
        self.hash = '#rx_array'

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            dl = self.kwargs['dl']
        except KeyError:
            logger.exception(self.params_str() + ' requires exactly 9 '
                             'parameters')
            raise

        xs, ys, zs = uip.check_src_rx_point(p1, self.params_str(), 'lower')
        xf, yf, zf = uip.check_src_rx_point(p2, self.params_str(), 'upper')
        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)
        dx, dy, dz = uip.discretise_point(dl)

        if xs > xf or ys > yf or zs > zf:
            logger.exception(self.params_str() + ' the lower coordinates ' 
                             'should be less than the upper coordinates.')
            raise ValueError
        if dx < 0 or dy < 0 or dz < 0:
            logger.exception(self.params_str() + ' the step size should not ' 
                             'be less than zero.')
            raise ValueError
        if dx < 1:
            if dx == 0:
                dx = 1
            else:
                logger.exception(self.params_str() + ' the step size should ' 
                                 'not be less than the spatial discretisation.')
                raise ValueError
        if dy < 1:
            if dy == 0:
                dy = 1
            else:
                logger.exception(self.params_str() + ' the step size should ' 
                                 'not be less than the spatial discretisation.')
                raise ValueError
        if dz < 1:
            if dz == 0:
                dz = 1
            else:
                logger.exception(self.params_str() + ' the step size should '
                                 'not be less than the spatial discretisation.')
                raise ValueError

        logger.info(self.grid_name(grid) + f'Receiver array {p3[0]:g}m, ' 
                    f'{p3[1]:g}m, {p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m, '
                    f'{p4[2]:g}m with steps {dx * grid.dx:g}m, '
                    f'{dy * grid.dy:g}m, {dz * grid.dz:g}m')

        for x in range(xs, xf + 1, dx):
            for y in range(ys, yf + 1, dy):
                for z in range(zs, zf + 1, dz):
                    r = RxUser()
                    r.xcoord = x
                    r.ycoord = y
                    r.zcoord = z
                    r.xcoordorigin = x
                    r.ycoordorigin = y
                    r.zcoordorigin = z
                    # Point relative to main grid
                    p5 = np.array([x, y, z])
                    p5 = uip.descretised_to_continuous(p5)
                    p5 = uip.round_to_grid_static_point(p5)
                    r.ID = (r.__class__.__name__ + '(' + str(x) + ',' + 
                            str(y) + ',' + str(z) + ')')
                    for key in RxUser.defaultoutputs:
                        r.outputs[key] = np.zeros(grid.iterations, dtype=config.sim_config.dtypes['float_or_double'])
                    logger.info(f"  Receiver at {p5[0]:g}m, {p5[1]:g}m, "
                                f"{p5[2]:g}m with output component(s) "
                                f"{', '.join(r.outputs)} created.")
                    grid.rxs.append(r)


class Snapshot(UserObjectMulti):
    """Obtains information about the electromagnetic fields within a volume 
        of the model at a given time instant.

    Attributes:
        p1: tuple required to specify lower left (x,y,z) coordinates of volume 
                of snapshot in metres.
        p2: tuple required to specify upper right (x,y,z) coordinates of volume 
                of snapshot in metres.
        dl: tuple require to specify spatial discretisation of the snapshot 
                in metres.
        filename: string required for name of the file to store snapshot.
        time/iterations: either a float for time or an int for iterations 
                            must be specified for point in time at which the 
                            snapshot will be taken.
        fileext: optional string to indicate type for snapshot file, either 
                            'vti' (default) or 'h5'
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 8
        self.hash = '#snapshot'

    def create(self, grid, uip):
        if isinstance(grid, SubGridBase):
            logger.exception(self.params_str() + ' do not add snapshots to '
                             'subgrids.')
            raise ValueError
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            dl = self.kwargs['dl']
            filename = self.kwargs['filename']
        except KeyError:
            logger.exception(self.params_str() + ' requires exactly 11 '
                             'parameters.')
            raise

        p1, p2 = uip.check_box_points(p1, p2, self.params_str())
        xs, ys, zs = p1
        xf, yf, zf = p2
        dx, dy, dz = uip.discretise_point(dl)

        p3 = uip.round_to_grid_static_point(p1)
        p4 = uip.round_to_grid_static_point(p2)

        # If number of iterations given
        try:
            iterations = self.kwargs['iterations']
        # If real floating point value given
        except KeyError:
            try:
                time = self.kwargs['time']
            except KeyError:
                logger.exception(self.params_str() + ' requires exactly 5 '
                                 'parameters.')
                raise
            if time > 0:
                iterations = round_value((time / grid.dt)) + 1
            else:
                logger.exception(self.params_str() + ' time value must be '
                                 'greater than zero.')
                raise ValueError

        try:
            fileext = self.kwargs['fileext']
            if fileext not in SnapshotUser.fileexts:
                logger.exception(f"'{fileext}' is not a valid format for a "
                                 "snapshot file. Valid options are: "
                                 f"{' '.join(SnapshotUser.fileexts)}.")
                raise ValueError
        except KeyError:
            fileext = SnapshotUser.fileexts[0]

        if dx < 0 or dy < 0 or dz < 0:
            logger.exception(self.params_str() + ' the step size should not '
                             'be less than zero.')
            raise ValueError
        if dx < 1 or dy < 1 or dz < 1:
            logger.exception(self.params_str() + ' the step size should not '
                             'be less than the spatial discretisation.')
            raise ValueError
        if iterations <= 0 or iterations > grid.iterations:
            logger.exception(self.params_str() + ' time value is not valid.')
            raise ValueError

        s = SnapshotUser(xs, ys, zs, xf, yf, zf, dx, dy, dz, iterations, 
                         filename, fileext=fileext)

        logger.info(f'Snapshot from {p3[0]:g}m, {p3[1]:g}m, {p3[2]:g}m, to '
                    f'{p4[0]:g}m, {p4[1]:g}m, {p4[2]:g}m, discretisation '
                    f'{dx * grid.dx:g}m, {dy * grid.dy:g}m, {dz * grid.dz:g}m, '
                    f'at {s.time * grid.dt:g} secs with filename '
                    f'{s.filename}{s.fileext} will be created.')

        grid.snapshots.append(s)


class Material(UserObjectMulti):
    """Specifies a material in the model described by a set of constitutive 
        parameters.

    Attributes:
        er: float required for the relative electric permittivity.
        se: float required for the electric conductivity (Siemens/metre).
        mr: float required for the relative magnetic permeability.
        sm: float required for the magnetic loss.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 9
        self.hash = '#material'

    def create(self, grid, uip):
        try:
            er = self.kwargs['er']
            se = self.kwargs['se']
            mr = self.kwargs['mr']
            sm = self.kwargs['sm']
            material_id = self.kwargs['id']
        except KeyError:
            logger.exception(self.params_str() + ' requires exactly five '
                             'parameters.')
            raise

        if er < 1:
            logger.exception(self.params_str() + ' requires a positive value '
                             'of one or greater for static (DC) permittivity.')
            raise ValueError
        if se != 'inf':
            se = float(se)
            if se < 0:
                logger.exception(self.params_str() + ' requires a positive '
                                 'value for electric conductivity.')
                raise ValueError
        else:
            se = float('inf')
        if mr < 1:
            logger.exception(self.params_str() + ' requires a positive value '
                             'of one or greater for magnetic permeability.')
            raise ValueError
        if sm < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for magnetic loss.')
            raise ValueError
        if any(x.ID == material_id for x in grid.materials):
            logger.exception(self.params_str() + f' with ID {material_id} '
                             'already exists')
            raise ValueError

        # Create a new instance of the Material class material 
        # (start index after pec & free_space)
        m = MaterialUser(len(grid.materials), material_id)
        m.er = er
        m.se = se
        m.mr = mr
        m.sm = sm

        # Set material averaging to False if infinite conductivity, i.e. pec
        if m.se == float('inf'):
            m.averagable = False

        logger.info(self.grid_name(grid) + f'Material {m.ID} with '
                    f'eps_r={m.er:g}, sigma={m.se:g} S/m; mu_r={m.mr:g}, '
                    f'sigma*={m.sm:g} Ohm/m created.')

        grid.materials.append(m)


class AddDebyeDispersion(UserObjectMulti):
    """Adds dispersive properties to already defined :class:`Material` based 
        on multi-pole Debye formulation.

    Attributes:
        n_poles: float required for number of Debye poles.
        er_delta: tuple required for difference between zero-frequency relative 
                    permittivity and relative permittivity at infinite frequency 
                    for each pole.
        tau: tuple required for relaxation time (secs) for each pole.
        material_ids: list required of material ids to apply disperive 
                        properties.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 10
        self.hash = '#add_dispersion_debye'

    def create(self, grid, uip):
        try:
            poles = self.kwargs['poles']
            er_delta = self.kwargs['er_delta']
            tau = self.kwargs['tau']
            material_ids = self.kwargs['material_ids']
        except KeyError:
            logger.exception(self.params_str() + ' requires at least four '
                             'parameters.')
            raise

        if poles < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for number of poles.')
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            logger.exception(self.params_str() + f' material(s) {notfound} do '
                             'not exist')
            raise ValueError

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.type = 'debye'
            disp_material.poles = poles
            disp_material.averagable = False
            for i in range(0, poles):
                if tau[i] > 0:
                    logger.debug('Not checking if relaxation times are '
                                 'greater than time-step.')
                    disp_material.deltaer.append(er_delta[i])
                    disp_material.tau.append(tau[i])
                else:
                    logger.exception(self.params_str() + ' requires positive '
                                     'values for the permittivity difference.')
                    raise ValueError
            if disp_material.poles > config.get_model_config().materials['maxpoles']:
                config.get_model_config().materials['maxpoles'] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [disp_material if mat.numID==material.numID else mat for mat in grid.materials]

            logger.info(self.grid_name(grid) + f"Debye disperion added to {disp_material.ID} "
                        f"with delta_eps_r={', '.join('%4.2f' % deltaer for deltaer in disp_material.deltaer)}, " 
                        f"and tau={', '.join('%4.3e' % tau for tau in disp_material.tau)} secs created.")


class AddLorentzDispersion(UserObjectMulti):
    """Add dispersive properties to already defined :class:`Material` based 
        on multi-pole Lorentz formulation.

    Attributes:
        n_poles: float required for number of Lorentz poles.
        er_delta: tuple required for difference between zero-frequency relative 
                    permittivity and relative permittivity at infinite frequency 
                    for each pole.
        omega: tuple required for frequency (Hz) for each pole.
        delta: tuple required for damping coefficient (Hz) for each pole.
        material_ids: list required of material ids to apply disperive 
                        properties.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 11
        self.hash = '#add_dispersion_lorentz'

    def create(self, grid, uip):
        try:
            poles = self.kwargs['poles']
            er_delta = self.kwargs['er_delta']
            tau = self.kwargs['omega']
            alpha = self.kwargs['delta']
            material_ids = self.kwargs['material_ids']
        except KeyError:
            logger.exception(self.params_str() + ' requires at least five '
                             'parameters.')
            raise

        if poles < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for number of poles.')
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            logger.exception(self.params_str() + f' material(s) {notfound} do '
                             'not exist')
            raise ValueError

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.type = 'lorentz'
            disp_material.poles = poles
            disp_material.averagable = False
            for i in range(0, poles):
                if er_delta[i] > 0 and tau[i] > grid.dt and alpha[i] > grid.dt:
                    disp_material.deltaer.append(er_delta[i])
                    disp_material.tau.append(tau[i])
                    disp_material.alpha.append(alpha[i])
                else:
                    logger.exception(self.params_str() + ' requires positive '
                                     'values for the permittivity difference '
                                     'and frequencies, and associated times '
                                     'that are greater than the time step for '
                                     'the model.')
                    raise ValueError
            if disp_material.poles > config.get_model_config().materials['maxpoles']:
                config.get_model_config().materials['maxpoles'] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [disp_material if mat.numID==material.numID else mat for mat in grid.materials]

            logger.info(self.grid_name(grid) + f"Lorentz disperion added to {disp_material.ID} "
                        f"with delta_eps_r={', '.join('%4.2f' % deltaer for deltaer in disp_material.deltaer)}, "
                        f"omega={', '.join('%4.3e' % tau for tau in disp_material.tau)} secs, "
                        f"and gamma={', '.join('%4.3e' % alpha for alpha in disp_material.alpha)} created.")


class AddDrudeDispersion(UserObjectMulti):
    """Adds dispersive properties to already defined :class:`Material` based 
        on multi-pole Drude formulation.

    Attributes:
        n_poles: float required for number of Drude poles.
        omega: tuple required for frequency (Hz) for each pole. 
        alpha: tuple required for inverse of relaxation time (secs) for each pole.
        material_ids: list required of material ids to apply disperive 
                        properties.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 12
        self.hash = '#add_dispersion_drude'

    def create(self, grid, uip):
        try:
            poles = self.kwargs['poles']
            tau = self.kwargs['tau']
            alpha = self.kwargs['alpha']
            material_ids = self.kwargs['material_ids']
        except KeyError:
            logger.exception(self.params_str() + ' requires at least four '
                             'parameters.')
            raise

        if poles < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for number of poles.')
            raise ValueError

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            logger.exception(self.params_str() + f' material(s) {notfound} do '
                             'not exist.')
            raise ValueError

        for material in materials:
            disp_material = DispersiveMaterialUser(material.numID, material.ID)
            disp_material.er = material.er
            disp_material.se = material.se
            disp_material.mr = material.mr
            disp_material.sm = material.sm
            disp_material.type = 'drude'
            disp_material.poles = poles
            disp_material.averagable = False
            for i in range(0, poles):
                if tau[i] > 0 and alpha[i] > grid.dt:
                    disp_material.tau.append(tau[i])
                    disp_material.alpha.append(alpha[i])
                else:
                    logger.exception(self.params_str() + ' requires positive '
                                     'values for the frequencies, and '
                                     'associated times that are greater than '
                                     'the time step for the model.')
                    raise ValueError
            if disp_material.poles > config.get_model_config().materials['maxpoles']:
                config.get_model_config().materials['maxpoles'] = disp_material.poles

            # Replace original material with newly created DispersiveMaterial
            grid.materials = [disp_material if mat.numID==material.numID else mat for mat in grid.materials]

            logger.info(self.grid_name(grid) + f"Drude disperion added to {disp_material.ID} "
                        f"with omega={', '.join('%4.3e' % tau for tau in disp_material.tau)} secs, "
                        f"and gamma={', '.join('%4.3e' % alpha for alpha in disp_material.alpha)} secs created.")


class SoilPeplinski(UserObjectMulti):
    """Mixing model for soils proposed by Peplinski et al.
        (http://dx.doi.org/10.1109/36.387598)

    Attributes:
        sand_fraction: float required for sand fraction of soil.
        clay_fraction: float required for clay of soil.
        bulk_density: float required for bulk density of soil (gm/cm^3).
        sand_density: float required for density of sand particles in soil (gm/cm^3).
        water_fraction_lower: float required for lower boundary of volumetric 
                                water fraction of the soil.
        water_fraction_upper: float required for upper boundary of volumetric 
                                water fraction of the soil.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 13
        self.hash = '#soil_peplinski'

    def create(self, grid, uip):
        try:
            sand_fraction = self.kwargs['sand_fraction']
            clay_fraction = self.kwargs['clay_fraction']
            bulk_density = self.kwargs['bulk_density']
            sand_density = self.kwargs['sand_density']
            water_fraction_lower = self.kwargs['water_fraction_lower']
            water_fraction_upper = self.kwargs['water_fraction_upper']
            ID = self.kwargs['id']
        except KeyError:
            logger.exception(self.params_str() + ' requires at exactly seven '
                             'parameters.')
            raise

        if sand_fraction < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for the sand fraction.')
            raise ValueError
        if clay_fraction < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for the clay fraction.')
            raise ValueError
        if bulk_density < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for the bulk density.')
            raise ValueError
        if sand_density < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for the sand particle density.')
            raise ValueError
        if water_fraction_lower < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for the lower limit of the water volumetric '
                             'fraction.')
            raise ValueError
        if water_fraction_upper < 0:
            logger.exception(self.params_str() + ' requires a positive value '
                             'for the upper limit of the water volumetric '
                             'fraction.')
            raise ValueError
        if any(x.ID == ID for x in grid.mixingmodels):
            logger.exception(self.params_str() + f' with ID {ID} already exists')
            raise ValueError

        # Create a new instance of the Material class material 
        # (start index after pec & free_space)
        s = PeplinskiSoilUser(ID, sand_fraction, clay_fraction, bulk_density, 
                              sand_density, (water_fraction_lower, water_fraction_upper))

        logger.info(self.grid_name(grid) + 'Mixing model (Peplinski) used to '
                    f'create {s.ID} with sand fraction {s.S:g}, clay fraction '
                    f'{s.C:g}, bulk density {s.rb:g}g/cm3, sand particle '
                    f'density {s.rs:g}g/cm3, and water volumetric fraction '
                    f'{s.mu[0]:g} to {s.mu[1]:g} created.')

        grid.mixingmodels.append(s)


class GeometryView(UserObjectMulti):
    """Outputs to file(s) information about the geometry (mesh) of model.

    The geometry information is saved in Visual Toolkit (VTK) formats.

    Attributes:
        p1: tuple required for lower left (x,y,z) coordinates of volume of 
                geometry view in metres.
        p2: tuple required for upper right (x,y,z) coordinates of volume of 
                geometry view in metres.
        dl: tuple required for spatial discretisation of geometry view in metres.
        output_tuple: string required for per-cell 'n' (normal) or per-cell-edge
                        'f' (fine) geometry views.
        filename: string required for filename where geometry view will be 
                    stored in the same directory as input file.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 14
        self.hash = '#geometry_view'

    def geometry_view_constructor(self, grid, output_type):
        """Selects appropriate class for geometry view dependent on grid type 
            and geometry view type, i.e. normal or fine.
        """

        if output_type == 'n':
            if isinstance(grid, SubGridBase):
                from .geometry_outputs import GeometryViewSubgridVoxels as GeometryViewUser            
            else:
                from .geometry_outputs import GeometryViewVoxels as GeometryViewUser            
        else:
            from .geometry_outputs import GeometryViewLines as GeometryViewUser

        return GeometryViewUser

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            dl = self.kwargs['dl']
            output_type = self.kwargs['output_type'].lower()
            filename = self.kwargs['filename']
        except KeyError:
            logger.exception(self.params_str() + ' requires exactly eleven '
                             'parameters.')
            raise
        GeometryViewUser = self.geometry_view_constructor(grid, output_type)
        try:
            p3 = uip.round_to_grid_static_point(p1)
            p4 = uip.round_to_grid_static_point(p2)
            p1, p2 = uip.check_box_points(p1, p2, self.params_str())
        except ValueError:
            logger.exception(self.params_str() + ' point is outside the domain.')
            raise
        xs, ys, zs = p1
        xf, yf, zf = p2

        dx, dy, dz = uip.discretise_static_point(dl)


        if dx < 0 or dy < 0 or dz < 0:
            logger.exception(self.params_str() + ' the step size should not be '
                             'less than zero.')
            raise ValueError
        if dx > grid.nx or dy > grid.ny or dz > grid.nz:
            logger.exception(self.params_str() + ' the step size should be '
                             'less than the domain size.')
            raise ValueError
        if dx < 1 or dy < 1 or dz < 1:
            logger.exception(self.params_str() + ' the step size should not '
                             'be less than the spatial discretisation.')
            raise ValueError
        if output_type != 'n' and output_type != 'f':
            logger.exception(self.params_str() + ' requires type to be either '
                             'n (normal) or f (fine).')
            raise ValueError
        if output_type == 'f' and (dx * grid.dx != grid.dx or 
                                   dy * grid.dy != grid.dy or 
                                   dz * grid.dz != grid.dz):
            logger.exception(self.params_str() + ' requires the spatial '
                             'discretisation for the geometry view to be the '
                             'same as the model for geometry view of '
                             'type f (fine)')
            raise ValueError

        g = GeometryViewUser(xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, grid)

        logger.info(self.grid_name(grid) + f'Geometry view from {p3[0]:g}m, '
                    f'{p3[1]:g}m, {p3[2]:g}m, to {p4[0]:g}m, {p4[1]:g}m, '
                    f'{p4[2]:g}m, discretisation {dx * grid.dx:g}m, '
                    f'{dy * grid.dy:g}m, {dz * grid.dz:g}m, with filename '
                    f'base {g.filename} created.')

        grid.geometryviews.append(g)


class GeometryObjectsWrite(UserObjectMulti):
    """Writes geometry generated in a model to file which can be imported into 
        other models.

    Attributes:
        p1: tuple required for lower left (x,y,z) coordinates of volume of 
                output in metres.
        p2: tuple required for upper right (x,y,z) coordinates of volume of 
                output in metres.
        filename: string required for filename where output will be stored in 
                    the same directory as input file.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 15
        self.hash = '#geometry_objects_write'

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            basefilename = self.kwargs['filename']
        except KeyError:
            logger.exception(self.params_str() + ' requires exactly seven '
                             'parameters.')
            raise

        p1, p2 = uip.check_box_points(p1, p2, self.params_str())
        x0, y0, z0 = p1
        x1, y1, z1 = p2

        g = GeometryObjectsUser(x0, y0, z0, x1, y1, z1, basefilename)

        logger.info(f'Geometry objects in the volume from {p1[0] * grid.dx:g}m, '
                    f'{p1[1] * grid.dy:g}m, {p1[2] * grid.dz:g}m, to '
                    f'{p2[0] * grid.dx:g}m, {p2[1] * grid.dy:g}m, '
                    f'{p2[2] * grid.dz:g}m, will be written to '
                    f'{g.filename_hdf5}, with materials written to '
                    f'{g.filename_materials}')

        grid.geometryobjectswrite.append(g)


class PMLCFS(UserObjectMulti):
    """Controls parameters that are used to build each order of PML. 

    Attributes:
        alphascalingprofile: string required for type of scaling to use for 
                                CFS alpha parameter.
        alphascalingdirection: string required for direction of scaling to use 
                                for CFS alpha parameter.
        alphamin: float required for minimum value for the CFS alpha parameter.
        alphamax: float required for maximum value for the CFS alpha parameter.
        kappascalingprofile: string required for type of scaling to use for 
                                CFS kappa parameter.
        kappascalingdirection: string required for direction of scaling to use 
                                for CFS kappa parameter.
        kappamin: float required for minimum value for the CFS kappa parameter.
        kappamax: float required for maximum value for the CFS kappa parameter.
        sigmascalingprofile: string required for type of scaling to use for 
                                CFS sigma parameter.
        sigmascalingdirection: string required for direction of scaling to use 
                                for CFS sigma parameter.
        sigmamin: float required for minimum value for the CFS sigma parameter.
        sigmamax: float required for maximum value for the CFS sigma parameter.
    """

    count = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 16
        self.hash = '#pml_cfs'
        PMLCFS.count += 1
        if PMLCFS.count == 2:
            logger.exception(self.params_str() + ' can only be used up to two '
                             'times, for up to a 2nd order PML.')
            raise ValueError

    def create(self, grid, uip):
        try:
            alphascalingprofile = self.kwargs['alphascalingprofile']
            alphascalingdirection = self.kwargs['alphascalingdirection']
            alphamin = self.kwargs['alphamin']
            alphamax = self.kwargs['alphamax']
            kappascalingprofile = self.kwargs['kappascalingprofile']
            kappascalingdirection = self.kwargs['kappascalingdirection']
            kappamin = self.kwargs['kappamin']
            kappamax = self.kwargs['kappamax']
            sigmascalingprofile = self.kwargs['sigmascalingprofile']
            sigmascalingdirection = self.kwargs['sigmascalingdirection']
            sigmamin = self.kwargs['sigmamin']
            sigmamax = self.kwargs['sigmamax']
        except KeyError:
            logger.exception(self.params_str() + ' requires exactly twelve '
                             'parameters.')
            raise

        if (alphascalingprofile not in CFSParameter.scalingprofiles.keys() or 
            kappascalingprofile not in CFSParameter.scalingprofiles.keys() or 
            sigmascalingprofile not in CFSParameter.scalingprofiles.keys()):
            logger.exception(self.params_str() + ' must have scaling type '
                             f"{','.join(CFSParameter.scalingprofiles.keys())}")
            raise ValueError
        if (alphascalingdirection not in CFSParameter.scalingdirections or 
            kappascalingdirection not in CFSParameter.scalingdirections or 
            sigmascalingdirection not in CFSParameter.scalingdirections):
            logger.exception(self.params_str() + ' must have scaling type '
                             f"{','.join(CFSParameter.scalingdirections)}")
            raise ValueError
        if (float(alphamin) < 0 or float(alphamax) < 0 or 
            float(kappamin) < 0 or float(kappamax) < 0 or float(sigmamin) < 0):
            logger.exception(self.params_str() + ' minimum and maximum scaling '
                             'values must be greater than zero.')
            raise ValueError
        if float(kappamin) < 1:
            logger.exception(self.params_str() + ' minimum scaling value for '
                             'kappa must be greater than or equal to one.')
            raise ValueError

        cfsalpha = CFSParameter()
        cfsalpha.ID = 'alpha'
        cfsalpha.scalingprofile = alphascalingprofile
        cfsalpha.scalingdirection = alphascalingdirection
        cfsalpha.min = float(alphamin)
        cfsalpha.max = float(alphamax)
        cfskappa = CFSParameter()
        cfskappa.ID = 'kappa'
        cfskappa.scalingprofile = kappascalingprofile
        cfskappa.scalingdirection = kappascalingdirection
        cfskappa.min = float(kappamin)
        cfskappa.max = float(kappamax)
        cfssigma = CFSParameter()
        cfssigma.ID = 'sigma'
        cfssigma.scalingprofile = sigmascalingprofile
        cfssigma.scalingdirection = sigmascalingdirection
        cfssigma.min = float(sigmamin)
        if sigmamax == 'None':
            cfssigma.max = None
        else:
            cfssigma.max = float(sigmamax)
        cfs = CFS()
        cfs.alpha = cfsalpha
        cfs.kappa = cfskappa
        cfs.sigma = cfssigma

        logger.info(f'PML CFS parameters: alpha (scaling: {cfsalpha.scalingprofile}, '
                    f'scaling direction: {cfsalpha.scalingdirection}, min: '
                    f'{cfsalpha.min:g}, max: {cfsalpha.max:g}), kappa (scaling: '
                    f'{cfskappa.scalingprofile}, scaling direction: '
                    f'{cfskappa.scalingdirection}, min: {cfskappa.min:g}, max: '
                    f'{cfskappa.max:g}), sigma (scaling: {cfssigma.scalingprofile}, '
                    f'scaling direction: {cfssigma.scalingdirection}, min: '
                    f'{cfssigma.min:g}, max: {cfssigma.max:g}) created.')

        grid.cfs.append(cfs)


class Subgrid(UserObjectMulti):
    """"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.children_multiple = []
        self.children_geometry = []

    def add(self, node):
        if isinstance(node, UserObjectMulti):
            self.children_multiple.append(node)
        elif isinstance(node, UserObjectGeometry):
            self.children_geometry.append(node)
        else:
            logger.exception('This object is unknown to gprMax.')
            raise ValueError
            