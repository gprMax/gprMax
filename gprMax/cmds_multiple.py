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

"""Object which can be created multiple times."""
import gprMax.config as config
from .config import z0
from .config import dtypes
from .utilities import round_value
from .cmds_geometry.cmds_geometry import UserObjectGeometry
from .waveforms import Waveform as WaveformUser
from .sources import VoltageSource as VoltageSourceUser
from .sources import HertzianDipole as HertzianDipoleUser
from .sources import MagneticDipole as MagneticDipoleUser
from .sources import TransmissionLine as TransmissionLineUser
from .snapshots import Snapshot as SnapshotUser
from .receivers import Rx as RxUser
from .materials import Material as MaterialUser
from .materials import PeplinskiSoil as PeplinskiSoilUser
from .geometry_outputs import GeometryObjects as GeometryObjectsUser
from .pml import CFSParameter
from .pml import CFS
from .subgrids.base import SubGridBase

from .exceptions import CmdInputError


import numpy as np
from tqdm import tqdm

floattype = dtypes['float_or_double']


class UserObjectMulti:
    """Specific multiobject object."""

    def __init__(self, **kwargs):
        """Constructor."""
        self.kwargs = kwargs
        self.order = None
        self.hash = '#example'

    def __str__(self):
        """Readble user string as per hash commands."""
        s = ''
        for k, v in self.kwargs.items():
            if isinstance(v, tuple) or isinstance(v, list):
                v = ' '.join([str(el) for el in v])
            s += str(v) + ' '

        return '{}: {}'.format(self.hash, s[:-1])

    def create(self, grid, uip):
        """Create the object and add it to the grid."""
        pass

    def params_str(self):
        """Readble string of parameters given to object."""
        return self.hash + ': ' + str(self.kwargs)


class Waveform(UserObjectMulti):
    """Allows you to specify waveforms to use with sources in the model.

    :param wave_type: wave type (see main documentation)
    :type wave_type: str, non-optional
    :param amp:  The scaling of the maximum amplitude of the waveform
    :type amp: float, non-optional
    :param freq: The centre frequency of the waveform (Hertz)
    :type freq: float, non-optional
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 0
        self.hash = '#waveform'

    def create(self, grid, uip):
        try:
            wavetype = self.kwargs['wave_type'].lower()
            amp = self.kwargs['amp']
            freq = self.kwargs['freq']
            ID = self.kwargs['id']

        except KeyError:
            raise CmdInputError(self.params_str() + ' requires exactly four parameters')

        if wavetype not in WaveformUser.types:
            raise CmdInputError(self.__str__() + ' must have one of the following types {}'.format(','.join(WaveformUser.types)))
        if freq <= 0:
            raise CmdInputError(self.__str__() + ' requires an excitation frequency value of greater than zero')
        if any(x.ID == ID for x in grid.waveforms):
            raise CmdInputError(self.__str__() + ' with ID {} already exists'.format(ID))

        w = WaveformUser()
        w.ID = ID
        w.type = wavetype
        w.amp = amp
        w.freq = freq

        if config.is_messages():
            print('Waveform {} of type {} with maximum amplitude scaling {:g}, frequency {:g}Hz created.'.format(w.ID, w.type, w.amp, w.freq))

        grid.waveforms.append(w)


class VoltageSource(UserObjectMulti):
    """Allows you to introduce a voltage source at an electric field location.

    :param polarisation: Polarisation of the source x, y, z
    :type polarisation: str, non-optional
    :param p1:  Position of the source x, y, z
    :type p1: list, non-optional
    :param resistance: Is the internal resistance of the voltage source in Ohms
    :type resistance: float, non-optional
    :param waveform_id: The identifier of the waveform that should be used with the source.
    :type waveform_id: str, non-optional
    :param start: Time to delay to start the source
    :type start: float, optional
    :param stop: Time to remove the source
    :type stop: float, optional
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 1
        self.hash = '#voltage_source'

    def create(self, grid, uip):
        """Create voltage source and add it to the grid."""
        try:
            p1 = self.kwargs['p1']
            polarisation = self.kwargs['polarisation'].lower()
            resistance = self.kwargs['resistance']
            waveform_id = self.kwargs['waveform_id']

        except KeyError:
            raise CmdInputError(self.__str__() + "  requires at least six parameters")

        # Check polarity & position parameters
        if polarisation not in ('x', 'y', 'z'):
            raise CmdInputError("'{}' polarisation must be x, y, or z".format(self.__str__()))
        if '2D TMx' in grid.mode and (polarisation == 'y' or polarisation == 'z'):
            raise CmdInputError("'{}' polarisation must be x in 2D TMx mode".format(self.__str__()))
        elif '2D TMy' in grid.mode and (polarisation == 'x' or polarisation == 'z'):
            raise CmdInputError("'{}' polarisation must be y in 2D TMy mode".format(self.__str__()))
        elif '2D TMz' in grid.mode and (polarisation == 'x' or polarisation == 'y'):
            raise CmdInputError("'{}' polarisation must be z in 2D TMz mode".format(self.__str__()))

        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.__str__())

        if resistance < 0:
            raise CmdInputError("'{}' requires a source resistance of zero or greater".format(self.__str__()))

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            raise CmdInputError("'{}' there is no waveform with the identifier {}'.format(tmp[5]".format(self.__str__()))

        v = VoltageSourceUser()
        v.polarisation = polarisation
        v.xcoord = xcoord
        v.ycoord = ycoord
        v.zcoord = zcoord
        v.ID = v.__class__.__name__ + '(' + str(v.xcoord) + ',' + str(v.ycoord) + ',' + str(v.zcoord) + ')'
        v.resistance = resistance
        v.waveformID = waveform_id

        try:
            start = self.kwargs['start']
            stop = self.kwargs['stop']
            # Check source start & source remove time parameters
            if start < 0:
                raise CmdInputError("'{}' delay of the initiation of the source should not be less than zero".format(self.__str__()))
            if stop < 0:
                raise CmdInputError("'{}' time to remove the source should not be less than zero".format(self.__str__()))
            if stop - start <= 0:
                raise CmdInputError("'{}' duration of the source should not be zero or less".format(self.__str__()))
            v.start = start
            if stop > grid.timewindow:
                v.stop = grid.timewindow
            else:
                v.stop = stop
            startstop = ' start time {:g} secs, finish time {:g} secs '.format(v.start, v.stop)
        except KeyError:
            v.start = 0
            v.stop = grid.timewindow
            startstop = ' '

        v.calculate_waveform_values(grid)

        if config.is_messages():
            print('Voltage source with polarity {} at {:g}m, {:g}m, {:g}m, resistance {:.1f} Ohms,'.format(v.polarisation, v.xcoord * grid.dx, v.ycoord * grid.dy, v.zcoord * grid.dz, v.resistance) + startstop + 'using waveform {} created.'.format(v.waveformID))

        grid.voltagesources.append(v)


class HertzianDipole(UserObjectMulti):
    """Allows you to specify a current density term at an electric field location
    - the simplest excitation, often referred to as an additive or soft source.

    :param polarisation: Polarisation of the source x, y, z
    :type polarisation: str, non-optional
    :param p1:  Position of the source x, y, z
    :type p1: list, non-optional
    :param waveform_id: The identifier of the waveform that should be used with the source.
    :type waveform_id: str, non-optional
    :param start: Time to delay to start the source
    :type start: float, optional
    :param stop: Time to remove the source
    :type stop: float, optional
    """

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 2
        self.hash = '#hertzian_dipole'

    def create(self, grid, uip):
        """Create HertzianDipole and add it to the grid."""
        try:
            polarisation = self.kwargs['polarisation'].lower()
            p1 = self.kwargs['p1']
            waveform_id = self.kwargs['waveform_id']

        except KeyError:
            raise CmdInputError("'{}' requires at least 3 parameters".format(self.params_str()))

        # Check polarity & position parameters
        if polarisation not in ('x', 'y', 'z'):
            raise CmdInputError("'{}' polarisation must be x, y, or z".format(self.__str__()))
        if '2D TMx' in grid.mode and (polarisation == 'y' or polarisation == 'z'):
            raise CmdInputError("'{}' polarisation must be x in 2D TMx mode".format(self.__str__()))
        elif '2D TMy' in grid.mode and (polarisation == 'x' or polarisation == 'z'):
            raise CmdInputError("'{}' polarisation must be y in 2D TMy mode".format(self.__str__()))
        elif '2D TMz' in grid.mode and (polarisation == 'x' or polarisation == 'y'):
            raise CmdInputError("'{}' polarisation must be z in 2D TMz mode".format(self.__str__()))

        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.__str__())

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            raise CmdInputError("'{}' there is no waveform with the identifier {}'.format(tmp[4]".format(self.__str__()))

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
        h.ID = h.__class__.__name__ + '(' + str(h.xcoord) + ',' + str(h.ycoord) + ',' + str(h.zcoord) + ')'
        h.waveformID = waveform_id

        try:
            # Check source start & source remove time parameters
            start = self.kwargs['start']
            stop = self.kwargs['stop']
            if start < 0:
                raise CmdInputError("'{}' delay of the initiation of the source should not be less than zero".format(self.__str__()))
            if stop < 0:
                raise CmdInputError("'{}' time to remove the source should not be less than zero".format(self.__str__()))
            if stop - start <= 0:
                raise CmdInputError("'{}' duration of the source should not be zero or less".format(self.__str__()))
            h.start = start
            if stop > grid.timewindow:
                h.stop = grid.timewindow
            else:
                h.stop = stop
            startstop = ' start time {:g} secs, finish time {:g} secs '.format(h.start, h.stop)
        except KeyError:
            h.start = 0
            h.stop = grid.timewindow
            print(grid.timewindow)
            startstop = ' '

        h.calculate_waveform_values(grid)

        if config.is_messages():
            if grid.mode == '2D':
                print('Hertzian dipole is a line source in 2D with polarity {} at {:g}m, {:g}m, {:g}m,'.format(h.polarisation, h.xcoord * grid.dx, h.ycoord * grid.dy, h.zcoord * grid.dz) + startstop + 'using waveform {} created.'.format(h.waveformID))
            else:
                print('Hertzian dipole with polarity {} at {:g}m, {:g}m, {:g}m,'.format(h.polarisation, h.xcoord * grid.dx, h.ycoord * grid.dy, h.zcoord * grid.dz) + startstop + 'using waveform {} created.'.format(h.waveformID))

        grid.hertziandipoles.append(h)


class MagneticDipole(UserObjectMulti):
    """This will simulate an infinitesimal magnetic dipole. This is often referred
    to as an additive or soft source.

    :param polarisation: Polarisation of the source x, y, z
    :type polarisation: str, non-optional
    :param p1:  Position of the source x, y, z
    :type p1: list, non-optional
    :param waveform_id: The identifier of the waveform that should be used with the source.
    :type waveform_id: str, non-optional
    :param start: Time to delay to start the source
    :type start: float, optional
    :param stop: Time to remove the source
    :type stop: float, optional
    """

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 3
        self.hash = '#magnetic_dipole'

    def create(self, grid, uip):
        """Create Magnetic Dipole and add it the grid."""
        try:
            polarisation = self.kwargs['polarisation'].lower()
            p1 = self.kwargs['p1']
            waveform_id = self.kwargs['waveform_id']
        except KeyError:
            raise CmdInputError("'{}' requires at least five parameters".format(self.__str__()))

        # Check polarity & position parameters
        if polarisation not in ('x', 'y', 'z'):
            raise CmdInputError("'{}' polarisation must be x, y, or z".format(self.__str__()))
        if '2D TMx' in grid.mode and (polarisation == 'y' or polarisation == 'z'):
            raise CmdInputError("'{}' polarisation must be x in 2D TMx mode".format(self.__str__()))
        elif '2D TMy' in grid.mode and (polarisation == 'x' or polarisation == 'z'):
            raise CmdInputError("'{}' polarisation must be y in 2D TMy mode".format(self.__str__()))
        elif '2D TMz' in grid.mode and (polarisation == 'x' or polarisation == 'y'):
            raise CmdInputError("'{}' polarisation must be z in 2D TMz mode".format(self.__str__()))

        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.__str__())

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            raise CmdInputError("'{}' there is no waveform with the identifier {}".format(self.__str__(), waveform_id))

        m = MagneticDipoleUser()
        m.polarisation = polarisation
        m.xcoord = xcoord
        m.ycoord = ycoord
        m.zcoord = zcoord
        m.xcoordorigin = xcoord
        m.ycoordorigin = ycoord
        m.zcoordorigin = zcoord
        m.ID = m.__class__.__name__ + '(' + str(m.xcoord) + ',' + str(m.ycoord) + ',' + str(m.zcoord) + ')'
        m.waveformID = waveform_id

        try:
            # Check source start & source remove time parameters
            start = self.kwargs['start']
            stop = self.kwargs['stop']
            if start < 0:
                raise CmdInputError("'{}' delay of the initiation of the source should not be less than zero".format(self.__str__()))
            if stop < 0:
                raise CmdInputError("'{}' time to remove the source should not be less than zero".format(self.__str__()))
            if stop - start <= 0:
                raise CmdInputError("'{}' duration of the source should not be zero or less".format(self.__str__()))
            m.start = start
            if stop > grid.timewindow:
                m.stop = grid.timewindow
            else:
                m.stop = stop
            startstop = ' start time {:g} secs, finish time {:g} secs '.format(m.start, m.stop)
        except KeyError:
            m.start = 0
            m.stop = grid.timewindow
            startstop = ' '

        m.calculate_waveform_values(grid)

        if config.is_messages():
            print('Magnetic dipole with polarity {} at {:g}m, {:g}m, {:g}m,'.format(m.polarisation, m.xcoord * grid.dx, m.ycoord * grid.dy, m.zcoord * grid.dz) + startstop + 'using waveform {} created.'.format(m.waveformID))

        grid.magneticdipoles.append(m)


class TransmissionLine(UserObjectMulti):
    """Allows you to introduce a one-dimensional transmission line model
    at an electric field location

    :param polarisation: Polarisation of the source x, y, z
    :type polarisation: str, non-optional
    :param p1:  Position of the source x, y, z
    :type p1: list, non-optional
    :param resistance: Is the internal resistance of the voltage source in Ohms
    :type resistance: float, non-optional
    :param waveform_id: The identifier of the waveform that should be used with the source.
    :type waveform_id: str, non-optional
    :param start: Time to delay to start the source
    :type start: float, optional
    :param stop: Time to remove the source
    :type stop: float, optional
    """
    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 4
        self.hash = '#transmission_line'

    def create(self, grid, uip):

        try:
            polarisation = self.kwargs['polarisation'].lower()
            p1 = self.kwargs['p1']
            waveform_id = self.kwargs['waveform_id']
            resistance = self.kwargs['resistance']

        except KeyError:
            raise CmdInputError("'{}' requires at least six parameters".format(self.params_str()))

        # Warn about using a transmission line on GPU
        if grid.gpu is not None:
            raise CmdInputError("'{}' A #transmission_line cannot currently be used with GPU solving. Consider using a #voltage_source instead.".format(self.__str__()))

        # Check polarity & position parameters
        if polarisation not in ('x', 'y', 'z'):
            raise CmdInputError("'{}' polarisation must be x, y, or z".format(self.__str__()))
        if '2D TMx' in grid.mode and (polarisation == 'y' or polarisation == 'z'):
            raise CmdInputError("'{}' polarisation must be x in 2D TMx mode".format(self.__str__()))
        elif '2D TMy' in grid.mode and (polarisation == 'x' or polarisation == 'z'):
            raise CmdInputError("'{}' polarisation must be y in 2D TMy mode".format(self.__str__()))
        elif '2D TMz' in grid.mode and (polarisation == 'x' or polarisation == 'y'):
            raise CmdInputError("'{}' polarisation must be z in 2D TMz mode".format(self.__str__()))

        xcoord, ycoord, zcoord = uip.check_src_rx_point(p1, self.__str__())

        if resistance <= 0 or resistance >= z0:
            raise CmdInputError("'{}' requires a resistance greater than zero and less than the impedance of free space, i.e. 376.73 Ohms".format(self.__str__()))

        # Check if there is a waveformID in the waveforms list
        if not any(x.ID == waveform_id for x in grid.waveforms):
            raise CmdInputError("'{}' there is no waveform with the identifier {}'.format(tmp[5]".format(self.__str__()))

        t = TransmissionLineUser(grid)
        t.polarisation = polarisation
        t.xcoord = xcoord
        t.ycoord = ycoord
        t.zcoord = zcoord
        t.ID = t.__class__.__name__ + '(' + str(t.xcoord) + ',' + str(t.ycoord) + ',' + str(t.zcoord) + ')'
        t.resistance = resistance
        t.waveformID = waveform_id

        try:
            # Check source start & source remove time parameters
            start = self.kwargs['start']
            stop = self.kwargs['stop']
            if start < 0:
                raise CmdInputError("'{}' delay of the initiation of the source should not be less than zero".format(self.__str__()))
            if stop < 0:
                raise CmdInputError("'{}' time to remove the source should not be less than zero".format(self.__str__()))
            if stop - start <= 0:
                raise CmdInputError("'{}' duration of the source should not be zero or less".format(self.__str__()))
            t.start = start
            if stop > grid.timewindow:
                t.stop = grid.timewindow
            else:
                t.stop = stop
            startstop = ' start time {:g} secs, finish time {:g} secs '.format(t.start, t.stop)
        except KeyError:
            t.start = 0
            t.stop = grid.timewindow
            startstop = ' '

        t.calculate_waveform_values(grid)
        t.calculate_incident_V_I(grid)

        if config.is_messages():
            print('Transmission line with polarity {} at {:g}m, {:g}m, {:g}m, resistance {:.1f} Ohms,'.format(t.polarisation, t.xcoord * grid.dx, t.ycoord * grid.dy, t.zcoord * grid.dz, t.resistance) + startstop + 'using waveform {} created.'.format(t.waveformID))

        grid.transmissionlines.append(t)


class Rx(UserObjectMulti):
    """Allows you to introduce output points into the model. These are locations
    where the values of the electric and magnetic field components over the number
    of iterations of the model will be saved to file. .

    :param p1: Position of the receiver x, y, z
    :type p1: list, non-optional
    :param id: Identifier for the receiver
    :type id: str, non-optional
    :param outputs: is a list of outputs with this receiver. It can be any
    selection from Ex, Ey, Ez, Hx, Hy, Hz, Ix, Iy, or Iz.
    :type outputs: list, non-optional

    """

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 5
        self.hash = '#rx'
        self.constructor = RxUser

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
        except KeyError:
            raise CmdInputError("'{}' has an incorrect number of parameters".format(self.__str__()))

        p = uip.check_src_rx_point(p1, self.__str__())

        r = self.constructor()
        r.xcoord, r.ycoord, r.zcoord = p
        r.xcoordorigin, r.ycoordorigin, r.zcoordorigin = p

        try:
            r.ID = self.kwargs['id']
            outputs = self.kwargs['outputs']
            # Get allowable outputs
            if grid.gpu is not None:
                allowableoutputs = RxUser.gpu_allowableoutputs
            else:
                allowableoutputs = RxUser.allowableoutputs
            # Check and add field output names
            for field in outputs:
                if field in allowableoutputs:
                    r.outputs[field] = np.zeros(grid.iterations, dtype=floattype)
                else:
                    raise CmdInputError("{} contains an output type that is not allowable. Allowable outputs in current context are {}".format(self.__str__(), allowableoutputs))

        # If no ID or outputs are specified, use default
        except KeyError:
            r.ID = r.__class__.__name__ + '(' + str(r.xcoord) + ',' + str(r.ycoord) + ',' + str(r.zcoord) + ')'
            for key in RxUser.defaultoutputs:
                r.outputs[key] = np.zeros(grid.iterations, dtype=floattype)
        if config.is_messages():
            print('Receiver at {:g}m, {:g}m, {:g}m with output component(s) {} created.'.format(r.xcoord * grid.dx, r.ycoord * grid.dy, r.zcoord * grid.dz, ', '.join(r.outputs)))

        grid.rxs.append(r)

        return r

class RxArray(UserObjectMulti):
    """Provides a simple method of defining multiple output points in the model.

    :param p1: Position of first receiver x, y, z
    :type p1: list, non-optional
    :param p2: Position of last receiver x, y, z
    :type p2: list, non-optional
    :param dl: Receiver spacing dx, dy, dz
    :type dl: list, non-optional
    """

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 6
        self.hash = '#rx_array'

    def create(self, grid, uip):

        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            dl = self.kwargs['dl']

        except KeyError:
            raise CmdInputError("'{}' requires exactly 9 parameters".format(self.__str__()))

        xs, ys, zs = uip.check_src_rx_point(p1, self.__str__(), 'lower')
        xf, yf, zf = uip.check_src_rx_point(p2, self.__str__(), 'upper')
        dx, dy, dz = uip.discretise_point(dl)

        if xs > xf or ys > yf or zs > zf:
            raise CmdInputError("'{}' the lower coordinates should be less than the upper coordinates".format(self.__str__()))
        if dx < 0 or dy < 0 or dz < 0:
            raise CmdInputError("'{}' the step size should not be less than zero".format(self.__str__()))
        if dx < 1:
            if dx == 0:
                dx = 1
            else:
                raise CmdInputError("'{}' the step size should not be less than the spatial discretisation".format(self.__str__()))
        if dy < 1:
            if dy == 0:
                dy = 1
            else:
                raise CmdInputError("'{}' the step size should not be less than the spatial discretisation".format(self.__str__()))
        if dz < 1:
            if dz == 0:
                dz = 1
            else:
                raise CmdInputError("'{}' the step size should not be less than the spatial discretisation".format(self.__str__()))

        if config.is_messages():
            print('Receiver array {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with steps {:g}m, {:g}m, {:g}m'.format(xs * grid.dx, ys * grid.dy, zs * grid.dz, xf * grid.dx, yf * grid.dy, zf * grid.dz, dx * grid.dx, dy * grid.dy, dz * grid.dz))

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
                    r.ID = r.__class__.__name__ + '(' + str(x) + ',' + str(y) + ',' + str(z) + ')'
                    for key in RxUser.defaultoutputs:
                        r.outputs[key] = np.zeros(grid.iterations, dtype=floattype)
                    if config.is_messages():
                        print('  Receiver at {:g}m, {:g}m, {:g}m with output component(s) {} created.'.format(r.xcoord * grid.dx, r.ycoord * grid.dy, r.zcoord * grid.dz, ', '.join(r.outputs)))
                    grid.rxs.append(r)


class Snapshot(UserObjectMulti):
    """Snapshot User Object."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 19
        self.hash = '#snapshot'

    def create(self, grid, uip):

        if isinstance(grid, SubGridBase):
            raise CmdInputError("'{}' Do not add Snapshots to Subgrids.".format(self.__str__()))
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            dl = self.kwargs['dl']
            filename = self.kwargs['filename']
        except KeyError:
            raise CmdInputError("'{}' requires exactly 11 parameters".format(self.__str__()))

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2
        dx, dy, dz = uip.discretise_point(dl)

        # If number of iterations given
        try:
            iterations = self.kwargs['iterations']
        # If real floating point value given
        except KeyError:
            try:
                time = self.kwargs['time']
            except KeyError:
                raise CmdInputError("'{}' requires exactly 5 parameters".format(self.__str__()))
            if time > 0:
                iterations = round_value((time / grid.dt)) + 1
            else:
                raise CmdInputError("'{}' time value must be greater than zero".format(self.__str__()))

        if dx < 0 or dy < 0 or dz < 0:
            raise CmdInputError("'{}' the step size should not be less than zero".format(self.__str__()))
        if dx < 1 or dy < 1 or dz < 1:
            raise CmdInputError("'{}' the step size should not be less than the spatial discretisation".format(self.__str__()))
        if iterations <= 0 or iterations > grid.iterations:
            raise CmdInputError("'{}' time value is not valid".format(self.__str__()))

        # Replace with old style snapshots if there are subgrids
        if grid.subgrids:
            from .snapshot_subgrid import Snapshot as SnapshotSub
            s = SnapshotSub(xs, ys, zs, xf, yf, zf, dx, dy, dz, iterations, filename)
        else:
            s = SnapshotUser(xs, ys, zs, xf, yf, zf, dx, dy, dz, iterations, filename)

        if config.is_messages():
            print('Snapshot from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, discretisation {:g}m, {:g}m, {:g}m, at {:g} secs with filename {} created.'.format(xs * grid.dx, ys * grid.dy, zs * grid.dz, xf * grid.dx, yf * grid.dy, zf * grid.dz, dx * grid.dx, dy * grid.dy, dz * grid.dz, s.time * grid.dt, s.basefilename))

        grid.snapshots.append(s)

class Material(UserObjectMulti):
    """Material User Object."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 8
        self.hash = '#material'

    def create(self, grid, uip):
        try:
            er = self.kwargs['er']
            se = self.kwargs['se']
            mr = self.kwargs['mr']
            sm = self.kwargs['sm']
            material_id = self.kwargs['id']
        except KeyError:
            raise CmdInputError('{} requires exactly five parameters'.format(self.params_str()))

        if er < 1:
            raise CmdInputError('{} requires a positive value of one or greater for static (DC) permittivity'.format(self.__str__()))
        if se != 'inf':
            se = float(se)
            if se < 0:
                raise CmdInputError('{} requires a positive value for conductivity'.format(self.__str__()))
        else:
            se = float('inf')
        if mr < 1:
            raise CmdInputError('{} requires a positive value of one or greater for permeability'.format(self.__str__()))
        if sm < 0:
            raise CmdInputError('{} requires a positive value for magnetic conductivity'.format(self.__str__()))
        if any(x.ID == material_id for x in grid.materials):
            raise CmdInputError('{} with ID {} already exists'.format(material_id).format(self.__str__()))

        # Create a new instance of the Material class material (start index after pec & free_space)
        m = MaterialUser(len(grid.materials), material_id)
        m.er = er
        m.se = se
        m.mr = mr
        m.sm = sm

        # Set material averaging to False if infinite conductivity, i.e. pec
        if m.se == float('inf'):
            m.averagable = False

        if config.is_messages():
            tqdm.write('Material {} with eps_r={:g}, sigma={:g} S/m; mu_r={:g}, sigma*={:g} Ohm/m created.'.format(m.ID, m.er, m.se, m.mr, m.sm))

        # Append the new material object to the materials list
        grid.materials.append(m)


class AddDebyeDispersion(UserObjectMulti):
    """Material User Object."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 9
        self.hash = '#add_dispersion_debye'

    def create(self, grid, uip):

        try:
            poles = self.kwargs['n_poles']
            er_delta = self.kwargs['er_delta']
            tau = self.kwargs['tau']
            material_ids = self.kwargs['material_ids']

        except KeyError:
            raise CmdInputError(self.__str__() + ' requires at least four parameters')

        if poles < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for number of poles')

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

        for material in materials:
            material.type = 'debye'
            material.poles = poles
            material.averagable = False
            for i in range(0, poles):
                # N.B Not checking if relaxation times are greater than time-step
                if tau[i] > 0:
                    material.deltaer.append(er_delta[i])
                    material.tau.append(tau[i])
                else:
                    raise CmdInputError(self.__str__() + ' requires positive values for the permittivity difference.')
            if material.poles > MaterialUser.maxpoles:
                MaterialUser.maxpoles = material.poles

            if config.is_messages():
                tqdm.write('Debye disperion added to {} with delta_eps_r={}, and tau={} secs created.'.format(material.ID, ', '.join('%4.2f' % deltaer for deltaer in material.deltaer), ', '.join('%4.3e' % tau for tau in material.tau)))


class AddLorentzDispersion(UserObjectMulti):
    """Material User Object."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 10
        self.hash = '#add_dispersion_lorentz'

    def create(self, grid, uip):

        try:
            poles = self.kwargs['n_poles']
            er_delta = self.kwargs['er_delta']
            tau = self.kwargs['omega']
            alpha = self.kwargs['delta']
            material_ids = self.kwargs['material_ids']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires at least five parameters')

        if poles < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for number of poles')

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

        for material in materials:
            material.type = 'lorentz'
            material.poles = poles
            material.averagable = False
            for i in range(0, poles):
                if er_delta[i] > 0 and tau[i] > grid.dt and alpha[i] > grid.dt:
                    material.deltaer.append(er_delta[i])
                    material.tau.append(tau[i])
                    material.alpha.append(alpha[i])
                else:
                    raise CmdInputError(self.__str__() + ' requires positive values for the permittivity difference and frequencies, and associated times that are greater than the time step for the model.')
            if material.poles > MaterialUser.maxpoles:
                MaterialUser.maxpoles = material.poles

            if config.is_messages():
                tqdm.write('Lorentz disperion added to {} with delta_eps_r={}, omega={} secs, and gamma={} created.'.format(material.ID, ', '.join('%4.2f' % deltaer for deltaer in material.deltaer), ', '.join('%4.3e' % tau for tau in material.tau), ', '.join('%4.3e' % alpha for alpha in material.alpha)))


class AddDrudeDispersion(UserObjectMulti):
    """Material User Object."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 11
        self.hash = '#add_dispersion_Drude'

    def create(self, grid, uip):

        try:
            poles = self.kwargs['n_poles']
            tau = self.kwargs['tau']
            alpha = self.kwargs['alpha']
            material_ids = self.kwargs['material_ids']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires at least four parameters')
        if poles < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for number of poles')

        # Look up requested materials in existing list of material instances
        materials = [y for x in material_ids for y in grid.materials if y.ID == x]

        if len(materials) != len(material_ids):
            notfound = [x for x in material_ids if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

        for material in materials:
            material.type = 'drude'
            material.poles = poles
            material.averagable = False
            for i in range(0, poles):
                if tau[i] > 0 and alpha[i] > grid.dt:
                    material.tau.append(tau[i])
                    material.alpha.append(alpha[i])
                else:
                    raise CmdInputError(self.__str__() + ' requires positive values for the frequencies, and associated times that are greater than the time step for the model.')
            if material.poles > MaterialUser.maxpoles:
                MaterialUser.maxpoles = material.poles

            if config.is_messages():
                tqdm.write('Drude disperion added to {} with omega={} secs, and gamma={} secs created.'.format(material.ID, ', '.join('%4.3e' % tau for tau in material.tau), ', '.join('%4.3e' % alpha for alpha in material.alpha)))


class SoilPeplinski(UserObjectMulti):
    """Material User Object."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 12
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
            raise CmdInputError(self.__str__() + ' requires at exactly seven parameters')

        if sand_fraction < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the sand fraction')
        if clay_fraction < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the clay fraction')
        if bulk_density < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the bulk density')
        if sand_density < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the sand particle density')
        if water_fraction_lower < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the lower limit of the water volumetric fraction')
        if water_fraction_upper < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the upper limit of the water volumetric fraction')
        if any(x.ID == ID for x in grid.mixingmodels):
            raise CmdInputError(self.__str__() + ' with ID {} already exists'.format(ID))

        # Create a new instance of the Material class material (start index after pec & free_space)
        s = PeplinskiSoilUser(ID, sand_fraction, clay_fraction, bulk_density, sand_density, (water_fraction_lower, water_fraction_upper))

        if config.is_messages():
            print('Mixing model (Peplinski) used to create {} with sand fraction {:g}, clay fraction {:g}, bulk density {:g}g/cm3, sand particle density {:g}g/cm3, and water volumetric fraction {:g} to {:g} created.'.format(s.ID, s.S, s.C, s.rb, s.rs, s.mu[0], s.mu[1]))

        # Append the new material object to the materials list
        grid.mixingmodels.append(s)


class GeometryView(UserObjectMulti):
    """Material User Object."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 18
        self.hash = '#geometry_view'
        self.multi_grid = False

    def geometry_view_constructor(self, grid, output_type):
        # Check if user want geometry output for all grids
        try:
            self.kwargs['multi_grid']
            # there is no voxel output for multi grid output
            if isinstance(grid, SubGridBase):
                    raise CmdInputError("'{}' Do not add multi_grid output to subgrid user object. Please add to Scene".format(self.__str__()))
            if output_type == 'n':
                raise CmdInputError("'{}' Voxel output type (n) is not supported for multigrid output :(".format(self.__str__()))
            # Change constructor to the multi grid output
            from .geometry_outputs import GeometryViewFineMultiGrid as GeometryViewUser
            self.multi_grid = True
        except KeyError:
            self.multi_grid = False
            from .geometry_outputs import GeometryView as GeometryViewUser

        return GeometryViewUser

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            dl = self.kwargs['dl']
            output_type = self.kwargs['output_type'].lower()
            filename = self.kwargs['filename']
        except KeyError:
            raise CmdInputError("'{}'  requires exactly eleven parameters".format(self.__str__()))

        GeometryViewUser = self.geometry_view_constructor(grid, output_type)

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        dx, dy, dz = uip.discretise_point(dl)

        if dx < 0 or dy < 0 or dz < 0:
            raise CmdInputError("'{}' the step size should not be less than zero".format(self.__str__()))
        if dx > grid.nx or dy > grid.ny or dz > grid.nz:
            raise CmdInputError("'{}' the step size should be less than the domain size".format(self.__str__()))
        if dx < 1 or dy < 1 or dz < 1:
            raise CmdInputError("'{}' the step size should not be less than the spatial discretisation".format(self.__str__()))
        if output_type != 'n' and output_type != 'f':
            raise CmdInputError("'{}' requires type to be either n (normal) or f (fine)".format(self.__str__()))
        if output_type == 'f' and (dx * grid.dx != grid.dx or dy * grid.dy != grid.dy or dz * grid.dz != grid.dz):
            raise CmdInputError("'{}' requires the spatial discretisation for the geometry view to be the same as the model for geometry view of type f (fine)".format(self.__str__()))

        # Set type of geometry file
        if output_type == 'n':
            fileext = '.vti'
        else:
            fileext = '.vtp'

        g = GeometryViewUser(xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, fileext, grid)

        if config.is_messages():
            print('Geometry view from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, discretisation {:g}m, {:g}m, {:g}m, multi_grid {}, grid={}, with filename base {} created.'.format(xs * grid.dx, ys * grid.dy, zs * grid.dz, xf * grid.dx, yf * grid.dy, zf * grid.dz, dx * grid.dx, dy * grid.dy, dz * grid.dz, self.multi_grid, grid.name, g.basefilename))

        # Append the new GeometryView object to the geometry views list
        grid.geometryviews.append(g)


class GeometryObjectsWrite(UserObjectMulti):
    """Material User Object."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 14
        self.hash = '#geometry_objects_write'

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            filename = self.kwargs['filename']
        except KeyError:
            raise CmdInputError("'{}' requires exactly seven parameters".format(self.__str__()))

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        x0, y0, z0 = p1
        x1, y1, z1 = p2

        g = GeometryObjectsUser(x0, y0, z0, x1, y1, z1, filename)

        if config.is_messages():
            print('Geometry objects in the volume from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, will be written to {}, with materials written to {}'.format(p1[0] * grid.dx, p1[1] * grid.dy, p1[2] * grid.dz, p2[0] * grid.dx, p2[1] * grid.dy, p2[2] * grid.dz, g.filename, g.materialsfilename))

        # Append the new GeometryView object to the geometry objects to write list
        grid.geometryobjectswrite.append(g)


class PMLCFS(UserObjectMulti):
    """Material User Object."""

    count = 0

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 15
        self.hash = '#pml_cfs'
        PMLCFS.count += 1
        if PMLCFS.count == 2:
            raise CmdInputError(self.__str__() + ' can only be used up to two times, for up to a 2nd order PML')

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
            raise CmdInputError(self.__str__() + ' requires exactly twelve parameters')

        if alphascalingprofile not in CFSParameter.scalingprofiles.keys() or kappascalingprofile not in CFSParameter.scalingprofiles.keys() or sigmascalingprofile not in CFSParameter.scalingprofiles.keys():
            raise CmdInputError(self.__str__() + ' must have scaling type {}'.format(','.join(CFSParameter.scalingprofiles.keys())))
        if alphascalingdirection not in CFSParameter.scalingdirections or kappascalingdirection not in CFSParameter.scalingdirections or sigmascalingdirection not in CFSParameter.scalingdirections:
            raise CmdInputError(self.__str__() + ' must have scaling type {}'.format(','.join(CFSParameter.scalingdirections)))
        if float(alphamin) < 0 or float(alphamax) < 0 or float(kappamin) < 0 or float(kappamax) < 0 or float(sigmamin) < 0:
            raise CmdInputError(self.__str__() + ' minimum and maximum scaling values must be greater than zero')
        if float(kappamin) < 1:
            raise CmdInputError(self.__str__() + ' minimum scaling value for kappa must be greater than or equal to one')

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

        if config.is_messages():
            print('PML CFS parameters: alpha (scaling: {}, scaling direction: {}, min: {:g}, max: {:g}), kappa (scaling: {}, scaling direction: {}, min: {:g}, max: {:g}), sigma (scaling: {}, scaling direction: {}, min: {:g}, max: {}) created.'.format(cfsalpha.scalingprofile, cfsalpha.scalingdirection, cfsalpha.min, cfsalpha.max, cfskappa.scalingprofile, cfskappa.scalingdirection, cfskappa.min, cfskappa.max, cfssigma.scalingprofile, cfssigma.scalingdirection, cfssigma.min, cfssigma.max))

        grid.cfs.append(cfs)

class Subgrid(UserObjectMulti):

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
            raise Exception('This Object is Unknown to gprMax')

class SubgridHSG(UserObjectMulti):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
