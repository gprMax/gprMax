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

from colorama import init
from colorama import Fore
from colorama import Style
init()
import numpy as np
from tqdm import tqdm

from gprMax.constants import z0
from gprMax.constants import floattype
from gprMax.exceptions import CmdInputError
from gprMax.geometry_outputs import GeometryView
from gprMax.geometry_outputs import GeometryObjects
from gprMax.materials import Material
from gprMax.materials import PeplinskiSoil
from gprMax.pml import CFSParameter
from gprMax.pml import CFS
from gprMax.receivers import Rx
from gprMax.snapshots import Snapshot
from gprMax.sources import VoltageSource
from gprMax.sources import HertzianDipole
from gprMax.sources import MagneticDipole
from gprMax.sources import TransmissionLine
from gprMax.utilities import round_value
from gprMax.waveforms import Waveform


def process_multicmds(multicmds, G):
    """
    Checks the validity of command parameters and creates instances of
        classes of parameters.

    Args:
        multicmds (dict): Commands that can have multiple instances in the model.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    # Check if coordinates are within the bounds of the grid
    def check_coordinates(x, y, z, name=''):
        try:
            G.within_bounds(x=x, y=y, z=z)
        except ValueError as err:
            s = "'{}: {} ' {} {}-coordinate is not within the model domain".format(cmdname, ' '.join(tmp), name, err.args[0])
            raise CmdInputError(s)

    # Waveform definitions
    cmdname = '#waveform'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 4:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly four parameters')
            if tmp[0].lower() not in Waveform.types:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' must have one of the following types {}'.format(','.join(Waveform.types)))
            if float(tmp[2]) <= 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires an excitation frequency value of greater than zero')
            if any(x.ID == tmp[3] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' with ID {} already exists'.format(tmp[3]))

            w = Waveform()
            w.ID = tmp[3]
            w.type = tmp[0].lower()
            w.amp = float(tmp[1])
            w.freq = float(tmp[2])

            if G.messages:
                print('Waveform {} of type {} with maximum amplitude scaling {:g}, frequency {:g}Hz created.'.format(w.ID, w.type, w.amp, w.freq))

            G.waveforms.append(w)

    # Voltage source
    cmdname = '#voltage_source'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least six parameters')

            # Check polarity & position parameters
            polarisation = tmp[0].lower()
            if polarisation not in ('x', 'y', 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x, y, or z')
            if '2D TMx' in G.mode and (polarisation == 'y' or polarisation == 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x in 2D TMx mode')
            elif '2D TMy' in G.mode and (polarisation == 'x' or polarisation == 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be y in 2D TMy mode')
            elif '2D TMz' in G.mode and (polarisation == 'x' or polarisation == 'y'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be z in 2D TMz mode')

            xcoord = G.calculate_coord('x', tmp[1])
            ycoord = G.calculate_coord('y', tmp[2])
            zcoord = G.calculate_coord('z', tmp[3])
            resistance = float(tmp[4])

            check_coordinates(xcoord, ycoord, zcoord)
            if xcoord < G.pmlthickness['x0'] or xcoord > G.nx - G.pmlthickness['xmax'] or ycoord < G.pmlthickness['y0'] or ycoord > G.ny - G.pmlthickness['ymax'] or zcoord < G.pmlthickness['z0'] or zcoord > G.nz - G.pmlthickness['zmax']:
                print(Fore.RED + "WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.' + Style.RESET_ALL)
            if resistance < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a source resistance of zero or greater')

            # Check if there is a waveformID in the waveforms list
            if not any(x.ID == tmp[5] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' there is no waveform with the identifier {}'.format(tmp[5]))

            v = VoltageSource()
            v.polarisation = polarisation
            v.xcoord = xcoord
            v.ycoord = ycoord
            v.zcoord = zcoord
            v.ID = v.__class__.__name__ + '(' + str(v.xcoord) + ',' + str(v.ycoord) + ',' + str(v.zcoord) + ')'
            v.resistance = resistance
            v.waveformID = tmp[5]

            if len(tmp) > 6:
                # Check source start & source remove time parameters
                start = float(tmp[6])
                stop = float(tmp[7])
                if start < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' delay of the initiation of the source should not be less than zero')
                if stop < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' time to remove the source should not be less than zero')
                if stop - start <= 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' duration of the source should not be zero or less')
                v.start = start
                if stop > G.timewindow:
                    v.stop = G.timewindow
                else:
                    v.stop = stop
                startstop = ' start time {:g} secs, finish time {:g} secs '.format(v.start, v.stop)
            else:
                v.start = 0
                v.stop = G.timewindow
                startstop = ' '

            v.calculate_waveform_values(G)

            if G.messages:
                print('Voltage source with polarity {} at {:g}m, {:g}m, {:g}m, resistance {:.1f} Ohms,'.format(v.polarisation, v.xcoord * G.dx, v.ycoord * G.dy, v.zcoord * G.dz, v.resistance) + startstop + 'using waveform {} created.'.format(v.waveformID))

            G.voltagesources.append(v)

    # Hertzian dipole
    cmdname = '#hertzian_dipole'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')

            # Check polarity & position parameters
            polarisation = tmp[0].lower()
            if polarisation not in ('x', 'y', 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x, y, or z')
            if '2D TMx' in G.mode and (polarisation == 'y' or polarisation == 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x in 2D TMx mode')
            elif '2D TMy' in G.mode and (polarisation == 'x' or polarisation == 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be y in 2D TMy mode')
            elif '2D TMz' in G.mode and (polarisation == 'x' or polarisation == 'y'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be z in 2D TMz mode')

            xcoord = G.calculate_coord('x', tmp[1])
            ycoord = G.calculate_coord('y', tmp[2])
            zcoord = G.calculate_coord('z', tmp[3])
            check_coordinates(xcoord, ycoord, zcoord)
            if xcoord < G.pmlthickness['x0'] or xcoord > G.nx - G.pmlthickness['xmax'] or ycoord < G.pmlthickness['y0'] or ycoord > G.ny - G.pmlthickness['ymax'] or zcoord < G.pmlthickness['z0'] or zcoord > G.nz - G.pmlthickness['zmax']:
                print(Fore.RED + "WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.' + Style.RESET_ALL)

            # Check if there is a waveformID in the waveforms list
            if not any(x.ID == tmp[4] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' there is no waveform with the identifier {}'.format(tmp[4]))

            h = HertzianDipole()
            h.polarisation = polarisation

            # Set length of dipole to grid size in polarisation direction
            if h.polarisation == 'x':
                h.dl = G.dx
            elif h.polarisation == 'y':
                h.dl = G.dy
            elif h.polarisation == 'z':
                h.dl = G.dz

            h.xcoord = xcoord
            h.ycoord = ycoord
            h.zcoord = zcoord
            h.xcoordorigin = xcoord
            h.ycoordorigin = ycoord
            h.zcoordorigin = zcoord
            h.ID = h.__class__.__name__ + '(' + str(h.xcoord) + ',' + str(h.ycoord) + ',' + str(h.zcoord) + ')'
            h.waveformID = tmp[4]

            if len(tmp) > 5:
                # Check source start & source remove time parameters
                start = float(tmp[5])
                stop = float(tmp[6])
                if start < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' delay of the initiation of the source should not be less than zero')
                if stop < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' time to remove the source should not be less than zero')
                if stop - start <= 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' duration of the source should not be zero or less')
                h.start = start
                if stop > G.timewindow:
                    h.stop = G.timewindow
                else:
                    h.stop = stop
                startstop = ' start time {:g} secs, finish time {:g} secs '.format(h.start, h.stop)
            else:
                h.start = 0
                h.stop = G.timewindow
                startstop = ' '

            h.calculate_waveform_values(G)

            if G.messages:
                if G.mode == '2D':
                    print('Hertzian dipole is a line source in 2D with polarity {} at {:g}m, {:g}m, {:g}m,'.format(h.polarisation, h.xcoord * G.dx, h.ycoord * G.dy, h.zcoord * G.dz) + startstop + 'using waveform {} created.'.format(h.waveformID))
                else:
                    print('Hertzian dipole with polarity {} at {:g}m, {:g}m, {:g}m,'.format(h.polarisation, h.xcoord * G.dx, h.ycoord * G.dy, h.zcoord * G.dz) + startstop + 'using waveform {} created.'.format(h.waveformID))

            G.hertziandipoles.append(h)

    # Magnetic dipole
    cmdname = '#magnetic_dipole'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')

            # Check polarity & position parameters
            polarisation = tmp[0].lower()
            if polarisation not in ('x', 'y', 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x, y, or z')
            if '2D TMx' in G.mode and (polarisation == 'y' or polarisation == 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x in 2D TMx mode')
            elif '2D TMy' in G.mode and (polarisation == 'x' or polarisation == 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be y in 2D TMy mode')
            elif '2D TMz' in G.mode and (polarisation == 'x' or polarisation == 'y'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be z in 2D TMz mode')

            xcoord = G.calculate_coord('x', tmp[1])
            ycoord = G.calculate_coord('y', tmp[2])
            zcoord = G.calculate_coord('z', tmp[3])
            check_coordinates(xcoord, ycoord, zcoord)
            if xcoord < G.pmlthickness['x0'] or xcoord > G.nx - G.pmlthickness['xmax'] or ycoord < G.pmlthickness['y0'] or ycoord > G.ny - G.pmlthickness['ymax'] or zcoord < G.pmlthickness['z0'] or zcoord > G.nz - G.pmlthickness['zmax']:
                print(Fore.RED + "WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.' + Style.RESET_ALL)

            # Check if there is a waveformID in the waveforms list
            if not any(x.ID == tmp[4] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' there is no waveform with the identifier {}'.format(tmp[4]))

            m = MagneticDipole()
            m.polarisation = polarisation
            m.xcoord = xcoord
            m.ycoord = ycoord
            m.zcoord = zcoord
            m.xcoordorigin = xcoord
            m.ycoordorigin = ycoord
            m.zcoordorigin = zcoord
            m.ID = m.__class__.__name__ + '(' + str(m.xcoord) + ',' + str(m.ycoord) + ',' + str(m.zcoord) + ')'
            m.waveformID = tmp[4]

            if len(tmp) > 5:
                # Check source start & source remove time parameters
                start = float(tmp[5])
                stop = float(tmp[6])
                if start < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' delay of the initiation of the source should not be less than zero')
                if stop < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' time to remove the source should not be less than zero')
                if stop - start <= 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' duration of the source should not be zero or less')
                m.start = start
                if stop > G.timewindow:
                    m.stop = G.timewindow
                else:
                    m.stop = stop
                startstop = ' start time {:g} secs, finish time {:g} secs '.format(m.start, m.stop)
            else:
                m.start = 0
                m.stop = G.timewindow
                startstop = ' '

            m.calculate_waveform_values(G)

            if G.messages:
                print('Magnetic dipole with polarity {} at {:g}m, {:g}m, {:g}m,'.format(m.polarisation, m.xcoord * G.dx, m.ycoord * G.dy, m.zcoord * G.dz) + startstop + 'using waveform {} created.'.format(m.waveformID))

            G.magneticdipoles.append(m)

    # Transmission line
    cmdname = '#transmission_line'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least six parameters')

            # Warn about using a transmission line on GPU
            if G.gpu is not None:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' A #transmission_line cannot currently be used with GPU solving. Consider using a #voltage_source instead.')

            # Check polarity & position parameters
            polarisation = tmp[0].lower()
            if polarisation not in ('x', 'y', 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x, y, or z')
            if '2D TMx' in G.mode and (polarisation == 'y' or polarisation == 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x in 2D TMx mode')
            elif '2D TMy' in G.mode and (polarisation == 'x' or polarisation == 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be y in 2D TMy mode')
            elif '2D TMz' in G.mode and (polarisation == 'x' or polarisation == 'y'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be z in 2D TMz mode')

            xcoord = G.calculate_coord('x', tmp[1])
            ycoord = G.calculate_coord('y', tmp[2])
            zcoord = G.calculate_coord('z', tmp[3])
            resistance = float(tmp[4])

            check_coordinates(xcoord, ycoord, zcoord)
            if xcoord < G.pmlthickness['x0'] or xcoord > G.nx - G.pmlthickness['xmax'] or ycoord < G.pmlthickness['y0'] or ycoord > G.ny - G.pmlthickness['ymax'] or zcoord < G.pmlthickness['z0'] or zcoord > G.nz - G.pmlthickness['zmax']:
                print(Fore.RED + "WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.' + Style.RESET_ALL)
            if resistance <= 0 or resistance >= z0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a resistance greater than zero and less than the impedance of free space, i.e. 376.73 Ohms')

            # Check if there is a waveformID in the waveforms list
            if not any(x.ID == tmp[5] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' there is no waveform with the identifier {}'.format(tmp[5]))

            t = TransmissionLine(G)
            t.polarisation = polarisation
            t.xcoord = xcoord
            t.ycoord = ycoord
            t.zcoord = zcoord
            t.ID = t.__class__.__name__ + '(' + str(t.xcoord) + ',' + str(t.ycoord) + ',' + str(t.zcoord) + ')'
            t.resistance = resistance
            t.waveformID = tmp[5]

            if len(tmp) > 6:
                # Check source start & source remove time parameters
                start = float(tmp[6])
                stop = float(tmp[7])
                if start < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' delay of the initiation of the source should not be less than zero')
                if stop < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' time to remove the source should not be less than zero')
                if stop - start <= 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' duration of the source should not be zero or less')
                t.start = start
                if stop > G.timewindow:
                    t.stop = G.timewindow
                else:
                    t.stop = stop
                startstop = ' start time {:g} secs, finish time {:g} secs '.format(t.start, t.stop)
            else:
                t.start = 0
                t.stop = G.timewindow
                startstop = ' '

            t.calculate_waveform_values(G)
            t.calculate_incident_V_I(G)

            if G.messages:
                print('Transmission line with polarity {} at {:g}m, {:g}m, {:g}m, resistance {:.1f} Ohms,'.format(t.polarisation, t.xcoord * G.dx, t.ycoord * G.dy, t.zcoord * G.dz, t.resistance) + startstop + 'using waveform {} created.'.format(t.waveformID))

            G.transmissionlines.append(t)

    # Receiver
    cmdname = '#rx'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 3 and len(tmp) < 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')

            # Check position parameters
            xcoord = round_value(float(tmp[0]) / G.dx)
            ycoord = round_value(float(tmp[1]) / G.dy)
            zcoord = round_value(float(tmp[2]) / G.dz)
            check_coordinates(xcoord, ycoord, zcoord)
            if xcoord < G.pmlthickness['x0'] or xcoord > G.nx - G.pmlthickness['xmax'] or ycoord < G.pmlthickness['y0'] or ycoord > G.ny - G.pmlthickness['ymax'] or zcoord < G.pmlthickness['z0'] or zcoord > G.nz - G.pmlthickness['zmax']:
                print(Fore.RED + "WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.' + Style.RESET_ALL)

            r = Rx()
            r.xcoord = xcoord
            r.ycoord = ycoord
            r.zcoord = zcoord
            r.xcoordorigin = xcoord
            r.ycoordorigin = ycoord
            r.zcoordorigin = zcoord

            # If no ID or outputs are specified, use default
            if len(tmp) == 3:
                r.ID = r.__class__.__name__ + '(' + str(r.xcoord) + ',' + str(r.ycoord) + ',' + str(r.zcoord) + ')'
                for key in Rx.defaultoutputs:
                    r.outputs[key] = np.zeros(G.iterations, dtype=floattype)
            else:
                r.ID = tmp[3]
                # Get allowable outputs
                if G.gpu is not None:
                    allowableoutputs = Rx.gpu_allowableoutputs
                else:
                    allowableoutputs = Rx.allowableoutputs
                # Check and add field output names
                for field in tmp[4::]:
                    if field in allowableoutputs:
                        r.outputs[field] = np.zeros(G.iterations, dtype=floattype)
                    else:
                        raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' contains an output type that is not allowable. Allowable outputs in current context are {}'.format(allowableoutputs))

            if G.messages:
                print('Receiver at {:g}m, {:g}m, {:g}m with output component(s) {} created.'.format(r.xcoord * G.dx, r.ycoord * G.dy, r.zcoord * G.dz, ', '.join(r.outputs)))

            G.rxs.append(r)

    # Receiver array
    cmdname = '#rx_array'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 9:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly nine parameters')

            xs = G.calculate_coord('x', tmp[0])
            ys = G.calculate_coord('y', tmp[1])
            zs = G.calculate_coord('z', tmp[2])

            xf = G.calculate_coord('x', tmp[3])
            yf = G.calculate_coord('y', tmp[4])
            zf = G.calculate_coord('z', tmp[5])

            dx = G.calculate_coord('x', tmp[6])
            dy = G.calculate_coord('y', tmp[7])
            dz = G.calculate_coord('z', tmp[8])

            check_coordinates(xs, ys, zs, name='lower')
            check_coordinates(xf, yf, zf, name='upper')

            if xcoord < G.pmlthickness['x0'] or xcoord > G.nx - G.pmlthickness['xmax'] or ycoord < G.pmlthickness['y0'] or ycoord > G.ny - G.pmlthickness['ymax'] or zcoord < G.pmlthickness['z0'] or zcoord > G.nz - G.pmlthickness['zmax']:
                print(Fore.RED + "WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.' + Style.RESET_ALL)
            if xs > xf or ys > yf or zs > zf:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
            if dx < 0 or dy < 0 or dz < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than zero')
            if dx < 1:
                if dx == 0:
                    dx = 1
                else:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than the spatial discretisation')
            if dy < 1:
                if dy == 0:
                    dy = 1
                else:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than the spatial discretisation')
            if dz < 1:
                if dz == 0:
                    dz = 1
                else:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than the spatial discretisation')

            if G.messages:
                print('Receiver array {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with steps {:g}m, {:g}m, {:g}m'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dy * G.dy, dz * G.dz))

            for x in range(xs, xf + 1, dx):
                for y in range(ys, yf + 1, dy):
                    for z in range(zs, zf + 1, dz):
                        r = Rx()
                        r.xcoord = x
                        r.ycoord = y
                        r.zcoord = z
                        r.xcoordorigin = x
                        r.ycoordorigin = y
                        r.zcoordorigin = z
                        r.ID = r.__class__.__name__ + '(' + str(x) + ',' + str(y) + ',' + str(z) + ')'
                        for key in Rx.defaultoutputs:
                            r.outputs[key] = np.zeros(G.iterations, dtype=floattype)
                        if G.messages:
                            print('  Receiver at {:g}m, {:g}m, {:g}m with output component(s) {} created.'.format(r.xcoord * G.dx, r.ycoord * G.dy, r.zcoord * G.dz, ', '.join(r.outputs)))
                        G.rxs.append(r)

    # Snapshot
    cmdname = '#snapshot'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly eleven parameters')

            xs = G.calculate_coord('x', tmp[0])
            ys = G.calculate_coord('y', tmp[1])
            zs = G.calculate_coord('z', tmp[2])

            xf = G.calculate_coord('x', tmp[3])
            yf = G.calculate_coord('y', tmp[4])
            zf = G.calculate_coord('z', tmp[5])

            dx = G.calculate_coord('x', tmp[6])
            dy = G.calculate_coord('y', tmp[7])
            dz = G.calculate_coord('z', tmp[8])

            # If number of iterations given
            try:
                time = int(tmp[9])
            # If real floating point value given
            except ValueError:
                time = float(tmp[9])
                if time > 0:
                    time = round_value((time / G.dt)) + 1
                else:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' time value must be greater than zero')

            check_coordinates(xs, ys, zs, name='lower')
            check_coordinates(xf, yf, zf, name='upper')

            if xs >= xf or ys >= yf or zs >= zf:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
            if dx < 0 or dy < 0 or dz < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than zero')
            if dx < 1 or dy < 1 or dz < 1:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than the spatial discretisation')
            if time <= 0 or time > G.iterations:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' time value is not valid')

            s = Snapshot(xs, ys, zs, xf, yf, zf, dx, dy, dz, time, tmp[10])

            if G.messages:
                print('Snapshot from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, discretisation {:g}m, {:g}m, {:g}m, at {:g} secs with filename {} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dy * G.dy, dz * G.dz, s.time * G.dt, s.basefilename))

            G.snapshots.append(s)

    # Materials
    cmdname = '#material'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly five parameters')
            if float(tmp[0]) < 1:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value of one or greater for static (DC) permittivity')
            if tmp[1] != 'inf':
                se = float(tmp[1])
                if se < 0:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for conductivity')
            else:
                se = float('inf')
            if float(tmp[2]) < 1:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value of one or greater for permeability')
            if float(tmp[3]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for magnetic conductivity')
            if any(x.ID == tmp[4] for x in G.materials):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' with ID {} already exists'.format(tmp[4]))

            # Create a new instance of the Material class material (start index after pec & free_space)
            m = Material(len(G.materials), tmp[4])
            m.er = float(tmp[0])
            m.se = se
            m.mr = float(tmp[2])
            m.sm = float(tmp[3])

            # Set material averaging to False if infinite conductivity, i.e. pec
            if m.se == float('inf'):
                m.averagable = False

            if G.messages:
                tqdm.write('Material {} with eps_r={:g}, sigma={:g} S/m; mu_r={:g}, sigma*={:g} Ohm/m created.'.format(m.ID, m.er, m.se, m.mr, m.sm))

            # Append the new material object to the materials list
            G.materials.append(m)

    cmdname = '#add_dispersion_debye'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 4:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least four parameters')
            if int(tmp[0]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for number of poles')
            poles = int(tmp[0])
            materialsrequested = tmp[(2 * poles) + 1:len(tmp)]

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            for material in materials:
                material.type = 'debye'
                material.poles = poles
                material.averagable = False
                for pole in range(1, 2 * poles, 2):
                    # N.B Not checking if relaxation times are greater than time-step
                    if float(tmp[pole]) > 0:
                        material.deltaer.append(float(tmp[pole]))
                        material.tau.append(float(tmp[pole + 1]))
                    else:
                        raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires positive values for the permittivity difference.')
                if material.poles > Material.maxpoles:
                    Material.maxpoles = material.poles

                if G.messages:
                    tqdm.write('Debye disperion added to {} with delta_eps_r={}, and tau={} secs created.'.format(material.ID, ', '.join('%4.2f' % deltaer for deltaer in material.deltaer), ', '.join('%4.3e' % tau for tau in material.tau)))

    cmdname = '#add_dispersion_lorentz'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
            if int(tmp[0]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for number of poles')
            poles = int(tmp[0])
            materialsrequested = tmp[(3 * poles) + 1:len(tmp)]

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            for material in materials:
                material.type = 'lorentz'
                material.poles = poles
                material.averagable = False
                for pole in range(1, 3 * poles, 3):
                    if float(tmp[pole]) > 0 and float(tmp[pole + 1]) > G.dt and float(tmp[pole + 2]) > G.dt:
                        material.deltaer.append(float(tmp[pole]))
                        material.tau.append(float(tmp[pole + 1]))
                        material.alpha.append(float(tmp[pole + 2]))
                    else:
                        raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires positive values for the permittivity difference and frequencies, and associated times that are greater than the time step for the model.')
                if material.poles > Material.maxpoles:
                    Material.maxpoles = material.poles

                if G.messages:
                    tqdm.write('Lorentz disperion added to {} with delta_eps_r={}, omega={} secs, and gamma={} created.'.format(material.ID, ', '.join('%4.2f' % deltaer for deltaer in material.deltaer), ', '.join('%4.3e' % tau for tau in material.tau), ', '.join('%4.3e' % alpha for alpha in material.alpha)))

    cmdname = '#add_dispersion_drude'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
            if int(tmp[0]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for number of poles')
            poles = int(tmp[0])
            materialsrequested = tmp[(3 * poles) + 1:len(tmp)]

            # Look up requested materials in existing list of material instances
            materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

            if len(materials) != len(materialsrequested):
                notfound = [x for x in materialsrequested if x not in materials]
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' material(s) {} do not exist'.format(notfound))

            for material in materials:
                material.type = 'drude'
                material.poles = poles
                material.averagable = False
                for pole in range(1, 2 * poles, 2):
                    if float(tmp[pole]) > 0 and float(tmp[pole + 1]) > G.dt:
                        material.tau.append(float(tmp[pole]))
                        material.alpha.append(float(tmp[pole + 1]))
                    else:
                        raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires positive values for the frequencies, and associated times that are greater than the time step for the model.')
                if material.poles > Material.maxpoles:
                    Material.maxpoles = material.poles

                if G.messages:
                    tqdm.write('Drude disperion added to {} with omega={} secs, and gamma={} secs created.'.format(material.ID, ', '.join('%4.3e' % tau for tau in material.tau), ', '.join('%4.3e' % alpha for alpha in material.alpha)))

    cmdname = '#soil_peplinski'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 7:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at exactly seven parameters')
            if float(tmp[0]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for the sand fraction')
            if float(tmp[1]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for the clay fraction')
            if float(tmp[2]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for the bulk density')
            if float(tmp[3]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for the sand particle density')
            if float(tmp[4]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for the lower limit of the water volumetric fraction')
            if float(tmp[5]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for the upper limit of the water volumetric fraction')
            if any(x.ID == tmp[6] for x in G.mixingmodels):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' with ID {} already exists'.format(tmp[6]))

            # Create a new instance of the Material class material (start index after pec & free_space)
            s = PeplinskiSoil(tmp[6], float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3]), (float(tmp[4]), float(tmp[5])))

            if G.messages:
                print('Mixing model (Peplinski) used to create {} with sand fraction {:g}, clay fraction {:g}, bulk density {:g}g/cm3, sand particle density {:g}g/cm3, and water volumetric fraction {:g} to {:g} created.'.format(s.ID, s.S, s.C, s.rb, s.rs, s.mu[0], s.mu[1]))

            # Append the new material object to the materials list
            G.mixingmodels.append(s)

    # Geometry views (creates VTK-based geometry files)
    cmdname = '#geometry_view'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly eleven parameters')

            xs = G.calculate_coord('x', tmp[0])
            ys = G.calculate_coord('y', tmp[1])
            zs = G.calculate_coord('z', tmp[2])

            xf = G.calculate_coord('x', tmp[3])
            yf = G.calculate_coord('y', tmp[4])
            zf = G.calculate_coord('z', tmp[5])

            dx = G.calculate_coord('x', tmp[6])
            dy = G.calculate_coord('y', tmp[7])
            dz = G.calculate_coord('z', tmp[8])

            check_coordinates(xs, ys, zs, name='lower')
            check_coordinates(xf, yf, zf, name='upper')

            if xs >= xf or ys >= yf or zs >= zf:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
            if dx < 0 or dy < 0 or dz < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than zero')
            if dx > G.nx or dy > G.ny or dz > G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should be less than the domain size')
            if dx < 1 or dy < 1 or dz < 1:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than the spatial discretisation')
            if tmp[10].lower() != 'n' and tmp[10].lower() != 'f':
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires type to be either n (normal) or f (fine)')
            if tmp[10].lower() == 'f' and (dx * G.dx != G.dx or dy * G.dy != G.dy or dz * G.dz != G.dz):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires the spatial discretisation for the geometry view to be the same as the model for geometry view of type f (fine)')

            # Set type of geometry file
            if tmp[10].lower() == 'n':
                fileext = '.vti'
            else:
                fileext = '.vtp'

            g = GeometryView(xs, ys, zs, xf, yf, zf, dx, dy, dz, tmp[9], fileext)

            if G.messages:
                print('Geometry view from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, discretisation {:g}m, {:g}m, {:g}m, with filename base {} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dy * G.dy, dz * G.dz, g.basefilename))

            # Append the new GeometryView object to the geometry views list
            G.geometryviews.append(g)

    # Geometry object(s) output
    cmdname = '#geometry_objects_write'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 7:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly seven parameters')

            xs = G.calculate_coord('x', tmp[0])
            ys = G.calculate_coord('y', tmp[1])
            zs = G.calculate_coord('z', tmp[2])

            xf = G.calculate_coord('x', tmp[3])
            yf = G.calculate_coord('y', tmp[4])
            zf = G.calculate_coord('z', tmp[5])

            check_coordinates(xs, ys, zs, name='lower')
            check_coordinates(xf, yf, zf, name='upper')

            if xs >= xf or ys >= yf or zs >= zf:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')

            g = GeometryObjects(xs, ys, zs, xf, yf, zf, tmp[6])

            if G.messages:
                print('Geometry objects in the volume from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, will be written to {}, with materials written to {}'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, g.filename, g.materialsfilename))

            # Append the new GeometryView object to the geometry objects to write list
            G.geometryobjectswrite.append(g)

    # Complex frequency shifted (CFS) PML parameter
    cmdname = '#pml_cfs'
    if multicmds[cmdname] is not None:
        if len(multicmds[cmdname]) > 2:
            raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' can only be used up to two times, for up to a 2nd order PML')
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 12:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly twelve parameters')
            if tmp[0] not in CFSParameter.scalingprofiles.keys() or tmp[4] not in CFSParameter.scalingprofiles.keys() or tmp[8] not in CFSParameter.scalingprofiles.keys():
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' must have scaling type {}'.format(','.join(CFSParameter.scalingprofiles.keys())))
            if tmp[1] not in CFSParameter.scalingdirections or tmp[5] not in CFSParameter.scalingdirections or tmp[9] not in CFSParameter.scalingdirections:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' must have scaling type {}'.format(','.join(CFSParameter.scalingdirections)))
            if float(tmp[2]) < 0 or float(tmp[3]) < 0 or float(tmp[6]) < 0 or float(tmp[7]) < 0 or float(tmp[10]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' minimum and maximum scaling values must be positive')
            if float(tmp[6]) < 1 and G.pmlformulation == 'HORIPML':
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' minimum scaling value for kappa must be greater than or equal to one')

            cfsalpha = CFSParameter()
            cfsalpha.ID = 'alpha'
            cfsalpha.scalingprofile = tmp[0]
            cfsalpha.scalingdirection = tmp[1]
            cfsalpha.min = float(tmp[2])
            cfsalpha.max = float(tmp[3])
            cfskappa = CFSParameter()
            cfskappa.ID = 'kappa'
            cfskappa.scalingprofile = tmp[4]
            cfskappa.scalingdirection = tmp[5]
            cfskappa.min = float(tmp[6])
            cfskappa.max = float(tmp[7])
            cfssigma = CFSParameter()
            cfssigma.ID = 'sigma'
            cfssigma.scalingprofile = tmp[8]
            cfssigma.scalingdirection = tmp[9]
            cfssigma.min = float(tmp[10])
            if tmp[11] == 'None':
                cfssigma.max = None
            else:
                cfssigma.max = float(tmp[11])
            cfs = CFS()
            cfs.alpha = cfsalpha
            cfs.kappa = cfskappa
            cfs.sigma = cfssigma

            if G.messages:
                print('PML CFS parameters: alpha (scaling: {}, scaling direction: {}, min: {:g}, max: {:g}), kappa (scaling: {}, scaling direction: {}, min: {:g}, max: {:g}), sigma (scaling: {}, scaling direction: {}, min: {:g}, max: {}) created.'.format(cfsalpha.scalingprofile, cfsalpha.scalingdirection, cfsalpha.min, cfsalpha.max, cfskappa.scalingprofile, cfskappa.scalingdirection, cfskappa.min, cfskappa.max, cfssigma.scalingprofile, cfssigma.scalingdirection, cfssigma.min, cfssigma.max))

            G.cfs.append(cfs)
