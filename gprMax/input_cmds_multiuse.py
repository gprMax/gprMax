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

from gprMax.constants import z0
from gprMax.exceptions import CmdInputError
from gprMax.geometry_views import GeometryView
from gprMax.materials import Material, PeplinskiSoil
from gprMax.pml import CFSParameter, CFS
from gprMax.receivers import Rx
from gprMax.snapshots import Snapshot
from gprMax.sources import VoltageSource, HertzianDipole, MagneticDipole, TransmissionLine
from gprMax.utilities import round_value
from gprMax.waveforms import Waveform


def process_multicmds(multicmds, G):
    """Checks the validity of command parameters and creates instances of classes of parameters.
        
    Args:
        multicmds (dict): Commands that can have multiple instances in the model.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    # Waveform definitions
    cmdname = '#waveform'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 4:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly four parameters')
            if tmp[0].lower() not in Waveform.waveformtypes:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' must have one of the following types {}'.format(','.join(Waveform.waveformtypes)))
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
                print('Waveform {} of type {} with amplitude {:g}, frequency {:g}Hz created.'.format(w.ID, w.type, w.amp, w.freq))
            
            G.waveforms.append(w)


    # Voltage source
    cmdname = '#voltage_source'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least six parameters')
            
            # Check polarity & position parameters
            if tmp[0].lower() not in ('x', 'y', 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x, y, or z')
            xcoord = round_value(float(tmp[1])/G.dx)
            ycoord = round_value(float(tmp[2])/G.dy)
            zcoord = round_value(float(tmp[3])/G.dz)
            resistance = float(tmp[4])
            if xcoord < 0 or xcoord >= G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' x-coordinate is not within the model domain')
            if ycoord < 0 or ycoord >= G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' y-coordinate is not within the model domain')
            if zcoord < 0 or zcoord >= G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' z-coordinate is not within the model domain')
            if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
                print("WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.')
            if resistance < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a source resistance of zero or greater')
                    
            # Check if there is a waveformID in the waveforms list
            if not any(x.ID == tmp[5] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' there is no waveform with the identifier {}'.format(tmp[5]))
            
            v = VoltageSource()
            v.polarisation= tmp[0]
            v.xcoord = xcoord
            v.ycoord = ycoord
            v.zcoord = zcoord
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
                startstop = ' start time {:g} secs, finish time {:g} secs '.format(v.start, v.stop)
            else:
                v.start = 0
                v.stop = G.timewindow
                startstop = ' '
            
            if G.messages:
                print('Voltage source with polarity {} at {:g}m, {:g}m, {:g}m, resistance {:.1f} Ohms,'.format(v.polarisation, v.xcoord * G.dx, v.ycoord * G.dy, v.zcoord * G.dz, v.resistance) + startstop + 'using waveform {} created.'.format(v.waveformID))
            
            G.voltagesources.append(v)


    # Hertzian dipole
    cmdname = '#hertzian_dipole'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
            
            # Check polarity & position parameters
            if tmp[0].lower() not in ('x', 'y', 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x, y, or z')
            xcoord = round_value(float(tmp[1])/G.dx)
            ycoord = round_value(float(tmp[2])/G.dy)
            zcoord = round_value(float(tmp[3])/G.dz)
            if xcoord < 0 or xcoord >= G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' x-coordinate is not within the model domain')
            if ycoord < 0 or ycoord >= G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' y-coordinate is not within the model domain')
            if zcoord < 0 or zcoord >= G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' z-coordinate is not within the model domain')
            if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
                print("WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.')
                    
            # Check if there is a waveformID in the waveforms list
            if not any(x.ID == tmp[4] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' there is no waveform with the identifier {}'.format(tmp[4]))
            
            h = HertzianDipole()
            h.polarisation = tmp[0]
            h.xcoord = xcoord
            h.ycoord = ycoord
            h.zcoord = zcoord
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
                startstop = ' start time {:g} secs, finish time {:g} secs '.format(h.start, h.stop)
            else:
                h.start = 0
                h.stop = G.timewindow
                startstop = ' '
            
            if G.messages:
                print('Hertzian dipole with polarity {} at {:g}m, {:g}m, {:g}m,'.format(h.polarisation, h.xcoord * G.dx, h.ycoord * G.dy, h.zcoord * G.dz) + startstop + 'using waveform {} created.'.format(h.waveformID))
            
            G.hertziandipoles.append(h)


    # Magnetic dipole
    cmdname = '#magnetic_dipole'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
            
            # Check polarity & position parameters
            if tmp[0].lower() not in ('x', 'y', 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x, y, or z')
            xcoord = round_value(float(tmp[1])/G.dx)
            ycoord = round_value(float(tmp[2])/G.dy)
            zcoord = round_value(float(tmp[3])/G.dz)
            if xcoord < 0 or xcoord >= G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' x-coordinate is not within the model domain')
            if ycoord < 0 or ycoord >= G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' y-coordinate is not within the model domain')
            if zcoord < 0 or zcoord >= G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' z-coordinate is not within the model domain')
            if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
                print("WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.')
                    
            # Check if there is a waveformID in the waveforms list
            if not any(x.ID == tmp[4] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' there is no waveform with the identifier {}'.format(tmp[4]))
            
            m = MagneticDipole()
            m.polarisation = tmp[0]
            m.xcoord = xcoord
            m.ycoord = ycoord
            m.zcoord = zcoord
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
                startstop = ' start time {:g} secs, finish time {:g} secs '.format(m.start, m.stop)
            else:
                m.start = 0
                m.stop = G.timewindow
                startstop = ' '
            
            if G.messages:
                print('Magnetic dipole with polarity {} at {:g}m, {:g}m, {:g}m,'.format(m.polarisation, m.xcoord * G.dx, m.ycoord * G.dy, m.zcoord * G.dz) + startstop + 'using waveform {} created.'.format(m.waveformID))
            
            G.magneticdipoles.append(m)


    # Transmission line
    cmdname = '#transmission_line'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least six parameters')
            
            # Check polarity & position parameters
            if tmp[0].lower() not in ('x', 'y', 'z'):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' polarisation must be x, y, or z')
            xcoord = round_value(float(tmp[1])/G.dx)
            ycoord = round_value(float(tmp[2])/G.dy)
            zcoord = round_value(float(tmp[3])/G.dz)
            resistance = float(tmp[4])
            if xcoord < 0 or xcoord >= G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' x-coordinate is not within the model domain')
            if ycoord < 0 or ycoord >= G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' y-coordinate is not within the model domain')
            if zcoord < 0 or zcoord >= G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' z-coordinate is not within the model domain')
            if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
                print("WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.')
            if resistance <= 0 or resistance > z0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a resistance greater than zero and less than the impedance of free space, i.e. 376.73 Ohms')
                    
            # Check if there is a waveformID in the waveforms list
            if not any(x.ID == tmp[5] for x in G.waveforms):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' there is no waveform with the identifier {}'.format(tmp[4]))
            
            t = TransmissionLine(G)
            t.polarisation = tmp[0]
            t.xcoord = xcoord
            t.ycoord = ycoord
            t.zcoord = zcoord
            t.resistance = resistance
            t.waveformID = tmp[5]
            t.calculate_incident_V_I(G)
            
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
                startstop = ' start time {:g} secs, finish time {:g} secs '.format(t.start, t.stop)
            else:
                t.start = 0
                t.stop = G.timewindow
                startstop = ' '
            
            if G.messages:
                print('Transmission line with polarity {} at {:g}m, {:g}m, {:g}m, resistance {:.1f} Ohms,'.format(t.polarisation, t.xcoord * G.dx, t.ycoord * G.dy, t.zcoord * G.dz, t.resistance) + startstop + 'using waveform {} created.'.format(t.waveformID))
            
            G.transmissionlines.append(t)


    # Receiver
    cmdname = '#rx'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 3 and len(tmp) < 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')

            # Check position parameters
            xcoord = round_value(float(tmp[0])/G.dx)
            ycoord = round_value(float(tmp[1])/G.dy)
            zcoord = round_value(float(tmp[2])/G.dz)
            if xcoord < 0 or xcoord >= G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' x-coordinate is not within the model domain')
            if ycoord < 0 or ycoord >= G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' y-coordinate is not within the model domain')
            if zcoord < 0 or zcoord >= G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' z-coordinate is not within the model domain')
            if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
                print("WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.')
            
            r = Rx(xcoord=xcoord, ycoord=ycoord, zcoord=zcoord)
            
            # If no ID or outputs are specified, use default i.e Ex, Ey, Ez, Hx, Hy, Hz, Ix, Iy, Iz
            if len(tmp) == 3:
                r.outputs = Rx.availableoutputs[0:9]
            else:
                r.ID = tmp[3]
                # Check and add field output names
                for field in tmp[4::]:
                    if field in Rx.availableoutputs:
                        r.outputs.append(field)
                    else:
                        raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' contains an output type that is not available')
            
            if G.messages:
                print('Receiver at {:g}m, {:g}m, {:g}m with output(s) {} created.'.format(r.xcoord * G.dx, r.ycoord * G.dy, r.zcoord * G.dz, ', '.join(r.outputs)))
            
            G.rxs.append(r)


    # Receiver box
    cmdname = '#rx_box'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 9:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly nine parameters')

            xs = round_value(float(tmp[0])/G.dx)
            xf = round_value(float(tmp[3])/G.dx)
            ys = round_value(float(tmp[1])/G.dy)
            yf = round_value(float(tmp[4])/G.dy)
            zs = round_value(float(tmp[2])/G.dz)
            zf = round_value(float(tmp[5])/G.dz)
            dx = round_value(float(tmp[6])/G.dx)
            dy = round_value(float(tmp[7])/G.dy)
            dz = round_value(float(tmp[8])/G.dz)

            if xs < 0 or xs >= G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs))
            if xf < 0 or xf >= G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf))
            if ys < 0 or ys >= G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys))
            if yf < 0 or yf >= G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf))
            if zs < 0 or zs >= G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs))
            if zf < 0 or zf >= G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf))
            if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
                print("WARNING: '" + cmdname + ': ' + ' '.join(tmp) + "'" + ' sources and receivers should not normally be positioned within the PML.')
            if xs >= xf or ys >= yf or zs >= zf:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
            if dx < 0 or dy < 0 or dz < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than zero')
            if dx < G.dx or dy < G.dy or dz < G.dz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than the spatial discretisation')

            for x in range(xs, xf, dx):
                for y in range(ys, yf, dy):
                    for z in range(zs, zf, dz):
                        r = Rx(xcoord=x, ycoord=y, zcoord=z)
                        G.rxs.append(r)

            if G.messages:
                print('Receiver box {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with steps {:g}m, {:g}m, {:g} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dy * G.dy, dz * G.dz))
                    
                    
    # Snapshot
    cmdname = '#snapshot'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly eleven parameters')

            xs = round_value(float(tmp[0])/G.dx)
            xf = round_value(float(tmp[3])/G.dx)
            ys = round_value(float(tmp[1])/G.dy)
            yf = round_value(float(tmp[4])/G.dy)
            zs = round_value(float(tmp[2])/G.dz)
            zf = round_value(float(tmp[5])/G.dz)
            dx = round_value(float(tmp[6])/G.dx)
            dy = round_value(float(tmp[7])/G.dy)
            dz = round_value(float(tmp[8])/G.dz)
            
            # If real floating point value given
            if '.' in tmp[9] or 'e' in tmp[9]:
                if float(tmp[9]) > 0:
                    time = round_value((float(tmp[9]) / G.dt)) + 1
                else:
                    raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' time value must be greater than zero')
            # If number of iterations given
            else:
                time = int(tmp[9])
            
            if xs < 0 or xs > G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
            if xf < 0 or xf > G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
            if ys < 0 or ys > G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
            if yf < 0 or yf > G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
            if zs < 0 or zs > G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
            if zf < 0 or zf > G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
            if xs >= xf or ys >= yf or zs >= zf:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
            if dx < 0 or dy < 0 or dz < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than zero')
            if dx < G.dx or dy < G.dy or dz < G.dz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than the spatial discretisation')
            if time <= 0 or time > G.iterations:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' time value is not valid')
                    
            s = Snapshot(xs, ys, zs, xf, yf, zf, dx, dy, dz, time, tmp[10])

            if G.messages:
                print('Snapshot from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, discretisation {:g}m, {:g}m, {:g}m, at {:g} secs with filename {} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dx * G.dy, dx * G.dz, s.time * G.dt, s.filename))
                    
            G.snapshots.append(s)


    # Materials
    # Create built-in materials
    m = Material(0, 'pec', G)
    m.average = False
    G.materials.append(m)
    
    m = Material(1, 'free_space', G)
    m.average = True
    G.materials.append(m)

    cmdname = '#material'
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 5:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly five parameters')
            if float(tmp[0]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for static (DC) permittivity')
            if float(tmp[1]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for conductivity')
            if float(tmp[2]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for permeability')
            if float(tmp[3]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires a positive value for magnetic conductivity')
            if any(x.ID == tmp[4] for x in G.materials):
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' with ID {} already exists'.format(tmp[4]))
            
            # Create a new instance of the Material class material (start index after pec & free_space)
            m = Material(len(G.materials), tmp[4], G)
            m.er = float(tmp[0])
            m.se = float(tmp[1])
            m.mr = float(tmp[2])
            m.sm = float(tmp[3])
            
            if G.messages:
                print('Material {} with epsr={:g}, sig={:g} S/m; mur={:g}, sig*={:g} S/m created.'.format(m.ID, m.er, m.se, m.mr, m.sm))
            
            # Append the new material object to the materials list
            G.materials.append(m)


    cmdname = '#add_dispersion_debye'
    if multicmds[cmdname] != 'None':
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
                material.average = False
                for pole in range(1, 2 * poles, 2):
                    if float(tmp[pole]) > 0 and float(tmp[pole + 1]) > G.dt:
                        material.deltaer.append(float(tmp[pole]))
                        material.tau.append(float(tmp[pole + 1]))
                    else:
                        raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires positive values for the permittivity difference and relaxation times, and relaxation times that are greater than the time step for the model.')
                if material.poles > Material.maxpoles:
                    Material.maxpoles = material.poles

                if G.messages:
                    print('Debye-type disperion added to {} with delta_epsr={:g}, and tau={:g} secs created.'.format(material.ID, ','.join('%4.2f' % deltaer for deltaer in material.deltaer), ','.join('%4.3e' % tau for tau in material.tau)))

    cmdname = '#add_dispersion_lorentz'
    if multicmds[cmdname] != 'None':
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
                material.average = False
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
                    print('Lorentz-type disperion added to {} with delta_epsr={:g}, omega={:g} secs, and gamma={:g} created.'.format(material.ID, ','.join('%4.2f' % deltaer for deltaer in material.deltaer), ','.join('%4.3e' % tau for tau in material.tau), ','.join('%4.3e' % alpha for alpha in material.alpha)))


    cmdname = '#add_dispersion_drude'
    if multicmds[cmdname] != 'None':
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
                material.average = False
                for pole in range(1, 2 * poles, 2):
                    if float(tmp[pole]) > 0 and float(tmp[pole + 1]) > G.dt:
                        material.tau.append(float(tmp[pole ]))
                        material.alpha.append(float(tmp[pole + 1]))
                    else:
                        raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires positive values for the frequencies, and associated times that are greater than the time step for the model.')
                if material.poles > Material.maxpoles:
                    Material.maxpoles = material.poles

                if G.messages:
                    print('Drude-type disperion added to {} with omega={:g} secs, and gamma={:g} secs created.'.format(material.ID, ','.join('%4.3e' % tau for tau in material.tau), ','.join('%4.3e' % alpha for alpha in material.alpha)))


    cmdname = '#soil_peplinski'
    if multicmds[cmdname] != 'None':
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
    if multicmds[cmdname] != 'None':
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly eleven parameters')

            xs = round_value(float(tmp[0])/G.dx)
            xf = round_value(float(tmp[3])/G.dx)
            ys = round_value(float(tmp[1])/G.dy)
            yf = round_value(float(tmp[4])/G.dy)
            zs = round_value(float(tmp[2])/G.dz)
            zf = round_value(float(tmp[5])/G.dz)
            dx = round_value(float(tmp[6])/G.dx)
            dy = round_value(float(tmp[7])/G.dy)
            dz = round_value(float(tmp[8])/G.dz)

            if xs < 0 or xs > G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower x-coordinate {:g}m is not within the model domain'.format(xs * G.dx))
            if xf < 0 or xf > G.nx:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper x-coordinate {:g}m is not within the model domain'.format(xf * G.dx))
            if ys < 0 or ys > G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower y-coordinate {:g}m is not within the model domain'.format(ys * G.dy))
            if yf < 0 or yf > G.ny:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper y-coordinate {:g}m is not within the model domain'.format(yf * G.dy))
            if zs < 0 or zs > G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower z-coordinate {:g}m is not within the model domain'.format(zs * G.dz))
            if zf < 0 or zf > G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the upper z-coordinate {:g}m is not within the model domain'.format(zf * G.dz))
            if xs >= xf or ys >= yf or zs >= zf:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the lower coordinates should be less than the upper coordinates')
            if dx < 0 or dy < 0 or dz < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than zero')
            if dx > G.nx or dy > G.ny or dz > G.nz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should be less than the domain size')
            if dx < G.dx or dy < G.dy or dz < G.dz:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' the step size should not be less than the spatial discretisation')
            if tmp[10].lower() != 'n' and tmp[10].lower() != 'f':
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires type to be either n (normal) or f (fine)')
            
            g = GeometryView(xs, ys, zs, xf, yf, zf, dx, dy, dz, tmp[9], tmp[10].lower())

            if G.messages:
                print('Geometry view from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, discretisation {:g}m, {:g}m, {:g}m, filename {} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dy * G.dy, dz * G.dz, g.filename))

            # Append the new GeometryView object to the geometry views list
            G.geometryviews.append(g)


    # Complex frequency shifted (CFS) PML parameter
    cmdname = '#pml_cfs'
    if multicmds[cmdname] != 'None':
        if len(multicmds[cmdname]) > 2:
            raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' can only be used up to two times, for up to a 2nd order PML')
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 12:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly twelve parameters')
            if tmp[0] not in CFSParameter.scalingprofiles.keys() or tmp[4] not in CFSParameter.scalingprofiles.keys() or tmp[8] not in CFSParameter.scalingprofiles.keys():
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' must have scaling type {}'.format(','.join(CFSParameter.scalingprofiles.keys())))
            if tmp[1] not in CFSParameter.scalingdirections or tmp[5] not in CFSParameter.scalingdirections or tmp[9] not in CFSParameter.scalingdirections:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' must have scaling type {}'.format(','.join(CFSParameter.scalingprofiles.keys())))
            if float(tmp[2]) < 0 or float(tmp[3]) < 0 or float(tmp[6]) < 0 or float(tmp[7]) < 0 or float(tmp[10]) < 0:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' minimum and maximum scaling values must be greater than zero')
            if float(tmp[6]) < 1:
                raise CmdInputError("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' minimum scaling value for kappa must be greater than zero')
            
            cfs = CFS()
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
                print('PML CFS parameters: alpha (scaling: {}, scaling direction: {}, min: {:g}, max: {:g}), kappa (scaling: {}, scaling direction: {}, min: {:g}, max: {:g}), sigma (scaling: {}, scaling direction: {}, min: {:g}, max: {:g}) created.'.format(cfsalpha.scalingprofile, cfsalpha.scalingdirection, cfsalpha.min, cfsalpha.max, cfskappa.scalingprofile, cfskappa.scalingdirection, cfskappa.min, cfskappa.max, cfssigma.scalingprofile, cfssigma.scalingdirection, cfssigma.min, cfssigma.max))
            
            G.cfs.append(cfs)

