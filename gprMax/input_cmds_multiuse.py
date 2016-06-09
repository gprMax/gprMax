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

from .constants import z0
from .exceptions import CmdInputError
from .geometry_views import GeometryView
from .materials import Material, PeplinskiSoil
from .pml import CFSParameter, CFS
from .receivers import Rx
from .snapshots import Snapshot
from .sources import (
    VoltageSource, HertzianDipole, MagneticDipole, TransmissionLine, TEMTransmissionLine)
from .utilities import round_value
from .waveforms import Waveform

from collections import OrderedDict


def check_coordinates(x, y, z, G, cmdname, tmp, name=''):
    # Check if coordinates are within the bounds of the grid
    try:
        G.within_bounds(x=x, y=y, z=z)
    except ValueError as err:

        s = "'{}: {} ' {} {}-coordinate is not within the model domain".format(
            cmdname, ' '.join(tmp), name, err.args[0])
        raise CmdInputError(s)


def create_cmd_msg(cmdname, cmdinstance):
    """Function to create a formatted str relating to a command
    and its parameters
    """
    return "'{}:{}'".format(cmdname, cmdinstance)


def create_transmission_line(cmdname, cmdinstance, G):

    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    tmp = cmdinstance.split()

    # Check correct number of args
    if len(tmp) < 6:
        raise CmdInputError(
            "{} requires at least six parameters".format(cmd_msg))

    # Check polarity & position parameters
    if tmp[0].lower() not in ('x', 'y', 'z', 'zx', 'xz'):
        raise CmdInputError(
            "{} polarisation must be x, y, z or zx".format(cmd_msg))

    xcoord = G.calculate_coord('x', tmp[1])
    ycoord = G.calculate_coord('y', tmp[2])
    zcoord = G.calculate_coord('z', tmp[3])
    resistance = float(tmp[4])

    check_coordinates(xcoord, ycoord, zcoord, G, cmdname, tmp)

    if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
        print("WARNING: {} sources and receivers should not normally be \
            positioned within the PML.".format(cmd_msg))
    if resistance <= 0 or resistance > z0:
        raise CmdInputError(
            "{} requires a resistance greater than zero and less than the impedance \
            of free space, i.e. 376.73 Ohms".format(cmd_msg))

    # Check if there is a waveformID in the waveforms list
    if not any(x.ID == tmp[5] for x in G.waveforms):
        raise CmdInputError("{} there is no waveform with the identifier \
         {}".format(cmd_msg, tmp[4]))

    # Can create several types of transmission line
    if cmdname == '#transmission_line':
        t = TransmissionLine(G)
    elif cmdname == '#tem_transmission_line':
        t = TEMTransmissionLine(G)
    else:
        raise CmdInputError("{} is not a valid transmission line type")

    t.polarisation = tmp[0]
    t.xcoord = xcoord
    t.ycoord = ycoord
    t.zcoord = zcoord
    t.setID()
    t.resistance = resistance
    t.waveformID = tmp[5]
    t.calculate_incident_V_I(G)

    if len(tmp) > 6:
        # Check source start & source remove time parameters
        start = float(tmp[6])
        stop = float(tmp[7])
        if start < 0:
            raise CmdInputError("{} delay of the initiation of the source should \
                not be less than zero".format(cmd_msg))
        if stop < 0:
            raise CmdInputError("{} time to remove the source should not be less \
                than zero".format(cmd_msg))
        if stop - start <= 0:
            raise CmdInputError("{} duration of the source should not be zero or \
                less".format(cmd_msg))
        t.start = start
        if stop > G.timewindow:
            t.stop = G.timewindow
        else:
            t.stop = stop
        startstop = ' start time {:g} secs, finish time {:g} secs '.format(
            t.start, t.stop)
    else:
        t.start = 0
        t.stop = G.timewindow
        startstop = ' '

    if G.messages:
        print('Transmission line with polarity {} at {:g}m, {:g}m, {:g}m, resistance {:.1f} Ohms,'.format(t.polarisation, t.xcoord * G.dx, t.ycoord * G.dy, t.zcoord * G.dz, t.resistance) + startstop + 'using waveform {} created.'.format(t.waveformID))

    G.transmissionlines.append(t)


def create_voltage_source(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) < 6:
        raise CmdInputError("{} requires at least six parameters".format(
            cmd_msg))

    # Check polarity & position parameters
    if tmp[0].lower() not in ('x', 'y', 'z'):
        raise CmdInputError("{} polarisation must be x, y, or z".format(
            cmd_msg))
    xcoord = G.calculate_coord('x', tmp[1])
    ycoord = G.calculate_coord('y', tmp[2])
    zcoord = G.calculate_coord('z', tmp[3])
    resistance = float(tmp[4])
    check_coordinates(xcoord, ycoord, zcoord, G, cmdname, tmp)
    if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
        print("WARNING: {} sources and receivers should not normally be \
            positioned within the PML.".format(cmd_msg))
    if resistance < 0:
        raise CmdInputError("{} requires a source resistance of zero or \
            greater".format(cmd_msg))

    # Check if there is a waveformID in the waveforms list
    if not any(x.ID == tmp[5] for x in G.waveforms):
        raise CmdInputError("{} there is no waveform with the \
            identifier {}".format(cmd_msg, tmp[5]))

    v = VoltageSource()
    v.polarisation = tmp[0]
    v.xcoord = xcoord
    v.ycoord = ycoord
    v.zcoord = zcoord
    v.ID = 'VoltageSource(' + str(v.xcoord) + ',' + str(v.ycoord) + ',' + str(v.zcoord) + ')'
    v.resistance = resistance
    v.waveformID = tmp[5]

    if len(tmp) > 6:
        # Check source start & source remove time parameters
        start = float(tmp[6])
        stop = float(tmp[7])
        if start < 0:
            raise CmdInputError("{} delay of the initiation of the source should \
                not be less than zero".format(cmd_msg))
        if stop < 0:
            raise CmdInputError("{} time to remove the source should not be less \
                than zero".format(cmd_msg))
        if stop - start <= 0:
            raise CmdInputError("{} duration of the source should not be zero or \
                less".format(cmd_msg))
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

    if G.messages:
        print('Voltage source with polarity {} at {:g}m, {:g}m, {:g}m, resistance {:.1f} Ohms,'.format(v.polarisation, v.xcoord * G.dx, v.ycoord * G.dy, v.zcoord * G.dz, v.resistance) + startstop + 'using waveform {} created.'.format(v.waveformID))

    G.voltagesources.append(v)


def create_waveforms(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) != 4:
        raise CmdInputError("{} requires exactly four parameters".format(
            cmd_msg))
    if tmp[0].lower() not in Waveform.types:
        raise CmdInputError("{} must have one of the following types \
            {}".format(cmd_msg, ','.join(Waveform.types)))
    if float(tmp[2]) <= 0:
        raise CmdInputError("{} requires an excitation frequency value of greater \
            than zero".format(cmd_msg))
    if any(x.ID == tmp[3] for x in G.waveforms):
        raise CmdInputError("{} with ID {} already exists".format(
            cmd_msg, tmp[3]))

    w = Waveform()
    w.ID = tmp[3]
    w.type = tmp[0].lower()
    w.amp = float(tmp[1])
    w.freq = float(tmp[2])

    if G.messages:
        print('Waveform {} of type {} with amplitude {:g}, frequency {:g}Hz created.'.format(w.ID, w.type, w.amp, w.freq))

    G.waveforms.append(w)


def create_hertzian_dipole(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) < 5:
        raise CmdInputError("{} requires at least five parameters".format(cmd_msg))

    # Check polarity & position parameters
    if tmp[0].lower() not in ('x', 'y', 'z'):
        raise CmdInputError("{} polarisation must be x, y, or z".format(cmd_msg))
    xcoord = G.calculate_coord('x', tmp[1])
    ycoord = G.calculate_coord('y', tmp[2])
    zcoord = G.calculate_coord('z', tmp[3])
    check_coordinates(xcoord, ycoord, zcoord, G, cmdname, tmp)
    if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
        print("WARNING: {} sources and receivers should not normally be positioned within the PML.".format(cmd_msg))

    # Check if there is a waveformID in the waveforms list
    if not any(x.ID == tmp[4] for x in G.waveforms):
        raise CmdInputError("{} there is no waveform with the identifier {}".format(cmd_msg, tmp[4]))

    h = HertzianDipole()
    h.polarisation = tmp[0]
    h.xcoord = xcoord
    h.ycoord = ycoord
    h.zcoord = zcoord
    h.xcoordbase = xcoord
    h.ycoordbase = ycoord
    h.zcoordbase = zcoord
    h.ID = 'HertzianDipole(' + str(h.xcoord) + ',' + str(h.ycoord) + ',' + str(h.zcoord) + ')'
    h.waveformID = tmp[4]

    if len(tmp) > 5:
        # Check source start & source remove time parameters
        start = float(tmp[5])
        stop = float(tmp[6])
        if start < 0:
            raise CmdInputError("{} delay of the initiation of the source should not be less than zero".format(cmd_msg))
        if stop < 0:
            raise CmdInputError("{} time to remove the source should not be less than zero".format(cmd_msg))
        if stop - start <= 0:
            raise CmdInputError("{} duration of the source should not be zero or less".format(cmd_msg))
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

    if G.messages:
        print('Hertzian dipole with polarity {} at {:g}m, {:g}m, {:g}m,'.format(h.polarisation, h.xcoord * G.dx, h.ycoord * G.dy, h.zcoord * G.dz) + startstop + 'using waveform {} created.'.format(h.waveformID))

    G.hertziandipoles.append(h)


def create_magnetic_dipole(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) < 5:
        raise CmdInputError("{} requires at least five parameters".format(cmd_msg))

    # Check polarity & position parameters
    if tmp[0].lower() not in ('x', 'y', 'z'):
        raise CmdInputError("{} polarisation must be x, y, or z".format(cmd_msg))
    xcoord = G.calculate_coord('x', tmp[1])
    ycoord = G.calculate_coord('y', tmp[2])
    zcoord = G.calculate_coord('z', tmp[3])
    check_coordinates(xcoord, ycoord, zcoord, G, cmdname, tmp)
    if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
        print("WARNING: {} sources and receivers should not normally be positioned within the PML.".format(cmd_msg))

    # Check if there is a waveformID in the waveforms list
    if not any(x.ID == tmp[4] for x in G.waveforms):
        raise CmdInputError("{} there is no waveform with the identifier {}".format(cmd_msg, tmp[4]))

    m = MagneticDipole()
    m.polarisation = tmp[0]
    m.xcoord = xcoord
    m.ycoord = ycoord
    m.zcoord = zcoord
    m.xcoordbase = xcoord
    m.ycoordbase = ycoord
    m.zcoordbase = zcoord
    m.ID = 'MagneticDipole(' + str(m.xcoord) + ',' + str(m.ycoord) + ',' + str(m.zcoord) + ')'
    m.waveformID = tmp[4]

    if len(tmp) > 5:
        # Check source start & source remove time parameters
        start = float(tmp[5])
        stop = float(tmp[6])
        if start < 0:
            raise CmdInputError("{} delay of the initiation of the source should not be less than zero".format(cmd_msg))
        if stop < 0:
            raise CmdInputError("{} time to remove the source should not be less than zero".format(cmd_msg))
        if stop - start <= 0:
            raise CmdInputError("{} duration of the source should not be zero or less".format(cmd_msg))
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

    if G.messages:
        print('Magnetic dipole with polarity {} at {:g}m, {:g}m, {:g}m,'.format(m.polarisation, m.xcoord * G.dx, m.ycoord * G.dy, m.zcoord * G.dz) + startstop + 'using waveform {} created.'.format(m.waveformID))

    G.magneticdipoles.append(m)


def create_rx(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) != 3 and len(tmp) < 5:
        raise CmdInputError("{} has an incorrect number of parameters".format(cmd_msg))

    # Check position parameters
    xcoord = round_value(float(tmp[0]) / G.dx)
    ycoord = round_value(float(tmp[1]) / G.dy)
    zcoord = round_value(float(tmp[2]) / G.dz)
    check_coordinates(xcoord, ycoord, zcoord)
    if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
        print("WARNING: {} sources and receivers should not normally be positioned within the PML.\n".format(cmd_msg))

    r = Rx()
    r.xcoord = xcoord
    r.ycoord = ycoord
    r.zcoord = zcoord
    r.xcoordbase = xcoord
    r.ycoordbase = ycoord
    r.zcoordbase = zcoord

    # If no ID or outputs are specified, use default i.e Ex, Ey, Ez, Hx, Hy, Hz, Ix, Iy, Iz
    if len(tmp) == 3:
        r.ID = 'Rx(' + str(r.xcoord) + ',' + str(r.ycoord) + ',' + str(r.zcoord) + ')'
        r.outputs = Rx.availableoutputs[0:9]
    else:
        r.ID = tmp[3]
        # Check and add field output names
        for field in tmp[4::]:
            if field in Rx.availableoutputs:
                r.outputs.append(field)
            else:
                raise CmdInputError("{} contains an output type that is not available".format(cmd_msg))

    if G.messages:
        print('Receiver at {:g}m, {:g}m, {:g}m with output(s) {} created.'.format(r.xcoord * G.dx, r.ycoord * G.dy, r.zcoord * G.dz, ', '.join(r.outputs)))

    G.rxs.append(r)


def create_rx_box(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) != 9:
        raise CmdInputError("{} requires exactly nine parameters".format(cmd_msg))

    xs = G.calculate_coord('x', tmp[0])
    ys = G.calculate_coord('y', tmp[1])
    zs = G.calculate_coord('z', tmp[2])

    xf = G.calculate_coord('x', tmp[3])
    yf = G.calculate_coord('y', tmp[4])
    zf = G.calculate_coord('z', tmp[5])

    dx = G.calculate_coord('x', tmp[6])
    dy = G.calculate_coord('y', tmp[7])
    dz = G.calculate_coord('z', tmp[8])

    check_coordinates(xs, ys, zs, G, cmdname, tmp, name='lower')
    check_coordinates(xf, yf, zf, G, cmdname, tmp, name='upper')

    if xcoord < G.pmlthickness[0] or xcoord > G.nx - G.pmlthickness[3] or ycoord < G.pmlthickness[1] or ycoord > G.ny - G.pmlthickness[4] or zcoord < G.pmlthickness[2] or zcoord > G.nz - G.pmlthickness[5]:
        print("WARNING: {} sources and receivers should not normally be positioned within the PML.".format(cmd_msg))
    if xs >= xf or ys >= yf or zs >= zf:
        raise CmdInputError("{} the lower coordinates should be less than the upper coordinates".format(cmd_msg))
    if dx < 0 or dy < 0 or dz < 0:
        raise CmdInputError("{} the step size should not be less than zero".format(cmd_msg))
    if dx < G.dx or dy < G.dy or dz < G.dz:
        raise CmdInputError("{} the step size should not be less than the spatial discretisation".format(cmd_msg))

    for x in range(xs, xf, dx):
        for y in range(ys, yf, dy):
            for z in range(zs, zf, dz):
                r = Rx()
                r.xcoord = x
                r.ycoord = y
                r.zcoord = z
                r.xcoordbase = x
                r.ycoordbase = y
                r.zcoordbase = z
                r.ID = 'Rx(' + str(x) + ',' + str(y) + ',' + str(z) + ')'
                G.rxs.append(r)

    if G.messages:
        print('Receiver box {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with steps {:g}m, {:g}m, {:g} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dy * G.dy, dz * G.dz))


def create_snapshot(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) != 11:
        raise CmdInputError("{} requires exactly eleven parameters".format(cmd_msg))

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
    except:
        time = float(tmp[9])
        if time > 0:
            time = round_value((time / G.dt)) + 1
        else:
            raise CmdInputError("{} time value must be greater than zero".format(cmd_msg))

    check_coordinates(xs, ys, zs, G, cmdname, tmp, name='lower')
    check_coordinates(xf, yf, zf, G, cmdname, tmp, name='upper')

    if xs >= xf or ys >= yf or zs >= zf:
        raise CmdInputError("{} the lower coordinates should be less than the upper coordinates".format(cmd_msg))
    if dx < 0 or dy < 0 or dz < 0:
        raise CmdInputError("{} the step size should not be less than zero".format(cmd_msg))
    if dx < G.dx or dy < G.dy or dz < G.dz:
        raise CmdInputError("{} the step size should not be less than the spatial discretisation".format(cmd_msg))
    if time <= 0 or time > G.iterations:
        raise CmdInputError("{} time value is not valid".format(cmd_msg))

    s = Snapshot(xs, ys, zs, xf, yf, zf, dx, dy, dz, time, tmp[10])

    if G.messages:
        print('Snapshot from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, discretisation {:g}m, {:g}m, {:g}m, at {:g} secs with filename {} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dx * G.dy, dx * G.dz, s.time * G.dt, s.basefilename))

    G.snapshots.append(s)


def create_material(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) != 5:
        raise CmdInputError("{} requires exactly five parameters".format(cmd_msg))
    if float(tmp[0]) < 0:
        raise CmdInputError("{} requires a positive value for static (DC) permittivity".format(cmd_msg))
    if float(tmp[1]) < 0:
        raise CmdInputError("{} requires a positive value for conductivity".format(cmd_msg))
    if float(tmp[2]) < 0:
        raise CmdInputError("{} requires a positive value for permeability".format(cmd_msg))
    if float(tmp[3]) < 0:
        raise CmdInputError("{} requires a positive value for magnetic conductivity".format(cmd_msg))
    if any(x.ID == tmp[4] for x in G.materials):
        raise CmdInputError("{} with ID {} already exists'.format(tmp[4]".format(cmd_msg))

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


def create_add_dispersion_debye(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)

    if len(tmp) < 4:
        raise CmdInputError("{} requires at least four parameters".format(cmd_msg))
    if int(tmp[0]) < 0:
        raise CmdInputError("{} requires a positive value for number of poles".format(cmd_msg))
    poles = int(tmp[0])
    materialsrequested = tmp[(2 * poles) + 1:len(tmp)]

    # Look up requested materials in existing list of material instances
    materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

    if len(materials) != len(materialsrequested):
        notfound = [x for x in materialsrequested if x not in materials]
        raise CmdInputError("{} material(s) {} do not exist".format(cmd_msg, notfound))

    for material in materials:
        material.type = 'debye'
        material.poles = poles
        material.average = False
        for pole in range(1, 2 * poles, 2):
            if float(tmp[pole]) > 0 and float(tmp[pole + 1]) > G.dt:
                material.deltaer.append(float(tmp[pole]))
                material.tau.append(float(tmp[pole + 1]))
            else:
                raise CmdInputError("{} requires positive values for the permittivity \
                    difference and relaxation times, and relaxation times that are greater \
                    than the time step for the model.".format(cmd_msg))
        if material.poles > Material.maxpoles:
            Material.maxpoles = material.poles

        if G.messages:
            print('Debye disperion added to {} with delta_epsr={}, and tau={} secs created.'.format(material.ID, ', '.join('%4.2f' % deltaer for deltaer in material.deltaer), ', '.join('%4.3e' % tau for tau in material.tau)))


def create_add_dispersion_lorentz(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)

    if len(tmp) < 5:
        raise CmdInputError("{} requires at least five parameters".format(cmd_msg))
    if int(tmp[0]) < 0:
        raise CmdInputError("{} requires a positive value for number of poles".format(cmd_msg))
    poles = int(tmp[0])
    materialsrequested = tmp[(3 * poles) + 1:len(tmp)]

    # Look up requested materials in existing list of material instances
    materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

    if len(materials) != len(materialsrequested):
        notfound = [x for x in materialsrequested if x not in materials]
        raise CmdInputError("{} material(s) {} do not exist".format(notfound, cmd_msg))

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
                raise CmdInputError("{} requires positive values for the permittivity difference and frequencies, and associated times that are greater than the time step for the model.".format(cmd_msg))
        if material.poles > Material.maxpoles:
            Material.maxpoles = material.poles

        if G.messages:
            print('Lorentz disperion added to {} with delta_epsr={}, omega={} secs, and gamma={} created.'.format(material.ID, ', '.join('%4.2f' % deltaer for deltaer in material.deltaer), ', '.join('%4.3e' % tau for tau in material.tau), ', '.join('%4.3e' % alpha for alpha in material.alpha)))


def create_add_dispersion_drude(cmdname, cmdinstance, G):

    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)

    if len(tmp) < 5:
        raise CmdInputError("{} requires at least five parameters".format(cmd_msg))
    if int(tmp[0]) < 0:
        raise CmdInputError("{} requires a positive value for number of poles".format(cmd_msg))
    poles = int(tmp[0])
    materialsrequested = tmp[(3 * poles) + 1:len(tmp)]

    # Look up requested materials in existing list of material instances
    materials = [y for x in materialsrequested for y in G.materials if y.ID == x]

    if len(materials) != len(materialsrequested):
        notfound = [x for x in materialsrequested if x not in materials]
        raise CmdInputError("{} material(s) {} do not exist".format(cmd_msg, notfound))

    for material in materials:
        material.type = 'drude'
        material.poles = poles
        material.average = False
        for pole in range(1, 2 * poles, 2):
            if float(tmp[pole]) > 0 and float(tmp[pole + 1]) > G.dt:
                material.tau.append(float(tmp[pole]))
                material.alpha.append(float(tmp[pole + 1]))
            else:
                raise CmdInputError("{} requires positive values for the frequencies, and associated times that are greater than the time step for the model.".format(cmd_msg))
        if material.poles > Material.maxpoles:
            Material.maxpoles = material.poles

        if G.messages:
            print('Drude disperion added to {} with omega={} secs, and gamma={} secs created.'.format(material.ID, ', '.join('%4.3e' % tau for tau in material.tau), ', '.join('%4.3e' % alpha for alpha in material.alpha)))


def create_soil_peplinski(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) != 7:
        raise CmdInputError("{} requires at exactly seven parameters".format(cmd_msg))
    if float(tmp[0]) < 0:
        raise CmdInputError("{} requires a positive value for the sand fraction".format(cmd_msg))
    if float(tmp[1]) < 0:
        raise CmdInputError("{} requires a positive value for the clay fraction".format(cmd_msg))
    if float(tmp[2]) < 0:
        raise CmdInputError("{} requires a positive value for the bulk density".format(cmd_msg))
    if float(tmp[3]) < 0:
        raise CmdInputError("{} requires a positive value for the sand particle density".format(cmd_msg))
    if float(tmp[4]) < 0:
        raise CmdInputError("{} requires a positive value for the lower limit of the water volumetric fraction".format(cmd_msg))
    if float(tmp[5]) < 0:
        raise CmdInputError("{} requires a positive value for the upper limit of the water volumetric fraction".format(cmd_msg))
    if any(x.ID == tmp[6] for x in G.mixingmodels):
        raise CmdInputError("{} with ID {} already exists".format(cmd_msg, tmp[6]))

    # Create a new instance of the Material class material (start index after pec & free_space)
    s = PeplinskiSoil(tmp[6], float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3]), (float(tmp[4]), float(tmp[5])))

    if G.messages:
        print('Mixing model (Peplinski) used to create {} with sand fraction {:g}, clay fraction {:g}, bulk density {:g}g/cm3, sand particle density {:g}g/cm3, and water volumetric fraction {:g} to {:g} created.'.format(s.ID, s.S, s.C, s.rb, s.rs, s.mu[0], s.mu[1]))

    # Append the new material object to the materials list
    G.mixingmodels.append(s)


def create_geometry_view(cmdname, cmdinstance, G):
    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)
    if len(tmp) != 11:
        raise CmdInputError("{} requires exactly eleven parameters".format(cmd_msg))

    xs = G.calculate_coord('x', tmp[0])
    ys = G.calculate_coord('y', tmp[1])
    zs = G.calculate_coord('z', tmp[2])

    xf = G.calculate_coord('x', tmp[3])
    yf = G.calculate_coord('y', tmp[4])
    zf = G.calculate_coord('z', tmp[5])

    dx = G.calculate_coord('x', tmp[6])
    dy = G.calculate_coord('y', tmp[7])
    dz = G.calculate_coord('z', tmp[8])

    check_coordinates(xs, ys, zs, G, cmdname, tmp, name='lower')
    check_coordinates(xf, yf, zf, G, cmdname, tmp, name='upper')

    if xs >= xf or ys >= yf or zs >= zf:
        raise CmdInputError("{} the lower coordinates should be less than the upper coordinates".format(cmd_msg))
    if dx < 0 or dy < 0 or dz < 0:
        raise CmdInputError("{} the step size should not be less than zero".format(cmd_msg))
    if dx > G.nx or dy > G.ny or dz > G.nz:
        raise CmdInputError("{} the step size should be less than the domain size".format(cmd_msg))
    if dx < G.dx or dy < G.dy or dz < G.dz:
        raise CmdInputError("{} the step size should not be less than the spatial discretisation".format(cmd_msg))
    if tmp[10].lower() != 'n' and tmp[10].lower() != 'f':
        raise CmdInputError("{} requires type to be either n (normal) or f (fine)".format(cmd_msg))
    if tmp[10].lower() == 'f' and (dx * G.dx != G.dx or dy * G.dy != G.dy or dz * G.dz != G.dz):
        raise CmdInputError("{} requires the spatial discretisation for the geometry view to be the same as the model for geometry view of type f (fine)".format(cmd_msg))

    g = GeometryView(xs, ys, zs, xf, yf, zf, dx, dy, dz, tmp[9], tmp[10].lower())

    if G.messages:
        print('Geometry view from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m, discretisation {:g}m, {:g}m, {:g}m, filename {} created.'.format(xs * G.dx, ys * G.dy, zs * G.dz, xf * G.dx, yf * G.dy, zf * G.dz, dx * G.dx, dy * G.dy, dz * G.dz, g.basefilename))

    # Append the new GeometryView object to the geometry views list
    G.geometryviews.append(g)


def create_pml_cfs(cmdname, cmdinstance, G):

    tmp = cmdinstance.split()
    cmd_msg = create_cmd_msg(cmdname, cmdinstance)

    if len(G.cfs) == 2:
        raise CmdInputError("{} can only be used up to two times, for up to a 2nd order PML".format(cmd_msg))

    if len(tmp) != 12:
        raise CmdInputError("{} requires exactly twelve parameters".format(cmd_msg))
    if tmp[0] not in CFSParameter.scalingprofiles.keys() or tmp[4] not in CFSParameter.scalingprofiles.keys() or tmp[8] not in CFSParameter.scalingprofiles.keys():
        raise CmdInputError("{} must have scaling type {}'.format(','.join(CFSParameter.scalingprofiles.keys())".format(cmd_msg))
    if tmp[1] not in CFSParameter.scalingdirections or tmp[5] not in CFSParameter.scalingdirections or tmp[9] not in CFSParameter.scalingdirections:
        raise CmdInputError("{} must have scaling type {}'.format(','.join(CFSParameter.scalingprofiles.keys())".format(cmd_msg))
    if float(tmp[2]) < 0 or float(tmp[3]) < 0 or float(tmp[6]) < 0 or float(tmp[7]) < 0 or float(tmp[10]) < 0:
        raise CmdInputError("{} minimum and maximum scaling values must be greater than zero".format(cmd_msg))
    if float(tmp[6]) < 1:
        raise CmdInputError("{} minimum scaling value for kappa must be greater than zero".format(cmd_msg))

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


def process_multicmds(multicmds, G):
    """Checks the validity of command parameters and creates instances of
    classes of parameters.

    Args:
        multicmds (dict): Commands that can have multiple instances in the
        model.
        G (class): Grid class instance - holds essential parameters
        describing the model.
    """

    # Order of commands is important as some cmds have dependencies.
    cmds = OrderedDict()

    cmds['#waveform'] = create_waveforms
    cmds['#voltage_source'] = create_voltage_source
    cmds['#hertzian_dipole'] = create_hertzian_dipole
    cmds['#magnetic_dipole'] = create_magnetic_dipole
    cmds['#transmission_line'] = create_transmission_line
    cmds['#tem_transmission_line'] = create_transmission_line
    cmds['#rx'] = create_rx
    cmds['#rx_box'] = create_rx_box
    cmds['#snapshot'] = create_snapshot
    cmds['#material'] = create_material
    cmds['#add_dispersion_debye'] = create_add_dispersion_debye
    cmds['#add_dispersion_lorentz'] = create_add_dispersion_lorentz
    cmds['#add_dispersion_drude'] = create_add_dispersion_drude
    cmds['#soil_peplinski'] = create_soil_peplinski
    cmds['#geometry_view'] = create_geometry_view
    cmds['#pml_cfs'] = create_pml_cfs

    for cmdname, func in cmds.items():
        if multicmds[cmdname] != 'None':
            for cmdinstance in multicmds[cmdname]:
                func(cmdname, cmdinstance, G)
