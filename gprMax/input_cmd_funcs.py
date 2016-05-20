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


"""This module contains functional forms of some of the most commonly used gprMax commands. It can be useful to use these within Python scripting in an input file.
For convenience, x, y, z coordinates are lumped in a namedtuple Coordinate:
>>> c = Coordinate(0.1, 0.2, 0.3)
>>> c
Coordinate(x=0.1, y=0.2, z=0.3)
>>> str(c)
'0.1 0.2 0.3'
# which can be accessed as a normal tuple:
>>> print c[0], c[1], c[2]
0.1 0.2 0.3
# or more explicitly
>>> print c.x, c.y, c.z
0.1 0.2 0.3
"""
import sys

from collections import namedtuple
Coordinate_tuple = namedtuple('Coordinate', ['x', 'y', 'z'])

class Coordinate(Coordinate_tuple):
    """Subclass of a namedtuple where __str__ outputs 'x y z'"""
    def __str__(self):
        return '{:g} {:g} {:g}'.format(self.x, self.y, self.z)


def command(cmd, *parameters):
    """Helper function. Prints the gprMax #<cmd>: <parameters>. None is ignored in the output.

    Args:
        cmd (str): the gprMax cmd string to be printed
        *parameters: one or more strings as arguments, any None values are ignored

    Returns:
        s (str): the printed string
    """
    # remove Nones
    filtered = filter(lambda x: x is not None, parameters)
    # convert to str
    filtered_str = map(str, filtered)
    # convert to list
    filtered_list = list(filtered_str)
    try:
        s = '#{}: {}'.format(cmd, " ".join(filtered_list))
    except TypeError as e:
        # append info about cmd and parameters to the exception:
        if not e.args: e.args=('', )
        additional_info = "Creating cmd = #{} with parameters {} -> {} failed".format(cmd, parameters, filtered_list)
        e.args = e.args + (additional_info,)
        raise e
    # and now we can print it:
    print(s)
    return s


def domain(x, y, z):
    """Prints the gprMax #domain command.

    Args:
        x, y, z (float): Extent of the domain in the x, y, and z directions.

    Returns:
        domain (Coordinate): Namedtuple of the extent of the domain.
    """
    domain = Coordinate(x, y, z)
    command('domain', domain)

    return domain


def dx_dy_dz(x, y, z):
    """Prints the gprMax #dx_dy_dz command.

    Args:
        x, y, z (float): Spatial resolution in the x, y, and z directions.

    Returns:
        dx_dy_dz (float): Tuple of the spatial resolutions.
    """
    dx_dy_dz = Coordinate(x, y, z)
    command('dx_dy_dz', dx_dy_dz)

    return dx_dy_dz


def time_window(time_window):
    """Prints the gprMax #time_window command.

    Args:
        time_window (float): Duration of simulation.

    Returns:
        time_window (float): Duration of simulation.
    """
    command('time_window', time_window)

    return time_window


def material(permittivity, conductivity, permeability, magconductivity, name):
    """Prints the gprMax #material command.

    Args:
        permittivity (float): Relative permittivity of the material.
        conductivity (float): Conductivity of the material.
        permeability (float): Relative permeability of the material.
        magconductivity (float): Magnetic loss of the material.
        name (str): Material identifier.
    """
    command('material', permittivity, conductivity, permeability, magconductivity, name)


def geometry_view(xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, type='n'):
    """Prints the gprMax #geometry_view command.

    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        dx, dy, dz (float): Spatial discretisation of geometry view.
        filename (str): Filename where geometry file information will be stored.
        type (str): Can be either n (normal) or f (fine) which specifies whether to output the geometry information on a per-cell basis (n) or a per-cell-edge basis (f).
    Returns:
        s, f, d (tuple): 3 namedtuple Coordinate for the start, finish coordinates and spatial discretisation
    """
    s = Coordinate(xs, ys, zs)
    f = Coordinate(xf, yf, zf)
    d = Coordinate(dx, dy, dz)
    command('geometry_view', *s, *f, *d, filename, type)

    return s, f, d


def snapshot(xs, ys, zs, xf, yf, zf, dx, dy, dz, time, filename):
    """Prints the gprMax #snapshot command.

    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        dx, dy, dz (float): Spatial discretisation of geometry view.
        time (float): Time in seconds (float) or the iteration number (integer) which denote the point in time at which the snapshot will be taken.
        filename (str): Filename where geometry file information will be stored.
    Returns:
        s, f, d (tuple): 3 namedtuple Coordinate for the start, finish coordinates and spatial discretisation
    """
    s = Coordinate(xs, ys, zs)
    f = Coordinate(xf, yf, zf)
    d = Coordinate(dx, dy, dz)

    if  '.' in str(time) or 'e' in str(time):
        time = '{:g}'.format(float(time))
    else:
        time = '{:d}'.format(int(time))

    command('snapshot', *s, *f, *d, time, filename)

    return s, f, d


def edge(xs, ys, zs, xf, yf, zf, material):
    """Prints the gprMax #edge command.

    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        material (str): Material identifier.
    Returns:
        s, f (tuple): 2 namedtuple Coordinate for the start and finish coordinates
    """
    s = Coordinate(xs, ys, zs)
    f = Coordinate(xf, yf, zf)
    command('edge', *s, *f, material)

    return s, f


def plate(xs, ys, zs, xf, yf, zf, material):
    """Prints the gprMax #plate command.

    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        material (str): Material identifier(s).
    Returns:
        s, f (tuple): 2 namedtuple Coordinate for the start and finish coordinates
    """
    s = Coordinate(xs, ys, zs)
    f = Coordinate(xf, yf, zf)
    command('plate', *s, *f, material)

    return s, f


def triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, thickness, material):
    """Prints the gprMax #triangle command.

    Args:
        x1, y1, z1, x2, y2, z2, x3, y3, z3 (float): Coordinates of the vertices.
        thickness (float): Thickness for a triangular prism, or zero for a triangular patch.
        material (str): Material identifier(s).
    Returns:
        v1, v2, v3 (tuple): 3 namedtuple Coordinate for the vertices
    """
    v1 = Coordinate(x1, y1, z1)
    v2 = Coordinate(x2, y2, z2)
    v3 = Coordinate(x3, y3, z3)
    command('triangle', *v1, *v2, *v3, thickness, material)

    return v1, v2, v3


def box(xs, ys, zs, xf, yf, zf, material, averaging=''):
    """Prints the gprMax #box command.

    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        material (str): Material identifier(s).
        averaging (str): Turn averaging on or off.

    Returns:
        s, f (tuple): 2 namedtuple Coordinate for the start and finish coordinates
    """
    s = Coordinate(xs, ys, zs)
    f = Coordinate(xf, yf, zf)
    command('box', *s, *f, material, averaging)

    return s, f


def sphere(x, y, z, radius, material, averaging=''):
    """Prints the gprMax #sphere command.

    Args:
        x, y, z (float): Coordinates of the centre of the sphere.
        radius (float): Radius.
        material (str): Material identifier(s).
        averaging (str): Turn averaging on or off.

    Returns:
        c (tuple): namedtuple Coordinate for the center of the sphere
    """
    c = Coordinate(x, y, z)
    command('sphere', *c, radius, material, averaging)

    return c


def cylinder(x1, y1, z1, x2, y2, z2, radius, material, averaging=''):
    """Prints the gprMax #cylinder command.

    Args:
        x1, y1, z1, x2, y2, z2 (float): Coordinates of the centres of the two faces of the cylinder.
        radius (float): Radius.
        material (str): Material identifier(s).
        averaging (str): Turn averaging on or off.

    Returns:
        c1, c2 (tuple): 2 namedtuple Coordinate for the centres of the two faces of the cylinder.
    """
    c1 = Coordinate(x1, y1, z1)
    c2 = Coordinate(x2, y2, z2)
    command('cylinder', *c1, *c2, radius, material, averaging)

    return c1, c2


def cylindrical_sector(axis, ctr1, ctr2, t1, t2, radius, startingangle, sweptangle, material, averaging=''):
    """Prints the gprMax #cylindrical_sector command.

    Args:
        axis (str): Axis of the cylinder from which the sector is defined and can be x, y, or z.
        ctr1, ctr2 (float): Coordinates of the centre of the cylindrical sector.
        t1, t2 (float): Lower and higher coordinates of the axis of the cylinder from which the sector is defined (in effect they specify the thickness of the sector).
        radius (float): Radius.
        startingangle (float): Starting angle (in degrees) for the cylindrical sector (with zero degrees defined on the positive first axis of the plane of the cylindrical sector).
        sweptangle (float): Angle (in degrees) swept by the cylindrical sector (the finishing angle of the sector is always anti-clockwise from the starting angle).
        material (str): Material identifier(s).
        averaging (str): Turn averaging on or off.
    """
    command('cylindrical_sector', axis, ctr1, ctr2, t1, t2, radius, startingangle, sweptangle, material, averaging)


def excitation_file(file1):
    """Prints the #excitation_file: <file1> command.

    Args:
        file1 (str): filename
    Returns:
        file1 (str): filename
    """
    command('excitation_file', file1)

    return file1


def waveform(shape, amplitude, frequency, identifier):
    """Prints the #waveform: shape amplitude frequency identifier

    Args:
        shape (str): is the type of waveform
        amplitude (float): is the amplitude of the waveform.
        frequency (float): is the frequency of the waveform in Hertz.
        identifier (str): is an identifier for the waveform used to assign it to a source.
    Returns:
        identifier (str): is an identifier for the waveform used to assign it to a source.
    """
    command('waveform', shape, amplitude, frequency, identifier)

    return identifier


def hertzian_dipole(polarization, f1, f2, f3, identifier, t0=None, t_remove=None):
    """Prints the #hertzian_dipole: polarization, f1, f2, f3, identifier, [t0, t_remove]

    Args:
        polarization (str):  is the polarisation of the source and can be 'x', 'y', or 'z'.
        f1 f2 f3 (float): are the coordinates (x,y,z) of the source in the model.
        identifier (str): is the identifier of the waveform that should be used with the source.
        t0 (float): is an optinal argument for the time delay in starting the source.
        t_remove (float): is a time to remove the source.
    Returns:
        coordinates (tuple): namedtuple Coordinate of the source location
    """
    c = Coordinate(f1, f2, f3)
    # since command ignores None, this is safe:
    command('hertzian_dipole', polarization, str(c), identifier, t0, t_remove)

    return c


def rx(x, y, z, identifier=None, to_save=None):
    """Prints the #rx: x, y, z, [identifier, to_save] command.

    Args:
        x, y, z (float): are the coordinates (x,y,z) of the receiver in the model.
        identifier (str): is the optional identifier of the receiver
        to_save (list):  is a list of outputs with this receiver. It can be any selection from 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', or 'Iz'.
    Returns:
        coordinates (tuple): namedtuple Coordinate of the receiver location
    """
    c = Coordinate(x, y, z)
    command('rx', str(c), identifier, to_save)

    return c


def src_steps(dx=0, dy=0, dz=0):
    """Prints the #src_steps: dx, dy, dz command.

    Args:
        dx, dy, dz (float): are the increments in (x, y, z) to move all simple sources or all receivers.
    Returns:
        coordinates (tuple): namedtuple Coordinate of the increments
    """

    c = Coordinate(dx, dy, dz)
    command('src_steps', str(c))

    return c


def rx_steps(dx=0, dy=0, dz=0):
    """Prints the #rx_steps: dx, dy, dz command.

    Args:
        dx, dy, dz (float): are the increments in (x, y, z) to move all simple sources or all receivers.
    Returns:
        coordinates (tuple): namedtuple Coordinate of the increments
    """

    c = Coordinate(dx, dy, dz)
    command('rx_steps', str(c))
    return c


