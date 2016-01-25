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


"""This module contains functional forms of some of the most commonly used gprMax commands. It can useful to use these within Python scripting in an input file."""


def domain(x, y, z):
    """Prints the gprMax #domain command.
        
    Args:
        x, y, z (float): Extent of the domain in the x, y, and z directions.
        
    Returns:
        domain (float): Tuple of the extent of the domain.
    """
    
    domain = (x, y, z)
    print('#domain: {:g} {:g} {:g}'.format(domain[0], domain[1], domain[2]))
          
    return domain


def dx_dy_dz(x, y, z):
    """Prints the gprMax #dx_dy_dz command.
        
    Args:
        x, y, z (float): Spatial resolution in the x, y, and z directions.
        
    Returns:
        dx_dy_dz (float): Tuple of the spatial resolutions.
    """
    
    dx_dy_dz = (x, y, z)
    print('#dx_dy_dz: {:g} {:g} {:g}'.format(dx_dy_dz[0], dx_dy_dz[1], dx_dy_dz[2]))
          
    return dx_dy_dz
          
    
def time_window(time_window):
    """Prints the gprMax #time_window command.
        
    Args:
        time_window (float): Duration of simulation.
        
    Returns:
        time_window (float): Duration of simulation.
    """
    
    print('#time_window: {:g}'.format(time_window))
          
    return time_window


def material(permittivity, conductivity, permeability, magconductivity, name):
    """Prints the gprMax #material command.
        
    Args:
        permittivity (float): Relative permittivity of material.
        conductivity (float): Conductivity of material.
        permeability (float): Relative permeability of material.
        magconductivity (float): Magnetic loss of material.
        name (str): Material identifier.
    """
    
    print('#material: {:g} {:g} {:g} {:g} {}'.format(permittivity, conductivity, permeability, magconductivity, name))


def geometry_view(xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, type='n'):
    """Prints the gprMax #geometry_view command.
        
    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        dx, dy, dz (float): Spatial discretisation of geometry view.
        filename (str): Filename where geometry file information will be stored.
        type (str): Can be either n (normal) or f (fine) which specifies whether to output the geometry information on a per-cell basis (n) or a per-cell-edge basis (f).
    """
    
    print('#geometry_view: {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {} {}'.format(xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, type))


def snapshot(xs, ys, zs, xf, yf, zf, dx, dy, dz, time, filename):
    """Prints the gprMax #snapshot command.
        
        Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        dx, dy, dz (float): Spatial discretisation of geometry view.
        time (float): Time in seconds (float) or the iteration number (integer) which denote the point in time at which the snapshot will be taken.
        filename (str): Filename where geometry file information will be stored.
    """
    
    if  '.' in time or 'e' in time:
        time = float(time)
        print('#snapshot: {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {}'.format(xs, ys, zs, xf, yf, zf, dx, dy, dz, time, filename))
    else:
        time = int(time)
        print('#snapshot: {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:d} {}'.format(xs, ys, zs, xf, yf, zf, dx, dy, dz, time, filename))

          
def edge(xs, ys, zs, xf, yf, zf, material):
    """Prints the gprMax #edge command.
        
    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        material (str): Material identifier.
    """
    
    print('#edge: {:g} {:g} {:g} {:g} {:g} {:g} {}'.format(xs, ys, zs, xf, yf, zf, material))
          
          
def plate(xs, ys, zs, xf, yf, zf, material):
    """Prints the gprMax #plate command.
        
    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        material (str): Material identifier(s).
    """
    
    print('#plate: {:g} {:g} {:g} {:g} {:g} {:g} {}'.format(xs, ys, zs, xf, yf, zf, material))
          
          
def triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, thickness, material):
    """Prints the gprMax #triangle command.
        
    Args:
        x1, y1, z1, x2, y2, z2, x3, y3, z3 (float): Coordinates of the vertices.
        thickness (float): Thickness for a triangular prism, or zero for a triangular patch.
        material (str): Material identifier(s).
    """
    
    print('#triangle: {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {}'.format(x1, y1, z1, x2, y2, z2, x3, y3, z3, thickness, material))
          
          
def box(xs, ys, zs, xf, yf, zf, material, averaging=''):
    """Prints the gprMax #box command.
        
    Args:
        xs, ys, zs, xf, yf, zf (float): Start and finish coordinates.
        material (str): Material identifier(s).
        averaging (str): Turn averaging on or off.
    """
    
    print('#box: {:g} {:g} {:g} {:g} {:g} {:g} {} {}'.format(xs, ys, zs, xf, yf, zf, material, averaging))
          
          
def sphere(x, y, z, radius, material, averaging=''):
    """Prints the gprMax #sphere command.
        
    Args:
        x, y, z (float): Coordinates of the centre of the sphere.
        radius (float): Radius.
        material (str): Material identifier(s).
        averaging (str): Turn averaging on or off.
    """
    
    print('#sphere: {:g} {:g} {:g} {:g} {} {}'.format(x, y, z, radius, material, averaging))
          
          
def cylinder(x1, y1, z1, x2, y2, z2, radius, material, averaging=''):
    """Prints the gprMax #cylinder command.
        
    Args:
        x1, y1, z1, x2, y2, z2 (float): Coordinates of the centres of the two faces of the cylinder.
        radius (float): Radius.
        material (str): Material identifier(s).
        averaging (str): Turn averaging on or off.
    """
    
    print('#cylinder: {:g} {:g} {:g} {:g} {:g} {:g} {:g} {} {}'.format(x1, y1, z1, x2, y2, z2, radius, material, averaging))
          
          
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
    
    print('#cylindrical_sector: {} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {} {}'.format(axis, ctr1, ctr2, t1, t2, radius, startingangle, sweptangle, material, averaging))
          
          
          
    

