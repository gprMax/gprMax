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

import numpy as np
cimport numpy as np

from gprMax.utilities import round_value
from gprMax.yee_cell_setget_rigid_ext cimport set_rigid_Ex
from gprMax.yee_cell_setget_rigid_ext cimport set_rigid_Ey
from gprMax.yee_cell_setget_rigid_ext cimport set_rigid_Ez
from gprMax.yee_cell_setget_rigid_ext cimport set_rigid_Hx
from gprMax.yee_cell_setget_rigid_ext cimport set_rigid_Hy
from gprMax.yee_cell_setget_rigid_ext cimport set_rigid_Hz
from gprMax.yee_cell_setget_rigid_ext cimport set_rigid_E
from gprMax.yee_cell_setget_rigid_ext cimport unset_rigid_E
from gprMax.yee_cell_setget_rigid_ext cimport set_rigid_H
from gprMax.yee_cell_setget_rigid_ext cimport unset_rigid_H

np.seterr(divide='raise')


cpdef bint are_clockwise(
                    float v1x,
                    float v1y,
                    float v2x,
                    float v2y
            ):
    """Find if vector 2 is clockwise relative to vector 1.

    Args:
        v1x, v1y, v2x, v2y (float): Coordinates of vectors.

    Returns:
        (boolean)
    """

    return -v1x*v2y + v1y*v2x > 0


cpdef bint is_within_radius(
                    float vx,
                    float vy,
                    float radius
            ):
    """Check if the point is within a given radius of the centre of the circle.

    Args:
        vx, vy (float): Coordinates of vector.
        radius (float): Radius.

    Returns:
        (boolean)
    """

    return vx*vx + vy*vy <= radius*radius


cpdef bint is_inside_sector(
                    float px,
                    float py,
                    float ctrx,
                    float ctry,
                    float sectorstartangle,
                    float sectorangle,
                    float radius
            ):
    """For a point to be inside a circular sector, it has to meet the following tests:
        It has to be positioned anti-clockwise from the start "arm" of the sector
        It has to be positioned clockwise from the end arm of the sector
        It has to be closer to the center of the circle than the sectors radius.
        Assumes sector start is always clockwise from sector end,
        i.e. sector defined in an anti-clockwise direction

    Args:
        px, py (float): Coordinates of point.
        ctrx, ctry (float): Coordinates of centre of circle.
        sectorstartangle (float): Angle (in radians) of start of sector.
        sectorangle (float): Angle (in radians) that sector makes.
        radius (float): Radius.

    Returns:
        (boolean)
    """

    cdef float sectorstart1, sectorstart2, sectorend1, sectorend2, relpoint1, relpoint2

    sectorstart1 = radius * np.cos(sectorstartangle)
    sectorstart2 = radius * np.sin(sectorstartangle)
    sectorend1 = radius * np.cos(sectorstartangle + sectorangle)
    sectorend2 = radius * np.sin(sectorstartangle + sectorangle)
    relpoint1 = px - ctrx
    relpoint2 = py - ctry

    return (not are_clockwise(sectorstart1, sectorstart2, relpoint1, relpoint2)
        and are_clockwise(sectorend1, sectorend2, relpoint1, relpoint2)
        and is_within_radius(relpoint1, relpoint2, radius))


cpdef bint point_in_polygon(
                    float px,
                    float py,
                    list polycoords
            ):
    """Calculates, using a ray casting algorithm, whether a point lies within a polygon.

    Args:
        px, py (float): Coordinates of point to test.
        polycoords (list): x, y tuples of coordinates that define the polygon.

    Returns:
        inside (boolean)
    """

    cdef int i
    cdef float p1x, p1y, p2x, p2y, pxints
    cdef bint inside

    # Check if point is a vertex
    if (px, py) in polycoords:
        return True

    # Check if point is on a boundary
    for i in range(len(polycoords)):
        p1 = None
        p2 = None
        if i == 0:
            p1x, p1y = polycoords[0]
            p2x, p2y = polycoords[1]
        else:
            p1x, p1y = polycoords[i - 1]
            p2x, p2y = polycoords[i]
        if p1y == p2y and p1y == py and px > min(p1x, p2x) and px < max(p1x, p2x):
            return True

    inside = False

    p1x, p1y = polycoords[0]
    for i in range(len(polycoords) + 1):
        p2x, p2y = polycoords[i % len(polycoords)]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        pxints = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= pxints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


cpdef void build_edge_x(
                    int i,
                    int j,
                    int k,
                    int numIDx,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Set x-orientated edges in the rigid and ID arrays for a Yee voxel.

    Args:
        i, j, k (int): Cell coordinates of edge.
        numIDz (int): Numeric ID of material.
        rigidE, rigidH, ID (memoryviews): Access to rigid and ID arrays.
    """

    set_rigid_Ex(i, j, k, rigidE)
    ID[0, i, j, k] = numIDx


cpdef void build_edge_y(
                    int i,
                    int j,
                    int k,
                    int numIDy,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Set y-orientated edges in the rigid and ID arrays for a Yee voxel.

    Args:
        i, j, k (int): Cell coordinates of edge.
        numIDz (int): Numeric ID of material.
        rigidE, rigidH, ID (memoryviews): Access to rigid and ID arrays.
    """

    set_rigid_Ey(i, j, k, rigidE)
    ID[1, i, j, k] = numIDy


cpdef void build_edge_z(
                    int i,
                    int j,
                    int k,
                    int numIDz,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Set z-orientated edges in the rigid and ID arrays for a Yee voxel.

    Args:
        i, j, k (int): Cell coordinates of edge.
        numIDz (int): Numeric ID of material.
        rigidE, rigidH, ID (memoryviews): Access to rigid and ID arrays.
    """

    set_rigid_Ez(i, j, k, rigidE)
    ID[2, i, j, k] = numIDz


cpdef void build_face_yz(
                    int i,
                    int j,
                    int k,
                    int numIDy,
                    int numIDz,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Set the edges of the yz-plane face of a Yell cell in the rigid and ID arrays.

    Args:
        i, j, k (int): Cell coordinates of the face.
        numIDx, numIDy (int): Numeric ID of material.
        rigidE, rigidH, ID (memoryviews): Access to rigid and ID arrays.
    """

    set_rigid_Ey(i, j, k, rigidE)
    set_rigid_Ez(i, j, k, rigidE)
    set_rigid_Ey(i, j, k + 1, rigidE)
    set_rigid_Ez(i, j + 1, k, rigidE)
    ID[1, i, j, k] = numIDy
    ID[2, i, j, k] = numIDz
    ID[1, i, j, k + 1] = numIDy
    ID[2, i, j + 1, k] = numIDz


cpdef void build_face_xz(
                    int i,
                    int j,
                    int k,
                    int numIDx,
                    int numIDz,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Set the edges of the xz-plane face of a Yell cell in the rigid and ID arrays.

    Args:
        i, j, k (int): Cell coordinates of the face.
        numIDx, numIDy (int): Numeric ID of material.
        rigidE, rigidH, ID (memoryviews): Access to rigid and ID arrays.
    """

    set_rigid_Ex(i, j, k, rigidE)
    set_rigid_Ez(i, j, k, rigidE)
    set_rigid_Ex(i, j, k + 1, rigidE)
    set_rigid_Ez(i + 1, j, k, rigidE)
    ID[0, i, j, k] = numIDx
    ID[2, i, j, k] = numIDz
    ID[0, i, j, k + 1] = numIDx
    ID[2, i + 1, j, k] = numIDz


cpdef void build_face_xy(
                    int i,
                    int j,
                    int k,
                    int numIDx,
                    int numIDy,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Set the edges of the xy-plane face of a Yell cell in the rigid and ID arrays.

    Args:
        i, j, k (int): Cell coordinates of the face.
        numIDx, numIDy (int): Numeric ID of material.
        rigidE, rigidH, ID (memoryviews): Access to rigid and ID arrays.
    """

    set_rigid_Ex(i, j, k, rigidE)
    set_rigid_Ey(i, j, k, rigidE)
    set_rigid_Ex(i, j + 1, k, rigidE)
    set_rigid_Ey(i + 1, j, k, rigidE)
    ID[0, i, j, k] = numIDx
    ID[1, i, j, k] = numIDy
    ID[0, i, j + 1, k] = numIDx
    ID[1, i + 1, j, k] = numIDy


cpdef void build_voxel(
                    int i,
                    int j,
                    int k,
                    int numID,
                    int numIDx,
                    int numIDy,
                    int numIDz,
                    bint averaging,
                    np.uint32_t[:, :, ::1] solid,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Set values in the solid, rigid and ID arrays for a Yee voxel.

    Args:
        i, j, k (int): Cell coordinates of voxel.
        numID, numIDx, numIDy, numIDz (int): Numeric ID of material.
        averaging (bint): Whether material property averaging will occur for the object.
        solid, rigidE, rigidH, ID (memoryviews): Access to solid, rigid and ID arrays.
    """

    if averaging:
        solid[i, j, k] = numID
        unset_rigid_E(i, j, k, rigidE)
        unset_rigid_H(i, j, k, rigidH)

    else:
        solid[i, j, k] = numID
        set_rigid_E(i, j, k, rigidE)
        set_rigid_H(i, j, k, rigidH)

        ID[0, i, j, k] = numIDx
        ID[0, i, j + 1, k + 1] = numIDx
        ID[0, i, j + 1, k] = numIDx
        ID[0, i, j, k + 1] = numIDx

        ID[1, i, j, k] = numIDy
        ID[1, i + 1, j, k + 1] = numIDy
        ID[1, i + 1, j, k] = numIDy
        ID[1, i, j, k + 1] = numIDy

        ID[2, i, j, k] = numIDz
        ID[2, i + 1, j + 1, k] = numIDz
        ID[2, i + 1, j, k] = numIDz
        ID[2, i, j + 1, k] = numIDz

        ID[3, i, j, k] = numIDx
        ID[3, i, j + 1, k + 1] = numIDx
        ID[3, i, j + 1, k] = numIDx
        ID[3, i, j, k + 1] = numIDx

        ID[4, i, j, k] = numIDy
        ID[4, i + 1, j, k + 1] = numIDy
        ID[4, i + 1, j, k] = numIDy
        ID[4, i, j, k + 1] = numIDy

        ID[5, i, j, k] = numIDz
        ID[5, i + 1, j + 1, k] = numIDz
        ID[5, i + 1, j, k] = numIDz
        ID[5, i, j + 1, k] = numIDz


cpdef void build_triangle(
                    float x1,
                    float y1,
                    float z1,
                    float x2,
                    float y2,
                    float z2,
                    float x3,
                    float y3,
                    float z3,
                    str normal,
                    float thickness,
                    float dx,
                    float dy,
                    float dz,
                    int numID,
                    int numIDx,
                    int numIDy,
                    int numIDz,
                    bint averaging,
                    np.uint32_t[:, :, ::1] solid,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """
    Builds #triangle and #triangular_prism commands which sets values in the
    solid, rigid and ID arrays for a Yee voxel.

    Args:
        x1, y1, z1, x2, y2, z2, x3, y3, z3 (float): Coordinates of the vertices
                of the triangular prism.
        normal (char): Normal direction to the plane of the triangular prism.
        thickness (float): Thickness of the triangular prism.
        dx, dy, dz (float): Spatial discretisation.
        numID, numIDx, numIDy, numIDz (int): Numeric ID of material.
        averaging (bint): Whether material property averaging will occur for the object.
        solid, rigidE, rigidH, ID (memoryviews): Access to solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k
    cdef int i1, i2, j1, j2, sign, level, thicknesscells
    cdef float area, s, t

    # Calculate a bounding box for the triangle
    if normal == 'x':
        area = 0.5 * (-z2 * y3 + z1 * (-y2 + y3) + y1 * (z2 - z3) + y2 * z3)
        i1 = round_value(np.amin([y1, y2, y3]) / dy) - 1
        i2 = round_value(np.amax([y1, y2, y3]) / dy) + 1
        j1 = round_value(np.amin([z1, z2, z3]) / dz) - 1
        j2 = round_value(np.amax([z1, z2, z3]) / dz) + 1
        level = round_value(x1 / dx)
        thicknesscells = round_value(thickness / dx)
    elif normal == 'y':
        area = 0.5 * (-z2 * x3 + z1 * (-x2 + x3) + x1 * (z2 - z3) + x2 * z3)
        i1 = round_value(np.amin([x1, x2, x3]) / dx) - 1
        i2 = round_value(np.amax([x1, x2, x3]) / dx) + 1
        j1 = round_value(np.amin([z1, z2, z3]) / dz) - 1
        j2 = round_value(np.amax([z1, z2, z3]) / dz) + 1
        level = round_value(y1 /dy)
        thicknesscells = round_value(thickness / dy)
    elif normal == 'z':
        area = 0.5 * (-y2 * x3 + y1 * (-x2 + x3) + x1 * (y2 - y3) + x2 * y3)
        i1 = round_value(np.amin([x1, x2, x3]) / dx) - 1
        i2 = round_value(np.amax([x1, x2, x3]) / dx) + 1
        j1 = round_value(np.amin([y1, y2, y3]) / dy) - 1
        j2 = round_value(np.amax([y1, y2, y3]) / dy) + 1
        level = round_value(z1 / dz)
        thicknesscells = round_value(thickness / dz)

    sign = np.sign(area)

    for i in range(i1, i2):
        for j in range(j1, j2):

            # Calculate the areas of the 3 triangles defined by the 3 vertices of the main triangle and the point under test
            if normal == 'x':
                ir = (i + 0.5) * dy
                jr = (j + 0.5) * dz
                s = sign * (z1 * y3 - y1 * z3 + (z3 - z1) * ir + (y1 - y3) * jr);
                t = sign * (y1 * z2 - z1 * y2 + (z1 - z2) * ir + (y2 - y1) * jr);
            elif normal == 'y':
                ir = (i + 0.5) * dx
                jr = (j + 0.5) * dz
                s = sign * (z1 * x3 - x1 * z3 + (z3 - z1) * ir + (x1 - x3) * jr);
                t = sign * (x1 * z2 - z1 * x2 + (z1 - z2) * ir + (x2 - x1) * jr);
            elif normal == 'z':
                ir = (i + 0.5) * dx
                jr = (j + 0.5) * dy
                s = sign * (y1 * x3 - x1 * y3 + (y3 - y1) * ir + (x1 - x3) * jr);
                t = sign * (x1 * y2 - y1 * x2 + (y1 - y2) * ir + (x2 - x1) * jr);

            # If these conditions are true then point is inside triangle
            if s > 0 and t > 0 and (s + t) < 2 * area * sign:
                if thicknesscells == 0:
                    if normal == 'x':
                        build_face_yz(level, i, j, numIDy, numIDz, rigidE, rigidH, ID)
                    elif normal == 'y':
                        build_face_xz(i, level, j, numIDx, numIDz, rigidE, rigidH, ID)
                    elif normal == 'z':
                        build_face_xy(i, j, level, numIDx, numIDy, rigidE, rigidH, ID)
                else:
                    for k in range(level, level + thicknesscells):
                        if normal == 'x':
                            build_voxel(k, i, j, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)
                        elif normal == 'y':
                            build_voxel(i, k, j, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)
                        elif normal == 'z':
                            build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)


cpdef void build_cylindrical_sector(
                    float ctr1,
                    float ctr2,
                    int level,
                    float sectorstartangle,
                    float sectorangle,
                    float radius,
                    str normal,
                    float thickness,
                    float dx,
                    float dy,
                    float dz,
                    int numID,
                    int numIDx,
                    int numIDy,
                    int numIDz,
                    bint averaging,
                    np.uint32_t[:, :, ::1] solid,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """
    Builds #cylindrical_sector commands which sets values in the solid, rigid
    and ID arrays for a Yee voxel. It defines a sector of cylinder given by the
    direction of the axis of the coordinates of the cylinder face centre, depth
    coordinates, sector start point, sector angle, and sector radius. N.B
    Assumes sector start is always clockwise from sector end, i.e. sector
    defined in an anti-clockwise direction.

    Args:
        ctr1, ctr2 (float): Coordinates of centre of circle.
        level (int): Third dimensional coordinate.
        sectorstartangle (float): Angle (in radians) of start of sector.
        sectorangle (float): Angle (in radians) that sector makes.
        radius (float): Radius of the cylindrical sector.
        normal (char): Normal direction to the plane of the cylindrical sector.
        thickness (float): Thickness of the cylindrical sector.
        dx, dy, dz (float): Spatial discretisation.
        numID, numIDx, numIDy, numIDz (int): Numeric ID of material.
        averaging (bint): Whether material property averaging will occur for the object.
        solid, rigidE, rigidH, ID (memoryviews): Access to solid, rigid and ID arrays.
    """

    cdef Py_ssize_t x, y, z
    cdef int x1, x2, y1, y2, z1, z2, thicknesscells

    if normal == 'x':
        # Angles are defined from zero degrees on the positive y-axis going towards positive z-axis
        y1 = round_value((ctr1 - radius)/dy)
        y2 = round_value((ctr1 + radius)/dy)
        z1 = round_value((ctr2 - radius)/dz)
        z2 = round_value((ctr2 + radius)/dz)
        thicknesscells = round_value(thickness/dx)

        # Set bounds to domain if they outside
        if y1 < 0:
            y1 = 0
        if y2 > solid.shape[1]:
            y2 = solid.shape[1]
        if z1 < 0:
            z1 = 0
        if z2 > solid.shape[2]:
            z2 = solid.shape[2]

        for y in range(y1, y2):
            for z in range(z1, z2):
                if is_inside_sector(y * dy + 0.5 * dy, z * dz + 0.5 * dz, ctr1, ctr2, sectorstartangle, sectorangle, radius):
                    if thicknesscells == 0:
                        build_face_yz(level, y, z, numIDy, numIDz, rigidE, rigidH, ID)
                    else:
                        for x in range(level, level + thicknesscells):
                            build_voxel(x, y, z, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)

    elif normal == 'y':
        # Angles are defined from zero degrees on the positive x-axis going towards positive z-axis
        x1 = round_value((ctr1 - radius)/dx)
        x2 = round_value((ctr1 + radius)/dx)
        z1 = round_value((ctr2 - radius)/dz)
        z2 = round_value((ctr2 + radius)/dz)
        thicknesscells = round_value(thickness/dy)

        # Set bounds to domain if they outside
        if x1 < 0:
            x1 = 0
        if x2 > solid.shape[0]:
            x2 = solid.shape[0]
        if z1 < 0:
            z1 = 0
        if z2 > solid.shape[2]:
            z2 = solid.shape[2]

        for x in range(x1, x2):
            for z in range(z1, z2):
                if is_inside_sector(x * dx + 0.5 * dx, z * dz + 0.5 * dz, ctr1, ctr2, sectorstartangle, sectorangle, radius):
                    if thicknesscells == 0:
                        build_face_xz(x, level, z, numIDx, numIDz, rigidE, rigidH, ID)
                    else:
                        for y in range(level, level + thicknesscells):
                            build_voxel(x, y, z, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)

    elif normal == 'z':
        # Angles are defined from zero degrees on the positive x-axis going towards positive y-axis
        x1 = round_value((ctr1 - radius)/dx)
        x2 = round_value((ctr1 + radius)/dx)
        y1 = round_value((ctr2 - radius)/dy)
        y2 = round_value((ctr2 + radius)/dy)
        thicknesscells = round_value(thickness/dz)

        # Set bounds to domain if they outside
        if x1 < 0:
            x1 = 0
        if x2 > solid.shape[0]:
            x2 = solid.shape[0]
        if y1 < 0:
            y1 = 0
        if y2 > solid.shape[1]:
            y2 = solid.shape[1]

        for x in range(x1, x2):
            for y in range(y1, y2):
                if is_inside_sector(x * dx + 0.5 * dx, y * dy + 0.5 * dy, ctr1, ctr2, sectorstartangle, sectorangle, radius):
                    if thicknesscells == 0:
                        build_face_xy(x, y, level, numIDx, numIDy, rigidE, rigidH, ID)
                    else:
                        for z in range(level, level + thicknesscells):
                            build_voxel(x, y, z, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)


cpdef void build_box(
                    int xs,
                    int xf,
                    int ys,
                    int yf,
                    int zs,
                    int zf,
                    int numID,
                    int numIDx,
                    int numIDy,
                    int numIDz,
                    bint averaging,
                    np.uint32_t[:, :, ::1] solid,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Builds #box commands which sets values in the solid, rigid and ID arrays.

    Args:
        xs, xf, ys, yf, zs, zf (int): Cell coordinates of entire box.
        numID, numIDx, numIDy, numIDz (int): Numeric ID of material.
        averaging (bint): Whether material property averaging will occur for the object.
        solid, rigidE, rigidH, ID (memoryviews): Access to solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k

    if averaging:
        for i in range(xs, xf):
            for j in range(ys, yf):
                for k in range(zs, zf):
                    solid[i, j, k] = numID
                    unset_rigid_E(i, j, k, rigidE)
                    unset_rigid_H(i, j, k, rigidH)
    else:
        for i in range(xs, xf):
            for j in range(ys, yf):
                for k in range(zs, zf):
                    solid[i, j, k] = numID
                    set_rigid_E(i, j, k, rigidE)
                    set_rigid_H(i, j, k, rigidH)

        for i in range(xs, xf):
            for j in range(ys, yf + 1):
                for k in range(zs, zf + 1):
                    ID[0, i, j, k] = numIDx

        for i in range(xs, xf + 1):
            for j in range(ys, yf):
                for k in range(zs, zf + 1):
                    ID[1, i, j, k] = numIDy

        for i in range(xs, xf + 1):
            for j in range(ys, yf + 1):
                for k in range(zs, zf):
                    ID[2, i, j, k] = numIDz

        for i in range(xs, xf + 1):
            for j in range(ys, yf):
                for k in range(zs, zf):
                    ID[3, i, j, k] = numIDx

        for i in range(xs, xf):
            for j in range(ys, yf + 1):
                for k in range(zs, zf):
                    ID[4, i, j, k] = numIDy

        for i in range(xs, xf):
            for j in range(ys, yf):
                for k in range(zs, zf + 1):
                    ID[5, i, j, k] = numIDz


cpdef void build_cylinder(
                    float x1,
                    float y1,
                    float z1,
                    float x2,
                    float y2,
                    float z2,
                    float r,
                    float dx,
                    float dy,
                    float dz,
                    int numID,
                    int numIDx,
                    int numIDy,
                    int numIDz,
                    bint averaging,
                    np.uint32_t[:, :, ::1] solid,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Builds #cylinder commands which sets values in the solid, rigid and ID arrays for a Yee voxel.

    Args:
        x1, y1, z1, x2, y2, z2 (float): Coordinates of the centres of cylinder faces.
        r (float): Radius of the cylinder.
        dx, dy, dz (float): Spatial discretisation.
        numID, numIDx, numIDy, numIDz (int): Numeric ID of material.
        averaging (bint): Whether material property averaging will occur for the object.
        solid, rigidE, rigidH, ID (memoryviews): Access to solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k
    cdef int xs, xf, ys, yf, zs, zf, xc, yc, zc
    cdef float f1f2mag, f2f1mag, f1ptmag, f2ptmag, dot1, dot2, factor1, factor2, theta1, theta2, distance1, distance2
    cdef bint build, x_align, y_align, z_align
    cdef np.ndarray f1f2, f2f1, f1pt, f2pt

    # Check if cylinder is aligned with an axis
    x_align = y_align = z_align = 0
    # x-aligned
    if round_value(y1 / dy) == round_value(y2 / dy) and round_value(z1 / dz) == round_value(z2 / dz):
        x_align = 1

    # y-aligned
    elif round_value(x1 / dx) == round_value(x2 / dx) and round_value(z1 / dz) == round_value(z2 / dz):
        y_align = 1

    # z-aligned
    elif round_value(x1 / dx) == round_value(x2 / dx) and round_value(y1 / dy) == round_value(y2 / dy):
        z_align = 1

    # Calculate a bounding box for the cylinder
    if x1 < x2:
        if x_align:
            xs = round_value(x1 / dx)
            xf = round_value(x2 / dx)
        else:
            xs = round_value((x1 - r) / dx) - 1
            xf = round_value((x2 + r) / dx) + 1
    else:
        if x_align:
            xs = round_value(x2 / dx)
            xf = round_value(x1 / dx)
        else:
            xs = round_value((x2 - r) / dx) - 1
            xf = round_value((x1 + r) / dx) + 1
    if y1 < y2:
        if y_align:
            ys = round_value(y1 / dy)
            yf = round_value(y2 / dy)
        else:
            ys = round_value((y1 - r) / dy) - 1
            yf = round_value((y2 + r) / dy) + 1
    else:
        if y_align:
            ys = round_value(y2 / dy)
            yf = round_value(y1 / dy)
        else:
            ys = round_value((y2 - r) / dy) - 1
            yf = round_value((y1 + r) / dy) + 1
    if z1 < z2:
        if z_align:
            zs = round_value(z1 / dz)
            zf = round_value(z2 / dz)
        else:
            zs = round_value((z1 - r) / dz) - 1
            zf = round_value((z2 + r) / dz) + 1
    else:
        if z_align:
            zs = round_value(z2 / dz)
            zf = round_value(z1 / dz)
        else:
            zs = round_value((z2 - r) / dz) - 1
            zf = round_value((z1 + r) / dz) + 1

    # Set bounds to domain if they outside
    if xs < 0:
        xs = 0
    if xf > solid.shape[0]:
        xf = solid.shape[0]
    if ys < 0:
        ys = 0
    if yf > solid.shape[1]:
        yf = solid.shape[1]
    if zs < 0:
        zs = 0
    if zf > solid.shape[2]:
        zf = solid.shape[2]

    # x-aligned cylinder
    if x_align:
        for j in range(ys, yf):
            for k in range(zs, zf):
                if np.sqrt((j * dy + 0.5 * dy - y1)**2 + (k * dz + 0.5 * dz - z1)**2) <= r:
                    for i in range(xs, xf):
                        build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)
    # y-aligned cylinder
    elif y_align:
        for i in range(xs, xf):
            for k in range(zs, zf):
                if np.sqrt((i * dx + 0.5 * dx - x1)**2 + (k * dz + 0.5 * dz - z1)**2) <= r:
                    for j in range(ys, yf):
                        build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)
    # z-aligned cylinder
    elif z_align:
        for i in range(xs, xf):
            for j in range(ys, yf):
                if np.sqrt((i * dx + 0.5 * dx - x1)**2 + (j * dy + 0.5 * dy - y1)**2) <= r:
                    for k in range(zs, zf):
                        build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)

    # Not aligned with any axis
    else:
        # Vectors between centres of cylinder faces
        f1f2 = np.array([x2 - x1, y2 - y1, z2 - z1], dtype=np.float32)
        f2f1 = np.array([x1 - x2, y1 - y2, z1 - z2], dtype=np.float32)

        # Magnitudes
        f1f2mag = np.sqrt((f1f2*f1f2).sum(axis=0))
        f2f1mag = np.sqrt((f2f1*f2f1).sum(axis=0))

        for i in range(xs, xf):
            for j in range(ys, yf):
                for k in range(zs, zf):
                    # Build flag - default false, set to True if point is in cylinder
                    build = 0
                    # Vector from centre of first cylinder face to test point
                    f1pt = np.array([i * dx + 0.5 * dx - x1, j * dy + 0.5 * dy - y1, k * dz + 0.5 * dz - z1], dtype=np.float32)
                    # Vector from centre of second cylinder face to test point
                    f2pt = np.array([i * dx + 0.5 * dx - x2, j * dy + 0.5 * dy - y2, k * dz + 0.5 * dz - z2], dtype=np.float32)
                    # Magnitudes
                    f1ptmag = np.sqrt((f1pt*f1pt).sum(axis=0))
                    f2ptmag = np.sqrt((f2pt*f2pt).sum(axis=0))
                    # Dot products
                    dot1 = np.dot(f1f2, f1pt)
                    dot2 = np.dot(f2f1, f2pt)

                    if f1ptmag == 0 or f2ptmag == 0:
                        build = 1
                    else:
                        factor1 = dot1 / (f1f2mag * f1ptmag)
                        factor2 = dot2 / (f2f1mag * f2ptmag)
                        # Catch cases where either factor1 or factor2 are 1
                        try:
                            theta1 = np.arccos(factor1)
                        except FloatingPointError:
                            theta1 = 0
                        try:
                            theta2 = np.arccos(factor2)
                        except FloatingPointError:
                            theta2 = 0
                        distance1 = f1ptmag * np.sin(theta1)
                        distance2 = f2ptmag * np.sin(theta2)
                        if (distance1 <= r or distance2 <= r) and theta1 <= np.pi/2 and theta2 <= np.pi/2:
                            build = 1

                    if build:
                        build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)


cpdef void build_sphere(
                    int xc,
                    int yc,
                    int zc,
                    float r,
                    float dx,
                    float dy,
                    float dz,
                    int numID,
                    int numIDx,
                    int numIDy,
                    int numIDz,
                    bint averaging,
                    np.uint32_t[:, :, ::1] solid,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Builds #sphere commands which sets values in the solid, rigid and ID arrays for a Yee voxel.

    Args:
        xc, yc, zc (int): Cell coordinates of the centre of the sphere.
        r (float): Radius of the sphere.
        dx, dy, dz (float): Spatial discretisation.
        numID, numIDx, numIDy, numIDz (int): Numeric ID of material.
        averaging (bint): Whether material property averaging will occur for the object.
        solid, rigidE, rigidH, ID (memoryviews): Access to solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k
    cdef int xs, xf, ys, yf, zs, zf

    # Calculate a bounding box for sphere
    xs = round_value(((xc * dx) - r) / dx) - 1
    xf = round_value(((xc * dx) + r) / dx) + 1
    ys = round_value(((yc * dy) - r) / dy) - 1
    yf = round_value(((yc * dy) + r) / dy) + 1
    zs = round_value(((zc * dz) - r) / dz) - 1
    zf = round_value(((zc * dz) + r) / dz) + 1

    # Set bounds to domain if they outside
    if xs < 0:
        xs = 0
    if xf > solid.shape[0]:
        xf = solid.shape[0]
    if ys < 0:
        ys = 0
    if yf > solid.shape[1]:
        yf = solid.shape[1]
    if zs < 0:
        zs = 0
    if zf > solid.shape[2]:
        zf = solid.shape[2]

    for i in range(xs, xf):
        for j in range(ys, yf):
            for k in range(zs, zf):
                if np.sqrt((i + 0.5 - xc)**2 * dx**2 + (j + 0.5 - yc)**2 * dy**2 + (k + 0.5 - zc)**2 * dz**2) <= r:
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)


cpdef void build_voxels_from_array(
                    int xs,
                    int ys,
                    int zs,
                    int numexistmaterials,
                    bint averaging,
                    np.int16_t[:, :, ::1] data,
                    np.uint32_t[:, :, ::1] solid,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Builds Yee voxels by reading integers from an array.

    Args:
        xs, ys, zs (int): Cell coordinates of position of start of array in domain.
        numexistmaterials (int): Number of existing materials in model prior to building voxels.
        averaging (bint): Whether material property averaging will occur for the object.
        data (memoryview): Access to array containing numeric IDs of voxels to create.
        solid, rigidE, rigidH, ID (memoryviews): Access to solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k
    cdef int xf, yf, zf, numID

    # Set bounds to domain if they outside
    if xs < 0:
        xs = 0
    if xs + data.shape[0] > solid.shape[0]:
        xf = solid.shape[0]
    else:
        xf = xs + data.shape[0]

    if ys < 0:
        ys = 0
    if ys + data.shape[1] > solid.shape[1]:
        yf = solid.shape[1]
    else:
        yf = ys + data.shape[1]

    if zs < 0:
        zs = 0
    if zs + data.shape[2] > solid.shape[2]:
        zf = solid.shape[2]
    else:
        zf = zs + data.shape[2]

    for i in range(xs, xf):
        for j in range(ys, yf):
            for k in range(zs, zf):
                numID = data[i - xs, j - ys, k - zs]
                if numID >= 0:
                    numID += numexistmaterials
                    build_voxel(i, j, k, numID, numID, numID, numID, averaging, solid, rigidE, rigidH, ID)


cpdef void build_voxels_from_array_mask(
                    int xs,
                    int ys,
                    int zs,
                    int waternumID,
                    int grassnumID,
                    bint averaging,
                    np.int8_t[:, :, ::1] mask,
                    np.int16_t[:, :, ::1] data,
                    np.uint32_t[:, :, ::1] solid,
                    np.int8_t[:, :, :, ::1] rigidE,
                    np.int8_t[:, :, :, ::1] rigidH,
                    np.uint32_t[:, :, :, ::1] ID
            ):
    """Builds Yee voxels by reading integers from an array.

    Args:
        xs, ys, zs (int): Cell coordinates of position of start of array in domain.
        waternumID, grassnumID (int): Numeric ID of water and grass materials.
        averaging (bint): Whether material property averaging will occur for the object.
        data (memoryview): Access to array containing numeric IDs of voxels to create.
        mask (memoryview): Access to array containing a mask of voxels to create.
        solid, rigidE, rigidH, ID (memoryviews): Access to solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k
    cdef int xf, yf, zf, numID, numIDx, numIDy, numIDz

    # Set upper bounds
    xf = xs + data.shape[0]
    yf = ys + data.shape[1]
    zf = zs + data.shape[2]

    for i in range(xs, xf):
        for j in range(ys, yf):
            for k in range(zs, zf):
                if mask[i - xs, j - ys, k - zs] == 1:
                    numID = numIDx = numIDy = numIDz = data[i - xs, j - ys, k - zs]
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)
                elif mask[i - xs, j - ys, k - zs] == 2:
                    numID = numIDx = numIDy = numIDz = waternumID
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)
                elif mask[i - xs, j - ys, k - zs] == 3:
                    numID = numIDx = numIDy = numIDz = grassnumID
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz, averaging, solid, rigidE, rigidH, ID)
