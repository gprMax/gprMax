# cython: cdivision=True
# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport ceil, cos, sin

np.seterr(divide='raise')

from gprMax.cython.yee_cell_setget_rigid cimport (
    set_rigid_E,
    set_rigid_Ex,
    set_rigid_Ey,
    set_rigid_Ez,
    set_rigid_Hx,
    set_rigid_Hy,
    set_rigid_Hz,
    unset_rigid_E,
    unset_rigid_H,
)

from gprMax.utilities.utilities import round_value


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void set_geometry_tag(
    np.uint8_t[::1] tag_bytes,
    int itemsize,
    Py_ssize_t ny,
    Py_ssize_t nz,
    Py_ssize_t i,
    Py_ssize_t j,
    Py_ssize_t k,
    unsigned int tag_id,
) noexcept nogil:
    """Write an adaptively sized tag ID to a flattened C-contiguous map."""

    cdef Py_ssize_t offset = ((i * ny + j) * nz + k) * itemsize
    if itemsize == 1:
        tag_bytes[offset] = <np.uint8_t>tag_id
    elif itemsize == 2:
        (<np.uint16_t*>&tag_bytes[offset])[0] = <np.uint16_t>tag_id
    else:
        (<np.uint32_t*>&tag_bytes[offset])[0] = <np.uint32_t>tag_id


cpdef bint are_clockwise(
    double v1x,
    double v1y,
    double v2x,
    double v2y
):
    """Find if vector 2 is clockwise relative to vector 1.

    Args:
        v1x, v1y, v2x, v2y: floats of coordinates of vectors.

    Returns:
        (boolean)
    """

    return -v1x*v2y + v1y*v2x > 0


cpdef bint is_within_radius(
    double vx,
    double vy,
    double radius
):
    """Check if the point is within a given radius of the centre of the circle.

    Args:
        vx, vy: floats of coordinates of vector.
        radius: float for radius.

    Returns:
        (boolean)
    """

    return vx*vx + vy*vy <= radius*radius


cpdef bint is_inside_sector(
    double px,
    double py,
    double ctrx,
    double ctry,
    double sectorstartangle,
    double sectorangle,
    double radius
):
    """For a point to be inside a circular sector, it has to meet the following tests:
        It has to be positioned anti-clockwise from the start "arm" of the sector
        It has to be positioned clockwise from the end arm of the sector
        It has to be closer to the center of the circle than the sectors radius.
        Assumes sector start is always clockwise from sector end,
        i.e. sector defined in an anti-clockwise direction

    Args:
        px, py: floats for coordinates of point.
        ctrx, ctry: floats for coordinates of centre of circle.
        sectorstartangle: float for angle (in radians) of start of sector.
        sectorangle: float for angle (in radians) that sector makes.
        radius: float for radius.

    Returns:
        (boolean)
    """

    cdef double sectorstart1, sectorstart2, sectorend1, sectorend2, relpoint1, relpoint2

    sectorstart1 = radius * cos(sectorstartangle)
    sectorstart2 = radius * sin(sectorstartangle)
    sectorend1 = radius * cos(sectorstartangle + sectorangle)
    sectorend2 = radius * sin(sectorstartangle + sectorangle)
    relpoint1 = px - ctrx
    relpoint2 = py - ctry

    if sectorangle <= np.pi:
        return (
            not are_clockwise(sectorstart1, sectorstart2, relpoint1, relpoint2)
            and are_clockwise(sectorend1, sectorend2, relpoint1, relpoint2)
            and is_within_radius(relpoint1, relpoint2, radius)
        )
    else:
        return (
            (
                not are_clockwise(sectorstart1, sectorstart2, relpoint1, relpoint2)
                or are_clockwise(sectorend1, sectorend2, relpoint1, relpoint2)
            ) and is_within_radius(relpoint1, relpoint2, radius)
        )


cpdef bint point_in_polygon(
    float px,
    float py,
    list polycoords
):
    """Calculates, using a ray casting algorithm, whether a point lies within a polygon.

    Args:
        px, py: float for coordinates of point to test.
        polycoords: list of x, y tuples of coordinates that define the polygon.

    Returns:
        inside: boolean
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
        i, j, k: ints for cell coordinates of edge.
        numIDx: int for numeric ID of material.
        rigidE, rigidH, ID: memoryviews to access rigid and ID arrays.
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
        i, j, k: ints for cell coordinates of edge.
        numIDy: int for numeric ID of material.
        rigidE, rigidH, ID: memoryviews to access rigid and ID arrays.
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
        i, j, k: ints for cell coordinates of edge.
        numIDz: int for numeric ID of material.
        rigidE, rigidH, ID: memoryviews to access rigid and ID arrays.
    """

    set_rigid_Ez(i, j, k, rigidE)
    ID[2, i, j, k] = numIDz


cpdef void build_magnetic_edge_x(
    int i,
    int j,
    int k,
    int numIDx,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID
):
    """Set an x-orientated magnetic edge in the rigid and ID arrays - the
    magnetic dual of build_edge_x, using the self-consistent single-position
    Hx marker (set_rigid_Hx). Has no electric counterpart to touch, unlike
    build_edge_x/y/z which take an (unused-by-them) rigidH parameter for
    signature uniformity - a magnetic edge has no such relationship to E.

    Args:
        i, j, k: ints for cell coordinates of edge.
        numIDx: int for numeric ID of material.
        rigidH, ID: memoryviews to access rigid and ID arrays.
    """

    set_rigid_Hx(i, j, k, rigidH)
    ID[3, i, j, k] = numIDx


cpdef void build_magnetic_edge_y(
    int i,
    int j,
    int k,
    int numIDy,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID
):
    """Set a y-orientated magnetic edge in the rigid and ID arrays.

    Args:
        i, j, k: ints for cell coordinates of edge.
        numIDy: int for numeric ID of material.
        rigidH, ID: memoryviews to access rigid and ID arrays.
    """

    set_rigid_Hy(i, j, k, rigidH)
    ID[4, i, j, k] = numIDy


cpdef void build_magnetic_edge_z(
    int i,
    int j,
    int k,
    int numIDz,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID
):
    """Set a z-orientated magnetic edge in the rigid and ID arrays.

    Args:
        i, j, k: ints for cell coordinates of edge.
        numIDz: int for numeric ID of material.
        rigidH, ID: memoryviews to access rigid and ID arrays.
    """

    set_rigid_Hz(i, j, k, rigidH)
    ID[5, i, j, k] = numIDz


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
        i, j, k: ints for cell coordinates of the face.
        numIDy, numIDz: int for numeric ID of material.
        rigidE, rigidH, ID: memoryviews to access rigid and ID arrays.
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
        i, j, k: ints for cell coordinates of the face.
        numIDx, numIDz: int for numeric ID of material.
        rigidE, rigidH, ID: memoryviews to access rigid and ID arrays.
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
        i, j, k: ints for cell coordinates of the face.
        numIDx, numIDy: int for numeric ID of material.
        rigidE, rigidH, ID: memoryviews to access rigid and ID arrays.
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
    bint pec_x,
    bint pec_y,
    bint pec_z,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID
) nogil:
    """Set values in the solid, rigid and ID arrays for a Yee voxel.

    Args:
        i, j, k: ints for cell coordinates of voxel.
        numID, numIDx, numIDy, numIDz: ints for numeric ID of material.
        averaging: bint for whether material property averaging will occur for
                    the object.
        pec_x, pec_y, pec_z: bints for whether the x/y/z-direction material is
                    PEC (or PEC-equivalent, se=inf). PEC has no well-defined
                    magnetic properties, so its H components are left
                    completely untouched (ID value and rigid state both kept
                    as whatever background was already there) rather than
                    being set - unlike other rigid (non-averaged) materials,
                    whose H is set at its correct 2 own-axis positions below.
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    if averaging:
        solid[i, j, k] = numID
        unset_rigid_E(i, j, k, rigidE)
        unset_rigid_H(i, j, k, rigidH)

    else:
        solid[i, j, k] = numID
        set_rigid_E(i, j, k, rigidE)

        # set_rigid_Hx/Hy/Hz are self-consistent single-position markers
        # (mirroring set_rigid_Ex/Ey/Ez's shape) - a solid cell has 2 true
        # H faces per component, so each is called twice, once per face,
        # matching the two ID writes below exactly.
        if not pec_x:
            set_rigid_Hx(i, j, k, rigidH)
            set_rigid_Hx(i + 1, j, k, rigidH)
        if not pec_y:
            set_rigid_Hy(i, j, k, rigidH)
            set_rigid_Hy(i, j + 1, k, rigidH)
        if not pec_z:
            set_rigid_Hz(i, j, k, rigidH)
            set_rigid_Hz(i, j, k + 1, rigidH)

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

        # H components have only 2 true positions each, varying along their
        # own dependency axis alone (tangential axes fixed to this cell) -
        # unlike E's 4 tangential-corner positions above. This matches the
        # established pattern in build_box(); the previous implementation
        # incorrectly mirrored the E-component pattern. PEC axes skip this
        # entirely - see the docstring above.
        if not pec_x:
            ID[3, i, j, k] = numIDx
            ID[3, i + 1, j, k] = numIDx

        if not pec_y:
            ID[4, i, j, k] = numIDy
            ID[4, i, j + 1, k] = numIDy

        if not pec_z:
            ID[5, i, j, k] = numIDz
            ID[5, i, j, k + 1] = numIDz


cpdef Py_ssize_t build_triangle(
    double x1,
    double y1,
    double z1,
    double x2,
    double y2,
    double z2,
    double x3,
    double y3,
    double z3,
    str normal,
    double thickness,
    double dx,
    double dy,
    double dz,
    int numID,
    int numIDx,
    int numIDy,
    int numIDz,
    bint averaging,
    bint pec_x,
    bint pec_y,
    bint pec_z,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0
):
    """
    Builds triangles and triangular prisms which sets values in the solid,
        rigid and ID arrays for a Yee voxel.

    Args:
        x1, y1, z1, x2, y2, z2, x3, y3, z3: floats of coordinates of the vertices
                                                of the triangular prism.
        normal: string for normal direction to the plane of the triangular prism.
        thickness: float for thickness of the triangular prism.
        dx, dy, dz: floats for spatial discretisation.
        numID, numIDx, numIDy, numIDz: ints for numeric ID of material.
        averaging: bint for whether material property averaging will occur for
                    the object.
        pec_x, pec_y, pec_z: bints for whether the x/y/z-direction material is
                    PEC (or PEC-equivalent) - see build_voxel().
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k, occupied = 0
    cdef int i1, i2, j1, j2, levelcells, thicknesscells
    cdef int u1, v1, u2, v2, u3, v3
    cdef long long bu, bv, cu, cv, qu2, qv2
    cdef long long denominator, s_num2, t_num2
    cdef bint inside
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

    # Work from snapped cell indices and relative coordinates. This avoids
    # subtracting large absolute floating-point coordinates, so translating an
    # otherwise identical object into an MPI rank or subgrid cannot change
    # which cell centres lie inside the triangle.
    if normal == 'x':
        u1, u2, u3 = round_value(y1 / dy), round_value(y2 / dy), round_value(y3 / dy)
        v1, v2, v3 = round_value(z1 / dz), round_value(z2 / dz), round_value(z3 / dz)
        i1, i2 = min(u1, u2, u3) - 1, max(u1, u2, u3) + 1
        j1, j2 = min(v1, v2, v3) - 1, max(v1, v2, v3) + 1
        levelcells = round_value(x1 / dx)
        thicknesscells = round_value(thickness / dx)
        i2, j2 = min(i2, solid.shape[1]), min(j2, solid.shape[2])
    elif normal == 'y':
        u1, u2, u3 = round_value(x1 / dx), round_value(x2 / dx), round_value(x3 / dx)
        v1, v2, v3 = round_value(z1 / dz), round_value(z2 / dz), round_value(z3 / dz)
        i1, i2 = min(u1, u2, u3) - 1, max(u1, u2, u3) + 1
        j1, j2 = min(v1, v2, v3) - 1, max(v1, v2, v3) + 1
        levelcells = round_value(y1 /dy)
        thicknesscells = round_value(thickness / dy)
        i2, j2 = min(i2, solid.shape[0]), min(j2, solid.shape[2])
    elif normal == 'z':
        u1, u2, u3 = round_value(x1 / dx), round_value(x2 / dx), round_value(x3 / dx)
        v1, v2, v3 = round_value(y1 / dy), round_value(y2 / dy), round_value(y3 / dy)
        i1, i2 = min(u1, u2, u3) - 1, max(u1, u2, u3) + 1
        j1, j2 = min(v1, v2, v3) - 1, max(v1, v2, v3) + 1
        levelcells = round_value(z1 / dz)
        thicknesscells = round_value(thickness / dz)
        i2, j2 = min(i2, solid.shape[0]), min(j2, solid.shape[1])

    # Bound to the start of the grid
    if i1 < 0:
        i1 = 0
    if j1 < 0:
        j1 = 0

    # Barycentric tests are performed entirely in snapped integer grid
    # coordinates. Scaling either in-plane axis by its cell spacing is an
    # affine transformation and therefore leaves these coordinates unchanged.
    # This also makes the zero-area test exact on every compiler/architecture.
    bu, bv = u2 - u1, v2 - v1
    cu, cv = u3 - u1, v3 - v1
    denominator = bu * cv - bv * cu
    if denominator == 0:
        return 0

    for i in range(i1, i2):
        for j in range(j1, j2):
            # Twice the cell-centre offset avoids representing 0.5 in
            # floating point. Numerators consequently share a denominator of
            # 2 * ``denominator``.
            qu2, qv2 = 2 * i + 1 - 2 * u1, 2 * j + 1 - 2 * v1
            s_num2 = qu2 * cv - qv2 * cu
            t_num2 = bu * qv2 - bv * qu2
            if denominator > 0:
                inside = (
                    s_num2 > 0
                    and t_num2 > 0
                    and s_num2 + t_num2 < 2 * denominator
                )
            else:
                inside = (
                    s_num2 < 0
                    and t_num2 < 0
                    and s_num2 + t_num2 > 2 * denominator
                )

            # Preserve the historical strict-edge convention: cells whose
            # centres lie exactly on an edge are not filled.
            if inside:
                if thicknesscells == 0:
                    if normal == 'x':
                        build_face_yz(levelcells, i, j, numIDy, numIDz,
                                      rigidE, rigidH, ID)
                    elif normal == 'y':
                        build_face_xz(i, levelcells, j, numIDx, numIDz,
                                      rigidE, rigidH, ID)
                    elif normal == 'z':
                        build_face_xy(i, j, levelcells, numIDx, numIDy,
                                      rigidE, rigidH, ID)
                    occupied += 1
                else:
                    for k in range(levelcells, levelcells + thicknesscells):
                        if normal == 'x':
                            build_voxel(k, i, j, numID, numIDx, numIDy, numIDz,
                                        averaging, pec_x, pec_y, pec_z,
                                        solid, rigidE, rigidH, ID)
                            if tag_itemsize:
                                set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                                 solid.shape[2], k, i, j, tag_id)
                        elif normal == 'y':
                            build_voxel(i, k, j, numID, numIDx, numIDy, numIDz,
                                        averaging, pec_x, pec_y, pec_z,
                                        solid, rigidE, rigidH, ID)
                            if tag_itemsize:
                                set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                                 solid.shape[2], i, k, j, tag_id)
                        elif normal == 'z':
                            build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                                        averaging, pec_x, pec_y, pec_z,
                                        solid, rigidE, rigidH, ID)
                            if tag_itemsize:
                                set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                                 solid.shape[2], i, j, k, tag_id)
                        occupied += 1

    return occupied


cpdef Py_ssize_t build_cylindrical_sector(
    double ctr1,
    double ctr2,
    double level,
    double sectorstartangle,
    double sectorangle,
    double radius,
    str normal,
    double thickness,
    double dx,
    double dy,
    double dz,
    int numID,
    int numIDx,
    int numIDy,
    int numIDz,
    bint averaging,
    bint pec_x,
    bint pec_y,
    bint pec_z,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0
):
    """
    Builds cylindrical sectors which sets values in the solid, rigid and ID
        arrays for a Yee voxel. It defines a sector of cylinder given by the
        direction of the axis of the coordinates of the cylinder face centre,
        depth coordinates, sector start point, sector angle, and sector radius.
        N.B Assumes sector start is always clockwise from sector end,
        i.e. sector defined in an anti-clockwise direction.

    Args:
        ctr1, ctr2: floats for coordinates of centre of circle.
        level: float for the third dimensional coordinate.
        sectorstartangle: float for angle (in radians) of start of sector.
        sectorangle: float for angle (in radians) that sector makes.
        radius: float for radius of the cylindrical sector.
        normal: string for normal direction to the plane of the triangular prism.
        thickness: float for thickness of the triangular prism.
        dx, dy, dz: floats for spatial discretisation.
        numID, numIDx, numIDy, numIDz: ints for numeric ID of material.
        averaging: bint for whether material property averaging will occur for
                    the object.
        pec_x, pec_y, pec_z: bints for whether the x/y/z-direction material is
                    PEC (or PEC-equivalent) - see build_voxel().
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t x, y, z, occupied = 0
    cdef int x1, x2, y1, y2, z1, z2, thicknesscells, ctr1cell, ctr2cell
    cdef int radiuscells1, radiuscells2
    cdef double rel1, rel2
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

    if normal == 'x':
        # Angles are defined from zero degrees on the positive y-axis going
        # towards positive z-axis.
        ctr1cell, ctr2cell = round_value(ctr1 / dy), round_value(ctr2 / dz)
        radiuscells1, radiuscells2 = <int>ceil(radius / dy), <int>ceil(radius / dz)
        y1, y2 = ctr1cell - radiuscells1 - 1, ctr1cell + radiuscells1 + 1
        z1, z2 = ctr2cell - radiuscells2 - 1, ctr2cell + radiuscells2 + 1
        levelcells = round_value(level / dx)
        thicknesscells = round_value(thickness / dx)

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
                rel1, rel2 = (y + 0.5 - ctr1cell) * dy, (z + 0.5 - ctr2cell) * dz
                if is_inside_sector(rel1, rel2, 0, 0, sectorstartangle, sectorangle, radius):
                    if thicknesscells == 0:
                        build_face_yz(levelcells, y, z, numIDy, numIDz,
                                      rigidE, rigidH, ID)
                        occupied += 1
                    else:
                        for x in range(levelcells, levelcells + thicknesscells):
                            build_voxel(x, y, z, numID, numIDx, numIDy, numIDz,
                                        averaging, pec_x, pec_y, pec_z,
                                        solid, rigidE, rigidH, ID)
                            if tag_itemsize:
                                set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                                 solid.shape[2], x, y, z, tag_id)
                            occupied += 1

    elif normal == 'y':
        # Angles are defined from zero degrees on the positive x-axis going
        # towards positive z-axis.
        ctr1cell, ctr2cell = round_value(ctr1 / dx), round_value(ctr2 / dz)
        radiuscells1, radiuscells2 = <int>ceil(radius / dx), <int>ceil(radius / dz)
        x1, x2 = ctr1cell - radiuscells1 - 1, ctr1cell + radiuscells1 + 1
        z1, z2 = ctr2cell - radiuscells2 - 1, ctr2cell + radiuscells2 + 1
        levelcells = round_value(level / dy)
        thicknesscells = round_value(thickness / dy)

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
                rel1, rel2 = (x + 0.5 - ctr1cell) * dx, (z + 0.5 - ctr2cell) * dz
                if is_inside_sector(rel1, rel2, 0, 0, sectorstartangle, sectorangle, radius):
                    if thicknesscells == 0:
                        build_face_xz(x, levelcells, z, numIDx, numIDz,
                                      rigidE, rigidH, ID)
                        occupied += 1
                    else:
                        for y in range(levelcells, levelcells + thicknesscells):
                            build_voxel(x, y, z, numID, numIDx, numIDy, numIDz,
                                        averaging, pec_x, pec_y, pec_z,
                                        solid, rigidE, rigidH, ID)
                            if tag_itemsize:
                                set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                                 solid.shape[2], x, y, z, tag_id)
                            occupied += 1

    elif normal == 'z':
        # Angles are defined from zero degrees on the positive x-axis going
        # towards positive y-axis.
        ctr1cell, ctr2cell = round_value(ctr1 / dx), round_value(ctr2 / dy)
        radiuscells1, radiuscells2 = <int>ceil(radius / dx), <int>ceil(radius / dy)
        x1, x2 = ctr1cell - radiuscells1 - 1, ctr1cell + radiuscells1 + 1
        y1, y2 = ctr2cell - radiuscells2 - 1, ctr2cell + radiuscells2 + 1
        levelcells = round_value(level / dz)
        thicknesscells = round_value(thickness / dz)

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
                rel1, rel2 = (x + 0.5 - ctr1cell) * dx, (y + 0.5 - ctr2cell) * dy
                if is_inside_sector(rel1, rel2, 0, 0, sectorstartangle, sectorangle, radius):
                    if thicknesscells == 0:
                        build_face_xy(x, y, levelcells, numIDx, numIDy,
                                      rigidE, rigidH, ID)
                        occupied += 1
                    else:
                        for z in range(levelcells, levelcells + thicknesscells):
                            build_voxel(x, y, z, numID, numIDx, numIDy, numIDz,
                                        averaging, pec_x, pec_y, pec_z,
                                        solid, rigidE, rigidH, ID)
                            if tag_itemsize:
                                set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                                 solid.shape[2], x, y, z, tag_id)
                            occupied += 1

    return occupied


cpdef Py_ssize_t build_box(
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
    bint pec_x,
    bint pec_y,
    bint pec_z,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0
):
    """Builds boxes which sets values in the solid, rigid and ID arrays.

    Args:
        xs, xf, ys, yf, zs, zf: ints for cell coordinates of entire box.
        numID, numIDx, numIDy, numIDz: ints for numeric ID of material.
        averaging: bint for whether material property averaging will occur for
                    the object.
        pec_x, pec_y, pec_z: bints for whether the x/y/z-direction material is
                    PEC (or PEC-equivalent) - see build_voxel().
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        if tag_array.shape != (solid.shape[0], solid.shape[1], solid.shape[2]):
            raise ValueError("Geometry tag map shape must match the solid array")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

    if averaging:
        for i in range(xs, xf):
            for j in range(ys, yf):
                for k in range(zs, zf):
                    solid[i, j, k] = numID
                    if tag_itemsize:
                        set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1], solid.shape[2],
                                         i, j, k, tag_id)
                    unset_rigid_E(i, j, k, rigidE)
                    unset_rigid_H(i, j, k, rigidH)
    else:
        for i in range(xs, xf):
            for j in range(ys, yf):
                for k in range(zs, zf):
                    solid[i, j, k] = numID
                    if tag_itemsize:
                        set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1], solid.shape[2],
                                         i, j, k, tag_id)
                    set_rigid_E(i, j, k, rigidE)

        # Each E/H component gets its own full-range loop. Ex/Ey/Ez are
        # node-based on their two tangential axes, so those need the full
        # ys..yf/zs..zf node range (+1), not just the cell range - matches
        # the pre-2022 (pre-9c1b6f06) structure, which this restores. See
        # project_2d_mode_framework.md for the investigation that found
        # the regression (narrowed to single-line patches when prange was
        # added in 9c1b6f06, and never restored when prange was later
        # removed as a performance regression in b96ef3c1).
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

        # PEC has no well-defined magnetic properties, so a PEC axis's H is
        # left completely untouched (background ID/rigid state kept as-is)
        # rather than set - see build_voxel()'s docstring for the rationale.
        # set_rigid_Hx/Hy/Hz are self-consistent single-position markers,
        # so calling them once per position in these full-range loops
        # (rather than once per cell in the loop above) correctly marks
        # every position exactly once, with no redundant double-calls at
        # shared interior boundaries.
        if not pec_x:
            for i in range(xs, xf + 1):
                for j in range(ys, yf):
                    for k in range(zs, zf):
                        set_rigid_Hx(i, j, k, rigidH)
                        ID[3, i, j, k] = numIDx

        if not pec_y:
            for i in range(xs, xf):
                for j in range(ys, yf + 1):
                    for k in range(zs, zf):
                        set_rigid_Hy(i, j, k, rigidH)
                        ID[4, i, j, k] = numIDy

        if not pec_z:
            for i in range(xs, xf):
                for j in range(ys, yf):
                    for k in range(zs, zf + 1):
                        set_rigid_Hz(i, j, k, rigidH)
                        ID[5, i, j, k] = numIDz

    return <Py_ssize_t>(xf - xs) * (yf - ys) * (zf - zs)


cpdef Py_ssize_t build_cylinder(
    double x1,
    double y1,
    double z1,
    double x2,
    double y2,
    double z2,
    double r,
    double dx,
    double dy,
    double dz,
    int numID,
    int numIDx,
    int numIDy,
    int numIDz,
    bint averaging,
    bint pec_x,
    bint pec_y,
    bint pec_z,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0
):
    """Builds cylinders which sets values in the solid, rigid and ID arrays for
        a Yee voxel.

    Args:
        x1, y1, z1, x2, y2, z2: floats for coordinates of the centres of cylinder
                                faces.
        r: float for radius of the cylinder.
        dx, dy, dz: floats for spatial discretisation.
        numID, numIDx, numIDy, numIDz: ints for numeric ID of material.
        averaging: bint for whether material property averaging will occur for
                    the object.
        pec_x, pec_y, pec_z: bints for whether the x/y/z-direction material is
                    PEC (or PEC-equivalent) - see build_voxel().
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k, occupied = 0
    cdef int xs, xf, ys, yf, zs, zf
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef int rx, ry, rz
    cdef double ax, ay, az, axis2, qx, qy, qz, projection, t
    cdef double radial2, radius2
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

    ix1, iy1, iz1 = round_value(x1 / dx), round_value(y1 / dy), round_value(z1 / dz)
    ix2, iy2, iz2 = round_value(x2 / dx), round_value(y2 / dy), round_value(z2 / dz)
    ax, ay, az = (ix2 - ix1) * dx, (iy2 - iy1) * dy, (iz2 - iz1) * dz
    axis2 = ax * ax + ay * ay + az * az
    if axis2 == 0:
        return 0

    rx, ry, rz = <int>ceil(r / dx), <int>ceil(r / dy), <int>ceil(r / dz)
    xs, xf = max(0, min(ix1, ix2) - rx - 1), min(solid.shape[0], max(ix1, ix2) + rx + 1)
    ys, yf = max(0, min(iy1, iy2) - ry - 1), min(solid.shape[1], max(iy1, iy2) + ry + 1)
    zs, zf = max(0, min(iz1, iz2) - rz - 1), min(solid.shape[2], max(iz1, iz2) + rz + 1)
    radius2 = r * r

    for i in range(xs, xf):
        for j in range(ys, yf):
            for k in range(zs, zf):
                qx, qy, qz = (i + 0.5 - ix1) * dx, (j + 0.5 - iy1) * dy, (k + 0.5 - iz1) * dz
                projection = qx * ax + qy * ay + qz * az
                t = projection / axis2
                if 0 <= t <= 1:
                    radial2 = qx * qx + qy * qy + qz * qz - projection * projection / axis2
                    if radial2 <= radius2:
                        build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                                    averaging, pec_x, pec_y, pec_z,
                                    solid, rigidE, rigidH, ID)
                        if tag_itemsize:
                            set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                             solid.shape[2], i, j, k, tag_id)
                        occupied += 1

    return occupied


cpdef Py_ssize_t build_cone(
    double x1,
    double y1,
    double z1,
    double x2,
    double y2,
    double z2,
    double r1,
    double r2,
    double dx,
    double dy,
    double dz,
    int numID,
    int numIDx,
    int numIDy,
    int numIDz,
    bint averaging,
    bint pec_x,
    bint pec_y,
    bint pec_z,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0
):
    """Builds cones which sets values in the solid, rigid and ID arrays for
        a Yee voxel.

    Args:
        x1, y1, z1, x2, y2, z2: floats for coordinates of the centres of the cone
                                faces.
        r1: float for radius of the first face of the cone.
        r2: float for radius of the second face of the cone.
        dx, dy, dz: floats for spatial discretisation.
        numID, numIDx, numIDy, numIDz: ints for numeric ID of material.
        averaging: bint for whether material property averaging will occur for
                    the object.
        pec_x, pec_y, pec_z: bints for whether the x/y/z-direction material is
                    PEC (or PEC-equivalent) - see build_voxel().
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k, occupied = 0
    cdef int xs, xf, ys, yf, zs, zf
    cdef int ix1, iy1, iz1, ix2, iy2, iz2
    cdef int rx, ry, rz
    cdef double ax, ay, az, axis2, qx, qy, qz, projection, t
    cdef double radial2, radius, Rmax
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

    ix1, iy1, iz1 = round_value(x1 / dx), round_value(y1 / dy), round_value(z1 / dz)
    ix2, iy2, iz2 = round_value(x2 / dx), round_value(y2 / dy), round_value(z2 / dz)
    ax, ay, az = (ix2 - ix1) * dx, (iy2 - iy1) * dy, (iz2 - iz1) * dz
    axis2 = ax * ax + ay * ay + az * az
    if axis2 == 0:
        return 0

    Rmax = max(r1, r2)
    rx, ry, rz = <int>ceil(Rmax / dx), <int>ceil(Rmax / dy), <int>ceil(Rmax / dz)
    xs, xf = max(0, min(ix1, ix2) - rx - 1), min(solid.shape[0], max(ix1, ix2) + rx + 1)
    ys, yf = max(0, min(iy1, iy2) - ry - 1), min(solid.shape[1], max(iy1, iy2) + ry + 1)
    zs, zf = max(0, min(iz1, iz2) - rz - 1), min(solid.shape[2], max(iz1, iz2) + rz + 1)

    for i in range(xs, xf):
        for j in range(ys, yf):
            for k in range(zs, zf):
                qx, qy, qz = (i + 0.5 - ix1) * dx, (j + 0.5 - iy1) * dy, (k + 0.5 - iz1) * dz
                projection = qx * ax + qy * ay + qz * az
                t = projection / axis2
                if 0 <= t <= 1:
                    radius = r1 + t * (r2 - r1)
                    radial2 = qx * qx + qy * qy + qz * qz - projection * projection / axis2
                    if radial2 <= radius * radius:
                        build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                                    averaging, pec_x, pec_y, pec_z,
                                    solid, rigidE, rigidH, ID)
                        if tag_itemsize:
                            set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                             solid.shape[2], i, j, k, tag_id)
                        occupied += 1

    return occupied


cpdef Py_ssize_t build_sphere(
    int xc,
    int yc,
    int zc,
    double r,
    double dx,
    double dy,
    double dz,
    int numID,
    int numIDx,
    int numIDy,
    int numIDz,
    bint averaging,
    bint pec_x,
    bint pec_y,
    bint pec_z,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0
):
    """Builds spheres which sets values in the solid, rigid and ID arrays for
        a Yee voxel.

    Args:
        xc, yc, zc: ints for cell coordinates of the centre of the sphere.
        r: float for radius of the sphere.
        dx, dy, dz: floats for spatial discretisation.
        numID, numIDx, numIDy, numIDz: ints for numeric ID of material.
        averaging: bint for whether material property averaging will occur for
                    the object.
        pec_x, pec_y, pec_z: bints for whether the x/y/z-direction material is
                    PEC (or PEC-equivalent) - see build_voxel().
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k, occupied = 0
    cdef int xs, xf, ys, yf, zs, zf, rx, ry, rz
    cdef double qx, qy, qz
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

    # Calculate a bounding box for sphere
    rx, ry, rz = <int>ceil(r / dx), <int>ceil(r / dy), <int>ceil(r / dz)
    xs, xf = xc - rx - 1, xc + rx + 1
    ys, yf = yc - ry - 1, yc + ry + 1
    zs, zf = zc - rz - 1, zc + rz + 1

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
                qx, qy, qz = (i + 0.5 - xc) * dx, (j + 0.5 - yc) * dy, (k + 0.5 - zc) * dz
                if qx * qx + qy * qy + qz * qz <= r * r:
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                                averaging, pec_x, pec_y, pec_z,
                                solid, rigidE, rigidH, ID)
                    if tag_itemsize:
                        set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                         solid.shape[2], i, j, k, tag_id)
                    occupied += 1

    return occupied


cpdef Py_ssize_t build_ellipsoid(
    int xc,
    int yc,
    int zc,
    double xr,
    double yr,
    double zr,
    double dx,
    double dy,
    double dz,
    int numID,
    int numIDx,
    int numIDy,
    int numIDz,
    bint averaging,
    bint pec_x,
    bint pec_y,
    bint pec_z,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0
):
    """Builds ellipsoids which sets values in the solid, rigid and ID arrays for
        a Yee voxel.

    Args:
        xc, yc, zc: ints for cell coordinates of the centre of the ellipsoid.
        xr: float for x-semiaxis of the elliposid.
        yr: float for y-semiaxis of the elliposid.
        zr: float for z-semiaxis of the elliposid.
        dx, dy, dz: floats for spatial discretisation.
        numID, numIDx, numIDy, numIDz: ints for numeric ID of material.
        averaging: bint for whether material property averaging will occur for
                    the object.
        pec_x, pec_y, pec_z: bints for whether the x/y/z-direction material is
                    PEC (or PEC-equivalent) - see build_voxel().
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k, occupied = 0
    cdef int xs, xf, ys, yf, zs, zf, rxcells, rycells, rzcells
    cdef double qx, qy, qz
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

    # Calculate an origin-independent bounding box.
    rxcells, rycells, rzcells = <int>ceil(xr / dx), <int>ceil(yr / dy), <int>ceil(zr / dz)
    xs, xf = xc - rxcells - 1, xc + rxcells + 1
    ys, yf = yc - rycells - 1, yc + rycells + 1
    zs, zf = zc - rzcells - 1, zc + rzcells + 1

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
                qx, qy, qz = (i + 0.5 - xc) * dx, (j + 0.5 - yc) * dy, (k + 0.5 - zc) * dz
                if (qx * qx) / (xr * xr) + (qy * qy) / (yr * yr) + (qz * qz) / (zr * zr) <= 1:
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                                averaging, pec_x, pec_y, pec_z,
                                solid, rigidE, rigidH, ID)
                    if tag_itemsize:
                        set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1],
                                         solid.shape[2], i, j, k, tag_id)
                    occupied += 1

    return occupied


cpdef void build_voxels_from_array(
    int xs,
    int ys,
    int zs,
    int numexistmaterials,
    bint averaging,
    np.uint8_t[::1] is_pec_lookup,
    np.uint8_t[::1] is_averagable_lookup,
    np.int16_t[:, :, ::1] data,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0
):
    """Builds Yee voxels by reading integers from an array.

    Args:
        xs, ys, zs: ints for cell coordinates of position of start of array in
                    domain.
        numexistmaterials: int for number of existing materials in model prior
                            to building voxels.
        averaging: bint for whether material property averaging will occur for
                    the object, requested by the user/grid default - combined
                    per-voxel with is_averagable_lookup below, since a mixing
                    model (e.g. #material_list/#material_range with a fractal
                    box) may reference a non-averagable material (PEC/PMC, or
                    any custom se=inf/sm=inf material) for only some of its
                    bins - those voxels must always take the rigid path
                    regardless of the requested averaging, matching how
                    Box/Cylinder/etc. already gate on materials[0].averagable.
        is_pec_lookup: memoryview indexed by numID, True where that material is
                    PEC (or PEC-equivalent) - see build_voxel().
        is_averagable_lookup: memoryview indexed by numID, True where that
                    material permits dielectric smoothing (Material.averagable).
        data: memoryview to access array containing numeric IDs of voxels to create.
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k
    cdef int xf, yf, zf, numID
    cdef bint pec, voxel_averaging
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

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
                    numID = numID + numexistmaterials
                    pec = is_pec_lookup[numID]
                    voxel_averaging = averaging and is_averagable_lookup[numID]
                    build_voxel(i, j, k, numID, numID, numID, numID, voxel_averaging,
                                pec, pec, pec, solid, rigidE, rigidH, ID)
                    if tag_itemsize:
                        set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1], solid.shape[2],
                                         i, j, k, tag_id)


cpdef void build_voxels_from_array_mask(
    int xs,
    int ys,
    int zs,
    int waternumID,
    int grassnumID,
    bint averaging,
    np.uint8_t[::1] is_pec_lookup,
    np.uint8_t[::1] is_averagable_lookup,
    np.int8_t[:, :, ::1] mask,
    np.int16_t[:, :, ::1] data,
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    object tag_data=None,
    unsigned int tag_id=0,
    unsigned int water_tag_id=0,
    unsigned int grass_tag_id=0
):
    """Builds Yee voxels by reading integers from an array.

    Args:
        xs, ys, zs: ints for cell coordinates of position of start of array in domain.
        waternumID, grassnumID: ints for numeric ID of water and grass materials.
        averaging: bint for whether material property averaging will occur for
                the object, requested by the user/grid default - see
                build_voxels_from_array() for why this is combined per-voxel
                with is_averagable_lookup rather than applied uniformly.
        is_pec_lookup: memoryview indexed by numID, True where that material is
                    PEC (or PEC-equivalent) - see build_voxel().
        is_averagable_lookup: memoryview indexed by numID, True where that
                    material permits dielectric smoothing (Material.averagable).
        data: memoryview to access array containing numeric IDs of voxels to create.
        mask: memoryview to access to array containing a mask of voxels to create.
        solid, rigidE, rigidH, ID: memoryviews to access solid, rigid and ID arrays.
    """

    cdef Py_ssize_t i, j, k
    cdef int xf, yf, zf, numID, numIDx, numIDy, numIDz
    cdef bint pec, voxel_averaging
    cdef int tag_itemsize = 0
    cdef np.uint8_t[::1] tag_bytes
    cdef object tag_array

    if tag_data is not None:
        tag_array = np.asarray(tag_data)
        if tag_array.ndim != 3 or not tag_array.flags.c_contiguous:
            raise ValueError("Geometry tag map must be a C-contiguous 3-D array")
        if tag_array.dtype not in (np.uint8, np.uint16, np.uint32):
            raise TypeError("Geometry tag map must use uint8, uint16, or uint32")
        tag_itemsize = tag_array.itemsize
        tag_bytes = tag_array.view(np.uint8).reshape(-1)

    # Set upper bounds
    xf = xs + data.shape[0]
    yf = ys + data.shape[1]
    zf = zs + data.shape[2]

    for i in range(xs, xf):
        for j in range(ys, yf):
            for k in range(zs, zf):
                if mask[i - xs, j - ys, k - zs] == 1:
                    numID = numIDx = numIDy = numIDz = data[i - xs, j - ys, k - zs]
                    pec = is_pec_lookup[numID]
                    voxel_averaging = averaging and is_averagable_lookup[numID]
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                                voxel_averaging, pec, pec, pec, solid, rigidE, rigidH, ID)
                    if tag_itemsize:
                        set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1], solid.shape[2],
                                         i, j, k, tag_id)
                elif mask[i - xs, j - ys, k - zs] == 2:
                    numID = numIDx = numIDy = numIDz = waternumID
                    pec = is_pec_lookup[numID]
                    voxel_averaging = averaging and is_averagable_lookup[numID]
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                                voxel_averaging, pec, pec, pec, solid, rigidE, rigidH, ID)
                    if tag_itemsize:
                        set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1], solid.shape[2],
                                         i, j, k, water_tag_id)
                elif mask[i - xs, j - ys, k - zs] == 3:
                    numID = numIDx = numIDy = numIDz = grassnumID
                    pec = is_pec_lookup[numID]
                    voxel_averaging = averaging and is_averagable_lookup[numID]
                    build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                                voxel_averaging, pec, pec, pec, solid, rigidE, rigidH, ID)
                    if tag_itemsize:
                        set_geometry_tag(tag_bytes, tag_itemsize, solid.shape[1], solid.shape[2],
                                         i, j, k, grass_tag_id)
