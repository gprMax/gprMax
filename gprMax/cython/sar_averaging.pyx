# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Compiled cuboid integrals and target-mass cube searches for SAR."""

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport ceil, floor, isfinite


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _integral_at(
    const double[:, :, :] prefix,
    const double* spacing,
    double x,
    double y,
    double z,
) noexcept nogil:
    cdef:
        double q[3]
        double frac[3]
        Py_ssize_t lo[3]
        Py_ssize_t shape[3]
        int ox, oy, oz
        double wx, wy, wz, value = 0
    shape[0] = prefix.shape[0] - 1
    shape[1] = prefix.shape[1] - 1
    shape[2] = prefix.shape[2] - 1
    q[0] = min(max(x / spacing[0], 0.0), <double>shape[0])
    q[1] = min(max(y / spacing[1], 0.0), <double>shape[1])
    q[2] = min(max(z / spacing[2], 0.0), <double>shape[2])
    for ox in range(3):
        lo[ox] = <Py_ssize_t>floor(q[ox])
        frac[ox] = q[ox] - lo[ox]
        if lo[ox] == shape[ox]:
            lo[ox] -= 1
            frac[ox] = 1.0
    for ox in range(2):
        wx = frac[0] if ox else 1.0 - frac[0]
        for oy in range(2):
            wy = frac[1] if oy else 1.0 - frac[1]
            for oz in range(2):
                wz = frac[2] if oz else 1.0 - frac[2]
                value += wx * wy * wz * prefix[lo[0] + ox, lo[1] + oy, lo[2] + oz]
    return value


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _cuboid(
    const double[:, :, :] prefix,
    const double* spacing,
    const double* lower,
    const double* upper,
) noexcept nogil:
    cdef int bx, by, bz
    cdef double x, y, z, value = 0
    for bx in range(2):
        x = upper[0] if bx else lower[0]
        for by in range(2):
            y = upper[1] if by else lower[1]
            for bz in range(2):
                z = upper[2] if bz else lower[2]
                if (bx + by + bz) % 2:
                    value += _integral_at(prefix, spacing, x, y, z)
                else:
                    value -= _integral_at(prefix, spacing, x, y, z)
    return max(0.0, value)


def cuboid_integral(
    const double[:, :, ::1] prefix,
    const double[::1] spacing,
    const double[::1] lower,
    const double[::1] upper,
):
    return _cuboid(prefix, &spacing[0], &lower[0], &upper[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def centered_shells_touch_tissue(
    const unsigned char[:, :, ::1] tissue,
    const int[::1] cell,
    double side,
    double minimum_spacing,
    int delta_min,
):
    """Return whether every checked centred-cube shell face touches tissue."""

    cdef:
        int final_delta = max(
            0, <int>ceil((side / minimum_spacing - 1.0) / 2.0 - 1e-12)
        )
        int first_delta = delta_min + 1 if delta_min > 0 else 0
        int delta, i, j, k
        int lo0, lo1, lo2, hi0, hi1, hi2
        unsigned char fm, fp, gm, gp, hm, hp
    for delta in range(first_delta, final_delta + 1):
        lo0 = cell[0] - delta
        lo1 = cell[1] - delta
        lo2 = cell[2] - delta
        hi0 = cell[0] + delta
        hi1 = cell[1] + delta
        hi2 = cell[2] + delta
        if (
            lo0 < 0 or lo1 < 0 or lo2 < 0
            or hi0 >= tissue.shape[0]
            or hi1 >= tissue.shape[1]
            or hi2 >= tissue.shape[2]
        ):
            return False
        fm = fp = gm = gp = hm = hp = 0
        for i in range(lo0, hi0 + 1):
            for j in range(lo1, hi1 + 1):
                hm |= tissue[i, j, lo2]
                hp |= tissue[i, j, hi2]
            for k in range(lo2, hi2 + 1):
                gm |= tissue[i, lo1, k]
                gp |= tissue[i, hi1, k]
        for j in range(lo1, hi1 + 1):
            for k in range(lo2, hi2 + 1):
                fm |= tissue[lo0, j, k]
                fp |= tissue[hi0, j, k]
        if not (fm and fp and gm and gp and hm and hp):
            return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
def find_mass_cube(
    const double[:, :, ::1] mass_prefix,
    const double[:, :, ::1] absorbed_prefix,
    const double[:, :, ::1] tissue_prefix,
    const double[::1] spacing,
    const int[::1] cell,
    double target_mass,
    int orientation,
):
    cdef:
        double domain[3]
        double center[3]
        double lower[3]
        double upper[3]
        double low = 0
        double high
        double side
        double maximum_side
        double axial
        double transverse
        double mass = 0
        double absorbed
        double tissue_volume
        double mass_tolerance = max(1e-12 * target_mass, np.finfo(np.float64).eps)
        int axis, positive, iteration, item
    for item in range(3):
        domain[item] = (mass_prefix.shape[item] - 1) * spacing[item]
        center[item] = (cell[item] + 0.5) * spacing[item]
    if orientation == 6:
        maximum_side = 2 * min(
            center[0], domain[0] - center[0],
            center[1], domain[1] - center[1],
            center[2], domain[2] - center[2],
        )
    else:
        axis = orientation // 2
        positive = orientation % 2 == 0
        axial = (
            domain[axis] - center[axis] + 0.5 * spacing[axis]
            if positive else center[axis] + 0.5 * spacing[axis]
        )
        maximum_side = axial
        for item in range(3):
            if item != axis:
                transverse = 2 * min(center[item], domain[item] - center[item])
                maximum_side = min(maximum_side, transverse)
    if maximum_side <= 0:
        return None
    high = min(spacing[0], spacing[1], spacing[2])
    for iteration in range(64):
        _set_bounds(center, &spacing[0], high, orientation, lower, upper)
        mass = _cuboid(mass_prefix, &spacing[0], lower, upper)
        if mass >= target_mass:
            break
        low = high
        if high >= maximum_side:
            return None
        high = min(2 * high, maximum_side)
    else:
        return None
    for iteration in range(64):
        side = 0.5 * (low + high)
        _set_bounds(center, &spacing[0], side, orientation, lower, upper)
        mass = _cuboid(mass_prefix, &spacing[0], lower, upper)
        if abs(mass - target_mass) <= mass_tolerance:
            break
        if mass < target_mass:
            low = side
        else:
            high = side
    side = 0.5 * (low + high)
    _set_bounds(center, &spacing[0], side, orientation, lower, upper)
    mass = _cuboid(mass_prefix, &spacing[0], lower, upper)
    if mass <= 0:
        return None
    absorbed = _cuboid(absorbed_prefix, &spacing[0], lower, upper)
    tissue_volume = _cuboid(tissue_prefix, &spacing[0], lower, upper)
    return (
        side,
        np.asarray((lower[0], lower[1], lower[2]), dtype=np.float64),
        np.asarray((upper[0], upper[1], upper[2]), dtype=np.float64),
        mass,
        absorbed,
        tissue_volume,
    )


cdef inline void _set_bounds(
    double* center,
    const double* spacing,
    double side,
    int orientation,
    double* lower,
    double* upper,
) noexcept nogil:
    cdef int item, axis, positive
    cdef double half_cell
    for item in range(3):
        lower[item] = center[item] - 0.5 * side
        upper[item] = center[item] + 0.5 * side
    if orientation < 6:
        axis = orientation // 2
        positive = orientation % 2 == 0
        half_cell = 0.5 * spacing[axis]
        if positive:
            lower[axis] = center[axis] - half_cell
            upper[axis] = lower[axis] + side
        else:
            upper[axis] = center[axis] + half_cell
            lower[axis] = upper[axis] - side


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _find_mass_geometry(
    const double[:, :, :] mass_prefix,
    const double[:, :, :] tissue_prefix,
    const double* spacing,
    const int* cell,
    double target_mass,
    int orientation,
    double* result_side,
    double* result_mass,
    double* result_tissue_volume,
) noexcept nogil:
    """Find target-mass cube geometry without evaluating absorbed power."""

    cdef:
        double domain[3]
        double center[3]
        double lower[3]
        double upper[3]
        double low = 0
        double high
        double side
        double maximum_side
        double axial
        double transverse
        double mass = 0
        double mass_tolerance = max(1e-12 * target_mass, 2.220446049250313e-16)
        int axis, positive, iteration, item
    for item in range(3):
        domain[item] = (mass_prefix.shape[item] - 1) * spacing[item]
        center[item] = (cell[item] + 0.5) * spacing[item]
    if orientation == 6:
        maximum_side = 2 * min(
            center[0], domain[0] - center[0],
            center[1], domain[1] - center[1],
            center[2], domain[2] - center[2],
        )
    else:
        axis = orientation // 2
        positive = orientation % 2 == 0
        axial = (
            domain[axis] - center[axis] + 0.5 * spacing[axis]
            if positive else center[axis] + 0.5 * spacing[axis]
        )
        maximum_side = axial
        for item in range(3):
            if item != axis:
                transverse = 2 * min(center[item], domain[item] - center[item])
                maximum_side = min(maximum_side, transverse)
    if maximum_side <= 0:
        return False
    high = min(spacing[0], spacing[1], spacing[2])
    for iteration in range(64):
        _set_bounds(center, spacing, high, orientation, lower, upper)
        mass = _cuboid(mass_prefix, spacing, lower, upper)
        if mass >= target_mass:
            break
        low = high
        if high >= maximum_side:
            return False
        high = min(2 * high, maximum_side)
    else:
        return False
    for iteration in range(64):
        side = 0.5 * (low + high)
        _set_bounds(center, spacing, side, orientation, lower, upper)
        mass = _cuboid(mass_prefix, spacing, lower, upper)
        if abs(mass - target_mass) <= mass_tolerance:
            break
        if mass < target_mass:
            low = side
        else:
            high = side
    side = 0.5 * (low + high)
    _set_bounds(center, spacing, side, orientation, lower, upper)
    mass = _cuboid(mass_prefix, spacing, lower, upper)
    if mass <= 0:
        return False
    result_side[0] = side
    result_mass[0] = mass
    result_tissue_volume[0] = _cuboid(tissue_prefix, spacing, lower, upper)
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _shells_touch_tissue(
    const unsigned char[:, :, :] tissue,
    const int* cell,
    double side,
    double minimum_spacing,
    int delta_min,
) noexcept nogil:
    cdef:
        int final_delta = max(
            0, <int>ceil((side / minimum_spacing - 1.0) / 2.0 - 1e-12)
        )
        int first_delta = delta_min + 1 if delta_min > 0 else 0
        int delta, i, j, k
        int lo0, lo1, lo2, hi0, hi1, hi2
        unsigned char fm, fp, gm, gp, hm, hp
    for delta in range(first_delta, final_delta + 1):
        lo0 = cell[0] - delta
        lo1 = cell[1] - delta
        lo2 = cell[2] - delta
        hi0 = cell[0] + delta
        hi1 = cell[1] + delta
        hi2 = cell[2] + delta
        if (
            lo0 < 0 or lo1 < 0 or lo2 < 0
            or hi0 >= tissue.shape[0]
            or hi1 >= tissue.shape[1]
            or hi2 >= tissue.shape[2]
        ):
            return False
        fm = fp = gm = gp = hm = hp = 0
        for i in range(lo0, hi0 + 1):
            for j in range(lo1, hi1 + 1):
                hm |= tissue[i, j, lo2]
                hp |= tissue[i, j, hi2]
            for k in range(lo2, hi2 + 1):
                gm |= tissue[i, lo1, k]
                gp |= tissue[i, hi1, k]
        for j in range(lo1, hi1 + 1):
            for k in range(lo2, hi2 + 1):
                fm |= tissue[lo0, j, k]
                fp |= tissue[hi0, j, k]
        if not (fm and fp and gm and gp and hm and hp):
            return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _build_one_centered_candidate(
    const double[:, :, :] mass_prefix,
    const double[:, :, :] tissue_prefix,
    const unsigned char[:, :, :] tissue,
    const double* spacing,
    int c0,
    int c1,
    int c2,
    double target_mass,
    double maximum_background_fraction,
    int delta_min,
    Py_ssize_t output_index,
    double[:] sides,
    double[:] masses,
    unsigned char[:] valid,
) noexcept nogil:
    cdef:
        int cell[3]
        double side, mass, tissue_volume, cube_volume, background
        double minimum_spacing = min(spacing[0], spacing[1], spacing[2])
    cell[0] = c0
    cell[1] = c1
    cell[2] = c2
    if not _find_mass_geometry(
        mass_prefix,
        tissue_prefix,
        spacing,
        cell,
        target_mass,
        6,
        &side,
        &mass,
        &tissue_volume,
    ):
        return
    cube_volume = side * side * side
    background = max(0.0, 1.0 - tissue_volume / cube_volume)
    if background > maximum_background_fraction:
        return
    if not _shells_touch_tissue(
        tissue, cell, side, minimum_spacing, delta_min
    ):
        return
    sides[output_index] = side
    masses[output_index] = mass
    valid[output_index] = 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _build_one_face_plan(
    const double[:, :, :] mass_prefix,
    const double[:, :, :] tissue_prefix,
    const double* spacing,
    int c0,
    int c1,
    int c2,
    double target_mass,
    Py_ssize_t output_index,
    double[:, :] sides,
    double[:, :] masses,
) noexcept nogil:
    cdef:
        int orientation
        int cell[3]
        double side, mass, tissue_volume
    cell[0] = c0
    cell[1] = c1
    cell[2] = c2
    for orientation in range(6):
        if _find_mass_geometry(
            mass_prefix,
            tissue_prefix,
            spacing,
            cell,
            target_mass,
            orientation,
            &side,
            &mass,
            &tissue_volume,
        ):
            sides[output_index, orientation] = side
            masses[output_index, orientation] = mass


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double _centered_average_value(
    const double[:, :, :] absorbed_prefix,
    const double* spacing,
    int c0,
    int c1,
    int c2,
    double side,
    double mass,
) noexcept nogil:
    cdef:
        int axis
        double center[3]
        double lower[3]
        double upper[3]
    center[0] = (c0 + 0.5) * spacing[0]
    center[1] = (c1 + 0.5) * spacing[1]
    center[2] = (c2 + 0.5) * spacing[2]
    _set_bounds(center, spacing, side, 6, lower, upper)
    return _cuboid(absorbed_prefix, spacing, lower, upper) / mass


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _apply_one_face_plan(
    const double[:, :, :] absorbed_prefix,
    const double* spacing,
    int c0,
    int c1,
    int c2,
    Py_ssize_t input_index,
    const double[:, :] face_sides,
    const double[:, :] face_masses,
    double face_volume_tolerance,
    double[:, :, :] output,
    double[:, :, :] output_mass,
    double[:, :, :] output_volume,
    signed char[:, :, :] orientation_output,
) noexcept nogil:
    cdef:
        int axis, orientation, best_orientation = -1
        double center[3]
        double lower[3]
        double upper[3]
        double minimum_volume = 1.7976931348623157e308
        double volume, absorbed, value, best_value = -1.0
    for orientation in range(6):
        if isfinite(face_sides[input_index, orientation]):
            volume = (
                face_sides[input_index, orientation]
                * face_sides[input_index, orientation]
                * face_sides[input_index, orientation]
            )
            minimum_volume = min(minimum_volume, volume)
    if minimum_volume == 1.7976931348623157e308:
        return
    center[0] = (c0 + 0.5) * spacing[0]
    center[1] = (c1 + 0.5) * spacing[1]
    center[2] = (c2 + 0.5) * spacing[2]
    for orientation in range(6):
        if not isfinite(face_sides[input_index, orientation]):
            continue
        volume = (
            face_sides[input_index, orientation]
            * face_sides[input_index, orientation]
            * face_sides[input_index, orientation]
        )
        if volume > (1.0 + face_volume_tolerance) * minimum_volume:
            continue
        _set_bounds(
            center,
            spacing,
            face_sides[input_index, orientation],
            orientation,
            lower,
            upper,
        )
        absorbed = _cuboid(absorbed_prefix, spacing, lower, upper)
        value = absorbed / face_masses[input_index, orientation]
        if best_orientation < 0 or value > best_value:
            best_orientation = orientation
            best_value = value
    if best_orientation >= 0:
        output[c0, c1, c2] = best_value
        output_mass[c0, c1, c2] = face_masses[input_index, best_orientation]
        output_volume[c0, c1, c2] = (
            face_sides[input_index, best_orientation]
            * face_sides[input_index, best_orientation]
            * face_sides[input_index, best_orientation]
        )
        orientation_output[c0, c1, c2] = best_orientation + 1


@cython.boundscheck(False)
@cython.wraparound(False)
def build_centered_plan(
    int nthreads,
    const double[:, :, ::1] mass_prefix,
    const double[:, :, ::1] tissue_prefix,
    const unsigned char[:, :, ::1] tissue,
    const int[:, ::1] cells,
    const double[::1] spacing,
    double target_mass,
    double maximum_background_fraction,
    int delta_min,
    double[::1] sides,
    double[::1] masses,
    unsigned char[::1] valid,
):
    """Build all centred-cube geometry in parallel."""

    cdef Py_ssize_t index
    for index in prange(
        cells.shape[0], nogil=True, schedule="static", num_threads=nthreads
    ):
        _build_one_centered_candidate(
            mass_prefix,
            tissue_prefix,
            tissue,
            &spacing[0],
            cells[index, 0],
            cells[index, 1],
            cells[index, 2],
            target_mass,
            maximum_background_fraction,
            delta_min,
            index,
            sides,
            masses,
            valid,
        )


@cython.boundscheck(False)
@cython.wraparound(False)
def mark_centered_plan(
    const unsigned char[:, :, ::1] tissue,
    const int[:, ::1] cells,
    const double[::1] spacing,
    const double[::1] sides,
    const unsigned char[::1] valid,
    double used_volume_threshold,
    unsigned char[:, :, ::1] status,
):
    """Set final geometry-only status values for valid centred cubes."""

    cdef:
        Py_ssize_t index
        int cell[3]
        int i, j, k, axis
        int first[3]
        int last[3]
        double center[3]
        double lower[3]
        double upper[3]
        double overlap_x, overlap_y, overlap_z, fraction
    for index in range(cells.shape[0]):
        if valid[index]:
            status[cells[index, 0], cells[index, 1], cells[index, 2]] = 3
    for index in range(cells.shape[0]):
        if not valid[index]:
            continue
        for axis in range(3):
            cell[axis] = cells[index, axis]
            center[axis] = (cell[axis] + 0.5) * spacing[axis]
        _set_bounds(center, &spacing[0], sides[index], 6, lower, upper)
        first[0] = max(0, <int>floor(lower[0] / spacing[0]))
        first[1] = max(0, <int>floor(lower[1] / spacing[1]))
        first[2] = max(0, <int>floor(lower[2] / spacing[2]))
        last[0] = min(tissue.shape[0], <int>ceil(upper[0] / spacing[0]))
        last[1] = min(tissue.shape[1], <int>ceil(upper[1] / spacing[1]))
        last[2] = min(tissue.shape[2], <int>ceil(upper[2] / spacing[2]))
        for i in range(first[0], last[0]):
            overlap_x = max(
                0.0,
                min((i + 1) * spacing[0], upper[0])
                - max(i * spacing[0], lower[0]),
            ) / spacing[0]
            for j in range(first[1], last[1]):
                overlap_y = max(
                    0.0,
                    min((j + 1) * spacing[1], upper[1])
                    - max(j * spacing[1], lower[1]),
                ) / spacing[1]
                for k in range(first[2], last[2]):
                    if not tissue[i, j, k] or status[i, j, k] == 3:
                        continue
                    overlap_z = max(
                        0.0,
                        min((k + 1) * spacing[2], upper[2])
                        - max(k * spacing[2], lower[2]),
                    ) / spacing[2]
                    fraction = overlap_x * overlap_y * overlap_z
                    if fraction > used_volume_threshold:
                        status[i, j, k] = 2


@cython.boundscheck(False)
@cython.wraparound(False)
def build_face_plan(
    int nthreads,
    const double[:, :, ::1] mass_prefix,
    const double[:, :, ::1] tissue_prefix,
    const int[:, ::1] cells,
    const double[::1] spacing,
    double target_mass,
    double[:, ::1] sides,
    double[:, ::1] masses,
):
    """Build six face-centred candidate geometries for boundary cells."""

    cdef Py_ssize_t index
    for index in prange(
        cells.shape[0], nogil=True, schedule="static", num_threads=nthreads
    ):
        _build_one_face_plan(
            mass_prefix,
            tissue_prefix,
            &spacing[0],
            cells[index, 0],
            cells[index, 1],
            cells[index, 2],
            target_mass,
            index,
            sides,
            masses,
        )


@cython.boundscheck(False)
@cython.wraparound(False)
def apply_spatial_average_plan(
    int nthreads,
    const double[:, :, ::1] absorbed_prefix,
    const unsigned char[:, :, ::1] tissue,
    const unsigned char[:, :, ::1] status,
    const double[::1] spacing,
    const int[:, ::1] centered_cells,
    const double[::1] centered_sides,
    const double[::1] centered_masses,
    const int[:, ::1] boundary_cells,
    const double[:, ::1] face_sides,
    const double[:, ::1] face_masses,
    double face_volume_tolerance,
    double used_volume_threshold,
    double[:, :, ::1] output,
    double[:, :, ::1] output_mass,
    double[:, :, ::1] output_volume,
    signed char[:, :, ::1] orientation_output,
):
    """Apply one absorbed-power field to reusable averaging geometry."""

    cdef:
        Py_ssize_t index
        int cell[3]
        int i, j, k, axis
        int first[3]
        int last[3]
        double center[3]
        double lower[3]
        double upper[3]
        double value, existing
        double overlap_x, overlap_y, overlap_z, fraction
        double[:] centered_values = np.empty(centered_cells.shape[0], dtype=np.float64)
    for index in prange(
        centered_cells.shape[0], nogil=True, schedule="static", num_threads=nthreads
    ):
        centered_values[index] = _centered_average_value(
            absorbed_prefix,
            &spacing[0],
            centered_cells[index, 0],
            centered_cells[index, 1],
            centered_cells[index, 2],
            centered_sides[index],
            centered_masses[index],
        )

    # Centred cubes overlap, so this deterministic maximum-deposition pass is
    # intentionally serial. All geometric searches and integrations remain
    # parallel above.
    for index in range(centered_cells.shape[0]):
        for axis in range(3):
            cell[axis] = centered_cells[index, axis]
            center[axis] = (cell[axis] + 0.5) * spacing[axis]
        value = centered_values[index]
        output[cell[0], cell[1], cell[2]] = value
        output_mass[cell[0], cell[1], cell[2]] = centered_masses[index]
        output_volume[cell[0], cell[1], cell[2]] = (
            centered_sides[index] * centered_sides[index] * centered_sides[index]
        )
        orientation_output[cell[0], cell[1], cell[2]] = 7
        _set_bounds(center, &spacing[0], centered_sides[index], 6, lower, upper)
        first[0] = max(0, <int>floor(lower[0] / spacing[0]))
        first[1] = max(0, <int>floor(lower[1] / spacing[1]))
        first[2] = max(0, <int>floor(lower[2] / spacing[2]))
        last[0] = min(tissue.shape[0], <int>ceil(upper[0] / spacing[0]))
        last[1] = min(tissue.shape[1], <int>ceil(upper[1] / spacing[1]))
        last[2] = min(tissue.shape[2], <int>ceil(upper[2] / spacing[2]))
        for i in range(first[0], last[0]):
            overlap_x = max(
                0.0,
                min((i + 1) * spacing[0], upper[0])
                - max(i * spacing[0], lower[0]),
            ) / spacing[0]
            for j in range(first[1], last[1]):
                overlap_y = max(
                    0.0,
                    min((j + 1) * spacing[1], upper[1])
                    - max(j * spacing[1], lower[1]),
                ) / spacing[1]
                for k in range(first[2], last[2]):
                    if not tissue[i, j, k] or status[i, j, k] == 3:
                        continue
                    overlap_z = max(
                        0.0,
                        min((k + 1) * spacing[2], upper[2])
                        - max(k * spacing[2], lower[2]),
                    ) / spacing[2]
                    fraction = overlap_x * overlap_y * overlap_z
                    if fraction <= used_volume_threshold:
                        continue
                    existing = output[i, j, k]
                    if not isfinite(existing) or value > existing:
                        output[i, j, k] = value

    # Face candidates belong to distinct boundary cells and are race-free.
    for index in prange(
        boundary_cells.shape[0], nogil=True, schedule="static", num_threads=nthreads
    ):
        _apply_one_face_plan(
            absorbed_prefix,
            &spacing[0],
            boundary_cells[index, 0],
            boundary_cells[index, 1],
            boundary_cells[index, 2],
            index,
            face_sides,
            face_masses,
            face_volume_tolerance,
            output,
            output_mass,
            output_volume,
            orientation_output,
        )
