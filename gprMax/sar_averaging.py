# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Mass-based spatial averaging of cell-centred local SAR.

The implementation follows the two-step cubical procedure described by
IEC/IEEE 62704-1.  Density and local SAR are constant within each FDTD cell;
only the geometrical fraction of a boundary cell contained by an averaging
cube is fractional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

try:
    from gprMax.cython.sar_averaging import (
        apply_spatial_average_plan as _apply_spatial_average_plan_cython,
    )
    from gprMax.cython.sar_averaging import build_centered_plan as _build_centered_plan_cython
    from gprMax.cython.sar_averaging import build_face_plan as _build_face_plan_cython
    from gprMax.cython.sar_averaging import (
        centered_shells_touch_tissue as _centered_shells_touch_tissue_cython,
    )
    from gprMax.cython.sar_averaging import cuboid_integral as _cuboid_integral_cython
    from gprMax.cython.sar_averaging import find_mass_cube as _find_mass_cube_cython
    from gprMax.cython.sar_averaging import mark_centered_plan as _mark_centered_plan_cython
except ImportError:  # pragma: no cover - source tree before Cython compilation
    _apply_spatial_average_plan_cython = None
    _build_centered_plan_cython = None
    _build_face_plan_cython = None
    _centered_shells_touch_tissue_cython = None
    _cuboid_integral_cython = None
    _find_mass_cube_cython = None
    _mark_centered_plan_cython = None


INVALID = np.uint8(0)
UNUSED = np.uint8(1)
USED = np.uint8(2)
VALID = np.uint8(3)
USED_CELL_VOLUME_THRESHOLD = 0.999


@dataclass(frozen=True)
class SpatialAverageSARResult:
    """One mass-based spatial-average SAR distribution."""

    target_mass: float
    sar: npt.NDArray[np.floating]
    status: npt.NDArray[np.uint8]
    averaging_mass: npt.NDArray[np.floating]
    averaging_volume: npt.NDArray[np.floating]
    orientation: npt.NDArray[np.int8]
    peak_sar: float
    peak_cell: tuple[int, int, int] | None


@dataclass(frozen=True)
class SpatialAverageSARPlan:
    """Geometry-only data reused across frequencies for one target mass."""

    target_mass: float
    spacing: npt.NDArray[np.float64]
    tissue: npt.NDArray[np.uint8]
    status: npt.NDArray[np.uint8]
    centered_cells: npt.NDArray[np.int32]
    centered_sides: npt.NDArray[np.float64]
    centered_masses: npt.NDArray[np.float64]
    boundary_cells: npt.NDArray[np.int32]
    face_sides: npt.NDArray[np.float64]
    face_masses: npt.NDArray[np.float64]
    face_volume_tolerance: float
    used_volume_threshold: float
    nthreads: int

    @property
    def nbytes(self) -> int:
        """Memory occupied by reusable plan arrays."""

        arrays = (
            self.spacing,
            self.tissue,
            self.status,
            self.centered_cells,
            self.centered_sides,
            self.centered_masses,
            self.boundary_cells,
            self.face_sides,
            self.face_masses,
        )
        return int(sum(array.nbytes for array in arrays))


def _prefix_integral(values, spacing):
    """Integral at grid vertices for a piecewise-constant cell quantity."""

    prefix = np.pad(np.asarray(values, dtype=np.float64), ((1, 0),) * 3)
    prefix = prefix.cumsum(0).cumsum(1).cumsum(2)
    return prefix * float(np.prod(spacing))


def _integral_at(prefix, spacing, point):
    """Trilinearly evaluate a cellwise-constant cumulative integral."""

    shape = np.asarray(prefix.shape) - 1
    coordinate = np.clip(np.asarray(point) / spacing, 0, shape)
    lower = np.floor(coordinate).astype(np.int32)
    fraction = coordinate - lower
    at_upper = lower == shape
    lower[at_upper] -= 1
    fraction[at_upper] = 1.0
    value = 0.0
    for ox in (0, 1):
        wx = fraction[0] if ox else 1 - fraction[0]
        for oy in (0, 1):
            wy = fraction[1] if oy else 1 - fraction[1]
            for oz in (0, 1):
                wz = fraction[2] if oz else 1 - fraction[2]
                value += wx * wy * wz * prefix[lower[0] + ox, lower[1] + oy, lower[2] + oz]
    return float(value)


def _cuboid_integral(prefix, spacing, lower, upper):
    if _cuboid_integral_cython is not None:
        return float(
            _cuboid_integral_cython(
                np.ascontiguousarray(prefix),
                np.ascontiguousarray(spacing),
                np.ascontiguousarray(lower),
                np.ascontiguousarray(upper),
            )
        )
    value = 0.0
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                point = np.asarray(
                    (
                        upper[0] if bx else lower[0],
                        upper[1] if by else lower[1],
                        upper[2] if bz else lower[2],
                    )
                )
                sign = 1 if (bx + by + bz) % 2 == 1 else -1
                value += sign * _integral_at(prefix, spacing, point)
    return max(0.0, float(value))


def _cube_integrals(integrals, spacing, lower, upper):
    """Integrate mass, absorbed power, and tissue volume over one cube."""

    return tuple(_cuboid_integral(prefix, spacing, lower, upper) for prefix in integrals)


def _bounds_for_cube(center, side, orientation, spacing=None):
    lower = center - 0.5 * side
    upper = center + 0.5 * side
    if orientation < 6:
        axis = orientation // 2
        # IEC/IEEE orientation 1 identifies the starting -x face, so the
        # associated cube expands toward +x (and likewise for y and z).
        positive = orientation % 2 == 0
        # The IEC/IEEE discrete Step-2 construction includes the complete
        # voxel whose centre identifies the starting face.
        half_cell = 0.0 if spacing is None else 0.5 * spacing[axis]
        if positive:
            lower[axis] = center[axis] - half_cell
            upper[axis] = lower[axis] + side
        else:
            upper[axis] = center[axis] + half_cell
            lower[axis] = upper[axis] - side
    return lower, upper


def _find_cube(density, integrals, spacing, cell, target_mass, orientation):
    """Find a centred (6) or face-centred (0..5) target-mass cube."""

    if _find_mass_cube_cython is not None:
        return _find_mass_cube_cython(
            np.ascontiguousarray(integrals[0]),
            np.ascontiguousarray(integrals[1]),
            np.ascontiguousarray(integrals[2]),
            np.ascontiguousarray(spacing),
            np.ascontiguousarray(cell, dtype=np.int32),
            target_mass,
            orientation,
        )

    shape = np.asarray(density.shape, dtype=np.int32)
    domain = shape * spacing
    center = (np.asarray(cell, dtype=np.float64) + 0.5) * spacing
    if orientation == 6:
        maximum_side = float(2 * np.min(np.minimum(center, domain - center)))
    else:
        axis = orientation // 2
        positive = orientation % 2 == 0
        axial = (
            domain[axis] - center[axis] + 0.5 * spacing[axis]
            if positive
            else center[axis] + 0.5 * spacing[axis]
        )
        transverse = [
            2 * min(center[item], domain[item] - center[item]) for item in range(3) if item != axis
        ]
        maximum_side = float(min(axial, *transverse))
    if maximum_side <= 0:
        return None
    low = 0.0
    high = float(np.min(spacing))
    for _ in range(64):
        lower, upper = _bounds_for_cube(center, high, orientation, spacing)
        if np.any(lower < 0) or np.any(upper > domain):
            return None
        mass, _, _ = _cube_integrals(integrals, spacing, lower, upper)
        if mass >= target_mass:
            break
        low = high
        if high >= maximum_side:
            return None
        high = min(2 * high, maximum_side)
    else:  # pragma: no cover - defensive bound
        return None

    mass_tolerance = max(1e-12 * target_mass, np.finfo(np.float64).eps)
    for _ in range(64):
        side = 0.5 * (low + high)
        lower, upper = _bounds_for_cube(center, side, orientation, spacing)
        mass, _, _ = _cube_integrals(integrals, spacing, lower, upper)
        if abs(mass - target_mass) <= mass_tolerance:
            break
        if mass < target_mass:
            low = side
        else:
            high = side
    side = 0.5 * (low + high)
    lower, upper = _bounds_for_cube(center, side, orientation, spacing)
    mass, absorbed, tissue_volume = _cube_integrals(integrals, spacing, lower, upper)
    if mass <= 0:
        return None
    return side, lower, upper, mass, absorbed, tissue_volume


def _centered_shells_touch_tissue(tissue, spacing, cell, side, delta_min):
    """Apply the standard's discrete face-touch test during cube growth."""

    cell = np.asarray(cell, dtype=np.int32)
    if _centered_shells_touch_tissue_cython is not None:
        return bool(
            _centered_shells_touch_tissue_cython(
                np.ascontiguousarray(tissue, dtype=np.uint8),
                cell,
                float(side),
                float(np.min(spacing)),
                int(delta_min),
            )
        )
    final_delta = max(
        0,
        int(np.ceil((side / float(np.min(spacing)) - 1) / 2 - 1e-12)),
    )
    # The reference implementation presets delta_min without checking its
    # faces, then checks every subsequently added complete voxel shell.
    first_delta = delta_min + 1 if delta_min > 0 else 0
    for delta in range(first_delta, final_delta + 1):
        lower = cell - delta
        upper = cell + delta + 1
        if np.any(lower < 0) or np.any(upper > tissue.shape):
            return False
        block = tissue[
            lower[0] : upper[0],
            lower[1] : upper[1],
            lower[2] : upper[2],
        ]
        if not (
            np.any(block[0, :, :])
            and np.any(block[-1, :, :])
            and np.any(block[:, 0, :])
            and np.any(block[:, -1, :])
            and np.any(block[:, :, 0])
            and np.any(block[:, :, -1])
        ):
            return False
    return True


def _spatial_average_sar_python(
    density: npt.ArrayLike,
    local_sar: npt.ArrayLike,
    spacing,
    target_mass: float,
    *,
    maximum_background_fraction: float = 0.1,
    face_volume_tolerance: float = 0.05,
) -> SpatialAverageSARResult:
    """Apply the IEC/IEEE two-step cubical mass-averaging procedure.

    ``density`` contains kg/m3 and uses NaN for background. ``local_sar`` is
    in W/kg. Both quantities are cell-centred and constant inside a cell.
    ``target_mass`` is in kg (0.001 for 1 g and 0.01 for 10 g).
    """

    rho = np.asarray(density, dtype=np.float64)
    sar = np.asarray(local_sar, dtype=np.float64)
    dl = np.asarray(spacing, dtype=np.float64)
    if rho.ndim != 3 or sar.shape != rho.shape:
        raise ValueError("density and local_sar must be identically shaped 3-D arrays")
    if dl.shape != (3,) or not np.all(np.isfinite(dl)) or np.any(dl <= 0):
        raise ValueError("spacing must contain three finite positive cell sizes")
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("target_mass must be finite and positive")
    tissue = np.isfinite(rho)
    if np.any(rho[tissue] <= 0) or not np.all(np.isfinite(sar[tissue])):
        raise ValueError("tissue density must be positive and local SAR finite")
    if not 0 <= maximum_background_fraction < 1:
        raise ValueError("maximum_background_fraction must lie in [0, 1)")
    if face_volume_tolerance < 0:
        raise ValueError("face_volume_tolerance must be non-negative")

    output = np.full(rho.shape, np.nan, dtype=np.float64)
    status = np.full(rho.shape, INVALID, dtype=np.uint8)
    status[tissue] = UNUSED
    masses = np.full(rho.shape, np.nan, dtype=np.float64)
    volumes = np.full(rho.shape, np.nan, dtype=np.float64)
    orientations = np.zeros(rho.shape, dtype=np.int8)
    integrals = (
        _prefix_integral(np.where(tissue, rho, 0.0), dl),
        _prefix_integral(np.where(tissue, rho * sar, 0.0), dl),
        _prefix_integral(tissue.astype(np.float64), dl),
    )
    maximum_cell_mass = float(np.nanmax(rho)) * float(np.prod(dl))
    centered_delta_min = max(
        0,
        int(np.floor((np.cbrt(target_mass / maximum_cell_mass) - 1) / 2)),
    )

    # Step 1: cubes centred on every tissue voxel.
    for cell_array in np.argwhere(tissue):
        cell = tuple(int(value) for value in cell_array)
        candidate = _find_cube(rho, integrals, dl, cell, target_mass, 6)
        if candidate is None:
            continue
        side, lower, upper, mass, absorbed, tissue_volume = candidate
        cube_volume = side**3
        background = max(0.0, 1.0 - tissue_volume / cube_volume)
        if background > maximum_background_fraction or not _centered_shells_touch_tissue(
            tissue, dl, cell, side, centered_delta_min
        ):
            continue
        value = absorbed / mass
        output[cell] = value
        masses[cell] = mass
        volumes[cell] = cube_volume
        orientations[cell] = 7
        status[cell] = VALID
        first = np.maximum(0, np.floor(lower / dl).astype(np.int32))
        last = np.minimum(rho.shape, np.ceil(upper / dl).astype(np.int32))
        if np.any(first >= last):
            continue
        ranges = tuple(np.arange(first[axis], last[axis]) for axis in range(3))
        ix = np.ix_(*ranges)
        included = tissue[ix]
        local_status = status[ix]
        fractions = []
        for axis, indices in enumerate(ranges):
            cell_lower = indices * dl[axis]
            cell_upper = cell_lower + dl[axis]
            fractions.append(
                np.clip(
                    np.minimum(cell_upper, upper[axis]) - np.maximum(cell_lower, lower[axis]),
                    0,
                    dl[axis],
                )
                / dl[axis]
            )
        included_fraction = (
            fractions[0][:, np.newaxis, np.newaxis]
            * fractions[1][np.newaxis, :, np.newaxis]
            * fractions[2][np.newaxis, np.newaxis, :]
        )
        mark = (
            included
            & ((local_status == UNUSED) | (local_status == USED))
            & (included_fraction > USED_CELL_VOLUME_THRESHOLD)
        )
        coordinates = tuple(axis_indices[mark] for axis_indices in np.broadcast_arrays(*ix))
        status[coordinates] = USED
        existing = output[coordinates]
        output[coordinates] = np.where(np.isnan(existing), value, np.maximum(existing, value))

    # Step 2: remaining boundary voxels use six face-centred candidates.
    for cell_array in np.argwhere(status == UNUSED):
        cell = tuple(int(value) for value in cell_array)
        candidates = []
        for orientation in range(6):
            candidate = _find_cube(rho, integrals, dl, cell, target_mass, orientation)
            if candidate is not None:
                candidates.append((orientation, candidate))
        if not candidates:
            continue
        minimum_volume = min(item[1][0] ** 3 for item in candidates)
        eligible = [
            item
            for item in candidates
            if item[1][0] ** 3 <= (1 + face_volume_tolerance) * minimum_volume
        ]
        orientation, chosen = max(eligible, key=lambda item: item[1][4] / item[1][3])
        side, _, _, mass, absorbed, _ = chosen
        output[cell] = absorbed / mass
        masses[cell] = mass
        volumes[cell] = side**3
        orientations[cell] = orientation + 1

    valid_values = np.flatnonzero(np.isfinite(output))
    if valid_values.size:
        peak_flat = int(valid_values[np.nanargmax(output.ravel()[valid_values])])
        peak_cell = tuple(int(value) for value in np.unravel_index(peak_flat, rho.shape))
        peak_sar = float(output[peak_cell])
    else:
        peak_cell = None
        peak_sar = float("nan")
    return SpatialAverageSARResult(
        target_mass=float(target_mass),
        sar=output,
        status=status,
        averaging_mass=masses,
        averaging_volume=volumes,
        orientation=orientations,
        peak_sar=peak_sar,
        peak_cell=peak_cell,
    )


def build_spatial_average_plan(
    density: npt.ArrayLike,
    spacing,
    target_mass: float,
    *,
    maximum_background_fraction: float = 0.1,
    face_volume_tolerance: float = 0.05,
    nthreads: int = 1,
) -> SpatialAverageSARPlan:
    """Precompute all density/tag/grid-dependent averaging geometry."""

    rho = np.ascontiguousarray(density, dtype=np.float64)
    dl = np.ascontiguousarray(spacing, dtype=np.float64)
    if rho.ndim != 3:
        raise ValueError("density must be a 3-D array")
    if dl.shape != (3,) or not np.all(np.isfinite(dl)) or np.any(dl <= 0):
        raise ValueError("spacing must contain three finite positive cell sizes")
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("target_mass must be finite and positive")
    tissue_bool = np.isfinite(rho)
    if np.any(rho[tissue_bool] <= 0):
        raise ValueError("tissue density must be positive")
    if not 0 <= maximum_background_fraction < 1:
        raise ValueError("maximum_background_fraction must lie in [0, 1)")
    if face_volume_tolerance < 0:
        raise ValueError("face_volume_tolerance must be non-negative")
    nthreads = int(nthreads)
    if nthreads < 1:
        raise ValueError("nthreads must be positive")
    if _build_centered_plan_cython is None:
        raise RuntimeError("compiled SAR spatial-averaging support is unavailable")

    tissue = np.ascontiguousarray(tissue_bool, dtype=np.uint8)
    cells = np.ascontiguousarray(np.argwhere(tissue_bool), dtype=np.int32)
    status = np.full(rho.shape, INVALID, dtype=np.uint8)
    status[tissue_bool] = UNUSED
    mass_prefix = _prefix_integral(np.where(tissue_bool, rho, 0.0), dl)
    tissue_prefix = _prefix_integral(tissue, dl)
    maximum_cell_mass = float(np.nanmax(rho)) * float(np.prod(dl))
    centered_delta_min = max(
        0,
        int(np.floor((np.cbrt(target_mass / maximum_cell_mass) - 1) / 2)),
    )
    centered_sides = np.full(cells.shape[0], np.nan, dtype=np.float64)
    centered_masses = np.full(cells.shape[0], np.nan, dtype=np.float64)
    centered_valid = np.zeros(cells.shape[0], dtype=np.uint8)
    _build_centered_plan_cython(
        nthreads,
        mass_prefix,
        tissue_prefix,
        tissue,
        cells,
        dl,
        float(target_mass),
        float(maximum_background_fraction),
        centered_delta_min,
        centered_sides,
        centered_masses,
        centered_valid,
    )
    _mark_centered_plan_cython(
        tissue,
        cells,
        dl,
        centered_sides,
        centered_valid,
        USED_CELL_VOLUME_THRESHOLD,
        status,
    )
    valid_selection = centered_valid.astype(bool)
    centered_cells = np.ascontiguousarray(cells[valid_selection], dtype=np.int32)
    centered_sides = np.ascontiguousarray(centered_sides[valid_selection])
    centered_masses = np.ascontiguousarray(centered_masses[valid_selection])
    cell_status = status[tuple(cells.T)]
    boundary_cells = np.ascontiguousarray(cells[cell_status == UNUSED], dtype=np.int32)
    face_sides = np.full((boundary_cells.shape[0], 6), np.nan, dtype=np.float64)
    face_masses = np.full((boundary_cells.shape[0], 6), np.nan, dtype=np.float64)
    _build_face_plan_cython(
        nthreads,
        mass_prefix,
        tissue_prefix,
        boundary_cells,
        dl,
        float(target_mass),
        face_sides,
        face_masses,
    )
    return SpatialAverageSARPlan(
        target_mass=float(target_mass),
        spacing=dl,
        tissue=tissue,
        status=status,
        centered_cells=centered_cells,
        centered_sides=centered_sides,
        centered_masses=centered_masses,
        boundary_cells=boundary_cells,
        face_sides=face_sides,
        face_masses=face_masses,
        face_volume_tolerance=float(face_volume_tolerance),
        used_volume_threshold=USED_CELL_VOLUME_THRESHOLD,
        nthreads=nthreads,
    )


def apply_spatial_average_plan(
    plan: SpatialAverageSARPlan,
    local_sar: npt.ArrayLike,
    density: npt.ArrayLike,
) -> SpatialAverageSARResult:
    """Apply a local-SAR field to a reusable spatial-averaging plan."""

    rho = np.ascontiguousarray(density, dtype=np.float64)
    sar = np.ascontiguousarray(local_sar, dtype=np.float64)
    if rho.shape != plan.tissue.shape or sar.shape != rho.shape:
        raise ValueError("density and local_sar must match the averaging-plan shape")
    tissue = plan.tissue.astype(bool)
    if np.any(rho[tissue] <= 0) or not np.all(np.isfinite(sar[tissue])):
        raise ValueError("tissue density must be positive and local SAR finite")
    if not np.array_equal(np.isfinite(rho), tissue):
        raise ValueError("density tissue membership differs from the averaging plan")

    absorbed_prefix = _prefix_integral(np.where(tissue, rho * sar, 0.0), plan.spacing)
    output = np.full(rho.shape, np.nan, dtype=np.float64)
    masses = np.full(rho.shape, np.nan, dtype=np.float64)
    volumes = np.full(rho.shape, np.nan, dtype=np.float64)
    orientations = np.zeros(rho.shape, dtype=np.int8)
    _apply_spatial_average_plan_cython(
        plan.nthreads,
        absorbed_prefix,
        plan.tissue,
        plan.status,
        plan.spacing,
        plan.centered_cells,
        plan.centered_sides,
        plan.centered_masses,
        plan.boundary_cells,
        plan.face_sides,
        plan.face_masses,
        plan.face_volume_tolerance,
        plan.used_volume_threshold,
        output,
        masses,
        volumes,
        orientations,
    )
    valid_values = np.flatnonzero(np.isfinite(output))
    if valid_values.size:
        peak_flat = int(valid_values[np.nanargmax(output.ravel()[valid_values])])
        peak_cell = tuple(int(value) for value in np.unravel_index(peak_flat, rho.shape))
        peak_sar = float(output[peak_cell])
    else:
        peak_cell = None
        peak_sar = float("nan")
    return SpatialAverageSARResult(
        target_mass=plan.target_mass,
        sar=output,
        status=plan.status.copy(),
        averaging_mass=masses,
        averaging_volume=volumes,
        orientation=orientations,
        peak_sar=peak_sar,
        peak_cell=peak_cell,
    )


def spatial_average_sar(
    density: npt.ArrayLike,
    local_sar: npt.ArrayLike,
    spacing,
    target_mass: float,
    *,
    maximum_background_fraction: float = 0.1,
    face_volume_tolerance: float = 0.05,
    nthreads: int = 1,
) -> SpatialAverageSARResult:
    """Apply IEC/IEEE averaging using compiled reusable geometry when available."""

    if _build_centered_plan_cython is None:
        return _spatial_average_sar_python(
            density,
            local_sar,
            spacing,
            target_mass,
            maximum_background_fraction=maximum_background_fraction,
            face_volume_tolerance=face_volume_tolerance,
        )
    plan = build_spatial_average_plan(
        density,
        spacing,
        target_mass,
        maximum_background_fraction=maximum_background_fraction,
        face_volume_tolerance=face_volume_tolerance,
        nthreads=nthreads,
    )
    return apply_spatial_average_plan(plan, local_sar, density)
