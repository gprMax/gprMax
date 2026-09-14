# Copyright (c) 2026 Mahdee Abir
# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# Originally developed for STEP-to-gprMax and distributed under the MIT
# License. This adapted version is part of gprMax and is distributed under the
# GNU General Public License, version 3 or (at your option) any later version.
# See LICENSE in this directory.

"""Classify triangle-mesh interiors on a cell grid and resolve overlaps.

World coordinates are converted to continuous voxel coordinates before
plane intersections and scanline voting. The public arrays use x, y, z
axes; local slice buffers use y, x and are transposed on return. STEP and
GeometryImport surface converters reuse this classifier, assigning their
own meaning to its non-negative mesh IDs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# HDF5 output for gprMax
try:
    import h5py

    _HAS_H5PY = True
except Exception:
    _HAS_H5PY = False

# =============================================================================
# 0) Data structures
# =============================================================================


@dataclass(frozen=True)
class GridSpec:
    """
    World-space voxel grid definition.

    ``origin_world`` is the lower corner of cell (0, 0, 0), not its centre.
    ``dxyz_world`` and ``nxyz`` are ordered x, y, z. Cell (i, j, k) has centre
    ``origin_world + ([i,j,k] + 0.5)*dxyz_world``. Low-level grid operations
    require consistent world length units; they do not convert units.
    """

    origin_world: np.ndarray  # (3,) float
    dxyz_world: np.ndarray  # (3,) float
    nxyz: np.ndarray  # (3,) int

    @property
    def nx(self) -> int:
        return int(self.nxyz[0])

    @property
    def ny(self) -> int:
        return int(self.nxyz[1])

    @property
    def nz(self) -> int:
        return int(self.nxyz[2])


@dataclass(frozen=True)
class TriangleMesh:
    """
    Indexed triangle mesh in world coordinates.

    Vertices have shape ``(N, 3)``; each row of ``triangles`` contains three
    zero-based vertex indices. ``material_id`` is a non-negative output ID;
    callers may use it for components before mapping to materials. Empty
    output cells use -1, so ID zero is available for an occupied component.
    ``priority`` is used only in priority merging: higher values win and
    equal values retain the earlier mesh.
    """

    vertices_world: np.ndarray  # (N,3) float
    triangles: np.ndarray  # (M,3) int
    material_id: int
    priority: int = 0
    name: str = ""


# =============================================================================
# 1) Grid utilities (world ↔ voxel coordinate space)
# =============================================================================


def make_grid_from_bbox(
    vmin_world: np.ndarray,
    vmax_world: np.ndarray,
    *,
    dx: float,
    dy: Optional[float] = None,
    dz: Optional[float] = None,
    pad: int = 2,
) -> GridSpec:
    """
    Build a grid from [vmin_world, vmax_world] with ``pad`` cells on each side.

    The lower bound is snapped down on a zero-origin voxel lattice. Values
    within 1e-9 voxel units of an integer are snapped to that integer first,
    so round-off just below a lattice plane does not add a lower cell.
    Origin and spacing are stored as float64.
    """
    vmin_world = np.asarray(vmin_world, dtype=np.float64)
    vmax_world = np.asarray(vmax_world, dtype=np.float64)

    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    dxyz = np.array([dx, dy, dz], dtype=np.float64)
    if not np.isfinite(dxyz).all() or np.any(dxyz <= 0):
        raise ValueError("voxel spacing must contain three positive finite values")
    if pad < 0:
        raise ValueError("pad must be non-negative")

    scaled_min = vmin_world / dxyz
    nearest = np.rint(scaled_min)
    scaled_min = np.where(
        np.isclose(scaled_min, nearest, rtol=0.0, atol=1e-9),
        nearest,
        scaled_min,
    )
    k = np.floor(scaled_min).astype(np.int64)
    origin = (k - pad) * dxyz

    # Ensure vmax fits with padding
    extent_from_origin = (vmax_world - origin) + pad * dxyz
    nxyz = np.ceil(extent_from_origin / dxyz).astype(np.int64)
    nxyz = np.maximum(nxyz, 1)

    return GridSpec(
        origin_world=origin,  # float64
        dxyz_world=dxyz,  # float64
        nxyz=nxyz.astype(np.int32),
    )


def compute_scene_bbox(meshes: Sequence[TriangleMesh]) -> Tuple[np.ndarray, np.ndarray]:
    """Return float64 world bounds without narrowing fine translated meshes.

    Bounding-box construction must retain the precision used by grid creation
    and world-to-voxel conversion; a later upcast cannot recover lost extents.
    """
    if not meshes:
        raise ValueError("meshes must be non-empty")
    vmins, vmaxs = [], []
    for m in meshes:
        V = np.asarray(m.vertices_world, dtype=np.float64)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("vertices_world must be (N,3)")
        vmins.append(V.min(axis=0))
        vmaxs.append(V.max(axis=0))
    return np.minimum.reduce(vmins), np.maximum.reduce(vmaxs)


def world_to_voxel_coords(V_world: np.ndarray, grid: GridSpec) -> np.ndarray:
    """
    Map world coords → voxel coord space (continuous).
    Keep float64 for robustness (float32 causes slice jitter at small dx).
    """
    V = np.asarray(V_world, dtype=np.float64)
    return (V - grid.origin_world[None, :]) / grid.dxyz_world[None, :]


# =============================================================================
# 2) Slice-based solid voxelisation
#    - Intersect mesh triangles with horizontal z-planes
#    - Reconstruct closed slice perimeters from intersection segments
#    - Fill interior pixels using scanline rasterisation
# =============================================================================


def _generate_tri_events(tris_xyz: np.ndarray, eps: float = 1e-7) -> List[Tuple[float, str, int]]:
    """
    Plane sweep events for z slicing.
    tris_xyz: (M,3,3) float, triangle vertices in voxel coords.

    ``eps`` is a fraction of a voxel, not a world-space length. It expands
    each triangle's active z interval to include nearby slicing planes.
    """
    events: List[Tuple[float, str, int]] = []
    for i, tri in enumerate(tris_xyz):
        zs = tri[:, 2].astype(np.float64, copy=False)
        zmin = float(np.min(zs))
        zmax = float(np.max(zs))

        # Expand slightly so triangles remain active when z_plane is extremely close.
        events.append((zmin - eps, "begin", i))
        events.append((zmax + eps, "end", i))

    # Sort by z, and ensure begin is processed before end at identical z.
    events.sort(key=lambda t: (t[0], 0 if t[1] == "begin" else 1))
    return events


def _generate_tri_events_int(tris_xyz: np.ndarray, nz: int) -> List[Tuple[int, str, int]]:
    """
    Integer z-layer events based on slicing at z_plane = z + 0.5.

    A triangle is relevant to layer z if:
      zmin <= (z + 0.5) <= zmax

    We convert that to an integer inclusive range [z0, z1] and sweep those indices.
    """
    events: List[Tuple[int, str, int]] = []
    for i, tri in enumerate(tris_xyz):
        zs = tri[:, 2].astype(np.float64, copy=False)
        zmin = float(np.min(zs))
        zmax = float(np.max(zs))

        # layer z corresponds to plane at z + 0.5
        z0 = int(np.ceil(zmin - 0.5))
        z1 = int(np.floor(zmax - 0.5))
        if z1 < z0:
            continue

        z0 = max(z0, 0)
        z1 = min(z1, nz - 1)
        if z1 < z0:
            # A triangle outside the cropped grid has no layer events.
            continue

        events.append((z0, "begin", i))
        events.append((z1, "end", i))  # end is inclusive; remove AFTER painting layer

    # sort by layer; begin before end at same layer
    events.sort(key=lambda t: (t[0], 0 if t[1] == "begin" else 1))
    return events


def _linear_interpolation(p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    return p1 * (1.0 - t) + p2 * t


def _where_line_crosses_z(p1: np.ndarray, p2: np.ndarray, z: float) -> np.ndarray:
    # Ensure p1 is below p2 in z
    if p1[2] > p2[2]:
        p1, p2 = p2, p1
    dz = float(p2[2] - p1[2])
    if dz == 0.0:
        t = 0.0
    else:
        t = float((z - p1[2]) / dz)
    return _linear_interpolation(p1, p2, t)


def _triangle_to_intersecting_points(
    tri: np.ndarray, z_plane: float, eps: float = 1e-7
) -> List[np.ndarray]:
    """
    Return intersection points between triangle edges and plane z=z_plane.
    Robust to near-plane degeneracy.

    eps is in voxel units. With float64 voxel coords, 1e-7 is conservative.
    """
    tri = np.asarray(tri, dtype=np.float64)
    points: List[np.ndarray] = []
    assert tri.shape == (3, 3)

    def near(z: float) -> bool:
        return abs(z - z_plane) <= eps

    def add_point(pt: np.ndarray) -> None:
        # Deduplicate by XY (Z is ~z_plane anyway)
        for p in points:
            if abs(float(p[0]) - float(pt[0])) <= eps and abs(float(p[1]) - float(pt[1])) <= eps:
                return
        points.append(pt)

    # Loop edges
    for e in range(3):
        p = tri[e]
        q = tri[(e + 1) % 3]
        pz, qz = float(p[2]), float(q[2])

        p_on = near(pz)
        q_on = near(qz)

        if p_on and q_on:
            # Edge lies (almost) in the plane: add both endpoints (dedup handles repeats)
            add_point(p)
            add_point(q)
            continue

        if p_on:
            add_point(p)
            continue
        if q_on:
            add_point(q)
            continue

        # Proper straddle (with epsilon band)
        below = pz < z_plane - eps and qz > z_plane + eps
        above = qz < z_plane - eps and pz > z_plane + eps
        if below or above:
            add_point(_where_line_crosses_z(p, q, z_plane))

    # Most of the time we want exactly 0 or 2 points for a clean segment.
    if len(points) <= 2:
        return points

    # Degenerate: choose the two farthest points in XY (stable segment)
    best_i, best_j = 0, 1
    best_d2 = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = float(points[i][0] - points[j][0])
            dy = float(points[i][1] - points[j][1])
            d2 = dx * dx + dy * dy
            if d2 > best_d2:
                best_d2 = d2
                best_i, best_j = i, j
    return [points[best_i], points[best_j]]


# -----------------------------------------------------------------------------
# 2.1) Polygon repair
# -----------------------------------------------------------------------------


def _find_polylines(
    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]]
) -> List[List[Tuple[int, int]]]:
    polylines: List[List[Tuple[int, int]]] = []
    fwd: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    bwd: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for s, e in segments:
        fwd.setdefault(s, []).append(e)
        bwd.setdefault(e, []).append(s)

    while fwd:
        start = next(iter(fwd.keys()))
        poly = [start]

        # forward
        while True:
            if start not in fwd:
                break
            end = fwd[start][0]
            fwd[start].remove(end)
            if not fwd[start]:
                del fwd[start]
            bwd[end].remove(start)
            if not bwd[end]:
                del bwd[end]
            poly.append(end)
            start = end

        # backward
        start = poly[0]
        while True:
            if start not in bwd:
                break
            end = bwd[start][0]
            bwd[start].remove(end)
            if not bwd[start]:
                del bwd[start]
            fwd[end].remove(start)
            if not fwd[end]:
                del fwd[end]
            poly.insert(0, end)
            start = end

        polylines.append(poly)

    return polylines


def _normalize2(v: Tuple[float, float]) -> Tuple[float, float]:
    x, y = v
    n = float(np.hypot(x, y))
    if n == 0.0:
        return (1.0, 0.0)
    return (x / n, y / n)


def _atan_sum(f1: Tuple[float, float], f2: Tuple[float, float]) -> Tuple[float, float]:
    x1, y1 = f1
    x2, y2 = f2
    return (x1 * x2 - y1 * y2, y1 * x2 + x1 * y2)


def _atan_diff(f1: Tuple[float, float], f2: Tuple[float, float]) -> Tuple[float, float]:
    x1, y1 = f1
    x2, y2 = f2
    return (x1 * x2 + y1 * y2, y1 * x2 - x1 * y2)


def _sub2(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _add2(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _winding_contour_pole(
    pos: Tuple[float, float], pt: Tuple[float, float], repel: bool
) -> Tuple[float, float]:
    x, y = _sub2(pos, pt)
    dist2 = x * x + y * y
    if dist2 == 0.0:
        return (0.0, 0.0)
    cx = x / dist2
    cy = y / dist2
    return (cx, cy) if repel else (-cx, -cy)


def _winding_contour(
    pos: Tuple[float, float], endpoints: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Tuple[float, float]:
    accum = (0.0, 0.0)
    for start, end in endpoints:
        accum = _add2(accum, _winding_contour_pole(pos, start, repel=False))
        accum = _add2(accum, _winding_contour_pole(pos, end, repel=True))
    return _normalize2(accum)


def _initial_direction(
    pt: Tuple[float, float], endpoints: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Tuple[float, float]:
    accum = (1.0, 0.0)
    for start, end in endpoints:
        if start != pt:
            accum = _atan_sum(accum, _sub2(start, pt))
        if end != pt:
            accum = _atan_diff(accum, _sub2(end, pt))
        accum = _normalize2(accum)
    return accum


def _winding_number_search(
    start: Tuple[float, float],
    ends: List[Tuple[float, float]],
    endpoints: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    max_iterations: int,
) -> Tuple[float, float]:
    direction = _initial_direction(start, endpoints)
    pos = (start[0] + direction[0] * 0.1, start[1] + direction[1] * 0.1)

    for _ in range(max_iterations):
        direction = _winding_contour(pos, endpoints)
        pos = (pos[0] + direction[0], pos[1] + direction[1])
        for end in ends:
            if _dist2(pos, end) < 1.0:
                return end

    raise RuntimeError("Polygon repair failed (winding search did not terminate).")


def _snap_key_xy(x: float, y: float, *, snap: float = 1e-6) -> tuple[float, float]:
    """
    Snap to a tiny lattice in *voxel units* so near-equal endpoints become identical keys.
    snap=1e-6 means 1e-6 of a voxel — far below any meaningful geometric scale.
    """
    return (round(x / snap) * snap, round(y / snap) * snap)


class _PolygonRepair:
    """
    Minimal wrapper around repair concept.

    Input segments are integer pixel coordinates: ((x1,y1),(x2,y2))
    """

    def __init__(
        self, segments: List[Tuple[Tuple[int, int], Tuple[int, int]]], dims_xy: Tuple[int, int]
    ):
        self.dims_xy = dims_xy  # (ny, nx) or (height,width); used only for max_iterations scale
        self.original_segments = list(segments)
        self.loops: List[List[Tuple[int, int]]] = []
        self.polylines: List[List[Tuple[int, int]]] = []
        self._collapse()

    def _collapse(self) -> None:
        self.loops = []
        self.polylines = []
        for poly in _find_polylines(self.original_segments):
            if poly and poly[0] == poly[-1]:
                self.loops.append(poly)
            else:
                self.polylines.append(poly)

    def repair_all(self) -> None:
        while self.polylines:
            self._repair_one()
            old = len(self.polylines)
            self._collapse()
            # Should reduce open polylines by 1 each time
            if len(self.polylines) != old - 1:
                # Don't hard-crash; just stop trying endlessly.
                break

    def _repair_one(self) -> None:
        search_start = self.polylines[0][-1]
        search_ends = [poly[0] for poly in self.polylines]

        endpoints = [(poly[0], poly[-1]) for poly in self.polylines]
        endpoints_f = [
            ((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for (a, b) in endpoints
        ]

        max_iterations = int(self.dims_xy[0] + self.dims_xy[1])  # heuristic
        end = _winding_number_search(
            (float(search_start[0]), float(search_start[1])),
            [(float(p[0]), float(p[1])) for p in search_ends],
            endpoints_f,
            max_iterations=max_iterations,
        )
        end_i = (int(round(end[0])), int(round(end[1])))
        self.original_segments.append((search_start, end_i))


# -----------------------------------------------------------------------------
# 2.2) Scanline fill (robust integer-column even-odd fill)
# -----------------------------------------------------------------------------


def _generate_line_events_int(
    line_list: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    nx: int,
) -> List[Tuple[int, str, int]]:
    """
    Integer x-events so a segment is active for columns x where (x+0.5) lies inside
    the segment's x-range. This avoids float-event bugs.
    """
    events: List[Tuple[int, str, int]] = []
    for i, (a, b) in enumerate(line_list):
        x1, _ = a
        x2, _ = b
        if x1 == x2:
            continue  # vertical segment: skip for scanline fill

        xmin = min(x1, x2)
        xmax = max(x1, x2)

        # Segment contributes to column x if x+0.5 ∈ [xmin, xmax]
        x_start = int(np.ceil(xmin - 0.5))
        x_end = int(np.floor(xmax - 0.5))

        if x_end < 0 or x_start >= nx:
            continue

        x_start = max(x_start, 0)
        x_end = min(x_end, nx - 1)

        events.append((x_start, "begin", i))
        events.append((x_end + 1, "end", i))  # deactivate after last column

    events.sort(key=lambda t: (t[0], 0 if t[1] == "begin" else 1))
    return events


def _y_at_x_halfopen(
    p1: Tuple[float, float], p2: Tuple[float, float], xq: float
) -> Optional[float]:
    """
    y-value where segment p1->p2 crosses vertical line x = xq.

    Robust rules:
      - ignore vertical segments
      - use half-open x-interval [xmin, xmax) to avoid double-counting shared vertices
    """
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        return None  # vertical -> skip (handled implicitly by other edges)

    # Half-open interval: include left, exclude right
    if x1 < x2:
        xmin, xmax = x1, x2
        if not (xmin <= xq < xmax):
            return None
    else:
        xmin, xmax = x2, x1
        if not (xmin <= xq < xmax):
            return None

    t = (xq - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def _dedup_sorted(values: List[float], eps: float) -> List[float]:
    """Assumes `values` is sorted; merges near-equal values."""
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        if abs(v - out[-1]) > eps:
            out.append(v)
    return out


def _column_intervals_even_odd(
    active_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    *,
    xq: float,
) -> List[Tuple[float, float]]:
    """Return half-open interior y intervals at a sampled x position."""
    ys: List[float] = []
    for a, b in active_lines:
        y = _y_at_x_halfopen(a, b, xq)  # IMPORTANT: uses the passed xq
        if y is not None:
            ys.append(float(y))

    if len(ys) < 2:
        return []

    ys.sort()
    ys = _dedup_sorted(ys, eps=1e-6)
    if len(ys) < 2:
        return []

    if len(ys) % 2 == 1:
        ys = ys[:-1]
    return [(ys[index], ys[index + 1]) for index in range(0, len(ys), 2)]


def _segment_active_at_xq(
    a: Tuple[float, float], b: Tuple[float, float], xq: float, *, eps: float = 1e-12
) -> bool:
    ax, ay = a
    bx, by = b
    if ax == bx:
        return False  # skip verticals (consistent with approach)

    xmin = ax if ax < bx else bx
    xmax = bx if ax < bx else ax

    # Half-open interval: xmin <= xq < xmax (with tiny eps)
    return (xq + eps) >= xmin and (xq - eps) < xmax


def _sample_offsets(samples: int) -> tuple[float, ...]:
    if samples < 1:
        raise ValueError("supersample must be at least one")
    return tuple((index + 0.5) / samples for index in range(samples))


def _lines_to_vote_counts(
    line_list: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    plane_shape_yx: Tuple[int, int],
    *,
    supersample: int,
) -> np.ndarray:
    """Count interior samples into a ``(ny, nx)`` plane using even-odd fill.

    Each cell is tested at ``supersample`` evenly centred offsets per axis.
    Half-open interval ends assign a sample on a shared contour only once;
    using inclusive bounds at both ends would change boundary votes.
    """
    ny, nx = plane_shape_yx
    votes = np.zeros((ny, nx), dtype=np.uint32)
    if not line_list:
        return votes

    offsets = _sample_offsets(supersample)

    for x in range(nx):
        for x_offset in offsets:
            xq = float(x) + x_offset
            active_lines = [seg for seg in line_list if _segment_active_at_xq(seg[0], seg[1], xq)]
            intervals = _column_intervals_even_odd(active_lines, xq=xq)
            for y_offset in offsets:
                for lower, upper in intervals:
                    y0 = int(np.ceil(lower - y_offset))
                    # The upper contour is half-open, avoiding a double vote
                    # when a sample lies exactly on a shared polygon boundary.
                    y1 = int(np.floor(np.nextafter(upper - y_offset, -np.inf)))
                    y0 = max(y0, 0)
                    y1 = min(y1, ny - 1)
                    if y1 >= y0:
                        votes[y0 : y1 + 1, x] += 1
    return votes


def _repaired_lines_to_votes(
    line_list: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    plane_shape_yx: Tuple[int, int],
    *,
    enable_repair: bool = False,
    supersample: int = 1,
) -> np.ndarray:
    """
    Optionally repair open polylines into loops before scanline voting.

    Repair rounds endpoints to integer pixel coordinates. If disabled, if
    ``repair_all`` raises, or if no usable segments/loops result, vote using
    the original lines. This fallback does not establish that they are closed.
    """
    if not line_list:
        return np.zeros(plane_shape_yx, dtype=np.uint32)

    def _fallback() -> np.ndarray:
        return _lines_to_vote_counts(line_list, plane_shape_yx, supersample=supersample)

    if not enable_repair:
        return _fallback()

    # Polygon repair operates on integer pixel endpoints.
    segs_int: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for a, b in line_list:
        ax, ay = int(round(a[0])), int(round(a[1]))
        bx, by = int(round(b[0])), int(round(b[1]))
        if (ax, ay) != (bx, by):
            segs_int.append(((ax, ay), (bx, by)))

    if not segs_int:
        return _fallback()

    repair = _PolygonRepair(segs_int, dims_xy=plane_shape_yx)
    try:
        repair.repair_all()
    except Exception:
        return _fallback()

    # An empty repair result must not erase votes from the original lines.
    if not repair.loops:
        return _fallback()

    # The scanline classifier consumes floating-point line segments.
    repaired_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for loop in repair.loops:
        for k in range(len(loop) - 1):
            p = loop[k]
            q = loop[k + 1]
            repaired_lines.append(((float(p[0]), float(p[1])), (float(q[0]), float(q[1]))))

    if not repaired_lines:
        return _fallback()

    return _lines_to_vote_counts(repaired_lines, plane_shape_yx, supersample=supersample)


# -----------------------------------------------------------------------------
# 2.3) Slice a mesh at a given z-layer
# -----------------------------------------------------------------------------


def _paint_z_plane(
    active_tris: np.ndarray,
    z_plane: float,
    plane_shape_yx: Tuple[int, int],
    *,
    supersample: int,
) -> np.ndarray:
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for tri in active_tris:
        pts = _triangle_to_intersecting_points(tri, z_plane, eps=1e-07)
        if len(pts) == 2:
            a, b = pts
            lines.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
        elif len(pts) == 3:
            for i in range(3):
                a = pts[i]
                b = pts[(i + 1) % 3]
                lines.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))

    return _repaired_lines_to_votes(
        lines,
        plane_shape_yx,
        supersample=supersample,
    )


_SWEEP_PERMUTATIONS = {
    "x": (1, 2, 0),
    "y": (2, 0, 1),
    "z": (0, 1, 2),
}


def resolve_sweep_axis(shape_nxyz: Sequence[int], sweep_axis: str) -> str:
    """Resolve ``auto`` to the shortest dimension, preferring z on ties."""
    if sweep_axis not in {"auto", "x", "y", "z"}:
        raise ValueError("sweep_axis must be auto, x, y, or z")
    if sweep_axis != "auto":
        return sweep_axis
    shape = tuple(int(value) for value in shape_nxyz)
    tie_order = {"z": 0, "y": 1, "x": 2}
    return min("xyz", key=lambda axis: (shape["xyz".index(axis)], tie_order[axis]))


def _voxelise_z_slices(
    V_vox: np.ndarray,
    T: np.ndarray,
    shape_nxyz: Tuple[int, int, int],
    *,
    supersample: int,
) -> np.ndarray:
    """Sweep local z, then return a boolean ``(nx, ny, nz)`` mask.

    The working volume is ``(nz, ny, nx)`` to hold contiguous slice planes.
    One sample per axis uses inclusive triangle layer events; larger sample
    counts accumulate ``supersample**3`` votes and require a strict majority.
    """
    V_vox = np.asarray(V_vox, dtype=np.float64)
    T = np.asarray(T, dtype=np.int32)
    nx, ny, nz = map(int, shape_nxyz)

    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError("triangles must be (M,3) int")
    if V_vox.ndim != 2 or V_vox.shape[1] != 3:
        raise ValueError("V_vox must be (N,3) float")

    tris_xyz = V_vox[T]  # (M,3,3)

    vol_zyx = np.zeros((nz, ny, nx), dtype=bool)
    if supersample == 1:
        events = _generate_tri_events_int(tris_xyz, nz)
        active: set[int] = set()
        event_index = 0
        for z in range(nz):
            while (
                event_index < len(events)
                and events[event_index][0] == z
                and events[event_index][1] == "begin"
            ):
                active.add(events[event_index][2])
                event_index += 1
            z_plane = float(z) + 0.5
            subset = tris_xyz[list(active)] if active else tris_xyz[:0]
            vol_zyx[z] = (
                _paint_z_plane(
                    subset,
                    z_plane,
                    (ny, nx),
                    supersample=1,
                )
                > 0
            )
            while (
                event_index < len(events)
                and events[event_index][0] == z
                and events[event_index][1] == "end"
            ):
                active.discard(events[event_index][2])
                event_index += 1
        return np.transpose(vol_zyx, (2, 1, 0))

    offsets = _sample_offsets(supersample)
    zmin = np.min(tris_xyz[:, :, 2], axis=1)
    zmax = np.max(tris_xyz[:, :, 2], axis=1)
    required_votes = supersample**3 // 2 + 1

    for z in range(nz):
        layer_votes = np.zeros((ny, nx), dtype=np.uint32)
        for offset in offsets:
            z_plane = float(z) + offset
            active = (zmin - 1e-7 <= z_plane) & (zmax + 1e-7 >= z_plane)
            layer_votes += _paint_z_plane(
                tris_xyz[active],
                z_plane,
                (ny, nx),
                supersample=supersample,
            )
        vol_zyx[z] = layer_votes >= required_votes

    return np.transpose(vol_zyx, (2, 1, 0))


def _preserve_empty_thin_solid(
    V_vox: np.ndarray,
    T: np.ndarray,
    shape_nxyz: Tuple[int, int, int],
) -> np.ndarray:
    """Paint a central slice for each bounding-box extent between 0 and 1 voxel.

    This fallback is called only when ordinary voting left the entire solid
    empty. Each eligible axis contributes a one-cell-thick slice at its
    bounding-box midpoint; it does not recover every thin feature of a mesh.
    """
    preserved = np.zeros(shape_nxyz, dtype=bool)
    extents = np.ptp(V_vox, axis=0)
    thin_axes = [index for index, extent in enumerate(extents) if 0 < extent < 1.0]
    for axis_index in thin_axes:
        axis = "xyz"[axis_index]
        permutation = _SWEEP_PERMUTATIONS[axis]
        inverse = tuple(int(value) for value in np.argsort(permutation))
        local_vertices = V_vox[:, permutation]
        local_shape = tuple(int(shape_nxyz[index]) for index in permutation)
        local_triangles = local_vertices[T]
        plane = 0.5 * (
            float(np.min(local_triangles[:, :, 2])) + float(np.max(local_triangles[:, :, 2]))
        )
        layer = int(np.clip(np.floor(plane), 0, local_shape[2] - 1))
        votes = _paint_z_plane(
            local_triangles,
            plane,
            (local_shape[1], local_shape[0]),
            supersample=1,
        )
        local_mask = np.zeros(local_shape, dtype=bool)
        local_mask[:, :, layer] = votes.T > 0
        preserved |= np.transpose(local_mask, inverse)
    return preserved


def voxelise_solid_scanline(
    V_vox: np.ndarray,
    T: np.ndarray,
    shape_nxyz: Tuple[int, int, int],
    *,
    supersample: int = 1,
    preserve_thin_features: bool = True,
    sweep_axis: str = "auto",
) -> np.ndarray:
    """Voxelise a closed triangle mesh using an axis-selectable plane sweep.

    ``V_vox`` contains continuous voxel coordinates, ``T`` zero-based vertex
    indices, and ``shape_nxyz`` the output cell counts. The chosen sweep axis
    is permuted into local z and restored in the returned boolean array.
    One sample evaluates each cell centre. Larger values sample equally
    along all three axes and require a strict majority. Optional thin-solid
    preservation runs only if that voting produced no occupied cells.
    """
    if supersample < 1:
        raise ValueError("supersample must be at least one")
    V_vox = np.asarray(V_vox, dtype=np.float64)
    T = np.asarray(T, dtype=np.int32)
    shape_nxyz = tuple(int(value) for value in shape_nxyz)
    axis = resolve_sweep_axis(shape_nxyz, sweep_axis)
    permutation = _SWEEP_PERMUTATIONS[axis]
    inverse = tuple(int(value) for value in np.argsort(permutation))

    local_mask = _voxelise_z_slices(
        V_vox[:, permutation],
        T,
        tuple(shape_nxyz[index] for index in permutation),
        supersample=supersample,
    )
    result = np.transpose(local_mask, inverse)
    if preserve_thin_features and not np.any(result):
        result |= _preserve_empty_thin_solid(V_vox, T, shape_nxyz)
    return result


# =============================================================================
# 3) Material grid assembly + overlap policy
# =============================================================================


def _merge_layer(
    mat_layer: np.ndarray,  # (nx,ny) int16
    solid_mask_layer: np.ndarray,  # (nx,ny) bool
    material_id: int,
    *,
    merge_mode: str,
    priority_layer: Optional[np.ndarray],  # (nx,ny) int16/int32
    new_priority: int,
) -> None:
    """
    Merge one z layer in place, retaining -1 as the empty-cell sentinel.

    Priority ties retain the existing owner because comparison is strict;
    changing ``>`` to ``>=`` would change overlap ownership.
    """
    if merge_mode == "last_wins":
        mat_layer[solid_mask_layer] = np.int16(material_id)

    elif merge_mode == "first_wins":
        empty = mat_layer == -1
        mat_layer[solid_mask_layer & empty] = np.int16(material_id)

    elif merge_mode == "priority":
        if priority_layer is None:
            raise ValueError("priority_layer required for merge_mode='priority'")
        # fill where empty OR new priority is higher
        empty = mat_layer == -1
        better = new_priority > priority_layer
        take = solid_mask_layer & (empty | better)
        mat_layer[take] = np.int16(material_id)
        priority_layer[take] = np.int32(new_priority)

    else:
        raise ValueError("merge_mode must be one of: 'last_wins', 'first_wins', 'priority'")


def voxelise_material_grid(
    meshes: Sequence[TriangleMesh],
    *,
    dx: float,
    dy: Optional[float] = None,
    dz: Optional[float] = None,
    pad: int = 2,
    bbox_world: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    merge_mode: str = "priority",
    supersample: int = 1,
    preserve_thin_features: bool = True,
    sweep_axis: str = "auto",
) -> Tuple[np.ndarray, GridSpec]:
    """
    Voxelise one or more triangle meshes into a single material-index grid.

    merge_mode:
      - "priority"  : higher mesh.priority overwrites lower on overlap;
                      equal priorities retain the earlier mesh
      - "last_wins" : later meshes overwrite earlier
      - "first_wins": earlier meshes keep material on overlap

    sweep_axis:
      - "auto": sweep along the shortest grid dimension (z wins ties)
      - "x", "y", or "z": use an explicitly reproducible slice direction

    Returns:
      mat_grid (nx,ny,nz) int16, with -1 for cells not assigned to a mesh
      grid     GridSpec
    """
    if not meshes:
        raise ValueError("meshes must be non-empty")

    # BBox + grid
    if bbox_world is None:
        vmin, vmax = compute_scene_bbox(meshes)
    else:
        vmin, vmax = bbox_world

    grid = make_grid_from_bbox(vmin, vmax, dx=dx, dy=dy, dz=dz, pad=pad)
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    mat_grid = np.full((nx, ny, nz), fill_value=-1, dtype=np.int16)

    priority_grid: Optional[np.ndarray]
    if merge_mode == "priority":
        priority_grid = np.full((nx, ny, nz), fill_value=np.int32(-(2**30)), dtype=np.int32)
    else:
        priority_grid = None

    # Voxelise each mesh → solid mask → merge
    for m in meshes:
        if m.material_id < 0 or m.material_id > np.iinfo(np.int16).max:
            raise ValueError(
                f"material_id must be in 0..{np.iinfo(np.int16).max}, got {m.material_id}"
            )

        Vv = world_to_voxel_coords(m.vertices_world, grid)
        solid = voxelise_solid_scanline(
            Vv,
            m.triangles,
            (nx, ny, nz),
            supersample=supersample,
            preserve_thin_features=preserve_thin_features,
            sweep_axis=sweep_axis,
        )

        if merge_mode == "priority":
            assert priority_grid is not None
            for z in range(nz):
                _merge_layer(
                    mat_grid[:, :, z],
                    solid[:, :, z],
                    m.material_id,
                    merge_mode=merge_mode,
                    priority_layer=priority_grid[:, :, z],
                    new_priority=m.priority,
                )
        else:
            for z in range(nz):
                _merge_layer(
                    mat_grid[:, :, z],
                    solid[:, :, z],
                    m.material_id,
                    merge_mode=merge_mode,
                    priority_layer=None,
                    new_priority=m.priority,
                )

    return mat_grid, grid


# =============================================================================
# 4) gprMax I/O
# =============================================================================


def write_gprmax_hdf5(path_h5: str, mat_grid: np.ndarray, grid: GridSpec) -> None:
    """
    Write gprMax geometry file:
      dataset "data": material index grid
      attrs   "dx_dy_dz": (dx,dy,dz)

    ``mat_grid`` uses x, y, z axes and -1 for unwritten cells. Spacing is
    written unchanged, so the caller must provide metres for gprMax input.
    Origin and shape are also stored as metadata; origin does not replace
    the placement coordinates supplied to the geometry-import command.
    """
    if not _HAS_H5PY:
        raise RuntimeError("h5py not installed. pip install h5py")

    mat_grid = np.asarray(mat_grid, dtype=np.int16)
    if mat_grid.ndim != 3:
        raise ValueError("mat_grid must be 3D (nx,ny,nz)")
    if mat_grid.shape != (grid.nx, grid.ny, grid.nz):
        raise ValueError("mat_grid shape does not match grid")

    with h5py.File(path_h5, "w") as f:
        f.create_dataset("data", data=mat_grid, compression="gzip")
        f.attrs["dx_dy_dz"] = (
            float(grid.dxyz_world[0]),
            float(grid.dxyz_world[1]),
            float(grid.dxyz_world[2]),
        )
        f.attrs["origin_xyz"] = (
            float(grid.origin_world[0]),
            float(grid.origin_world[1]),
            float(grid.origin_world[2]),
        )
        f.attrs["shape_nxyz"] = (int(grid.nx), int(grid.ny), int(grid.nz))


# =============================================================================
# 4.5) Voxel cache I/O
# =============================================================================


def write_voxel_cache_hdf5(
    path_h5: str,
    mat_grid: np.ndarray,
    grid: GridSpec,
    *,
    cache_key: str,
    meta: Optional[Dict[str, Any]] = None,
    compression: Optional[str] = "gzip",
) -> None:
    """
    Write an existing voxel grid through a temporary file and os.replace.

    Stored format (HDF5):
      dataset "data": int16 grid (nx, ny, nz)
      attrs:
        - dx_dy_dz    : (dx, dy, dz) float
        - origin_xyz  : (ox, oy, oz) float
        - shape_nxyz  : (nx, ny, nz) int
        - cache_key   : str
        - meta_json   : str (optional, JSON)

    Notes:
      - No voxel classification or unit conversion occurs here.
      - The completed .tmp file replaces the destination after HDF5 is closed.
    """
    if not _HAS_H5PY:
        raise RuntimeError("h5py not installed. pip install h5py")

    mat_grid = np.asarray(mat_grid, dtype=np.int16)
    if mat_grid.ndim != 3:
        raise ValueError("mat_grid must be 3D (nx,ny,nz)")
    if mat_grid.shape != (grid.nx, grid.ny, grid.nz):
        raise ValueError(
            f"mat_grid shape {mat_grid.shape} does not match grid {(grid.nx, grid.ny, grid.nz)}"
        )

    # Prepare metadata JSON
    meta_json = ""
    if meta is not None:
        # ensure serializable and deterministic ordering
        meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))

    tmp_path = path_h5 + ".tmp"

    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("data", data=mat_grid, compression=compression)

        f.attrs["dx_dy_dz"] = (
            float(grid.dxyz_world[0]),
            float(grid.dxyz_world[1]),
            float(grid.dxyz_world[2]),
        )
        f.attrs["origin_xyz"] = (
            float(grid.origin_world[0]),
            float(grid.origin_world[1]),
            float(grid.origin_world[2]),
        )
        f.attrs["shape_nxyz"] = (int(grid.nx), int(grid.ny), int(grid.nz))
        f.attrs["cache_key"] = str(cache_key)

        if meta_json:
            f.attrs["meta_json"] = meta_json

    # Publish only after the temporary HDF5 file has been closed.
    os.replace(tmp_path, path_h5)


def read_voxel_cache_hdf5(path_h5: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read cached voxel grid and relevant attrs.

    Returns:
      mat_grid : np.ndarray (int16) shape (nx,ny,nz)
      attrs    : dict with keys like dx_dy_dz, origin_xyz, shape_nxyz, cache_key, meta
    """
    if not _HAS_H5PY:
        raise RuntimeError("h5py not installed. pip install h5py")

    with h5py.File(path_h5, "r") as f:
        if "data" not in f:
            raise ValueError("HDF5 file missing dataset 'data'")

        mat_grid = np.asarray(f["data"], dtype=np.int16)

        attrs: Dict[str, Any] = {}
        for k in ("dx_dy_dz", "origin_xyz", "shape_nxyz", "cache_key", "meta_json"):
            if k in f.attrs:
                attrs[k] = f.attrs[k]

    # Normalize attribute types
    if "dx_dy_dz" in attrs:
        attrs["dx_dy_dz"] = tuple(float(x) for x in attrs["dx_dy_dz"])
    if "origin_xyz" in attrs:
        attrs["origin_xyz"] = tuple(float(x) for x in attrs["origin_xyz"])
    if "shape_nxyz" in attrs:
        attrs["shape_nxyz"] = tuple(int(x) for x in attrs["shape_nxyz"])
    if "cache_key" in attrs:
        attrs["cache_key"] = str(attrs["cache_key"])

    # Parse meta_json if present
    if "meta_json" in attrs:
        try:
            attrs["meta"] = json.loads(str(attrs["meta_json"]))
        except Exception:
            attrs["meta"] = None

    return mat_grid, attrs


def validate_cached_voxel_grid(
    mat_grid: np.ndarray,
    attrs: Dict[str, Any],
    *,
    expect_shape: Optional[Tuple[int, int, int]] = None,
) -> bool:
    """
    Check array rank, integer dtype, optional shape and required attributes.

    Attribute values and material IDs are not validated here. The caller
    must compare cache keys separately before reusing a grid for another mesh.
    """
    try:
        if not isinstance(mat_grid, np.ndarray):
            return False
        if mat_grid.ndim != 3:
            return False

        # Shape checks
        if expect_shape is not None and tuple(mat_grid.shape) != tuple(expect_shape):
            return False
        if "shape_nxyz" in attrs and tuple(mat_grid.shape) != tuple(attrs["shape_nxyz"]):
            return False

        # Required attrs
        if "dx_dy_dz" not in attrs or "origin_xyz" not in attrs:
            return False

        # Dtype should be integer-like
        if mat_grid.dtype.kind not in ("i", "u"):
            return False

        return True
    except Exception:
        return False


# =============================================================================
# 5) Convenience wrapper for voxelising a single triangle mesh
# =============================================================================


def world_mesh_to_gprmax_grid(
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    *,
    material_id: int = 1,
    priority: int = 0,
    dx: float = 0.005,
    dy: Optional[float] = None,
    dz: Optional[float] = None,
    pad: int = 2,
    merge_mode: str = "priority",
    supersample: int = 1,
    preserve_thin_features: bool = True,
    sweep_axis: str = "auto",
) -> Tuple[np.ndarray, GridSpec]:
    """
    Convenience wrapper for a single mesh.
    """
    mesh = TriangleMesh(
        vertices_world=np.asarray(vertices_world),
        triangles=np.asarray(triangles),
        material_id=int(material_id),
        priority=int(priority),
        name="mesh",
    )
    return voxelise_material_grid(
        [mesh],
        dx=dx,
        dy=dy,
        dz=dz,
        pad=pad,
        merge_mode=merge_mode,
        supersample=supersample,
        preserve_thin_features=preserve_thin_features,
        sweep_axis=sweep_axis,
    )
