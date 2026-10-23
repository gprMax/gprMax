# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Voxelise STL surfaces on a gprMax cell grid.

The grid is snapped to the lattice of whole cells from the coordinate origin,
so its lower corner is a multiple of the cell size, and a cell belongs to a
part when its centre is inside that part's closed surface. Voxelisation uses
STEPtoVoxel's scanline voxeliser, which needs no CAD kernel.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from os import PathLike
from typing import Sequence

import numpy as np
from stl import mesh

from gprMax.toolboxes.STEPtoVoxel.voxeliser import TriangleMesh, voxelise_material_grid

UNIT_TO_METRES = {"m": 1.0, "mm": 1e-3, "um": 1e-6}
ROTATE_ABOUT = ("bbox", "origin")


@dataclass(frozen=True)
class STLVoxels:
    """Voxelised STL parts on a cell grid.

    Attributes:
        data: int16 array with x, y, z axes. Each cell holds the index of the
            part that contains its centre, or -1 where no part does.
        origin: lower corner of cell (0, 0, 0), in metres. Placing the file
            with ``GeometryObjectsRead`` at ``p1 = origin`` puts the cells at
            the STL's (transformed) coordinates.
        spacing: cell size in x, y, z, in metres.
    """

    data: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]


def unit_scale(source_unit: str) -> float:
    """Return the length in metres of one STL coordinate unit.

    Args:
        source_unit: ``"m"``, ``"mm"`` or ``"um"`` (``"µm"`` is accepted).

    Returns:
        Metres per unit.

    Raises:
        ValueError: if the unit is not one of those above.
    """

    try:
        return UNIT_TO_METRES[str(source_unit).lower().replace("µ", "u")]
    except KeyError as exc:
        raise ValueError("source_unit must be m, mm, or um") from exc


def read_stl_triangles(path: str | PathLike[str]) -> np.ndarray:
    """Read an STL file's triangles in its own coordinate unit.

    Args:
        path: ASCII or binary STL file.

    Returns:
        float64 array of shape ``(M, 3, 3)``: triangle, vertex, x/y/z.
    """

    return np.asarray(mesh.Mesh.from_file(str(path)).vectors, dtype=np.float64).copy()


def rotation_matrix(rotate_deg: Sequence[float]) -> np.ndarray:
    """Return the rotation by ``(rx, ry, rz)`` degrees about fixed x, y, z axes.

    The rotation about x is applied first, then y, then z (right-handed,
    anticlockwise when looking down each axis towards the origin).

    Args:
        rotate_deg: angles about x, y and z, in degrees.

    Returns:
        3 x 3 matrix ``R`` such that a rotated point is ``R @ p``.
    """

    rx, ry, rz = np.radians(np.asarray(rotate_deg, dtype=np.float64))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    about_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    about_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    about_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return about_z @ about_y @ about_x


def transform_triangles(
    triangle_sets: Sequence[np.ndarray],
    *,
    rotate: Sequence[float] | None = None,
    rotate_about: str = "bbox",
    scale: float = 1.0,
    translate: Sequence[float] | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Scale, rotate and translate several parts together, as one rigid body.

    Each vertex ``p`` becomes ``s * (R @ (p - c) + c) + t``: a uniform scale
    ``s`` about the coordinate origin, a rotation ``R`` about the centre ``c``
    (see :func:`rotation_matrix`) and a translation ``t``. ``c`` is the centre
    of the bounding box of all parts with ``rotate_about="bbox"``, or the
    origin with ``"origin"``. Translations are in the coordinate unit after
    scaling, i.e. the STL's unit.

    Args:
        triangle_sets: one ``(M, 3, 3)`` array per part.
        rotate: angles about x, y and z, in degrees, or None.
        rotate_about: ``"bbox"`` or ``"origin"``.
        scale: positive uniform scale factor.
        translate: shift in x, y and z, or None.

    Returns:
        The transformed parts (new arrays) and the rotation centre ``c`` before
        scaling (None when there is no rotation).

    Raises:
        ValueError: for an invalid angle, centre, scale or translation.
    """

    if rotate_about not in ROTATE_ABOUT:
        raise ValueError("rotate_about must be 'bbox' or 'origin'")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    shift = np.zeros(3) if translate is None else np.asarray(translate, dtype=np.float64)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("translate must contain three finite values")

    parts = [np.array(triangles, dtype=np.float64, copy=True) for triangles in triangle_sets]
    centre = None
    if rotate is not None:
        angles = np.asarray(rotate, dtype=np.float64)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError("rotate must contain three finite angles in degrees")
        if rotate_about == "bbox":
            lower = np.min([part.reshape(-1, 3).min(axis=0) for part in parts], axis=0)
            upper = np.max([part.reshape(-1, 3).max(axis=0) for part in parts], axis=0)
            centre = 0.5 * (lower + upper)
        else:
            centre = np.zeros(3)
        if np.any(angles):
            matrix = rotation_matrix(angles)
            for part in parts:
                part[...] = (part - centre) @ matrix.T + centre
    for part in parts:
        if scale != 1.0:
            part *= scale
        part += shift
    return parts, centre


def voxelise_triangles(
    triangle_sets: Sequence[np.ndarray],
    spacing: Sequence[float],
    *,
    pad: int = 0,
    supersample: int = 1,
) -> STLVoxels:
    """Voxelise closed triangle surfaces given in metres.

    Later parts overwrite earlier ones where they overlap, so pass parts in
    ascending priority.

    Args:
        triangle_sets: one ``(M, 3, 3)`` array per part, in metres.
        spacing: cell size in x, y, z, in metres.
        pad: empty cells added on every side.
        supersample: samples per cell and axis; 1 tests each cell centre,
            larger values fill a cell when most of its samples are inside.

    Returns:
        The part-index grid and its placement.
    """

    if not triangle_sets:
        raise ValueError("at least one mesh is required")
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("discretization must contain three positive finite values")
    if int(pad) != pad or pad < 0:
        raise ValueError("pad must be a non-negative whole number of cells")
    meshes = []
    for index, triangles in enumerate(triangle_sets):
        triangles = np.asarray(triangles, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
            raise ValueError("each mesh must be a non-empty (M, 3, 3) triangle array")
        meshes.append(
            TriangleMesh(
                triangles.reshape(-1, 3),
                np.arange(3 * len(triangles), dtype=np.int32).reshape(-1, 3),
                index,
            )
        )
    data, grid = voxelise_material_grid(
        meshes,
        dx=float(spacing[0]),
        dy=float(spacing[1]),
        dz=float(spacing[2]),
        pad=int(pad),
        merge_mode="last_wins",
        supersample=int(supersample),
    )
    return STLVoxels(
        data,
        tuple(float(value) for value in grid.origin_world),
        tuple(float(value) for value in grid.dxyz_world),
    )


def voxelise_stl_files(
    input_file_paths: Sequence[str | PathLike[str]],
    discretization: Sequence[float],
    *,
    source_unit: str = "mm",
    pad: int = 0,
    supersample: int = 1,
    rotate: Sequence[float] | None = None,
    rotate_about: str = "bbox",
    scale: float = 1.0,
    translate: Sequence[float] | None = None,
) -> STLVoxels:
    """Voxelise STL files, with their placement, on the model's cell grid.

    Args:
        input_file_paths: STL files in ascending priority (later files win
            where parts overlap).
        discretization: cell size in x, y, z, in metres.
        source_unit: unit of the STL coordinates, ``"m"``, ``"mm"`` or
            ``"um"``.
        pad: empty cells added on every side.
        supersample: samples per cell and axis (see :func:`voxelise_triangles`).
        rotate, rotate_about, scale, translate: placement applied to the
            vertices before voxelisation, in the STL's unit (see
            :func:`transform_triangles`).

    Returns:
        The part-index grid (part ``i`` is ``input_file_paths[i]``) and its
        placement in metres.
    """

    metres = unit_scale(source_unit)
    parts, _ = transform_triangles(
        [read_stl_triangles(path) for path in input_file_paths],
        rotate=rotate,
        rotate_about=rotate_about,
        scale=scale,
        translate=translate,
    )
    return voxelise_triangles(
        [part * metres for part in parts],
        discretization,
        pad=pad,
        supersample=supersample,
    )


def _warn_parallel(parallel: bool) -> None:
    if parallel:
        warnings.warn(
            "STLtoVoxel's parallel option is deprecated and has no effect: the scanline "
            "voxeliser is serial and fast",
            DeprecationWarning,
            stacklevel=3,
        )


def convert_meshes(meshes, discretization, parallel=False):
    """Voxelise triangle arrays given in millimetres.

    Args:
        meshes: one ``(M, 3, 3)`` triangle array per part, in millimetres, in
            ascending priority.
        discretization: cell size in x, y, z, in metres.
        parallel: deprecated and ignored.

    Returns:
        ``(vol, scale, shift)``: ``vol`` is the int16 part-index grid with
        z, y, x axes (-1 where empty); a vertex ``p`` (mm) is at voxel
        coordinates ``(p - shift) * scale``, so ``shift`` is the grid's lower
        corner in millimetres and ``scale`` is cells per millimetre.
    """

    _warn_parallel(parallel)
    if not len(meshes):
        raise ValueError("at least one mesh is required")
    voxels = voxelise_triangles(
        [np.asarray(part, dtype=np.float64) * 1e-3 for part in meshes], discretization
    )
    scale = 1 / (np.asarray(voxels.spacing) * 1000)
    shift = np.asarray(voxels.origin) * 1000
    return np.ascontiguousarray(np.transpose(voxels.data)), scale, shift


def convert_file(input_file_path, discretization, pad=1, parallel=False, source_unit="mm"):
    """Voxelise one STL file; see :func:`convert_files`."""

    _warn_parallel(parallel)
    return convert_files([input_file_path], discretization, pad=pad, source_unit=source_unit)


def convert_files(
    input_file_paths,
    discretization,
    colors=None,
    pad=0,
    parallel=False,
    source_unit="mm",
    **placement,
):
    """Convert one or more STL files to a gprMax material-index array.

    ``colors`` is retained for compatibility with the upstream API but is not
    used by gprMax. ``pad`` adds background cells around the converted object.
    ``parallel`` is deprecated and ignored. ``placement`` takes the
    ``rotate``, ``rotate_about``, ``scale``, ``translate`` and ``supersample``
    arguments of :func:`voxelise_stl_files`, which also returns the grid's
    origin.

    Returns:
        int16 array with x, y, z axes: the index of the file whose surface
        contains each cell centre, or -1.
    """

    _warn_parallel(parallel)
    if pad < 0:
        raise ValueError("pad must be non-negative")
    return voxelise_stl_files(
        input_file_paths, discretization, pad=pad, source_unit=source_unit, **placement
    ).data
