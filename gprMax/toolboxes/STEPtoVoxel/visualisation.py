# Copyright (c) 2026 Mahdee Abir
# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# Originally developed for STEP-to-gprMax and distributed under the MIT
# License. This adapted version is part of gprMax and is distributed under the
# GNU General Public License, version 3 or (at your option) any later version.
# See LICENSE in this directory.

"""Optional PyVista and Matplotlib visualisation utilities."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def image_data(mat_grid: np.ndarray, grid):
    """Create PyVista ImageData with cell-based material/component IDs."""
    import pyvista as pv

    mat_grid = np.asarray(mat_grid)
    image = pv.ImageData()
    image.dimensions = tuple(np.asarray(mat_grid.shape, dtype=int) + 1)
    image.origin = tuple(map(float, grid.origin_world))
    image.spacing = tuple(map(float, grid.dxyz_world))
    return image


def write_vti(path, mat_grid: np.ndarray, grid, scalar_name: str = "material_id") -> None:
    """Write a cell-labelled VTK XML ImageData file for ParaView."""
    image = image_data(mat_grid, grid)
    image.cell_data[scalar_name] = np.asarray(mat_grid, dtype=np.int16).ravel(order="F")
    image.save(path)


def write_reference_geometry(
    path,
    items: Sequence[tuple[int, str, np.ndarray, np.ndarray]],
) -> None:
    """Write non-physical CAD reference geometry as VTK PolyData.

    ``items`` contains ``(reference_id, name, vertices, triangles)`` tuples.
    Triangulated faces, lines and points are all supported. The integer IDs
    are mapped back to names in ``conversion.json``. Keeping this geometry
    separate from the voxel grid allows ports, source edges and construction
    geometry to be inspected without assigning material to them.
    """
    import pyvista as pv

    meshes = []
    for reference_id, _name, vertices, triangles in items:
        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        if not len(vertices):
            continue
        if len(triangles):
            faces = np.column_stack((np.full(len(triangles), 3, dtype=np.int64), triangles)).ravel()
            mesh = pv.PolyData(vertices, faces)
        elif len(vertices) == 1:
            mesh = pv.PolyData(vertices)
            mesh.verts = np.array((1, 0), dtype=np.int64)
        else:
            mesh = pv.PolyData()
            mesh.points = vertices
            segments = np.column_stack(
                (
                    np.full(len(vertices) - 1, 2, dtype=np.int64),
                    np.arange(len(vertices) - 1),
                    np.arange(1, len(vertices)),
                )
            )
            mesh.lines = segments.ravel()
        mesh.cell_data["reference_geometry_id"] = np.full(
            mesh.n_cells, reference_id, dtype=np.int16
        )
        meshes.append(mesh)

    if not meshes:
        raise ValueError("No reference geometry was supplied")

    combined = meshes[0]
    for mesh in meshes[1:]:
        combined = combined.merge(mesh, merge_points=False)
    combined.save(path)


def translate_reference_geometry(source, destination, offset) -> None:
    """Translate reference VTK geometry into an imported model coordinate system.

    ``reference_geometry_cad.vtp`` and ``geometry.vti`` use the original CAD
    coordinates. To overlay the references on a gprMax ``GeometryView``, pass
    ``geometry_import_origin - voxel_grid_origin`` as *offset*.
    """
    import pyvista as pv

    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (3,):
        raise ValueError("reference-geometry offset must contain three coordinates")
    geometry = pv.read(source)
    geometry.translate(offset, inplace=True)
    geometry.save(destination)


def show_voxels_3d(mat_grid: np.ndarray, grid, threshold: int = 0) -> None:
    import pyvista as pv

    mat_grid = np.asarray(mat_grid)
    nx, ny, nz = mat_grid.shape

    filled_count = int((mat_grid >= threshold).sum())
    print(f"Filled voxels (mat>={threshold}): {filled_count}")
    if filled_count == 0:
        print("Nothing to render.")
        return

    ug = image_data(mat_grid, grid)

    # IMPORTANT: keep order="F" to match voxel grid convention
    ug.cell_data["mat"] = mat_grid.ravel(order="F")

    filled = ug.threshold(value=threshold - 0.5, scalars="mat")

    pl = pv.Plotter()
    pl.add_mesh(filled, show_edges=False)
    pl.add_axes()
    pl.show_grid()
    pl.show(title="Voxelised geometry (cell-based)")


def show_voxels_cutaway(
    mat_grid: np.ndarray,
    grid,
    axis: str = "z",
    frac: float = 0.5,
    threshold: int = 0,
) -> None:
    import pyvista as pv

    mat_grid = np.asarray(mat_grid)
    nx, ny, nz = mat_grid.shape

    filled_count = int((mat_grid >= threshold).sum())
    if filled_count == 0:
        print("Nothing to render.")
        return

    ug = image_data(mat_grid, grid)

    ug.cell_data["mat"] = mat_grid.ravel(order="F")

    filled = ug.threshold(value=threshold - 0.5, scalars="mat")

    axis = axis.lower()
    bounds = filled.bounds

    if axis == "x":
        x0 = bounds[0] + frac * (bounds[1] - bounds[0])
        cut = filled.clip(normal=(1, 0, 0), origin=(x0, 0, 0))
    elif axis == "y":
        y0 = bounds[2] + frac * (bounds[3] - bounds[2])
        cut = filled.clip(normal=(0, 1, 0), origin=(0, y0, 0))
    elif axis == "z":
        z0 = bounds[4] + frac * (bounds[5] - bounds[4])
        cut = filled.clip(normal=(0, 0, 1), origin=(0, 0, z0))
    else:
        raise ValueError("axis must be x, y, or z")

    pl = pv.Plotter()
    pl.add_mesh(cut, show_edges=False)
    pl.add_axes()
    pl.show_grid()
    pl.show(title=f"Cutaway axis={axis}, frac={frac}")


def debug_plot_slice(
    mat_grid: np.ndarray,
    grid,
    axis: str = "z",
    index: Optional[int] = None,
) -> None:
    import matplotlib.pyplot as plt

    axis = axis.lower()
    nx, ny, nz = mat_grid.shape

    if axis == "z":
        k = nz // 2 if index is None else int(index)
        img = (mat_grid[:, :, k] >= 0).T
        title = f"Occupancy slice z={k}"
    elif axis == "y":
        k = ny // 2 if index is None else int(index)
        img = (mat_grid[:, k, :] >= 0).T
        title = f"Occupancy slice y={k}"
    elif axis == "x":
        k = nx // 2 if index is None else int(index)
        img = (mat_grid[k, :, :] >= 0).T
        title = f"Occupancy slice x={k}"
    else:
        raise ValueError("axis must be x/y/z")

    print(title, "fill ratio:", float(img.mean()))

    plt.figure()
    plt.imshow(img, origin="lower", interpolation="nearest")
    plt.title(title)
    plt.show()
