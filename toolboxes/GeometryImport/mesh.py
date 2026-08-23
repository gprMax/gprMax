# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Voxelise labelled surface or unstructured meshes for gprMax."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gprMax.material_database import make_database_id
from toolboxes.STEPtoVoxel.voxeliser import (
    GridSpec,
    TriangleMesh,
    make_grid_from_bbox,
    voxelise_material_grid,
)

from .common import (
    build_tag_volume,
    normalise_tag_name,
    unique_normalised_tags,
    write_geometry_hdf5,
    write_geometry_preview,
    write_null_material_database,
)

MESH_COLUMNS = ("region", "include", "priority", "material_name", "geometry_tag")
_UNIT_FACTORS = {"m": 1.0, "mm": 1e-3, "um": 1e-6}
_INTERNAL_REGION_ARRAY = "__gprmax_region_id"
_PREFERRED_REGION_ARRAYS = (
    "gmsh:physical",
    "PhysicalGroup",
    "RegionId",
    "region_id",
    "material_id",
    "part_id",
)


@dataclass(frozen=True)
class MeshRegion:
    source_value: str
    name: str
    compact_id: int
    cell_count: int


@dataclass(frozen=True)
class MeshSource:
    dataset: object
    regions: tuple[MeshRegion, ...]
    kind: str
    source_region_array: str | None


@dataclass(frozen=True)
class MeshAssignment:
    region: str
    include: bool
    priority: int
    material_name: str | None
    geometry_tag: str | None


@dataclass(frozen=True)
class MeshConversionResult:
    geometry_file: Path
    preview_file: Path | None
    materials_file: Path
    manifest_file: Path
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    kind: str


def _pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Gmsh and VTK-family mesh import requires PyVista/VTK") from exc
    return pv


def _cell_dimensions(dataset) -> np.ndarray:
    """Return the topological dimension of every cell in a dataset."""

    pv = _pyvista()
    if isinstance(dataset, pv.PolyData):
        if dataset.n_faces:
            return np.full(dataset.n_cells, 2, dtype=np.int8)
        raise ValueError("PolyData input has no polygonal surface cells")
    try:
        import vtk
    except ImportError as exc:
        raise RuntimeError("Mesh dimensionality inspection requires VTK") from exc
    utility = getattr(vtk, "vtkCellTypeUtilities", vtk.vtkCellTypes)
    celltypes = dataset.cast_to_unstructured_grid().celltypes
    dimensions = np.asarray(
        [int(utility.GetDimension(int(cell_type))) for cell_type in celltypes],
        dtype=np.int8,
    )
    if not np.any(dimensions >= 2):
        raise ValueError("Mesh has no surface or volume cells")
    return dimensions


def _retain_highest_dimension(dataset):
    """Discard lower-dimensional boundary entities from mixed FEM meshes.

    Gmsh commonly stores 2-D physical boundary groups alongside the 3-D
    tetrahedra or hexahedra which define a volume.  Those boundary entities
    are useful to FEM solvers but do not own FDTD cells and must not become
    gprMax material/tag regions.
    """

    dimensions = _cell_dimensions(dataset)
    highest = int(dimensions.max())
    if highest not in {2, 3}:
        raise ValueError("Mesh has no surface or volume cells")
    keep = dimensions == highest
    if np.all(keep):
        return dataset, highest
    return dataset.cast_to_unstructured_grid().extract_cells(keep), highest


def _field_region_names(dataset, dimension: int) -> dict[str, str]:
    """Recover Gmsh physical names where the reader retained field data."""

    names = {}
    for name, value in dataset.field_data.items():
        array = np.asarray(value).ravel()
        if not array.size:
            continue
        try:
            source_value = array[0].item()
            if not isinstance(source_value, (int, float, np.integer, np.floating)):
                continue
            field_dimension = int(array[1]) if array.size >= 2 else dimension
        except (TypeError, ValueError, OverflowError):
            continue
        if field_dimension == dimension:
            names[str(source_value)] = str(name)
    return names


def _meshio_field_region_names(path: Path, dimension: int) -> dict[str, str]:
    if path.suffix.lower() != ".msh":
        return {}
    try:
        import meshio
    except ImportError:
        return {}
    mesh = meshio.read(path)
    return {
        str(np.asarray(value).ravel()[0]): str(name)
        for name, value in mesh.field_data.items()
        if np.asarray(value).size
        and (np.asarray(value).size < 2 or int(np.asarray(value).ravel()[1]) == dimension)
    }


def _available_cell_arrays(source) -> set[str]:
    pv = _pyvista()
    if isinstance(source, pv.MultiBlock):
        populated = [block for block in source if block is not None and block.n_cells]
        if not populated:
            return set()
        highest = max(int(_cell_dimensions(block).max()) for block in populated)
        blocks = [
            block
            for block in populated
            if np.any(_cell_dimensions(block) == highest)
        ]
        available = set(blocks[0].cell_data.keys())
        for block in blocks[1:]:
            available &= set(block.cell_data.keys())
        return available
    return set(source.cell_data.keys())


def _combine_blocks(source, source_path: Path, region_array: str | None):
    pv = _pyvista()
    if not isinstance(source, pv.MultiBlock):
        return source, None, {}
    populated = [block for block in source if block is not None and block.n_cells]
    if not populated:
        raise ValueError(f"No mesh cells were read from {source_path}")
    highest = max(int(_cell_dimensions(block).max()) for block in populated)
    blocks = []
    block_names = {}
    for index, block in enumerate(source):
        if block is None or block.n_cells == 0:
            continue
        dimensions = _cell_dimensions(block)
        keep = dimensions == highest
        if not np.any(keep):
            continue
        item = block.copy(deep=True)
        if not np.all(keep):
            item = item.cast_to_unstructured_grid().extract_cells(keep)
        if region_array is None:
            item.cell_data[_INTERNAL_REGION_ARRAY] = np.full(item.n_cells, index, dtype=np.int32)
            block_names[str(index)] = source.get_block_name(index) or f"block_{index}"
        blocks.append(item)
    if not blocks:
        raise ValueError(f"No mesh cells were read from {source_path}")
    combined = pv.MultiBlock(blocks).combine(merge_points=False)
    return (
        combined,
        _INTERNAL_REGION_ARRAY if region_array is None else region_array,
        block_names,
    )


def load_mesh_source(
    path: str | Path,
    *,
    unit: str,
    region_array: str | None = None,
) -> MeshSource:
    """Load Gmsh/VTK/VTP/VTU geometry and identify its semantic regions."""

    pv = _pyvista()
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    try:
        factor = _UNIT_FACTORS[unit.lower().replace("µ", "u")]
    except KeyError as exc:
        raise ValueError("Mesh coordinates require an explicit unit: m, mm, or um") from exc
    loaded = pv.read(source_path)
    if region_array is None:
        available = _available_cell_arrays(loaded)
        region_array = next(
            (candidate for candidate in _PREFERRED_REGION_ARRAYS if candidate in available),
            None,
        )
    dataset, effective_array, block_names = _combine_blocks(loaded, source_path, region_array)
    dataset = dataset.copy(deep=True)
    dataset, dimension = _retain_highest_dimension(dataset)
    dataset.points = np.asarray(dataset.points, dtype=np.float64) * factor
    if dataset.n_cells == 0:
        raise ValueError(f"No mesh cells were read from {source_path}")

    if effective_array is None:
        effective_array = region_array
    if effective_array is None:
        dataset.cell_data[_INTERNAL_REGION_ARRAY] = np.zeros(dataset.n_cells, dtype=np.int32)
        effective_array = _INTERNAL_REGION_ARRAY
        values = np.asarray([0])
        names_by_value = {"0": source_path.stem}
    else:
        if effective_array not in dataset.cell_data:
            available = sorted(dataset.cell_data.keys())
            raise ValueError(f"Cell-data region array {effective_array!r} was not found; available={available}")
        raw = np.asarray(dataset.cell_data[effective_array])
        if raw.ndim != 1 or len(raw) != dataset.n_cells:
            raise ValueError("The selected region array must contain one scalar per mesh cell")
        values = np.unique(raw)
        names_by_value = _field_region_names(dataset, dimension)
        names_by_value.update(_meshio_field_region_names(source_path, dimension))
        names_by_value.update(block_names)
        compact = np.empty(dataset.n_cells, dtype=np.int32)
        for index, value in enumerate(values):
            compact[raw == value] = index
        dataset.cell_data[_INTERNAL_REGION_ARRAY] = compact

    compact_values = np.asarray(dataset.cell_data[_INTERNAL_REGION_ARRAY], dtype=np.int32)
    regions = []
    for compact_id, value in enumerate(values):
        source_value = str(value.item() if hasattr(value, "item") else value)
        name = names_by_value.get(source_value, f"region_{source_value}")
        regions.append(
            MeshRegion(
                source_value,
                name,
                compact_id,
                int(np.count_nonzero(compact_values == compact_id)),
            )
        )
    kind = "volume" if dimension == 3 else "surface"
    return MeshSource(dataset, tuple(regions), kind, region_array)


def write_mesh_template(source: MeshSource, path: str | Path, *, overwrite=False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    tags = unique_normalised_tags([region.name for region in source.regions])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MESH_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for priority, (region, tag) in enumerate(zip(source.regions, tags)):
            writer.writerow(
                {
                    "region": region.source_value,
                    "include": "y",
                    "priority": priority,
                    "material_name": region.name,
                    "geometry_tag": tag,
                }
            )
    return target


def _yes(value: str) -> bool:
    clean = value.strip().lower()
    if clean in {"y", "yes", "true", "1"}:
        return True
    if clean in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"include must be y/n, got {value!r}")


def read_mesh_assignments(path: str | Path) -> tuple[MeshAssignment, ...]:
    result = []
    seen = set()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(MESH_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            region = row["region"].strip()
            if not region or region in seen:
                raise ValueError(f"Mesh regions must be non-empty and unique: {region!r}")
            seen.add(region)
            include = _yes(row["include"])
            material = row["material_name"].strip() or None
            if include and material is None:
                raise ValueError(f"material_name is required for region {region}")
            raw_tag = row["geometry_tag"].strip()
            result.append(
                MeshAssignment(
                    region,
                    include,
                    int(row["priority"]),
                    material,
                    normalise_tag_name(raw_tag) if include and raw_tag else None,
                )
            )
    return tuple(result)


def _surface_components(source: MeshSource, selected, spacing, pad_cells, supersample):
    dataset = source.dataset
    source_by_value = {region.source_value: region for region in source.regions}
    meshes = []
    ordered = sorted(selected, key=lambda item: item.priority)
    for component_id, assignment in enumerate(ordered):
        region = source_by_value[assignment.region]
        selected_cells = np.asarray(dataset.cell_data[_INTERNAL_REGION_ARRAY]) == region.compact_id
        surface = dataset.extract_cells(selected_cells).extract_surface(algorithm="dataset_surface").triangulate()
        if surface.n_open_edges:
            raise ValueError(
                f"Surface region {region.name!r} is not watertight " f"({surface.n_open_edges} open edges)"
            )
        faces = np.asarray(surface.faces).reshape(-1, 4)
        if faces.size and not np.all(faces[:, 0] == 3):
            raise ValueError(f"Surface region {region.name!r} could not be triangulated")
        meshes.append(
            TriangleMesh(
                np.asarray(surface.points, dtype=np.float64),
                np.asarray(faces[:, 1:4], dtype=np.int32),
                component_id,
                assignment.priority,
                region.name,
            )
        )
    return (
        *voxelise_material_grid(
            meshes,
            dx=spacing[0],
            dy=spacing[1],
            dz=spacing[2],
            pad=pad_cells,
            merge_mode="priority",
            supersample=supersample,
            sweep_axis="auto",
            preserve_thin_features=True,
        ),
        ordered,
    )


def _sample_volume(dataset, grid: GridSpec, *, z_chunk_cells: int = 32) -> np.ndarray:
    """Probe an unstructured volume at FDTD cell centres in bounded chunks."""

    pv = _pyvista()
    dataset = dataset.copy(deep=False)
    dataset.set_active_scalars(_INTERNAL_REGION_ARRAY, preference="cell")
    result = np.full((grid.nx, grid.ny, grid.nz), -1, dtype=np.int32)
    for z0 in range(0, grid.nz, z_chunk_cells):
        nz = min(z_chunk_cells, grid.nz - z0)
        image = pv.ImageData(
            dimensions=(grid.nx + 1, grid.ny + 1, nz + 1),
            spacing=tuple(float(value) for value in grid.dxyz_world),
            origin=(
                float(grid.origin_world[0]),
                float(grid.origin_world[1]),
                float(grid.origin_world[2] + z0 * grid.dxyz_world[2]),
            ),
        )
        centres = image.cell_centers()
        # Cell data are copied from the containing volume cell; they are not
        # interpolated like point data, so region IDs remain discrete without
        # VTK's categorical point-scalar mode.
        sampled = centres.sample(dataset, categorical=False, pass_cell_data=False)
        valid = np.asarray(sampled["vtkValidPointMask"], dtype=bool).reshape((grid.nx, grid.ny, nz), order="F")
        values = np.asarray(sampled[_INTERNAL_REGION_ARRAY], dtype=np.int32).reshape((grid.nx, grid.ny, nz), order="F")
        block = result[:, :, z0 : z0 + nz]
        block[valid] = values[valid]
    return result


def _volume_components(source: MeshSource, selected, spacing, pad_cells):
    bounds = np.asarray(source.dataset.bounds, dtype=float)
    grid = make_grid_from_bbox(
        np.asarray((bounds[0], bounds[2], bounds[4])),
        np.asarray((bounds[1], bounds[3], bounds[5])),
        dx=spacing[0],
        dy=spacing[1],
        dz=spacing[2],
        pad=pad_cells,
    )
    all_regions = _sample_volume(source.dataset, grid)
    ordered = sorted(selected, key=lambda item: item.priority)
    source_by_value = {region.source_value: region for region in source.regions}
    components = np.full(all_regions.shape, -1, dtype=np.int32)
    for component_id, assignment in enumerate(ordered):
        compact_id = source_by_value[assignment.region].compact_id
        components[all_regions == compact_id] = component_id
    return components, grid, ordered


def convert_mesh(
    source_path: str | Path,
    assignments_csv: str | Path,
    output_dir: str | Path,
    *,
    voxel_size: tuple[float, float, float],
    unit: str,
    region_array: str | None = None,
    pad_cells: int = 2,
    supersample: int = 1,
) -> MeshConversionResult:
    """Voxelise a labelled surface or unstructured-volume mesh."""

    if (
        len(voxel_size) != 3
        or not np.isfinite(voxel_size).all()
        or any(value <= 0 for value in voxel_size)
    ):
        raise ValueError("voxel_size must contain three positive values in metres")
    if pad_cells < 0:
        raise ValueError("pad_cells must be non-negative")
    if supersample < 1:
        raise ValueError("supersample must be at least one")
    source = load_mesh_source(source_path, unit=unit, region_array=region_array)
    assignments = read_mesh_assignments(assignments_csv)
    actual = {region.source_value for region in source.regions}
    declared = {assignment.region for assignment in assignments}
    if actual != declared:
        raise ValueError(
            f"Assignments do not match mesh regions; missing={sorted(actual-declared)}, "
            f"unknown={sorted(declared-actual)}"
        )
    selected = [assignment for assignment in assignments if assignment.include]
    if not selected:
        raise ValueError("No mesh regions have include=y")
    if source.kind == "surface":
        component_grid, grid, ordered = _surface_components(source, selected, voxel_size, pad_cells, supersample)
    else:
        component_grid, grid, ordered = _volume_components(source, selected, voxel_size, pad_cells)

    material_names = []
    material_ids = {}
    component_material_ids = []
    component_tags = []
    for assignment in ordered:
        assert assignment.material_name is not None
        material_id = material_ids.get(assignment.material_name)
        if material_id is None:
            material_id = len(material_names)
            material_ids[assignment.material_name] = material_id
            material_names.append(assignment.material_name)
        component_material_ids.append(material_id)
        component_tags.append(assignment.geometry_tag or normalise_tag_name(assignment.region))
    occupied = component_grid >= 0
    if len(material_names) > np.iinfo(np.int16).max + 1:
        raise ValueError("Imported geometry exceeds the int16 material-index schema")
    material_grid = np.full(component_grid.shape, -1, dtype=np.int16)
    lookup = np.asarray(component_material_ids, dtype=np.int16)
    material_grid[occupied] = lookup[component_grid[occupied]]
    tag_data, tag_names = build_tag_volume(component_grid, component_tags)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometry_file = output / "geometry.h5"
    preview_file = output / "geometry_preview.vti"
    materials_file = output / "materials.json"
    manifest_file = output / "conversion.json"
    database_id = make_database_id(materials_file.stem, prefix="geometry")
    if database_id != materials_file.stem:
        materials_file = output / f"{database_id}.json"
    material_keys = write_null_material_database(
        materials_file,
        database_id,
        material_names,
        source="GeometryImport Gmsh/VTK mesh",
    )
    write_geometry_hdf5(
        geometry_file,
        material_grid,
        grid.dxyz_world,
        origin=grid.origin_world,
        material_keys=material_keys,
        material_database=database_id,
        tag_data=tag_data,
        tag_names=tag_names,
    )
    preview_written = write_geometry_preview(
        preview_file,
        material_grid,
        grid.dxyz_world,
        origin=grid.origin_world,
        material_keys=material_keys,
        tag_data=tag_data,
        tag_names=tag_names,
    )
    if not preview_written:
        preview_file = None
    region_by_value = {region.source_value: region for region in source.regions}
    manifest = {
        "source": str(Path(source_path).expanduser().resolve()),
        "mesh_kind": source.kind,
        "source_region_array": source.source_region_array,
        "shape_cells": list(component_grid.shape),
        "spacing_m": list(grid.dxyz_world),
        "grid_origin_m": list(grid.origin_world),
        "regions": [
            {
                "source_value": assignment.region,
                "source_name": region_by_value[assignment.region].name,
                "included": assignment.include,
                "material_name": assignment.material_name,
                "geometry_tag": assignment.geometry_tag,
                "source_cell_count": region_by_value[assignment.region].cell_count,
                "voxel_count": int(np.count_nonzero(component_grid == ordered.index(assignment)))
                if assignment.include
                else 0,
            }
            for assignment in assignments
        ],
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return MeshConversionResult(
        geometry_file,
        preview_file,
        materials_file,
        manifest_file,
        tuple(int(value) for value in component_grid.shape),
        tuple(float(value) for value in grid.dxyz_world),
        source.kind,
    )
