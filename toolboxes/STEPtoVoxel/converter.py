# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.
#
# The STEP conversion workflow is derived from STEP-to-gprMax, originally
# developed by Mahdee Abir and distributed under the MIT License. See LICENSE
# and README.rst in this directory for attribution and licence details.

"""High-level STEP assembly to gprMax voxel conversion workflow."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gprMax.material_database import create_database_document, write_database
from toolboxes.GeometryImport.common import (
    build_tag_volume,
    normalise_tag_name,
    unique_normalised_tags,
    write_geometry_hdf5,
)

from .grouping import suggest_material_groups
from .markers import classify_marker_name, marker_record
from .voxeliser import TriangleMesh, resolve_sweep_axis, voxelise_material_grid

LEGACY_MATERIAL_COLUMNS = (
    "part_name",
    "include",
    "priority",
    "material_name",
    "relative_permittivity",
    "conductivity",
    "relative_permeability",
    "magnetic_loss",
)

MATERIAL_COLUMNS = (
    "group_id",
    "group_confidence",
    "similar_group",
    "part_count",
    "part_names",
    "include",
    "priority",
    "material_name",
    "geometry_tag",
    "relative_permittivity",
    "conductivity",
    "relative_permeability",
    "magnetic_loss",
)

# ``geometry_tag`` was added after the original grouped assignment format.
# It is intentionally optional when reading so existing assignment CSV files
# remain valid.
GROUPED_REQUIRED_COLUMNS = tuple(column for column in MATERIAL_COLUMNS if column != "geometry_tag")


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for parsing, tessellating, and voxelising a STEP file."""

    voxel_size: tuple[float, float, float] = (1e-3, 1e-3, 1e-3)
    pad_cells: int = 2
    supersample: int = 1
    sweep_axis: str = "auto"
    merge_mode: str = "priority"
    preserve_thin_features: bool = True
    material_grouping: str = "exact"
    grouping_relative_tolerance: float = 0.01
    linear_deflection: float | None = None
    angular_deflection: float = 0.1
    relative_deflection: bool = False
    force_units_to_metres: bool = True

    def __post_init__(self) -> None:
        if (
            len(self.voxel_size) != 3
            or not np.isfinite(self.voxel_size).all()
            or any(value <= 0 for value in self.voxel_size)
        ):
            raise ValueError("voxel_size must contain three positive values")
        if self.pad_cells < 0:
            raise ValueError("pad_cells must be non-negative")
        if self.supersample < 1:
            raise ValueError("supersample must be at least one")
        if self.sweep_axis not in {"auto", "x", "y", "z"}:
            raise ValueError("sweep_axis must be auto, x, y, or z")
        if self.merge_mode not in {"priority", "first_wins", "last_wins"}:
            raise ValueError("merge_mode must be priority, first_wins, or last_wins")
        if self.material_grouping not in {"none", "exact", "similar"}:
            raise ValueError("material_grouping must be none, exact, or similar")
        if not 0 < self.grouping_relative_tolerance < 1:
            raise ValueError("grouping_relative_tolerance must be between zero and one")


@dataclass(frozen=True)
class ConversionResult:
    """Files and grid metadata produced by :func:`convert_step`."""

    geometry_file: Path
    materials_file: Path
    manifest_file: Path
    markers_file: Path
    vtk_file: Path | None
    reference_geometry_cad_file: Path | None
    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    component_cell_counts: dict[str, int]


@dataclass(frozen=True)
class _Material:
    name: str
    er: float | None
    se: float | None
    mr: float | None
    sm: float | None


@dataclass(frozen=True)
class _Assignment:
    include: bool
    priority: int
    material: _Material | None
    geometry_tag: str | None = None
    group_id: str = ""
    group_confidence: str = "legacy"


def _parser_config(config: ConversionConfig):
    try:
        from .step_parser import ParserConfig
    except ImportError as exc:
        if exc.name and (exc.name == "OCC" or exc.name.startswith("OCC.")):
            raise RuntimeError(
                "STEPtoVoxel requires pythonocc-core. Install it in a compatible "
                "conda environment, for example: conda install -c conda-forge pythonocc-core"
            ) from exc
        raise

    linear_deflection = config.linear_deflection
    if linear_deflection is None:
        linear_deflection = 0.5 * min(config.voxel_size)
    return ParserConfig(
        force_units_to_metres=config.force_units_to_metres,
        linear_deflection=linear_deflection,
        angular_deflection=config.angular_deflection,
        is_relative_deflection=config.relative_deflection,
    )


def inspect_step(step_file: str | Path, config: ConversionConfig | None = None):
    """Parse *step_file* and return its OpenCascade-backed component records."""
    step_file = Path(step_file).expanduser().resolve()
    if not step_file.is_file():
        raise FileNotFoundError(step_file)
    config = config or ConversionConfig()
    parser_config = _parser_config(config)
    from . import step_parser

    parts = step_parser.main(str(step_file), parser_config)
    if not parts:
        raise RuntimeError(f"No components could be read from {step_file}")
    return parts


def _component_records(parts: Sequence[Any]) -> list[dict[str, Any]]:
    records = []
    for part in parts:
        cad = dict(getattr(part, "cad", None) or {})
        volume = float(cad.get("vol_m3") or 0.0)
        records.append(
            {
                "uid": int(part.uid),
                "name": str(part.name),
                "raw_step_name": getattr(part, "raw_step_name", None),
                "name_source": getattr(part, "name_source", None),
                "name_confidence": getattr(part, "name_confidence", None),
                "step_entity_id": getattr(part, "step_entity_id", None),
                "is_solid": volume > 0.0,
                "volume_m3": volume,
                "surface_area_m2": cad.get("area_m2"),
                "bbox_xyzxyz_m": cad.get("bbox_xyzxyz"),
                "bbox_dimensions_m": cad.get("bbox_dims_xyz"),
            }
        )
    return records


def write_material_template(
    step_file: str | Path,
    csv_file: str | Path,
    config: ConversionConfig | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Create an editable part-to-material CSV for a STEP assembly."""
    csv_file = Path(csv_file)
    if csv_file.exists() and not overwrite:
        raise FileExistsError(csv_file)

    config = config or ConversionConfig()
    parts = inspect_step(step_file, config)
    part_by_name = {str(part.name): part for part in parts}
    groups = suggest_material_groups(
        parts,
        mode=config.material_grouping,
        relative_tolerance=config.grouping_relative_tolerance,
    )

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATERIAL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for group in groups:
            members = [part_by_name[name] for name in group.part_names]
            is_solid = all(float(part.cad.get("vol_m3") or 0.0) > 0.0 for part in members)
            is_marker = any(classify_marker_name(part.name) is not None for part in members)
            writer.writerow(
                {
                    "group_id": group.identifier,
                    "group_confidence": group.confidence,
                    "similar_group": group.similar_group,
                    "part_count": len(group.part_names),
                    "part_names": "|".join(group.part_names),
                    "include": "y" if is_solid and not is_marker else "n",
                    "priority": group.priority if is_solid else 0,
                    "material_name": "",
                    "geometry_tag": "",
                    "relative_permittivity": "",
                    "conductivity": "",
                    "relative_permeability": 1,
                    "magnetic_loss": 0,
                }
            )
    return csv_file


def _yes(value: str) -> bool:
    value = value.strip().lower()
    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"include must be y/n, got {value!r}")


def _float(value: str, field: str, part_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{part_name}: {field} must be numeric, got {value!r}") from exc


def _optional_float(value: str, field: str, part_name: str) -> float | None:
    """Return ``None`` for an unassigned template value."""

    return None if not value.strip() else _float(value, field, part_name)


def _validate_material(material: _Material, assignment_name: str) -> None:
    """Reject values which cannot form a valid material-database entry."""

    for field, value in (
        ("relative_permittivity", material.er),
        ("relative_permeability", material.mr),
    ):
        if value is None:
            continue
        if not np.isfinite(value) or value < 1:
            raise ValueError(f"{assignment_name}: {field} must be finite and at least one")
    for field, value in (("conductivity", material.se), ("magnetic_loss", material.sm)):
        if value is None:
            continue
        if np.isnan(value) or np.isneginf(value) or value < 0:
            raise ValueError(f"{assignment_name}: {field} must be non-negative")


def _read_assignments(csv_file: Path) -> dict[str, _Assignment]:
    assignments: dict[str, _Assignment] = {}
    with csv_file.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        grouped = set(GROUPED_REQUIRED_COLUMNS).issubset(fields)
        required = set(GROUPED_REQUIRED_COLUMNS if grouped else LEGACY_MATERIAL_COLUMNS)
        missing = required - fields
        if missing:
            raise ValueError(f"{csv_file} is missing columns: {sorted(missing)}")
        for row in reader:
            if grouped:
                group_id = row["group_id"].strip()
                group_confidence = row["group_confidence"].strip() or "unknown"
                part_names = tuple(
                    name.strip() for name in row["part_names"].split("|") if name.strip()
                )
                if not group_id or not part_names:
                    raise ValueError("group_id and part_names must be non-empty")
                try:
                    expected_count = int(row["part_count"])
                except ValueError as exc:
                    raise ValueError(f"{group_id}: part_count must be an integer") from exc
                if expected_count != len(part_names):
                    raise ValueError(
                        f"{group_id}: part_count={expected_count} does not match "
                        f"the {len(part_names)} names in part_names"
                    )
            else:
                group_id = ""
                group_confidence = "legacy"
                part_names = (row["part_name"].strip(),)
            if any(not name or name in assignments for name in part_names):
                raise ValueError(f"Part names must be non-empty and occur once: {part_names!r}")
            include = _yes(row["include"])
            try:
                priority = int(row["priority"])
            except ValueError as exc:
                raise ValueError(
                    f"{group_id or part_names[0]}: priority must be an integer"
                ) from exc
            material = None
            requested_tag = row.get("geometry_tag", "").strip()
            geometry_tag = normalise_tag_name(requested_tag) if requested_tag else None
            if include:
                material_name = re.sub(r"\s+", "_", row["material_name"].strip())
                if not material_name:
                    raise ValueError(
                        f"{group_id or part_names[0]}: material_name is required when include=y"
                    )
                material = _Material(
                    material_name,
                    _optional_float(
                        row["relative_permittivity"],
                        "relative_permittivity",
                        group_id or part_names[0],
                    ),
                    _optional_float(row["conductivity"], "conductivity", group_id or part_names[0]),
                    _optional_float(
                        row["relative_permeability"],
                        "relative_permeability",
                        group_id or part_names[0],
                    ),
                    _optional_float(
                        row["magnetic_loss"], "magnetic_loss", group_id or part_names[0]
                    ),
                )
                _validate_material(material, group_id or part_names[0])
            for part_name in part_names:
                assignments[part_name] = _Assignment(
                    include,
                    priority,
                    material,
                    geometry_tag=geometry_tag,
                    group_id=group_id,
                    group_confidence=group_confidence,
                )
    return assignments


def _material_key(index: int, name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-") or "material"
    if not slug[0].isalpha():
        slug = f"m_{slug}"
    return f"material_{index:03d}_{slug}"


def _write_material_database(path: Path, materials: Sequence[_Material]) -> list[str]:
    """Write a material database once, preserving later user edits.

    STEP carries geometry and names but not the constitutive data needed by
    gprMax. The assignment CSV can seed constant values, while the generated
    JSON is the authoritative runtime database and may subsequently be edited
    to add dispersion or richer metadata.
    """

    keys = [_material_key(index, material.name) for index, material in enumerate(materials)]
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Existing material database {path} is invalid JSON: {exc.msg}"
            ) from exc
        if existing.get("database", {}).get("id") != path.stem:
            raise ValueError(f"Existing material database {path} has the wrong database ID")
        existing_materials = existing.get("materials")
        if not isinstance(existing_materials, dict) or list(existing_materials) != keys:
            raise ValueError(
                f"Existing material database {path} has material keys that do not match the "
                "STEP assignments; move it aside before converting the changed geometry"
            )
        return keys

    entries = {}
    for key, material in zip(keys, materials):
        # JSON has no portable representation for infinity. gprMax material
        # databases use the explicit string ``"inf"`` for ideal electric or
        # magnetic conductivity, matching material_to_database_entry().
        electric_conductivity = (
            "inf" if material.se is not None and np.isposinf(material.se) else material.se
        )
        magnetic_conductivity = (
            "inf" if material.sm is not None and np.isposinf(material.sm) else material.sm
        )
        entries[key] = {
            "name": material.name,
            "model": "constant",
            "base": {
                "relative_permittivity": material.er,
                "electric_conductivity_s_per_m": electric_conductivity,
                "relative_permeability": material.mr,
                "magnetic_conductivity_s_per_m": magnetic_conductivity,
            },
            "metadata": {
                "original_id": material.name,
                "source": "STEPtoVoxel material assignment CSV",
            },
        }
    write_database(
        path,
        create_database_document(
            path.stem,
            entries,
            name="STEPtoVoxel material assignments",
            description="Material order is fixed by /material_keys in geometry.h5.",
        ),
    )
    return keys


def convert_step(
    step_file: str | Path,
    materials_csv: str | Path,
    output_dir: str | Path,
    config: ConversionConfig | None = None,
    *,
    write_vtk: bool = True,
) -> ConversionResult:
    """Convert a material-assigned STEP assembly into gprMax and VTK files."""
    config = config or ConversionConfig()
    step_file = Path(step_file).expanduser().resolve()
    materials_csv = Path(materials_csv).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = inspect_step(step_file, config)
    from . import step_parser

    assignments = _read_assignments(materials_csv)
    part_names = {part.name for part in parts}
    unknown = set(assignments) - part_names
    missing = part_names - set(assignments)
    if unknown or missing:
        raise ValueError(
            f"Material CSV does not match STEP components; unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )

    selected = [part for part in parts if assignments[part.name].include]
    if not selected:
        raise ValueError("No components have include=y")
    selected_markers = [
        part.name for part in selected if classify_marker_name(part.name) is not None
    ]
    if selected_markers:
        raise ValueError(
            "CAD source/receiver/port markers are non-physical and must use include=n: "
            + ", ".join(selected_markers)
        )
    surfaces = [part.name for part in selected if float(part.cad.get("vol_m3") or 0.0) <= 0.0]
    if surfaces:
        raise ValueError(
            "Open or zero-volume surfaces cannot be solid-voxelised; set include=n for "
            + ", ".join(surfaces)
        )

    material_indices: dict[_Material, int] = {}
    materials: list[_Material] = []
    component_material_ids: list[int] = []
    default_component_tags = unique_normalised_tags([str(part.name) for part in selected])
    component_tags: list[str] = []
    meshes = []
    parser_config = _parser_config(config)
    reference_parts = [
        part
        for part in parts
        if float(part.cad.get("vol_m3") or 0.0) <= 0.0
        or classify_marker_name(part.name) is not None
    ]
    reference_geometry = []
    reference_meshes = {}
    for reference_id, part in enumerate(reference_parts):
        shape = step_parser.shape_for_ops(part, parser_config)
        vertices, triangles = step_parser.tessellate_shape(
            shape,
            parser_config,
        )
        if not vertices:
            vertices = step_parser.topology_vertices(shape)
        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int32)
        reference_meshes[part.uid] = (vertices, triangles)
        if len(vertices):
            reference_geometry.append((reference_id, part.name, vertices, triangles))

    for component_id, part in enumerate(selected):
        assignment = assignments[part.name]
        assert assignment.material is not None
        if assignment.material not in material_indices:
            material_indices[assignment.material] = len(materials)
            materials.append(assignment.material)
        component_material_ids.append(material_indices[assignment.material])
        component_tags.append(assignment.geometry_tag or default_component_tags[component_id])

        vertices, triangles = step_parser.tessellate_shape(
            step_parser.shape_for_ops(part, parser_config),
            parser_config,
        )
        if not triangles:
            raise RuntimeError(f"Tessellation produced no triangles for {part.name}")
        meshes.append(
            TriangleMesh(
                vertices_world=np.asarray(vertices, dtype=np.float64),
                triangles=np.asarray(triangles, dtype=np.int32),
                material_id=component_id,
                priority=assignment.priority,
                name=part.name,
            )
        )

    component_grid, grid = voxelise_material_grid(
        meshes,
        dx=config.voxel_size[0],
        dy=config.voxel_size[1],
        dz=config.voxel_size[2],
        pad=config.pad_cells,
        merge_mode=config.merge_mode,
        supersample=config.supersample,
        sweep_axis=config.sweep_axis,
        preserve_thin_features=config.preserve_thin_features,
    )
    if len(materials) > np.iinfo(np.int16).max + 1:
        raise ValueError("Imported geometry exceeds the int16 material-index schema")
    material_grid = component_grid.copy()
    material_map = np.asarray(component_material_ids, dtype=np.int16)
    occupied = component_grid >= 0
    material_grid[occupied] = material_map[component_grid[occupied]]
    tag_data, tag_names = build_tag_volume(component_grid, component_tags)

    geometry_file = output_dir / "geometry.h5"
    materials_file = output_dir / "materials.json"
    manifest_file = output_dir / "conversion.json"
    markers_file = output_dir / "markers.json"
    vtk_file = output_dir / "geometry.vti" if write_vtk else None
    reference_geometry_cad_file = None
    material_keys = _write_material_database(materials_file, materials)
    write_geometry_hdf5(
        geometry_file,
        material_grid,
        grid.dxyz_world,
        origin=grid.origin_world,
        material_keys=material_keys,
        material_database=materials_file.stem,
        tag_data=tag_data,
        tag_names=tag_names,
    )

    if vtk_file is not None:
        try:
            from .visualisation import write_reference_geometry, write_vti

            write_vti(vtk_file, component_grid, grid, scalar_name="component_id")

            if reference_geometry:
                reference_geometry_cad_file = output_dir / "reference_geometry_cad.vtp"
                write_reference_geometry(reference_geometry_cad_file, reference_geometry)
        except ImportError as exc:
            raise RuntimeError(
                "VTK output requires PyVista. Install it with "
                "'conda install -c conda-forge pyvista', or use write_vtk=False/--no-vtk."
            ) from exc

    marker_records = []
    for part in parts:
        vertices, triangles = reference_meshes.get(part.uid, (None, None))
        record = marker_record(
            part,
            grid.origin_world,
            grid.dxyz_world,
            vertices=vertices,
            triangles=triangles,
        )
        if record is not None:
            marker_records.append(record)

    markers_payload = {
        "source_step": str(step_file),
        "voxel_grid_origin_cad_m": grid.origin_world.tolist(),
        "voxel_size_m": grid.dxyz_world.tolist(),
        "coordinate_translation": "model_position = geometry_import_p1 + local_position_m",
        "markers": marker_records,
    }
    markers_file.write_text(json.dumps(markers_payload, indent=2) + "\n", encoding="utf-8")

    counts = {
        part.name: int(np.count_nonzero(component_grid == index))
        for index, part in enumerate(selected)
    }
    manifest = {
        "source_step": str(step_file),
        "voxel_size_m": list(config.voxel_size),
        "shape_cells": list(component_grid.shape),
        "origin_m": grid.origin_world.tolist(),
        "extent_m": (grid.dxyz_world * grid.nxyz).tolist(),
        "preserve_thin_features": config.preserve_thin_features,
        "supersample": config.supersample,
        "sweep_axis": config.sweep_axis,
        "resolved_sweep_axis": resolve_sweep_axis(grid.nxyz, config.sweep_axis),
        "material_grouping": config.material_grouping,
        "grouping_relative_tolerance": config.grouping_relative_tolerance,
        "markers_file": markers_file.name,
        "markers": marker_records,
        "components": _component_records(parts),
        "included_components": [
            {
                "component_id": index,
                "name": part.name,
                "material_id": component_material_ids[index],
                "material_name": assignments[part.name].material.name,
                "geometry_tag": component_tags[index],
                "priority": assignments[part.name].priority,
                "material_group": assignments[part.name].group_id,
                "material_group_confidence": assignments[part.name].group_confidence,
                "cell_count": counts[part.name],
            }
            for index, part in enumerate(selected)
        ],
        "reference_geometry": [
            {
                "reference_id": reference_id,
                "name": part.name,
                "surface_area_m2": part.cad.get("area_m2"),
                "bbox_xyzxyz_m": part.cad.get("bbox_xyzxyz"),
            }
            for reference_id, part in enumerate(reference_parts)
        ],
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return ConversionResult(
        geometry_file=geometry_file,
        materials_file=materials_file,
        manifest_file=manifest_file,
        markers_file=markers_file,
        vtk_file=vtk_file,
        reference_geometry_cad_file=reference_geometry_cad_file,
        shape=tuple(int(value) for value in component_grid.shape),
        origin=tuple(float(value) for value in grid.origin_world),
        spacing=tuple(float(value) for value in grid.dxyz_world),
        component_cell_counts=counts,
    )
