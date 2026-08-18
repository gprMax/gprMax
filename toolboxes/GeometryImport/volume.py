# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Import labelled medical-image volumes as tagged gprMax geometries."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gprMax.material_database import make_database_id

from .common import (
    build_tag_volume,
    normalise_tag_name,
    unique_normalised_tags,
    write_geometry_hdf5,
    write_geometry_preview,
    write_null_material_database,
)

LABEL_COLUMNS = ("label", "name", "include", "material_name", "geometry_tag")
_UNIT_FACTORS = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimetre": 1e-3,
    "um": 1e-6,
    "micron": 1e-6,
    "micrometer": 1e-6,
    "micrometre": 1e-6,
}


@dataclass(frozen=True)
class LabelVolume:
    labels: np.ndarray
    spacing_m: tuple[float, float, float]
    # NIfTI, NRRD, and MetaImage define the physical location of the first
    # image sample (voxel centre), not the lower face of the voxel volume.
    first_cell_centre_m: tuple[float, float, float]
    label_names: Mapping[int, str]
    source_format: str


@dataclass(frozen=True)
class LabelAssignment:
    label: int
    name: str
    include: bool
    material_name: str | None
    geometry_tag: str | None


@dataclass(frozen=True)
class VolumeConversionResult:
    geometry_file: Path
    preview_file: Path | None
    materials_file: Path
    manifest_file: Path
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    labels: tuple[int, ...]


def _unit_factor(requested: str, encoded: str | None) -> float:
    unit = encoded if requested == "auto" else requested
    if unit is None:
        raise ValueError("The source format does not define physical units; specify m, mm, or um")
    clean = str(unit).strip().strip('"').lower().replace("µ", "u")
    try:
        return _UNIT_FACTORS[clean]
    except KeyError as exc:
        raise ValueError(f"Unsupported or unknown spatial unit: {unit!r}") from exc


def _integer_labels(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim != 3:
        raise ValueError("A labelled anatomy volume must have exactly three spatial dimensions")
    if data.dtype.kind == "c":
        raise ValueError("Label volumes cannot contain complex values")
    if data.dtype.kind == "f":
        if not np.isfinite(data).all() or not np.allclose(data, np.rint(data), atol=1e-6):
            raise ValueError("Label volumes must contain finite integer-valued labels")
        data = np.rint(data)
    if data.dtype.kind not in "iuf":
        raise ValueError("Label volumes must contain numeric integer-valued labels")
    minimum = int(data.min()) if data.size else 0
    maximum = int(data.max()) if data.size else 0
    candidates = (
        (np.uint8, np.uint16, np.uint32, np.uint64)
        if minimum >= 0
        else (np.int8, np.int16, np.int32, np.int64)
    )
    for dtype in candidates:
        limits = np.iinfo(dtype)
        if minimum >= limits.min and maximum <= limits.max:
            return np.asarray(data, dtype=dtype, order="C")
    raise ValueError("Label values exceed supported 64-bit integer storage")


def _canonicalise_axis_aligned(
    data: np.ndarray,
    axis_vectors: np.ndarray,
    origin: Sequence[float],
    *,
    unit_factor: float,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """Permute/flip an axis-aligned physical volume into x-y-z order."""

    vectors = np.asarray(axis_vectors, dtype=float)
    if vectors.shape != (3, 3):
        raise ValueError("The image must define three spatial axis vectors")
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Image spacing must be positive")
    directions = vectors / lengths[:, None]
    physical_for_source = np.argmax(np.abs(directions), axis=1)
    if len(set(int(value) for value in physical_for_source)) != 3:
        raise ValueError("Image axes do not map uniquely to physical x, y, and z")
    dominant = directions[np.arange(3), physical_for_source]
    residual = directions.copy()
    residual[np.arange(3), physical_for_source] = 0
    if np.max(np.abs(residual)) > tolerance or np.max(np.abs(np.abs(dominant) - 1)) > tolerance:
        raise ValueError("Oblique or sheared label volumes must be resampled to an axis-aligned grid first")

    source_for_physical = tuple(int(np.where(physical_for_source == axis)[0][0]) for axis in range(3))
    canonical = np.transpose(data, source_for_physical)
    shifted_origin = np.asarray(origin, dtype=float).copy()
    spacing = []
    for physical_axis, source_axis in enumerate(source_for_physical):
        spacing.append(float(lengths[source_axis]) * unit_factor)
        if dominant[source_axis] < 0:
            shifted_origin += vectors[source_axis] * (data.shape[source_axis] - 1)
            canonical = np.flip(canonical, axis=physical_axis)
    return (
        np.asarray(canonical, order="C"),
        tuple(spacing),
        tuple(float(value) * unit_factor for value in shifted_origin),
    )


def _load_nifti(path: Path, unit: str) -> LabelVolume:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("NIfTI import requires nibabel") from exc
    image = nib.load(str(path))
    data = _integer_labels(np.asanyarray(image.dataobj))
    affine = np.asarray(image.affine, dtype=float)
    encoded_unit = image.header.get_xyzt_units()[0] or None
    factor = _unit_factor(unit, encoded_unit)
    canonical, spacing, origin = _canonicalise_axis_aligned(
        data,
        affine[:3, :3].T,
        affine[:3, 3],
        unit_factor=factor,
    )
    return LabelVolume(canonical, spacing, origin, {}, "nifti")


def _nrrd_segment_names(header: Mapping) -> dict[int, str]:
    names = {}
    for key, value in header.items():
        text = str(key)
        if not text.startswith("Segment") or not text.endswith("_Name"):
            continue
        prefix = text[: -len("_Name")]
        label_key = f"{prefix}_LabelValue"
        if label_key in header:
            names[int(header[label_key])] = str(value)
    return names


def _load_nrrd(path: Path, unit: str) -> LabelVolume:
    try:
        import nrrd
    except ImportError as exc:
        raise RuntimeError("NRRD import requires pynrrd") from exc
    # Fortran index order preserves the NRRD dimension order described by
    # ``space directions``. C order reverses the array axes on read.
    values, header = nrrd.read(str(path), index_order="F")
    data = _integer_labels(values)
    raw_vectors = header.get("space directions")
    if raw_vectors is None:
        raise ValueError("NRRD label volume has no 'space directions' metadata")
    vectors = np.asarray(raw_vectors, dtype=float)
    raw_units = header.get("space units")
    encoded_unit = None
    if raw_units:
        clean_units = {str(value).strip().strip('"') for value in raw_units}
        if len(clean_units) != 1:
            raise ValueError("NRRD spatial axes use inconsistent units")
        encoded_unit = clean_units.pop()
    factor = _unit_factor(unit, encoded_unit)
    origin_native = np.asarray(header.get("space origin", (0.0, 0.0, 0.0)), dtype=float)
    canonical, spacing, origin = _canonicalise_axis_aligned(
        data,
        vectors,
        origin_native,
        unit_factor=factor,
    )
    return LabelVolume(canonical, spacing, origin, _nrrd_segment_names(header), "nrrd")


def _load_metaimage(path: Path, unit: str) -> LabelVolume:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("MetaImage import requires SimpleITK") from exc
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError("A MetaImage anatomy label map must be three-dimensional")
    # SimpleITK exposes arrays in z-y-x order but geometry in x-y-z order.
    data = _integer_labels(np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0)))
    spacing_native = np.asarray(image.GetSpacing(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    vectors = (direction * spacing_native[None, :]).T
    factor = _unit_factor(unit, None)
    canonical, spacing, origin = _canonicalise_axis_aligned(
        data,
        vectors,
        image.GetOrigin(),
        unit_factor=factor,
    )
    return LabelVolume(canonical, spacing, origin, {}, "metaimage")


def load_label_volume(path: str | Path, *, unit: str = "auto") -> LabelVolume:
    """Load an axis-aligned NIfTI, NRRD, or MetaImage integer label map."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    lower = source.name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return _load_nifti(source, unit)
    if lower.endswith((".nrrd", ".nhdr", ".seg.nrrd")):
        return _load_nrrd(source, unit)
    if lower.endswith((".mha", ".mhd")):
        return _load_metaimage(source, unit)
    raise ValueError("Supported label volumes are NIfTI, NRRD, and MetaImage")


def _yes(value: str) -> bool:
    clean = value.strip().lower()
    if clean in {"y", "yes", "true", "1"}:
        return True
    if clean in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"include must be y/n, got {value!r}")


def write_label_template(volume: LabelVolume, path: str | Path, *, overwrite=False) -> Path:
    """Write an editable label-to-material/tag assignment CSV."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    labels = tuple(int(value) for value in np.unique(volume.labels))
    source_names = [volume.label_names.get(value, f"label_{value}") for value in labels]
    tags = unique_normalised_tags(source_names)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for label, name, tag in zip(labels, source_names, tags):
            writer.writerow(
                {
                    "label": label,
                    "name": name,
                    "include": "n" if label == 0 else "y",
                    "material_name": name,
                    "geometry_tag": tag,
                }
            )
    return target


def read_label_assignments(path: str | Path) -> tuple[LabelAssignment, ...]:
    assignments = []
    seen = set()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(LABEL_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            label = int(row["label"])
            if label in seen:
                raise ValueError(f"Label {label} occurs more than once")
            seen.add(label)
            include = _yes(row["include"])
            name = row["name"].strip() or f"label_{label}"
            material = row["material_name"].strip() or None
            if include and material is None:
                raise ValueError(f"material_name is required for included label {label}")
            raw_tag = row["geometry_tag"].strip()
            tag = normalise_tag_name(raw_tag or name) if include else None
            assignments.append(LabelAssignment(label, name, include, material, tag))
    return tuple(assignments)


def convert_label_volume(
    source: str | Path,
    assignments_csv: str | Path,
    output_dir: str | Path,
    *,
    unit: str = "auto",
) -> VolumeConversionResult:
    """Convert a labelled image volume to reusable tagged gprMax geometry."""

    source = Path(source).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    volume = load_label_volume(source, unit=unit)
    assignments = read_label_assignments(assignments_csv)
    actual = {int(value) for value in np.unique(volume.labels)}
    declared = {assignment.label for assignment in assignments}
    if actual != declared:
        raise ValueError(
            f"Assignment labels do not match source; missing={sorted(actual-declared)}, "
            f"unknown={sorted(declared-actual)}"
        )
    selected = [assignment for assignment in assignments if assignment.include]
    if not selected:
        raise ValueError("No labels have include=y")

    material_names = []
    material_ids = {}
    component_dtype = np.int16 if len(selected) <= np.iinfo(np.int16).max + 1 else np.int32
    component_grid = np.full(volume.labels.shape, -1, dtype=component_dtype)
    component_material_ids = []
    component_tags = []
    for component_id, assignment in enumerate(selected):
        assert assignment.material_name is not None
        material_id = material_ids.get(assignment.material_name)
        if material_id is None:
            material_id = len(material_names)
            material_ids[assignment.material_name] = material_id
            material_names.append(assignment.material_name)
        component_material_ids.append(material_id)
        component_tags.append(assignment.geometry_tag)
        component_grid[volume.labels == assignment.label] = component_id

    if len(material_names) > np.iinfo(np.int16).max + 1:
        raise ValueError("Imported geometry exceeds the int16 material-index schema")
    material_grid = np.full(component_grid.shape, -1, dtype=np.int16)
    occupied = component_grid >= 0
    lookup = np.asarray(component_material_ids, dtype=np.int16)
    material_grid[occupied] = lookup[component_grid[occupied]]
    tag_data, tag_names = build_tag_volume(component_grid, component_tags)

    geometry_file = output / "geometry.h5"
    preview_file = output / "geometry_preview.vti"
    materials_file = output / "materials.json"
    manifest_file = output / "conversion.json"
    database_id = make_database_id(materials_file.stem, prefix="geometry")
    if database_id != materials_file.stem:
        materials_file = output / f"{database_id}.json"
    material_metadata = []
    for name in material_names:
        labels_for_material = [assignment.label for assignment in selected if assignment.material_name == name]
        material_metadata.append({"source_labels": labels_for_material})
    material_keys = write_null_material_database(
        materials_file,
        database_id,
        material_names,
        source=f"GeometryImport {volume.source_format} label map",
        metadata=material_metadata,
    )
    grid_origin_m = tuple(
        centre - 0.5 * spacing for centre, spacing in zip(volume.first_cell_centre_m, volume.spacing_m)
    )
    write_geometry_hdf5(
        geometry_file,
        material_grid,
        volume.spacing_m,
        origin=grid_origin_m,
        material_keys=material_keys,
        material_database=database_id,
        tag_data=tag_data,
        tag_names=tag_names,
    )
    preview_written = write_geometry_preview(
        preview_file,
        material_grid,
        volume.spacing_m,
        origin=grid_origin_m,
        material_keys=material_keys,
        tag_data=tag_data,
        tag_names=tag_names,
    )
    if not preview_written:
        preview_file = None
    manifest = {
        "source": str(source),
        "source_format": volume.source_format,
        "shape_cells": list(volume.labels.shape),
        "spacing_m": list(volume.spacing_m),
        "first_cell_centre_m": list(volume.first_cell_centre_m),
        "grid_origin_m": list(grid_origin_m),
        "regions": [
            {
                "source_label": assignment.label,
                "source_name": assignment.name,
                "included": assignment.include,
                "material_name": assignment.material_name,
                "geometry_tag": assignment.geometry_tag,
                "cell_count": int(np.count_nonzero(volume.labels == assignment.label)),
            }
            for assignment in assignments
        ],
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return VolumeConversionResult(
        geometry_file,
        preview_file,
        materials_file,
        manifest_file,
        tuple(int(value) for value in volume.labels.shape),
        volume.spacing_m,
        tuple(sorted(actual)),
    )
