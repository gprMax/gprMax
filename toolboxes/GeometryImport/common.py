# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Common schema helpers used by geometry conversion toolboxes.

The functions in this module deliberately know nothing about the source
geometry format.  Importers first produce a final cell-centred material grid
and, when semantic component information exists, a final cell-centred tag
grid.  This keeps material identity and geometry identity independent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from gprMax.geometry_tags import UNTAGGED_NAME, validate_geometry_tag
from gprMax.material_database import create_database_document, write_database


def normalise_tag_name(name: str, *, fallback: str = "region") -> str:
    """Return a valid geometry tag derived from an external object name.

    STEP, STL, VTK, and medical-image labels may contain whitespace or
    punctuation which is not valid in a gprMax geometry tag.  Conversion is
    deterministic and keeps the original source name separately in importer
    manifests.
    """

    value = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(name).strip()).strip("_.:-")
    if not value:
        value = fallback
    if value == UNTAGGED_NAME:
        value = f"{value}_region"
    validate_geometry_tag(value)
    return value


def unique_normalised_tags(names: Sequence[str]) -> tuple[str, ...]:
    """Normalise external names without accidentally merging distinct parts."""

    used: set[str] = set()
    result = []
    for index, name in enumerate(names, start=1):
        base = normalise_tag_name(name, fallback=f"region_{index}")
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return tuple(result)


def material_key(index: int, name: str) -> str:
    """Build a stable schema-safe material key for an imported region."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_.-") or "material"
    if not slug[0].isalpha():
        slug = f"m_{slug}"
    return f"material_{index:03d}_{slug}"


def write_null_material_database(
    path: str | PathLike[str],
    database_id: str,
    material_names: Sequence[str],
    *,
    source: str,
    metadata: Sequence[dict] | None = None,
) -> tuple[str, ...]:
    """Create, but never overwrite, an editable imported-material database."""

    target = Path(path)
    names = tuple(str(name) for name in material_names)
    keys = tuple(material_key(index, name) for index, name in enumerate(names))
    if metadata is None:
        metadata = tuple({} for _ in names)
    if len(metadata) != len(names):
        raise ValueError("metadata must have one entry per material")

    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing material database {target} is invalid JSON") from exc
        recorded = existing.get("materials")
        if existing.get("database", {}).get("id") != database_id or not isinstance(recorded, dict):
            raise ValueError(f"Existing material database {target} is not compatible")
        if tuple(recorded) != keys:
            raise ValueError(f"Existing material database {target} has keys which do not match this import")
        return keys

    entries = {}
    for key, name, extra in zip(keys, names, metadata):
        item_metadata = {"original_id": name, "source": source}
        item_metadata.update(dict(extra))
        entries[key] = {
            "name": name,
            "model": "constant",
            "base": {
                "relative_permittivity": None,
                "electric_conductivity_s_per_m": None,
                "relative_permeability": None,
                "magnetic_conductivity_s_per_m": None,
            },
            "metadata": item_metadata,
        }
    write_database(
        target,
        create_database_document(
            database_id,
            entries,
            name="Imported geometry material assignments",
            description="Replace null constitutive values before importing this geometry.",
        ),
    )
    return keys


def build_tag_volume(
    component_grid: np.ndarray,
    component_tags: Sequence[str | None],
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Map final component IDs to compact geometry-tag IDs.

    Negative component values mean that an importer does not write that cell.
    ``None`` maps an occupied component to tag ID zero.  When no non-zero tag
    exists, ``None`` is returned for the volume so untagged models retain zero
    tag-storage overhead.
    """

    components = np.asarray(component_grid)
    if components.ndim != 3:
        raise ValueError("component_grid must be a 3D cell-centred array")
    occupied = components >= 0
    if occupied.any() and int(components[occupied].max()) >= len(component_tags):
        raise ValueError("component_grid references a component absent from component_tags")

    names = [UNTAGGED_NAME]
    ids = {UNTAGGED_NAME: 0}
    lookup = []
    for tag in component_tags:
        if tag is None:
            lookup.append(0)
            continue
        validate_geometry_tag(tag)
        tag_id = ids.get(tag)
        if tag_id is None:
            tag_id = len(names)
            ids[tag] = tag_id
            names.append(tag)
        lookup.append(tag_id)

    if len(names) == 1:
        return None, tuple(names)
    max_id = len(names) - 1
    if max_id <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif max_id <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    else:
        dtype = np.uint32
    tag_data = np.zeros(components.shape, dtype=dtype)
    if occupied.any():
        mapping = np.asarray(lookup, dtype=dtype)
        tag_data[occupied] = mapping[components[occupied]]
    return tag_data, tuple(names)


def _validate_tags(
    tag_data: np.ndarray | None,
    tag_names: Sequence[str] | None,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray | None, tuple[str, ...] | None]:
    if tag_data is None:
        supplied_names = None if tag_names is None else tuple(tag_names)
        if supplied_names not in (None, (), (UNTAGGED_NAME,)):
            raise ValueError("tag_names were supplied without tag_data")
        return None, None
    tags = np.asarray(tag_data)
    if tags.shape != shape:
        raise ValueError("tag_data shape must match material data")
    if tags.dtype.kind != "u":
        raise ValueError("tag_data must use an unsigned integer dtype")
    if not tag_names or tag_names[0] != UNTAGGED_NAME:
        raise ValueError("tag_names[0] must be 'untagged'")
    names = tuple(str(name) for name in tag_names)
    if len(set(names)) != len(names):
        raise ValueError("tag_names must be unique")
    for name in names[1:]:
        validate_geometry_tag(name)
    if tags.size and int(tags.max()) >= len(names):
        raise ValueError("tag_data references an ID absent from tag_names")
    return tags, names


def write_tag_datasets(
    h5: h5py.File,
    tag_data: np.ndarray | None,
    tag_names: Sequence[str] | None,
    *,
    shape: tuple[int, int, int],
    compression: str | None = "gzip",
) -> None:
    """Write the established gprMax geometry-tag schema to an open file."""

    tags, names = _validate_tags(tag_data, tag_names, shape)
    if tags is None:
        return
    h5.create_dataset("tag_data", data=tags, compression=compression)
    h5.create_dataset("tag_names", data=np.asarray(names, dtype="S"))
    h5.attrs["GeometryTagsSchemaVersion"] = 1


def write_geometry_hdf5(
    path: str | PathLike[str],
    data: np.ndarray,
    spacing: Sequence[float],
    *,
    origin: Sequence[float] | None = None,
    material_keys: Sequence[str] | None = None,
    material_database: str | None = None,
    tag_data: np.ndarray | None = None,
    tag_names: Sequence[str] | None = None,
    compression: str | None = "gzip",
) -> None:
    """Write a material/tag cell volume in the gprMax geometry-object schema."""

    source_materials = np.asarray(data)
    if source_materials.ndim != 3:
        raise ValueError("data must be a 3D cell-centred array")
    if source_materials.dtype.kind not in "iu":
        raise ValueError("data must contain integer material indices")
    if source_materials.size and (source_materials.min() < -1 or source_materials.max() > np.iinfo(np.int16).max):
        raise ValueError("material indices must be -1 or fit in signed int16 storage")
    materials = np.asarray(source_materials, dtype=np.int16, order="C")
    dxyz = tuple(float(value) for value in spacing)
    if len(dxyz) != 3 or not np.isfinite(dxyz).all() or any(value <= 0 for value in dxyz):
        raise ValueError("spacing must contain three positive values in metres")
    if material_keys is not None:
        keys = tuple(str(key) for key in material_keys)
        written = materials >= 0
        if written.any() and int(materials[written].max()) >= len(keys):
            raise ValueError("material data references an index absent from material_keys")
    else:
        keys = None

    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=materials, compression=compression)
        h5.attrs["dx_dy_dz"] = dxyz
        h5.attrs["shape_nxyz"] = materials.shape
        if origin is not None:
            xyz = tuple(float(value) for value in origin)
            if len(xyz) != 3 or not np.isfinite(xyz).all():
                raise ValueError("origin must contain three finite coordinates")
            h5.attrs["origin_xyz"] = xyz
        if keys is not None:
            h5.create_dataset("material_keys", data=np.asarray(keys, dtype="S"))
        if material_database is not None:
            h5.attrs["MaterialDatabase"] = str(material_database)
            h5.attrs["MaterialDatabaseSchemaVersion"] = 1
        write_tag_datasets(
            h5,
            tag_data,
            tag_names,
            shape=materials.shape,
            compression=compression,
        )


def write_geometry_preview(
    path: str | PathLike[str],
    data: np.ndarray,
    spacing: Sequence[float],
    *,
    origin: Sequence[float],
    material_keys: Sequence[str] | None = None,
    tag_data: np.ndarray | None = None,
    tag_names: Sequence[str] | None = None,
) -> bool:
    """Write a cell-centred VTK ImageData preview for ParaView.

    This is a visualisation companion to ``geometry.h5`` rather than an input
    to :class:`GeometryObjectsRead`. ``False`` is returned when the optional
    PyVista/VTK visualisation dependency is unavailable; conversion of the
    reusable HDF5 geometry remains independent of that dependency.
    """

    try:
        import pyvista as pv
    except ImportError:
        return False

    materials = np.asarray(data)
    if materials.ndim != 3:
        raise ValueError("data must be a 3D cell-centred array")
    dxyz = tuple(float(value) for value in spacing)
    xyz = tuple(float(value) for value in origin)
    if len(dxyz) != 3 or not np.isfinite(dxyz).all() or any(value <= 0 for value in dxyz):
        raise ValueError("spacing must contain three positive values in metres")
    if len(xyz) != 3 or not np.isfinite(xyz).all():
        raise ValueError("origin must contain three finite coordinates")
    tags, names = _validate_tags(tag_data, tag_names, materials.shape)

    image = pv.ImageData(
        dimensions=tuple(int(value) + 1 for value in materials.shape),
        spacing=dxyz,
        origin=xyz,
    )
    image.cell_data["MaterialIndex"] = materials.ravel(order="F")
    if tags is not None:
        image.cell_data["TagID"] = tags.ravel(order="F")
    if material_keys is not None:
        image.field_data["MaterialKeys"] = np.asarray(tuple(material_keys), dtype=str)
    if names is not None:
        image.field_data["TagNames"] = np.asarray(names, dtype=str)
    image.save(Path(path))
    return True
