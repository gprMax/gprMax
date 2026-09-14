# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Non-destructive conversion of legacy geometry/material file pairs."""

import re
import shutil
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from gprMax.geometry_outputs.geometry_objects_read import read_geometry_tag_names
from gprMax.geometry_tags import validate_geometry_tag_ids
from gprMax.material_database import (
    create_database_document,
    make_database_id,
    resolve_database_path,
    validate_material_database,
    write_database,
)

_SPACING_ATTRIBUTE = "dx_dy_dz"
_LEGACY_SPACING_ATTRIBUTE = "dx, dy, dz"


def _number(value):
    number = float(value)
    return "inf" if np.isposinf(number) else number


def _safe_key(index, material_id):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", material_id).strip("_.-") or "material"
    if not slug[0].isalpha():
        slug = f"m_{slug}"
    return f"material_{index:03d}_{slug}"


def parse_legacy_materials(path):
    """Translate only the material commands accepted by legacy geometry files."""

    entries = []
    by_id = {}
    commands = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("##"):
            continue
        if not line.startswith("#") or ":" not in line:
            raise ValueError(f"Unsupported content at {path}:{line_number}: {raw!r}")
        command, parameters = line.split(":", 1)
        tokens = parameters.split()
        commands.append((line_number, command, tokens))

        if command == "#material":
            if len(tokens) != 5:
                raise ValueError(f"Invalid #material at {path}:{line_number}")
            material_id = tokens[4]
            if material_id in by_id:
                raise ValueError(f"Duplicate material ID {material_id!r} at {path}:{line_number}")
            entry = {
                "name": material_id,
                "model": "constant",
                "base": {
                    "relative_permittivity": float(tokens[0]),
                    "electric_conductivity_s_per_m": _number(tokens[1]),
                    "relative_permeability": float(tokens[2]),
                    "magnetic_conductivity_s_per_m": _number(tokens[3]),
                },
                "metadata": {"original_id": material_id, "legacy_source_line": line_number},
            }
            by_id[material_id] = entry
            entries.append((_safe_key(len(entries), material_id), entry))
        elif command not in (
            "#add_dispersion_debye",
            "#add_dispersion_lorentz",
            "#add_dispersion_drude",
            "#material_density",
        ):
            raise ValueError(
                f"Unsupported command {command!r} at {path}:{line_number}; only material, "
                "dispersion and material_density commands are permitted"
            )

    for line_number, command, tokens in commands:
        if command == "#material":
            continue
        if command == "#material_density":
            if len(tokens) < 2:
                raise ValueError(f"Invalid #material_density at {path}:{line_number}")
            density = float(tokens[0])
            if not np.isfinite(density) or density <= 0:
                raise ValueError(f"Density must be finite and positive at {path}:{line_number}")
            for material_id in tokens[1:]:
                if material_id not in by_id:
                    raise ValueError(
                        f"Density at {path}:{line_number} references unknown material {material_id!r}"
                    )
                by_id[material_id]["mass_density_kg_per_m3"] = density
            continue
        try:
            count = int(tokens[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid pole count at {path}:{line_number}") from exc
        width = 2 if command != "#add_dispersion_lorentz" else 3
        split = 1 + width * count
        if count <= 0 or len(tokens) <= split:
            raise ValueError(f"Invalid dispersion command at {path}:{line_number}")
        values = [float(value) for value in tokens[1:split]]
        material_ids = tokens[split:]
        for material_id in material_ids:
            if material_id not in by_id:
                raise ValueError(
                    f"Dispersion at {path}:{line_number} references unknown material {material_id!r}"
                )
            entry = by_id[material_id]
            if entry["model"] != "constant":
                raise ValueError(f"Material {material_id!r} has multiple dispersion definitions")
            poles = []
            for index in range(count):
                pole = values[index * width : (index + 1) * width]
                if command == "#add_dispersion_debye":
                    poles.append(
                        {
                            "relative_permittivity_difference": pole[0],
                            "relaxation_time_s": pole[1],
                        }
                    )
                    entry["model"] = "debye"
                elif command == "#add_dispersion_lorentz":
                    poles.append(
                        {
                            "relative_permittivity_difference": pole[0],
                            "resonance_frequency_hz": pole[1],
                            "damping_coefficient_per_s": pole[2],
                        }
                    )
                    entry["model"] = "lorentz"
                else:
                    poles.append(
                        {
                            "plasma_frequency_hz": pole[0],
                            "collision_frequency_per_s": pole[1],
                        }
                    )
                    entry["model"] = "drude"
            entry["poles"] = poles
    return entries


def _blocks(dataset, max_bytes=8 * 1024 * 1024):
    """Read regular blocks bounded in size, including contiguous anatomy files."""
    shape = dataset.shape
    budget = max(1, max_bytes // max(1, dataset.dtype.itemsize))
    block_shape = [1] * len(shape)
    for axis in reversed(range(len(shape))):
        block_shape[axis] = max(1, min(shape[axis], budget))
        budget = max(1, budget // block_shape[axis])
    for start in product(*(range(0, n, step) for n, step in zip(shape, block_shape))):
        selection = tuple(
            slice(i, min(i + step, n)) for i, step, n in zip(start, block_shape, shape)
        )
        yield dataset[selection]


def _required_material_count(geometry):
    maximum = -1
    for dataset_name in ("/data", "/ID"):
        if dataset_name in geometry:
            dataset = geometry[dataset_name]
            if dataset.dtype.kind not in "iu":
                raise ValueError(f"Geometry {dataset_name} material indices must be integers")
            for values in _blocks(dataset):
                if values.size:
                    if int(values.min()) < -1:
                        raise ValueError(
                            f"Geometry {dataset_name} indices must be -1 or non-negative"
                        )
                    maximum = max(maximum, int(values.max()))
    return maximum + 1


def _geometry_spacing(geometry, filename):
    """Validate spacing without modifying the source file."""

    current = geometry.attrs.get(_SPACING_ATTRIBUTE)
    legacy = geometry.attrs.get(_LEGACY_SPACING_ATTRIBUTE)
    if current is None and legacy is None:
        raise ValueError(
            f"Geometry file {filename} has neither {_SPACING_ATTRIBUTE!r} nor legacy "
            f"{_LEGACY_SPACING_ATTRIBUTE!r} spacing metadata"
        )
    for spacing in (current, legacy):
        if spacing is not None:
            values = np.asarray(spacing, dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(
                    f"Geometry spacing in {filename} must contain three positive finite values"
                )
    if current is not None and legacy is not None:
        if not np.array_equal(current, legacy):
            raise ValueError(
                f"Geometry file {filename} contains inconsistent current and legacy spacings"
            )
    return current if current is not None else legacy


def _require_self_contained(group, seen=None):
    """A file copy must not leave relative external data dependencies behind."""
    if seen is None:
        seen = set()
    if group.id in seen:
        return
    seen.add(group.id)
    for name in group:
        link = group.get(name, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            raise ValueError("Conversion requires self-contained HDF5 files, not external links")
        if isinstance(link, h5py.SoftLink):
            continue  # Internal paths remain valid in the file copy.
        obj = group[name]
        if isinstance(obj, h5py.Group):
            _require_self_contained(obj, seen)
        elif isinstance(obj, h5py.Dataset) and (obj.is_virtual or obj.external):
            raise ValueError(
                "Conversion requires self-contained HDF5 datasets, not virtual or external storage"
            )


def _validate_geometry(geometry, filename, material_count):
    """Validate the arrays the simulator will use; never rerasterise geometry."""
    if "material_keys" in geometry:
        raise ValueError("Source geometry is already a material-database geometry file")
    _require_self_contained(geometry)
    if "data" not in geometry or not isinstance(geometry["data"], h5py.Dataset):
        raise ValueError(f"Geometry file {filename} has no /data dataset")
    data = geometry["data"]
    if data.ndim != 3 or any(n == 0 for n in data.shape):
        raise ValueError("Geometry /data must be a non-empty three-dimensional cell array")
    component_names = ("ID", "rigidE", "rigidH")
    present = [name in geometry for name in component_names]
    if any(present) and not all(present):
        raise ValueError("Component geometry requires all of /ID, /rigidE and /rigidH")
    if all(present):
        expected = {
            "ID": (6, *(n + 1 for n in data.shape)),
            "rigidE": (12, *data.shape),
            "rigidH": (6, *data.shape),
        }
        for name, shape in expected.items():
            dataset = geometry[name]
            if not isinstance(dataset, h5py.Dataset) or dataset.shape != shape:
                raise ValueError(f"Geometry /{name} must have shape {shape}")
            if dataset.dtype.kind not in "iu":
                raise ValueError(f"Geometry /{name} must contain integers")
    required = _required_material_count(geometry)
    if required > material_count:
        raise ValueError(
            f"Geometry references {required} material indices but the text file only declares "
            f"{material_count} material(s); include the complete original material table in file order"
        )
    names = read_geometry_tag_names(geometry)
    if names:
        for values in _blocks(geometry["tag_data"]):
            validate_geometry_tag_ids(values, len(names))


def convert_geometry(
    geometry,
    materials,
    *,
    output_geometry=None,
    output_database=None,
):
    """Convert an HDF5/text pair without changing source arrays or material values.

    Outputs must be new, adjacent files: GeometryObjectsRead resolves a database
    beside its HDF5 partner. Validate first, reserve paths exclusively, and remove
    files created by this call if it fails. File contents are copied, not rebuilt
    with current geometry/averaging algorithms. Validation uses bounded blocks.
    """

    geometry = Path(geometry)
    materials = Path(materials)
    if output_geometry is None:
        output_geometry = geometry.with_name(f"{geometry.stem}_converted{geometry.suffix}")
    else:
        output_geometry = Path(output_geometry)
    if output_database is None:
        database_id = make_database_id(f"{geometry.stem}_materials", prefix="geometry")
        output_database = output_geometry.with_name(f"{database_id}.json")
    else:
        output_database = Path(output_database)
        database_id = make_database_id(output_database.stem, prefix="geometry")
        if database_id != output_database.stem:
            raise ValueError(
                "The output database filename must already be a valid database name; "
                f"use '{database_id}.json'"
            )
    if output_database.suffix != ".json":
        raise ValueError("The output material database must have the .json extension")
    if output_geometry.parent.resolve() != output_database.parent.resolve():
        raise ValueError("Output geometry and JSON material database must be in the same directory")
    outputs = (output_geometry, output_database)
    if outputs[0].resolve() == outputs[1].resolve():
        raise ValueError("Output geometry and database must be different files")
    sources = {geometry.resolve(), materials.resolve()}
    for output in outputs:
        if output.resolve() in sources:
            raise ValueError("Refusing to overwrite a source file; choose another output path")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Conversion output {output} already exists; choose another path")

    entries = parse_legacy_materials(materials)
    document = create_database_document(
        database_id,
        dict(entries),
        name=f"Converted materials for {geometry.name}",
        description=f"Converted non-destructively from legacy material file {materials.name}.",
    )
    with h5py.File(geometry, "r") as source:
        spacing = _geometry_spacing(source, geometry)
        _validate_geometry(source, geometry, len(entries))

    # Validate the generated JSON through the same schema reader as the solver.
    # Official names are reserved: an output with such a name would load the
    # installed catalogue instead of this converted material table.
    with TemporaryDirectory(prefix="gprmax-material-conversion-") as directory:
        directory = Path(directory)
        staged_database = directory / output_database.name
        write_database(staged_database, document)
        _, official = resolve_database_path(database_id, search_directory=directory)
        if official:
            raise ValueError(
                f"Output database name {database_id!r} is reserved; choose another name"
            )
        validate_material_database(database_id, search_directory=directory)

    created = []
    try:
        for output in outputs:
            with output.open("xb"):
                created.append(output)
        shutil.copyfile(geometry, output_geometry)
        with h5py.File(output_geometry, "r+") as converted:
            if _SPACING_ATTRIBUTE not in converted.attrs:
                converted.attrs[_SPACING_ATTRIBUTE] = spacing
            converted.create_dataset(
                "material_keys", data=np.asarray([key for key, _ in entries], dtype="S")
            )
            converted.attrs["MaterialDatabase"] = database_id
            converted.attrs["MaterialDatabaseSchemaVersion"] = 1
            converted.attrs["LegacyMaterialsSource"] = str(materials)
        write_database(output_database, document)
    except BaseException:
        for output in reversed(created):
            output.unlink(missing_ok=True)
        raise
    return output_geometry, output_database
