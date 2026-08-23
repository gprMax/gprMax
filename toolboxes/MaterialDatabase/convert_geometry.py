# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Non-destructive conversion of legacy geometry/material file pairs."""

import re
import shutil
from pathlib import Path

import h5py
import numpy as np

from gprMax.material_database import create_database_document, make_database_id, write_database

_SPACING_ATTRIBUTE = "dx_dy_dz"
_LEGACY_SPACING_ATTRIBUTE = "dx, dy, dz"


def _number(value):
    return "inf" if value == "inf" else float(value)


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
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
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
        ):
            raise ValueError(
                f"Unsupported command {command!r} at {path}:{line_number}; only material and "
                "dispersion commands are permitted"
            )

    for line_number, command, tokens in commands:
        if command == "#material":
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


def _required_material_count(geometry):
    maximum = -1
    for dataset_name in ("/data", "/ID"):
        if dataset_name in geometry:
            values = geometry[dataset_name][:]
            nonnegative = values[values >= 0]
            if nonnegative.size:
                maximum = max(maximum, int(nonnegative.max()))
    return maximum + 1


def _normalise_geometry_spacing(geometry, filename):
    """Add the current spacing attribute to legacy geometry files."""

    current = geometry.attrs.get(_SPACING_ATTRIBUTE)
    legacy = geometry.attrs.get(_LEGACY_SPACING_ATTRIBUTE)
    if current is None and legacy is None:
        raise ValueError(
            f"Geometry file {filename} has neither {_SPACING_ATTRIBUTE!r} nor legacy "
            f"{_LEGACY_SPACING_ATTRIBUTE!r} spacing metadata"
        )
    if current is not None and legacy is not None:
        if len(current) != len(legacy) or not np.allclose(current, legacy, rtol=0, atol=0):
            raise ValueError(
                f"Geometry file {filename} contains inconsistent current and legacy spacings"
            )
    if current is None:
        geometry.attrs[_SPACING_ATTRIBUTE] = legacy


def convert_geometry(
    geometry,
    materials,
    *,
    output_geometry=None,
    output_database=None,
):
    """Copy a geometry file and attach stable keys for a converted JSON database."""

    geometry = Path(geometry)
    materials = Path(materials)
    if output_geometry is None:
        output_geometry = geometry.with_name(f"{geometry.stem}_converted{geometry.suffix}")
    else:
        output_geometry = Path(output_geometry)
    if output_database is None:
        database_id = make_database_id(f"{geometry.stem}_materials", prefix="geometry")
        output_database = geometry.with_name(f"{database_id}.json")
    else:
        output_database = Path(output_database)
        database_id = make_database_id(output_database.stem, prefix="geometry")
        if database_id != output_database.stem:
            raise ValueError(
                "The output database filename must already be a valid database name; "
                f"use '{database_id}.json'"
            )
    if output_geometry.resolve() == geometry.resolve():
        raise ValueError("Refusing to overwrite the source geometry; choose another output path")
    if output_geometry.exists() or output_database.exists():
        raise FileExistsError("Conversion outputs already exist; remove or rename them explicitly")

    entries = parse_legacy_materials(materials)
    shutil.copy2(geometry, output_geometry)
    try:
        with h5py.File(output_geometry, "r+") as converted:
            if "/data" not in converted:
                raise ValueError(f"Geometry file {geometry} has no /data dataset")
            _normalise_geometry_spacing(converted, geometry)
            required = _required_material_count(converted)
            if required > len(entries):
                raise ValueError(
                    f"Geometry references {required} material indices but {materials} declares only "
                    f"{len(entries)} materials"
                )
            if "/material_keys" in converted:
                raise ValueError("Source geometry is already a material-database geometry file")
            keys = [key for key, _ in entries]
            converted.create_dataset("/material_keys", data=np.asarray(keys, dtype="S"))
            converted.attrs["MaterialDatabase"] = output_database.stem
            converted.attrs["MaterialDatabaseSchemaVersion"] = 1
            converted.attrs["LegacyMaterialsSource"] = str(materials)
    except Exception:
        output_geometry.unlink()
        raise

    document = create_database_document(
        database_id,
        dict(entries),
        name=f"Converted materials for {geometry.name}",
        description=f"Converted non-destructively from legacy material file {materials.name}.",
    )
    write_database(output_database, document)
    return output_geometry, output_database
