# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Versioned JSON material databases and their gprMax translation layer."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import DispersiveMaterial, Material

SCHEMA_NAME = "gprMax-material-database"
SCHEMA_VERSION = 1
DATABASE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
MATERIAL_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
SUPPORTED_MODELS = frozenset(("constant", "debye", "lorentz", "drude", "general", "builtin"))
BUILTIN_MATERIALS = frozenset(("pec", "pmc", "free_space"))


@dataclass(frozen=True)
class MaterialDatabaseSource:
    """Identity of the database from which a material was resolved."""

    database_id: str
    database_version: str
    path: str
    official: bool


@dataclass(frozen=True)
class MaterialSpec:
    """Validated, grid-independent material definition."""

    key: str
    name: str
    model: str
    relative_permittivity: float
    electric_conductivity: float
    relative_permeability: float
    magnetic_conductivity: float
    poles: Tuple[Mapping[str, Any], ...]
    inclusive_conductivity: float
    averagable: Optional[bool]
    metadata: Mapping[str, Any]
    source: MaterialDatabaseSource
    entry_sha256: str


_DOCUMENT_CACHE: Dict[Path, Tuple[int, Mapping[str, Any]]] = {}


def make_database_id(value: str, *, prefix: str = "database") -> str:
    """Return a schema-safe identifier for a generated database artefact."""

    identifier = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_-")
    if not identifier:
        identifier = prefix
    if not identifier[0].isalpha():
        identifier = f"{prefix}_{identifier}"
    return identifier


def official_material_directory() -> Path:
    """Return the installed directory containing official databases."""

    return Path(__file__).resolve().parent / "data" / "materials"


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Material database file '{path}' does not exist") from exc

    cached = _DOCUMENT_CACHE.get(path)
    if cached is not None and cached[0] == stat.st_mtime_ns:
        return cached[1]

    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Material database '{path}' is not valid JSON: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(f"Material database '{path}' must contain a JSON object at its root")
    _DOCUMENT_CACHE[path] = (stat.st_mtime_ns, document)
    return document


def _official_manifest() -> Mapping[str, str]:
    manifest_path = official_material_directory() / "databases.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported official database manifest '{manifest_path}'")
    databases = manifest.get("databases")
    if not isinstance(databases, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in databases.items()
    ):
        raise ValueError(f"Official database manifest '{manifest_path}' has an invalid mapping")
    return databases


def resolve_database_path(
    database: str, search_directory: Optional[Path] = None
) -> Tuple[Path, bool]:
    """Resolve a database name using the fixed official/local search policy.

    Official names are reserved and always resolve to packaged data. Other
    names resolve to ``<name>.json`` in the input file directory (or the
    current working directory for a direct Python API model).
    """

    if not isinstance(database, str) or not DATABASE_NAME_PATTERN.fullmatch(database):
        raise ValueError(
            "Material database names must start with a letter and contain only letters, "
            "numbers, underscores, or hyphens"
        )

    official = _official_manifest()
    if database in official:
        return official_material_directory() / official[database], True

    directory = Path.cwd() if search_directory is None else Path(search_directory)
    path = directory / f"{database}.json"
    if not path.is_file():
        available = ", ".join(sorted(official))
        raise FileNotFoundError(
            f"Material database '{database}' was not found at '{path}'. "
            f"Official databases are: {available}"
        )
    return path, False


def _finite_number(value: Any, field: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Material field '{field}' must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Material field '{field}' must be a number") from exc
    if not np.isfinite(number):
        raise ValueError(f"Material field '{field}' must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"Material field '{field}' must be at least {minimum:g}")
    return number


def _conductivity(value: Any, field: str) -> float:
    if value == "inf":
        return float("inf")
    return _finite_number(value, field, minimum=0.0)


def _complex_pair(value: Any, field: str) -> complex:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Material field '{field}' must be a [real, imaginary] pair")
    return complex(
        _finite_number(value[0], f"{field}[0]"),
        _finite_number(value[1], f"{field}[1]"),
    )


def _check_keys(
    values: Mapping[str, Any],
    context: str,
    *,
    allowed: Sequence[str],
    required: Sequence[str] = (),
) -> None:
    unknown = set(values) - set(allowed)
    missing = set(required) - set(values)
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{context} is missing required fields: {sorted(missing)}")


def _validate_root(document: Mapping[str, Any], path: Path) -> Tuple[str, str, Mapping[str, Any]]:
    _check_keys(
        document,
        f"Material database '{path}'",
        allowed=("schema", "schema_version", "database", "materials"),
        required=("schema", "schema_version", "database", "materials"),
    )
    if document.get("schema") != SCHEMA_NAME or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Material database '{path}' must use {SCHEMA_NAME!r} schema version {SCHEMA_VERSION}"
        )
    database = document.get("database")
    if not isinstance(database, dict):
        raise ValueError(f"Material database '{path}' is missing its 'database' metadata object")
    database_id = database.get("id")
    version = database.get("version")
    if not isinstance(database_id, str) or not DATABASE_NAME_PATTERN.fullmatch(database_id):
        raise ValueError(f"Material database '{path}' has an invalid database ID")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Material database '{path}' has an invalid version")
    materials = document.get("materials")
    if not isinstance(materials, dict):
        raise ValueError(f"Material database '{path}' is missing its 'materials' mapping")
    return database_id, version, materials


def _normalise_poles(model: str, poles: Any, context: str) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(poles, list) or not poles:
        raise ValueError(f"{context} model '{model}' requires at least one pole")
    normalised = []
    for index, pole in enumerate(poles):
        field = f"{context}.poles[{index}]"
        if not isinstance(pole, dict):
            raise ValueError(f"{field} must be an object")
        if model == "debye":
            _check_keys(
                pole,
                field,
                allowed=("relative_permittivity_difference", "relaxation_time_s"),
                required=("relative_permittivity_difference", "relaxation_time_s"),
            )
            normalised.append(
                {
                    "relative_permittivity_difference": _finite_number(
                        pole.get("relative_permittivity_difference"),
                        f"{field}.relative_permittivity_difference",
                        minimum=0.0,
                    ),
                    "relaxation_time_s": _finite_number(
                        pole.get("relaxation_time_s"),
                        f"{field}.relaxation_time_s",
                        minimum=np.finfo(float).tiny,
                    ),
                }
            )
        elif model == "lorentz":
            _check_keys(
                pole,
                field,
                allowed=(
                    "relative_permittivity_difference",
                    "resonance_frequency_hz",
                    "damping_coefficient_per_s",
                ),
                required=(
                    "relative_permittivity_difference",
                    "resonance_frequency_hz",
                    "damping_coefficient_per_s",
                ),
            )
            normalised.append(
                {
                    "relative_permittivity_difference": _finite_number(
                        pole.get("relative_permittivity_difference"),
                        f"{field}.relative_permittivity_difference",
                        minimum=0.0,
                    ),
                    "resonance_frequency_hz": _finite_number(
                        pole.get("resonance_frequency_hz"),
                        f"{field}.resonance_frequency_hz",
                        minimum=np.finfo(float).tiny,
                    ),
                    "damping_coefficient_per_s": _finite_number(
                        pole.get("damping_coefficient_per_s"),
                        f"{field}.damping_coefficient_per_s",
                        minimum=0.0,
                    ),
                }
            )
        elif model == "drude":
            _check_keys(
                pole,
                field,
                allowed=("plasma_frequency_hz", "collision_frequency_per_s"),
                required=("plasma_frequency_hz", "collision_frequency_per_s"),
            )
            normalised.append(
                {
                    "plasma_frequency_hz": _finite_number(
                        pole.get("plasma_frequency_hz"),
                        f"{field}.plasma_frequency_hz",
                        minimum=np.finfo(float).tiny,
                    ),
                    "collision_frequency_per_s": _finite_number(
                        pole.get("collision_frequency_per_s"),
                        f"{field}.collision_frequency_per_s",
                        minimum=np.finfo(float).tiny,
                    ),
                }
            )
        else:
            _check_keys(
                pole,
                field,
                allowed=("w_per_s", "q_per_s"),
                required=("w_per_s", "q_per_s"),
            )
            q = _complex_pair(pole.get("q_per_s"), f"{field}.q_per_s")
            if q.real >= 0:
                raise ValueError(f"{field}.q_per_s must have a negative real part for stability")
            normalised.append(
                {
                    "w_per_s": _complex_pair(pole.get("w_per_s"), f"{field}.w_per_s"),
                    "q_per_s": q,
                }
            )
    return tuple(normalised)


def _entry_hash(entry: Mapping[str, Any]) -> str:
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_material_spec(
    database: str,
    material: str,
    *,
    search_directory: Optional[Path] = None,
    _visited: Sequence[Tuple[str, str]] = (),
) -> MaterialSpec:
    """Resolve and validate one entry, following aliases if necessary."""

    if not isinstance(material, str) or not MATERIAL_KEY_PATTERN.fullmatch(material):
        raise ValueError(
            "Material keys must start with a letter and contain only letters, numbers, "
            "underscores, hyphens, or dots"
        )
    identity = (database, material)
    if identity in _visited:
        chain = " -> ".join(f"{db}:{key}" for db, key in (*_visited, identity))
        raise ValueError(f"Circular material database alias: {chain}")

    path, official = resolve_database_path(database, search_directory)
    document = _read_json(path)
    database_id, database_version, materials = _validate_root(document, path)
    if database_id != database:
        raise ValueError(
            f"Material database '{path}' declares ID '{database_id}', but was selected as "
            f"'{database}'"
        )
    if material not in materials:
        choices = ", ".join(sorted(materials)) or "(none)"
        raise KeyError(
            f"Material '{material}' does not exist in database '{database}'. "
            f"Available entries: {choices}"
        )
    entry = materials[material]
    if not isinstance(entry, dict):
        raise ValueError(f"Material entry '{database}:{material}' must be an object")

    alias = entry.get("alias")
    if alias is not None:
        _check_keys(
            entry,
            f"Material alias '{database}:{material}'",
            allowed=("name", "alias", "metadata"),
            required=("alias",),
        )
        if not isinstance(alias, dict) or not isinstance(alias.get("material"), str):
            raise ValueError(f"Material alias '{database}:{material}' is invalid")
        _check_keys(
            alias,
            f"Material alias '{database}:{material}'.alias",
            allowed=("database", "material"),
            required=("material",),
        )
        alias_database = alias.get("database", database)
        if not isinstance(alias_database, str):
            raise ValueError(f"Material alias '{database}:{material}' has an invalid database")
        return load_material_spec(
            alias_database,
            alias["material"],
            search_directory=search_directory,
            _visited=(*_visited, identity),
        )

    context = f"Material '{database}:{material}'"
    model = entry.get("model")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"{context} has unsupported model {model!r}")
    model_fields = {
        "constant": ("name", "model", "base", "averagable", "metadata"),
        "debye": ("name", "model", "base", "poles", "averagable", "metadata"),
        "lorentz": ("name", "model", "base", "poles", "averagable", "metadata"),
        "drude": ("name", "model", "base", "poles", "averagable", "metadata"),
        "general": (
            "name",
            "model",
            "base",
            "poles",
            "inclusive_conductivity_s_per_m",
            "averagable",
            "metadata",
        ),
        "builtin": ("name", "model", "builtin", "averagable", "metadata"),
    }
    required_fields = ("model", "builtin") if model == "builtin" else ("model", "base")
    _check_keys(
        entry,
        context,
        allowed=model_fields[model],
        required=required_fields,
    )
    name = entry.get("name", material)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{context} has an invalid name")

    if model == "builtin":
        builtin = entry.get("builtin")
        if builtin not in BUILTIN_MATERIALS:
            raise ValueError(f"{context} has unsupported builtin material {builtin!r}")
        builtins = {
            "pec": (1.0, float("inf"), 1.0, 0.0, False),
            "pmc": (1.0, 0.0, 1.0, float("inf"), False),
            "free_space": (1.0, 0.0, 1.0, 0.0, True),
        }
        er, se, mr, sm, default_averagable = builtins[builtin]
        poles: Tuple[Mapping[str, Any], ...] = ()
        inclusive_conductivity = 0.0
    else:
        base = entry.get("base")
        if not isinstance(base, dict):
            raise ValueError(f"{context} is missing its 'base' constitutive parameters")
        _check_keys(
            base,
            f"{context}.base",
            allowed=(
                "relative_permittivity",
                "electric_conductivity_s_per_m",
                "relative_permeability",
                "magnetic_conductivity_s_per_m",
            ),
            required=("relative_permittivity",),
        )
        er = _finite_number(base.get("relative_permittivity"), "relative_permittivity", minimum=1.0)
        se = _conductivity(
            base.get("electric_conductivity_s_per_m", 0.0), "electric_conductivity_s_per_m"
        )
        mr = _finite_number(
            base.get("relative_permeability", 1.0), "relative_permeability", minimum=1.0
        )
        sm = _conductivity(
            base.get("magnetic_conductivity_s_per_m", 0.0), "magnetic_conductivity_s_per_m"
        )
        default_averagable = not (np.isinf(se) or np.isinf(sm))
        poles = () if model == "constant" else _normalise_poles(model, entry.get("poles"), context)
        inclusive_conductivity = (
            _finite_number(
                entry.get("inclusive_conductivity_s_per_m", 0.0),
                "inclusive_conductivity_s_per_m",
                minimum=0.0,
            )
            if model == "general"
            else 0.0
        )

    averagable = entry.get("averagable")
    if averagable is not None and not isinstance(averagable, bool):
        raise ValueError(f"{context} field 'averagable' must be true or false")
    if averagable is None:
        averagable = default_averagable

    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} field 'metadata' must be an object")
    return MaterialSpec(
        key=material,
        name=name,
        model=model,
        relative_permittivity=er,
        electric_conductivity=se,
        relative_permeability=mr,
        magnetic_conductivity=sm,
        poles=poles,
        inclusive_conductivity=inclusive_conductivity,
        averagable=averagable,
        metadata=metadata,
        source=MaterialDatabaseSource(database_id, database_version, str(path), official),
        entry_sha256=_entry_hash(entry),
    )


def validate_material_database(
    database: str, *, search_directory: Optional[Path] = None
) -> Tuple[Tuple[str, MaterialSpec], ...]:
    """Validate every entry in a database and return an immutable catalogue.

    The requested key is retained separately because an alias resolves to the
    canonical target specification. This function is also the implementation
    behind the command-line ``list`` and ``validate`` utilities.
    """

    path, _ = resolve_database_path(database, search_directory)
    document = _read_json(path)
    database_id, _, materials = _validate_root(document, path)
    if database_id != database:
        raise ValueError(
            f"Material database '{path}' declares ID '{database_id}', but was selected as "
            f"'{database}'"
        )
    return tuple(
        (
            key,
            load_material_spec(database, key, search_directory=search_directory),
        )
        for key in sorted(materials)
    )


def build_material_from_spec(grid: FDTDGrid, spec: MaterialSpec, material_id: str) -> Material:
    """Create a grid material from a validated, immutable specification."""

    if any(existing.ID == material_id for existing in grid.materials):
        raise ValueError(f"Material with ID '{material_id}' already exists")

    if spec.model in ("constant", "builtin"):
        result: Material = Material(len(grid.materials), material_id)
    else:
        result = DispersiveMaterial(len(grid.materials), material_id)
        result.type = spec.model
        result.poles = len(spec.poles)

    result.er = spec.relative_permittivity
    result.se = spec.electric_conductivity
    result.mr = spec.relative_permeability
    result.sm = spec.magnetic_conductivity
    result.averagable = bool(spec.averagable)

    if isinstance(result, DispersiveMaterial):
        if result.is_pec or result.is_pmc:
            raise ValueError("Perfect conductors cannot contain electric dispersion")
        if spec.model == "debye":
            result.deltaer = [pole["relative_permittivity_difference"] for pole in spec.poles]
            result.tau = [pole["relaxation_time_s"] for pole in spec.poles]
        elif spec.model == "lorentz":
            result.deltaer = [pole["relative_permittivity_difference"] for pole in spec.poles]
            result.tau = [pole["resonance_frequency_hz"] for pole in spec.poles]
            result.alpha = [pole["damping_coefficient_per_s"] for pole in spec.poles]
            if any(
                frequency >= (2.0 * np.pi) / grid.dt
                or damping >= 1.0 / grid.dt
                or frequency == damping
                for frequency, damping in zip(result.tau, result.alpha)
            ):
                raise ValueError(
                    f"Lorentz material '{material_id}' has a pole incompatible with dt={grid.dt:g} s"
                )
        elif spec.model == "drude":
            result.tau = [pole["plasma_frequency_hz"] for pole in spec.poles]
            result.alpha = [pole["collision_frequency_per_s"] for pole in spec.poles]
            if any(
                frequency >= (2.0 * np.pi) / grid.dt or collision >= 1.0 / grid.dt
                for frequency, collision in zip(result.tau, result.alpha)
            ):
                raise ValueError(
                    f"Drude material '{material_id}' has a pole incompatible with dt={grid.dt:g} s"
                )
        else:
            result.inclusive_w = [pole["w_per_s"] for pole in spec.poles]
            result.inclusive_q = [pole["q_per_s"] for pole in spec.poles]
            result.inclusive_conductivity = spec.inclusive_conductivity

        result.averagable = bool(spec.averagable) and config.get_model_config().dispersive_averaging
        config.get_model_config().materials["maxpoles"] = max(
            config.get_model_config().materials["maxpoles"], result.poles
        )

    result.database_provenance = {
        "database_id": spec.source.database_id,
        "database_version": spec.source.database_version,
        "entry_key": spec.key,
        "entry_sha256": spec.entry_sha256,
        "official": spec.source.official,
        "source": spec.source.path,
    }
    grid.materials.append(result)
    return result


def material_matches_spec(material: Material, spec: MaterialSpec) -> bool:
    """Return whether a live material is constitutively equivalent to *spec*."""

    scalar_values = (
        (material.er, spec.relative_permittivity),
        (material.se, spec.electric_conductivity),
        (material.mr, spec.relative_permeability),
        (material.sm, spec.magnetic_conductivity),
    )
    if not all(np.isclose(actual, expected, equal_nan=False) for actual, expected in scalar_values):
        return False
    if spec.model in ("constant", "builtin"):
        return not isinstance(material, DispersiveMaterial) or material.poles == 0
    if not isinstance(material, DispersiveMaterial) or material.poles != len(spec.poles):
        return False
    if spec.model != "general" and spec.model not in material.type:
        return False
    if spec.model == "debye":
        return np.allclose(
            material.deltaer,
            [pole["relative_permittivity_difference"] for pole in spec.poles],
        ) and np.allclose(material.tau, [pole["relaxation_time_s"] for pole in spec.poles])
    if spec.model == "lorentz":
        return (
            np.allclose(
                material.deltaer,
                [pole["relative_permittivity_difference"] for pole in spec.poles],
            )
            and np.allclose(
                material.tau,
                [pole["resonance_frequency_hz"] for pole in spec.poles],
            )
            and np.allclose(
                material.alpha,
                [pole["damping_coefficient_per_s"] for pole in spec.poles],
            )
        )
    if spec.model == "drude":
        return np.allclose(
            material.tau,
            [pole["plasma_frequency_hz"] for pole in spec.poles],
        ) and np.allclose(
            material.alpha,
            [pole["collision_frequency_per_s"] for pole in spec.poles],
        )
    return (
        np.allclose(material.inclusive_w, [pole["w_per_s"] for pole in spec.poles])
        and np.allclose(material.inclusive_q, [pole["q_per_s"] for pole in spec.poles])
        and np.isclose(material.inclusive_conductivity, spec.inclusive_conductivity)
    )


def material_to_database_entry(material: Material) -> Mapping[str, Any]:
    """Serialise a live material without losing its dispersive representation."""

    base = {
        "relative_permittivity": float(material.er),
        "electric_conductivity_s_per_m": "inf" if np.isinf(material.se) else float(material.se),
        "relative_permeability": float(material.mr),
        "magnetic_conductivity_s_per_m": "inf" if np.isinf(material.sm) else float(material.sm),
    }
    entry: Dict[str, Any] = {
        "name": material.ID,
        "model": "constant",
        "base": base,
        "averagable": bool(material.averagable),
    }
    if not isinstance(material, DispersiveMaterial) or material.poles == 0:
        return entry

    if material.inclusive_w:
        entry["model"] = "general"
        entry["inclusive_conductivity_s_per_m"] = float(material.inclusive_conductivity)
        entry["poles"] = [
            {
                "w_per_s": [float(complex(w).real), float(complex(w).imag)],
                "q_per_s": [float(complex(q).real), float(complex(q).imag)],
            }
            for w, q in zip(material.inclusive_w, material.inclusive_q)
        ]
    elif "debye" in material.type:
        entry["model"] = "debye"
        entry["poles"] = [
            {
                "relative_permittivity_difference": float(delta),
                "relaxation_time_s": float(tau),
            }
            for delta, tau in zip(material.deltaer, material.tau)
        ]
    elif "lorentz" in material.type:
        entry["model"] = "lorentz"
        entry["poles"] = [
            {
                "relative_permittivity_difference": float(delta),
                "resonance_frequency_hz": float(frequency),
                "damping_coefficient_per_s": float(damping),
            }
            for delta, frequency, damping in zip(material.deltaer, material.tau, material.alpha)
        ]
    elif "drude" in material.type:
        entry["model"] = "drude"
        entry["poles"] = [
            {
                "plasma_frequency_hz": float(frequency),
                "collision_frequency_per_s": float(collision),
            }
            for frequency, collision in zip(material.tau, material.alpha)
        ]
    else:
        raise ValueError(f"Cannot serialise unsupported dispersive material type '{material.type}'")
    return entry


def create_database_document(
    database_id: str,
    materials: Mapping[str, Mapping[str, Any]],
    *,
    name: Optional[str] = None,
    version: str = "1.0.0",
    description: str = "",
) -> Mapping[str, Any]:
    """Create a schema-versioned database document for converters/exporters."""

    if not isinstance(database_id, str) or not DATABASE_NAME_PATTERN.fullmatch(database_id):
        raise ValueError(f"Invalid generated material database ID {database_id!r}")

    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "database": {
            "id": database_id,
            "name": name or database_id,
            "version": version,
            "description": description,
        },
        "materials": dict(materials),
    }


def write_database(path: Path, document: Mapping[str, Any]) -> None:
    """Write a deterministic, human-readable material database."""

    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=False, ensure_ascii=False)
        stream.write("\n")
