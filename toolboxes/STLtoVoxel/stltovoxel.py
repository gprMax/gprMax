import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gprMax.material_database import create_database_document, make_database_id, write_database
from toolboxes.GeometryImport.common import (
    build_tag_volume,
    normalise_tag_name,
    unique_normalised_tags,
    write_geometry_hdf5,
)

from .convert import convert_files

logger = logging.getLogger(__name__)


ASSIGNMENT_COLUMNS = ("file", "include", "priority", "material_name", "geometry_tag")


@dataclass(frozen=True)
class STLAssignment:
    path: Path
    include: bool
    priority: int
    material_name: str | None
    geometry_tag: str | None


def _yes(value: str) -> bool:
    value = value.strip().lower()
    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"include must be y/n, got {value!r}")


def write_assignment_template(files, path, *, overwrite=False):
    """Write an editable STL part/material/tag assignment table."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    tags = unique_normalised_tags([item.stem for item in files])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ASSIGNMENT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for priority, (source, tag) in enumerate(zip(files, tags)):
            writer.writerow(
                {
                    "file": source.name,
                    "include": "y",
                    "priority": priority,
                    "material_name": source.stem,
                    "geometry_tag": tag,
                }
            )
    return path


def read_assignments(files, path=None):
    """Return validated assignments, defaulting to one tag/material per STL."""

    files = tuple(Path(item) for item in files)
    defaults = unique_normalised_tags([item.stem for item in files])
    if path is None:
        return [
            STLAssignment(source, True, index, source.stem, tag)
            for index, (source, tag) in enumerate(zip(files, defaults))
        ]

    by_name = {source.name: source for source in files}
    assignments = []
    seen = set()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(ASSIGNMENT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            name = row["file"].strip()
            if name not in by_name:
                raise ValueError(f"Assignment references unknown STL file: {name}")
            if name in seen:
                raise ValueError(f"STL file occurs more than once in assignments: {name}")
            seen.add(name)
            include = _yes(row["include"])
            material_name = row["material_name"].strip() or None
            if include and material_name is None:
                raise ValueError(f"material_name is required for {name}")
            requested_tag = row["geometry_tag"].strip()
            tag = normalise_tag_name(requested_tag or Path(name).stem) if include else None
            assignments.append(
                STLAssignment(
                    by_name[name],
                    include,
                    int(row["priority"]),
                    material_name,
                    tag,
                )
            )
    missing_files = set(by_name) - seen
    if missing_files:
        raise ValueError(f"Assignments omit STL files: {sorted(missing_files)}")
    return assignments


def _write_or_preserve_database(database_file, database_id, material_keys, entries):
    """Create a material template without overwriting user-supplied values."""

    if database_file.exists():
        try:
            existing = json.loads(database_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Existing material database {database_file} is invalid JSON: {exc.msg}"
            ) from exc
        recorded_id = existing.get("database", {}).get("id")
        recorded_materials = existing.get("materials")
        if recorded_id != database_id or not isinstance(recorded_materials, dict):
            raise ValueError(
                f"Existing material database {database_file} is not the companion database "
                "for this conversion"
            )
        if list(recorded_materials) != material_keys:
            raise ValueError(
                f"Existing material database {database_file} has material keys that do not "
                "match the STL inputs; move it aside before converting the changed geometry"
            )
        logger.info(f"Preserved existing editable material database: {database_file}")
        return

    write_database(
        database_file,
        create_database_document(
            database_id,
            entries,
            name="STLtoVoxel material assignments",
            description=(
                "Template generated by STLtoVoxel. Replace null constitutive values before use."
            ),
        ),
    )
    logger.info(f"Written editable material database: {database_file}")


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert one or more STL files to a voxelised gprMax geometry.",
    )
    parser.add_argument(
        "stlfiles",
        help=(
            "can be the filename of a single STL file, or the path to a folder "
            "containing multiple STL files"
        ),
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        help="optional CSV mapping STL files independently to materials and geometry tags",
    )
    parser.add_argument(
        "--prepare",
        type=Path,
        metavar="CSV",
        help="write an editable assignment CSV and exit",
    )
    parser.add_argument(
        "-dxdydz",
        type=float,
        help="discretisation to use in voxelisation process (required for conversion)",
    )
    parser.add_argument(
        "--unit",
        choices=("m", "mm", "um"),
        default="mm",
        help="unit used by STL coordinates (default: mm for backwards compatibility)",
    )
    args = parser.parse_args()

    input_path = Path(args.stlfiles)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.stl"))
        filename_hdf5 = input_path / f"{input_path.name}_geo.h5"
    elif input_path.is_file():
        files = [input_path]
        filename_hdf5 = input_path.with_name(f"{input_path.stem}_geo.h5")
    else:
        parser.error(f"STL file or directory does not exist: {input_path}")

    if not files:
        parser.error(f"No STL files found in: {input_path}")

    if args.prepare is not None:
        write_assignment_template(files, args.prepare)
        logger.info(f"Written STL assignment template: {args.prepare}")
        return
    if args.dxdydz is None:
        parser.error("-dxdydz is required when converting STL geometry")

    try:
        assignments = read_assignments(files, args.assignments)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    selected = sorted(
        (assignment for assignment in assignments if assignment.include),
        key=lambda assignment: (assignment.priority, files.index(assignment.path)),
    )
    if not selected:
        parser.error("No STL files have include=y")
    files = [assignment.path for assignment in selected]

    dxdydz = (args.dxdydz, args.dxdydz, args.dxdydz)

    newline = "\n\t"
    logger.info(f"\nConverting STL file(s): {newline.join(map(str, files))}")
    component_array = convert_files(files, dxdydz, source_unit=args.unit)
    logger.info(
        "Number of voxels: "
        f"{component_array.shape[0]} x {component_array.shape[1]} x "
        f"{component_array.shape[2]}"
    )
    logger.info(f"Spatial discretisation: {dxdydz[0]} x {dxdydz[1]} x {dxdydz[2]}m")

    # Write HDF5 file for gprMax using voxels
    material_names = []
    material_index = {}
    component_material_ids = []
    component_tags = []
    material_sources = {}
    for assignment in selected:
        assert assignment.material_name is not None
        if assignment.material_name not in material_index:
            material_index[assignment.material_name] = len(material_names)
            material_names.append(assignment.material_name)
            material_sources[assignment.material_name] = []
        component_material_ids.append(material_index[assignment.material_name])
        component_tags.append(assignment.geometry_tag)
        material_sources[assignment.material_name].append(str(assignment.path))

    if len(material_names) > np.iinfo(np.int16).max + 1:
        raise ValueError("Imported geometry exceeds the int16 material-index schema")
    material_array = component_array.copy()
    occupied = component_array >= 0
    lookup = np.asarray(component_material_ids, dtype=np.int16)
    material_array[occupied] = lookup[component_array[occupied]]
    tag_data, tag_names = build_tag_volume(component_array, component_tags)

    material_keys = []
    for index, name in enumerate(material_names):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-") or "material"
        material_keys.append(f"material_{index:03d}_{slug}")
    database_id = make_database_id(f"{filename_hdf5.stem}_materials", prefix="geometry")
    database_file = filename_hdf5.with_name(f"{database_id}.json")
    write_geometry_hdf5(
        filename_hdf5,
        material_array,
        dxdydz,
        material_keys=material_keys,
        material_database=database_id,
        tag_data=tag_data,
        tag_names=tag_names,
    )
    logger.info(f"Written geometry object file: {filename_hdf5}")

    # Do not silently invent properties for CAD/STL solids. The null values
    # make this an editable template and produce a clear validation error if
    # it is used before the constitutive parameters are filled in.
    entries = {}
    for key, name in zip(material_keys, material_names):
        entries[key] = {
            "name": name,
            "model": "constant",
            "base": {
                "relative_permittivity": None,
                "electric_conductivity_s_per_m": None,
                "relative_permeability": None,
                "magnetic_conductivity_s_per_m": None,
            },
            "metadata": {
                "original_id": name,
                "source_stl": material_sources[name],
            },
        }
    _write_or_preserve_database(
        database_file,
        database_id,
        material_keys,
        entries,
    )


if __name__ == "__main__":
    main()
