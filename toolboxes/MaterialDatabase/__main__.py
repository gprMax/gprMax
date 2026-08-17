# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Command-line entry point for material database utilities."""

import argparse
from pathlib import Path

from gprMax.material_database import validate_material_database

from .convert_geometry import convert_geometry


def main():
    parser = argparse.ArgumentParser(description="gprMax material database utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser(
        "convert-geometry",
        help="copy a legacy geometry HDF5 file and convert its material text file to JSON",
    )
    convert.add_argument("geometry")
    convert.add_argument("materials")
    convert.add_argument("--output-geometry")
    convert.add_argument("--output-database")
    list_command = subparsers.add_parser(
        "list",
        help="validate a database and list its available material entries",
    )
    list_command.add_argument("database")
    list_command.add_argument("--directory", type=Path)
    validate = subparsers.add_parser(
        "validate",
        help="validate every entry in a material database",
    )
    validate.add_argument("database")
    validate.add_argument("--directory", type=Path)
    args = parser.parse_args()

    if args.command == "convert-geometry":
        geometry, database = convert_geometry(
            args.geometry,
            args.materials,
            output_geometry=args.output_geometry,
            output_database=args.output_database,
        )
        print(f"Converted geometry: {geometry}")
        print(f"Material database: {database}")
        print(f"Use database name: {database.stem}")
    elif args.command in {"list", "validate"}:
        catalogue = validate_material_database(
            args.database,
            search_directory=args.directory,
        )
        if args.command == "list":
            for key, specification in catalogue:
                print(f"{key}\t{specification.model}\t{specification.name}")
        print(f"Validated {len(catalogue)} material entries in {args.database}")


if __name__ == "__main__":
    main()
