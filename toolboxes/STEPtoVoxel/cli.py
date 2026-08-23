# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Command-line interface for the STEPtoVoxel toolbox."""

from __future__ import annotations

import argparse
from pathlib import Path

from .converter import (
    ConversionConfig,
    convert_step,
    inspect_step,
    write_material_template,
)


def _config(args: argparse.Namespace) -> ConversionConfig:
    return ConversionConfig(
        voxel_size=tuple(args.voxel_size),
        pad_cells=args.pad_cells,
        supersample=args.supersample,
        sweep_axis=args.sweep_axis,
        material_grouping=getattr(args, "group_mode", "exact"),
        grouping_relative_tolerance=getattr(args, "group_tolerance", 0.01),
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        default=(1e-3, 1e-3, 1e-3),
        help="voxel size in metres (default: 0.001 0.001 0.001)",
    )
    parser.add_argument("--pad-cells", type=int, default=2)
    parser.add_argument(
        "--supersample",
        type=int,
        default=1,
        help="symmetric samples per cell axis (default: one cell-centre sample)",
    )
    parser.add_argument(
        "--sweep-axis",
        choices=("auto", "x", "y", "z"),
        default="auto",
        help="internal slice direction; auto uses the shortest grid dimension",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m toolboxes.STEPtoVoxel",
        description="Convert STEP CAD assemblies into gprMax voxel geometry.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="list STEP components and dimensions")
    inspect.add_argument("step_file", type=Path)
    _common(inspect)

    prepare = commands.add_parser("prepare", help="create an editable material-assignment CSV")
    prepare.add_argument("step_file", type=Path)
    prepare.add_argument("materials_csv", type=Path)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument(
        "--group-mode",
        choices=("none", "exact", "similar"),
        default="exact",
        help="group repeated STEP instances, approximate geometry, or neither",
    )
    prepare.add_argument(
        "--group-tolerance",
        type=float,
        default=0.01,
        help="relative tolerance for similar-geometry suggestions (default: 0.01)",
    )
    _common(prepare)

    convert = commands.add_parser("convert", help="voxelise and export gprMax/VTK files")
    convert.add_argument("step_file", type=Path)
    convert.add_argument("materials_csv", type=Path)
    convert.add_argument("output_dir", type=Path)
    convert.add_argument("--no-vtk", action="store_true")
    _common(convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args)

    if args.command == "inspect":
        parts = inspect_step(args.step_file, config)
        for part in parts:
            cad = part.cad
            kind = "solid" if float(cad.get("vol_m3") or 0.0) > 0 else "surface"
            print(
                f"{part.name}: {kind}; bbox={cad.get('bbox_dims_xyz')}; "
                f"volume={cad.get('vol_m3')}; name_source={part.name_source}; "
                f"confidence={part.name_confidence}; STEP=#{part.step_entity_id}"
            )
        return 0

    if args.command == "prepare":
        path = write_material_template(
            args.step_file,
            args.materials_csv,
            config,
            overwrite=args.overwrite,
        )
        print(f"Wrote material template: {path}")
        return 0

    result = convert_step(
        args.step_file,
        args.materials_csv,
        args.output_dir,
        config,
        write_vtk=not args.no_vtk,
    )
    print(f"Wrote gprMax geometry: {result.geometry_file}")
    print(f"Wrote gprMax materials: {result.materials_file}")
    print(f"Wrote CAD markers: {result.markers_file}")
    if result.vtk_file:
        print(f"Wrote VTK geometry: {result.vtk_file}")
    if result.reference_geometry_cad_file:
        print(f"Wrote CAD-coordinate VTK reference geometry: {result.reference_geometry_cad_file}")
    print(f"Grid: {result.shape}; spacing={result.spacing}")
    return 0
