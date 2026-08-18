# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Command-line interface for labelled volumes and general meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .mesh import convert_mesh, load_mesh_source, write_mesh_template
from .volume import convert_label_volume, load_label_volume, write_label_template


def _voxel_size(values):
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) == 3:
        return tuple(values)
    raise argparse.ArgumentTypeError("voxel size requires one isotropic or three x-y-z values")


def _add_volume_commands(parent):
    commands = parent.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("inspect", help="inspect labels and image geometry")
    prepare = commands.add_parser("prepare", help="write an editable label assignment CSV")
    convert = commands.add_parser("convert", help="convert a label map to gprMax HDF5")
    for command in (inspect, prepare, convert):
        command.add_argument("source", type=Path)
        command.add_argument(
            "--unit",
            default="auto",
            choices=("auto", "m", "mm", "um"),
            help="physical unit; auto uses NIfTI/NRRD metadata",
        )
    prepare.add_argument("assignments", type=Path)
    convert.add_argument("assignments", type=Path)
    convert.add_argument("output", type=Path)


def _add_mesh_commands(parent):
    commands = parent.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("inspect", help="inspect mesh type and region arrays")
    prepare = commands.add_parser("prepare", help="write an editable mesh assignment CSV")
    convert = commands.add_parser("convert", help="voxelise a mesh to gprMax HDF5")
    for command in (inspect, prepare, convert):
        command.add_argument("source", type=Path)
        command.add_argument("--unit", required=True, choices=("m", "mm", "um"))
        command.add_argument("--region-array", help="cell-data array defining mesh regions")
    prepare.add_argument("assignments", type=Path)
    convert.add_argument("assignments", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument(
        "--voxel-size",
        required=True,
        nargs="+",
        type=float,
        metavar="DL",
        help="one isotropic or three x-y-z cell sizes in metres",
    )
    convert.add_argument("--pad-cells", type=int, default=2)
    convert.add_argument("--supersample", type=int, default=1)


def _volume(args):
    volume = load_label_volume(args.source, unit=args.unit)
    if args.action == "inspect":
        print(
            json.dumps(
                {
                    "format": volume.source_format,
                    "shape": list(volume.labels.shape),
                    "spacing_m": list(volume.spacing_m),
                    "first_cell_centre_m": list(volume.first_cell_centre_m),
                    "labels": [int(value) for value in np.unique(volume.labels)],
                    "label_names": dict(volume.label_names),
                },
                indent=2,
            )
        )
    elif args.action == "prepare":
        print(write_label_template(volume, args.assignments))
    else:
        result = convert_label_volume(args.source, args.assignments, args.output, unit=args.unit)
        print(result.geometry_file)


def _mesh(args):
    source = load_mesh_source(
        args.source,
        unit=args.unit,
        region_array=args.region_array,
    )
    if args.action == "inspect":
        print(
            json.dumps(
                {
                    "kind": source.kind,
                    "regions": [region.__dict__ for region in source.regions],
                    "cell_data": sorted(source.dataset.cell_data.keys()),
                },
                indent=2,
            )
        )
    elif args.action == "prepare":
        print(write_mesh_template(source, args.assignments))
    else:
        result = convert_mesh(
            args.source,
            args.assignments,
            args.output,
            voxel_size=_voxel_size(args.voxel_size),
            unit=args.unit,
            region_array=args.region_array,
            pad_cells=args.pad_cells,
            supersample=args.supersample,
        )
        print(result.geometry_file)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m toolboxes.GeometryImport",
        description="Import labelled medical volumes and general meshes into gprMax",
    )
    formats = parser.add_subparsers(dest="source_type", required=True)
    _add_volume_commands(formats.add_parser("volume", help="NIfTI/NRRD/MetaImage labels"))
    _add_mesh_commands(formats.add_parser("mesh", help="Gmsh/VTK/VTP/VTU meshes"))
    args = parser.parse_args(argv)
    if args.source_type == "volume":
        _volume(args)
    else:
        _mesh(args)
