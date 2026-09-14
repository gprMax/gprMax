# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Shared import tools for tagged, cell-centred gprMax geometries."""

from .common import (
    build_tag_volume,
    material_key,
    normalise_tag_name,
    unique_normalised_tags,
    write_geometry_hdf5,
    write_null_material_database,
    write_tag_datasets,
)
from .mesh import convert_mesh, load_mesh_source, write_mesh_template
from .volume import convert_label_volume, load_label_volume, write_label_template

__all__ = [
    "build_tag_volume",
    "convert_label_volume",
    "convert_mesh",
    "load_label_volume",
    "load_mesh_source",
    "material_key",
    "normalise_tag_name",
    "unique_normalised_tags",
    "write_geometry_hdf5",
    "write_null_material_database",
    "write_label_template",
    "write_mesh_template",
    "write_tag_datasets",
]
