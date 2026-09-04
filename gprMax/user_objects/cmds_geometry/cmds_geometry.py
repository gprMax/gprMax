# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
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
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import logging

import numpy as np

import gprMax.config as config

logger = logging.getLogger(__name__)


def validate_surface_impedance_geometry_materials(
    materials,
    *,
    geometry,
    cell_volume=True,
    directional=False,
):
    """Validate use of surface-impedance markers by a geometry object.

    A scalar surface impedance owns the boundary of one opaque voxel volume.
    It therefore cannot be assigned directionally and cannot be attached to a
    zero-thickness face or edge, where fields exist on both sides.

    Returns:
        bool: ``True`` when the geometry uses a surface impedance.
    """

    impedance_materials = [
        material for material in materials if hasattr(material, "surface_impedance_id")
    ]
    if not impedance_materials:
        return False

    IDs = ", ".join(repr(material.surface_impedance_id) for material in impedance_materials)
    if not cell_volume:
        raise ValueError(
            f"{geometry} surface-impedance material {IDs} requires a closed, "
            "cell-occupying volume; sheet and edge geometry are unsupported"
        )
    if directional or len(materials) != 1:
        raise ValueError(
            f"{geometry} a surface impedance must be selected with isotropic "
            "material_id; directional material_ids are unsupported"
        )
    return True


def resolve_geometry_materials(
    grid,
    material_ids,
    *,
    geometry,
    cell_volume=True,
    directional=False,
):
    """Resolve bulk or surface-impedance IDs for a geometry object only."""

    from gprMax.impedance_surfaces import create_impedance_marker_material

    requested = list(material_ids)
    resolved = []
    missing = []
    for material_id in requested:
        bulk = next((item for item in grid.materials if item.ID == material_id), None)
        surface = getattr(grid, "surface_impedance_models", {}).get(material_id)
        if bulk is not None and surface is not None:
            raise ValueError(
                f"{geometry} material ID {material_id!r} is ambiguous between a bulk "
                "material and a surface impedance"
            )
        if surface is not None:
            resolved.append(("surface", surface))
        elif bulk is not None:
            resolved.append(("bulk", bulk))
        else:
            missing.append(material_id)

    if missing:
        raise ValueError(f"{geometry} material(s) {missing} do not exist")

    surface_models = [value for kind, value in resolved if kind == "surface"]
    if surface_models and not cell_volume:
        IDs = ", ".join(repr(model.ID) for model in surface_models)
        raise ValueError(
            f"{geometry} surface-impedance material {IDs} requires a closed, "
            "cell-occupying volume; sheet and edge geometry are unsupported"
        )
    if surface_models and (directional or len(requested) != 1):
        raise ValueError(
            f"{geometry} a surface impedance must be selected with isotropic "
            "material_id; directional material_ids are unsupported"
        )

    materials = [
        create_impedance_marker_material(grid, value.ID) if kind == "surface" else value
        for kind, value in resolved
    ]
    validate_surface_impedance_geometry_materials(
        materials,
        geometry=geometry,
        cell_volume=cell_volume,
        directional=directional,
    )
    return materials


def geometry_tag_args(grid, tag):
    """Return the optional dense map and compact ID expected by Cython builders."""

    tag_map = getattr(grid, "geometry_tag_map", None)
    if tag_map is None:
        if tag is not None:
            raise RuntimeError(f"Geometry tag '{tag}' was not registered before rasterisation")
        return None, 0
    return tag_map.data, tag_map.id_for(tag)


def check_averaging(averaging):
    """Check and set material averaging value.

    Args:
        averaging: string for input value from hash command - should be 'y'
                    or 'n'.

    Returns:
        averaging: boolean for geometry object material averaging.
    """

    if averaging.lower() == "y":
        averaging = True
    elif averaging.lower() == "n":
        averaging = False
    else:
        logger.exception("Averaging should be either y or n")

    return averaging
