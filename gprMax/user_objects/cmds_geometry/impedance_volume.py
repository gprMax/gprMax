# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Tag-driven closed surface-impedance volumes."""

from __future__ import annotations

import logging

import numpy as np

import gprMax.config as config
from gprMax.geometry_tags import validate_geometry_tag
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.impedance_surfaces import create_impedance_marker_material
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class ImpedanceVolume(GeometryUserObject):
    """Turn the currently surviving cells of a geometry tag into a closed
    surface-impedance volume.

    The tagged geometry must be rasterised before this object in scene order.
    Any volumetric primitive that writes the cell-centred geometry-tag map can
    therefore be used without a shape-specific impedance implementation. A
    later geometry object can overwrite the marked cells normally, preserving
    the established last-object-wins rule.

    Zero-thickness plates and patches are intentionally outside this API: they
    do not occupy cells and require a two-sided sheet transition condition.
    """

    @property
    def hash(self):
        return "#impedance_volume"

    def __init__(self, *, geometry_tag: str, surface_impedance_id: str):
        if geometry_tag is None:
            raise ValueError("geometry_tag must name a cell-centred geometry tag")
        geometry_tag = validate_geometry_tag(geometry_tag)
        if not isinstance(surface_impedance_id, str) or not surface_impedance_id:
            raise ValueError("surface_impedance_id must be a non-empty string")
        super().__init__(
            geometry_tag=geometry_tag,
            surface_impedance_id=surface_impedance_id,
        )
        self.geometry_tag = geometry_tag
        self.surface_impedance_id = surface_impedance_id

    def declared_geometry_tags(self) -> tuple[str, ...]:
        """Ensure the target tag has storage before geometry rasterisation."""

        return (self.geometry_tag,)

    def build(self, grid: FDTDGrid) -> None:
        if config.get_model_config().mode != "3D":
            raise ValueError(f"{self.params_str()} currently supports only 3-D models")

        try:
            model = grid.surface_impedance_models[self.surface_impedance_id]
        except KeyError as exc:
            raise ValueError(
                f"{self.params_str()} there is no surface impedance "
                f"with ID {self.surface_impedance_id!r}"
            ) from exc

        tag_map = getattr(grid, "geometry_tag_map", None)
        registry = getattr(grid, "geometry_tag_registry", None)
        if tag_map is None or registry is None:
            raise ValueError(
                f"{self.params_str()} geometry tag {self.geometry_tag!r} is unavailable"
            )
        try:
            tag_id = registry.id_for(self.geometry_tag)
        except KeyError as exc:
            raise ValueError(
                f"{self.params_str()} geometry tag {self.geometry_tag!r} is not registered"
            ) from exc

        selected = tag_map.data == tag_id
        cell_count = int(np.count_nonzero(selected))
        if cell_count == 0:
            raise ValueError(
                f"{self.params_str()} geometry tag {self.geometry_tag!r} has no occupied cells "
                "at this point in scene order"
            )

        marker = create_impedance_marker_material(grid, model.ID)
        grid.solid[selected] = marker.numID

        occupied_axes = (
            np.flatnonzero(np.any(selected, axis=(1, 2))),
            np.flatnonzero(np.any(selected, axis=(0, 2))),
            np.flatnonzero(np.any(selected, axis=(0, 1))),
        )
        lower = tuple(int(values[0]) for values in occupied_axes)
        upper = tuple(int(values[-1]) + 1 for values in occupied_axes)
        grid.impedance_volume_specs.append(
            {
                "kind": "tagged",
                "model_id": model.ID,
                "geometry_tag": self.geometry_tag,
                "cell_count": cell_count,
                "lower": lower,
                "upper": upper,
            }
        )
        logger.info(
            f"{self.grid_name(grid)}Marked {cell_count} surviving voxel(s) tagged "
            f"{self.geometry_tag!r} as a closed impedance volume using {model.ID!r}."
        )
