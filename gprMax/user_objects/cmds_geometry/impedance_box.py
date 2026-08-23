# Copyright (C) 2026: The University of Edinburgh, United Kingdom

"""Closed, grid-aligned surface-impedance volume."""

import logging

import gprMax.config as config
from gprMax.cython.geometry_primitives import build_box
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.impedance_surfaces import create_impedance_marker_material
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class ImpedanceBox(GeometryUserObject):
    """Create an opaque closed box governed by a surface-impedance model."""

    @property
    def hash(self):
        return "#impedance_box"

    def __init__(self, p1, p2, surface_impedance_id: str):
        super().__init__(p1=p1, p2=p2, surface_impedance_id=surface_impedance_id)
        self.p1 = p1
        self.p2 = p2
        self.surface_impedance_id = surface_impedance_id

    def build(self, grid: FDTDGrid):
        if config.get_model_config().mode != "3D":
            raise ValueError(f"{self.params_str()} currently supports only 3-D models")
        try:
            model = grid.surface_impedance_models[self.surface_impedance_id]
        except KeyError as exc:
            raise ValueError(
                f"{self.params_str()} there is no surface impedance "
                f"with ID {self.surface_impedance_id!r}"
            ) from exc

        uip = self._create_uip(grid)
        p1 = uip.resolve_inf_point(self.p1, role="lower")
        p2 = uip.resolve_inf_point(self.p2, role="upper")
        contains, lower, upper = uip.check_box_points(p1, p2, self.__str__())
        if not contains:
            return
        if any(stop <= start for start, stop in zip(lower, upper)):
            raise ValueError(f"{self.params_str()} must occupy at least one cell on every axis")

        marker = create_impedance_marker_material(grid, model.ID)
        xs, ys, zs = lower
        xf, yf, zf = upper
        build_box(
            xs,
            xf,
            ys,
            yf,
            zs,
            zf,
            marker.numID,
            marker.numID,
            marker.numID,
            marker.numID,
            False,
            False,
            False,
            False,
            grid.solid,
            grid.rigidE,
            grid.rigidH,
            grid.ID,
        )
        grid.impedance_volume_specs.append(
            {
                "model_id": model.ID,
                "lower": tuple(int(value) for value in lower),
                "upper": tuple(int(value) for value in upper),
            }
        )
        rounded_lower = uip.round_to_grid_static_point(p1)
        rounded_upper = uip.round_to_grid_static_point(p2)
        logger.info(
            f"{self.grid_name(grid)}Impedance box from {rounded_lower[0]:g}m, "
            f"{rounded_lower[1]:g}m, {rounded_lower[2]:g}m to {rounded_upper[0]:g}m, "
            f"{rounded_upper[1]:g}m, {rounded_upper[2]:g}m using {model.ID!r}."
        )
