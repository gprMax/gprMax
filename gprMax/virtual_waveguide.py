# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
# Authors: Craig Warren, Antonis Giannopoulos, John Hartley, and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

"""Virtual FDTD waveguides coupled to eigenmode-port apertures."""

from __future__ import annotations

import copy
import logging

import numpy as np

import gprMax.config as config
from gprMax.cython.virtual_waveguide import (
    couple_virtual_waveguide_electric,
    couple_virtual_waveguide_magnetic,
)
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.updates.cpu_updates import CPUUpdates

logger = logging.getLogger(__name__)


class VirtualWaveguide:
    """An auxiliary Yee grid joined bidirectionally to one modal aperture."""

    def __init__(self, main_grid, port, spec):
        self.main_grid = main_grid
        self.port = port
        self.spec = spec
        self.normal_axis = int(port.normal_axis)
        self.direction_sign = 1 if port.direction == "+" else -1
        self.transverse_axes = tuple(int(value) for value in port.transverse_axes)
        self.plane_index = int(port.plane_index)
        self.u0, self.v0 = (int(value) for value in port.transverse_start)
        self.u1, self.v1 = (int(value) for value in port.transverse_stop)
        self.nu = self.u1 - self.u0
        self.nv = self.v1 - self.v0

        self._validate()
        self.aux_grid = self._build_auxiliary_grid()
        self.aux_source = self._build_auxiliary_source()
        self.aux_updates = CPUUpdates(self.aux_grid)

        # A virtual split makes only the main-domain side of the H plane a
        # valid sampling plane. This is already the source-monitor policy;
        # passive monitors need the same policy once their rear is detached.
        if self.port.port_monitor is not None:
            self.port.port_monitor.magnetic_side = 1

    def _validate(self):
        mode = config.get_model_config().mode
        if mode != "3D":
            raise ValueError("Virtual waveguides currently require a 3D model.")
        if config.sim_config.general["solver"] != "cpu":
            raise ValueError(
                "Virtual waveguides currently require the CPU solver; GPU support "
                "will follow the eigenmode-port GPU implementation."
            )
        if config.sim_config.mpi:
            raise ValueError("Virtual waveguides do not yet support MPI.")
        if self.port.invariant_axis is not None:
            raise ValueError("Virtual waveguides do not support a 2D eigenmode port.")
        if self.spec.pml_cells < 2:
            raise ValueError("A virtual-waveguide PML must contain at least two cells.")
        if self.spec.source_clearance_cells < 1:
            raise ValueError("Virtual-waveguide source clearance must be at least one cell.")
        minimum_length = self.spec.pml_cells + self.spec.source_clearance_cells + 3
        if self.spec.length_cells < minimum_length:
            raise ValueError(
                "Virtual-waveguide length must be at least PML cells + source "
                f"clearance + 3 cells ({minimum_length} cells for this request)."
            )
        normal_cells = int(self.main_grid.size[self.normal_axis])
        if not 1 <= self.plane_index < normal_cells:
            raise ValueError("A virtual-waveguide aperture must be an internal Yee plane.")
        if self.nu < 2 or self.nv < 2:
            raise ValueError(
                "A virtual-waveguide cross-section must be at least two cells "
                "along each transverse axis."
            )

        first_ids, second_ids = self._adjacent_component_ids()
        first_solid, second_solid = self._adjacent_solids()
        if not np.array_equal(first_solid, second_solid) or not np.array_equal(
            first_ids, second_ids
        ):
            raise ValueError(
                "A virtual-waveguide aperture must lie in a locally uniform guide: "
                "cell materials or interior Yee-component IDs differ across the "
                f"{port_plane_label(self.normal_axis, self.plane_index)} split."
            )

        material_ids = np.unique(self._component_cross_section())
        dispersive = [
            material.ID
            for material in self.main_grid.materials
            if material.numID in material_ids and getattr(material, "poles", 0) > 0
        ]
        if dispersive:
            raise ValueError(
                "Virtual-waveguide aperture coupling does not yet support "
                "dispersive guide materials; found " + ", ".join(dispersive) + "."
            )

    def _adjacent_solids(self):
        grid = self.main_grid
        p = self.plane_index
        if self.normal_axis == 0:
            return (
                grid.solid[p - 1, self.u0 : self.u1, self.v0 : self.v1],
                grid.solid[p, self.u0 : self.u1, self.v0 : self.v1],
            )
        if self.normal_axis == 1:
            return (
                grid.solid[self.u0 : self.u1, p - 1, self.v0 : self.v1],
                grid.solid[self.u0 : self.u1, p, self.v0 : self.v1],
            )
        return (
            grid.solid[self.u0 : self.u1, self.v0 : self.v1, p - 1],
            grid.solid[self.u0 : self.u1, self.v0 : self.v1, p],
        )

    def _adjacent_component_ids(self):
        grid = self.main_grid
        p = self.plane_index
        # Perimeter IDs may deliberately contain zero-thickness PEC connector
        # walls. Compare the interior to detect a longitudinal discontinuity
        # without rejecting the physical wall at the aperture.
        if self.normal_axis == 0:
            return (
                grid.ID[:, p - 1, self.u0 + 1 : self.u1, self.v0 + 1 : self.v1],
                grid.ID[:, p, self.u0 + 1 : self.u1, self.v0 + 1 : self.v1],
            )
        if self.normal_axis == 1:
            return (
                grid.ID[:, self.u0 + 1 : self.u1, p - 1, self.v0 + 1 : self.v1],
                grid.ID[:, self.u0 + 1 : self.u1, p, self.v0 + 1 : self.v1],
            )
        return (
            grid.ID[:, self.u0 + 1 : self.u1, self.v0 + 1 : self.v1, p - 1],
            grid.ID[:, self.u0 + 1 : self.u1, self.v0 + 1 : self.v1, p],
        )

    def _component_cross_section(self):
        grid = self.main_grid
        p = self.plane_index
        if self.normal_axis == 0:
            return grid.ID[:, p, self.u0 : self.u1 + 1, self.v0 : self.v1 + 1]
        if self.normal_axis == 1:
            return grid.ID[:, self.u0 : self.u1 + 1, p, self.v0 : self.v1 + 1]
        return grid.ID[:, self.u0 : self.u1 + 1, self.v0 : self.v1 + 1, p]

    def _solid_cross_section(self):
        grid = self.main_grid
        # The detached side is represented by the auxiliary guide.
        cell = self.plane_index if self.direction_sign < 0 else self.plane_index - 1
        if self.normal_axis == 0:
            return grid.solid[cell, self.u0 : self.u1, self.v0 : self.v1]
        if self.normal_axis == 1:
            return grid.solid[self.u0 : self.u1, cell, self.v0 : self.v1]
        return grid.solid[self.u0 : self.u1, self.v0 : self.v1, cell]

    def _resolve_pml_profile(self):
        grid = self.main_grid
        if self.spec.profile_id is None:
            return grid.pmls["formulation"], copy.deepcopy(grid.pmls["cfs"])
        try:
            profile = grid.pmls["profiles"][self.spec.profile_id]
        except KeyError as exc:
            raise ValueError(
                f"Virtual waveguide on port {self.spec.port} refers to unknown "
                f"PML profile {self.spec.profile_id!r}."
            ) from exc
        if profile["formulation"] is None:
            raise ValueError(f"PML profile {self.spec.profile_id!r} has no formulation.")
        cfs = profile["cfs"] or copy.deepcopy(grid.pmls["cfs"])
        return profile["formulation"], copy.deepcopy(cfs)

    def _build_auxiliary_grid(self):
        main = self.main_grid
        aux = FDTDGrid()
        aux.name = f"virtual_waveguide_port_{self.spec.port}"
        aux.size[:] = 1
        aux.size[self.normal_axis] = self.spec.length_cells
        aux.size[self.transverse_axes[0]] = self.nu
        aux.size[self.transverse_axes[1]] = self.nv
        aux.dl[:] = main.dl
        aux.dt = main.dt
        aux.iterations = main.iterations
        aux.timewindow = main.timewindow
        aux.materials = main.materials

        formulation, cfs = self._resolve_pml_profile()
        aux.pmls["formulation"] = formulation
        aux.pmls["cfs"] = cfs
        thickness = [0] * 6
        face_offset = 3 if self.direction_sign < 0 else 0
        thickness[self.normal_axis + face_offset] = self.spec.pml_cells
        aux.set_pml_thickness(tuple(thickness))

        aux.initialise_geometry_arrays()
        solid = self._solid_cross_section()
        component_ids = self._component_cross_section()
        if self.normal_axis == 0:
            aux.solid[:] = solid[np.newaxis, :, :]
            aux.ID[:] = component_ids[:, np.newaxis, :, :]
        elif self.normal_axis == 1:
            aux.solid[:] = solid[:, np.newaxis, :]
            aux.ID[:] = component_ids[:, :, np.newaxis, :]
        else:
            aux.solid[:] = solid[:, :, np.newaxis]
            aux.ID[:] = component_ids[:, :, :, np.newaxis]

        aux._build_pmls()
        aux._terminate_pmls_with_pec()
        aux.initialise_field_arrays()
        aux.updatecoeffsE = np.array(main.updatecoeffsE, copy=True)
        aux.updatecoeffsH = np.array(main.updatecoeffsH, copy=True)
        if config.get_model_config().materials["maxpoles"] > 0:
            aux.initialise_dispersive_arrays()
            aux.updatecoeffsdispersive = np.array(main.updatecoeffsdispersive, copy=True)
        return aux

    def _build_auxiliary_source(self):
        if self.port not in self.main_grid.eigenmodesources:
            return None
        source = copy.copy(self.port)
        source.transverse_start = np.asarray((0, 0), dtype=np.int32)
        source.transverse_stop = np.asarray((self.nu, self.nv), dtype=np.int32)
        distance = self.spec.length_cells - self.spec.pml_cells - self.spec.source_clearance_cells
        source.plane_index = (
            distance if self.direction_sign < 0 else self.spec.length_cells - distance
        )
        source.port_monitor = None
        return source

    def update_magnetic(self, iteration):
        """Advance auxiliary H, apply modal injection, and join the aperture."""

        self.aux_updates.update_magnetic()
        self.aux_updates.update_magnetic_pml()
        if self.aux_source is not None:
            self.aux_source.update_eigenmode_magnetic(iteration, self.aux_grid)
        couple_virtual_waveguide_magnetic(
            config.get_model_config().ompthreads,
            self.normal_axis,
            self.direction_sign,
            self.u0,
            self.v0,
            self.u1,
            self.v1,
            self.plane_index,
            self.main_grid.Hx,
            self.main_grid.Hy,
            self.main_grid.Hz,
            self.aux_grid.Hx,
            self.aux_grid.Hy,
            self.aux_grid.Hz,
        )

    def update_electric(self, iteration):
        """Advance auxiliary E and close its curl with main-grid H."""

        self.aux_updates.update_electric_a()
        self.aux_updates.update_electric_pml()
        if self.aux_source is not None:
            self.aux_source.update_eigenmode_electric(iteration, self.aux_grid)
        self.aux_updates.update_electric_b()
        couple_virtual_waveguide_electric(
            config.get_model_config().ompthreads,
            self.normal_axis,
            self.direction_sign,
            self.u0,
            self.v0,
            self.u1,
            self.v1,
            self.plane_index,
            self.aux_grid.updatecoeffsE,
            self.aux_grid.ID,
            self.main_grid.Ex,
            self.main_grid.Ey,
            self.main_grid.Ez,
            self.main_grid.Hx,
            self.main_grid.Hy,
            self.main_grid.Hz,
            self.aux_grid.Ex,
            self.aux_grid.Ey,
            self.aux_grid.Ez,
            self.aux_grid.Hx,
            self.aux_grid.Hy,
            self.aux_grid.Hz,
        )


def port_plane_label(normal_axis, plane_index):
    return f"{'xyz'[normal_axis]}={plane_index}"


def initialise_virtual_waveguides(grid):
    """Construct deferred guides after modal bases and monitors are ready."""

    if not grid.virtual_waveguide_specs:
        return
    runtime_ports = {
        int(port.port_index): port for port in (*grid.eigenmodesources, *grid.eigenmodereceivers)
    }
    for port_number, spec in sorted(grid.virtual_waveguide_specs.items()):
        try:
            port = runtime_ports[port_number]
        except KeyError as exc:
            raise ValueError(
                f"Virtual waveguide references unknown eigenmode port {port_number}."
            ) from exc
        guide = VirtualWaveguide(grid, port, spec)
        grid.virtual_waveguides.append(guide)
        if port in grid.eigenmodesources:
            grid.eigenmodesources.remove(port)
        source_description = (
            f", source plane {guide.aux_source.plane_index}"
            if guide.aux_source is not None
            else ", passive"
        )
        logger.info(
            f"Virtual waveguide for eigenmode port {port_number}: "
            f"{spec.length_cells} cells long, {spec.pml_cells} PML cells"
            f"{source_description}."
        )
