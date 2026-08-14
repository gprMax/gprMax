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
    couple_virtual_waveguide_electric_aperture,
    couple_virtual_waveguide_magnetic,
)
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.materials import process_materials
from gprMax.subgrids.grid import SubGridBaseGrid
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
        self.mpi = hasattr(main_grid, "global_size")
        if self.mpi:
            self.plane_index = int(port.global_plane_index)
            self.u0, self.v0 = (int(value) for value in port.global_transverse_start)
            self.u1, self.v1 = (int(value) for value in port.global_transverse_stop)
        else:
            self.plane_index = int(port.plane_index)
            self.u0, self.v0 = (int(value) for value in port.transverse_start)
            self.u1, self.v1 = (int(value) for value in port.transverse_stop)
        self.nu = self.u1 - self.u0
        self.nv = self.v1 - self.v0
        self._mpi_materials = None
        self._mpi_adjacent_solid = None
        self._mpi_adjacent_ids = None
        self._mpi_component_ids = None
        self._mpi_solid_ids = None
        self._mpi_h_local = None
        self._mpi_h_global = None

        self._validate()
        self.aux_grid = self._build_auxiliary_grid()
        self.aux_source = self._build_auxiliary_source()
        if self.aux_source is not None:
            self.aux_grid.eigenmodesources.append(self.aux_source)
        self.aux_updates = (
            CPUUpdates(self.aux_grid) if config.sim_config.general["solver"] == "cpu" else None
        )

        # A virtual split makes only the main-domain side of the H plane a
        # valid sampling plane. This is already the source-monitor policy;
        # passive monitors need the same policy once their rear is detached.
        if self.port.port_monitor is not None:
            self.port.port_monitor.magnetic_side = 1

    def _validate(self):
        mode = config.get_model_config().mode
        if mode != "3D":
            raise ValueError("Virtual waveguides currently require a 3D model.")
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
        domain_size = np.asarray(
            getattr(self.main_grid, "global_size", self.main_grid.size), dtype=np.int32
        )
        normal_cells = int(domain_size[self.normal_axis])
        if not 1 <= self.plane_index < normal_cells:
            raise ValueError("A virtual-waveguide aperture must be an internal Yee plane.")
        if self.nu < 2 or self.nv < 2:
            raise ValueError(
                "A virtual-waveguide cross-section must be at least two cells "
                "along each transverse axis."
            )
        int32_max = np.iinfo(np.int32).max
        main_points = int(np.prod(np.asarray(domain_size, dtype=object) + 1))
        auxiliary_size = [self.nu, self.nv, self.spec.length_cells]
        auxiliary_points = int(np.prod(np.asarray(auxiliary_size, dtype=object) + 1))
        if max(main_points, auxiliary_points, (self.nu + 1) * (self.nv + 1)) > int32_max:
            raise ValueError("Virtual-waveguide device indexing exceeds the signed 32-bit range.")

        if self.mpi:
            self._prepare_mpi_cross_section()

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
        materials = self._mpi_materials if self.mpi else self.main_grid.materials
        dispersive = [
            material.ID
            for material in materials
            if material.numID in material_ids and getattr(material, "poles", 0) > 0
        ]
        if dispersive:
            raise ValueError(
                "Virtual-waveguide aperture coupling does not yet support "
                "dispersive guide materials; found " + ", ".join(dispersive) + "."
            )

    def _adjacent_solids(self):
        if self.mpi:
            return self._mpi_adjacent_solid
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
        if self.mpi:
            return self._mpi_adjacent_ids
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
        if self.mpi:
            return self._mpi_component_ids
        grid = self.main_grid
        p = self.plane_index
        if self.normal_axis == 0:
            return grid.ID[:, p, self.u0 : self.u1 + 1, self.v0 : self.v1 + 1]
        if self.normal_axis == 1:
            return grid.ID[:, self.u0 : self.u1 + 1, p, self.v0 : self.v1 + 1]
        return grid.ID[:, self.u0 : self.u1 + 1, self.v0 : self.v1 + 1, p]

    def _solid_cross_section(self):
        if self.mpi:
            return self._mpi_solid_ids
        grid = self.main_grid
        # The detached side is represented by the auxiliary guide.
        cell = self.plane_index if self.direction_sign < 0 else self.plane_index - 1
        if self.normal_axis == 0:
            return grid.solid[cell, self.u0 : self.u1, self.v0 : self.v1]
        if self.normal_axis == 1:
            return grid.solid[self.u0 : self.u1, cell, self.v0 : self.v1]
        return grid.solid[self.u0 : self.u1, self.v0 : self.v1, cell]

    def _prepare_mpi_cross_section(self):
        """Collect one global material cross-section on every rank.

        Numeric IDs for dielectric-smoothed materials are local to an MPI
        rank. Build a deterministic catalogue keyed by material name, then
        communicate the catalogue indices at the aperture. The auxiliary
        guide is consequently identical on every rank even when the modal
        plane crosses several partitions.
        """

        from mpi4py import MPI

        grid = self.main_grid
        local_catalogue = {material.ID: material for material in grid.materials}
        catalogues = grid.comm.allgather(local_catalogue)
        catalogue = {}
        for rank_catalogue in catalogues:
            for material_id, material in rank_catalogue.items():
                catalogue.setdefault(material_id, material)

        def material_sort_key(material_id):
            material = catalogue[material_id]
            return (
                bool(material.is_compound_material()),
                material.numID if not material.is_compound_material() else 0,
                material_id,
            )

        material_names = sorted(catalogue, key=material_sort_key)
        material_index = {name: index for index, name in enumerate(material_names)}
        self._mpi_materials = []
        for index, name in enumerate(material_names):
            material = copy.deepcopy(catalogue[name])
            material.numID = index
            self._mpi_materials.append(material)

        local_materials = {material.numID: material for material in grid.materials}

        def collect(component, normal_index, u_start, v_start, shape):
            local_values = np.zeros(shape, dtype=np.int64)
            local_count = np.zeros(shape, dtype=np.int8)
            for u in range(shape[0]):
                for v in range(shape[1]):
                    coordinate = np.zeros(3, dtype=np.int32)
                    coordinate[self.normal_axis] = normal_index
                    coordinate[self.transverse_axes[0]] = u_start + u
                    coordinate[self.transverse_axes[1]] = v_start + v
                    if grid.get_rank_from_coordinate(coordinate) != grid.rank:
                        continue
                    local_coordinate = grid.global_to_local_coordinate(coordinate)
                    if component is None:
                        numeric_id = int(grid.solid[tuple(local_coordinate)])
                    else:
                        numeric_id = int(grid.ID[(component, *local_coordinate)])
                    name = local_materials[numeric_id].ID
                    local_values[u, v] = material_index[name] + 1
                    local_count[u, v] = 1

            values = np.empty_like(local_values)
            count = np.empty_like(local_count)
            grid.comm.Allreduce(local_values, values, op=MPI.SUM)
            grid.comm.Allreduce(local_count, count, op=MPI.SUM)
            if np.any(count != 1):
                raise RuntimeError(
                    "MPI virtual-waveguide cross-section samples must have exactly one owner."
                )
            return (values - 1).astype(np.uint32)

        component_shape = (self.nu + 1, self.nv + 1)
        component_ids = np.empty((6, *component_shape), dtype=np.uint32)
        for component in range(6):
            component_ids[component] = collect(
                component, self.plane_index, self.u0, self.v0, component_shape
            )
        self._mpi_component_ids = component_ids

        interior_shape = (self.nu - 1, self.nv - 1)
        adjacent_ids = []
        for normal_index in (self.plane_index - 1, self.plane_index):
            ids = np.empty((6, *interior_shape), dtype=np.uint32)
            for component in range(6):
                ids[component] = collect(
                    component,
                    normal_index,
                    self.u0 + 1,
                    self.v0 + 1,
                    interior_shape,
                )
            adjacent_ids.append(ids)
        self._mpi_adjacent_ids = tuple(adjacent_ids)

        solid_shape = (self.nu, self.nv)
        self._mpi_adjacent_solid = tuple(
            collect(None, normal_index, self.u0, self.v0, solid_shape)
            for normal_index in (self.plane_index - 1, self.plane_index)
        )
        detached_cell = self.plane_index if self.direction_sign < 0 else self.plane_index - 1
        self._mpi_solid_ids = collect(None, detached_cell, self.u0, self.v0, solid_shape)

        h_points = self.nu * self.nv
        h_points += (self.nu + 1) * self.nv
        h_points += self.nu * (self.nv + 1)
        dtype = config.sim_config.dtypes["float_or_double"]
        self._mpi_h_local = np.zeros(h_points, dtype=dtype)
        self._mpi_h_global = np.zeros(h_points, dtype=dtype)

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
        # An HSG-owned guide follows the fine grid's numerical parameters but
        # is an independent, ordinary auxiliary Yee grid. Constructing a
        # second SubGridHSG would incorrectly require an HSG coupling region
        # and a parent coarse grid around a guide that is deliberately
        # detached from the physical domain.
        aux = FDTDGrid() if self.mpi or isinstance(main, SubGridBaseGrid) else type(main)()
        aux.name = f"virtual_waveguide_port_{self.spec.port}"
        aux.size[:] = 1
        aux.size[self.normal_axis] = self.spec.length_cells
        aux.size[self.transverse_axes[0]] = self.nu
        aux.size[self.transverse_axes[1]] = self.nv
        aux.dl[:] = main.dl
        aux.dt = main.dt
        aux.iterations = main.iterations
        aux.timewindow = main.timewindow
        aux.materials = copy.deepcopy(self._mpi_materials) if self.mpi else main.materials

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
        if self.mpi:
            aux.initialise_std_update_coeff_arrays()
            if config.get_model_config().materials["maxpoles"] > 0:
                aux.initialise_dispersive_arrays()
                aux.initialise_dispersive_update_coeff_array()
            process_materials(aux)
        else:
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
        source.global_plane_index = source.plane_index
        source.global_transverse_start = np.asarray((0, 0), dtype=np.int32)
        source.global_transverse_stop = np.asarray((self.nu, self.nv), dtype=np.int32)
        source.tfsf_owned_lower = np.zeros(3, dtype=np.int32)
        source.tfsf_owned_upper = np.asarray(self.aux_grid.size + 1, dtype=np.int32)
        source.port_monitor = None
        return source

    def initialise_device(self, parent_updates):
        """Create the auxiliary solver in the parent's accelerator context."""

        if self.aux_updates is not None:
            return
        self.aux_updates = type(parent_updates)(self.aux_grid, shared=parent_updates)
        solver = config.sim_config.general["solver"]
        if not hasattr(self.aux_grid, "updatecoeffsE_dev"):
            if solver == "cuda":
                self.aux_grid.htod_mat_coeff_arrays()
            elif solver == "opencl":
                self.aux_grid.htod_mat_coeff_arrays(parent_updates.queue)
            else:
                self.aux_grid.htod_material_arrays(parent_updates.dev)

    def _mpi_owned_global_bounds(self):
        lower = np.asarray(
            self.main_grid.lower_extent + self.main_grid.negative_halo_offset,
            dtype=np.int32,
        )
        upper = np.asarray(
            self.main_grid.lower_extent + self.main_grid.size,
            dtype=np.int32,
        )
        return lower, upper

    def _mpi_component_sheet(self, component, normal_index, u_points, v_points, output):
        """Pack the locally owned part of one global H sheet."""

        lower, upper = self._mpi_owned_global_bounds()
        if not lower[self.normal_axis] <= normal_index < upper[self.normal_axis]:
            return
        global_u0 = max(self.u0, int(lower[self.transverse_axes[0]]))
        global_u1 = min(self.u0 + u_points, int(upper[self.transverse_axes[0]]))
        global_v0 = max(self.v0, int(lower[self.transverse_axes[1]]))
        global_v1 = min(self.v0 + v_points, int(upper[self.transverse_axes[1]]))
        if global_u0 >= global_u1 or global_v0 >= global_v1:
            return

        local_slices = [slice(None)] * 3
        local_slices[self.normal_axis] = (
            normal_index - self.main_grid.lower_extent[self.normal_axis]
        )
        local_slices[self.transverse_axes[0]] = slice(
            global_u0 - self.main_grid.lower_extent[self.transverse_axes[0]],
            global_u1 - self.main_grid.lower_extent[self.transverse_axes[0]],
        )
        local_slices[self.transverse_axes[1]] = slice(
            global_v0 - self.main_grid.lower_extent[self.transverse_axes[1]],
            global_v1 - self.main_grid.lower_extent[self.transverse_axes[1]],
        )
        output[
            global_u0 - self.u0 : global_u1 - self.u0,
            global_v0 - self.v0 : global_v1 - self.v0,
        ] = (self.main_grid.Hx, self.main_grid.Hy, self.main_grid.Hz,)[component][
            tuple(local_slices)
        ]

    def _collect_mpi_aperture_magnetic_fields(self):
        """All-reduce the three H sheets needed by the aperture update."""

        from mpi4py import MPI

        normal_points = self.nu * self.nv
        u_points = (self.nu + 1) * self.nv
        normal = self._mpi_h_local[:normal_points].reshape(self.nu, self.nv)
        h_u = self._mpi_h_local[normal_points : normal_points + u_points].reshape(
            self.nu + 1, self.nv
        )
        h_v = self._mpi_h_local[normal_points + u_points :].reshape(self.nu, self.nv + 1)
        self._mpi_h_local.fill(0)
        self._mpi_component_sheet(self.normal_axis, self.plane_index, self.nu, self.nv, normal)
        cross_plane = self.plane_index - 1 if self.direction_sign < 0 else self.plane_index
        self._mpi_component_sheet(self.transverse_axes[0], cross_plane, self.nu + 1, self.nv, h_u)
        self._mpi_component_sheet(self.transverse_axes[1], cross_plane, self.nu, self.nv + 1, h_v)
        self.main_grid.comm.Allreduce(self._mpi_h_local, self._mpi_h_global, op=MPI.SUM)
        normal = self._mpi_h_global[:normal_points].reshape(self.nu, self.nv)
        h_u = self._mpi_h_global[normal_points : normal_points + u_points].reshape(
            self.nu + 1, self.nv
        )
        h_v = self._mpi_h_global[normal_points + u_points :].reshape(self.nu, self.nv + 1)
        return normal, h_u, h_v

    def _set_auxiliary_normal_magnetic(self, values):
        aperture = 0 if self.direction_sign < 0 else int(self.aux_grid.size[self.normal_axis])
        slices = [slice(None)] * 3
        slices[self.normal_axis] = aperture
        slices[self.transverse_axes[0]] = slice(0, self.nu)
        slices[self.transverse_axes[1]] = slice(0, self.nv)
        (self.aux_grid.Hx, self.aux_grid.Hy, self.aux_grid.Hz)[self.normal_axis][
            tuple(slices)
        ] = values

    def _write_mpi_component_sheet(self, component, normal_index, u_points, v_points, values):
        """Write an auxiliary aperture sheet to locally owned main fields."""

        lower, upper = self._mpi_owned_global_bounds()
        if not lower[self.normal_axis] <= normal_index < upper[self.normal_axis]:
            return
        global_u0 = max(self.u0, int(lower[self.transverse_axes[0]]))
        global_u1 = min(self.u0 + u_points, int(upper[self.transverse_axes[0]]))
        global_v0 = max(self.v0, int(lower[self.transverse_axes[1]]))
        global_v1 = min(self.v0 + v_points, int(upper[self.transverse_axes[1]]))
        if global_u0 >= global_u1 or global_v0 >= global_v1:
            return

        local_slices = [slice(None)] * 3
        local_slices[self.normal_axis] = (
            normal_index - self.main_grid.lower_extent[self.normal_axis]
        )
        local_slices[self.transverse_axes[0]] = slice(
            global_u0 - self.main_grid.lower_extent[self.transverse_axes[0]],
            global_u1 - self.main_grid.lower_extent[self.transverse_axes[0]],
        )
        local_slices[self.transverse_axes[1]] = slice(
            global_v0 - self.main_grid.lower_extent[self.transverse_axes[1]],
            global_v1 - self.main_grid.lower_extent[self.transverse_axes[1]],
        )
        (self.main_grid.Ex, self.main_grid.Ey, self.main_grid.Ez)[component][
            tuple(local_slices)
        ] = values[
            global_u0 - self.u0 : global_u1 - self.u0,
            global_v0 - self.v0 : global_v1 - self.v0,
        ]

    def _clear_mpi_component_box(
        self, fields, component, normal_start, normal_stop, u_points, v_points
    ):
        """Clear the owned intersection of a detached rear-field box."""

        lower, upper = self._mpi_owned_global_bounds()
        starts = np.zeros(3, dtype=np.int32)
        stops = np.zeros(3, dtype=np.int32)
        starts[self.normal_axis] = normal_start
        stops[self.normal_axis] = normal_stop
        starts[self.transverse_axes[0]] = self.u0
        stops[self.transverse_axes[0]] = self.u0 + u_points
        starts[self.transverse_axes[1]] = self.v0
        stops[self.transverse_axes[1]] = self.v0 + v_points
        starts = np.maximum(starts, lower)
        stops = np.minimum(stops, upper)
        if np.any(starts >= stops):
            return
        slices = tuple(
            slice(
                int(starts[axis] - self.main_grid.lower_extent[axis]),
                int(stops[axis] - self.main_grid.lower_extent[axis]),
            )
            for axis in range(3)
        )
        fields[component][slices] = 0

    def _clear_mpi_rear_magnetic(self):
        fields = (self.main_grid.Hx, self.main_grid.Hy, self.main_grid.Hz)
        domain_stop = int(self.main_grid.global_size[self.normal_axis]) + 1
        if self.direction_sign < 0:
            normal_start = (self.plane_index + 1, self.plane_index, self.plane_index)
            normal_stop = (domain_stop, domain_stop - 1, domain_stop - 1)
        else:
            normal_start = (0, 0, 0)
            normal_stop = (self.plane_index, self.plane_index, self.plane_index)

        component_points = {
            self.normal_axis: (self.nu, self.nv),
            self.transverse_axes[0]: (self.nu + 1, self.nv),
            self.transverse_axes[1]: (self.nu, self.nv + 1),
        }
        for component, (u_points, v_points) in component_points.items():
            index = (
                0
                if component == self.normal_axis
                else 1
                if component == self.transverse_axes[0]
                else 2
            )
            self._clear_mpi_component_box(
                fields,
                component,
                normal_start[index],
                normal_stop[index],
                u_points,
                v_points,
            )

    def _clear_mpi_rear_electric(self):
        fields = (self.main_grid.Ex, self.main_grid.Ey, self.main_grid.Ez)
        domain_stop = int(self.main_grid.global_size[self.normal_axis]) + 1
        if self.direction_sign < 0:
            normal_start = self.plane_index + 1
            normal_stops = (domain_stop, domain_stop, domain_stop)
        else:
            normal_start = 0
            normal_stops = (
                self.plane_index - 1,
                self.plane_index,
                self.plane_index,
            )
        component_points = {
            self.normal_axis: (self.nu + 1, self.nv + 1),
            self.transverse_axes[0]: (self.nu, self.nv + 1),
            self.transverse_axes[1]: (self.nu + 1, self.nv),
        }
        for component, (u_points, v_points) in component_points.items():
            index = (
                0
                if component == self.normal_axis
                else 1
                if component == self.transverse_axes[0]
                else 2
            )
            self._clear_mpi_component_box(
                fields,
                component,
                normal_start,
                normal_stops[index],
                u_points,
                v_points,
            )

    def _deposit_mpi_aperture_electric(self):
        aperture = 0 if self.direction_sign < 0 else int(self.aux_grid.size[self.normal_axis])
        inside = 0 if self.direction_sign < 0 else aperture - 1
        aux_fields = (self.aux_grid.Ex, self.aux_grid.Ey, self.aux_grid.Ez)

        def aux_sheet(component, normal_index, u_points, v_points):
            slices = [slice(None)] * 3
            slices[self.normal_axis] = normal_index
            slices[self.transverse_axes[0]] = slice(0, u_points)
            slices[self.transverse_axes[1]] = slice(0, v_points)
            return aux_fields[component][tuple(slices)]

        self._write_mpi_component_sheet(
            self.transverse_axes[0],
            self.plane_index,
            self.nu,
            self.nv + 1,
            aux_sheet(self.transverse_axes[0], aperture, self.nu, self.nv + 1),
        )
        self._write_mpi_component_sheet(
            self.transverse_axes[1],
            self.plane_index,
            self.nu + 1,
            self.nv,
            aux_sheet(self.transverse_axes[1], aperture, self.nu + 1, self.nv),
        )
        main_normal_index = self.plane_index if self.direction_sign < 0 else self.plane_index - 1
        self._write_mpi_component_sheet(
            self.normal_axis,
            main_normal_index,
            self.nu + 1,
            self.nv + 1,
            aux_sheet(self.normal_axis, inside, self.nu + 1, self.nv + 1),
        )

    def update_magnetic(self, iteration):
        """Advance auxiliary H, apply modal injection, and join the aperture."""

        self.aux_updates.update_magnetic()
        self.aux_updates.update_magnetic_pml()
        self.aux_updates.update_eigenmode_sources_magnetic(iteration)
        if self.mpi:
            self._clear_mpi_rear_magnetic()
            return
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

    def complete_magnetic_mpi(self):
        """Join the auxiliary H plane after the main MPI halo exchange."""

        if not self.mpi:
            return
        normal, _, _ = self._collect_mpi_aperture_magnetic_fields()
        self._set_auxiliary_normal_magnetic(normal)

    def update_electric(self, iteration):
        """Advance auxiliary E and close its curl with main-grid H."""

        self.aux_updates.update_electric_a()
        self.aux_updates.update_electric_pml()
        self.aux_updates.update_eigenmode_sources_electric(iteration)
        self.aux_updates.update_electric_b()
        if self.mpi:
            normal_points = self.nu * self.nv
            u_points = (self.nu + 1) * self.nv
            h_u = self._mpi_h_global[normal_points : normal_points + u_points].reshape(
                self.nu + 1, self.nv
            )
            h_v = self._mpi_h_global[normal_points + u_points :].reshape(self.nu, self.nv + 1)
            couple_virtual_waveguide_electric_aperture(
                config.get_model_config().ompthreads,
                self.normal_axis,
                self.direction_sign,
                self.aux_grid.updatecoeffsE,
                self.aux_grid.ID,
                h_u,
                h_v,
                self.aux_grid.Ex,
                self.aux_grid.Ey,
                self.aux_grid.Ez,
                self.aux_grid.Hx,
                self.aux_grid.Hy,
                self.aux_grid.Hz,
            )
            self._deposit_mpi_aperture_electric()
            self._clear_mpi_rear_electric()
            return
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
