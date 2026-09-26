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
from gprMax.modal_window import pec_electric_masks, sibc_window_pec_masks
from gprMax.mode2d import mode2d_geometry
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
        self.reduced = mode2d_geometry(config.get_model_config().mode)
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
        self._impedance_edges = ()
        self._impedance_window_edges = ()

        self._validate()
        self.aux_grid = self._build_auxiliary_grid()
        self._prepare_dispersive_aperture()
        if self._impedance_edges:
            self._prepare_impedance_edge_indices()
        if self._impedance_window_edges:
            self._freeze_impedance_main_rows()
        self.aux_source = self._build_auxiliary_source()
        self.aux_sources = [] if self.aux_source is None else [self.aux_source]
        if self.aux_source is not None:
            self.aux_grid.eigenmodesources.append(self.aux_source)
        self.aux_updates = (
            CPUUpdates(self.aux_grid) if config.sim_config.general["solver"] == "cpu" else None
        )
        if self.aux_updates is not None and self.aux_grid.maxpoles > 0:
            self.aux_updates.set_dispersive_updates()

        # A virtual split makes only the main-domain side of the H plane a
        # valid sampling plane. This is already the source-monitor policy;
        # passive monitors need the same policy once their rear is detached.
        if self.port.port_monitor is not None:
            self.port.port_monitor.magnetic_side = 1

    def _validate(self):
        if self.reduced:
            if self.mpi or config.sim_config.general["solver"] != "cpu":
                raise ValueError(
                    "2D virtual waveguides currently require the non-distributed CPU solver."
                )
            if self.port.invariant_axis != self.reduced.invariant_axis:
                raise ValueError("Virtual-waveguide port must use the model's invariant axis.")
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
            getattr(self.main_grid, "global_size", self.main_grid.size), dtype=object
        )
        normal_cells = int(domain_size[self.normal_axis])
        if not 1 <= self.plane_index < normal_cells:
            raise ValueError("A virtual-waveguide aperture must be an internal Yee plane.")
        transverse_sizes = dict(zip(self.transverse_axes, (self.nu, self.nv)))
        if any(
            size < 2
            for axis, size in transverse_sizes.items()
            if not self.reduced or axis != self.reduced.invariant_axis
        ):
            raise ValueError(
                "A virtual-waveguide cross-section must be at least two cells "
                "along each transverse axis."
            )
        int32_max = np.iinfo(np.int32).max
        auxiliary_size = [self.nu, self.nv, self.spec.length_cells]
        auxiliary_points = int(np.prod(np.asarray(auxiliary_size, dtype=object) + 1))
        device_counts = (
            auxiliary_points,
            (self.nu + 1) * (self.nv + 1),
            self._rear_clear_points(magnetic=True),
            self._rear_clear_points(magnetic=False),
        )
        if config.sim_config.general["solver"] != "cpu" and max(device_counts) > int32_max:
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
        if dispersive and (self.mpi or config.sim_config.general["solver"] != "cpu"):
            raise ValueError(
                "Dispersive virtual-waveguide aperture coupling requires the "
                "non-distributed CPU solver; found " + ", ".join(dispersive) + "."
            )
        self._validate_impedance_cross_section()

    def _validate_impedance_cross_section(self):
        """Limit sparse guide coupling to propagation-invariant surface impedance."""

        grid = self.main_grid
        system = getattr(grid, "impedance_surfaces", None)
        if system is None:
            return
        if self.mpi or config.sim_config.general["solver"] != "cpu":
            raise ValueError(
                "Surface-impedance virtual waveguides currently require the CPU solver."
            )
        selected, window_edges = [], []
        invariant_local = (
            None if not self.reduced else self.transverse_axes.index(self.reduced.invariant_axis)
        )
        window_pec = sibc_window_pec_masks(
            system,
            self.plane_index,
            self.transverse_axes,
            (self.u0, self.v0),
            (self.u1, self.v1),
            invariant_local,
        )
        rim = pec_electric_masks((self.nu, self.nv), invariant_local)
        local_axes = (*self.transverse_axes, self.normal_axis)
        for index, edge in enumerate(system.edge_info):
            coordinate = edge[1:4]
            if coordinate[self.normal_axis] != self.plane_index:
                continue
            u, v = coordinate[list(self.transverse_axes)]
            mask = window_pec[local_axes.index(int(edge[0]))]
            local_index = (int(u - self.u0), int(v - self.v0))
            if not all(0 <= value < size for value, size in zip(local_index, mask.shape)):
                continue
            window_edges.append(index)
            if mask[local_index]:
                # The dense guide/coupling kernels leave tangential rim E
                # at zero. Do not let a sparse ADE row overwrite that PEC.
                continue
            if rim[local_axes.index(int(edge[0]))][local_index]:
                raise ValueError(
                    "A physical SIBC wall on the virtual-waveguide rim needs opaque "
                    "padding beyond it for transverse PML coupling."
                )
            ports = slice(int(edge[6]), int(edge[6] + edge[7]))
            if np.any(system.port_normal[ports, 0] == self.normal_axis):
                raise ValueError(
                    "Surface-impedance virtual-waveguide walls must be propagation-invariant."
                )
            selected.append(index)

        self._impedance_window_edges = tuple(window_edges)
        self._impedance_edges = tuple(selected)
        if not selected:
            return
        for material_id in np.unique(self._solid_cross_section()):
            material = grid.materials[int(material_id)]
            if material.numID in grid.impedance_marker_models or material.is_pec:
                continue
            if (
                material.is_pmc
                or material.directional_materials is not None
                or not np.isfinite(material.se)
                or material.se < 0
                or not np.isfinite(material.sm)
                or material.sm < 0
                or not np.isfinite(material.er)
                or not np.isfinite(material.mr)
                or material.er <= 0
                or material.mr <= 0
            ):
                raise ValueError(
                    "Surface-impedance virtual waveguides require isotropic retained materials "
                    "with finite nonnegative conductivities and positive finite er and mr."
                )

    def _build_impedance_auxiliary_system(self, aux):
        """Extrude full dual-cell rows and independent ADE histories.

        Compiling a truncated auxiliary solid would halve the aperture mass.
        Translate the already compiled invariant main rows instead, retaining
        the same Yee operator on both sides of the split.
        """

        from gprMax.impedance_pml import prepare_impedance_pml
        from gprMax.impedance_surfaces import ImpedanceSurfaceSystem

        source = self.main_grid.impedance_surfaces
        normal = self.normal_axis
        offset = np.zeros(3, dtype=np.int32)
        offset[list(self.transverse_axes)] = (-self.u0, -self.v0)
        edge_info, original_rows, h_info, h_weight = [], [], [], []
        port_info, port_g, port_normal, port_area = [], [], [], []
        port_inv_Z0, port_g_over_Z0 = [], []
        state_count = 0
        aperture = 0 if self.direction_sign < 0 else self.spec.length_cells
        for original_index in self._impedance_edges:
            original = source.edge_info[original_index]
            if original[0] == normal:
                positions = range(self.spec.length_cells)
            else:
                positions = (
                    range(self.spec.length_cells)
                    if aperture == 0
                    else range(1, self.spec.length_cells + 1)
                )
            for position in positions:
                offset[normal] = position - self.plane_index
                row = original.copy()
                row[1:4] += offset
                row[4], row[6] = len(h_info), len(port_info)
                h_slice = slice(int(original[4]), int(original[4] + original[5]))
                translated = source.h_info[h_slice].copy()
                translated[:, 1:4] += offset
                # Tangential H at the allocated upper padding plane is not
                # advanced by the dense Yee kernels. It stores the main-side
                # cross-aperture samples, including the low-side -1 images.
                outside = translated[:, normal + 1] < 0
                translated[outside, normal + 1] = self.spec.length_cells
                invalid = np.any(translated[:, 1:4] < 0) or np.any(translated[:, 1:4] > aux.size)
                for axis in self.transverse_axes:
                    # H is node-sampled only along its own component axis;
                    # allocated upper padding on the other axes is not a DOF.
                    limit = aux.size[axis] + (translated[:, 0] == axis)
                    invalid = invalid or np.any(translated[:, axis + 1] >= limit)
                if invalid:
                    raise ValueError(
                        "Surface-impedance modal window omits a required magnetic sample."
                    )
                h_info.extend(translated)
                h_weight.extend(source.h_weight[h_slice])
                port_slice = slice(int(original[6]), int(original[6] + original[7]))
                for model_index, _ in source.port_info[port_slice]:
                    port_info.append((int(model_index), state_count))
                    state_count += int(source.model_info[model_index, 0])
                port_g.extend(source.port_g[port_slice])
                port_inv_Z0.extend(source.port_inv_Z0[port_slice])
                port_g_over_Z0.extend(source.port_g_over_Z0[port_slice])
                port_normal.extend(source.port_normal[port_slice])
                port_area.extend(source.port_area[port_slice])
                edge_info.append(row)
                original_rows.append(original_index)

        dtype = source.edge_runtime.dtype

        def packed(values, dtype=dtype):
            return np.ascontiguousarray(values, dtype=dtype)

        # Each extruded row owns an independent copy of every retained bulk
        # pole, just as it owns independent Foster surface-current histories.
        pole_offsets, pole_coeffs = [0], []
        if source.pole_coeffs.size:
            for original_index in original_rows:
                start, stop = source.pole_offsets[original_index : original_index + 2]
                pole_coeffs.extend(source.pole_coeffs[start:stop])
                pole_offsets.append(len(pole_coeffs))

        system = ImpedanceSurfaceSystem(
            edge_info=packed(edge_info, np.int32).reshape(-1, 8),
            edge_params=packed(source.edge_params[original_rows]),
            edge_runtime=packed(source.edge_runtime[original_rows]),
            edge_fraction=packed(source.edge_fraction[original_rows]),
            h_info=packed(h_info, np.int32).reshape(-1, 4),
            h_weight=packed(h_weight),
            port_info=packed(port_info, np.int32).reshape(-1, 2),
            port_g=packed(port_g),
            port_g_over_Z0=packed(port_g_over_Z0),
            port_inv_Z0=packed(port_inv_Z0),
            port_normal=packed(port_normal, np.int8).reshape(-1, 2),
            port_area=packed(port_area),
            model_info=source.model_info.copy(),
            model_f=source.model_f.copy(),
            model_q=source.model_q.copy(),
            model_Z0=source.model_Z0.copy(),
            state_y=np.zeros(max(1, state_count), dtype=dtype),
            model_ids=source.model_ids,
            edge_dispersion=packed(source.edge_dispersion[original_rows]) if pole_coeffs else None,
            pole_offsets=packed(pole_offsets, np.int32) if pole_coeffs else None,
            pole_coeffs=packed(pole_coeffs) if pole_coeffs else None,
        )
        aux.impedance_surfaces = system
        prepare_impedance_pml(aux, system)

    def _prepare_dispersive_aperture(self):
        """Index bulk ADE samples omitted by the auxiliary boundary kernels."""

        self._dispersive_aperture = []
        aux = self.aux_grid
        if not aux.maxpoles or self.mpi or config.sim_config.general["solver"] != "cpu":
            return
        aperture = 0 if self.direction_sign < 0 else self.spec.length_cells
        material_poles = np.asarray([getattr(material, "poles", 0) for material in aux.materials])
        for component in self.transverse_axes:
            if self.reduced and "E" + "xyz"[component] not in self.reduced.active_electric:
                continue
            ranges = []
            for axis in range(3):
                if axis == self.normal_axis:
                    values = [aperture]
                elif self.reduced and axis == self.reduced.invariant_axis:
                    values = [self.reduced.live_index]
                else:
                    values = np.arange(int(axis != component), aux.size[axis])
                ranges.append(values)
            coordinates = tuple(values.ravel() for values in np.meshgrid(*ranges, indexing="ij"))
            ids = aux.ID[(component,) + coordinates]
            selected = material_poles[ids] > 0
            if not np.any(selected):
                continue
            coordinates = tuple(values[selected] for values in coordinates)
            main_coordinates = [values.copy() for values in coordinates]
            main_coordinates[self.normal_axis][:] = self.plane_index
            main_coordinates[self.transverse_axes[0]] += self.u0
            main_coordinates[self.transverse_axes[1]] += self.v0
            ids = ids[selected]
            coefficients = aux.updatecoeffsdispersive[ids].T
            self._dispersive_aperture.append(
                (
                    component, coordinates, tuple(main_coordinates), aux.updatecoeffsE[ids, 4],
                    coefficients[0::3], coefficients[1::3], coefficients[2::3],
                )
            )

    def _complete_dispersive_aperture(self, old_fields):
        """Finish the same bulk A/B recurrence as an interior Yee sample.

        Aperture curl coupling already includes the instantaneous dispersive
        mass and conductivity. Add the old polarization load, advance its
        independent state with the final E, and publish the corrected field.
        Sparse SIBC rows have pole-free hold IDs and are advanced separately.
        """

        aux, main = self.aux_grid, self.main_grid
        for data, old_e in zip(self._dispersive_aperture, old_fields):
            component, coordinates, main_coordinates, scale, a, f, b = data
            electric = getattr(aux, "E" + "xyz"[component])
            history = getattr(aux, "T" + "xyz"[component])
            history_indices = (slice(None),) + coordinates
            old_t = history[history_indices]
            phi = np.sum((a * old_t).real, axis=0)
            electric[coordinates] -= scale * phi
            # Match the native A-then-B arithmetic, including complex poles.
            history[history_indices] = (f * old_t + b * old_e) - b * electric[coordinates]
            getattr(main, "E" + "xyz"[component])[main_coordinates] = electric[coordinates]

    def _prepare_impedance_edge_indices(self):
        """Index auxiliary source and aperture rows for coupling and diagnostics."""

        grid = self.aux_grid
        system = grid.impedance_surfaces
        source_plane = self.spec.pml_cells + self.spec.source_clearance_cells
        if self.direction_sign < 0:
            source_plane = self.spec.length_cells - source_plane
        rows = system.edge_info
        self._impedance_source_edge_indices = np.flatnonzero(
            (rows[:, 0] != self.normal_axis) & (rows[:, 1 + self.normal_axis] == source_plane)
        ).astype(np.int32)
        aperture = 0 if self.direction_sign < 0 else self.spec.length_cells
        inside = 0 if self.direction_sign < 0 else aperture - 1
        self._impedance_deposit_indices = np.flatnonzero(
            rows[:, 1 + self.normal_axis]
            == np.where(rows[:, 0] == self.normal_axis, inside, aperture)
        )

    def _freeze_impedance_main_rows(self):
        """The auxiliary solver owns detached E, including the aperture sheet."""

        system = self.main_grid.impedance_surfaces
        frozen = list(getattr(system, "virtual_frozen_edges", ()))
        for edge in system.edge_info:
            component, i, j, k = (int(value) for value in edge[:4])
            coordinate = np.asarray((i, j, k))
            u, v = coordinate[list(self.transverse_axes)]
            if not (
                self.u0 <= u < self.u1 + int(component != self.transverse_axes[0])
                and self.v0 <= v < self.v1 + int(component != self.transverse_axes[1])
            ):
                continue
            position = coordinate[self.normal_axis]
            detached = (
                position >= self.plane_index
                if self.direction_sign < 0
                else position < self.plane_index
                or (position == self.plane_index and component != self.normal_axis)
            )
            if detached:
                frozen.append((component, i, j, k))
        system.virtual_frozen_edges = np.ascontiguousarray(frozen, dtype=np.int32).reshape(-1, 4)

    def _copy_impedance_aperture_magnetic(self):
        """Fill the unused upper tangential-H padding plane from main H."""

        main, aux = self.main_grid, self.aux_grid
        main_h, aux_h = (main.Hx, main.Hy, main.Hz), (aux.Hx, aux.Hy, aux.Hz)
        for component in self.transverse_axes:
            main_slice, aux_slice = [slice(None)] * 3, [slice(None)] * 3
            main_slice[self.normal_axis] = (
                self.plane_index - 1 if self.direction_sign < 0 else self.plane_index
            )
            main_slice[self.transverse_axes[0]] = slice(self.u0, self.u1 + 1)
            main_slice[self.transverse_axes[1]] = slice(self.v0, self.v1 + 1)
            aux_slice[self.normal_axis] = self.spec.length_cells
            aux_h[component][tuple(aux_slice)] = main_h[component][tuple(main_slice)]

    def _deposit_impedance_aperture_electric(self):
        """Publish newly advanced sparse aperture and adjacent normal E."""

        main_e = (self.main_grid.Ex, self.main_grid.Ey, self.main_grid.Ez)
        aux_e = (self.aux_grid.Ex, self.aux_grid.Ey, self.aux_grid.Ez)
        aperture = 0 if self.direction_sign < 0 else self.spec.length_cells
        for edge in self.aux_grid.impedance_surfaces.edge_info[self._impedance_deposit_indices]:
            component = int(edge[0])
            coordinate = edge[1:4]
            target = coordinate.copy()
            target[list(self.transverse_axes)] += (self.u0, self.v0)
            target[self.normal_axis] += self.plane_index - aperture
            main_e[component][tuple(target)] = aux_e[component][tuple(coordinate)]

    def _rear_clear_points(self, *, magnetic):
        """Return the compact accelerator dispatch size for the detached rear.

        The rear occupies only the aperture cross-section extruded from the
        split plane to the relevant main-domain boundary. Magnetic Yee
        components on the far normal boundary require one additional plane;
        the component-specific bounds remain in the device kernel.
        """

        domain_size = getattr(self.main_grid, "global_size", self.main_grid.size)
        normal_cells = int(domain_size[self.normal_axis])
        if self.direction_sign < 0:
            normal_points = normal_cells - self.plane_index + int(magnetic)
        else:
            normal_points = self.plane_index
        return normal_points * (self.nu + 1) * (self.nv + 1)

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
        if self.reduced:
            slices = [slice(None)] * 3
            for axis, lower, upper in zip(
                self.transverse_axes, (self.u0, self.v0), (self.u1, self.v1)
            ):
                slices[axis] = (
                    self.reduced.live_index
                    if axis == self.reduced.invariant_axis
                    else slice(lower + 1, upper)
                )
            components = [
                "xyz".index(name[1].lower()) + (3 if name[0] == "H" else 0)
                for name in (*self.reduced.active_electric, *self.reduced.active_magnetic)
            ]
            slices[self.normal_axis] = p - 1
            first = grid.ID[(components, *slices)]
            slices[self.normal_axis] = p
            return first, grid.ID[(components, *slices)]
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
        solver = config.sim_config.general["solver"]
        if self.mpi or solver == "cpu":
            AuxiliaryGrid = FDTDGrid
        elif solver == "cuda":
            from gprMax.grid.cuda_grid import CUDAGrid as AuxiliaryGrid
        elif solver == "opencl":
            from gprMax.grid.opencl_grid import OpenCLGrid as AuxiliaryGrid
        elif solver == "metal":
            from gprMax.grid.metal_grid import MetalGrid as AuxiliaryGrid
        else:
            raise ValueError(f"Unsupported virtual-waveguide backend: {solver}")
        aux = AuxiliaryGrid()
        aux.name = f"virtual_waveguide_port_{self.spec.port}"
        aux.size[:] = 1
        aux.size[self.normal_axis] = self.spec.length_cells
        aux.size[self.transverse_axes[0]] = self.nu
        aux.size[self.transverse_axes[1]] = self.nv
        aux.dl[:] = main.dl
        aux.dt = main.dt
        aux.iterations = main.iterations
        aux.timewindow = main.timewindow
        aux.materials = (
            copy.deepcopy(self._mpi_materials)
            if self.mpi
            else copy.deepcopy(main.materials)
            if self._impedance_edges
            else main.materials
        )
        if self._impedance_edges:
            aux.surface_impedance_models = dict(main.surface_impedance_models)
            aux.impedance_marker_models = dict(main.impedance_marker_models)
        # The detached guide uses the same material catalogue and arithmetic
        # requirements as its owning grid, but maintains independent state.
        aux.maxpoles = main.maxpoles
        aux.drudelorentz = main.drudelorentz
        aux.dispersivedtype = main.dispersivedtype
        aux.dispersiveCdtype = main.dispersiveCdtype
        aux.crealfunc = main.crealfunc

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

        try:
            aux._build_pmls()
        except ValueError as exc:
            raise ValueError(
                f"Virtual waveguide port {self.spec.port}, PML profile "
                f"{self.spec.profile_id!r}: {exc}"
            ) from exc
        if self.reduced:
            aux._2d_mode_grid_update()
        aux._terminate_pmls_with_pec()
        if self._impedance_edges:
            self._build_impedance_auxiliary_system(aux)
        aux.initialise_field_arrays()
        if self.mpi or self._impedance_edges:
            aux.initialise_std_update_coeff_arrays()
            if aux.maxpoles > 0:
                aux.initialise_dispersive_arrays()
                aux.initialise_dispersive_update_coeff_array()
            process_materials(aux)
        else:
            aux.updatecoeffsE = np.array(main.updatecoeffsE, copy=True)
            aux.updatecoeffsH = np.array(main.updatecoeffsH, copy=True)
            if aux.maxpoles > 0:
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

    def set_active_source(self, source):
        """Install a configured cached modal source in the auxiliary guide."""

        self.set_active_sources((source,))

    def set_active_sources(self, sources):
        """Install all configured modal drives belonging to this aperture."""

        sources = tuple(sources)
        if not sources:
            self.clear_active_source()
            return
        self.port = sources[0]
        distance = self.spec.length_cells - self.spec.pml_cells - self.spec.source_clearance_cells
        plane_index = distance if self.direction_sign < 0 else self.spec.length_cells - distance
        self.aux_sources = []
        for source in sources:
            aux_source = copy.copy(source)
            aux_source.transverse_start = np.asarray((0, 0), dtype=np.int32)
            aux_source.transverse_stop = np.asarray((self.nu, self.nv), dtype=np.int32)
            aux_source.plane_index = plane_index
            aux_source.global_plane_index = plane_index
            aux_source.global_transverse_start = np.asarray((0, 0), dtype=np.int32)
            aux_source.global_transverse_stop = np.asarray((self.nu, self.nv), dtype=np.int32)
            aux_source.tfsf_owned_lower = np.zeros(3, dtype=np.int32)
            aux_source.tfsf_owned_upper = np.asarray(self.aux_grid.size + 1, dtype=np.int32)
            aux_source.port_monitor = None
            self.aux_sources.append(aux_source)
        self.aux_source = self.aux_sources[0]
        self.aux_grid.eigenmodesources[:] = self.aux_sources

    def clear_active_source(self):
        """Leave this guide as a passive matched modal load."""

        self.aux_source = None
        self.aux_sources = []
        self.aux_grid.eigenmodesources.clear()

    def reset_run_state(self):
        """Clear auxiliary fields/PML history and stale accelerator bindings."""

        self.aux_grid.reset_fields()
        if config.sim_config.general["solver"] == "cpu":
            self.aux_updates = CPUUpdates(self.aux_grid)
            if self.aux_grid.maxpoles > 0:
                self.aux_updates.set_dispersive_updates()
        else:
            # Device update objects share their parent's context/queue.  A new
            # parent is created for every reused run, so the auxiliary object
            # must be rebound as well.
            self.aux_updates = None

    def initialise_device(self, parent_updates):
        """Create the auxiliary solver in the parent's accelerator context."""

        if self.aux_updates is not None:
            return
        solver = config.sim_config.general["solver"]
        # The auxiliary grid is an ordinary Yee grid even when its owner is
        # an HSG subgrid or orchestrator. Those specialised updater classes
        # have different constructors and time-stepping responsibilities.
        # Select the plain backend, retaining the owner's context/queue and
        # the auxiliary grid's own (possibly fine-grid) timestep.
        if solver == "cuda":
            from gprMax.updates.cuda_updates import CUDAUpdates as AuxiliaryUpdates
        elif solver == "opencl":
            from gprMax.updates.opencl_updates import OpenCLUpdates as AuxiliaryUpdates
        elif solver == "metal":
            from gprMax.updates.metal_updates import MetalUpdates as AuxiliaryUpdates
        else:
            raise ValueError(f"Unsupported virtual-waveguide device backend: {solver}")
        self.aux_updates = AuxiliaryUpdates(self.aux_grid, shared=parent_updates)
        # Always refresh these arrays: a geometry-reuse run creates a new
        # accelerator context/queue, while Python attributes from the former
        # context may still exist on the persistent auxiliary grid.
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
        if self.reduced:
            from gprMax.virtual_waveguide_2d import couple_magnetic

            couple_magnetic(self)
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
        old_aperture_fields = [
            getattr(self.aux_grid, "E" + "xyz"[component])[coordinates].copy()
            for component, coordinates, *_ in self._dispersive_aperture
        ]
        if self.reduced:
            from gprMax.virtual_waveguide_2d import couple_electric

            couple_electric(self)
        else:
            self._couple_electric_3d()
        self._complete_dispersive_aperture(old_aperture_fields)
        if self._impedance_edges:
            self._copy_impedance_aperture_magnetic()
            self.aux_updates.update_impedance_surfaces()
            self._deposit_impedance_aperture_electric()

    def _couple_electric_3d(self):
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
    all_runtime_sources = tuple(grid.eigenmodesources)
    runtime_ports = {int(monitor.port_index): monitor.owner for monitor in grid.eigenmodeports}
    for port_number, spec in sorted(grid.virtual_waveguide_specs.items()):
        try:
            port = runtime_ports[port_number]
        except KeyError as exc:
            raise ValueError(
                f"Virtual waveguide references unknown eigenmode port {port_number}."
            ) from exc
        guide = VirtualWaveguide(grid, port, spec)
        active_sources = [
            source for source in all_runtime_sources if int(source.port_index) == port_number
        ]
        if active_sources:
            guide.set_active_sources(active_sources)
        grid.virtual_waveguides.append(guide)
        grid.eigenmodesources[:] = [
            source for source in grid.eigenmodesources if int(source.port_index) != port_number
        ]
        source_description = (
            f", {len(guide.aux_sources)} source drive(s) at plane "
            f"{guide.aux_source.plane_index}"
            if guide.aux_source is not None
            else ", passive"
        )
        logger.info(
            f"Virtual waveguide for eigenmode port {port_number}: "
            f"{spec.length_cells} cells long, {spec.pml_cells} PML cells"
            f"{source_description}."
        )
