# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
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
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Compiled reusable-surface interface for NTFF field transformations."""

import logging
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.constants import c, epsilon_0, mu_0

import gprMax.config as config
from gprMax.ntff.antenna import (
    directivity_from_intensity,
    radiation_intensity,
    spherical_quadrature,
)
from gprMax.ntff.closures import SymmetryCompletion, resolve_closure
from gprMax.ntff.conventions import (
    FORWARD_TRANSFORM_KERNEL,
    OUTGOING_GREEN_RADIAL_FACTOR,
    PHASOR_TIME_DEPENDENCE,
)
from gprMax.ntff.equivalent_current_time import EquivalentCurrentTimeMonitor
from gprMax.ntff.equivalent_currents import evaluate_equivalent_current_far_zone
from gprMax.ntff.evaluator import (
    evaluate_exact_points_patches,
    project_cartesian_to_spherical,
    spherical_basis,
    spherical_directions,
)
from gprMax.ntff.frequency_domain import (
    KSIRFrequencyDomainMonitor,
    _evaluate_component_with_closure,
)
from gprMax.ntff.surfaces import COMPONENT_OFFSETS, COMPONENTS, build_component_surface
from gprMax.ntff.time_domain import KSIRTimeDomainMonitor
from gprMax.ports import evaluate_port_power_spectrum, model_port_ids, model_port_output_registry

logger = logging.getLogger(__name__)

ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")
MAGNETIC_COMPONENTS = ("Hx", "Hy", "Hz")
CARTESIAN_OUTPUTS = ELECTRIC_COMPONENTS + MAGNETIC_COMPONENTS
SPHERICAL_OUTPUTS = (
    "Er",
    "Etheta",
    "Ephi",
    "Hr",
    "Htheta",
    "Hphi",
)
DIRECTIVITY_OUTPUTS = ("directivity", "directivity_dbi")
GAIN_OUTPUTS = ("gain", "gain_dbi", "realized_gain", "realized_gain_dbi")
EFFICIENCY_OUTPUTS = ("radiation_efficiency", "total_efficiency")
PORT_METRICS = GAIN_OUTPUTS + EFFICIENCY_OUTPUTS
FAR_METRICS = (
    ("radiation_intensity", "rcs") + DIRECTIVITY_OUTPUTS + GAIN_OUTPUTS + EFFICIENCY_OUTPUTS
)
TIME_ORIGINS = ("simulation", "first_arrival")
WINDOWS = ("rectangular", "hann")
# Post-processing block sizes are derived per transform from this cache-sized
# working-set target. It is a performance bound, not a model-size limit.
FAR_ZONE_TARGET_WORKING_SET_BYTES = 32 * 1024 * 1024
MAX_FAR_ZONE_DIRECTION_BLOCK = 1024


def _readonly(values: npt.ArrayLike, dtype=None) -> npt.NDArray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def validate_identifier(kind: str, value: str) -> str:
    """Validate an ID that will become one HDF5 path component."""

    if (
        not isinstance(value, str)
        or not value.strip()
        or "/" in value
        or "\x00" in value
        or value in (".", "..")
    ):
        raise ValueError(f"{kind} must be a valid, non-empty HDF5 path component")
    return value


def component_dependencies(outputs: Sequence[str]) -> tuple[str, ...]:
    """Return Cartesian surface components required for public outputs."""

    dependencies = []
    for output in outputs:
        if output in CARTESIAN_OUTPUTS:
            requested = (output,)
        elif output in SPHERICAL_OUTPUTS:
            requested = ELECTRIC_COMPONENTS if output.startswith("E") else MAGNETIC_COMPONENTS
        elif output in FAR_METRICS:
            requested = ELECTRIC_COMPONENTS
        else:
            raise ValueError(f"unknown NTFF output {output!r}")
        for component in requested:
            if component not in dependencies:
                dependencies.append(component)
    return tuple(dependencies)


@dataclass(frozen=True)
class NTFFSurfaceSpec:
    surface_id: str
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]
    origin: Optional[tuple[float, float, float]] = None


@dataclass(frozen=True)
class KSIRFrequencyTransformSpec:
    surface_id: str
    transform_id: str
    frequencies: tuple[float, ...]
    window: str = "rectangular"
    save_surface_dft: bool = True
    plane_wave_index: Optional[int] = None

    @property
    def formulation(self) -> str:
        return "ksir"


@dataclass(frozen=True)
class NTFFFrequencyTransformSpec:
    """Conventional equivalent-current frequency transform."""

    surface_id: str
    transform_id: str
    frequencies: tuple[float, ...]
    window: str = "rectangular"
    save_surface_dft: bool = True
    plane_wave_index: Optional[int] = None

    @property
    def formulation(self) -> str:
        return "equivalent_current"


@dataclass(frozen=True)
class KSIRAntennaPortsSpec:
    transform_id: str
    port_ids: tuple[str, ...]


@dataclass(frozen=True)
class KSIRTimeRequestSpec:
    key: str
    surface_id: str
    output_id: str
    points: npt.NDArray[np.floating]
    outputs: tuple[str, ...]
    time_origin: str
    coordinate_system: str
    spherical_coordinates: Optional[npt.NDArray[np.floating]] = None


@dataclass(frozen=True)
class KSIRFrequencyRequestSpec:
    key: str
    transform_id: str
    output_id: str
    points: npt.NDArray[np.floating]
    outputs: tuple[str, ...]
    coordinate_system: str
    spherical_coordinates: Optional[npt.NDArray[np.floating]] = None


@dataclass(frozen=True)
class KSIRFarFieldRequestSpec:
    key: str
    transform_id: str
    output_id: str
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class NTFFTimeFarFieldRequestSpec:
    key: str
    surface_id: str
    output_id: str
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class KSIRTimeReceiverResult:
    """Time histories for one receiver or receiver array."""

    output_id: str
    points: npt.NDArray[np.floating]
    times: npt.NDArray[np.floating]
    time_origins: npt.NDArray[np.floating]
    valid_lengths: npt.NDArray[np.int64]
    fully_supported_lengths: npt.NDArray[np.int64]
    terminal_field_ratios: npt.NDArray[np.floating]
    terminal_decay_ok: npt.NDArray[np.bool_]
    terminal_decay_threshold: float
    terminal_decay_window_samples: int
    fields: Mapping[str, npt.NDArray[np.floating]]
    coordinate_system: str
    time_origin: str
    spherical_coordinates: Optional[npt.NDArray[np.floating]]

    def point_times(self, point_index: int) -> npt.NDArray[np.floating]:
        """Return physical times for the fully supported field trace."""

        length = int(self.fully_supported_lengths[point_index])
        return self.time_origins[point_index] + self.times[:length]

    def point_field(self, output: str, point_index: int) -> npt.NDArray:
        """Return the fully supported field trace for one point."""

        length = int(self.fully_supported_lengths[point_index])
        return self.fields[output][point_index, :length]

    def point_raw_times(self, point_index: int) -> npt.NDArray[np.floating]:
        """Return all stored times, including the partial retarded tail."""

        length = int(self.valid_lengths[point_index])
        return self.time_origins[point_index] + self.times[:length]

    def point_raw_field(self, output: str, point_index: int) -> npt.NDArray:
        """Return all stored bins, including the partial retarded tail."""

        length = int(self.valid_lengths[point_index])
        return self.fields[output][point_index, :length]


@dataclass(frozen=True)
class KSIRFrequencyReceiverResult:
    """Exact finite-distance frequency-domain fields."""

    output_id: str
    frequencies: npt.NDArray[np.floating]
    points: npt.NDArray[np.floating]
    fields: Mapping[str, npt.NDArray[np.complexfloating]]
    coordinate_system: str
    spherical_coordinates: Optional[npt.NDArray[np.floating]]
    range_normalized: bool = False


@dataclass(frozen=True)
class KSIRFarFieldResult:
    """Range-normalized frequency-domain far fields."""

    output_id: str
    frequencies: npt.NDArray[np.floating]
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    directions: npt.NDArray[np.floating]
    fields: Mapping[str, npt.NDArray]
    origin: npt.NDArray[np.floating]
    radiation_metrics: Optional["KSIRRadiationMetrics"]
    port_metrics: Optional["KSIRPortMetrics"]
    range_normalized: bool = True


@dataclass(frozen=True)
class NTFFTimeFarFieldResult:
    """Range-normalized 1997 equivalent-current time-domain far fields."""

    output_id: str
    times: npt.NDArray[np.floating]
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    directions: npt.NDArray[np.floating]
    fields: Mapping[str, npt.NDArray[np.floating]]
    terminal_field_ratios: npt.NDArray[np.floating]
    terminal_decay_ok: npt.NDArray[np.bool_]
    terminal_decay_threshold: float
    terminal_decay_window_samples: int
    range_normalized: bool = True


@dataclass(frozen=True)
class KSIRRadiationMetrics:
    """Full-sphere quantities shared by far-field cuts and points."""

    radiated_power: npt.NDArray[np.floating]
    maximum_directivity: npt.NDArray[np.floating]
    maximum_directivity_dbi: npt.NDArray[np.floating]
    maximum_theta: npt.NDArray[np.floating]
    maximum_phi: npt.NDArray[np.floating]
    theta_order: int
    phi_order: int
    enclosure_radius: float


def _refine_radiation_maximum(
    metrics: KSIRRadiationMetrics,
    intensity: npt.NDArray[np.floating],
    theta: npt.NDArray[np.floating],
    phi: npt.NDArray[np.floating],
) -> KSIRRadiationMetrics:
    """Include any more accurately sampled requested peak in the summary."""

    local_index = np.argmax(intensity, axis=1)
    local_intensity = intensity[np.arange(intensity.shape[0]), local_index]
    local_directivity, local_directivity_dbi = directivity_from_intensity(
        local_intensity[:, np.newaxis],
        metrics.radiated_power,
    )
    local_directivity = local_directivity[:, 0]
    local_directivity_dbi = local_directivity_dbi[:, 0]
    update = np.isfinite(local_directivity) & (local_directivity > metrics.maximum_directivity)
    if not np.any(update):
        return metrics

    maximum_directivity = np.array(metrics.maximum_directivity, copy=True)
    maximum_directivity_dbi = np.array(metrics.maximum_directivity_dbi, copy=True)
    maximum_theta = np.array(metrics.maximum_theta, copy=True)
    maximum_phi = np.array(metrics.maximum_phi, copy=True)
    maximum_directivity[update] = local_directivity[update]
    maximum_directivity_dbi[update] = local_directivity_dbi[update]
    maximum_theta[update] = theta[local_index[update]]
    maximum_phi[update] = phi[local_index[update]]
    return replace(
        metrics,
        maximum_directivity=_readonly(maximum_directivity),
        maximum_directivity_dbi=_readonly(maximum_directivity_dbi),
        maximum_theta=_readonly(maximum_theta),
        maximum_phi=_readonly(maximum_phi),
    )


@dataclass(frozen=True)
class KSIRPortMetrics:
    """Exact-frequency powers for one coherently excited antenna port set."""

    port_ids: tuple[str, ...]
    source_types: tuple[str, ...]
    reference_impedances: npt.NDArray[np.floating]
    incident_voltage_per_port: npt.NDArray[np.complexfloating]
    terminal_voltage_per_port: npt.NDArray[np.complexfloating]
    terminal_current_per_port: npt.NDArray[np.complexfloating]
    incident_power_per_port: npt.NDArray[np.floating]
    accepted_power_per_port: npt.NDArray[np.floating]
    incident_power: npt.NDArray[np.floating]
    accepted_power: npt.NDArray[np.floating]
    reflected_power: npt.NDArray[np.floating]
    incident_relative_db: npt.NDArray[np.floating]
    mesh_valid: npt.NDArray[np.bool_]
    terminal_valid: npt.NDArray[np.bool_]
    gain_valid: npt.NDArray[np.bool_]
    realized_gain_valid: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class _CompiledSurface:
    spec: NTFFSurfaceSpec
    closure: object
    surfaces: Mapping[str, object]
    origin: npt.NDArray[np.floating]
    pml_limits: tuple[tuple[int, int], ...]


def _resolve_surface_closure(spec: NTFFSurfaceSpec, grid, real_dtype):
    lower = np.asarray(spec.lower)
    upper = np.asarray(spec.upper)
    touches_symmetry = any(
        (face.endswith("0") and lower["xyz".index(face[0])] == 0)
        or (face.endswith("max") and upper["xyz".index(face[0])] == grid.size["xyz".index(face[0])])
        for face in grid.symmetry_boundaries
    )
    return resolve_closure(
        SymmetryCompletion() if touches_symmetry else "closed",
        grid.symmetry_boundaries,
        spec.lower,
        spec.upper,
        grid.size,
        grid.dl,
        real_dtype=real_dtype,
    )


def _completed_bounds(lower, upper, closure):
    lower = np.asarray(lower).copy()
    upper = np.asarray(upper).copy()
    for plane in closure.symmetry_planes:
        reflected_lower = 2 * plane.coordinate - upper[plane.axis]
        reflected_upper = 2 * plane.coordinate - lower[plane.axis]
        lower[plane.axis] = min(lower[plane.axis], reflected_lower)
        upper[plane.axis] = max(upper[plane.axis], reflected_upper)
    return lower, upper


def surface_reference_origin(spec: NTFFSurfaceSpec, grid, real_dtype) -> npt.NDArray:
    """Return the configured origin or centre of the completed surface."""

    if spec.origin is not None:
        return _readonly(spec.origin, real_dtype)
    closure = _resolve_surface_closure(spec, grid, real_dtype)
    spacing = np.asarray(grid.dl, dtype=real_dtype)
    lower, upper = _completed_bounds(
        np.asarray(spec.lower, dtype=real_dtype) * spacing,
        np.asarray(spec.upper, dtype=real_dtype) * spacing,
        closure,
    )
    return _readonly(0.5 * (lower + upper), real_dtype)


def _surface_pml_limits(grid) -> tuple[tuple[int, int], ...]:
    pml = grid.pmls["thickness"]
    return (
        (pml["x0"], grid.nx if pml["xmax"] == 0 else grid.nx - pml["xmax"] - 1),
        (pml["y0"], grid.ny if pml["ymax"] == 0 else grid.ny - pml["ymax"] - 1),
        (pml["z0"], grid.nz if pml["zmax"] == 0 else grid.nz - pml["zmax"] - 1),
    )


def _validate_pml_samples(surface_id: str, surfaces: Mapping, limits) -> None:
    for surface in surfaces.values():
        for face in surface.faces:
            samples = np.concatenate((face.inside_indices, face.outside_indices))
            for axis, (minimum, maximum) in enumerate(limits):
                if np.any(samples[:, axis] < minimum) or np.any(samples[:, axis] > maximum):
                    raise ValueError(
                        f"NTFF surface {surface_id!r} {surface.component} samples "
                        f"enter the PML on the {'xyz'[axis]}-axis"
                    )


def _surface_material_id(surfaces: Mapping, closure, grid) -> int:
    sampled_ids = []
    real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    for component, surface in surfaces.items():
        component_ids = grid.ID[grid.IDlookup[component]]
        for face in surface.faces:
            keep = np.ones(face.npatches, dtype=bool)
            for plane in closure.symmetry_planes:
                tolerance = (
                    16
                    * np.finfo(real_dtype).eps
                    * max(abs(plane.coordinate), surface.grid_spacing[plane.axis])
                )
                keep &= ~np.isclose(
                    face.patch_positions[:, plane.axis],
                    plane.coordinate,
                    rtol=0,
                    atol=tolerance,
                )
            if np.any(keep):
                sampled_ids.append(component_ids[tuple(face.inside_indices[keep].T)])
                sampled_ids.append(component_ids[tuple(face.outside_indices[keep].T)])
    if not sampled_ids:
        raise ValueError("NTFF surface has no off-symmetry samples for material validation")
    unique = np.unique(np.concatenate(sampled_ids))
    if unique.size != 1:
        raise ValueError(f"NTFF surface straddles multiple material IDs: {unique.tolist()}")
    return int(unique[0])


def _background_properties(surfaces: Mapping, closure, grid):
    material_id = _surface_material_id(surfaces, closure, grid)
    material = next((item for item in grid.materials if item.numID == material_id), None)
    if material is None:
        raise ValueError(f"cannot resolve NTFF background material ID {material_id}")
    if (
        not np.isfinite(material.er)
        or not np.isfinite(material.mr)
        or material.er <= 0
        or material.mr <= 0
        or material.se != 0
        or material.sm != 0
        or getattr(material, "poles", 0) != 0
    ):
        raise ValueError(
            "NTFF requires a homogeneous, lossless, non-dispersive background; "
            f"got {material.ID!r}"
        )
    wave_speed = c / np.sqrt(float(material.er) * float(material.mr))
    impedance = np.sqrt(mu_0 * float(material.mr) / (epsilon_0 * float(material.er)))
    return material_id, wave_speed, impedance


def _validate_external_points(points: npt.NDArray, surfaces: Mapping, closure) -> None:
    for component, surface in surfaces.items():
        lower = np.asarray(surface.physical_lower).copy()
        upper = np.asarray(surface.physical_upper).copy()
        for plane in closure.symmetry_planes:
            if plane.face.endswith("0"):
                lower[plane.axis] = plane.coordinate
            else:
                upper[plane.axis] = plane.coordinate

        corners = np.asarray(
            [
                (x, y, z)
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ],
            dtype=points.dtype,
        )
        image_positions = []
        for image in closure.component_images(component):
            positions, _ = image.transform(corners, np.zeros_like(corners))
            image_positions.append(positions)
        completed = np.concatenate(image_positions)
        lower = np.min(completed, axis=0)
        upper = np.max(completed, axis=0)
        scale = max(1.0, np.max(np.abs(lower)), np.max(np.abs(upper)))
        tolerance = 10 * np.finfo(points.dtype).eps * scale
        on_or_inside = np.all((points >= lower - tolerance) & (points <= upper + tolerance), axis=1)
        if np.any(on_or_inside):
            raise ValueError(
                "KSIR observation points must be strictly outside the completed "
                f"{component} surface"
            )


def _completed_logical_bounds(surface, closure):
    """Return the physical bounds of a closed or symmetry-completed box."""

    spacing = np.asarray(surface.grid_spacing, dtype=np.float64)
    lower = np.asarray(surface.lower, dtype=np.float64) * spacing
    upper = np.asarray(surface.upper, dtype=np.float64) * spacing
    return _completed_bounds(lower, upper, closure)


def _subgrid_outer_bounds(subgrid, main_grid, *, physical=False):
    """Return the HSG outer-surface bounds in the main-grid frame."""

    lower = np.asarray((subgrid.i0, subgrid.j0, subgrid.k0), dtype=np.float64)
    upper = np.asarray((subgrid.i1, subgrid.j1, subgrid.k1), dtype=np.float64)
    lower -= subgrid.is_os_sep
    upper += subgrid.is_os_sep
    if physical:
        spacing = np.asarray(main_grid.dl, dtype=np.float64)
        lower *= spacing
        upper *= spacing
    return lower, upper


def _boxes_overlap(lower_a, upper_a, lower_b, upper_b):
    """Return whether two closed axis-aligned boxes touch or overlap."""

    return bool(np.all(lower_a <= upper_b) and np.all(upper_a >= lower_b))


def validate_tfsf_subgrid_enclosure(model) -> None:
    """Prevent a main-grid TFSF correction surface cutting an HSG region.

    A subgrid may be wholly inside or wholly outside a TFSF box. If their
    extents touch or overlap, the TFSF box must strictly enclose the HSG
    outer surface, leaving its correction stencil on the main grid.
    """

    for index, plane_wave in enumerate(getattr(model.G, "discreteplanewaves", ())):
        corners = np.asarray(plane_wave.corners, dtype=np.float64)
        box_lower, box_upper = corners[:3], corners[3:]
        for subgrid in model.subgrids:
            outer_lower, outer_upper = _subgrid_outer_bounds(subgrid, model.G)
            if not _boxes_overlap(box_lower, box_upper, outer_lower, outer_upper):
                continue
            if not (np.all(box_lower < outer_lower) and np.all(box_upper > outer_upper)):
                raise ValueError(
                    f"DiscretePlaneWave[{index}] TFSF box must strictly enclose "
                    f"the complete outer coupling surface of subgrid {subgrid.name!r}, "
                    "or be disjoint from it."
                )


def _validate_ntff_subgrid_enclosure(model, compiled_surfaces) -> None:
    """Prevent a main-grid NTFF surface cutting an HSG coupling region."""

    for surface_id, compiled in compiled_surfaces.items():
        sample_surface = next(iter(compiled.surfaces.values()))
        lower, upper = _completed_logical_bounds(sample_surface, compiled.closure)
        for subgrid in model.subgrids:
            outer_lower, outer_upper = _subgrid_outer_bounds(subgrid, model.G, physical=True)
            if not _boxes_overlap(lower, upper, outer_lower, outer_upper):
                continue
            if not (np.all(lower < outer_lower) and np.all(upper > outer_upper)):
                raise ValueError(
                    f"NTFF surface {surface_id!r} must strictly enclose the "
                    f"complete outer coupling surface of subgrid {subgrid.name!r}, "
                    "or be disjoint from it."
                )


def validate_ntff_source_enclosure(model, grid) -> None:
    """Require every active NTFF monitor to enclose impressed sources.

    This is called after ``#src_steps`` has moved simple sources for the
    current model, so the check covers the actual source positions used by
    the solve rather than only their original scene coordinates.
    """

    source_groups = (
        ("voltagesources", "E"),
        ("hertziandipoles", "E"),
        ("magneticdipoles", "H"),
        ("transmissionlines", "E"),
        ("magneticfrillsources", "E"),
    )
    for monitor in grid.ntff_monitors:
        if getattr(monitor, "allow_external_sources", False):
            continue
        surfaces = getattr(monitor, "surfaces", None)
        if not surfaces:
            continue
        surface = next(iter(surfaces.values()))
        closure = monitor.closure
        lower, upper = _completed_logical_bounds(surface, closure)
        offenders = []
        for source_grid in (model.G, *model.subgrids):
            grid_name = "main grid" if source_grid is model.G else f"subgrid {source_grid.name!r}"
            for collection_name, field_prefix in source_groups:
                for source in getattr(source_grid, collection_name, ()):
                    component = f"{field_prefix}{source.polarisation}"
                    local_position = np.asarray(source.coord, dtype=np.float64) + np.asarray(
                        COMPONENT_OFFSETS[component], dtype=np.float64
                    )
                    if source_grid is model.G:
                        position = local_position * np.asarray(source_grid.dl, dtype=np.float64)
                    else:
                        position = np.asarray(
                            source_grid.local_to_global(local_position), dtype=np.float64
                        )
                    if not np.all((position > lower) & (position < upper)):
                        source_id = getattr(source, "ID", source.__class__.__name__)
                        offenders.append(
                            f"{source.__class__.__name__} {source_id!r} on {grid_name} at "
                            f"({position[0]:g}, {position[1]:g}, {position[2]:g}) m"
                        )

            spacing = np.asarray(source_grid.dl, dtype=np.float64)
            for index, plane_wave in enumerate(getattr(source_grid, "discreteplanewaves", ())):
                corners = np.asarray(plane_wave.corners, dtype=np.float64)
                if source_grid is model.G:
                    box_lower = corners[:3] * spacing
                    box_upper = corners[3:] * spacing
                else:
                    box_lower = source_grid.local_to_global(corners[:3])
                    box_upper = source_grid.local_to_global(corners[3:])
                if not (np.all(lower < box_lower) and np.all(upper > box_upper)):
                    offenders.append(
                        f"DiscretePlaneWave[{index}] on {grid_name} TFSF box from "
                        f"{tuple(box_lower)} m to {tuple(box_upper)} m"
                    )

            for index, source in enumerate(getattr(source_grid, "eigenmodesources", ())):
                transverse_axes = np.asarray(source.transverse_axes, dtype=np.intp)
                local_lower = np.zeros(3, dtype=np.float64)
                local_upper = np.zeros(3, dtype=np.float64)
                local_lower[source.normal_axis] = source.plane_index
                local_upper[source.normal_axis] = source.plane_index
                local_lower[transverse_axes] = source.transverse_start
                local_upper[transverse_axes] = source.transverse_stop
                if source_grid is model.G:
                    box_lower = local_lower * spacing
                    box_upper = local_upper * spacing
                else:
                    box_lower = np.asarray(
                        source_grid.local_to_global(local_lower), dtype=np.float64
                    )
                    box_upper = np.asarray(
                        source_grid.local_to_global(local_upper), dtype=np.float64
                    )
                if not (np.all(lower < box_lower) and np.all(upper > box_upper)):
                    offenders.append(
                        f"EigenmodeSource[{index}] on {grid_name} injection plane from "
                        f"{tuple(box_lower)} m to {tuple(box_upper)} m"
                    )

        if offenders:
            details = "; ".join(offenders)
            raise ValueError(
                f"NTFF monitor {monitor.name!r} integration surface must "
                f"strictly enclose every impressed source; outside or "
                f"boundary source(s): {details}."
            )


def _project_time_fields(
    cartesian: Mapping[str, npt.NDArray],
    outputs: Sequence[str],
    theta: npt.NDArray,
    phi: npt.NDArray,
) -> Mapping[str, npt.NDArray]:
    radial, polar, azimuthal = spherical_basis(theta, phi, degrees=True)
    basis = {"r": radial, "theta": polar, "phi": azimuthal}
    result = {}
    for output in outputs:
        if output in CARTESIAN_OUTPUTS:
            result[output] = _readonly(cartesian[output])
            continue
        prefix = output[0]
        components = ELECTRIC_COMPONENTS if prefix == "E" else MAGNETIC_COMPONENTS
        vector = np.stack([cartesian[item] for item in components], axis=-1)
        projected = np.sum(vector * basis[output[1:]][:, np.newaxis, :], axis=-1)
        result[output] = _readonly(projected)
    return MappingProxyType(result)


def _project_frequency_fields(
    cartesian: Mapping[str, npt.NDArray],
    outputs: Sequence[str],
    theta: npt.NDArray,
    phi: npt.NDArray,
) -> dict[str, npt.NDArray]:
    result = {}
    for output in outputs:
        if output in CARTESIAN_OUTPUTS:
            result[output] = _readonly(cartesian[output])
    for prefix, components in (("E", ELECTRIC_COMPONENTS), ("H", MAGNETIC_COMPONENTS)):
        requested = [item for item in outputs if item.startswith(prefix) and item not in components]
        if not requested:
            continue
        vector = np.stack([cartesian[item] for item in components], axis=-1)
        spherical = project_cartesian_to_spherical(vector, theta, phi, degrees=True)
        for output in requested:
            axis = {f"{prefix}r": 0, f"{prefix}theta": 1, f"{prefix}phi": 2}[output]
            result[output] = _readonly(spherical[:, :, axis])
    return result


class NTFFCompiledOutputs:
    """Own grouped monitors and expose per-command results/HDF5 output."""

    def __init__(
        self,
        model,
        grid,
        surfaces,
        transforms,
        time_requests,
        frequency_requests,
        far_requests,
        antenna_port_specs,
        time_far_requests=(),
    ):
        self.model = model
        self.grid = grid
        self.surfaces = surfaces
        self.transforms = transforms
        self.time_requests = {item.key: item for item in time_requests}
        self.frequency_requests = {item.key: item for item in frequency_requests}
        self.far_requests = {item.key: item for item in far_requests}
        self.time_far_requests = {item.key: item for item in time_far_requests}
        self.antenna_port_specs = dict(antenna_port_specs)
        self.time_bindings = {}
        self.time_far_bindings = {}
        self.frequency_monitors = {}
        self._results = {}
        self._radiation_cache = {}
        self._port_cache = {}

    def result_for(self, key: str):
        if key not in self._results:
            if key in self.time_requests:
                self._results[key] = self._time_result(self.time_requests[key])
            elif key in self.frequency_requests:
                self._results[key] = self._frequency_result(self.frequency_requests[key])
            elif key in self.far_requests:
                self._results[key] = self._far_result(self.far_requests[key])
            elif key in self.time_far_requests:
                self._results[key] = self._time_far_result(self.time_far_requests[key])
            else:
                raise KeyError(key)
        return self._results[key]

    def transform_monitor(self, transform_id: str):
        return self.frequency_monitors[transform_id]

    @staticmethod
    def _enclosure_radius(compiled: _CompiledSurface) -> float:
        radius = 0.0
        for component, surface in compiled.surfaces.items():
            for image in compiled.closure.component_images(component):
                positions, _ = image.transform(
                    surface.patch_positions,
                    surface.normals,
                )
                radius = max(
                    radius,
                    float(
                        np.max(
                            np.linalg.norm(
                                positions - compiled.origin[np.newaxis, :],
                                axis=1,
                            )
                        )
                    ),
                )
        if radius <= 0:
            raise RuntimeError("NTFF surface has no finite angular extent")
        return radius

    def _radiation_metrics(self, transform_id: str) -> KSIRRadiationMetrics:
        """Integrate a temporary full sphere without retaining its fields."""

        if transform_id in self._radiation_cache:
            return self._radiation_cache[transform_id]
        monitor = self.frequency_monitors[transform_id]
        transform = self.transforms[transform_id]
        compiled = self.surfaces[transform.surface_id]
        radius = self._enclosure_radius(compiled)
        maximum_wavenumber = (
            2 * np.pi * float(np.max(monitor.frequencies, initial=0)) / monitor.wave_speed
        )
        quadrature = spherical_quadrature(
            radius,
            maximum_wavenumber,
            monitor.real_dtype,
        )
        logger.info(
            f"NTFF transform {transform_id!r}: evaluating {quadrature.theta.size} "
            f"temporary full-sphere directions ({quadrature.theta_order} x "
            f"{quadrature.phi_order}) for radiation normalisation"
        )
        nfrequencies = monitor.frequencies.size
        radiated_power = np.zeros(nfrequencies, dtype=monitor.real_dtype)
        maximum_intensity = np.full(nfrequencies, -np.inf, dtype=monitor.real_dtype)
        maximum_theta = np.full(nfrequencies, np.nan, dtype=monitor.real_dtype)
        maximum_phi = np.full(nfrequencies, np.nan, dtype=monitor.real_dtype)

        complex_bytes = np.dtype(monitor.complex_dtype).itemsize
        real_bytes = np.dtype(monitor.real_dtype).itemsize
        bytes_per_direction = max(1, nfrequencies) * (8 * complex_bytes + 2 * real_bytes)
        direction_block_size = max(
            1,
            min(
                MAX_FAR_ZONE_DIRECTION_BLOCK,
                FAR_ZONE_TARGET_WORKING_SET_BYTES // bytes_per_direction,
            ),
        )
        for start in range(0, quadrature.theta.size, direction_block_size):
            stop = min(start + direction_block_size, quadrature.theta.size)
            theta = quadrature.theta[start:stop]
            phi = quadrature.phi[start:stop]
            directions = spherical_directions(theta, phi, degrees=True)
            cartesian = self._far_cartesian(transform_id, directions, ELECTRIC_COMPONENTS)
            intensity = radiation_intensity(
                np.stack(
                    [cartesian[component] for component in ELECTRIC_COMPONENTS],
                    axis=-1,
                ),
                theta,
                phi,
                monitor.impedance,
            )
            radiated_power += np.sum(
                intensity * quadrature.weights[np.newaxis, start:stop],
                axis=1,
            )
            local_index = np.argmax(intensity, axis=1)
            local_maximum = intensity[np.arange(nfrequencies), local_index]
            update = local_maximum > maximum_intensity
            maximum_intensity[update] = local_maximum[update]
            maximum_theta[update] = theta[local_index[update]]
            maximum_phi[update] = phi[local_index[update]]

        maximum_directivity, maximum_directivity_dbi = directivity_from_intensity(
            maximum_intensity[:, np.newaxis],
            radiated_power,
        )
        valid_pattern = np.isfinite(radiated_power) & (radiated_power > 0)
        maximum_theta[~valid_pattern] = np.nan
        maximum_phi[~valid_pattern] = np.nan
        result = KSIRRadiationMetrics(
            radiated_power=_readonly(radiated_power),
            maximum_directivity=_readonly(maximum_directivity[:, 0]),
            maximum_directivity_dbi=_readonly(maximum_directivity_dbi[:, 0]),
            maximum_theta=_readonly(maximum_theta),
            maximum_phi=_readonly(maximum_phi),
            theta_order=quadrature.theta_order,
            phi_order=quadrature.phi_order,
            enclosure_radius=quadrature.enclosure_radius,
        )
        self._radiation_cache[transform_id] = result
        return result

    def _port_metrics(self, transform_id: str) -> KSIRPortMetrics:
        if transform_id in self._port_cache:
            return self._port_cache[transform_id]
        if transform_id not in self.antenna_port_specs:
            raise RuntimeError(f"NTFF transform {transform_id!r} has no antenna-port association")
        spec = self.antenna_port_specs[transform_id]
        monitor = self.frequency_monitors[transform_id]
        registry = model_port_output_registry(self.model)
        missing = set(spec.port_ids) - set(registry)
        if missing:
            raise RuntimeError(f"NTFF antenna ports were not finalised: {sorted(missing)}")
        spectra = []
        for port_id in spec.port_ids:
            binding = registry[port_id]
            spectrum = evaluate_port_power_spectrum(
                binding.output,
                binding.grid,
                monitor.frequencies,
                window=self.transforms[transform_id].window,
            )
            spectra.append(replace(spectrum, port_id=port_id))
        incident_voltage_per_port = np.stack([item.incident_voltage for item in spectra])
        terminal_voltage_per_port = np.stack([item.terminal_voltage for item in spectra])
        terminal_current_per_port = np.stack([item.terminal_current for item in spectra])
        incident_per_port = np.stack([item.incident_power for item in spectra])
        accepted_per_port = np.stack([item.accepted_power for item in spectra])
        incident_power = np.sum(incident_per_port, axis=0, dtype=monitor.real_dtype)
        accepted_power = np.sum(accepted_per_port, axis=0, dtype=monitor.real_dtype)
        reflected_power = np.asarray(
            incident_power - accepted_power,
            dtype=monitor.real_dtype,
        )
        mesh_valid = np.logical_and.reduce([item.mesh_valid for item in spectra])
        terminal_valid = np.logical_and.reduce([item.terminal_valid for item in spectra])

        incident_relative_db = np.full(
            incident_power.shape,
            -np.inf,
            dtype=monitor.real_dtype,
        )
        incident_peak = float(np.max(incident_power, initial=0.0))
        if incident_peak > 0:
            nonzero = incident_power > 0
            incident_relative_db[nonzero] = np.asarray(
                10 * np.log10(incident_power[nonzero] / incident_peak),
                dtype=monitor.real_dtype,
            )
        source_valid = incident_relative_db >= -40
        scale = max(
            incident_peak,
            float(np.max(np.abs(accepted_power), initial=0.0)),
        )
        threshold = 64 * np.finfo(monitor.real_dtype).eps * scale
        common_valid = mesh_valid & terminal_valid & source_valid
        gain_valid = common_valid & (accepted_power > threshold)
        realized_gain_valid = common_valid & (incident_power > threshold)
        result = KSIRPortMetrics(
            port_ids=spec.port_ids,
            source_types=tuple(item.source_type for item in spectra),
            reference_impedances=_readonly(
                [item.reference_impedance for item in spectra],
                monitor.real_dtype,
            ),
            incident_voltage_per_port=_readonly(
                incident_voltage_per_port,
                monitor.complex_dtype,
            ),
            terminal_voltage_per_port=_readonly(
                terminal_voltage_per_port,
                monitor.complex_dtype,
            ),
            terminal_current_per_port=_readonly(
                terminal_current_per_port,
                monitor.complex_dtype,
            ),
            incident_power_per_port=_readonly(incident_per_port, monitor.real_dtype),
            accepted_power_per_port=_readonly(accepted_per_port, monitor.real_dtype),
            incident_power=_readonly(incident_power, monitor.real_dtype),
            accepted_power=_readonly(accepted_power, monitor.real_dtype),
            reflected_power=_readonly(reflected_power, monitor.real_dtype),
            incident_relative_db=_readonly(incident_relative_db, monitor.real_dtype),
            mesh_valid=_readonly(mesh_valid, bool),
            terminal_valid=_readonly(terminal_valid, bool),
            gain_valid=_readonly(gain_valid, bool),
            realized_gain_valid=_readonly(realized_gain_valid, bool),
        )
        self._port_cache[transform_id] = result
        return result

    def _time_result(self, spec: KSIRTimeRequestSpec) -> KSIRTimeReceiverResult:
        monitor, point_slice = self.time_bindings[spec.key]
        source = monitor.result
        cartesian = {component: values[point_slice] for component, values in source.fields.items()}
        if spec.coordinate_system == "spherical":
            coordinates = spec.spherical_coordinates
            fields = _project_time_fields(
                cartesian, spec.outputs, coordinates[:, 1], coordinates[:, 2]
            )
        else:
            fields = MappingProxyType(
                {output: _readonly(cartesian[output]) for output in spec.outputs}
            )
        return KSIRTimeReceiverResult(
            output_id=spec.output_id,
            points=spec.points,
            times=source.times,
            time_origins=_readonly(source.time_origins[point_slice]),
            valid_lengths=_readonly(source.valid_lengths[point_slice], np.int64),
            fully_supported_lengths=_readonly(
                source.fully_supported_lengths[point_slice], np.int64
            ),
            terminal_field_ratios=_readonly(source.terminal_field_ratios[point_slice]),
            terminal_decay_ok=_readonly(source.terminal_decay_ok[point_slice], bool),
            terminal_decay_threshold=source.terminal_decay_threshold,
            terminal_decay_window_samples=source.terminal_decay_window_samples,
            fields=fields,
            coordinate_system=spec.coordinate_system,
            time_origin=spec.time_origin,
            spherical_coordinates=spec.spherical_coordinates,
        )

    def _exact_cartesian(self, spec: KSIRFrequencyRequestSpec) -> dict:
        monitor = self.frequency_monitors[spec.transform_id]
        transform = self.transforms[spec.transform_id]
        compiled_surface = self.surfaces[transform.surface_id]
        dependencies = component_dependencies(spec.outputs)
        _validate_external_points(
            spec.points,
            {item: compiled_surface.surfaces[item] for item in dependencies},
            compiled_surface.closure,
        )
        values = {}
        for component in dependencies:
            data = monitor.surface_data[component]
            total = np.zeros(
                (monitor.frequencies.size, spec.points.shape[0]),
                dtype=monitor.complex_dtype,
            )
            for (
                _,
                positions,
                normals,
                areas,
                field,
                derivative,
            ) in compiled_surface.closure.transformed_faces(
                data.surface, data.field, data.normal_derivative
            ):
                total += evaluate_exact_points_patches(
                    positions,
                    normals,
                    areas,
                    monitor.frequencies,
                    spec.points,
                    field,
                    derivative,
                    wave_speed=monitor.wave_speed,
                )
            values[component] = _readonly(total)
        return values

    def _frequency_result(self, spec: KSIRFrequencyRequestSpec):
        monitor = self.frequency_monitors[spec.transform_id]
        cartesian = self._exact_cartesian(spec)
        if spec.coordinate_system == "spherical":
            coordinates = spec.spherical_coordinates
            fields = MappingProxyType(
                _project_frequency_fields(
                    cartesian, spec.outputs, coordinates[:, 1], coordinates[:, 2]
                )
            )
        else:
            fields = MappingProxyType(
                {output: _readonly(cartesian[output]) for output in spec.outputs}
            )
        return KSIRFrequencyReceiverResult(
            output_id=spec.output_id,
            frequencies=monitor.frequencies,
            points=spec.points,
            fields=fields,
            coordinate_system=spec.coordinate_system,
            spherical_coordinates=spec.spherical_coordinates,
        )

    def _far_cartesian(self, transform_id, directions, components):
        """Evaluate requested far-zone Cartesian components by formulation."""

        monitor = self.frequency_monitors[transform_id]
        transform = self.transforms[transform_id]
        compiled_surface = self.surfaces[transform.surface_id]
        if transform.formulation == "equivalent_current":
            electric = evaluate_equivalent_current_far_zone(
                monitor.surface_data,
                monitor.frequencies,
                directions,
                origin=compiled_surface.origin,
                wave_speed=monitor.wave_speed,
                impedance=monitor.impedance,
                nthreads=monitor.nthreads,
            )
            magnetic = np.asarray(
                np.cross(directions[np.newaxis, :, :], electric) / monitor.impedance,
                dtype=monitor.complex_dtype,
            )
            vectors = {"E": electric, "H": magnetic}
            return {
                component: _readonly(vectors[component[0]][:, :, "xyz".index(component[1].lower())])
                for component in components
            }

        result = {}
        for component in components:
            data = monitor.surface_data[component]
            values, _ = _evaluate_component_with_closure(
                data.surface,
                data.field,
                data.normal_derivative,
                compiled_surface.closure,
                monitor.frequencies,
                directions,
                monitor.wave_speed,
                compiled_surface.origin,
                monitor.nthreads,
            )
            result[component] = values
        return result

    def _time_far_result(self, spec: NTFFTimeFarFieldRequestSpec):
        monitor, direction_slice = self.time_far_bindings[spec.key]
        source = monitor.result
        electric_theta = source.fields["Etheta"][direction_slice]
        electric_phi = source.fields["Ephi"][direction_slice]
        theta_basis = monitor.theta_basis[direction_slice]
        phi_basis = monitor.phi_basis[direction_slice]
        electric = (
            electric_theta[:, :, np.newaxis] * theta_basis[:, np.newaxis, :]
            + electric_phi[:, :, np.newaxis] * phi_basis[:, np.newaxis, :]
        )
        magnetic_theta = -electric_phi / monitor.impedance
        magnetic_phi = electric_theta / monitor.impedance
        magnetic = (
            magnetic_theta[:, :, np.newaxis] * theta_basis[:, np.newaxis, :]
            + magnetic_phi[:, :, np.newaxis] * phi_basis[:, np.newaxis, :]
        )
        fields = {}
        for output in spec.outputs:
            if output == "Etheta":
                values = electric_theta
            elif output == "Ephi":
                values = electric_phi
            elif output == "Htheta":
                values = magnetic_theta
            elif output == "Hphi":
                values = magnetic_phi
            elif output in ("Er", "Hr"):
                values = np.zeros_like(electric_theta)
            else:
                vector = electric if output.startswith("E") else magnetic
                values = vector[:, :, "xyz".index(output[1].lower())]
            fields[output] = _readonly(values, monitor.real_dtype)
        return NTFFTimeFarFieldResult(
            output_id=spec.output_id,
            times=source.times,
            theta=spec.theta,
            phi=spec.phi,
            directions=source.directions[direction_slice],
            fields=MappingProxyType(fields),
            terminal_field_ratios=_readonly(
                source.terminal_field_ratios[direction_slice], monitor.real_dtype
            ),
            terminal_decay_ok=_readonly(source.terminal_decay_ok[direction_slice], bool),
            terminal_decay_threshold=source.terminal_decay_threshold,
            terminal_decay_window_samples=source.terminal_decay_window_samples,
        )

    def _far_result(self, spec: KSIRFarFieldRequestSpec):
        monitor = self.frequency_monitors[spec.transform_id]
        transform = self.transforms[spec.transform_id]
        compiled_surface = self.surfaces[transform.surface_id]
        directions = _readonly(
            spherical_directions(spec.theta, spec.phi, degrees=True),
            monitor.real_dtype,
        )
        dependencies = component_dependencies(spec.outputs)
        cartesian = self._far_cartesian(spec.transform_id, directions, dependencies)
        ordinary_outputs = [item for item in spec.outputs if item not in FAR_METRICS]
        fields = _project_frequency_fields(cartesian, ordinary_outputs, spec.theta, spec.phi)
        radiation_metrics = None
        port_metrics = None
        if any(item in spec.outputs for item in FAR_METRICS):
            electric = np.stack([cartesian[item] for item in ELECTRIC_COMPONENTS], axis=-1)
            intensity = radiation_intensity(
                electric,
                spec.theta,
                spec.phi,
                monitor.impedance,
            )
            if "radiation_intensity" in spec.outputs:
                fields["radiation_intensity"] = _readonly(intensity, monitor.real_dtype)
            if any(item in spec.outputs for item in DIRECTIVITY_OUTPUTS):
                radiation_metrics = self._radiation_metrics(spec.transform_id)
                directivity, directivity_dbi = directivity_from_intensity(
                    intensity,
                    radiation_metrics.radiated_power,
                )
                if "directivity" in spec.outputs:
                    fields["directivity"] = _readonly(directivity, monitor.real_dtype)
                if "directivity_dbi" in spec.outputs:
                    fields["directivity_dbi"] = _readonly(
                        directivity_dbi,
                        monitor.real_dtype,
                    )
            if any(item in spec.outputs for item in PORT_METRICS):
                port_metrics = self._port_metrics(spec.transform_id)
                if any(item in spec.outputs for item in GAIN_OUTPUTS):
                    gain, gain_dbi = directivity_from_intensity(
                        intensity,
                        port_metrics.accepted_power,
                    )
                    gain[~port_metrics.gain_valid] = np.nan
                    gain_dbi[~port_metrics.gain_valid] = np.nan
                    realized_gain, realized_gain_dbi = directivity_from_intensity(
                        intensity,
                        port_metrics.incident_power,
                    )
                    realized_gain[~port_metrics.realized_gain_valid] = np.nan
                    realized_gain_dbi[~port_metrics.realized_gain_valid] = np.nan
                    for name, values in (
                        ("gain", gain),
                        ("gain_dbi", gain_dbi),
                        ("realized_gain", realized_gain),
                        ("realized_gain_dbi", realized_gain_dbi),
                    ):
                        if name in spec.outputs:
                            fields[name] = _readonly(values, monitor.real_dtype)
                if any(item in spec.outputs for item in EFFICIENCY_OUTPUTS):
                    if radiation_metrics is None:
                        radiation_metrics = self._radiation_metrics(spec.transform_id)
                    radiation_efficiency = np.full(
                        monitor.frequencies.shape,
                        np.nan,
                        dtype=monitor.real_dtype,
                    )
                    total_efficiency = np.full_like(radiation_efficiency, np.nan)
                    radiation_efficiency[port_metrics.gain_valid] = (
                        radiation_metrics.radiated_power[port_metrics.gain_valid]
                        / port_metrics.accepted_power[port_metrics.gain_valid]
                    )
                    total_efficiency[port_metrics.realized_gain_valid] = (
                        radiation_metrics.radiated_power[port_metrics.realized_gain_valid]
                        / port_metrics.incident_power[port_metrics.realized_gain_valid]
                    )
                    efficiency_tolerance = max(
                        0.02,
                        512 * np.finfo(monitor.real_dtype).eps,
                    )
                    if np.any(
                        radiation_efficiency[np.isfinite(radiation_efficiency)]
                        > 1 + efficiency_tolerance
                    ):
                        logger.warning(
                            "NTFF radiation efficiency exceeds unity for output %s; "
                            "check the integration surface, time window, mesh, and "
                            "port definitions",
                            spec.output_id,
                        )
                    if "radiation_efficiency" in spec.outputs:
                        fields["radiation_efficiency"] = _readonly(radiation_efficiency)
                    if "total_efficiency" in spec.outputs:
                        fields["total_efficiency"] = _readonly(total_efficiency)
            if radiation_metrics is not None:
                radiation_metrics = _refine_radiation_maximum(
                    radiation_metrics,
                    intensity,
                    spec.theta,
                    spec.phi,
                )
            if "rcs" in spec.outputs:
                incident = monitor.result.incident_electric
                if incident is None:
                    raise RuntimeError(
                        "RCS output requires an NTFF surface enclosing one TFSF plane wave"
                    )
                incident_power = np.sum(np.abs(incident) ** 2, axis=1)
                tangential_squared = 2 * monitor.impedance * intensity
                rcs = np.full(tangential_squared.shape, np.nan, dtype=monitor.real_dtype)
                valid = incident_power > 0
                rcs[valid] = (
                    4 * np.pi * tangential_squared[valid] / incident_power[valid, np.newaxis]
                )
                fields["rcs"] = _readonly(rcs)
        return KSIRFarFieldResult(
            output_id=spec.output_id,
            frequencies=monitor.frequencies,
            theta=spec.theta,
            phi=spec.phi,
            directions=directions,
            fields=MappingProxyType(fields),
            origin=compiled_surface.origin,
            radiation_metrics=radiation_metrics,
            port_metrics=port_metrics,
        )

    def _write_surface_metadata(self, group, compiled: _CompiledSurface):
        first = next(iter(compiled.surfaces.values()))
        group.attrs["formulation"] = "shared_ntff_surface"
        group.attrs["logical_bounds"] = (*first.lower, *first.upper)
        group.attrs["physical_origin"] = compiled.origin
        group.attrs["closure"] = compiled.closure.name
        group.attrs["mathematically_closed"] = compiled.closure.mathematically_closed
        group.attrs["closure_exact"] = compiled.closure.exact
        group.attrs["omitted_faces"] = np.asarray(compiled.closure.omitted_faces, dtype="S5")
        group.attrs["symmetry_plane_faces"] = np.asarray(
            [plane.face for plane in compiled.closure.symmetry_planes], dtype="S5"
        )
        group.attrs["symmetry_plane_types"] = np.asarray(
            [plane.boundary_type for plane in compiled.closure.symmetry_planes],
            dtype="S3",
        )
        group.attrs["symmetry_plane_coordinates"] = np.asarray(
            [plane.coordinate for plane in compiled.closure.symmetry_planes],
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        group.attrs["symmetry_image_count"] = compiled.closure.image_count

    @staticmethod
    def _write_fields(group, result):
        field_group = group.create_group("fields")
        for output, values in result.fields.items():
            field_group[output] = values

    def write_hdf5(self, base_group) -> None:
        ntff_group = base_group.require_group("ntff")
        for surface_id, compiled in self.surfaces.items():
            if surface_id in ntff_group:
                raise ValueError(f"duplicate NTFF output group {surface_id!r}")
            surface_group = ntff_group.create_group(surface_id)
            self._write_surface_metadata(surface_group, compiled)

        for key, spec in self.time_requests.items():
            result = self.result_for(key)
            monitor, _ = self.time_bindings[key]
            group = (
                base_group[f"ntff/{spec.surface_id}"]
                .require_group("time")
                .create_group(spec.output_id)
            )
            group.attrs["coordinate_system"] = spec.coordinate_system
            group.attrs["time_origin"] = spec.time_origin
            group.attrs["outputs"] = np.asarray(spec.outputs, dtype="S20")
            group.attrs["solver"] = monitor.device_backend or "cpu"
            group.attrs["collection_backend"] = monitor.collection_backend
            group.attrs["terminal_decay_threshold"] = result.terminal_decay_threshold
            group.attrs["terminal_decay_window_samples"] = result.terminal_decay_window_samples
            group.attrs[
                "raw_tail_policy"
            ] = "stored_for_research_use; use fully_supported_lengths by default"
            group["points"] = result.points
            group["times"] = result.times
            group["time_origins"] = result.time_origins
            group["valid_lengths"] = result.valid_lengths
            group["fully_supported_lengths"] = result.fully_supported_lengths
            group["terminal_field_ratios"] = result.terminal_field_ratios
            group["terminal_decay_ok"] = result.terminal_decay_ok
            if result.spherical_coordinates is not None:
                group["spherical_coordinates"] = result.spherical_coordinates
            self._write_fields(group, result)

        for key, spec in self.time_far_requests.items():
            result = self.result_for(key)
            monitor, _ = self.time_far_bindings[key]
            group = (
                base_group[f"ntff/{spec.surface_id}"]
                .require_group("time_far_field")
                .create_group(spec.output_id)
            )
            group.attrs["formulation"] = "equivalent_current_1997"
            group.attrs["coordinate_system"] = "spherical"
            group.attrs["range_normalized"] = True
            group.attrs["normalization"] = "r * field at reduced time t - r/c"
            group.attrs["interpolation"] = "linear"
            group.attrs["solver"] = "cpu"
            group.attrs["collection_backend"] = monitor.collection_backend
            group.attrs["outputs"] = np.asarray(spec.outputs, dtype="S20")
            group.attrs["terminal_decay_threshold"] = result.terminal_decay_threshold
            group.attrs["terminal_decay_window_samples"] = result.terminal_decay_window_samples
            group.attrs[
                "retarded_window_policy"
            ] = "only bins supported by every integration-surface patch are stored"
            group["times"] = result.times
            group["theta"] = result.theta
            group["phi"] = result.phi
            group["directions"] = result.directions
            group["terminal_field_ratios"] = result.terminal_field_ratios
            group["terminal_decay_ok"] = result.terminal_decay_ok
            self._write_fields(group, result)

        for transform_id, transform in self.transforms.items():
            monitor = self.frequency_monitors[transform_id]
            group = (
                base_group[f"ntff/{transform.surface_id}"]
                .require_group("frequency")
                .create_group(transform_id)
            )
            group.attrs["window"] = transform.window
            group.attrs["formulation"] = transform.formulation
            group.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
            group.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
            group.attrs["green_radial_sign"] = OUTGOING_GREEN_RADIAL_FACTOR
            group.attrs["wave_speed"] = monitor.wave_speed
            group.attrs["impedance"] = monitor.impedance
            group.attrs["precision"] = monitor.precision
            group.attrs["solver"] = monitor.solver_backend
            group.attrs["collection_backend"] = monitor.collection_backend
            group["frequencies"] = monitor.frequencies
            if transform.save_surface_dft:
                dft_group = group.create_group("surface_dft")
                for component, data in monitor.surface_data.items():
                    component_group = dft_group.create_group(component)
                    component_group.attrs["compatibility_signature"] = data.compatibility_signature
                    component_group["field"] = data.field
                    component_group["normal_derivative"] = data.normal_derivative
                    component_group["patch_positions"] = data.surface.patch_positions
                    component_group["patch_normals"] = data.surface.normals
                    component_group["area_weights"] = data.surface.area_weights

        for key, spec in self.frequency_requests.items():
            result = self.result_for(key)
            transform = self.transforms[spec.transform_id]
            parent = base_group[
                f"ntff/{transform.surface_id}/frequency/{spec.transform_id}"
            ].require_group("receivers")
            group = parent.create_group(spec.output_id)
            group.attrs["coordinate_system"] = spec.coordinate_system
            group.attrs["range_normalized"] = False
            group.attrs["outputs"] = np.asarray(spec.outputs, dtype="S20")
            group["points"] = result.points
            if result.spherical_coordinates is not None:
                group["spherical_coordinates"] = result.spherical_coordinates
            self._write_fields(group, result)

        for key, spec in self.far_requests.items():
            result = self.result_for(key)
            transform = self.transforms[spec.transform_id]
            parent = base_group[
                f"ntff/{transform.surface_id}/frequency/{spec.transform_id}"
            ].require_group("far_field")
            group = parent.create_group(spec.output_id)
            group.attrs["coordinate_system"] = "spherical"
            group.attrs["range_normalized"] = True
            group.attrs["normalization"] = "r * exp(+j*k*r) * field"
            group.attrs["outputs"] = np.asarray(spec.outputs, dtype="S20")
            group["theta"] = result.theta
            group["phi"] = result.phi
            group["directions"] = result.directions
            if result.radiation_metrics is not None:
                metrics = result.radiation_metrics
                group.attrs["radiation_quadrature"] = "Gauss-Legendre theta, periodic phi"
                group.attrs["radiation_spectral_power_units"] = "W s^2"
                group.attrs["radiation_quadrature_theta_order"] = metrics.theta_order
                group.attrs["radiation_quadrature_phi_order"] = metrics.phi_order
                group.attrs[
                    "maximum_directivity_sampling"
                ] = "full-sphere quadrature plus requested directions"
                group.attrs["radiation_enclosure_radius"] = metrics.enclosure_radius
                group.attrs["radiation_enclosure_radius_units"] = "m"
                group["radiated_power"] = metrics.radiated_power
                group["maximum_directivity"] = metrics.maximum_directivity
                group["maximum_directivity_dbi"] = metrics.maximum_directivity_dbi
                group["maximum_directivity_theta"] = metrics.maximum_theta
                group["maximum_directivity_phi"] = metrics.maximum_phi
            if result.port_metrics is not None:
                metrics = result.port_metrics
                power_group = group.create_group("port_power")
                power_group.attrs[
                    "accepted_power_definition"
                ] = "sum(0.5*Re(Vterminal*conj(Iterminal)))"
                power_group.attrs["incident_power_definition"] = "sum(abs(Vincident)**2/(2*Z0))"
                power_group.attrs["incident_floor_db"] = -40.0
                power_group.attrs["voltage_spectrum_units"] = "V s"
                power_group.attrs["current_spectrum_units"] = "A s"
                power_group.attrs["spectral_power_units"] = "W s^2"
                power_group["port_ids"] = np.asarray(metrics.port_ids, dtype="S64")
                power_group["source_types"] = np.asarray(metrics.source_types, dtype="S40")
                power_group["reference_impedances"] = metrics.reference_impedances
                power_group["incident_voltage_per_port"] = metrics.incident_voltage_per_port
                power_group["terminal_voltage_per_port"] = metrics.terminal_voltage_per_port
                power_group["terminal_current_per_port"] = metrics.terminal_current_per_port
                power_group["incident_power_per_port"] = metrics.incident_power_per_port
                power_group["accepted_power_per_port"] = metrics.accepted_power_per_port
                power_group["incident_power"] = metrics.incident_power
                power_group["accepted_power"] = metrics.accepted_power
                power_group["reflected_power"] = metrics.reflected_power
                power_group["incident_relative_db"] = metrics.incident_relative_db
                power_group["mesh_valid"] = metrics.mesh_valid.astype(np.uint8)
                power_group["terminal_valid"] = metrics.terminal_valid.astype(np.uint8)
                power_group["gain_valid"] = metrics.gain_valid.astype(np.uint8)
                power_group["realized_gain_valid"] = metrics.realized_gain_valid.astype(np.uint8)
            self._write_fields(group, result)


def _associate_plane_wave(monitor, surfaces, lower, upper, grid, requested_index):
    if requested_index is not None:
        if not isinstance(requested_index, (int, np.integer)):
            raise ValueError("NTFF plane_wave_index must be an integer")
        if requested_index < 0 or requested_index >= len(grid.discreteplanewaves):
            raise ValueError("NTFF plane_wave_index is not valid")
        candidates = [(requested_index, grid.discreteplanewaves[requested_index])]
    else:
        candidates = []
        for index, plane_wave in enumerate(grid.discreteplanewaves):
            corners = np.asarray(plane_wave.corners)
            if np.all(lower < corners[:3]) and np.all(upper > corners[3:]):
                candidates.append((index, plane_wave))
        if len(candidates) > 1:
            raise ValueError(
                "NTFF surface encloses multiple plane waves; select plane_wave_index "
                "through the Python API"
            )
    if not candidates:
        if grid.discreteplanewaves:
            raise ValueError(
                "NTFF surface must enclose the TFSF box of every discrete plane-wave source"
            )
        return
    index, plane_wave = candidates[0]
    corners = np.asarray(plane_wave.corners)
    if not (np.all(lower < corners[:3]) and np.all(upper > corners[3:])):
        raise ValueError("NTFF surface must enclose the selected TFSF box")
    correction_lower = corners[:3] - 1
    correction_upper = corners[3:]
    for surface in surfaces.values():
        for face in surface.faces:
            samples = np.concatenate((face.inside_indices, face.outside_indices))
            coordinates = samples[:, face.normal_axis]
            if face.normal_sign < 0:
                clear = np.all(coordinates < correction_lower[face.normal_axis])
            else:
                clear = np.all(coordinates > correction_upper[face.normal_axis])
            if not clear:
                raise ValueError(
                    f"NTFF {surface.component} {face.face_id} samples touch the "
                    "TFSF correction stencil; move the surface at least one cell away"
                )
    monitor.associate_plane_wave(plane_wave, grid.dl, index)


def compile_ntff_outputs(model, grid) -> Optional[NTFFCompiledOutputs]:
    """Compile declarative NTFF commands after Yee material construction."""

    surface_specs = getattr(grid, "ntff_surface_specs", {})
    ksir_transform_specs = dict(getattr(grid, "ksir_transform_specs", {}))
    equivalent_transform_specs = dict(getattr(grid, "ntff_transform_specs", {}))
    duplicate_transform_ids = set(ksir_transform_specs) & set(equivalent_transform_specs)
    if duplicate_transform_ids:
        raise ValueError(
            f"NTFF transform IDs must be globally unique: {sorted(duplicate_transform_ids)}"
        )
    transform_specs = {**ksir_transform_specs, **equivalent_transform_specs}
    time_requests = list(getattr(grid, "ksir_time_requests", ()))
    frequency_requests = list(getattr(grid, "ksir_frequency_requests", ()))
    far_requests = list(getattr(grid, "ksir_far_field_requests", ()))
    far_requests.extend(getattr(grid, "ntff_far_field_requests", ()))
    time_far_requests = list(getattr(grid, "ntff_time_far_field_requests", ()))
    antenna_port_specs = dict(getattr(grid, "ksir_antenna_port_specs", {}))
    for transform_id, spec in getattr(grid, "ntff_antenna_port_specs", {}).items():
        if transform_id in antenna_port_specs:
            raise ValueError(
                f"NTFF transform {transform_id!r} has duplicate antenna-port associations"
            )
        antenna_port_specs[transform_id] = spec
    if not (
        surface_specs
        or transform_specs
        or time_requests
        or frequency_requests
        or far_requests
        or time_far_requests
    ):
        return None
    # A surface is a reusable definition, not by itself an output request.
    if not (
        transform_specs or time_requests or frequency_requests or far_requests or time_far_requests
    ):
        return None
    if config.sim_config.mpi:
        raise ValueError("the reusable NTFF interface does not yet support MPI")
    if config.sim_config.general["solver"] not in ("cpu", "cuda", "opencl", "metal"):
        raise ValueError(
            "the reusable NTFF interface supports CPU, CUDA, OpenCL, and Metal solvers"
        )
    if config.get_model_config().mode != "3D":
        raise ValueError("the reusable NTFF interface currently supports only 3-D models")
    if time_far_requests and config.sim_config.general["solver"] != "cpu":
        raise ValueError(
            "the 1997 equivalent-current time-domain transform currently supports "
            "the CPU solver; device-resident kernels are the next implementation stage"
        )

    for transform in transform_specs.values():
        if transform.surface_id not in surface_specs:
            raise ValueError(
                f"NTFF transform {transform.transform_id!r} refers to unknown surface "
                f"{transform.surface_id!r}"
            )
    for request in time_requests:
        if request.surface_id not in surface_specs:
            raise ValueError(
                f"KSIR time receiver {request.output_id!r} refers to unknown surface "
                f"{request.surface_id!r}"
            )
    for request in time_far_requests:
        if request.surface_id not in surface_specs:
            raise ValueError(
                f"NTFF time far field {request.output_id!r} refers to unknown surface "
                f"{request.surface_id!r}"
            )
    for request in frequency_requests + far_requests:
        if request.transform_id not in transform_specs:
            raise ValueError(
                f"NTFF output {request.output_id!r} refers to unknown transform "
                f"{request.transform_id!r}"
            )

    gain_transforms = {
        request.transform_id
        for request in far_requests
        if any(output in PORT_METRICS for output in request.outputs)
    }
    available_port_ids = model_port_ids(model) if antenna_port_specs else ()
    for transform_id, antenna_spec in antenna_port_specs.items():
        unknown = set(antenna_spec.port_ids) - set(available_port_ids)
        if unknown:
            raise ValueError(
                f"NTFF antenna-port group for transform {transform_id!r} refers "
                f"to unknown port IDs {sorted(unknown)}; available ports are "
                f"{list(available_port_ids)}"
            )
    for transform_id in gain_transforms:
        if transform_id not in antenna_port_specs:
            raise ValueError(
                f"NTFF transform {transform_id!r} requests gain or efficiency "
                "without an antenna-port association"
            )
        transform = transform_specs[transform_id]
        if transform.window != "rectangular":
            raise ValueError(
                f"NTFF transform {transform_id!r} requests gain with window "
                f"{transform.window!r}; antenna gain currently requires rectangular"
            )

        expected_ids = available_port_ids
        requested_ids = set(antenna_port_specs[transform_id].port_ids)
        missing = set(expected_ids) - requested_ids
        if missing:
            raise ValueError(
                f"NTFF antenna-port group for transform {transform_id!r} must "
                f"include every physical port; missing {sorted(missing)}"
            )
        monitored_voltage_sources = set()
        voltage_sources = []
        nonport_sources = []
        for source_grid in (model.G, *model.subgrids):
            monitored_voltage_sources.update(
                monitor.source for monitor in getattr(source_grid, "port_monitors", ())
            )
            voltage_sources.extend(getattr(source_grid, "voltagesources", ()))
            nonport_sources.extend(getattr(source_grid, "hertziandipoles", ()))
            nonport_sources.extend(getattr(source_grid, "magneticdipoles", ()))
            nonport_sources.extend(getattr(source_grid, "discreteplanewaves", ()))
            nonport_sources.extend(getattr(source_grid, "eigenmodesources", ()))
        unmonitored_voltage_sources = [
            source for source in voltage_sources if source not in monitored_voltage_sources
        ]
        if unmonitored_voltage_sources:
            raise ValueError(
                "antenna gain requires an #rx_port for every voltage source; "
                f"found {len(unmonitored_voltage_sources)} unmonitored source(s)"
            )

        def source_is_active(source):
            for name in ("waveformvalues_wholedt", "waveformvalues_halfdt"):
                values = getattr(source, name, None)
                if values is not None and np.any(np.asarray(values) != 0):
                    return True
            waveform = getattr(source, "waveform", None)
            return waveform is not None and getattr(waveform, "amp", 0) != 0

        active_nonport = [source for source in nonport_sources if source_is_active(source)]
        if active_nonport:
            raise ValueError(
                "antenna gain cannot be normalised while active non-port sources "
                f"contribute to the field; found {len(active_nonport)} source(s)"
            )

    needed_surface_ids = {item.surface_id for item in transform_specs.values()}
    needed_surface_ids.update(item.surface_id for item in time_requests)
    needed_surface_ids.update(item.surface_id for item in time_far_requests)
    real_dtype = config.sim_config.dtypes["float_or_double"]
    field_shape = tuple(int(value + 1) for value in grid.size)
    compiled_surfaces = {}
    for surface_id in needed_surface_ids:
        spec = surface_specs[surface_id]
        closure = _resolve_surface_closure(spec, grid, real_dtype)
        surfaces = MappingProxyType(
            {
                component: closure.apply_quadrature(
                    build_component_surface(
                        component,
                        spec.lower,
                        spec.upper,
                        grid.dl,
                        field_shape,
                        excluded_faces=closure.omitted_faces,
                        real_dtype=real_dtype,
                    )
                )
                for component in COMPONENTS
            }
        )
        limits = _surface_pml_limits(grid)
        _validate_pml_samples(surface_id, surfaces, limits)
        origin = surface_reference_origin(spec, grid, real_dtype)
        compiled_surfaces[surface_id] = _CompiledSurface(
            spec=spec,
            closure=closure,
            surfaces=surfaces,
            origin=origin,
            pml_limits=limits,
        )

    for transform in equivalent_transform_specs.values():
        closure = compiled_surfaces[transform.surface_id].closure
        if closure.image_count != 1 or closure.omitted_faces:
            raise ValueError(
                "equivalent-current NTFF currently requires all six physical faces; "
                "symmetry-completed surfaces will be enabled after primitive E/H "
                "image-parity validation"
            )
    for request in time_far_requests:
        closure = compiled_surfaces[request.surface_id].closure
        if closure.image_count != 1 or closure.omitted_faces:
            raise ValueError(
                "equivalent-current time-domain NTFF currently requires all six physical faces"
            )

    _validate_ntff_subgrid_enclosure(model, compiled_surfaces)

    writer = NTFFCompiledOutputs(
        model,
        grid,
        compiled_surfaces,
        transform_specs,
        time_requests,
        frequency_requests,
        far_requests,
        antenna_port_specs,
        time_far_requests,
    )

    groups = {}
    for request in time_requests:
        groups.setdefault((request.surface_id, request.time_origin), []).append(request)
    for group_index, ((surface_id, time_origin), requests) in enumerate(groups.items()):
        compiled = compiled_surfaces[surface_id]
        dependencies = []
        points = []
        offset = 0
        for request in requests:
            for component in component_dependencies(request.outputs):
                if component not in dependencies:
                    dependencies.append(component)
            points.append(request.points)
            stop = offset + request.points.shape[0]
            writer.time_bindings[request.key] = (None, slice(offset, stop))
            offset = stop
        point_array = _readonly(np.concatenate(points), real_dtype)
        selected_surfaces = {item: compiled.surfaces[item] for item in dependencies}
        _, wave_speed, _ = _background_properties(selected_surfaces, compiled.closure, grid)
        monitor = KSIRTimeDomainMonitor(
            f"_ksir_time_{surface_id}_{group_index}",
            selected_surfaces,
            point_array,
            grid.dt,
            grid.iterations,
            real_dtype=real_dtype,
            wave_speed=wave_speed,
            nthreads=config.get_model_config().ompthreads,
            time_origin=time_origin,
            device_backend=(
                config.sim_config.general["solver"]
                if config.sim_config.general["solver"] in ("cuda", "opencl", "metal")
                else None
            ),
            closure=compiled.closure,
        )
        monitor.managed_output = True
        grid.ntff_monitors.append(monitor)
        for request in requests:
            _, point_slice = writer.time_bindings[request.key]
            writer.time_bindings[request.key] = (monitor, point_slice)

    time_far_groups = {}
    for request in time_far_requests:
        time_far_groups.setdefault(request.surface_id, []).append(request)
    for group_index, (surface_id, requests) in enumerate(time_far_groups.items()):
        compiled = compiled_surfaces[surface_id]
        theta = []
        phi = []
        offset = 0
        for request in requests:
            theta.append(request.theta)
            phi.append(request.phi)
            stop = offset + request.theta.size
            writer.time_far_bindings[request.key] = (None, slice(offset, stop))
            offset = stop
        _, wave_speed, impedance = _background_properties(compiled.surfaces, compiled.closure, grid)
        monitor = EquivalentCurrentTimeMonitor(
            f"_ntff_time_far_{surface_id}_{group_index}",
            compiled.spec.lower,
            compiled.spec.upper,
            grid.dl,
            field_shape,
            grid.dt,
            grid.iterations,
            np.concatenate(theta),
            np.concatenate(phi),
            compiled.origin,
            real_dtype=real_dtype,
            wave_speed=wave_speed,
            impedance=impedance,
            nthreads=config.get_model_config().ompthreads,
        )
        monitor.surfaces = compiled.surfaces
        monitor.closure = compiled.closure
        grid.ntff_monitors.append(monitor)
        for request in requests:
            _, direction_slice = writer.time_far_bindings[request.key]
            writer.time_far_bindings[request.key] = (monitor, direction_slice)

    for transform_id, transform in transform_specs.items():
        compiled = compiled_surfaces[transform.surface_id]
        dependencies = []
        related = [item for item in frequency_requests if item.transform_id == transform_id] + [
            item for item in far_requests if item.transform_id == transform_id
        ]
        if transform.formulation == "equivalent_current" or not related:
            dependencies = list(COMPONENTS)
        else:
            for request in related:
                for component in component_dependencies(request.outputs):
                    if component not in dependencies:
                        dependencies.append(component)
        selected_surfaces = {item: compiled.surfaces[item] for item in dependencies}
        monitor = KSIRFrequencyDomainMonitor(
            f"_ntff_frequency_{transform_id}",
            selected_surfaces,
            transform.frequencies,
            (0.0,),
            (0.0,),
            grid.dt,
            grid.iterations,
            real_dtype=real_dtype,
            complex_dtype=config.sim_config.dtypes["complex"],
            nthreads=config.get_model_config().ompthreads,
            solver_backend=config.sim_config.general["solver"],
            origin=compiled.origin,
            window=transform.window,
            save_surface_dft=transform.save_surface_dft,
            exterior_index_bounds=compiled.pml_limits,
            closure=compiled.closure,
        )
        _associate_plane_wave(
            monitor,
            selected_surfaces,
            np.asarray(compiled.spec.lower),
            np.asarray(compiled.spec.upper),
            grid,
            transform.plane_wave_index,
        )
        monitor.managed_output = True
        grid.ntff_monitors.append(monitor)
        writer.frequency_monitors[transform_id] = monitor

    for owner in getattr(grid, "ksir_request_owners", {}).values():
        owner._compiled_outputs = writer
    for transform_id, owner in getattr(grid, "ksir_transform_owners", {}).items():
        owner._compiled_outputs = writer
    for owner in getattr(grid, "ntff_request_owners", {}).values():
        owner._compiled_outputs = writer
    for transform_id, owner in getattr(grid, "ntff_transform_owners", {}).items():
        owner._compiled_outputs = writer
    grid.ntff_output_writers.append(writer)
    return writer
