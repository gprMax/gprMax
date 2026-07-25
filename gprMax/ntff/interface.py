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

"""Compiled reusable-surface interface for KSIR field transformations."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.constants import c, epsilon_0, mu_0

import gprMax.config as config
from gprMax.ntff.closures import SymmetryCompletion, resolve_closure
from gprMax.ntff.conventions import (
    FORWARD_TRANSFORM_KERNEL,
    OUTGOING_GREEN_RADIAL_FACTOR,
    PHASOR_TIME_DEPENDENCE,
)
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
FAR_METRICS = ("radiation_intensity", "rcs")
TIME_ORIGINS = ("simulation", "first_arrival")
WINDOWS = ("rectangular", "hann")


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
            raise ValueError(f"unknown KSIR output {output!r}")
        for component in requested:
            if component not in dependencies:
                dependencies.append(component)
    return tuple(dependencies)


@dataclass(frozen=True)
class KSIRSurfaceSpec:
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
class KSIRTimeReceiverResult:
    """Time histories for one receiver or receiver array."""

    output_id: str
    points: npt.NDArray[np.floating]
    times: npt.NDArray[np.floating]
    time_origins: npt.NDArray[np.floating]
    valid_lengths: npt.NDArray[np.int64]
    fields: Mapping[str, npt.NDArray[np.floating]]
    coordinate_system: str
    time_origin: str
    spherical_coordinates: Optional[npt.NDArray[np.floating]]

    def point_times(self, point_index: int) -> npt.NDArray[np.floating]:
        length = int(self.valid_lengths[point_index])
        return self.time_origins[point_index] + self.times[:length]

    def point_field(self, output: str, point_index: int) -> npt.NDArray:
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
    range_normalized: bool = True


@dataclass(frozen=True)
class _CompiledSurface:
    spec: KSIRSurfaceSpec
    closure: object
    surfaces: Mapping[str, object]
    origin: npt.NDArray[np.floating]
    pml_limits: tuple[tuple[int, int], ...]


def _resolve_surface_closure(spec: KSIRSurfaceSpec, grid, real_dtype):
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


def surface_reference_origin(spec: KSIRSurfaceSpec, grid, real_dtype) -> npt.NDArray:
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
                        f"KSIR surface {surface_id!r} {surface.component} samples "
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
        raise ValueError("KSIR surface has no off-symmetry samples for material validation")
    unique = np.unique(np.concatenate(sampled_ids))
    if unique.size != 1:
        raise ValueError(f"KSIR surface straddles multiple material IDs: {unique.tolist()}")
    return int(unique[0])


def _background_properties(surfaces: Mapping, closure, grid):
    material_id = _surface_material_id(surfaces, closure, grid)
    material = next((item for item in grid.materials if item.numID == material_id), None)
    if material is None:
        raise ValueError(f"cannot resolve KSIR background material ID {material_id}")
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
            "KSIR requires a homogeneous, lossless, non-dispersive background; "
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


def validate_ksir_source_enclosure(grid) -> None:
    """Require every active KSIR monitor to enclose impressed sources.

    This is called after ``#src_steps`` has moved simple sources for the
    current model, so the check covers the actual source positions used by
    the solve rather than only their original scene coordinates.
    """

    source_groups = (
        ("voltagesources", "E"),
        ("hertziandipoles", "E"),
        ("magneticdipoles", "H"),
        ("transmissionlines", "E"),
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
        for collection_name, field_prefix in source_groups:
            for source in getattr(grid, collection_name, ()):
                component = f"{field_prefix}{source.polarisation}"
                position = (
                    np.asarray(source.coord, dtype=np.float64)
                    + np.asarray(COMPONENT_OFFSETS[component], dtype=np.float64)
                ) * np.asarray(grid.dl, dtype=np.float64)
                if not np.all((position > lower) & (position < upper)):
                    source_id = getattr(source, "ID", source.__class__.__name__)
                    offenders.append(
                        f"{source.__class__.__name__} {source_id!r} at "
                        f"({position[0]:g}, {position[1]:g}, {position[2]:g}) m"
                    )

        spacing = np.asarray(grid.dl, dtype=np.float64)
        for index, plane_wave in enumerate(getattr(grid, "discreteplanewaves", ())):
            corners = np.asarray(plane_wave.corners, dtype=np.float64)
            box_lower = corners[:3] * spacing
            box_upper = corners[3:] * spacing
            if not (np.all(lower < box_lower) and np.all(upper > box_upper)):
                offenders.append(
                    f"DiscretePlaneWave[{index}] TFSF box from "
                    f"{tuple(box_lower)} m to {tuple(box_upper)} m"
                )

        if offenders:
            details = "; ".join(offenders)
            raise ValueError(
                f"KSIR monitor {monitor.name!r} integration surface must "
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


class KSIRCompiledOutputs:
    """Own grouped monitors and expose per-command results/HDF5 output."""

    def __init__(self, grid, surfaces, transforms, time_requests, frequency_requests, far_requests):
        self.grid = grid
        self.surfaces = surfaces
        self.transforms = transforms
        self.time_requests = {item.key: item for item in time_requests}
        self.frequency_requests = {item.key: item for item in frequency_requests}
        self.far_requests = {item.key: item for item in far_requests}
        self.time_bindings = {}
        self.frequency_monitors = {}
        self._results = {}

    def result_for(self, key: str):
        if key not in self._results:
            if key in self.time_requests:
                self._results[key] = self._time_result(self.time_requests[key])
            elif key in self.frequency_requests:
                self._results[key] = self._frequency_result(self.frequency_requests[key])
            elif key in self.far_requests:
                self._results[key] = self._far_result(self.far_requests[key])
            else:
                raise KeyError(key)
        return self._results[key]

    def transform_monitor(self, transform_id: str):
        return self.frequency_monitors[transform_id]

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

    def _far_result(self, spec: KSIRFarFieldRequestSpec):
        monitor = self.frequency_monitors[spec.transform_id]
        transform = self.transforms[spec.transform_id]
        compiled_surface = self.surfaces[transform.surface_id]
        directions = _readonly(
            spherical_directions(spec.theta, spec.phi, degrees=True),
            monitor.real_dtype,
        )
        dependencies = component_dependencies(spec.outputs)
        cartesian = {}
        for component in dependencies:
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
            )
            cartesian[component] = values
        ordinary_outputs = [item for item in spec.outputs if item not in FAR_METRICS]
        fields = _project_frequency_fields(cartesian, ordinary_outputs, spec.theta, spec.phi)
        if any(item in spec.outputs for item in FAR_METRICS):
            electric = np.stack([cartesian[item] for item in ELECTRIC_COMPONENTS], axis=-1)
            spherical = project_cartesian_to_spherical(electric, spec.theta, spec.phi, degrees=True)
            tangential_squared = np.abs(spherical[:, :, 1]) ** 2 + np.abs(spherical[:, :, 2]) ** 2
            if "radiation_intensity" in spec.outputs:
                fields["radiation_intensity"] = _readonly(
                    0.5 * tangential_squared / monitor.impedance,
                    monitor.real_dtype,
                )
            if "rcs" in spec.outputs:
                incident = monitor.result.incident_electric
                if incident is None:
                    raise RuntimeError(
                        "RCS output requires a KSIR surface enclosing one TFSF plane wave"
                    )
                incident_power = np.sum(np.abs(incident) ** 2, axis=1)
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
        )

    def _write_surface_metadata(self, group, compiled: _CompiledSurface):
        first = next(iter(compiled.surfaces.values()))
        group.attrs["formulation"] = "KSIR"
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
            group = (
                base_group[f"ntff/{spec.surface_id}"]
                .require_group("time")
                .create_group(spec.output_id)
            )
            group.attrs["coordinate_system"] = spec.coordinate_system
            group.attrs["time_origin"] = spec.time_origin
            group.attrs["outputs"] = np.asarray(spec.outputs, dtype="S20")
            group["points"] = result.points
            group["times"] = result.times
            group["time_origins"] = result.time_origins
            group["valid_lengths"] = result.valid_lengths
            if result.spherical_coordinates is not None:
                group["spherical_coordinates"] = result.spherical_coordinates
            self._write_fields(group, result)

        for transform_id, transform in self.transforms.items():
            monitor = self.frequency_monitors[transform_id]
            group = (
                base_group[f"ntff/{transform.surface_id}"]
                .require_group("frequency")
                .create_group(transform_id)
            )
            group.attrs["window"] = transform.window
            group.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
            group.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
            group.attrs["green_radial_sign"] = OUTGOING_GREEN_RADIAL_FACTOR
            group.attrs["wave_speed"] = monitor.wave_speed
            group.attrs["impedance"] = monitor.impedance
            group.attrs["precision"] = monitor.precision
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
            self._write_fields(group, result)


def _associate_plane_wave(monitor, surfaces, lower, upper, grid, requested_index):
    if requested_index is not None:
        if not isinstance(requested_index, (int, np.integer)):
            raise ValueError("KSIR plane_wave_index must be an integer")
        if requested_index < 0 or requested_index >= len(grid.discreteplanewaves):
            raise ValueError("KSIR plane_wave_index is not valid")
        candidates = [(requested_index, grid.discreteplanewaves[requested_index])]
    else:
        candidates = []
        for index, plane_wave in enumerate(grid.discreteplanewaves):
            corners = np.asarray(plane_wave.corners)
            if np.all(lower < corners[:3]) and np.all(upper > corners[3:]):
                candidates.append((index, plane_wave))
        if len(candidates) > 1:
            raise ValueError(
                "KSIR surface encloses multiple plane waves; select plane_wave_index "
                "through the Python API"
            )
    if not candidates:
        if grid.discreteplanewaves:
            raise ValueError(
                "KSIR surface must enclose the TFSF box of every discrete plane-wave source"
            )
        return
    index, plane_wave = candidates[0]
    corners = np.asarray(plane_wave.corners)
    if not (np.all(lower < corners[:3]) and np.all(upper > corners[3:])):
        raise ValueError("KSIR surface must enclose the selected TFSF box")
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
                    f"KSIR {surface.component} {face.face_id} samples touch the "
                    "TFSF correction stencil; move the surface at least one cell away"
                )
    monitor.associate_plane_wave(plane_wave, grid.dl, index)


def compile_ksir_outputs(model, grid) -> Optional[KSIRCompiledOutputs]:
    """Compile declarative KSIR commands after Yee material construction."""

    surface_specs = getattr(grid, "ksir_surface_specs", {})
    transform_specs = getattr(grid, "ksir_transform_specs", {})
    time_requests = list(getattr(grid, "ksir_time_requests", ()))
    frequency_requests = list(getattr(grid, "ksir_frequency_requests", ()))
    far_requests = list(getattr(grid, "ksir_far_field_requests", ()))
    if not (
        surface_specs or transform_specs or time_requests or frequency_requests or far_requests
    ):
        return None
    # A surface is a reusable definition, not by itself an output request.
    if not (transform_specs or time_requests or frequency_requests or far_requests):
        return None
    if config.sim_config.mpi:
        raise ValueError("the reusable KSIR interface does not yet support MPI")
    if config.sim_config.general["solver"] != "cpu":
        raise ValueError("the reusable KSIR interface currently supports only the CPU solver")
    if config.get_model_config().mode != "3D":
        raise ValueError("the reusable KSIR interface currently supports only 3-D models")

    for transform in transform_specs.values():
        if transform.surface_id not in surface_specs:
            raise ValueError(
                f"KSIR transform {transform.transform_id!r} refers to unknown surface "
                f"{transform.surface_id!r}"
            )
    for request in time_requests:
        if request.surface_id not in surface_specs:
            raise ValueError(
                f"KSIR time receiver {request.output_id!r} refers to unknown surface "
                f"{request.surface_id!r}"
            )
    for request in frequency_requests + far_requests:
        if request.transform_id not in transform_specs:
            raise ValueError(
                f"KSIR output {request.output_id!r} refers to unknown transform "
                f"{request.transform_id!r}"
            )

    needed_surface_ids = {item.surface_id for item in transform_specs.values()}
    needed_surface_ids.update(item.surface_id for item in time_requests)
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

    writer = KSIRCompiledOutputs(
        grid,
        compiled_surfaces,
        transform_specs,
        time_requests,
        frequency_requests,
        far_requests,
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
            closure=compiled.closure,
        )
        monitor.managed_output = True
        grid.ntff_monitors.append(monitor)
        for request in requests:
            _, point_slice = writer.time_bindings[request.key]
            writer.time_bindings[request.key] = (monitor, point_slice)

    for transform_id, transform in transform_specs.items():
        compiled = compiled_surfaces[transform.surface_id]
        dependencies = []
        related = [item for item in frequency_requests if item.transform_id == transform_id] + [
            item for item in far_requests if item.transform_id == transform_id
        ]
        if not related:
            dependencies = list(COMPONENTS)
        else:
            for request in related:
                for component in component_dependencies(request.outputs):
                    if component not in dependencies:
                        dependencies.append(component)
        selected_surfaces = {item: compiled.surfaces[item] for item in dependencies}
        monitor = KSIRFrequencyDomainMonitor(
            f"_ksir_frequency_{transform_id}",
            selected_surfaces,
            transform.frequencies,
            (0.0,),
            (0.0,),
            grid.dt,
            grid.iterations,
            real_dtype=real_dtype,
            complex_dtype=config.sim_config.dtypes["complex"],
            nthreads=config.get_model_config().ompthreads,
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
    grid.ntff_output_writers.append(writer)
    return writer
