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

"""Solver-independent near-to-far-field transformation utilities."""

from .closures import (
    ExperimentalMask,
    ResolvedKSIRClosure,
    SymmetryCompletion,
    SymmetryImage,
    SymmetryPlane,
    closure_from_metadata,
    component_parity,
    resolve_closure,
)
from .conventions import (
    FORWARD_TRANSFORM_KERNEL,
    OUTGOING_GREEN_RADIAL_FACTOR,
    PHASOR_TIME_DEPENDENCE,
    engineering_dft,
)
from .evaluator import (
    evaluate_exact_points_patches,
    evaluate_far_zone,
    evaluate_far_zone_patches,
    project_cartesian_to_spherical,
    spherical_basis,
    spherical_directions,
    spherical_observation_points,
)
from .frequency_domain import (
    KSIRComponentPhasors,
    KSIRFrequencyDomainMonitor,
    KSIRFrequencyResult,
    KSIRSavedFarField,
    evaluate_saved_surface_dft,
    surface_compatibility_signature,
    validate_nyquist_frequencies,
)
from .surfaces import (
    COMPONENTS,
    COMPONENT_OFFSETS,
    FACES,
    KSIRComponentSurface,
    KSIRSurfaceFace,
    build_all_component_surfaces,
    build_component_surface,
)

__all__ = [
    "COMPONENTS",
    "COMPONENT_OFFSETS",
    "FACES",
    "FORWARD_TRANSFORM_KERNEL",
    "ExperimentalMask",
    "KSIRComponentSurface",
    "KSIRComponentPhasors",
    "KSIRFrequencyDomainMonitor",
    "KSIRFrequencyResult",
    "KSIRSavedFarField",
    "KSIRSurfaceFace",
    "OUTGOING_GREEN_RADIAL_FACTOR",
    "PHASOR_TIME_DEPENDENCE",
    "ResolvedKSIRClosure",
    "SymmetryCompletion",
    "SymmetryImage",
    "SymmetryPlane",
    "build_all_component_surfaces",
    "build_component_surface",
    "engineering_dft",
    "closure_from_metadata",
    "component_parity",
    "evaluate_exact_points_patches",
    "evaluate_far_zone",
    "evaluate_far_zone_patches",
    "evaluate_saved_surface_dft",
    "project_cartesian_to_spherical",
    "resolve_closure",
    "spherical_basis",
    "spherical_directions",
    "spherical_observation_points",
    "surface_compatibility_signature",
    "validate_nyquist_frequencies",
]
