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

"""Closed, symmetry-completed, and experimental KSIR surface policies."""

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .surfaces import COMPONENT_OFFSETS, FACES, KSIRComponentSurface


_FACE_AXIS_SIGN = {
    "x0": (0, -1),
    "xmax": (0, 1),
    "y0": (1, -1),
    "ymax": (1, 1),
    "z0": (2, -1),
    "zmax": (2, 1),
}
_COMPONENT_AXIS = {"x": 0, "y": 1, "z": 2}


def _face_tuple(name: str, faces: Sequence[str]) -> Tuple[str, ...]:
    values = tuple(faces)
    unknown = set(values) - set(FACES)
    if unknown:
        raise ValueError(f"{name} contains unknown faces: {sorted(unknown)}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(face for face in FACES if face in values)


@dataclass(frozen=True)
class SymmetryCompletion:
    """Request exact completion from resolved gprMax PEC/PMC boundaries.

    If faces is omitted, every declared symmetry boundary touched by the
    logical KSIR box is used.
    """

    faces: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.faces is not None:
            object.__setattr__(self, "faces", _face_tuple("faces", self.faces))


@dataclass(frozen=True)
class ExperimentalMask:
    """Deliberately omit complete physical faces from the KSIR integral."""

    omit_faces: Tuple[str, ...]

    def __post_init__(self):
        omitted = _face_tuple("omit_faces", self.omit_faces)
        if not omitted:
            raise ValueError("ExperimentalMask must omit at least one face")
        if len(omitted) == len(FACES):
            raise ValueError("ExperimentalMask must leave at least one active face")
        object.__setattr__(self, "omit_faces", omitted)


@dataclass(frozen=True)
class HuygensOpenSurface:
    """Select an open Huygens box by omitting complete physical faces."""

    omit_faces: Tuple[str, ...]

    def __post_init__(self):
        omitted = _face_tuple("omit_faces", self.omit_faces)
        if not omitted:
            raise ValueError("HuygensOpenSurface must omit at least one face")
        if len(omitted) == len(FACES):
            raise ValueError("HuygensOpenSurface must leave at least one active face")
        object.__setattr__(self, "omit_faces", omitted)


@dataclass(frozen=True)
class SymmetryPlane:
    """One resolved physical reflection plane."""

    face: str
    axis: int
    coordinate: float
    boundary_type: str

    def __post_init__(self):
        if self.face not in _FACE_AXIS_SIGN:
            raise ValueError(f"unknown symmetry face {self.face!r}")
        expected_axis, _ = _FACE_AXIS_SIGN[self.face]
        if self.axis != expected_axis:
            raise ValueError(
                f"symmetry face {self.face!r} must use axis {expected_axis}"
            )
        if not np.isfinite(self.coordinate):
            raise ValueError("symmetry-plane coordinate must be finite")
        if self.boundary_type not in ("pec", "pmc"):
            raise ValueError("symmetry boundary_type must be 'pec' or 'pmc'")


@dataclass(frozen=True)
class SymmetryImage:
    """One member of the finite reflection group."""

    planes: Tuple[SymmetryPlane, ...]
    parity: int

    def transform(
        self,
        positions: npt.NDArray[np.floating],
        normals: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        image_positions = positions.copy()
        image_normals = normals.copy()
        for plane in self.planes:
            image_positions[:, plane.axis] = (
                2 * plane.coordinate - image_positions[:, plane.axis]
            )
            image_normals[:, plane.axis] *= -1
        return image_positions, image_normals


@dataclass(frozen=True)
class ResolvedKSIRClosure:
    """Closure policy resolved against one built gprMax grid."""

    name: str
    omitted_faces: Tuple[str, ...]
    symmetry_planes: Tuple[SymmetryPlane, ...]
    mathematically_closed: bool
    exact: bool

    @property
    def active_faces(self) -> Tuple[str, ...]:
        return tuple(face for face in FACES if face not in self.omitted_faces)

    def material_validation_mask(
        self,
        surface: KSIRComponentSurface,
        face,
        real_dtype,
    ) -> npt.NDArray[np.bool_]:
        """Exclude active-face edge rows adjacent to omitted physical planes."""

        keep = np.ones(face.npatches, dtype=bool)
        symmetry_faces = {plane.face for plane in self.symmetry_planes}
        for plane in self.symmetry_planes:
            tolerance = (
                16
                * np.finfo(np.dtype(real_dtype)).eps
                * max(abs(plane.coordinate), surface.grid_spacing[plane.axis])
            )
            keep &= ~np.isclose(
                face.patch_positions[:, plane.axis],
                plane.coordinate,
                rtol=0,
                atol=tolerance,
            )
        for omitted_face in self.omitted_faces:
            if omitted_face in symmetry_faces:
                continue
            axis = "xyz".index(omitted_face[0])
            coordinate = (
                surface.physical_lower[axis]
                if omitted_face.endswith("0")
                else surface.physical_upper[axis]
            )
            tolerance = (
                16
                * np.finfo(np.dtype(real_dtype)).eps
                * max(abs(coordinate), surface.grid_spacing[axis])
            )
            edge_clearance = surface.grid_spacing[axis] + tolerance
            if omitted_face.endswith("0"):
                keep &= face.patch_positions[:, axis] > coordinate + edge_clearance
            else:
                keep &= face.patch_positions[:, axis] < coordinate - edge_clearance
        return keep

    @property
    def signature(self) -> str:
        planes = ",".join(
            f"{plane.face}:{plane.boundary_type}:{plane.coordinate:.17g}"
            for plane in self.symmetry_planes
        )
        omitted = ",".join(self.omitted_faces)
        return f"{self.name}|{omitted}|{planes}"

    @property
    def image_count(self) -> int:
        return 2 ** len(self.symmetry_planes)

    def component_images(self, component: str) -> Tuple[SymmetryImage, ...]:
        images = []
        for count in range(len(self.symmetry_planes) + 1):
            for selected in combinations(self.symmetry_planes, count):
                parity = 1
                for plane in selected:
                    parity *= component_parity(
                        component, plane.axis, plane.boundary_type
                    )
                images.append(SymmetryImage(tuple(selected), parity))
        return tuple(images)

    def apply_quadrature(
        self, surface: KSIRComponentSurface
    ) -> KSIRComponentSurface:
        """Half-weight patches centred on reflected tangential boundaries."""

        if not self.symmetry_planes:
            return surface
        adjusted_faces = []
        offsets = COMPONENT_OFFSETS[surface.component]
        for face in surface.faces:
            weights = face.area_weights.copy()
            for plane in self.symmetry_planes:
                if plane.axis == face.normal_axis or offsets[plane.axis] != 0:
                    continue
                tolerance = (
                    16
                    * np.finfo(face.area_weights.dtype).eps
                    * max(
                        abs(plane.coordinate),
                        surface.grid_spacing[plane.axis],
                    )
                )
                on_plane = np.isclose(
                    face.patch_positions[:, plane.axis],
                    plane.coordinate,
                    rtol=0,
                    atol=tolerance,
                )
                weights[on_plane] *= 0.5
            weights.setflags(write=False)
            adjusted_faces.append(replace(face, area_weights=weights))
        return replace(surface, faces=tuple(adjusted_faces))

    def transformed_faces(
        self,
        surface: KSIRComponentSurface,
        field: npt.NDArray,
        normal_derivative: npt.NDArray,
    ) -> Iterator[
        tuple[
            str,
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray[np.floating],
            npt.NDArray,
            npt.NDArray,
        ]
    ]:
        """Yield physical and virtual face patches ready for evaluation."""

        start = 0
        images = self.component_images(surface.component)
        for face in surface.faces:
            stop = start + face.npatches
            positions = face.patch_positions
            normals = np.broadcast_to(face.normal, positions.shape)
            face_field = field[:, start:stop]
            face_derivative = normal_derivative[:, start:stop]
            for image in images:
                image_positions, image_normals = image.transform(
                    positions, normals
                )
                yield (
                    face.face_id,
                    image_positions,
                    image_normals,
                    face.area_weights,
                    image.parity * face_field,
                    image.parity * face_derivative,
                )
            start = stop


def component_parity(component: str, axis: int, boundary_type: str) -> int:
    """Return scalar component parity under a PEC or PMC reflection."""

    if len(component) != 2 or component[0] not in ("E", "H"):
        raise ValueError(f"unknown field component {component!r}")
    if component[1].lower() not in _COMPONENT_AXIS:
        raise ValueError(f"unknown field component {component!r}")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if boundary_type not in ("pec", "pmc"):
        raise ValueError("boundary_type must be 'pec' or 'pmc'")

    normal_component = _COMPONENT_AXIS[component[1].lower()] == axis
    if boundary_type == "pec":
        even = normal_component if component[0] == "E" else not normal_component
    else:
        even = not normal_component if component[0] == "E" else normal_component
    return 1 if even else -1


def resolve_closure(
    policy,
    symmetry_boundaries: Mapping[str, str],
    lower: Sequence[int],
    upper: Sequence[int],
    grid_size: Sequence[int],
    grid_spacing: Sequence[float],
    real_dtype=None,
) -> ResolvedKSIRClosure:
    """Resolve a public closure policy against actual grid boundaries."""

    if policy == "closed":
        return ResolvedKSIRClosure("closed", (), (), True, True)
    if isinstance(policy, ExperimentalMask):
        return ResolvedKSIRClosure(
            "experimental_mask",
            policy.omit_faces,
            (),
            False,
            False,
        )
    if isinstance(policy, HuygensOpenSurface):
        return ResolvedKSIRClosure(
            "huygens_open",
            policy.omit_faces,
            (),
            False,
            False,
        )
    if not isinstance(policy, SymmetryCompletion):
        raise ValueError(
            "closure must be 'closed', SymmetryCompletion, HuygensOpenSurface, "
            "or ExperimentalMask"
        )

    unknown_boundaries = set(symmetry_boundaries) - set(FACES)
    if unknown_boundaries:
        raise ValueError(
            f"symmetry_boundaries contains unknown faces: {sorted(unknown_boundaries)}"
        )
    invalid_types = {
        face: boundary_type
        for face, boundary_type in symmetry_boundaries.items()
        if boundary_type not in ("pec", "pmc")
    }
    if invalid_types:
        raise ValueError(
            "symmetry_boundaries types must be 'pec' or 'pmc': "
            f"{invalid_types}"
        )

    lower_values = np.asarray(lower, dtype=np.int64)
    upper_values = np.asarray(upper, dtype=np.int64)
    size_values = np.asarray(grid_size, dtype=np.int64)
    if real_dtype is None:
        candidate = np.asarray(grid_spacing).dtype
        real_dtype = candidate if candidate.kind == "f" else np.dtype(float)
    spacing = np.asarray(grid_spacing, dtype=real_dtype)
    touched = []
    for face, boundary_type in symmetry_boundaries.items():
        axis, sign = _FACE_AXIS_SIGN[face]
        if (sign < 0 and lower_values[axis] == 0) or (
            sign > 0 and upper_values[axis] == size_values[axis]
        ):
            touched.append(face)
    requested = tuple(touched) if policy.faces is None else policy.faces
    if not requested:
        raise ValueError(
            "SymmetryCompletion requires the KSIR box to touch a declared "
            "symmetry boundary"
        )
    for face in requested:
        if face not in symmetry_boundaries:
            raise ValueError(
                f"KSIR symmetry face {face!r} is not a declared boundary"
            )
        axis, sign = _FACE_AXIS_SIGN[face]
        touches = (
            lower_values[axis] == 0
            if sign < 0
            else upper_values[axis] == size_values[axis]
        )
        if not touches:
            raise ValueError(
                f"KSIR box does not touch declared symmetry face {face!r}"
            )
    axes = [_FACE_AXIS_SIGN[face][0] for face in requested]
    if len(set(axes)) != len(axes):
        raise ValueError(
            "SymmetryCompletion supports at most one reflection plane per axis"
        )

    planes = []
    for face in FACES:
        if face not in requested:
            continue
        axis, sign = _FACE_AXIS_SIGN[face]
        coordinate = 0.0 if sign < 0 else size_values[axis] * spacing[axis]
        planes.append(
            SymmetryPlane(
                face,
                axis,
                float(coordinate),
                symmetry_boundaries[face],
            )
        )
    return ResolvedKSIRClosure(
        "symmetry",
        tuple(plane.face for plane in planes),
        tuple(planes),
        True,
        True,
    )


def closure_from_metadata(
    name: str,
    omitted_faces: Sequence[str],
    plane_faces: Sequence[str],
    plane_types: Sequence[str],
    plane_coordinates: Sequence[float],
) -> ResolvedKSIRClosure:
    """Rebuild a resolved closure from persisted HDF5 metadata."""

    if name not in ("closed", "symmetry", "huygens_open", "experimental_mask"):
        raise ValueError(f"unknown saved closure policy {name!r}")
    if not (
        len(plane_faces) == len(plane_types) == len(plane_coordinates)
    ):
        raise ValueError("saved symmetry-plane metadata has inconsistent lengths")
    omitted = _face_tuple("omitted_faces", omitted_faces)
    if name == "closed" and (omitted or plane_faces):
        raise ValueError("saved closed policy must not omit faces or contain planes")
    if name == "symmetry" and (not omitted or not plane_faces):
        raise ValueError("saved symmetry policy requires omitted faces and planes")
    if name == "experimental_mask" and (not omitted or plane_faces):
        raise ValueError(
            "saved experimental mask requires omitted faces and no symmetry planes"
        )
    if name == "huygens_open" and (
        not omitted or len(omitted) == len(FACES) or plane_faces
    ):
        raise ValueError(
            "saved open Huygens surface requires one to five omitted faces and "
            "no symmetry planes"
        )
    if name == "symmetry" and _face_tuple("plane_faces", plane_faces) != omitted:
        raise ValueError("saved symmetry plane faces must match omitted faces")
    planes = []
    for face, boundary_type, coordinate in zip(
        plane_faces, plane_types, plane_coordinates
    ):
        axis, _ = _FACE_AXIS_SIGN[face]
        planes.append(
            SymmetryPlane(face, axis, float(coordinate), boundary_type)
        )
    mathematically_closed = name not in ("huygens_open", "experimental_mask")
    return ResolvedKSIRClosure(
        name,
        omitted,
        tuple(planes),
        mathematically_closed,
        mathematically_closed,
    )
