# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""CAD marker conventions and coordinate translation helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_MARKER_NAME = re.compile(
    r"^(?:gprmax[\s_.:|\-]*)?" r"(?P<kind>source|src|receiver|rx|port)" r"[\s_.:|\-]*(?P<identifier>.*)$",
    flags=re.IGNORECASE,
)
_KINDS = {
    "source": "source",
    "src": "source",
    "receiver": "receiver",
    "rx": "receiver",
    "port": "port",
}


@dataclass(frozen=True)
class CADMarker:
    """A non-physical CAD marker expressed relative to voxel geometry."""

    name: str
    kind: str
    identifier: str
    geometry: str
    cad_position: tuple[float, float, float]
    local_position: tuple[float, float, float]
    cad_bounds: tuple[float, float, float, float, float, float]
    local_bounds: tuple[float, float, float, float, float, float]
    direction: tuple[float, float, float] | None
    axis: str | None
    cad_endpoints: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    local_endpoints: tuple[tuple[float, float, float], tuple[float, float, float]] | None
    length: float | None

    def model_position(self, geometry_import_origin: Sequence[float]) -> tuple[float, float, float]:
        """Translate the marker to a model importing geometry at ``p1``."""
        return tuple(float(a + b) for a, b in zip(geometry_import_origin, self.local_position))

    def model_bounds(self, geometry_import_origin: Sequence[float]) -> tuple[float, ...]:
        """Translate the marker bounds to a model importing geometry at ``p1``."""
        origin = tuple(float(value) for value in geometry_import_origin)
        return (
            origin[0] + self.local_bounds[0],
            origin[1] + self.local_bounds[1],
            origin[2] + self.local_bounds[2],
            origin[0] + self.local_bounds[3],
            origin[1] + self.local_bounds[4],
            origin[2] + self.local_bounds[5],
        )

    def model_endpoints(
        self, geometry_import_origin: Sequence[float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Translate a line marker's endpoints to model coordinates."""
        if self.local_endpoints is None:
            return None
        origin = np.asarray(geometry_import_origin, dtype=float)
        return tuple(tuple(float(value) for value in origin + endpoint) for endpoint in self.local_endpoints)


def classify_marker_name(name: str) -> tuple[str, str] | None:
    """Return ``(kind, identifier)`` for a portable gprMax CAD marker name.

    Recognised examples include ``gprmax_source_tx1``, ``source1``,
    ``receiver_rx2``, ``rx3`` and ``port1``.
    """
    match = _MARKER_NAME.fullmatch((name or "").strip())
    if not match:
        return None
    kind = _KINDS[match.group("kind").lower()]
    identifier = match.group("identifier").strip(" _.:|-") or name
    return kind, identifier


def _surface_direction(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray | None:
    if len(triangles) == 0:
        return None
    edge1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edge2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    normals = np.cross(edge1, edge2)
    normal = normals.sum(axis=0)
    magnitude = np.linalg.norm(normal)
    if magnitude <= np.finfo(float).eps:
        return None
    normal /= magnitude
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal *= -1
    return normal


def _axis(direction: np.ndarray | None, tolerance: float = 1e-6) -> str | None:
    if direction is None:
        return None
    dominant = int(np.argmax(np.abs(direction)))
    if abs(direction[dominant]) < 1 - tolerance:
        return None
    return "xyz"[dominant]


def _line_endpoints(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the most widely separated pair of topological line vertices."""
    if len(vertices) < 2:
        return None
    offsets = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    distances2 = np.einsum("ijk,ijk->ij", offsets, offsets)
    first, second = np.unravel_index(int(np.argmax(distances2)), distances2.shape)
    if distances2[first, second] <= np.finfo(float).eps:
        return None
    return vertices[first].copy(), vertices[second].copy()


def marker_record(
    part: Any,
    grid_origin: Sequence[float],
    spacing: Sequence[float],
    *,
    vertices: np.ndarray | None = None,
    triangles: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Build a JSON-safe marker record from a named CAD component."""
    classification = classify_marker_name(part.name)
    if classification is None:
        return None
    kind, identifier = classification

    bounds = np.asarray(part.cad.get("bbox_xyzxyz"), dtype=float)
    if bounds.shape != (6,):
        raise ValueError(f"CAD marker {part.name!r} has no valid bounding box")
    lower = bounds[:3]
    upper = bounds[3:]
    centre = 0.5 * (lower + upper)
    grid_origin = np.asarray(grid_origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    local_lower = lower - grid_origin
    local_upper = upper - grid_origin
    local_centre = centre - grid_origin
    dimensions = upper - lower

    volume = float(part.cad.get("vol_m3") or 0.0)
    area = float(part.cad.get("area_m2") or 0.0)
    tolerance = max(1e-12, 0.01 * float(np.min(spacing)))
    nonzero_dimensions = int(np.count_nonzero(dimensions > tolerance))
    if volume > 0:
        geometry = "solid"
    elif area > 0 or nonzero_dimensions >= 2:
        geometry = "surface"
    elif nonzero_dimensions == 1:
        geometry = "line"
    else:
        geometry = "point"

    direction = None
    endpoints = None
    length = None
    if geometry == "surface" and vertices is not None and triangles is not None:
        direction = _surface_direction(np.asarray(vertices), np.asarray(triangles))
    elif geometry == "line":
        vertex_array = np.empty((0, 3)) if vertices is None else np.asarray(vertices, dtype=float)
        endpoints = _line_endpoints(vertex_array)
        if endpoints is None:
            endpoints = (lower.copy(), upper.copy())
        vector = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(vector))
        if length > np.finfo(float).eps:
            direction = vector / length
            dominant = int(np.argmax(np.abs(direction)))
            if direction[dominant] < 0:
                endpoints = (endpoints[1], endpoints[0])
                direction *= -1
    elif geometry == "solid":
        order = np.argsort(dimensions)
        if dimensions[order[-1]] > 1.5 * dimensions[order[-2]]:
            direction = np.zeros(3)
            direction[order[-1]] = 1

    return {
        "name": part.name,
        "kind": kind,
        "identifier": identifier,
        "geometry": geometry,
        "cad_position_m": centre.tolist(),
        "local_position_m": local_centre.tolist(),
        "cad_bounds_xyzxyz_m": bounds.tolist(),
        "local_bounds_xyzxyz_m": np.concatenate((local_lower, local_upper)).tolist(),
        "grid_coordinates": (local_centre / spacing).tolist(),
        "direction": None if direction is None else direction.tolist(),
        "axis": _axis(direction),
        "direction_sign_is_arbitrary": direction is not None,
        "cad_endpoints_m": None if endpoints is None else [endpoint.tolist() for endpoint in endpoints],
        "local_endpoints_m": (
            None if endpoints is None else [(endpoint - grid_origin).tolist() for endpoint in endpoints]
        ),
        "length_m": length,
        "step_entity_id": getattr(part, "step_entity_id", None),
        "name_source": getattr(part, "name_source", None),
        "name_confidence": getattr(part, "name_confidence", None),
    }


def load_markers(path: str | Path) -> dict[str, CADMarker]:
    """Load ``markers.json`` and return markers keyed by CAD name."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    markers = {}
    for item in payload["markers"]:
        marker = CADMarker(
            name=item["name"],
            kind=item["kind"],
            identifier=item["identifier"],
            geometry=item["geometry"],
            cad_position=tuple(item["cad_position_m"]),
            local_position=tuple(item["local_position_m"]),
            cad_bounds=tuple(item["cad_bounds_xyzxyz_m"]),
            local_bounds=tuple(item["local_bounds_xyzxyz_m"]),
            direction=None if item["direction"] is None else tuple(item["direction"]),
            axis=item["axis"],
            cad_endpoints=(
                None
                if item.get("cad_endpoints_m") is None
                else tuple(tuple(endpoint) for endpoint in item["cad_endpoints_m"])
            ),
            local_endpoints=(
                None
                if item.get("local_endpoints_m") is None
                else tuple(tuple(endpoint) for endpoint in item["local_endpoints_m"])
            ),
            length=item.get("length_m"),
        )
        markers[marker.name] = marker
    return markers
