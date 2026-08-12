# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Conservative material-group suggestions for STEP assemblies."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite, log, log1p
from statistics import median
from typing import Any, Sequence


@dataclass(frozen=True)
class MaterialGroup:
    """Components that may conveniently share one material assignment."""

    identifier: str
    confidence: str
    part_names: tuple[str, ...]
    priority: int
    similar_group: str = ""


def _positive_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(result) or result <= 0:
        return None
    return result


def _relative_bin(value: Any, tolerance: float) -> int | None:
    """Return a scale-independent logarithmic bin for a positive quantity."""
    value = _positive_number(value)
    if value is None:
        return None
    return int(round(log(value) / log1p(tolerance)))


def _candidate_signature(part: Any, tolerance: float) -> tuple[Any, ...] | None:
    """Build a rotation-resistant, approximate CAD-geometry signature.

    The signature is deliberately only a *candidate* match. Equal volume,
    surface area, topology and inertia do not prove that two solids are the
    same, nor that they should have the same electromagnetic material.
    """
    cad = dict(getattr(part, "cad", None) or {})
    volume = _positive_number(cad.get("vol_m3"))
    area = _positive_number(cad.get("area_m2"))
    topology = cad.get("topology_counts")
    if volume is None or area is None or not isinstance(topology, dict):
        return None

    counts = tuple(int(topology.get(key, -1)) for key in ("solids", "shells", "faces", "edges", "vertices"))
    moments = cad.get("principal_moments_m5")
    if isinstance(moments, (tuple, list)) and len(moments) == 3:
        # Volume properties use unit density, so moments have dimensions L^5.
        # Normalising them removes overall scale while retaining shape ratios.
        scale = volume ** (5.0 / 3.0)
        shape_values = tuple(_relative_bin(float(value) / scale, tolerance) for value in sorted(moments))
    else:
        dimensions = cad.get("bbox_dims_xyz")
        if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 3:
            return None
        # This fallback is invariant to axis permutations, but not to an
        # arbitrary rotation of an asymmetric object.
        shape_values = tuple(_relative_bin(value, tolerance) for value in sorted(dimensions))

    return (
        counts,
        _relative_bin(volume, tolerance),
        _relative_bin(area, tolerance),
        shape_values,
    )


def _candidate_labels(parts: Sequence[Any], tolerance: float) -> dict[str, str]:
    buckets: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for part in parts:
        signature = _candidate_signature(part, tolerance)
        if signature is not None:
            buckets[signature].append(part)

    labels: dict[str, str] = {}
    repeated = sorted(
        (members for members in buckets.values() if len(members) > 1),
        key=lambda members: tuple(sorted(str(part.name) for part in members)),
    )
    for index, members in enumerate(repeated, start=1):
        label = f"S{index:03d}"
        for part in members:
            labels[str(part.name)] = label
    return labels


def _priorities(groups: Sequence[tuple[str, list[Any]]]) -> dict[str, int]:
    """Give smaller groups higher overlap priority without arbitrary ties."""
    volumes: dict[str, float | None] = {}
    for key, members in groups:
        values = [_positive_number(getattr(part, "cad", {}).get("vol_m3")) for part in members]
        values = [value for value in values if value is not None]
        volumes[key] = median(values) if values else None
    distinct = sorted({value for value in volumes.values() if value is not None}, reverse=True)
    rank = {value: index + 1 for index, value in enumerate(distinct)}
    return {key: rank[volume] if volume is not None else 0 for key, volume in volumes.items()}


def suggest_material_groups(
    parts: Sequence[Any],
    *,
    mode: str = "exact",
    relative_tolerance: float = 0.01,
) -> list[MaterialGroup]:
    """Return deterministic, editable material groups.

    ``exact`` groups repeated occurrences transferred from the same STEP shape
    entity. ``similar`` additionally groups approximate CAD signatures.
    ``none`` emits one row per component. Approximate matches should always be
    inspected by the user before material properties are assigned.
    """
    if mode not in {"none", "exact", "similar"}:
        raise ValueError("group mode must be 'none', 'exact', or 'similar'")
    if not 0 < relative_tolerance < 1:
        raise ValueError("grouping relative tolerance must be between zero and one")

    parts = list(parts)
    candidate_labels = _candidate_labels(parts, relative_tolerance)
    buckets: dict[str, list[Any]] = defaultdict(list)
    confidence: dict[str, str] = {}

    if mode == "none":
        for part in parts:
            key = f"part:{part.name}"
            buckets[key].append(part)
            confidence[key] = "unique"
    elif mode == "similar":
        for part in parts:
            label = candidate_labels.get(str(part.name))
            if label:
                key = f"similar:{label}"
                confidence[key] = "similar_candidate"
            else:
                key = f"part:{part.name}"
                confidence[key] = "unique"
            buckets[key].append(part)
    else:
        exact_counts: dict[int, int] = defaultdict(int)
        for part in parts:
            entity_id = getattr(part, "step_entity_id", None)
            if entity_id is not None:
                exact_counts[int(entity_id)] += 1
        for part in parts:
            entity_id = getattr(part, "step_entity_id", None)
            if entity_id is not None and exact_counts[int(entity_id)] > 1:
                key = f"step:{int(entity_id)}"
                confidence[key] = "exact_instance"
            else:
                key = f"part:{part.name}"
                confidence[key] = "unique"
            buckets[key].append(part)

    ordered = sorted(
        buckets.items(),
        key=lambda item: (-len(item[1]), tuple(sorted(str(part.name) for part in item[1]))),
    )
    priority = _priorities(ordered)
    return [
        MaterialGroup(
            identifier=f"G{index:03d}",
            confidence=confidence[key],
            part_names=tuple(sorted(str(part.name) for part in members)),
            priority=priority[key],
            similar_group=(
                candidate_labels.get(str(members[0].name), "")
                if len({candidate_labels.get(str(part.name), "") for part in members}) == 1
                else ""
            ),
        )
        for index, (key, members) in enumerate(ordered, start=1)
    ]
