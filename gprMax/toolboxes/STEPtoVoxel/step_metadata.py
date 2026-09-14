# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Recover semantic names and relationships from STEP exchange metadata.

CAD exporters attach names to several different ISO 10303 entities. This
module parses the entity graph and resolves a name from the exact STEP entity
that OpenCascade reports for a transferred shape. It intentionally does not
infer semantic names from geometry.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

_ENTITY_START = re.compile(r"#(?P<identifier>\d+)\s*=", re.IGNORECASE)
_ENTITY_TYPE = re.compile(r"\s*(?P<type>[A-Z][A-Z0-9_]*)\s*\(", re.IGNORECASE)
_STRING = re.compile(r"'((?:[^']|'')*)'")
_REFERENCE = re.compile(r"#(\d+)")
_GENERATED_NAME = re.compile(r"[0-9_:\[\]=>-]+")
_GENERIC_CAD_NAME = re.compile(
    r"(?:SOLID|BODY|PART|COMPONENT|SHAPE|SHELL|FACE)[\s_.-]*\d*",
    flags=re.IGNORECASE,
)

_GENERIC_NAMES = {
    "",
    "DEFAULT",
    "NONE",
    "NULL",
    "UNDEFINED",
    "UNKNOWN",
    "UNNAMED",
    "UNSPECIFIED",
}

_TYPE_RANK = {
    "MANIFOLD_SOLID_BREP": 0,
    "BREP_WITH_VOIDS": 0,
    "FACETED_BREP": 0,
    "SHELL_BASED_SURFACE_MODEL": 0,
    "GEOMETRIC_SET": 0,
    "VERTEX_POINT": 0,
    "EDGE_CURVE": 0,
    "ADVANCED_BREP_SHAPE_REPRESENTATION": 1,
    "MANIFOLD_SURFACE_SHAPE_REPRESENTATION": 1,
    "SHAPE_REPRESENTATION": 1,
    "PRODUCT": 2,
    "PRODUCT_DEFINITION_FORMATION": 3,
    "NEXT_ASSEMBLY_USAGE_OCCURRENCE": 3,
    "PRESENTATION_LAYER_ASSIGNMENT": 4,
}


@dataclass(frozen=True)
class StepEntity:
    """A lightweight ISO 10303 entity record."""

    identifier: int
    entity_type: str
    name: str | None
    references: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedStepName:
    """A semantic name resolved through the STEP entity graph."""

    name: str
    raw_name: str
    entity_id: int
    source_entity_id: int
    source_entity_type: str
    graph_distance: int
    confidence: str


def is_generated_name(name: str) -> bool:
    """Whether *name* resembles an OpenCascade-generated label path."""
    return bool(_GENERATED_NAME.fullmatch((name or "").strip()))


def is_generic_cad_name(name: str) -> bool:
    """Whether *name* resembles an exporter-generated solid/body label."""
    return bool(_GENERIC_CAD_NAME.fullmatch((name or "").strip()))


def semantic_leaf_name(name: str) -> str:
    """Remove a pipe-delimited exporter hierarchy while preserving its leaf."""
    name = (name or "").strip()
    if "|" in name:
        leaf = name.rsplit("|", 1)[-1].strip()
        if leaf:
            return leaf
    return name


def _entity_statements(text: str):
    """Yield ``(identifier, definition)`` across multiline STEP entities."""
    position = 0
    while match := _ENTITY_START.search(text, position):
        identifier = int(match.group("identifier"))
        index = match.end()
        in_string = False
        while index < len(text):
            char = text[index]
            if char == "'":
                if in_string and index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                in_string = not in_string
            elif char == ";" and not in_string:
                yield identifier, text[match.end() : index].strip()
                position = index + 1
                break
            index += 1
        else:
            break


def _entity_name(entity_type: str, strings: list[str]) -> str | None:
    """Select the semantic-name field used by common STEP entity types."""
    if not strings:
        return None
    if entity_type in {"PRODUCT", "NEXT_ASSEMBLY_USAGE_OCCURRENCE"}:
        order = (1, 0, 2)
    else:
        order = tuple(range(len(strings)))
    for index in order:
        if index >= len(strings):
            continue
        value = strings[index].replace("''", "'").strip()
        if value.upper() not in _GENERIC_NAMES:
            return value
    return None


def parse_step_entities(path: str | Path) -> dict[int, StepEntity]:
    """Parse names and references from an ISO-10303-21 STEP file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    entities = {}
    for identifier, definition in _entity_statements(text):
        type_match = _ENTITY_TYPE.match(definition)
        entity_type = type_match.group("type").upper() if type_match else "COMPLEX_ENTITY"
        strings = _STRING.findall(definition)
        name = _entity_name(entity_type, strings)
        references = tuple(int(value) for value in _REFERENCE.findall(definition))
        entities[identifier] = StepEntity(identifier, entity_type, name, references)
    return entities


def _meaningful_name(name: str | None) -> bool:
    if name is None:
        return False
    value = name.strip()
    return value.upper() not in _GENERIC_NAMES and not is_generated_name(value)


def _is_metadata_dead_end(entity_type: str) -> bool:
    """Avoid graph traversal through shared units and representation contexts."""
    return any(
        token in entity_type
        for token in (
            "CONTEXT",
            "UNIT",
            "UNCERTAINTY",
            "AXIS2_PLACEMENT",
            "CARTESIAN_POINT",
            "DIRECTION",
        )
    )


class StepMetadata:
    """Reference graph used to resolve exporter-dependent STEP names."""

    def __init__(self, entities: dict[int, StepEntity]):
        self.entities = entities
        reverse = defaultdict(set)
        for entity in entities.values():
            for reference in entity.references:
                reverse[reference].add(entity.identifier)
        self.reverse_references = {key: frozenset(value) for key, value in reverse.items()}

    @classmethod
    def from_file(cls, path: str | Path) -> StepMetadata:
        return cls(parse_step_entities(path))

    def resolve_name(self, entity_id: int, max_depth: int = 7) -> ResolvedStepName | None:
        """Resolve the best semantic name connected to *entity_id*.

        The search is reference based. Direct shape names win, followed by
        representation, product, assembly-occurrence, and layer names. Shared
        unit/context records are not traversed, preventing names from leaking
        between unrelated solids in one assembly.
        """
        if entity_id not in self.entities:
            return None

        queue = deque([(entity_id, 0)])
        visited = {entity_id}
        candidates = []
        while queue:
            current_id, distance = queue.popleft()
            entity = self.entities.get(current_id)
            if entity is None:
                continue
            if _meaningful_name(entity.name):
                candidates.append(
                    (
                        is_generic_cad_name(entity.name or ""),
                        distance,
                        _TYPE_RANK.get(entity.entity_type, 10),
                        current_id,
                        entity,
                    )
                )
            if distance >= max_depth:
                continue

            neighbours = set(self.reverse_references.get(current_id, ()))
            neighbours.update(entity.references)
            for neighbour_id in neighbours:
                if neighbour_id in visited:
                    continue
                neighbour = self.entities.get(neighbour_id)
                if neighbour is None or _is_metadata_dead_end(neighbour.entity_type):
                    continue
                visited.add(neighbour_id)
                queue.append((neighbour_id, distance + 1))

        if not candidates:
            return None
        generic, distance, _rank, source_id, source = min(candidates)
        confidence = "exact" if distance <= 1 else "high" if distance <= 3 else "inferred"
        if generic:
            confidence = "low"
        raw_name = source.name or ""
        return ResolvedStepName(
            name=semantic_leaf_name(raw_name),
            raw_name=raw_name,
            entity_id=entity_id,
            source_entity_id=source_id,
            source_entity_type=source.entity_type,
            graph_distance=distance,
            confidence=confidence,
        )
