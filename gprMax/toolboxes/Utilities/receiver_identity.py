"""Public receiver selection and cross-file identity, independent of HDF5 order.

Numbers and paths select a receiver in ONE file. A sequence is anchored to that
receiver in its first file, then matched by StudyID, unique Name, or a versioned
construction layout/index. Coordinates are not identities (receivers can move
or coincide). Legacy multi-receiver files without unique labels are ambiguous.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import h5py


def text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def natural_key(value):
    """Numeric receiver/path order, including rx9 before rx10."""
    return tuple((1, int(part)) if part.isdigit() else (0, part) for part in re.split(r"(\d+)", str(value)))


@dataclass(frozen=True)
class ReceiverIdentity:
    path: str
    name: str = ""
    study_id: str = ""
    name_kind: str = ""
    build_index: int | None = None
    layout: str = ""
    name_unique: bool = False
    study_unique: bool = False
    count: int = 0


class ReceiverCatalogue(dict):
    """Indexed identity lookup keeps multi-receiver merges linear in count."""

    def __init__(self, receivers):
        super().__init__(receivers)
        self.names = defaultdict(list)
        self.studies = defaultdict(list)
        self.ordinals = defaultdict(list)
        for rx in self.values():
            self.names[rx.name].append(rx)
            self.studies[rx.study_id].append(rx)
            self.ordinals[(rx.layout, rx.build_index)].append(rx)


def receiver_catalogue(grid) -> dict[str, ReceiverIdentity]:
    """Snapshot identities in one grid namespace; safe after the file closes."""
    if "rxs" not in grid:
        return {}
    groups = {f"rxs/{key}": group for key, group in grid["rxs"].items() if isinstance(group, h5py.Group)}
    names = Counter(text(g.attrs.get("Name", "")) for g in groups.values())
    studies = Counter(text(g.attrs.get("StudyID", "")) for g in groups.values())
    versioned = (
        grid.attrs.get("ReceiverOrderSchemaVersion") == 1
        and text(grid.attrs.get("ReceiverOrder", "")) == "construction"
    )
    layout = text(grid.attrs.get("ReceiverLayout", "")) if versioned else ""
    result = {}
    for path in sorted(groups, key=natural_key):
        attrs = groups[path].attrs
        name = text(attrs.get("Name", ""))
        study = text(attrs.get("StudyID", ""))
        index = attrs.get("BuildIndex") if versioned else None
        if index is not None and (int(index) != index or index < 0):
            raise ValueError(f"Invalid receiver BuildIndex at {groups[path].name}")
        result[path] = ReceiverIdentity(
            path,
            name,
            study,
            text(attrs.get("NameKind", "")),
            int(index) if index is not None else None,
            layout,
            bool(name) and names[name] == 1,
            bool(study) and studies[study] == 1,
            len(groups),
        )
    indices = [rx.build_index for rx in result.values()]
    if versioned and (None in indices or len(set(indices)) != len(indices)):
        raise ValueError(f"Missing or duplicate receiver BuildIndex in {grid.name}")
    return ReceiverCatalogue(result)


def select_receiver(catalogue, selector) -> ReceiverIdentity:
    """Resolve a number/path, ``name:label``, ``study:id``, or ``build:N``."""
    value = str(selector)
    for prefix, attr in (("name:", "name"), ("study:", "study_id"), ("build:", "build_index")):
        if value.startswith(prefix):
            wanted = value[len(prefix) :]
            matches = [rx for rx in catalogue.values() if str(getattr(rx, attr)) == wanted]
            if len(matches) != 1:
                raise ValueError(f"Receiver selector {value!r} has {len(matches)} matches; use a unique identity")
            return matches[0]
    path = f"rxs/rx{value}" if value.isdigit() else value.strip("/")
    if path.startswith("rx") and "/" not in path:
        path = "rxs/" + path
    if path not in catalogue:
        raise ValueError(f"Receiver {value!r} is not present; available: {list(catalogue)}")
    return catalogue[path]


def match_receiver(reference: ReceiverIdentity, catalogue) -> ReceiverIdentity:
    """Find the same declared receiver, never silently fall back to its path.

    A construction layout describes ordered labels/output requests, not an
    entire model. Ordinal matching is valid for the same declared acquisition
    layout, including moving/generated or duplicate-labelled receivers.
    """
    candidates = catalogue.values()
    indexed = isinstance(catalogue, ReceiverCatalogue)
    if reference.study_id:
        matches = (
            catalogue.studies.get(reference.study_id, [])
            if indexed
            else [rx for rx in candidates if rx.study_id == reference.study_id]
        )
        if reference.study_unique and len(matches) == 1:
            return matches[0]
        raise ValueError(f"Receiver identity StudyID {reference.study_id!r} is missing or ambiguous")

    ordinal = (
        catalogue.ordinals.get((reference.layout, reference.build_index), [])
        if indexed
        else [rx for rx in candidates if rx.layout == reference.layout and rx.build_index == reference.build_index]
    )
    if not reference.layout or reference.build_index is None:
        ordinal = []
    # Generated coordinate labels can change (or cross another receiver's old
    # position) during a scan. In the same layout the ordinal is authoritative.
    if reference.name_kind == "generated" and len(ordinal) == 1:
        return ordinal[0]
    if reference.name and reference.name_unique:
        matches = (
            catalogue.names.get(reference.name, [])
            if indexed
            else [rx for rx in candidates if rx.name == reference.name]
        )
        if len(matches) == 1:
            if matches[0].study_id:
                raise ValueError("Receiver identity StudyID differs between files")
            return matches[0]
    if len(ordinal) == 1:
        return ordinal[0]
    # Preserve genuinely anonymous, single-receiver legacy files. Different
    # nonempty labels are NOT made compatible just because both files have rx1.
    if reference.count == 1 and len(candidates) == 1:
        other = next(iter(candidates))
        if not reference.name and not other.name and not other.study_id:
            return other
    raise ValueError(
        f"Receiver identity {reference.name or reference.path!r} is missing or ambiguous; "
        "group numbers are file-local. Supply unique receiver names/StudyIDs or "
        "an explicit per-file receiver selection."
    )


def matching_receiver_path(reference_filename, reference_path, filename) -> str:
    """Match within the reference receiver's grid, including subgrids."""
    grid_path, _, leaf = str(reference_path).rpartition("/rxs/")
    if not leaf:
        raise ValueError(f"Not a public receiver path: {reference_path!r}")
    with h5py.File(reference_filename, "r") as source, h5py.File(filename, "r") as target:
        reference = select_receiver(receiver_catalogue(source[grid_path or "/"]), f"rxs/{leaf}")
        if (grid_path or "/") not in target:
            raise ValueError(f"Receiver grid {grid_path or '/'} is not present in {filename}")
        match = match_receiver(reference, receiver_catalogue(target[grid_path or "/"]))
        return f"{grid_path}/{match.path}"
