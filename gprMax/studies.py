# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""Reusable-geometry parameter studies.

The first implementation deliberately supports only stateless local sources
and ordinary receivers. Stateful ports, plane waves, and eigenmode objects
need family-specific reset/rebuild hooks before they can safely participate.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


_POSITION_COLUMNS = ("x_m", "y_m", "z_m")
_SOURCE_PARAMETERS = {"active", "position", "waveform_id", "start", "stop", "scale"}
_RECEIVER_PARAMETERS = {"position", "record"}
_CSV_COLUMNS = {
    "case_id",
    "object_id",
    "active",
    *_POSITION_COLUMNS,
    "waveform_id",
    "start_s",
    "stop_s",
    "scale",
    "record",
}


@dataclass
class ObjectState:
    """Overrides for one study object in one case.

    ``object`` may be a deterministic object ID such as
    ``"hertzian_dipole_1"``/``"rx_1"`` or the corresponding Python API
    user-object instance.
    """

    object: Any
    parameters: dict[str, Any] = field(default_factory=dict)

    def __init__(self, object: Any, **parameters: Any):
        self.object = object
        self.parameters = dict(parameters)


@dataclass
class StudyCase:
    """A named set of object overrides."""

    id: str
    states: list[ObjectState]

    def __init__(self, id: str, states: Iterable[ObjectState]):
        self.id = str(id).strip()
        self.states = list(states)
        if not self.id:
            raise ValueError("A study case ID cannot be empty.")
        if not all(isinstance(state, ObjectState) for state in self.states):
            raise TypeError("StudyCase states must all be ObjectState instances.")


class Study:
    """A sequence of cases evaluated while reusing one built geometry."""

    supported_types = ("gpr",)

    def __init__(
        self,
        type: str,
        cases: Sequence[StudyCase],
        *,
        source_path: Optional[Union[str, Path]] = None,
        source_text: Optional[str] = None,
    ):
        self.type = str(type).strip().lower()
        if self.type not in self.supported_types:
            raise ValueError(
                f"Study type '{type}' is not supported. Currently supported types: "
                f"{', '.join(self.supported_types)}."
            )
        self.cases = list(cases)
        if not self.cases:
            raise ValueError("A study must contain at least one case.")
        if not all(isinstance(case, StudyCase) for case in self.cases):
            raise TypeError("Study cases must all be StudyCase instances.")
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Study case IDs must be unique.")

        self.source_path = Path(source_path).resolve() if source_path is not None else None
        self.source_text = source_text
        self._scene_bound = False
        self._runtime_baselines: dict[str, dict[str, Any]] = {}
        self._current_resolved_case: Optional[dict[str, Any]] = None

    def reset_runtime(self) -> None:
        """Clear bindings/state retained only for one call to :func:`gprMax.run`."""

        self._scene_bound = False
        self._runtime_baselines.clear()
        self._current_resolved_case = None

    @classmethod
    def from_csv(cls, type: str, path: Union[str, Path]) -> "Study":
        """Read a sparse, wide study table from CSV."""

        path = Path(path).expanduser().resolve()
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            raise ValueError(f"Study CSV '{path}' has no header row.")
        fieldnames = [name.strip() for name in reader.fieldnames]
        missing = {"case_id", "object_id"} - set(fieldnames)
        if missing:
            raise ValueError(
                f"Study CSV '{path}' is missing required column(s): {', '.join(sorted(missing))}."
            )
        unknown = set(fieldnames) - _CSV_COLUMNS
        if unknown:
            raise ValueError(
                f"Study CSV '{path}' contains unsupported column(s): "
                f"{', '.join(sorted(unknown))}."
            )

        case_rows: dict[str, list[ObjectState]] = {}
        seen: set[tuple[str, str]] = set()
        for line_number, raw_row in enumerate(reader, start=2):
            row = {(key or "").strip(): (value or "").strip() for key, value in raw_row.items()}
            case_id = row.get("case_id", "")
            object_id = row.get("object_id", "")
            if not case_id or not object_id:
                raise ValueError(
                    f"Study CSV '{path}', line {line_number}: case_id and object_id are required."
                )
            key = (case_id, object_id)
            if key in seen:
                raise ValueError(
                    f"Study CSV '{path}', line {line_number}: object '{object_id}' is listed "
                    f"more than once in case '{case_id}'."
                )
            seen.add(key)

            position_values = [row.get(name, "") for name in _POSITION_COLUMNS]
            if any(position_values) and not all(position_values):
                raise ValueError(
                    f"Study CSV '{path}', line {line_number}: x_m, y_m, and z_m must be "
                    "provided together."
                )

            parameters: dict[str, Any] = {}
            if all(position_values):
                parameters["position"] = tuple(
                    _parse_finite_float(value, path, line_number, name)
                    for name, value in zip(_POSITION_COLUMNS, position_values)
                )
            if row.get("active", ""):
                parameters["active"] = _parse_bool(row["active"], path, line_number, "active")
            if row.get("waveform_id", ""):
                parameters["waveform_id"] = row["waveform_id"]
            for csv_name, parameter_name in (
                ("start_s", "start"),
                ("stop_s", "stop"),
                ("scale", "scale"),
            ):
                if row.get(csv_name, ""):
                    parameters[parameter_name] = _parse_finite_float(
                        row[csv_name], path, line_number, csv_name
                    )
            if row.get("record", ""):
                parameters["record"] = _parse_bool(row["record"], path, line_number, "record")

            case_rows.setdefault(case_id, []).append(ObjectState(object_id, **parameters))

        cases = [StudyCase(case_id, states) for case_id, states in case_rows.items()]
        return cls(type, cases, source_path=path, source_text=text)

    def bind_scene(self, scene) -> None:
        """Assign stable IDs and resolve API object references before build."""

        if self._scene_bound:
            return

        from gprMax.user_objects.cmds_multiuse import (
            DiscretePlaneWaveAngles,
            DiscretePlaneWaveAxial,
            DiscretePlaneWaveVector,
            EigenmodeExcitation,
            HertzianDipole,
            MagneticDipole,
            MagneticFrillSource,
            NetworkExcitation,
            Rx,
            TransmissionLine,
            VoltageSource,
        )

        supported = (
            (HertzianDipole, "hertzian_dipole"),
            (MagneticDipole, "magnetic_dipole"),
            (Rx, "rx"),
        )
        registry: dict[str, Any] = {}
        aliases: dict[str, str] = {}
        reference_ids: dict[int, str] = {}
        all_objects = list(scene.grid_objects) + list(scene.output_objects)
        unsupported_excitation_types = (
            VoltageSource,
            TransmissionLine,
            MagneticFrillSource,
            NetworkExcitation,
            DiscretePlaneWaveAngles,
            DiscretePlaneWaveAxial,
            DiscretePlaneWaveVector,
            EigenmodeExcitation,
        )
        unsupported = [
            type(user_object).__name__
            for user_object in all_objects
            if isinstance(user_object, unsupported_excitation_types)
        ]
        if unsupported:
            raise ValueError(
                "This study stage cannot safely manage source type(s): "
                f"{', '.join(sorted(set(unsupported)))}. It currently supports only "
                "HertzianDipole and MagneticDipole excitations."
            )
        for object_type, prefix in supported:
            index = 0
            for user_object in all_objects:
                if not isinstance(user_object, object_type):
                    continue
                index += 1
                study_id = f"{prefix}_{index}"
                setattr(user_object, "_study_id", study_id)
                registry[study_id] = user_object
                reference_ids[id(user_object)] = study_id
                explicit_id = getattr(user_object, "id", None)
                if explicit_id:
                    explicit_id = str(explicit_id)
                    if explicit_id == study_id:
                        aliases[explicit_id] = study_id
                    elif explicit_id in aliases or explicit_id in registry:
                        raise ValueError(f"Study object alias '{explicit_id}' is not unique.")
                    else:
                        aliases[explicit_id] = study_id

        if not registry:
            raise ValueError(
                "The study scene contains no supported objects. The first implementation "
                "supports HertzianDipole, MagneticDipole, and Rx objects."
            )

        for case in self.cases:
            case_seen: set[str] = set()
            for state in case.states:
                if isinstance(state.object, str):
                    requested_id = state.object.strip()
                    study_id = aliases.get(requested_id, requested_id)
                else:
                    try:
                        study_id = reference_ids[id(state.object)]
                    except KeyError as exc:
                        raise ValueError(
                            f"Study case '{case.id}' references an object that is not a supported "
                            "top-level object in its Scene."
                        ) from exc
                if study_id not in registry:
                    available = ", ".join(sorted(registry))
                    raise ValueError(
                        f"Study case '{case.id}' references unknown object '{study_id}'. "
                        f"Available study objects: {available}."
                    )
                if study_id in case_seen:
                    raise ValueError(
                        f"Study case '{case.id}' contains duplicate state for '{study_id}'."
                    )
                case_seen.add(study_id)
                state.object = study_id
                allowed = (
                    _RECEIVER_PARAMETERS
                    if isinstance(registry[study_id], Rx)
                    else _SOURCE_PARAMETERS
                )
                unknown = set(state.parameters) - allowed
                if unknown:
                    raise ValueError(
                        f"Study object '{study_id}' does not support parameter(s) "
                        f"{', '.join(sorted(unknown))}. Allowed parameters: "
                        f"{', '.join(sorted(allowed))}."
                    )

        self.registry = registry
        self.aliases = aliases
        self._scene_bound = True
        logger.info(
            "Study objects: "
            + ", ".join(
                f"{study_id} ({type(user_object).__name__})"
                for study_id, user_object in registry.items()
            )
            + "\n"
        )

    def apply_case(self, model) -> None:
        """Restore baselines and apply the current absolute case to a built model."""

        import gprMax.config as config

        case_index = config.sim_config.current_model
        if case_index < 0 or case_index >= len(self.cases):
            raise ValueError(
                f"Study case index {case_index + 1} is outside the {len(self.cases)} available cases."
            )

        runtime = self._runtime_registry(model)
        self._capture_baselines(runtime)
        self._restore_baselines(runtime)

        case = self.cases[case_index]
        explicit_sources: set[str] = set()
        resolved: dict[str, Any] = {"case_id": case.id, "objects": {}}
        for state in case.states:
            study_id = str(state.object)
            item = runtime[study_id]
            if _is_source(item):
                explicit_sources.add(study_id)
            applied = self._apply_state(model.G, study_id, item, state.parameters)
            resolved["objects"][study_id] = applied

        # Sparse tables describe the active source set for each case. A source
        # omitted from a case is therefore disabled, while receivers retain
        # their baseline location and remain recorded.
        for study_id, item in runtime.items():
            if _is_source(item) and study_id not in explicit_sources:
                self._resample_source(model.G, item, scale=0.0)
                resolved["objects"][study_id] = {
                    "active": False,
                    "position": [float(item.coord[i] * model.G.dl[i]) for i in range(3)],
                    "waveform_id": item.waveformID,
                    "start": float(item.start),
                    "stop": float(item.stop),
                    "scale": 0.0,
                }
            elif not _is_source(item) and study_id not in resolved["objects"]:
                resolved["objects"][study_id] = {
                    "position": [float(item.coord[i] * model.G.dl[i]) for i in range(3)],
                    "record": True,
                }

        self._current_resolved_case = resolved

    def write_hdf5(self, h5file) -> None:
        """Persist the exact case and study source alongside model outputs."""

        import gprMax.config as config

        case_index = config.sim_config.current_model
        case = self.cases[case_index]
        group = h5file.create_group("study")
        group.attrs["Type"] = self.type
        group.attrs["CaseID"] = case.id
        group.attrs["CaseIndex"] = case_index + 1
        group.attrs["CaseCount"] = len(self.cases)
        group.attrs["GeometryReused"] = bool(config.get_model_config().reuse_geometry())
        if self.source_path is not None:
            group.attrs["SourcePath"] = str(self.source_path)
        if self.source_text is not None:
            group.create_dataset("source", data=self.source_text)
        definition = {
            "type": self.type,
            "cases": [
                {
                    "case_id": item.id,
                    "objects": {
                        str(state.object): _jsonable(state.parameters) for state in item.states
                    },
                }
                for item in self.cases
            ],
        }
        group.create_dataset("definition", data=json.dumps(definition, sort_keys=True))
        group.create_dataset(
            "resolved_case",
            data=json.dumps(self._current_resolved_case or {}, sort_keys=True),
        )

    def _runtime_registry(self, model) -> dict[str, Any]:
        registry: dict[str, Any] = {}
        objects = model.G.hertziandipoles + model.G.magneticdipoles + model.G.rxs
        for item in objects:
            study_id = getattr(item, "study_id", None)
            if study_id:
                registry[study_id] = item
        expected = set(self.registry)
        if set(registry) != expected:
            missing = ", ".join(sorted(expected - set(registry)))
            raise RuntimeError(f"Study runtime object binding is incomplete; missing: {missing}.")
        return registry

    def _capture_baselines(self, runtime: Mapping[str, Any]) -> None:
        if self._runtime_baselines:
            return
        for study_id, item in runtime.items():
            baseline: dict[str, Any] = {"coord": np.array(item.coordorigin, copy=True)}
            if _is_source(item):
                baseline.update(
                    waveform_id=item.waveformID,
                    start=float(item.start),
                    stop=float(item.stop),
                )
            self._runtime_baselines[study_id] = baseline

    def _restore_baselines(self, runtime: Mapping[str, Any]) -> None:
        for study_id, item in runtime.items():
            baseline = self._runtime_baselines[study_id]
            item.coord = np.array(baseline["coord"], copy=True)
            if _is_source(item):
                item.waveformID = baseline["waveform_id"]
                item.start = baseline["start"]
                item.stop = baseline["stop"]
            else:
                for values in item.outputs.values():
                    values.fill(0)

    def _apply_state(
        self, grid, study_id: str, item, parameters: Mapping[str, Any]
    ) -> dict[str, Any]:
        from gprMax.user_inputs import MainGridUserInput

        applied: dict[str, Any] = {}
        if "position" in parameters:
            point = tuple(float(value) for value in parameters["position"])
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError(
                    f"Study object '{study_id}' position must contain three finite values."
                )
            uip = MainGridUserInput(grid)
            point = uip.resolve_inf_point(point)
            _, coord = uip.check_src_rx_point(point, "#study")
            import gprMax.config as config

            mode = config.get_model_config().mode
            if mode.startswith("2D"):
                axis = "xyz".index(mode[-1])
                baseline_coord = self._runtime_baselines[study_id]["coord"]
                if coord[axis] != baseline_coord[axis]:
                    raise ValueError(
                        f"Study object '{study_id}' cannot move off the active {mode} invariant layer."
                    )
            item.coord = coord
        applied["position"] = [float(item.coord[i] * grid.dl[i]) for i in range(3)]

        if not _is_source(item):
            if parameters.get("record") is False:
                raise ValueError(
                    f"Study object '{study_id}' requested record=false. Selective receiver output "
                    "is reserved for a later study implementation; omit the row instead."
                )
            applied["record"] = True
            return applied

        if "waveform_id" in parameters:
            waveform_id = str(parameters["waveform_id"])
            if not any(waveform.ID == waveform_id for waveform in grid.waveforms):
                raise ValueError(
                    f"Study object '{study_id}' references unknown waveform '{waveform_id}'."
                )
            item.waveformID = waveform_id
        if "start" in parameters:
            item.start = float(parameters["start"])
        if "stop" in parameters:
            item.stop = float(parameters["stop"])
        if item.start < 0 or item.stop <= item.start or item.stop > grid.timewindow:
            raise ValueError(
                f"Study object '{study_id}' requires 0 <= start < stop <= the model time window."
            )
        scale = float(parameters.get("scale", 1.0))
        if not math.isfinite(scale):
            raise ValueError(f"Study object '{study_id}' scale must be finite.")
        if parameters.get("active") is False:
            scale = 0.0
        self._resample_source(grid, item, scale)
        applied.update(
            active=bool(scale != 0),
            waveform_id=item.waveformID,
            start=float(item.start),
            stop=float(item.stop),
            scale=scale,
        )
        return applied

    def _resample_source(self, grid, source, scale: float) -> None:
        waveform = next(w for w in grid.waveforms if w.ID == source.waveformID)
        values = np.zeros(grid.iterations + 1, dtype=grid.Ex.dtype)
        half_step = source.waveformvalues_halfdt is not None
        for iteration in range(grid.iterations + 1):
            time = grid.dt * iteration
            if source.start <= time <= source.stop:
                evaluation_time = time - source.start + (0.5 * grid.dt if half_step else 0.0)
                values[iteration] = scale * waveform.calculate_value(evaluation_time, grid.dt)
        if half_step:
            source.waveformvalues_halfdt = values
        else:
            source.waveformvalues_wholedt = values
        source.study_scale = scale


class GPRStudy(Study):
    """Convenience API for a ``gpr`` study."""

    def __init__(self, cases: Sequence[StudyCase]):
        super().__init__("gpr", cases)


def preflight_study_args(args) -> Optional[Study]:
    """Resolve API/hash study input before SimulationConfig sizes its run."""

    study = getattr(args, "study", None)
    if study is not None and not isinstance(study, Study):
        raise TypeError("The study API argument must be a Study instance.")

    hash_spec = _find_hash_study(getattr(args, "inputfile", None))
    if study is not None and hash_spec is not None:
        raise ValueError("Specify a study through either the Python API or #study, not both.")
    if study is None and hash_spec is not None:
        study_type, csv_path = hash_spec
        study = Study.from_csv(study_type, csv_path)
    if study is None:
        setattr(args, "study", None)
        return None

    if getattr(args, "taskfarm", False):
        raise ValueError("Studies do not yet support MPI task farming.")
    if getattr(args, "mpi", None) is not None:
        raise ValueError("Studies do not yet support MPI domain decomposition.")
    scenes = getattr(args, "scenes", None)
    if scenes is not None and len(scenes) != 1:
        raise ValueError("A study requires exactly one reusable Scene.")

    count = len(study.cases)
    start = getattr(args, "i", None)
    if start is None:
        args.n = count
    else:
        if start <= 0 or start > count:
            raise ValueError(
                f"Study restart index i={start} is outside the {count} available cases."
            )
        args.n = count - start + 1
    args.geometry_fixed = True
    args.study = study
    study.reset_runtime()
    return study


def _find_hash_study(inputfile: Optional[Union[str, Path]]) -> Optional[tuple[str, Path]]:
    if inputfile is None:
        return None
    main_path = Path(inputfile).expanduser().resolve()
    found: list[tuple[str, Path]] = []
    visited: set[Path] = set()

    def scan(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        in_python = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("##"):
                continue
            if stripped.startswith("#python:"):
                in_python = True
                continue
            if stripped.startswith("#end_python:"):
                in_python = False
                continue
            if in_python:
                # Preflight intentionally does not execute deprecated Python.
                continue
            if stripped.startswith("#include_file:"):
                values = shlex.split(stripped.split(":", 1)[1])
                if len(values) != 1:
                    raise ValueError("#include_file requires exactly one parameter.")
                include = Path(values[0]).expanduser()
                if not include.exists():
                    include = main_path.parent / include
                scan(include)
            elif stripped.startswith("#study:"):
                values = shlex.split(stripped.split(":", 1)[1])
                if len(values) != 2:
                    raise ValueError("#study requires exactly two parameters: type and CSV path.")
                csv_path = Path(values[1]).expanduser()
                if not csv_path.is_absolute():
                    csv_path = main_path.parent / csv_path
                found.append((values[0].lower(), csv_path.resolve()))

    scan(main_path)
    if len(found) > 1:
        raise ValueError("Only one #study command may be specified.")
    return found[0] if found else None


def _is_source(item: Any) -> bool:
    return hasattr(item, "waveformID") and (
        getattr(item, "waveformvalues_halfdt", None) is not None
        or getattr(item, "waveformvalues_wholedt", None) is not None
    )


def _parse_bool(value: str, path: Path, line: int, name: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"true", "yes", "1"}:
        return True
    if normalised in {"false", "no", "0"}:
        return False
    raise ValueError(f"Study CSV '{path}', line {line}: {name} must be true/false, yes/no, or 1/0.")


def _parse_finite_float(value: str, path: Path, line: int, name: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Study CSV '{path}', line {line}: {name} must be a number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"Study CSV '{path}', line {line}: {name} must be finite.")
    return result


def _jsonable(value: Any) -> Any:
    """Convert ordinary API parameter values to JSON-compatible builtins."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
