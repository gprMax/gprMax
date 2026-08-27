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

"""Reusable-geometry source, receiver, and port studies.

General GPR studies manage stateless local sources and ordinary receivers.
Port studies additionally manage source-bound voltage-port monitors and
assemble their frequency-domain response after the reused solves.
Eigenmode studies reuse phase-aligned FDFD modal bases and assemble a full
modal S matrix from one active port/mode channel per run.
Plane-wave studies rebuild only the auxiliary DPW source and declarative NTFF
accumulators while retaining the main Yee geometry.
Source studies reset fixed-topology transmission-line, magnetic-frill, and
rational-network terminals while varying only their generator drives.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import operator
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import h5py
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


_POSITION_COLUMNS = ("x_m", "y_m", "z_m")
_SOURCE_PARAMETERS = {"active", "position", "waveform_id", "start", "stop", "scale"}
_RECEIVER_PARAMETERS = {"position", "record"}
_PLANE_WAVE_COMMON_PARAMETERS = {"psi", "waveform_id", "start", "stop", "scale"}
_STATEFUL_SOURCE_PARAMETERS = {"active", "waveform_id", "start", "stop", "scale"}
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
    "port",
    "mode",
    "theta_deg",
    "phi_deg",
    "psi_deg",
    "axis",
    "m_x",
    "m_y",
    "m_z",
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
            raise ValueError(f"Study CSV '{path}' is missing required column(s): {', '.join(sorted(missing))}.")
        unknown = set(fieldnames) - _CSV_COLUMNS
        if unknown:
            raise ValueError(f"Study CSV '{path}' contains unsupported column(s): " f"{', '.join(sorted(unknown))}.")

        case_rows: dict[str, list[ObjectState]] = {}
        seen: set[tuple[str, str]] = set()
        for line_number, raw_row in enumerate(reader, start=2):
            row = {(key or "").strip(): (value or "").strip() for key, value in raw_row.items()}
            case_id = row.get("case_id", "")
            object_id = row.get("object_id", "")
            if not case_id or not object_id:
                raise ValueError(f"Study CSV '{path}', line {line_number}: case_id and object_id are required.")
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
                    f"Study CSV '{path}', line {line_number}: x_m, y_m, and z_m must be " "provided together."
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
                    parameters[parameter_name] = _parse_finite_float(row[csv_name], path, line_number, csv_name)
            for name in ("port", "mode"):
                if row.get(name, ""):
                    parameters[name] = _parse_positive_int(row[name], path, line_number, name)
            for csv_name, parameter_name in (
                ("theta_deg", "theta"),
                ("phi_deg", "phi"),
                ("psi_deg", "psi"),
            ):
                if row.get(csv_name, ""):
                    parameters[parameter_name] = _parse_finite_float(row[csv_name], path, line_number, csv_name)
            mapping_values = [row.get(name, "") for name in ("m_x", "m_y", "m_z")]
            if any(mapping_values) and not all(mapping_values):
                raise ValueError(
                    f"Study CSV '{path}', line {line_number}: m_x, m_y, and m_z must be " "provided together."
                )
            if all(mapping_values):
                parameters["m_vec"] = tuple(
                    _parse_int(value, path, line_number, name)
                    for name, value in zip(("m_x", "m_y", "m_z"), mapping_values)
                )
            if row.get("axis", ""):
                parameters["axis"] = row["axis"]
            if row.get("record", ""):
                parameters["record"] = _parse_bool(row["record"], path, line_number, "record")

            case_rows.setdefault(case_id, []).append(ObjectState(object_id, **parameters))

        cases = [StudyCase(case_id, states) for case_id, states in case_rows.items()]
        if str(type).strip().lower() == "port":
            return PortStudy(cases, source_path=path, source_text=text)
        if str(type).strip().lower() == "eigenmode":
            return EigenmodeStudy(cases, source_path=path, source_text=text)
        if str(type).strip().lower() == "plane_wave":
            return PlaneWaveStudy(cases, source_path=path, source_text=text)
        if str(type).strip().lower() == "source":
            return SourceStudy(cases, source_path=path, source_text=text)
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
            (VoltageSource, "voltage_source"),
            (HertzianDipole, "hertzian_dipole"),
            (MagneticDipole, "magnetic_dipole"),
            (Rx, "rx"),
        )
        registry: dict[str, Any] = {}
        aliases: dict[str, str] = {}
        reference_ids: dict[int, str] = {}
        all_objects = list(scene.grid_objects) + list(scene.output_objects)
        unsupported_excitation_types = (
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
                "voltage sources, HertzianDipole, and MagneticDipole excitations."
            )
        if self.type != "port" and any(isinstance(user_object, VoltageSource) for user_object in all_objects):
            raise ValueError(
                "VoltageSource reuse is available through PortStudy so its fixed terminal "
                "resistance and automatic source-owned port are validated together."
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
                "The study scene contains no supported objects. Studies currently support "
                "VoltageSource, HertzianDipole, MagneticDipole, and Rx objects."
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
                    raise ValueError(f"Study case '{case.id}' contains duplicate state for '{study_id}'.")
                case_seen.add(study_id)
                state.object = study_id
                if isinstance(registry[study_id], Rx):
                    allowed = _RECEIVER_PARAMETERS
                elif isinstance(registry[study_id], VoltageSource):
                    # A finite-resistance source changes the electric-edge
                    # update coefficients during geometry construction. Its
                    # position and resistance are therefore immutable in a
                    # reused model; only the generator drive may vary.
                    allowed = _SOURCE_PARAMETERS - {"position"}
                else:
                    allowed = _SOURCE_PARAMETERS
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
            + ", ".join(f"{study_id} ({type(user_object).__name__})" for study_id, user_object in registry.items())
            + "\n"
        )

    def apply_case(self, model) -> None:
        """Restore baselines and apply the current absolute case to a built model."""

        import gprMax.config as config

        case_index = config.sim_config.current_model
        if case_index < 0 or case_index >= len(self.cases):
            raise ValueError(f"Study case index {case_index + 1} is outside the {len(self.cases)} available cases.")

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
                    "objects": {str(state.object): _jsonable(state.parameters) for state in item.states},
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
        objects = model.G.voltagesources + model.G.hertziandipoles + model.G.magneticdipoles + model.G.rxs
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

    def _apply_state(self, grid, study_id: str, item, parameters: Mapping[str, Any]) -> dict[str, Any]:
        from gprMax.user_inputs import MainGridUserInput

        applied: dict[str, Any] = {}
        if "position" in parameters:
            point = tuple(float(value) for value in parameters["position"])
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError(f"Study object '{study_id}' position must contain three finite values.")
            uip = MainGridUserInput(grid)
            point = uip.resolve_inf_point(point)
            _, coord = uip.check_src_rx_point(point, "#study")
            import gprMax.config as config

            mode = config.get_model_config().mode
            if mode.startswith("2D"):
                axis = "xyz".index(mode[-1])
                baseline_coord = self._runtime_baselines[study_id]["coord"]
                if coord[axis] != baseline_coord[axis]:
                    raise ValueError(f"Study object '{study_id}' cannot move off the active {mode} invariant layer.")
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
                raise ValueError(f"Study object '{study_id}' references unknown waveform '{waveform_id}'.")
            item.waveformID = waveform_id
        if "start" in parameters:
            item.start = float(parameters["start"])
        if "stop" in parameters:
            item.stop = float(parameters["stop"])
        if item.start < 0 or item.stop <= item.start or item.stop > grid.timewindow:
            raise ValueError(f"Study object '{study_id}' requires 0 <= start < stop <= the model time window.")
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
        if source.waveformvalues_halfdt is not None:
            half_values = np.zeros(grid.iterations + 1, dtype=grid.Ex.dtype)
            for iteration in range(grid.iterations + 1):
                time = grid.dt * iteration
                if source.start <= time <= source.stop:
                    half_values[iteration] = scale * waveform.calculate_value(
                        time - source.start + 0.5 * grid.dt,
                        grid.dt,
                    )
            source.waveformvalues_halfdt = half_values
        if source.waveformvalues_wholedt is not None:
            whole_values = np.zeros(grid.iterations + 1, dtype=grid.Ex.dtype)
            for iteration in range(grid.iterations + 1):
                time = grid.dt * iteration
                if source.start <= time <= source.stop:
                    whole_values[iteration] = scale * waveform.calculate_value(
                        time - source.start,
                        grid.dt,
                    )
            source.waveformvalues_wholedt = whole_values
        source.study_scale = scale

    def collect_case(self, model) -> None:
        """Collect derived results after a case has solved."""

    def finalise(self):
        """Finalise any study-level result after all requested cases."""

        return None


class GPRStudy(Study):
    """Convenience API for a ``gpr`` study."""

    def __init__(self, cases: Sequence[StudyCase]):
        super().__init__("gpr", cases)


class SourceStudy(Study):
    """Reusable study for fixed-topology stateful terminal sources.

    Transmission-line resistance, magnetic-frill geometry/Z0, and rational
    network topology remain fixed. Cases may select the active generators and
    change only their waveform, time window, and scale. An omitted or inactive
    source retains its physical terminal with a zero generator drive.
    """

    supported_types = ("source",)

    def __init__(
        self,
        cases: Sequence[StudyCase],
        *,
        source_path: Optional[Union[str, Path]] = None,
        source_text: Optional[str] = None,
    ):
        super().__init__(
            "source",
            cases,
            source_path=source_path,
            source_text=source_text,
        )

    def bind_scene(self, scene) -> None:
        """Bind fixed main-grid terminal definitions and validate cases."""

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
            TransmissionLine,
            VoltageSource,
        )

        supported = (
            (TransmissionLine, "transmission_line"),
            (MagneticFrillSource, "magnetic_frill_source"),
            (NetworkExcitation, "network_excitation"),
        )
        unsupported_types = (
            VoltageSource,
            HertzianDipole,
            MagneticDipole,
            DiscretePlaneWaveAngles,
            DiscretePlaneWaveAxial,
            DiscretePlaneWaveVector,
            EigenmodeExcitation,
        )
        supported_types = tuple(item_type for item_type, _ in supported)
        all_source_types = supported_types + unsupported_types
        unsupported = [type(item).__name__ for item in scene.grid_objects if isinstance(item, unsupported_types)]
        subgrid_sources = [
            f"{type(item).__name__} on subgrid {subgrid.kwargs.get('id', '')!r}"
            for subgrid in scene.subgrid_objects
            for item in subgrid.children_grid
            if isinstance(item, all_source_types)
        ]
        if unsupported or subgrid_sources:
            details = sorted(set(unsupported + subgrid_sources))
            raise ValueError(
                "SourceStudy supports only main-grid TransmissionLine, "
                "MagneticFrillSource, and NetworkExcitation objects; remove " + ", ".join(details) + "."
            )

        registry: dict[str, Any] = {}
        reference_ids: dict[int, str] = {}
        for object_type, prefix in supported:
            index = 0
            for item in scene.grid_objects:
                if not isinstance(item, object_type):
                    continue
                index += 1
                study_id = f"{prefix}_{index}"
                setattr(item, "_study_id", study_id)
                registry[study_id] = item
                reference_ids[id(item)] = study_id

        if not registry:
            raise ValueError(
                "SourceStudy requires at least one TransmissionLine, " "MagneticFrillSource, or NetworkExcitation."
            )

        for case in self.cases:
            seen: set[str] = set()
            for state in case.states:
                if isinstance(state.object, str):
                    study_id = state.object.strip()
                else:
                    try:
                        study_id = reference_ids[id(state.object)]
                    except KeyError as exc:
                        raise ValueError(
                            f"SourceStudy case '{case.id}' references an object that is not "
                            "a supported main-grid source in its Scene."
                        ) from exc
                if study_id not in registry:
                    available = ", ".join(sorted(registry))
                    raise ValueError(
                        f"SourceStudy case '{case.id}' references unknown object "
                        f"'{study_id}'. Available objects: {available}."
                    )
                if study_id in seen:
                    raise ValueError(f"SourceStudy case '{case.id}' contains duplicate state for " f"'{study_id}'.")
                unknown = set(state.parameters) - _STATEFUL_SOURCE_PARAMETERS
                if unknown:
                    raise ValueError(
                        f"SourceStudy object '{study_id}' does not support parameter(s) "
                        f"{', '.join(sorted(unknown))}. Allowed parameters: "
                        f"{', '.join(sorted(_STATEFUL_SOURCE_PARAMETERS))}."
                    )
                if "scale" in state.parameters:
                    scale = float(state.parameters["scale"])
                    if not math.isfinite(scale):
                        raise ValueError(f"SourceStudy case '{case.id}' scale must be finite.")
                if "active" in state.parameters and not isinstance(state.parameters["active"], (bool, np.bool_)):
                    raise ValueError(f"SourceStudy case '{case.id}' active must be a boolean.")
                state.object = study_id
                seen.add(study_id)

        self.registry = registry
        self.aliases = {}
        self._scene_bound = True
        logger.info(
            "SourceStudy objects: "
            + ", ".join(f"{study_id} ({type(item).__name__})" for study_id, item in registry.items())
            + "\n"
        )

    def _runtime_registry(self, model) -> dict[str, Any]:
        runtime: dict[str, Any] = {}
        for item in (
            list(model.G.transmissionlines) + list(model.G.magneticfrillsources) + list(model.G.networkterminals)
        ):
            study_id = getattr(item, "study_id", None)
            if study_id:
                runtime[study_id] = item
        expected = set(self.registry)
        if set(runtime) != expected:
            missing = ", ".join(sorted(expected - set(runtime)))
            raise RuntimeError(f"SourceStudy runtime object binding is incomplete; missing: {missing}.")
        return runtime

    def _capture_baselines(self, runtime: Mapping[str, Any]) -> None:
        if self._runtime_baselines:
            return
        for study_id, item in runtime.items():
            self._runtime_baselines[study_id] = {
                "waveform_id": item.waveformID,
                "start": float(item.start),
                "stop": float(item.stop),
            }

    def apply_case(self, model) -> None:
        """Reset every terminal and apply the current absolute drive state."""

        import gprMax.config as config

        case_index = config.sim_config.current_model
        if case_index < 0 or case_index >= len(self.cases):
            raise ValueError(
                f"SourceStudy case index {case_index + 1} is outside the " f"{len(self.cases)} available cases."
            )

        runtime = self._runtime_registry(model)
        self._capture_baselines(runtime)
        states = {str(state.object): state for state in self.cases[case_index].states}
        resolved: dict[str, Any] = {
            "case_id": self.cases[case_index].id,
            "objects": {},
        }
        for study_id, item in runtime.items():
            baseline = self._runtime_baselines[study_id]
            parameters = dict(states[study_id].parameters) if study_id in states else {}
            active = bool(parameters.pop("active", study_id in states))
            waveform_id = str(parameters.pop("waveform_id", baseline["waveform_id"]))
            start = float(parameters.pop("start", baseline["start"]))
            stop = min(
                float(parameters.pop("stop", baseline["stop"])),
                float(model.G.timewindow),
            )
            scale = float(parameters.pop("scale", 1.0)) if active else 0.0
            item.configure_study_excitation(
                model.G,
                waveform_id=waveform_id,
                start=start,
                stop=stop,
                scale=scale,
            )
            resolved["objects"][study_id] = {
                "type": type(self.registry[study_id]).__name__,
                "active": bool(scale != 0),
                "waveform_id": waveform_id,
                "start": start,
                "stop": stop,
                "scale": scale,
            }
            if hasattr(item, "coord"):
                resolved["objects"][study_id]["position"] = [
                    float(item.coord[index] * model.G.dl[index]) for index in range(3)
                ]
            if hasattr(item, "ID"):
                resolved["objects"][study_id]["terminal_id"] = str(item.ID)

        for receiver in model.G.rxs:
            for values in receiver.outputs.values():
                values.fill(0)
        for monitor in getattr(model.G, "port_monitors", ()):
            reset = getattr(monitor, "reset_run_state", None)
            if reset is not None:
                reset()

        _recompile_declarative_ntff(model, model.G, "SourceStudy")
        self._current_resolved_case = resolved


class PlaneWaveStudy(Study):
    """Reusable-geometry study with one rebuilt discrete plane wave per case.

    The main Yee geometry is retained, but the auxiliary one-dimensional DPW
    grid is deliberately reconstructed because its integer direction mapping,
    length, projections, material profile, and PML state depend on the case.
    Declarative NTFF monitors are reconstructed with it so their accumulated
    DFT and incident-wave normalisation state cannot leak between cases.
    """

    supported_types = ("plane_wave",)

    def __init__(
        self,
        cases: Sequence[StudyCase],
        *,
        source_path: Optional[Union[str, Path]] = None,
        source_text: Optional[str] = None,
    ):
        super().__init__(
            "plane_wave",
            cases,
            source_path=source_path,
            source_text=source_text,
        )
        self._definition = None
        self._definition_type = None
        self._baseline_kwargs: dict[str, Any] = {}

    def reset_runtime(self) -> None:
        super().reset_runtime()
        self._definition = None
        self._definition_type = None
        self._baseline_kwargs = {}

    def bind_scene(self, scene) -> None:
        """Bind one main-grid DPW template and validate every case override."""

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
            TransmissionLine,
            VoltageSource,
        )

        plane_wave_types = (
            DiscretePlaneWaveAngles,
            DiscretePlaneWaveAxial,
            DiscretePlaneWaveVector,
        )
        containers = [("main grid", list(scene.grid_objects))]
        containers.extend(
            (
                f"subgrid {subgrid.kwargs.get('id', '')!r}",
                list(subgrid.children_grid),
            )
            for subgrid in scene.subgrid_objects
        )
        definitions = [item for item in scene.grid_objects if isinstance(item, plane_wave_types)]
        subgrid_plane_waves = [
            label for label, objects in containers[1:] for item in objects if isinstance(item, plane_wave_types)
        ]
        if subgrid_plane_waves:
            raise ValueError(
                "PlaneWaveStudy requires its plane-wave template on the main grid; "
                "plane waves were also found on " + ", ".join(sorted(set(subgrid_plane_waves))) + "."
            )
        if len(definitions) != 1:
            raise ValueError(
                "PlaneWaveStudy requires exactly one plane-wave template on the main grid; "
                f"found {len(definitions)}."
            )
        other_source_types = (
            VoltageSource,
            HertzianDipole,
            MagneticDipole,
            TransmissionLine,
            MagneticFrillSource,
            NetworkExcitation,
            EigenmodeExcitation,
        )
        other_sources = [
            f"{type(item).__name__} on {label}"
            for label, objects in containers
            for item in objects
            if isinstance(item, other_source_types)
        ]
        if other_sources:
            raise ValueError(
                "PlaneWaveStudy requires the plane wave to be the only excitation; remove "
                + ", ".join(sorted(set(other_sources)))
                + "."
            )

        definition = definitions[0]
        setattr(definition, "_study_id", "plane_wave_1")
        if isinstance(definition, DiscretePlaneWaveAngles):
            allowed = _PLANE_WAVE_COMMON_PARAMETERS | {"theta", "phi"}
        elif isinstance(definition, DiscretePlaneWaveVector):
            allowed = _PLANE_WAVE_COMMON_PARAMETERS | {"m_vec"}
        else:
            allowed = _PLANE_WAVE_COMMON_PARAMETERS | {"axis"}

        for case in self.cases:
            if len(case.states) != 1:
                raise ValueError(f"PlaneWaveStudy case '{case.id}' must contain exactly one plane-wave state.")
            state = case.states[0]
            if isinstance(state.object, str):
                if state.object.strip() != "plane_wave_1":
                    raise ValueError(
                        f"PlaneWaveStudy case '{case.id}' references '{state.object}', " "expected 'plane_wave_1'."
                    )
            elif state.object is not definition:
                raise ValueError(f"PlaneWaveStudy case '{case.id}' must reference its Scene's plane wave.")
            unknown = set(state.parameters) - allowed
            if unknown:
                raise ValueError(
                    f"PlaneWaveStudy case '{case.id}' has unsupported parameter(s) "
                    f"{', '.join(sorted(unknown))}. Allowed parameters: "
                    f"{', '.join(sorted(allowed))}."
                )
            if "m_vec" in state.parameters:
                mapping = tuple(state.parameters["m_vec"])
                if len(mapping) != 3 or not all(isinstance(value, (int, np.integer)) for value in mapping):
                    raise ValueError(f"PlaneWaveStudy case '{case.id}' m_vec must contain three integers.")
                if not any(mapping):
                    raise ValueError(f"PlaneWaveStudy case '{case.id}' m_vec cannot be the zero vector.")
                state.parameters["m_vec"] = mapping
            scale = float(state.parameters.get("scale", 1.0))
            if not math.isfinite(scale) or scale == 0:
                raise ValueError(f"PlaneWaveStudy case '{case.id}' scale must be finite and non-zero.")
            state.object = "plane_wave_1"

        self._definition = definition
        self._definition_type = type(definition)
        self._baseline_kwargs = copy.deepcopy(definition.kwargs)
        self.registry = {"plane_wave_1": definition}
        self.aliases = {}
        self._scene_bound = True
        logger.info(f"PlaneWaveStudy object: plane_wave_1 ({self._definition_type.__name__})\n")

    def apply_case(self, model) -> None:
        """Rebuild the case DPW and all declarative NTFF accumulator state."""

        import gprMax.config as config

        if self._definition_type is None:
            raise RuntimeError("PlaneWaveStudy was not bound to a Scene before model build.")
        case_index = config.sim_config.current_model
        if case_index < 0 or case_index >= len(self.cases):
            raise ValueError(
                f"PlaneWaveStudy case index {case_index + 1} is outside the " f"{len(self.cases)} available cases."
            )

        case = self.cases[case_index]
        parameters = dict(case.states[0].parameters)
        scale = float(parameters.pop("scale", 1.0))
        kwargs = copy.deepcopy(self._baseline_kwargs)
        kwargs.update(parameters)
        definition = self._definition_type(**kwargs)

        grid = model.G
        previous = list(grid.discreteplanewaves)
        try:
            definition.build(grid)
            plane_wave = grid.discreteplanewaves[-1]
            grid.discreteplanewaves[:] = [plane_wave]
            if plane_wave.axial != 0:
                plane_wave.grid_init(grid)
        except Exception:
            grid.discreteplanewaves[:] = previous
            raise

        for name in ("waveformvalues_wholedt", "waveformvalues_halfdt"):
            values = getattr(plane_wave, name, None)
            if values is not None:
                values *= scale
        plane_wave.study_id = "plane_wave_1"
        plane_wave.study_scale = scale

        # Receiver histories are ordinary per-run outputs, but reset_fields()
        # intentionally clears only field/PML state. Clear them explicitly.
        for receiver in grid.rxs:
            for values in receiver.outputs.values():
                values.fill(0)

        _recompile_declarative_ntff(model, grid, "PlaneWaveStudy")
        self._current_resolved_case = {
            "case_id": case.id,
            "objects": {
                "plane_wave_1": {
                    "type": self._definition_type.__name__,
                    "theta": float(plane_wave.theta),
                    "phi": float(plane_wave.phi),
                    "psi": float(plane_wave.psi),
                    "actual_angles": [float(value) for value in plane_wave.actual_angles],
                    "integer_mapping": [int(value) for value in plane_wave.m[:3]],
                    "waveform_id": plane_wave.waveformID,
                    "start": float(plane_wave.start),
                    "stop": float(plane_wave.stop),
                    "scale": scale,
                    "tfsf_corners": [int(value) for value in plane_wave.corners],
                }
            },
        }


def _recompile_declarative_ntff(model, grid, study_name: str) -> None:
    """Replace declarative NTFF monitors with pristine per-case instances."""

    if not grid.ntff_monitors and not grid.ntff_output_writers:
        return
    if not grid.ntff_output_writers:
        raise ValueError(
            f"{study_name} cannot reset directly constructed NTFF monitors; use the "
            "declarative NTFF surface/transform/output API or hash commands."
        )

    referenced = set()
    for writer in grid.ntff_output_writers:
        referenced.update(writer.frequency_monitors.values())
        for mapping_name in ("time_bindings", "time_far_bindings"):
            for binding in getattr(writer, mapping_name, {}).values():
                referenced.add(binding[0])
    unmanaged = [monitor for monitor in grid.ntff_monitors if monitor not in referenced]
    if unmanaged:
        raise ValueError(
            f"{study_name} found NTFF monitors outside the declarative reusable "
            "interface; these cannot be reset safely between cases."
        )

    grid.ntff_monitors.clear()
    grid.ntff_output_writers.clear()
    from gprMax.ntff.interface import compile_ntff_outputs

    compile_ntff_outputs(model, grid)


@dataclass(frozen=True)
class PortStudyResult:
    """Assembled source-plane and gap-corrected multiport S parameters.

    Attributes:
        frequency: frequency axis in Hz with shape ``(F,)``.
        port_ids: output-port ordering used by both S-matrix port axes.
        source_ids: voltage-source identifiers corresponding to ``port_ids``.
        case_ids: study-case ordering used for the driven input columns.
        reference_impedance: real port impedances in ohms with shape ``(P,)``.
        gap_correction: dimensionless parallel gap admittance with shape
            ``(F, P)``.
        s_source: uncorrected source-plane S matrix with shape ``(F, P, P)``.
        s: gap-corrected S matrix with shape ``(F, P, P)``; its axes are
            ``(frequency, output_port, input_port)``.
        valid_s_source: validity mask for ``s_source``.
        valid_s: validity mask for ``s``.
        output_file: aggregate study HDF5 path.
    """

    frequency: npt.NDArray[np.floating]
    port_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    reference_impedance: npt.NDArray[np.floating]
    gap_correction: npt.NDArray[np.complexfloating]
    s_source: npt.NDArray[np.complexfloating]
    s: npt.NDArray[np.complexfloating]
    valid_s_source: npt.NDArray[np.bool_]
    valid_s: npt.NDArray[np.bool_]
    output_file: Path


class PortStudy(Study):
    """One-active-port-per-case finite-resistance voltage-source study."""

    supported_types = ("port",)

    def __init__(
        self,
        cases: Sequence[StudyCase],
        *,
        source_path: Optional[Union[str, Path]] = None,
        source_text: Optional[str] = None,
    ):
        super().__init__(
            "port",
            cases,
            source_path=source_path,
            source_text=source_text,
        )
        self._case_drive_ids: list[str] = []
        self._port_source_ids: tuple[str, ...] = ()
        self._port_ids: tuple[str, ...] = ()
        self._port_monitors: dict[str, Any] = {}
        self._columns: dict[str, dict[str, Any]] = {}
        self._current_port_column: Optional[dict[str, Any]] = None
        self._aggregate_output_path: Optional[Path] = None
        self.result: Optional[PortStudyResult] = None

    def reset_runtime(self) -> None:
        super().reset_runtime()
        self._case_drive_ids = []
        self._port_source_ids = ()
        self._port_ids = ()
        self._port_monitors.clear()
        self._columns.clear()
        self._current_port_column = None
        self._aggregate_output_path = None
        self.result = None

    def bind_scene(self, scene) -> None:
        """Validate the fixed finite-resistance port topology and drive cases."""

        super().bind_scene(scene)
        from gprMax.user_objects.cmds_multiuse import HertzianDipole, MagneticDipole, VoltageSource

        other_sources = [
            study_id
            for study_id, user_object in self.registry.items()
            if isinstance(user_object, (HertzianDipole, MagneticDipole))
        ]
        if other_sources:
            raise ValueError(
                "PortStudy cases may contain only finite-resistance VoltageSource "
                f"excitations; remove {', '.join(other_sources)}."
            )
        voltage_ids = [
            study_id for study_id, user_object in self.registry.items() if isinstance(user_object, VoltageSource)
        ]
        if not voltage_ids:
            raise ValueError("PortStudy requires at least one finite-resistance VoltageSource.")
        for study_id in voltage_ids:
            source = self.registry[study_id]
            if not np.isfinite(source.resistance) or source.resistance <= 0:
                raise ValueError(
                    f"PortStudy source '{study_id}' must have a finite resistance greater "
                    "than zero; hard voltage sources are not matched passive ports."
                )

        drive_ids: list[str] = []
        for case in self.cases:
            active = []
            for state in case.states:
                study_id = str(state.object)
                if study_id not in voltage_ids:
                    continue
                scale = float(state.parameters.get("scale", 1.0))
                if state.parameters.get("active") is not False and scale != 0:
                    active.append(study_id)
            if len(active) != 1:
                raise ValueError(
                    f"PortStudy case '{case.id}' must explicitly drive exactly one "
                    f"VoltageSource; found {len(active)}. Omitted sources remain passive."
                )
            drive_ids.append(active[0])
        if len(set(drive_ids)) != len(drive_ids):
            raise ValueError("PortStudy must drive every voltage port in exactly one case.")
        if set(drive_ids) != set(voltage_ids):
            missing = ", ".join(study_id for study_id in voltage_ids if study_id not in drive_ids)
            raise ValueError(f"PortStudy has no driven case for: {missing}.")
        self._case_drive_ids = drive_ids

    def apply_case(self, model) -> None:
        for monitor in getattr(model.G, "port_monitors", ()):
            reset = getattr(monitor, "reset_run_state", None)
            if reset is not None:
                reset()
        super().apply_case(model)
        self._bind_runtime_ports(model)

        import gprMax.config as config

        drive_id = self._case_drive_ids[config.sim_config.current_model]
        resolved = self._current_resolved_case or {}
        active = [
            study_id
            for study_id in self._port_source_ids
            if resolved.get("objects", {}).get(study_id, {}).get("active", False)
        ]
        if active != [drive_id]:
            raise RuntimeError(
                f"PortStudy case '{resolved.get('case_id', '')}' resolved active ports "
                f"{active}, expected [{drive_id!r}]."
            )
        self._current_drive_id = drive_id

    def _bind_runtime_ports(self, model) -> None:
        if self._port_monitors:
            return
        from gprMax.user_objects.cmds_multiuse import VoltageSource

        voltage_ids = tuple(
            study_id for study_id, user_object in self.registry.items() if isinstance(user_object, VoltageSource)
        )
        monitors = {}
        for monitor in getattr(model.G, "port_monitors", ()):
            study_id = getattr(monitor.source, "study_id", None)
            if study_id in voltage_ids:
                if study_id in monitors:
                    raise ValueError(f"PortStudy source '{study_id}' has more than one port monitor.")
                monitors[study_id] = monitor
        if set(monitors) != set(voltage_ids):
            missing = ", ".join(study_id for study_id in voltage_ids if study_id not in monitors)
            raise ValueError(f"PortStudy has no source-owned port output for: {missing}.")
        port_ids = tuple(monitors[study_id].output_id for study_id in voltage_ids)
        if len(port_ids) != len(set(port_ids)):
            raise ValueError("PortStudy voltage-source port IDs must be unique.")
        self._port_source_ids = voltage_ids
        self._port_ids = port_ids
        self._port_monitors = monitors

    def collect_case(self, model) -> None:
        """Collect one power-normalised source-plane S-matrix column."""

        import gprMax.config as config

        self._bind_runtime_ports(model)
        drive_id = self._current_drive_id
        drive_index = self._port_source_ids.index(drive_id)
        ordered = [self._port_monitors[study_id] for study_id in self._port_source_ids]
        results = [monitor.result for monitor in ordered]
        if any(result is None for result in results):
            raise RuntimeError("PortStudy results were collected before every voltage-source port finalised.")

        frequency = np.asarray(results[0].frequency)
        for port_id, result in zip(self._port_ids[1:], results[1:]):
            if not np.array_equal(result.frequency, frequency):
                raise ValueError(
                    f"PortStudy port '{port_id}' has a different frequency axis; use the "
                    "same spectrum_limit for every voltage-source port."
                )

        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        impedances = np.asarray([monitor.reference_impedance for monitor in ordered], dtype=real_dtype)
        drive_incident = np.asarray(results[drive_index].incident_spectrum, dtype=complex_dtype)
        drive_valid = np.asarray(results[drive_index].source_valid, dtype=bool)
        column = np.full((frequency.size, len(ordered)), np.nan + 1j * np.nan, dtype=complex_dtype)
        valid_column = np.zeros(column.shape, dtype=bool)
        for output_index, result in enumerate(results):
            ratio, defined = _safe_divide(
                np.asarray(result.reflected_source_spectrum, dtype=complex_dtype),
                drive_incident,
                complex_dtype,
            )
            ratio *= np.sqrt(impedances[drive_index] / impedances[output_index])
            column[:, output_index] = ratio
            valid_column[:, output_index] = defined & drive_valid & np.asarray(result.mesh_valid, dtype=bool)

        gap_correction = np.stack(
            [np.asarray(result.gap_correction, dtype=complex_dtype) for result in results], axis=1
        )
        # Scalar S11 validity on an unexcited port is false by definition,
        # but the port's gap admittance remains perfectly well defined and
        # is required for the matrix correction. Build this mask from the
        # gap data itself and exclude only the exact discrete Nyquist pole.
        gap_valid = np.logical_and.reduce([np.isfinite(result.gap_correction) for result in results])
        for monitor in ordered:
            if monitor.gap_capacitance != 0:
                gap_valid &= ~np.isclose(
                    frequency,
                    monitor.nyquist_frequency,
                    rtol=64 * np.finfo(real_dtype).eps,
                    atol=0,
                )
        case_index = config.sim_config.current_model
        case_data = {
            "case_id": self.cases[case_index].id,
            "case_index": case_index,
            "drive_id": drive_id,
            "drive_port_id": self._port_ids[drive_index],
            "frequency": np.asarray(frequency, dtype=real_dtype),
            "reference_impedance": impedances,
            "gap_correction": gap_correction,
            "gap_valid": gap_valid,
            "column": column,
            "valid_column": valid_column,
        }
        self._columns[drive_id] = case_data
        self._current_port_column = case_data
        if self._aggregate_output_path is None:
            model_config = config.get_model_config()
            base = Path(model_config.output_file_path)
            suffix = model_config.appendmodelnumber
            if suffix and base.name.endswith(suffix):
                base = base.with_name(base.name[: -len(suffix)])
            self._aggregate_output_path = base.with_name(base.name + "_study").with_suffix(".h5")

    def write_hdf5(self, h5file) -> None:
        super().write_hdf5(h5file)
        if self._current_port_column is None:
            raise RuntimeError("PortStudy case data was not collected before HDF5 output.")
        data = self._current_port_column
        group = h5file["study"].create_group("port_response")
        group.attrs["DrivenSourceID"] = data["drive_id"]
        group.attrs["DrivenPortID"] = data["drive_port_id"]
        group.attrs["MatrixConvention"] = "S[frequency, output_port, input_port]"
        group.create_dataset("port_ids", data=np.asarray(self._port_ids, dtype="S"))
        group.create_dataset("source_ids", data=np.asarray(self._port_source_ids, dtype="S"))
        group.create_dataset("frequency", data=data["frequency"])
        group.create_dataset("reference_impedance", data=data["reference_impedance"])
        group.create_dataset("gap_correction_c", data=data["gap_correction"])
        group.create_dataset("S_source_column", data=data["column"])
        group.create_dataset("valid_S_source_column", data=data["valid_column"].astype(np.uint8))
        group.create_dataset("gap_correction_valid", data=data["gap_valid"].astype(np.uint8))
        if self._aggregate_output_path is not None:
            group.attrs["AggregateOutput"] = str(self._aggregate_output_path)

    def finalise(self) -> Optional[PortStudyResult]:
        """Assemble, gap-correct, store, and expose the multiport S matrix."""

        if not self._columns or self._aggregate_output_path is None:
            return None
        first = next(iter(self._columns.values()))
        frequency = first["frequency"]
        nfrequency = frequency.size
        nports = len(self._port_source_ids)
        complex_dtype = first["column"].dtype
        s_source = np.full((nfrequency, nports, nports), np.nan + 1j * np.nan, dtype=complex_dtype)
        valid_source = np.zeros(s_source.shape, dtype=bool)
        case_ids = [""] * nports
        if len(self._columns) < nports and self._aggregate_output_path.exists():
            with h5py.File(self._aggregate_output_path, "r") as previous:
                previous_ports = tuple(item.decode() for item in previous["port_ids"][...])
                previous_sources = tuple(item.decode() for item in previous["source_ids"][...])
                compatible = (
                    previous.attrs.get("StudyType", "") == "port"
                    and previous_ports == self._port_ids
                    and previous_sources == self._port_source_ids
                    and np.array_equal(previous["frequency"][...], frequency)
                    and np.allclose(
                        previous["reference_impedance"][...],
                        first["reference_impedance"],
                    )
                    and np.allclose(
                        previous["gap_correction_c"][...],
                        first["gap_correction"],
                        equal_nan=True,
                    )
                )
                if not compatible:
                    raise ValueError(
                        f"Existing PortStudy output '{self._aggregate_output_path}' is not "
                        "compatible with this restarted study. Remove or rename it, or use "
                        "a different output base."
                    )
                s_source[...] = previous["S_source"][...]
                valid_source[...] = previous["valid_S_source"][...].astype(bool)
                case_ids = [item.decode() for item in previous["case_ids"][...]]
        for input_index, source_id in enumerate(self._port_source_ids):
            data = self._columns.get(source_id)
            if data is None:
                continue
            if not np.array_equal(data["frequency"], frequency):
                raise RuntimeError("PortStudy collected inconsistent frequency axes.")
            if not np.allclose(data["gap_correction"], first["gap_correction"], equal_nan=True):
                raise RuntimeError("PortStudy gap correction changed between reused runs.")
            s_source[:, :, input_index] = data["column"]
            valid_source[:, :, input_index] = data["valid_column"]
            case_ids[input_index] = data["case_id"]

        if not all(case_ids):
            missing_ports = ", ".join(self._port_ids[index] for index, case_id in enumerate(case_ids) if not case_id)
            logger.warning(
                "PortStudy output is incomplete; no compatible prior aggregate supplied "
                f"columns for: {missing_ports}. Corrected matrix values remain invalid."
            )

        from gprMax.ports import correct_smatrix_for_parallel_gaps

        s_corrected, matrix_valid = correct_smatrix_for_parallel_gaps(
            s_source,
            first["gap_correction"],
            complex_dtype,
        )
        matrix_valid &= first["gap_valid"] & np.all(valid_source, axis=(1, 2))
        valid_corrected = np.broadcast_to(matrix_valid[:, np.newaxis, np.newaxis], s_source.shape).copy()
        s_corrected[~valid_corrected] = np.nan + 1j * np.nan

        self.result = PortStudyResult(
            frequency=np.asarray(frequency),
            port_ids=self._port_ids,
            source_ids=self._port_source_ids,
            case_ids=tuple(case_ids),
            reference_impedance=np.asarray(first["reference_impedance"]),
            gap_correction=np.asarray(first["gap_correction"]),
            s_source=s_source,
            s=s_corrected,
            valid_s_source=valid_source,
            valid_s=valid_corrected,
            output_file=self._aggregate_output_path,
        )
        self._write_aggregate_hdf5()
        logger.info(f"Written PortStudy output file: {self._aggregate_output_path.name}\n")
        return self.result

    def _write_aggregate_hdf5(self) -> None:
        if self.result is None:
            return
        from gprMax._version import __version__
        from gprMax.ntff.conventions import FORWARD_TRANSFORM_KERNEL, PHASOR_TIME_DEPENDENCE

        result = self.result
        with h5py.File(result.output_file, "w") as output:
            output.attrs["gprMax"] = __version__
            output.attrs["StudyType"] = self.type
            output.attrs["MatrixConvention"] = "S[frequency, output_port, input_port]"
            output.attrs["WaveNormalisation"] = "power_wave_real_reference_impedance"
            output.attrs["GapCorrection"] = "multiport_diagonal_shunt_admittance"
            output.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
            output.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
            output.attrs["CasesCompleted"] = sum(bool(case_id) for case_id in result.case_ids)
            output.attrs["CaseCount"] = len(self.cases)
            output.attrs["Complete"] = all(bool(case_id) for case_id in result.case_ids)
            if self.source_path is not None:
                output.attrs["SourcePath"] = str(self.source_path)
            if self.source_text is not None:
                output.create_dataset("source", data=self.source_text)
            output.create_dataset("frequency", data=result.frequency)
            output.create_dataset("port_ids", data=np.asarray(result.port_ids, dtype="S"))
            output.create_dataset("source_ids", data=np.asarray(result.source_ids, dtype="S"))
            output.create_dataset("case_ids", data=np.asarray(result.case_ids, dtype="S"))
            output.create_dataset("reference_impedance", data=result.reference_impedance)
            output.create_dataset("gap_correction_c", data=result.gap_correction)
            output.create_dataset("S_source", data=result.s_source)
            output.create_dataset("S", data=result.s)
            output.create_dataset("valid_S_source", data=result.valid_s_source.astype(np.uint8))
            output.create_dataset("valid_S", data=result.valid_s.astype(np.uint8))


def _deembed_modal_responses(
    incident_matrix,
    response_matrix,
    *,
    incident_valid=None,
    response_valid=None,
):
    """Recover basis responses from independent modal excitation runs.

    The measured incident waves form ``A[frequency, channel, case]`` and an
    arbitrary linear response forms ``R[frequency, observation, ..., case]``.
    Since ``R = X A``, the desired response ``X`` is obtained with a
    conditioned solve of ``A.T X.T = R.T`` at every frequency.  The final
    response axis is therefore the incident modal-basis channel, not the run.

    Invalid observations are rejected independently, while an invalid or
    ill-conditioned incident basis rejects the complete frequency bin.  This
    helper deliberately uses :func:`numpy.linalg.solve`, never an explicit
    matrix inverse.
    """

    from gprMax.eigenmode_ports import (
        CONDITION_RELATIVE_ERROR_BUDGET,
        MAX_CONDITION_NUMBER,
    )

    incident = np.asarray(incident_matrix)
    response = np.asarray(response_matrix)
    if incident.ndim != 3 or incident.shape[1] != incident.shape[2]:
        raise ValueError("incident_matrix must have shape (frequency, channel, excitation_case)")
    if response.ndim < 2 or response.shape[0] != incident.shape[0]:
        raise ValueError("response_matrix must use the incident-matrix frequency axis")
    if response.shape[-1] != incident.shape[-1]:
        raise ValueError("response_matrix final axis must be the excitation-case axis")

    if incident_valid is None:
        incident_mask = np.ones(incident.shape, dtype=bool)
    else:
        incident_mask = np.asarray(incident_valid, dtype=bool)
        if incident_mask.shape != incident.shape:
            raise ValueError("incident_valid must have the incident_matrix shape")
    if response_valid is None:
        response_mask = np.ones(response.shape, dtype=bool)
    else:
        response_mask = np.asarray(response_valid, dtype=bool)
        try:
            response_mask = np.broadcast_to(response_mask, response.shape)
        except ValueError as exc:
            raise ValueError("response_valid must be broadcastable to the response_matrix shape") from exc

    complex_dtype = np.result_type(incident.dtype, response.dtype)
    result = np.full(response.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
    observation_valid = np.zeros(response.shape[:-1], dtype=bool)
    condition_number = np.full(incident.shape[0], np.inf, dtype=np.float64)
    deembedding_valid = np.zeros(incident.shape[0], dtype=bool)
    component_dtype = np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64
    condition_limit = min(
        MAX_CONDITION_NUMBER,
        CONDITION_RELATIVE_ERROR_BUDGET / np.finfo(component_dtype).eps,
    )

    for frequency_index in range(incident.shape[0]):
        local_incident = incident[frequency_index]
        if not (np.all(incident_mask[frequency_index]) and np.all(np.isfinite(local_incident))):
            continue
        try:
            local_condition = float(np.linalg.cond(local_incident))
        except np.linalg.LinAlgError:
            continue
        condition_number[frequency_index] = local_condition
        if not np.isfinite(local_condition) or local_condition > condition_limit:
            continue

        local_response = response[frequency_index].reshape(-1, incident.shape[-1])
        local_mask = response_mask[frequency_index].reshape(-1, incident.shape[-1])
        usable = np.all(local_mask & np.isfinite(local_response), axis=1)
        if np.any(usable):
            try:
                solved = np.linalg.solve(
                    local_incident.T,
                    local_response[usable].T,
                ).T
            except np.linalg.LinAlgError:
                continue
            finite = np.all(np.isfinite(solved), axis=1)
            usable_indices = np.flatnonzero(usable)
            solved_indices = usable_indices[finite]
            result[frequency_index].reshape(-1, incident.shape[-1])[solved_indices] = solved[finite]
            observation_valid[frequency_index].reshape(-1)[solved_indices] = True
        deembedding_valid[frequency_index] = True

    return result, observation_valid, condition_number, deembedding_valid


@dataclass(frozen=True)
class EigenmodeStudyResult:
    """Assembled modal S matrix from one active port/mode per run.

    Attributes:
        frequency: frequency axis in Hz with shape ``(F,)``.
        channel_ports: one-based port number for each modal channel.
        channel_modes: one-based mode number for each modal channel.
        case_ids: identifiers of the independent excitation cases.
        s: generalized modal S matrix with shape ``(F, C, C)`` and axes
            ``(frequency, output_channel, input_channel)``.
        valid_s: physical propagating-power validity mask for ``s``.
        generalized_valid_s: generalized-coefficient validity mask for ``s``.
        output_file: aggregate study HDF5 path.
        embedded_far_fields: mapping from ``transform_id/output_id`` to
            de-embedded :class:`EmbeddedFarFieldBank` objects.
        incident_matrix: measured incident modal matrix for all cases, when
            retained.
        outgoing_matrix: measured outgoing modal matrix for all cases, when
            retained.
        wave_valid_matrix: physical power-wave validity for the measured
            modal matrices.
        generalized_wave_valid_matrix: generalized-coefficient validity for
            the measured modal matrices.
        deembedding_condition_number: condition number of the incident-wave
            solve at each frequency.
        deembedding_valid: validity of that solve at each frequency.
    """

    frequency: npt.NDArray[np.floating]
    channel_ports: npt.NDArray[np.integer]
    channel_modes: npt.NDArray[np.integer]
    case_ids: tuple[str, ...]
    s: npt.NDArray[np.complexfloating]
    valid_s: npt.NDArray[np.bool_]
    generalized_valid_s: npt.NDArray[np.bool_]
    output_file: Path
    embedded_far_fields: Mapping[str, "EmbeddedFarFieldBank"] = field(default_factory=dict)
    incident_matrix: Optional[npt.NDArray[np.complexfloating]] = None
    outgoing_matrix: Optional[npt.NDArray[np.complexfloating]] = None
    wave_valid_matrix: Optional[npt.NDArray[np.bool_]] = None
    generalized_wave_valid_matrix: Optional[npt.NDArray[np.bool_]] = None
    deembedding_condition_number: Optional[npt.NDArray[np.floating]] = None
    deembedding_valid: Optional[npt.NDArray[np.bool_]] = None

    @classmethod
    def from_hdf5(cls, filename: Union[str, Path]) -> "EigenmodeStudyResult":
        """Load an aggregate eigenmode study for offline array synthesis."""

        path = Path(filename).expanduser().resolve()
        with h5py.File(path, "r") as source:
            if source.attrs.get("StudyType", "") != "eigenmode":
                raise ValueError(f"{path!s} is not an aggregate EigenmodeStudy output")
            banks = {}
            root = source.get("embedded_far_fields")
            if root is not None:
                for transform_id, transform_group in root.items():
                    for output_id, group in transform_group.items():
                        bank = EmbeddedFarFieldBank(
                            transform_id=transform_id,
                            output_id=output_id,
                            frequency=group["frequency"][...],
                            theta=group["theta"][...],
                            phi=group["phi"][...],
                            etheta=group["Etheta"][...],
                            ephi=group["Ephi"][...],
                            sphere_theta=group["sphere/theta"][...],
                            sphere_phi=group["sphere/phi"][...],
                            sphere_weights=group["sphere/weights"][...],
                            sphere_etheta=group["sphere/Etheta"][...],
                            sphere_ephi=group["sphere/Ephi"][...],
                            valid=group["valid"][...].astype(bool),
                            impedance=float(group.attrs["Impedance"]),
                            theta_order=int(group.attrs["ThetaOrder"]),
                            phi_order=int(group.attrs["PhiOrder"]),
                            enclosure_radius=float(group.attrs["EnclosureRadius"]),
                        )
                        banks[bank.key] = bank
            return cls(
                frequency=source["frequency"][...],
                channel_ports=source["channel_ports"][...],
                channel_modes=source["channel_modes"][...],
                case_ids=tuple(
                    item.decode() if isinstance(item, bytes) else str(item) for item in source["case_ids"][...]
                ),
                s=source["S"][...],
                valid_s=source["valid_S"][...].astype(bool),
                generalized_valid_s=source["generalized_valid_S"][...].astype(bool),
                output_file=path,
                embedded_far_fields=banks,
                incident_matrix=(source["incident_matrix"][...] if "incident_matrix" in source else None),
                outgoing_matrix=(source["outgoing_matrix"][...] if "outgoing_matrix" in source else None),
                wave_valid_matrix=(
                    source["valid_wave_matrix"][...].astype(bool) if "valid_wave_matrix" in source else None
                ),
                generalized_wave_valid_matrix=(
                    source["generalized_valid_wave_matrix"][...].astype(bool)
                    if "generalized_valid_wave_matrix" in source
                    else None
                ),
                deembedding_condition_number=(
                    source["deembedding_condition_number"][...] if "deembedding_condition_number" in source else None
                ),
                deembedding_valid=(
                    source["deembedding_valid"][...].astype(bool) if "deembedding_valid" in source else None
                ),
            )

    def excitation_weights(self, excitations: Iterable["ModalWeight"]):
        """Return frequency-dependent incident power-wave weights."""

        return modal_array_weights(
            self.frequency,
            self.channel_ports,
            self.channel_modes,
            excitations,
        )

    def outgoing(self, excitations: Iterable["ModalWeight"]):
        """Apply the assembled S matrix to a weighted incident-wave vector.

        Zero-weight channels are omitted rather than multiplied by their
        stored NaNs.  An output is NaN when any active input lacks a valid
        generalized coefficient to that output.  Use ``valid_s`` separately
        when the result must also represent propagating real-power waves.
        """

        incident = self.excitation_weights(excitations)
        active = np.abs(incident) > 0
        selected_s = np.where(active[:, np.newaxis, :], self.s, 0)
        result = np.einsum("foi,fi->fo", selected_s, incident)
        valid = np.all(
            ~active[:, np.newaxis, :] | self.generalized_valid_s,
            axis=-1,
        )
        result[~valid] = np.nan + 1j * np.nan
        return result

    def evaluate_array_state(self, state: "ArrayState") -> "ArrayStateResult":
        """Evaluate one post-processed array state from the embedded basis."""

        return evaluate_array_state(self, state)

    def evaluate_codebook(self, codebook: "ArrayCodebook") -> tuple["ArrayStateResult", ...]:
        """Evaluate every named state in a validated array codebook."""

        if not isinstance(codebook, ArrayCodebook):
            raise TypeError("codebook must be an ArrayCodebook instance")
        return tuple(self.evaluate_array_state(state) for state in codebook.states)


@dataclass(frozen=True)
class ModalWeight:
    """One modal array weight using power, phase, and/or true time delay."""

    port: int
    mode: int
    power: float = 1.0
    phase_deg: float = 0.0
    delay_s: float = 0.0


@dataclass(frozen=True)
class EmbeddedFarFieldSpec:
    """One frequency-domain far-field request retained as an embedded basis."""

    transform_id: str
    output_id: str

    def __post_init__(self):
        for label, value in (("transform_id", self.transform_id), ("output_id", self.output_id)):
            if not isinstance(value, str) or not value.strip() or "/" in value or "\x00" in value:
                raise ValueError(f"Embedded far-field {label} must be a valid non-empty ID")

    @property
    def key(self) -> str:
        return f"{self.transform_id}/{self.output_id}"


@dataclass(frozen=True)
class ArrayState:
    """A named set of modal power, phase-shifter, and true-time-delay weights."""

    id: str
    drives: tuple[ModalWeight, ...]
    description: str = ""

    def __init__(
        self,
        id: str,
        drives: Iterable[ModalWeight],
        *,
        description: str = "",
    ):
        state_id = str(id).strip()
        if not state_id or "/" in state_id or "\x00" in state_id:
            raise ValueError("Array-state ID must be a valid non-empty HDF5 path component")
        values = tuple(drives)
        if not values:
            raise ValueError(f"Array state {state_id!r} requires at least one modal drive")
        if not all(isinstance(item, ModalWeight) for item in values):
            raise TypeError("Array-state drives must all be ModalWeight instances")
        channels = [(int(item.port), int(item.mode)) for item in values]
        if len(channels) != len(set(channels)):
            raise ValueError(f"Array state {state_id!r} contains a duplicate port/mode drive")
        object.__setattr__(self, "id", state_id)
        object.__setattr__(self, "drives", values)
        object.__setattr__(self, "description", str(description))


@dataclass(frozen=True)
class ArrayCodebook:
    """Versioned modal-array states and requested embedded far-field bases."""

    states: tuple[ArrayState, ...]
    embedded_far_fields: tuple[EmbeddedFarFieldSpec, ...] = ()
    description: str = ""
    source_path: Optional[Path] = None
    source_text: Optional[str] = None

    schema = "gprMax-array-codebook-v1"

    def __init__(
        self,
        states: Iterable[ArrayState],
        *,
        embedded_far_fields: Iterable[EmbeddedFarFieldSpec] = (),
        description: str = "",
        source_path: Optional[Union[str, Path]] = None,
        source_text: Optional[str] = None,
    ):
        state_values = tuple(states)
        field_values = tuple(embedded_far_fields)
        if not all(isinstance(item, ArrayState) for item in state_values):
            raise TypeError("ArrayCodebook states must all be ArrayState instances")
        if not all(isinstance(item, EmbeddedFarFieldSpec) for item in field_values):
            raise TypeError("ArrayCodebook embedded_far_fields must all be EmbeddedFarFieldSpec instances")
        state_ids = [item.id for item in state_values]
        field_ids = [item.key for item in field_values]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("ArrayCodebook state IDs must be unique")
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("ArrayCodebook embedded far-field selections must be unique")
        object.__setattr__(self, "states", state_values)
        object.__setattr__(self, "embedded_far_fields", field_values)
        object.__setattr__(self, "description", str(description))
        object.__setattr__(
            self,
            "source_path",
            None if source_path is None else Path(source_path).expanduser().resolve(),
        )
        object.__setattr__(self, "source_text", source_text)

    def to_definition(self) -> dict[str, Any]:
        """Return the complete canonical, versioned codebook definition."""

        return _jsonable(
            {
                "schema": self.schema,
                "description": self.description,
                "embedded_far_fields": [
                    {
                        "transform_id": item.transform_id,
                        "output_id": item.output_id,
                    }
                    for item in self.embedded_far_fields
                ],
                "states": [
                    {
                        "id": state.id,
                        "description": state.description,
                        "drives": [
                            {
                                "port": drive.port,
                                "mode": drive.mode,
                                "power_w": drive.power,
                                "phase_deg": drive.phase_deg,
                                "delay_s": drive.delay_s,
                            }
                            for drive in state.drives
                        ],
                    }
                    for state in self.states
                ],
            }
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize the complete canonical codebook as JSON."""

        return json.dumps(self.to_definition(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, filename: Union[str, Path]) -> "ArrayCodebook":
        """Load and strictly validate a versioned JSON array codebook."""

        path = Path(filename).expanduser().resolve()
        text = path.read_text(encoding="utf-8")
        try:
            definition = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Array codebook {path!s} is not valid JSON: {exc}") from exc
        if not isinstance(definition, dict):
            raise ValueError("Array codebook root must be a JSON object")
        allowed = {"schema", "description", "embedded_far_fields", "states"}
        unknown = set(definition) - allowed
        if unknown:
            raise ValueError(f"Array codebook has unknown key(s): {', '.join(sorted(unknown))}")
        if definition.get("schema") != cls.schema:
            raise ValueError(f"Array codebook schema must be {cls.schema!r}")

        selections = []
        for index, item in enumerate(definition.get("embedded_far_fields", ())):
            if not isinstance(item, dict) or set(item) != {"transform_id", "output_id"}:
                raise ValueError("Each embedded_far_fields entry must contain only transform_id and output_id")
            selections.append(EmbeddedFarFieldSpec(**item))

        states = []
        for state_index, item in enumerate(definition.get("states", ())):
            if not isinstance(item, dict):
                raise ValueError(f"Array codebook state {state_index + 1} must be an object")
            unknown = set(item) - {"id", "description", "drives"}
            if unknown or "id" not in item or "drives" not in item:
                raise ValueError(
                    f"Array codebook state {state_index + 1} requires id and drives and has "
                    f"unknown key(s): {', '.join(sorted(unknown)) or 'none'}"
                )
            if not isinstance(item["drives"], list):
                raise ValueError(f"Array codebook state {item['id']!r} drives must be a list")
            drives = []
            for drive_index, drive in enumerate(item["drives"]):
                if not isinstance(drive, dict):
                    raise ValueError(
                        f"Array codebook state {item['id']!r} drive {drive_index + 1} " "must be an object"
                    )
                unknown = set(drive) - {"port", "mode", "power_w", "phase_deg", "delay_s"}
                if unknown or "port" not in drive or "mode" not in drive:
                    raise ValueError(
                        f"Array drive requires port and mode and has unknown key(s): "
                        f"{', '.join(sorted(unknown)) or 'none'}"
                    )
                drives.append(
                    ModalWeight(
                        port=drive["port"],
                        mode=drive["mode"],
                        power=drive.get("power_w", 1.0),
                        phase_deg=drive.get("phase_deg", 0.0),
                        delay_s=drive.get("delay_s", 0.0),
                    )
                )
            states.append(
                ArrayState(
                    item["id"],
                    drives,
                    description=item.get("description", ""),
                )
            )
        return cls(
            states,
            embedded_far_fields=selections,
            description=definition.get("description", ""),
            source_path=path,
            source_text=text,
        )


@dataclass(frozen=True)
class EmbeddedFarFieldBank:
    """Per-unit-incident-power complex field basis ordered by modal channel.

    Requested-direction fields have shape ``(F, D, C)`` and full-sphere
    fields have shape ``(F, Q, C)``, where ``F`` is frequency, ``D`` is the
    number of requested directions, ``Q`` is the internal quadrature size,
    and ``C`` is the modal-channel count. Complex fields are range-normalized
    field per square-root watt of incident modal power.

    Attributes:
        transform_id: source frequency-transform identifier.
        output_id: source far-field request identifier.
        frequency: frequency axis in Hz.
        theta: requested polar angles in degrees.
        phi: requested azimuthal angles in degrees.
        etheta: embedded requested-direction :math:`E_\\theta` fields.
        ephi: embedded requested-direction :math:`E_\\phi` fields.
        sphere_theta: internal full-sphere polar angles in degrees.
        sphere_phi: internal full-sphere azimuthal angles in degrees.
        sphere_weights: full-sphere solid-angle quadrature weights.
        sphere_etheta: embedded full-sphere :math:`E_\\theta` fields.
        sphere_ephi: embedded full-sphere :math:`E_\\phi` fields.
        valid: physical modal-channel validity mask with shape ``(F, C)``.
        impedance: lossless exterior wave impedance in ohms.
        theta_order: full-sphere polar quadrature order.
        phi_order: full-sphere azimuthal quadrature order.
        enclosure_radius: radius in metres enclosing the transformed source.
    """

    transform_id: str
    output_id: str
    frequency: npt.NDArray[np.floating]
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    etheta: npt.NDArray[np.complexfloating]
    ephi: npt.NDArray[np.complexfloating]
    sphere_theta: npt.NDArray[np.floating]
    sphere_phi: npt.NDArray[np.floating]
    sphere_weights: npt.NDArray[np.floating]
    sphere_etheta: npt.NDArray[np.complexfloating]
    sphere_ephi: npt.NDArray[np.complexfloating]
    valid: npt.NDArray[np.bool_]
    impedance: float
    theta_order: int
    phi_order: int
    enclosure_radius: float

    @property
    def key(self) -> str:
        return f"{self.transform_id}/{self.output_id}"


@dataclass(frozen=True)
class ArrayFarFieldResult:
    """Coherently synthesized fields and radiation metrics for one array state.

    Direction-dependent arrays have shape ``(F, D)``; integrated, maximum,
    efficiency, and validity arrays have shape ``(F,)``.

    Attributes:
        transform_id: source frequency-transform identifier.
        output_id: source far-field request identifier.
        theta: requested polar angles in degrees.
        phi: requested azimuthal angles in degrees.
        etheta: coherently combined range-normalized :math:`E_\\theta`.
        ephi: coherently combined range-normalized :math:`E_\\phi`.
        radiation_intensity: radiation intensity in W/sr.
        radiated_power: full-sphere integrated radiated power in W.
        directivity: linear directivity.
        directivity_dbi: directivity in dBi.
        gain: linear accepted-power gain.
        gain_dbi: accepted-power gain in dBi.
        realized_gain: linear incident-power gain.
        realized_gain_dbi: incident-power gain in dBi.
        radiation_efficiency: radiated power divided by accepted power.
        total_efficiency: radiated power divided by incident power.
        maximum_directivity: maximum linear directivity.
        maximum_directivity_dbi: maximum directivity in dBi.
        maximum_theta: polar angle of maximum directivity in degrees.
        maximum_phi: azimuthal angle of maximum directivity in degrees.
        valid: frequency-bin validity mask.
    """

    transform_id: str
    output_id: str
    theta: npt.NDArray[np.floating]
    phi: npt.NDArray[np.floating]
    etheta: npt.NDArray[np.complexfloating]
    ephi: npt.NDArray[np.complexfloating]
    radiation_intensity: npt.NDArray[np.floating]
    radiated_power: npt.NDArray[np.floating]
    directivity: npt.NDArray[np.floating]
    directivity_dbi: npt.NDArray[np.floating]
    gain: npt.NDArray[np.floating]
    gain_dbi: npt.NDArray[np.floating]
    realized_gain: npt.NDArray[np.floating]
    realized_gain_dbi: npt.NDArray[np.floating]
    radiation_efficiency: npt.NDArray[np.floating]
    total_efficiency: npt.NDArray[np.floating]
    maximum_directivity: npt.NDArray[np.floating]
    maximum_directivity_dbi: npt.NDArray[np.floating]
    maximum_theta: npt.NDArray[np.floating]
    maximum_phi: npt.NDArray[np.floating]
    valid: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ArrayStateResult:
    """Network and optional radiation result for one named array state.

    Attributes:
        id: array-state identifier.
        frequency: frequency axis in Hz with shape ``(F,)``.
        incident: incident modal power waves with shape ``(F, C)``.
        outgoing: outgoing modal power waves with shape ``(F, C)``.
        active_reflection: active reflection coefficient for each driven
            modal channel, with shape ``(F, C)``.
        incident_power: total incident power in W.
        reflected_power: total reflected power in W.
        accepted_power: net accepted power in W.
        tarc: total active reflection coefficient.
        valid: frequency-bin network validity mask.
        far_fields: mapping from ``transform_id/output_id`` to synthesized
            :class:`ArrayFarFieldResult` objects.
    """

    id: str
    frequency: npt.NDArray[np.floating]
    incident: npt.NDArray[np.complexfloating]
    outgoing: npt.NDArray[np.complexfloating]
    active_reflection: npt.NDArray[np.complexfloating]
    incident_power: npt.NDArray[np.floating]
    reflected_power: npt.NDArray[np.floating]
    accepted_power: npt.NDArray[np.floating]
    tarc: npt.NDArray[np.floating]
    valid: npt.NDArray[np.bool_]
    far_fields: Mapping[str, ArrayFarFieldResult]


def modal_array_weights(
    frequency,
    channel_ports,
    channel_modes,
    excitations: Iterable[ModalWeight],
):
    """Build incident modal power waves using the engineering FFT convention.

    A constant phase shifter contributes ``exp(+j phase)``.  A true time
    delay contributes ``exp(-j 2 pi f delay)``.  The coefficient magnitude is
    the square root of the requested incident power in watts.
    """

    frequency = np.asarray(frequency)
    channel_ports = np.asarray(channel_ports, dtype=np.int64)
    channel_modes = np.asarray(channel_modes, dtype=np.int64)
    if frequency.ndim != 1:
        raise ValueError("Modal array frequencies must be one-dimensional.")
    if channel_ports.shape != channel_modes.shape or channel_ports.ndim != 1:
        raise ValueError("Modal channel port/mode arrays must be matching vectors.")
    channels = {(int(port), int(mode)): index for index, (port, mode) in enumerate(zip(channel_ports, channel_modes))}
    if len(channels) != channel_ports.size:
        raise ValueError("Modal channel port/mode pairs must be unique.")
    result = np.zeros((frequency.size, channel_ports.size), dtype=np.complex128)
    seen = set()
    for excitation in excitations:
        if not isinstance(excitation, ModalWeight):
            raise TypeError("Modal excitations must be ModalWeight instances.")
        channel = (int(excitation.port), int(excitation.mode))
        if channel not in channels:
            raise ValueError(f"Modal weight references unavailable port {channel[0]}, mode {channel[1]}.")
        if channel in seen:
            raise ValueError(f"Modal weight for port {channel[0]}, mode {channel[1]} is duplicated.")
        seen.add(channel)
        values = (excitation.power, excitation.phase_deg, excitation.delay_s)
        if not all(np.isfinite(value) for value in values) or excitation.power < 0:
            raise ValueError("Modal weight power must be finite and non-negative; phase and " "delay must be finite.")
        phase = np.deg2rad(excitation.phase_deg) - 2 * np.pi * frequency * excitation.delay_s
        result[:, channels[channel]] = np.sqrt(excitation.power) * np.exp(1j * phase)
    return result


def combine_embedded_modal_responses(responses, weights, *, channel_axis=-1):
    """Linearly combine complex embedded responses with modal weights.

    ``responses`` must have frequency as its first axis and one axis ordered
    like the study's modal channels.  All intervening axes (observation point,
    angle, field component, and so on) are retained.
    """

    responses = np.asarray(responses)
    weights = np.asarray(weights)
    if weights.ndim != 2:
        raise ValueError("Modal weights must have shape (frequency, channel).")
    if responses.ndim < 2:
        raise ValueError("Embedded responses require frequency and channel axes.")
    try:
        channel_axis = operator.index(channel_axis)
    except TypeError as exc:
        raise TypeError("The embedded-response channel axis must be an integer.") from exc
    if not -responses.ndim <= channel_axis < responses.ndim:
        raise ValueError("The embedded-response channel axis is outside the response array.")
    channel_axis %= responses.ndim
    if channel_axis == 0:
        raise ValueError("The embedded-response channel axis cannot be its frequency axis.")
    moved = np.moveaxis(responses, channel_axis, -1)
    if moved.shape[0] != weights.shape[0]:
        raise ValueError("Embedded responses and weights require the same frequency axis.")
    if moved.shape[-1] != weights.shape[1]:
        raise ValueError("Embedded response channels do not match modal weights.")
    shape = (weights.shape[0],) + (1,) * (moved.ndim - 2) + (weights.shape[1],)
    reshaped_weights = weights.reshape(shape)
    selected = np.where(np.abs(reshaped_weights) > 0, moved, 0)
    return np.sum(selected * reshaped_weights, axis=-1)


def _ratio_and_db(
    intensity: npt.NDArray[np.floating],
    power: npt.NDArray[np.floating],
):
    result = np.full(intensity.shape, np.nan, dtype=intensity.dtype)
    valid = np.isfinite(power) & (power > 0)
    result[valid] = 4 * np.pi * intensity[valid] / power[valid, np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        result_db = 10 * np.log10(result)
    return result, np.asarray(result_db, dtype=intensity.dtype)


def evaluate_array_state(
    study: EigenmodeStudyResult,
    state: ArrayState,
) -> ArrayStateResult:
    """Apply a named modal state to an S matrix and embedded field banks."""

    if not isinstance(study, EigenmodeStudyResult):
        raise TypeError("study must be an EigenmodeStudyResult")
    if not isinstance(state, ArrayState):
        raise TypeError("state must be an ArrayState")
    incident = study.excitation_weights(state.drives)
    active = np.abs(incident) > 0
    propagating = np.diagonal(study.valid_s, axis1=1, axis2=2)
    active_inputs_valid = np.all(~active | propagating, axis=1)
    required_power_waves = propagating[:, :, np.newaxis] & active[:, np.newaxis, :]
    physical_transfer_valid = np.all(~required_power_waves | study.valid_s, axis=(1, 2))

    selected_s = np.where(
        active[:, np.newaxis, :] & study.generalized_valid_s,
        study.s,
        0,
    )
    outgoing = np.einsum("foi,fi->fo", selected_s, incident)
    has_incident = np.any(active, axis=1)
    outgoing_valid = active_inputs_valid[:, np.newaxis] & np.all(
        ~active[:, np.newaxis, :] | study.generalized_valid_s,
        axis=2,
    )
    outgoing[~outgoing_valid] = np.nan + 1j * np.nan
    network_valid = active_inputs_valid & physical_transfer_valid & has_incident

    real_dtype = np.empty((), dtype=study.s.dtype).real.dtype
    incident_power = np.asarray(np.sum(np.abs(incident) ** 2, axis=1), dtype=real_dtype)
    reflected_power = np.asarray(
        np.sum(np.where(propagating, np.abs(outgoing) ** 2, 0), axis=1),
        dtype=real_dtype,
    )
    accepted_power = np.asarray(incident_power - reflected_power, dtype=real_dtype)
    for values in (reflected_power, accepted_power):
        values[~network_valid] = np.nan
    tarc = np.full(incident_power.shape, np.nan, dtype=real_dtype)
    tarc[network_valid] = np.sqrt(np.maximum(reflected_power[network_valid], 0) / incident_power[network_valid])
    complex_dtype = np.result_type(study.s.dtype, incident.dtype)
    active_reflection = np.full(incident.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
    np.divide(
        outgoing,
        incident,
        out=active_reflection,
        where=active & propagating & network_valid[:, None],
    )

    far_fields: dict[str, ArrayFarFieldResult] = {}
    for key, bank in study.embedded_far_fields.items():
        if not np.array_equal(bank.frequency, study.frequency):
            raise ValueError(f"Embedded far-field basis {key!r} has an incompatible frequency axis")
        if bank.etheta.shape[-1] != incident.shape[1]:
            raise ValueError(f"Embedded far-field basis {key!r} has an incompatible channel axis")
        field_valid = network_valid & np.all(~active | bank.valid, axis=1)
        etheta = combine_embedded_modal_responses(bank.etheta, incident)
        ephi = combine_embedded_modal_responses(bank.ephi, incident)
        sphere_etheta = combine_embedded_modal_responses(bank.sphere_etheta, incident)
        sphere_ephi = combine_embedded_modal_responses(bank.sphere_ephi, incident)
        for values in (etheta, ephi, sphere_etheta, sphere_ephi):
            values[~field_valid] = np.nan + 1j * np.nan

        intensity = np.asarray(
            0.5 * (np.abs(etheta) ** 2 + np.abs(ephi) ** 2) / bank.impedance,
            dtype=real_dtype,
        )
        sphere_intensity = np.asarray(
            0.5 * (np.abs(sphere_etheta) ** 2 + np.abs(sphere_ephi) ** 2) / bank.impedance,
            dtype=real_dtype,
        )
        radiated_power = np.asarray(
            np.sum(sphere_intensity * bank.sphere_weights[np.newaxis, :], axis=1),
            dtype=real_dtype,
        )
        radiated_power[~field_valid] = np.nan
        directivity, directivity_dbi = _ratio_and_db(intensity, radiated_power)
        gain, gain_dbi = _ratio_and_db(intensity, accepted_power)
        realized_gain, realized_gain_dbi = _ratio_and_db(intensity, incident_power)
        radiation_efficiency = np.full(radiated_power.shape, np.nan, dtype=real_dtype)
        total_efficiency = np.full(radiated_power.shape, np.nan, dtype=real_dtype)
        accepted_valid = field_valid & np.isfinite(accepted_power) & (accepted_power > 0)
        incident_valid = field_valid & (incident_power > 0)
        radiation_efficiency[accepted_valid] = radiated_power[accepted_valid] / accepted_power[accepted_valid]
        total_efficiency[incident_valid] = radiated_power[incident_valid] / incident_power[incident_valid]

        maximum_index = np.argmax(np.where(np.isfinite(sphere_intensity), sphere_intensity, -np.inf), axis=1)
        maximum_intensity = sphere_intensity[np.arange(study.frequency.size), maximum_index]
        maximum_directivity, maximum_directivity_dbi = _ratio_and_db(maximum_intensity[:, np.newaxis], radiated_power)
        maximum_theta = np.asarray(bank.sphere_theta[maximum_index], dtype=real_dtype)
        maximum_phi = np.asarray(bank.sphere_phi[maximum_index], dtype=real_dtype)
        maximum_theta[~field_valid] = np.nan
        maximum_phi[~field_valid] = np.nan
        far_fields[key] = ArrayFarFieldResult(
            transform_id=bank.transform_id,
            output_id=bank.output_id,
            theta=bank.theta,
            phi=bank.phi,
            etheta=etheta,
            ephi=ephi,
            radiation_intensity=intensity,
            radiated_power=radiated_power,
            directivity=directivity,
            directivity_dbi=directivity_dbi,
            gain=gain,
            gain_dbi=gain_dbi,
            realized_gain=realized_gain,
            realized_gain_dbi=realized_gain_dbi,
            radiation_efficiency=radiation_efficiency,
            total_efficiency=total_efficiency,
            maximum_directivity=maximum_directivity[:, 0],
            maximum_directivity_dbi=maximum_directivity_dbi[:, 0],
            maximum_theta=maximum_theta,
            maximum_phi=maximum_phi,
            valid=field_valid,
        )

    return ArrayStateResult(
        id=state.id,
        frequency=study.frequency,
        incident=incident,
        outgoing=outgoing,
        active_reflection=active_reflection,
        incident_power=incident_power,
        reflected_power=reflected_power,
        accepted_power=accepted_power,
        tarc=tarc,
        valid=network_valid,
        far_fields=far_fields,
    )


class EigenmodeStudy(Study):
    """One-active-modal-channel-per-case reusable eigenmode-port study.

    All transverse modal anchor problems are solved during the first geometry
    build.  Later cases only synthesize an injection from those cached bases,
    clear modal/virtual-guide state, and select a new active ``(port, mode)``.
    """

    supported_types = ("eigenmode",)

    def __init__(
        self,
        cases: Sequence[StudyCase],
        *,
        codebook: Optional[ArrayCodebook] = None,
        source_path: Optional[Union[str, Path]] = None,
        source_text: Optional[str] = None,
    ):
        super().__init__(
            "eigenmode",
            cases,
            source_path=source_path,
            source_text=source_text,
        )
        if codebook is not None and not isinstance(codebook, ArrayCodebook):
            raise TypeError("EigenmodeStudy codebook must be an ArrayCodebook instance")
        self.codebook = codebook
        self._excitation = None
        self._case_channels: list[tuple[int, int]] = []
        self._channels: tuple[tuple[int, int], ...] = ()
        self._runtime_grid = None
        self._runtime_ports: dict[int, Any] = {}
        self._runtime_monitors: dict[int, Any] = {}
        self._runtime_guides: dict[int, Any] = {}
        self._waveform = None
        self._columns: dict[tuple[int, int], dict[str, Any]] = {}
        self._current_column: Optional[dict[str, Any]] = None
        self._embedded_bindings: dict[str, tuple[Any, str]] = {}
        self._raw_embedded_far_fields: dict[str, dict[str, Any]] = {}
        self._aggregate_wave_valid_matrix = None
        self._aggregate_generalized_wave_valid_matrix = None
        self._aggregate_output_path: Optional[Path] = None
        self.result: Optional[EigenmodeStudyResult] = None

    def reset_runtime(self) -> None:
        super().reset_runtime()
        self._runtime_grid = None
        self._runtime_ports.clear()
        self._runtime_monitors.clear()
        self._runtime_guides.clear()
        self._waveform = None
        self._columns.clear()
        self._current_column = None
        self._embedded_bindings.clear()
        self._raw_embedded_far_fields.clear()
        self._aggregate_wave_valid_matrix = None
        self._aggregate_generalized_wave_valid_matrix = None
        self._aggregate_output_path = None
        self.result = None

    def bind_scene(self, scene) -> None:
        """Validate a complete, unambiguous modal-channel schedule."""

        if self._scene_bound:
            return
        from gprMax.user_objects.cmds_multiuse import EigenmodeExcitation, EigenmodePort

        containers = [("main grid", list(scene.grid_objects) + list(scene.output_objects))]
        containers.extend(
            (
                f"subgrid {subgrid.kwargs.get('id', '')!r}",
                list(subgrid.children_grid) + list(subgrid.children_output),
            )
            for subgrid in scene.subgrid_objects
        )
        excitations = [
            (label, objects, item)
            for label, objects in containers
            for item in objects
            if isinstance(item, EigenmodeExcitation)
        ]
        if len(excitations) != 1:
            raise ValueError(
                "EigenmodeStudy requires exactly one EigenmodeExcitation in its Scene; " f"found {len(excitations)}."
            )
        owner_label, owner_objects, excitation = excitations[0]
        ports = [item for item in owner_objects if isinstance(item, EigenmodePort)]
        if not ports:
            raise ValueError("EigenmodeStudy requires at least one EigenmodePort.")
        foreign_ports = [
            label
            for label, objects in containers
            if objects is not owner_objects
            for item in objects
            if isinstance(item, EigenmodePort)
        ]
        if foreign_ports:
            raise ValueError(
                "EigenmodeStudy requires its excitation and every eigenmode port to "
                f"belong to one grid ({owner_label}); ports were also found on "
                + ", ".join(sorted(set(foreign_ports)))
                + "."
            )

        study_id = "eigenmode_excitation_1"
        setattr(excitation, "_study_id", study_id)
        # The builder uses this marker to resolve source-grade anchor coverage
        # at every port, not only at the initially selected source.
        setattr(excitation, "_reusable_study", True)
        self._excitation = excitation
        self.registry = {study_id: excitation}
        self.aliases = {}

        available: set[tuple[int, int]] = set()
        for port in ports:
            port_number = int(port.kwargs["port"])
            modes = port.kwargs.get("modes", (1,))
            if np.isscalar(modes):
                modes = tuple(range(1, int(modes) + 1))
            available.update((port_number, int(mode)) for mode in modes)

        scheduled: list[tuple[int, int]] = []
        for case in self.cases:
            if len(case.states) != 1:
                raise ValueError(
                    f"EigenmodeStudy case '{case.id}' must contain exactly one " "EigenmodeExcitation state."
                )
            state = case.states[0]
            if isinstance(state.object, str):
                requested = state.object.strip()
                if requested != study_id:
                    raise ValueError(
                        f"EigenmodeStudy case '{case.id}' references '{requested}', " f"expected '{study_id}'."
                    )
            elif state.object is not excitation:
                raise ValueError(
                    f"EigenmodeStudy case '{case.id}' must reference the Scene's " "EigenmodeExcitation object."
                )
            unknown = set(state.parameters) - {"port", "mode"}
            if unknown:
                raise ValueError(
                    f"EigenmodeStudy case '{case.id}' has unsupported parameter(s) "
                    f"{', '.join(sorted(unknown))}; allowed parameters are port and mode."
                )
            if "port" not in state.parameters or "mode" not in state.parameters:
                raise ValueError(f"EigenmodeStudy case '{case.id}' requires both port and mode.")
            channel = (int(state.parameters["port"]), int(state.parameters["mode"]))
            if channel not in available:
                raise ValueError(
                    f"EigenmodeStudy case '{case.id}' requests unavailable channel "
                    f"port {channel[0]}, mode {channel[1]}."
                )
            state.object = study_id
            scheduled.append(channel)

        if len(scheduled) != len(set(scheduled)):
            raise ValueError("EigenmodeStudy may excite each port/mode channel only once.")
        if set(scheduled) != available:
            missing = ", ".join(f"port {port}, mode {mode}" for port, mode in sorted(available - set(scheduled)))
            raise ValueError(
                "EigenmodeStudy requires one case for every declared modal channel; " f"missing {missing}."
            )
        self._case_channels = scheduled
        self._channels = tuple(sorted(available))
        if self.codebook is not None:
            channel_ports = np.asarray([item[0] for item in self._channels])
            channel_modes = np.asarray([item[1] for item in self._channels])
            for state in self.codebook.states:
                # Validate every reference and numerical drive value before
                # starting an otherwise potentially expensive FDTD study.
                modal_array_weights((0.0,), channel_ports, channel_modes, state.drives)
        self._scene_bound = True
        logger.info(
            "EigenmodeStudy channels: " + ", ".join(f"port {port}/mode {mode}" for port, mode in self._channels) + "\n"
        )

    def _bind_runtime(self, model) -> None:
        if self._runtime_grid is not None:
            return
        grids = [model.G] + list(model.subgrids)
        matches = [grid for grid in grids if grid.eigenmodeexcitation is self._excitation]
        if len(matches) != 1:
            raise RuntimeError("EigenmodeStudy could not identify exactly one built grid for its excitation.")
        grid = matches[0]
        monitors = {int(monitor.port_index): monitor for monitor in grid.eigenmodeports}
        ports = {port_number: monitor.owner for port_number, monitor in monitors.items()}
        expected_ports = {port for port, _ in self._channels}
        if set(ports) != expected_ports:
            raise RuntimeError("EigenmodeStudy runtime ports do not match its declared modal channels.")
        guides = {int(guide.spec.port): guide for guide in grid.virtual_waveguides}
        waveforms = [port.waveform for port in ports.values() if getattr(port, "waveform", None) is not None]
        if len(waveforms) != 1:
            raise RuntimeError("EigenmodeStudy requires exactly one baseline excitation waveform.")
        self._runtime_grid = grid
        self._runtime_ports = ports
        self._runtime_monitors = monitors
        self._runtime_guides = guides
        self._waveform = waveforms[0]

    def _bind_embedded_ntff(self, grid) -> None:
        self._embedded_bindings.clear()
        if self.codebook is not None:
            for selection in self.codebook.embedded_far_fields:
                matches = []
                for writer in grid.ntff_output_writers:
                    for request_key, request in writer.far_requests.items():
                        if request.transform_id == selection.transform_id and request.output_id == selection.output_id:
                            matches.append((writer, request_key))
                if len(matches) != 1:
                    raise ValueError(
                        f"Array codebook embedded far field {selection.key!r} matched "
                        f"{len(matches)} NTFF requests; exactly one is required."
                    )
                self._embedded_bindings[selection.key] = matches[0]

    def apply_case(self, model) -> None:
        """Reset persistent state and activate the scheduled modal channel."""

        import gprMax.config as config

        self._bind_runtime(model)
        grid = self._runtime_grid
        case_index = config.sim_config.current_model
        port_number, mode_index = self._case_channels[case_index]

        for monitor in self._runtime_monitors.values():
            monitor.reset_run_state(grid)
            monitor.is_source = False
            monitor.excitation_mode_index = None
            monitor.excitation_mode_indices = ()
            monitor.drive_metadata = ()
            monitor.magnetic_side = 1 if int(monitor.port_index) in self._runtime_guides else -1
        for guide in self._runtime_guides.values():
            guide.clear_active_source()
            guide.reset_run_state()

        all_ports = list(self._runtime_ports.values())
        grid.eigenmodesources.clear()
        grid.eigenmodereceivers[:] = all_ports

        source = self._runtime_ports[port_number]
        source.spectral_threshold = grid.eigenmodeband.spectral_threshold
        monitor = self._runtime_monitors[port_number]
        drive = self._excitation
        # The scheduled channel changes, while the reusable excitation keeps
        # its waveform, amplitude, phase, and delay controls.
        drive.port_index = port_number
        drive.mode_index = mode_index
        source.set_drive_parameters(drive)
        source.configure_cached_excitation(grid, mode_index, self._waveform)
        monitor.set_drive_metadata((drive,))
        monitor.magnetic_side = 1
        guide = self._runtime_guides.get(port_number)
        if guide is None:
            grid.eigenmodereceivers.remove(source)
            grid.eigenmodesources.append(source)
        else:
            guide.set_active_source(source)

        # NTFF transforms always live on the main grid. Their closed surface
        # may contain a complete fine subgrid and an eigenmode source on that
        # subgrid, but the reusable transform itself must still be rebuilt on
        # the main grid for every study case.
        _recompile_declarative_ntff(model, model.G, "EigenmodeStudy")
        self._bind_embedded_ntff(model.G)

        case = self.cases[case_index]
        self._current_resolved_case = {
            "case_id": case.id,
            "objects": {
                "eigenmode_excitation_1": {
                    "port": port_number,
                    "mode": mode_index,
                }
            },
        }
        if self._aggregate_output_path is None:
            model_config = config.get_model_config()
            base = Path(model_config.output_file_path)
            suffix = model_config.appendmodelnumber
            if suffix and base.name.endswith(suffix):
                base = base.with_name(base.name[: -len(suffix)])
            self._aggregate_output_path = base.with_name(base.name + "_study").with_suffix(".h5")

    def collect_case(self, model) -> None:
        """Collect one excitation run's complete incident/outgoing modal waves."""

        import gprMax.config as config

        grid = self._runtime_grid
        input_channel = self._case_channels[config.sim_config.current_model]
        first = self._runtime_monitors[self._channels[0][0]]
        frequency = np.asarray(first.result.frequency)
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        column = np.full(
            (frequency.size, len(self._channels)),
            np.nan + 1j * np.nan,
            dtype=complex_dtype,
        )
        valid = np.zeros(column.shape, dtype=bool)
        generalized_valid = np.zeros(column.shape, dtype=bool)
        incident = np.full_like(column, np.nan + 1j * np.nan)
        outgoing = np.full_like(column, np.nan + 1j * np.nan)
        wave_valid = np.zeros(column.shape, dtype=bool)
        generalized_wave_valid = np.zeros(column.shape, dtype=bool)
        for output_index, (port_number, mode_index) in enumerate(self._channels):
            monitor = self._runtime_monitors[port_number]
            if not np.array_equal(monitor.result.frequency, frequency):
                raise ValueError("All EigenmodeStudy ports must use identical DFT bins.")
            mode_position = monitor.mode_indices.index(mode_index)
            column[:, output_index] = monitor.s_parameters[mode_position]
            valid[:, output_index] = monitor.s_valid[mode_position]
            generalized_valid[:, output_index] = monitor.s_generalized_valid[mode_position]
            incident[:, output_index] = monitor.result.incident[mode_position]
            outgoing[:, output_index] = monitor.result.outgoing[mode_position]
            wave_valid[:, output_index] = monitor.result.valid[mode_position]
            monitor_generalized_valid = getattr(
                monitor.result,
                "generalized_valid",
                monitor.result.valid,
            )
            generalized_wave_valid[:, output_index] = monitor_generalized_valid[mode_position]

        embedded: dict[str, dict[str, Any]] = {}
        for selection_key, (writer, request_key) in self._embedded_bindings.items():
            basis = writer.linear_far_field_basis(request_key)
            if not np.array_equal(basis.frequencies, frequency):
                raise ValueError(
                    f"Embedded far field {selection_key!r} and EigenmodeStudy ports " "must use identical DFT bins."
                )

            raw_fields = (
                np.asarray(basis.etheta),
                np.asarray(basis.ephi),
                np.asarray(basis.sphere_etheta),
                np.asarray(basis.sphere_ephi),
            )
            field_valid = np.ones(frequency.shape, dtype=bool)
            for values in raw_fields:
                field_valid &= np.all(np.isfinite(values), axis=1)

            embedded[selection_key] = {
                "transform_id": basis.transform_id,
                "output_id": basis.output_id,
                "theta": np.asarray(basis.theta),
                "phi": np.asarray(basis.phi),
                "etheta": raw_fields[0].astype(complex_dtype, copy=False),
                "ephi": raw_fields[1].astype(complex_dtype, copy=False),
                "sphere_theta": np.asarray(basis.sphere_theta),
                "sphere_phi": np.asarray(basis.sphere_phi),
                "sphere_weights": np.asarray(basis.sphere_weights),
                "sphere_etheta": raw_fields[2].astype(complex_dtype, copy=False),
                "sphere_ephi": raw_fields[3].astype(complex_dtype, copy=False),
                "valid": field_valid,
                "impedance": basis.impedance,
                "theta_order": basis.theta_order,
                "phi_order": basis.phi_order,
                "enclosure_radius": basis.enclosure_radius,
            }

        data = {
            "case_id": self.cases[config.sim_config.current_model].id,
            "input_channel": input_channel,
            "frequency": frequency.copy(),
            "column": column,
            "valid": valid,
            "generalized_valid": generalized_valid,
            "incident": incident,
            "outgoing": outgoing,
            "wave_valid": wave_valid,
            "generalized_wave_valid": generalized_wave_valid,
            "embedded": embedded,
        }
        self._columns[input_channel] = data
        self._current_column = data

    def write_hdf5(self, h5file) -> None:
        super().write_hdf5(h5file)
        if self._current_column is None:
            raise RuntimeError("EigenmodeStudy case data was not collected before HDF5 output.")
        data = self._current_column
        group = h5file["study"].create_group("eigenmode_response")
        group.attrs["InputPort"] = data["input_channel"][0]
        group.attrs["InputMode"] = data["input_channel"][1]
        group.attrs["MatrixConvention"] = "S[frequency, output_channel, input_channel]"
        group.create_dataset("channel_ports", data=[item[0] for item in self._channels])
        group.create_dataset("channel_modes", data=[item[1] for item in self._channels])
        group.create_dataset("frequency", data=data["frequency"])
        group.create_dataset("S_column", data=data["column"])
        group.create_dataset("valid_S_column", data=data["valid"].astype(np.uint8))
        group.create_dataset("power_wave_valid_S_column", data=data["valid"].astype(np.uint8))
        group.create_dataset(
            "generalized_valid_S_column",
            data=data["generalized_valid"].astype(np.uint8),
        )
        group.create_dataset(
            "coefficient_valid_S_column",
            data=data["generalized_valid"].astype(np.uint8),
        )
        group.create_dataset("incident", data=data["incident"])
        group.create_dataset("outgoing", data=data["outgoing"])
        group.create_dataset("valid_wave", data=data["wave_valid"].astype(np.uint8))
        group.create_dataset("power_wave_valid", data=data["wave_valid"].astype(np.uint8))
        group.create_dataset(
            "generalized_valid_wave",
            data=data["generalized_wave_valid"].astype(np.uint8),
        )
        group.create_dataset(
            "coefficient_valid_wave",
            data=data["generalized_wave_valid"].astype(np.uint8),
        )
        if self._aggregate_output_path is not None:
            group.attrs["AggregateOutput"] = str(self._aggregate_output_path)

    def finalise(self) -> Optional[EigenmodeStudyResult]:
        """Assemble and store the de-embedded complete modal S matrix."""

        if not self._columns or self._aggregate_output_path is None:
            return None
        first = next(iter(self._columns.values()))
        frequency = first["frequency"]
        channel_count = len(self._channels)
        shape = (frequency.size, channel_count, channel_count)
        incident_matrix = np.full(
            shape,
            np.nan + 1j * np.nan,
            dtype=first["column"].dtype,
        )
        outgoing_matrix = np.full_like(incident_matrix, np.nan + 1j * np.nan)
        wave_valid_matrix = np.zeros(shape, dtype=bool)
        generalized_wave_valid_matrix = np.zeros(shape, dtype=bool)
        case_ids = [""] * channel_count

        if len(self._columns) < channel_count and self._aggregate_output_path.exists():
            with h5py.File(self._aggregate_output_path, "r") as previous:
                previous_channels = tuple(
                    zip(
                        previous["channel_ports"][...].astype(int),
                        previous["channel_modes"][...].astype(int),
                    )
                )
                compatible = (
                    previous.attrs.get("StudyType", "") == "eigenmode"
                    and previous_channels == self._channels
                    and np.array_equal(previous["frequency"][...], frequency)
                )
                if not compatible:
                    raise ValueError(
                        f"Existing EigenmodeStudy output '{self._aggregate_output_path}' "
                        "is not compatible with this restarted study."
                    )
                required = (
                    "incident_matrix",
                    "outgoing_matrix",
                    "valid_wave_matrix",
                    "generalized_valid_wave_matrix",
                )
                missing = [name for name in required if name not in previous]
                if missing:
                    raise ValueError(
                        f"Existing EigenmodeStudy output '{self._aggregate_output_path}' "
                        "predates full incident-matrix de-embedding and cannot be "
                        f"restarted (missing {', '.join(missing)})."
                    )
                incident_matrix[...] = previous["incident_matrix"][...]
                outgoing_matrix[...] = previous["outgoing_matrix"][...]
                wave_valid_matrix[...] = previous["valid_wave_matrix"][...].astype(bool)
                generalized_wave_valid_matrix[...] = previous["generalized_valid_wave_matrix"][...].astype(bool)
                case_ids = [item.decode() for item in previous["case_ids"][...]]
        elif len(self._columns) < channel_count:
            logger.warning(
                "EigenmodeStudy is incomplete and no compatible aggregate output "
                "exists from an earlier run; the incident basis is incomplete and "
                "the aggregate S matrix will be NaN."
            )

        for input_index, channel in enumerate(self._channels):
            data = self._columns.get(channel)
            if data is None:
                continue
            if not np.array_equal(data["frequency"], frequency):
                raise RuntimeError("EigenmodeStudy collected inconsistent frequency axes.")
            incident_matrix[:, :, input_index] = data["incident"]
            outgoing_matrix[:, :, input_index] = data["outgoing"]
            wave_valid_matrix[:, :, input_index] = data["wave_valid"]
            generalized_wave_valid_matrix[:, :, input_index] = data["generalized_wave_valid"]
            case_ids[input_index] = data["case_id"]

        s, generalized_output_valid, condition_number, deembedding_valid = _deembed_modal_responses(
            incident_matrix,
            outgoing_matrix,
            incident_valid=generalized_wave_valid_matrix,
            response_valid=generalized_wave_valid_matrix,
        )
        generalized_valid = np.broadcast_to(
            generalized_output_valid[:, :, np.newaxis],
            shape,
        ).copy()
        physical_incident_basis_valid = np.all(wave_valid_matrix, axis=(1, 2))
        physical_output_valid = np.all(wave_valid_matrix, axis=2)
        valid = (
            generalized_valid
            & physical_incident_basis_valid[:, np.newaxis, np.newaxis]
            & physical_output_valid[:, :, np.newaxis]
            & deembedding_valid[:, np.newaxis, np.newaxis]
        )

        embedded_far_fields = self._assemble_embedded_far_fields(
            frequency,
            channel_count,
            incident_matrix=incident_matrix,
            wave_valid_matrix=wave_valid_matrix,
            load_previous=len(self._columns) < channel_count,
        )
        self._aggregate_wave_valid_matrix = wave_valid_matrix
        self._aggregate_generalized_wave_valid_matrix = generalized_wave_valid_matrix

        self.result = EigenmodeStudyResult(
            frequency=np.asarray(frequency),
            channel_ports=np.asarray([item[0] for item in self._channels], dtype=np.int32),
            channel_modes=np.asarray([item[1] for item in self._channels], dtype=np.int32),
            case_ids=tuple(case_ids),
            s=s,
            valid_s=valid,
            generalized_valid_s=generalized_valid,
            output_file=self._aggregate_output_path,
            embedded_far_fields=embedded_far_fields,
            incident_matrix=incident_matrix,
            outgoing_matrix=outgoing_matrix,
            wave_valid_matrix=wave_valid_matrix,
            generalized_wave_valid_matrix=generalized_wave_valid_matrix,
            deembedding_condition_number=condition_number,
            deembedding_valid=deembedding_valid,
        )
        self._write_aggregate_hdf5()
        logger.info(f"Written EigenmodeStudy output file: {self._aggregate_output_path.name}\n")
        return self.result

    def _assemble_embedded_far_fields(
        self,
        frequency,
        channel_count: int,
        *,
        incident_matrix,
        wave_valid_matrix,
        load_previous: bool,
    ) -> dict[str, EmbeddedFarFieldBank]:
        if self.codebook is None or not self.codebook.embedded_far_fields:
            self._raw_embedded_far_fields.clear()
            return {}
        previous_file = None
        if load_previous and self._aggregate_output_path is not None:
            previous_file = h5py.File(self._aggregate_output_path, "r")
        try:
            assembled = {}
            raw_assembled = {}
            for selection in self.codebook.embedded_far_fields:
                template = next(
                    (
                        column["embedded"][selection.key]
                        for column in self._columns.values()
                        if selection.key in column["embedded"]
                    ),
                    None,
                )
                previous_group = None
                previous_raw = None
                if previous_file is not None:
                    path = f"embedded_far_fields/{selection.transform_id}/{selection.output_id}"
                    previous_group = previous_file.get(path)
                    if previous_group is not None:
                        previous_raw = previous_group.get("raw_runs")
                        if previous_raw is None:
                            raise ValueError(
                                f"Existing embedded far field {selection.key!r} predates "
                                "full incident-matrix de-embedding and cannot be restarted."
                            )
                if template is None and previous_group is None:
                    raise RuntimeError(f"No embedded far-field data were collected for {selection.key!r}.")

                if template is not None:
                    requested_shape = template["etheta"].shape
                    sphere_shape = template["sphere_etheta"].shape
                    metadata = template
                else:
                    requested_shape = previous_raw["Etheta"].shape[:2]
                    sphere_shape = previous_raw["sphere/Etheta"].shape[:2]
                    metadata = {
                        "theta": previous_group["theta"][...],
                        "phi": previous_group["phi"][...],
                        "sphere_theta": previous_group["sphere/theta"][...],
                        "sphere_phi": previous_group["sphere/phi"][...],
                        "sphere_weights": previous_group["sphere/weights"][...],
                        "impedance": previous_group.attrs["Impedance"],
                        "theta_order": previous_group.attrs["ThetaOrder"],
                        "phi_order": previous_group.attrs["PhiOrder"],
                        "enclosure_radius": previous_group.attrs["EnclosureRadius"],
                    }
                complex_dtype = next(iter(self._columns.values()))["column"].dtype
                raw_etheta = np.full(
                    requested_shape + (channel_count,),
                    np.nan + 1j * np.nan,
                    dtype=complex_dtype,
                )
                raw_ephi = np.full_like(raw_etheta, np.nan + 1j * np.nan)
                raw_sphere_etheta = np.full(
                    sphere_shape + (channel_count,),
                    np.nan + 1j * np.nan,
                    dtype=complex_dtype,
                )
                raw_sphere_ephi = np.full_like(
                    raw_sphere_etheta,
                    np.nan + 1j * np.nan,
                )
                raw_valid = np.zeros((len(frequency), channel_count), dtype=bool)
                if previous_group is not None:
                    if not np.array_equal(previous_group["frequency"][...], frequency):
                        raise ValueError(
                            f"Existing embedded far field {selection.key!r} has an " "incompatible frequency axis."
                        )
                    raw_etheta[...] = previous_raw["Etheta"][...]
                    raw_ephi[...] = previous_raw["Ephi"][...]
                    raw_sphere_etheta[...] = previous_raw["sphere/Etheta"][...]
                    raw_sphere_ephi[...] = previous_raw["sphere/Ephi"][...]
                    raw_valid[...] = previous_raw["valid"][...].astype(bool)

                for channel_index, channel in enumerate(self._channels):
                    data = self._columns.get(channel)
                    if data is None:
                        continue
                    item = data["embedded"].get(selection.key)
                    if item is None:
                        raise RuntimeError(f"Case {data['case_id']!r} did not collect {selection.key!r}.")
                    for name in ("theta", "phi", "sphere_theta", "sphere_phi", "sphere_weights"):
                        if not np.array_equal(item[name], metadata[name]):
                            raise RuntimeError(f"Embedded far field {selection.key!r} changed {name} between cases.")
                    raw_etheta[..., channel_index] = item["etheta"]
                    raw_ephi[..., channel_index] = item["ephi"]
                    raw_sphere_etheta[..., channel_index] = item["sphere_etheta"]
                    raw_sphere_ephi[..., channel_index] = item["sphere_ephi"]
                    raw_valid[:, channel_index] = item["valid"]

                field_response_valid = raw_valid[:, np.newaxis, :]
                etheta, etheta_valid, _, etheta_basis_valid = _deembed_modal_responses(
                    incident_matrix,
                    raw_etheta,
                    incident_valid=wave_valid_matrix,
                    response_valid=field_response_valid,
                )
                ephi, ephi_valid, _, ephi_basis_valid = _deembed_modal_responses(
                    incident_matrix,
                    raw_ephi,
                    incident_valid=wave_valid_matrix,
                    response_valid=field_response_valid,
                )
                sphere_etheta, sphere_etheta_valid, _, sphere_etheta_basis_valid = _deembed_modal_responses(
                    incident_matrix,
                    raw_sphere_etheta,
                    incident_valid=wave_valid_matrix,
                    response_valid=field_response_valid,
                )
                sphere_ephi, sphere_ephi_valid, _, sphere_ephi_basis_valid = _deembed_modal_responses(
                    incident_matrix,
                    raw_sphere_ephi,
                    incident_valid=wave_valid_matrix,
                    response_valid=field_response_valid,
                )
                frequency_valid = (
                    etheta_basis_valid
                    & ephi_basis_valid
                    & sphere_etheta_basis_valid
                    & sphere_ephi_basis_valid
                    & np.all(etheta_valid, axis=1)
                    & np.all(ephi_valid, axis=1)
                    & np.all(sphere_etheta_valid, axis=1)
                    & np.all(sphere_ephi_valid, axis=1)
                )
                valid = np.broadcast_to(
                    frequency_valid[:, np.newaxis],
                    (len(frequency), channel_count),
                ).copy()

                raw_assembled[selection.key] = {
                    "etheta": raw_etheta,
                    "ephi": raw_ephi,
                    "sphere_etheta": raw_sphere_etheta,
                    "sphere_ephi": raw_sphere_ephi,
                    "valid": raw_valid,
                }

                assembled[selection.key] = EmbeddedFarFieldBank(
                    transform_id=selection.transform_id,
                    output_id=selection.output_id,
                    frequency=np.asarray(frequency),
                    theta=np.asarray(metadata["theta"]),
                    phi=np.asarray(metadata["phi"]),
                    etheta=etheta,
                    ephi=ephi,
                    sphere_theta=np.asarray(metadata["sphere_theta"]),
                    sphere_phi=np.asarray(metadata["sphere_phi"]),
                    sphere_weights=np.asarray(metadata["sphere_weights"]),
                    sphere_etheta=sphere_etheta,
                    sphere_ephi=sphere_ephi,
                    valid=valid,
                    impedance=float(metadata["impedance"]),
                    theta_order=int(metadata["theta_order"]),
                    phi_order=int(metadata["phi_order"]),
                    enclosure_radius=float(metadata["enclosure_radius"]),
                )
            self._raw_embedded_far_fields = raw_assembled
            return assembled
        finally:
            if previous_file is not None:
                previous_file.close()

    def _write_aggregate_hdf5(self) -> None:
        if self.result is None:
            return
        from gprMax._version import __version__
        from gprMax.ntff.conventions import FORWARD_TRANSFORM_KERNEL, PHASOR_TIME_DEPENDENCE

        result = self.result
        with h5py.File(result.output_file, "w") as output:
            output.attrs["gprMax"] = __version__
            output.attrs["StudyType"] = self.type
            output.attrs["MatrixConvention"] = "S[frequency, output_channel, input_channel]"
            output.attrs["RunMatrixConvention"] = (
                "A_or_B[frequency, modal_channel, excitation_case]"
            )
            output.attrs["WaveNormalisation"] = "modal_power_wave"
            output.attrs["Deembedding"] = "conditioned_full_incident_matrix_solve"
            output.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
            output.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
            output.attrs["CasesCompleted"] = sum(bool(case_id) for case_id in result.case_ids)
            output.attrs["CaseCount"] = len(self.cases)
            output.attrs["Complete"] = all(bool(case_id) for case_id in result.case_ids)
            if self.source_path is not None:
                output.attrs["SourcePath"] = str(self.source_path)
            if self.source_text is not None:
                output.create_dataset("source", data=self.source_text)
            output.create_dataset("frequency", data=result.frequency)
            output.create_dataset("channel_ports", data=result.channel_ports)
            output.create_dataset("channel_modes", data=result.channel_modes)
            output.create_dataset("case_ids", data=np.asarray(result.case_ids, dtype="S"))
            output.create_dataset("S", data=result.s)
            output.create_dataset("valid_S", data=result.valid_s.astype(np.uint8))
            output.create_dataset("power_wave_valid_S", data=result.valid_s.astype(np.uint8))
            output.create_dataset("generalized_valid_S", data=result.generalized_valid_s.astype(np.uint8))
            output.create_dataset(
                "coefficient_valid_S",
                data=result.generalized_valid_s.astype(np.uint8),
            )
            output.create_dataset("incident_matrix", data=result.incident_matrix)
            output.create_dataset("outgoing_matrix", data=result.outgoing_matrix)
            output.create_dataset(
                "deembedding_condition_number",
                data=result.deembedding_condition_number,
            )
            output.create_dataset(
                "deembedding_valid",
                data=result.deembedding_valid.astype(np.uint8),
            )
            output.create_dataset(
                "valid_wave_matrix",
                data=result.wave_valid_matrix.astype(np.uint8),
            )
            output.create_dataset(
                "power_wave_valid_matrix",
                data=result.wave_valid_matrix.astype(np.uint8),
            )
            output.create_dataset(
                "generalized_valid_wave_matrix",
                data=result.generalized_wave_valid_matrix.astype(np.uint8),
            )
            output.create_dataset(
                "coefficient_valid_wave_matrix",
                data=result.generalized_wave_valid_matrix.astype(np.uint8),
            )
            if self.codebook is not None:
                codebook_group = output.create_group("array_codebook")
                codebook_group.attrs["Schema"] = self.codebook.schema
                codebook_group.attrs["Description"] = self.codebook.description
                codebook_group.create_dataset(
                    "definition",
                    data=self.codebook.to_json(),
                )
                if self.codebook.source_path is not None:
                    codebook_group.attrs["SourcePath"] = str(self.codebook.source_path)
                if self.codebook.source_text is not None:
                    codebook_group.create_dataset("source", data=self.codebook.source_text)

            if result.embedded_far_fields:
                embedded_group = output.create_group("embedded_far_fields")
                for bank in result.embedded_far_fields.values():
                    group = embedded_group.require_group(bank.transform_id).create_group(bank.output_id)
                    group.attrs["Normalisation"] = "field_per_sqrt_watt_incident_power_wave"
                    group.attrs["Impedance"] = bank.impedance
                    group.attrs["ThetaOrder"] = bank.theta_order
                    group.attrs["PhiOrder"] = bank.phi_order
                    group.attrs["EnclosureRadius"] = bank.enclosure_radius
                    group.create_dataset("frequency", data=bank.frequency)
                    group.create_dataset("theta", data=bank.theta)
                    group.create_dataset("phi", data=bank.phi)
                    group.create_dataset("Etheta", data=bank.etheta)
                    group.create_dataset("Ephi", data=bank.ephi)
                    group.create_dataset("valid", data=bank.valid.astype(np.uint8))
                    sphere = group.create_group("sphere")
                    sphere.create_dataset("theta", data=bank.sphere_theta)
                    sphere.create_dataset("phi", data=bank.sphere_phi)
                    sphere.create_dataset("weights", data=bank.sphere_weights)
                    sphere.create_dataset("Etheta", data=bank.sphere_etheta)
                    sphere.create_dataset("Ephi", data=bank.sphere_ephi)
                    raw = self._raw_embedded_far_fields[bank.key]
                    raw_group = group.create_group("raw_runs")
                    raw_group.attrs["FinalAxis"] = "excitation_case"
                    raw_group.attrs["Normalisation"] = "raw_range_normalized_run_field"
                    raw_group.create_dataset("Etheta", data=raw["etheta"])
                    raw_group.create_dataset("Ephi", data=raw["ephi"])
                    raw_group.create_dataset("valid", data=raw["valid"].astype(np.uint8))
                    raw_sphere = raw_group.create_group("sphere")
                    raw_sphere.create_dataset("Etheta", data=raw["sphere_etheta"])
                    raw_sphere.create_dataset("Ephi", data=raw["sphere_ephi"])

            if self.codebook is not None and self.codebook.states:
                states_group = output.create_group("array_states")
                states_group.attrs["DefinitionSchema"] = self.codebook.schema
                for state, state_result in zip(
                    self.codebook.states,
                    result.evaluate_codebook(self.codebook),
                ):
                    group = states_group.create_group(state.id)
                    group.attrs["Description"] = state.description
                    group.attrs["NetworkWaveNormalisation"] = "modal_power_wave"
                    drives = group.create_group("drives")
                    drives.create_dataset("port", data=[item.port for item in state.drives])
                    drives.create_dataset("mode", data=[item.mode for item in state.drives])
                    drives.create_dataset("power_w", data=[item.power for item in state.drives])
                    drives.create_dataset("phase_deg", data=[item.phase_deg for item in state.drives])
                    drives.create_dataset("delay_s", data=[item.delay_s for item in state.drives])
                    group.create_dataset("incident", data=state_result.incident)
                    group.create_dataset("outgoing", data=state_result.outgoing)
                    group.create_dataset("active_reflection", data=state_result.active_reflection)
                    group.create_dataset("incident_power", data=state_result.incident_power)
                    group.create_dataset("reflected_power", data=state_result.reflected_power)
                    group.create_dataset("accepted_power", data=state_result.accepted_power)
                    group.create_dataset("tarc", data=state_result.tarc)
                    group.create_dataset("valid", data=state_result.valid.astype(np.uint8))
                    for field_result in state_result.far_fields.values():
                        field_group = (
                            group.require_group("far_fields")
                            .require_group(field_result.transform_id)
                            .create_group(field_result.output_id)
                        )
                        field_group.create_dataset("theta", data=field_result.theta)
                        field_group.create_dataset("phi", data=field_result.phi)
                        for dataset_name, result_name in (
                            ("Etheta", "etheta"),
                            ("Ephi", "ephi"),
                            ("radiation_intensity", "radiation_intensity"),
                            ("radiated_power", "radiated_power"),
                            ("directivity", "directivity"),
                            ("directivity_dbi", "directivity_dbi"),
                            ("gain", "gain"),
                            ("gain_dbi", "gain_dbi"),
                            ("realized_gain", "realized_gain"),
                            ("realized_gain_dbi", "realized_gain_dbi"),
                            ("radiation_efficiency", "radiation_efficiency"),
                            ("total_efficiency", "total_efficiency"),
                            ("maximum_directivity", "maximum_directivity"),
                            ("maximum_directivity_dbi", "maximum_directivity_dbi"),
                            ("maximum_theta", "maximum_theta"),
                            ("maximum_phi", "maximum_phi"),
                        ):
                            field_group.create_dataset(
                                dataset_name,
                                data=getattr(field_result, result_name),
                            )
                        field_group.create_dataset("valid", data=field_result.valid.astype(np.uint8))


def preflight_study_args(args) -> Optional[Study]:
    """Resolve API/hash study input before SimulationConfig sizes its run."""

    study = getattr(args, "study", None)
    if study is not None and not isinstance(study, Study):
        raise TypeError("The study API argument must be a Study instance.")

    hash_spec = _find_hash_study(getattr(args, "inputfile", None))
    hash_codebook = _find_hash_array_codebook(getattr(args, "inputfile", None))
    if study is not None and hash_spec is not None:
        raise ValueError("Specify a study through either the Python API or #study, not both.")
    if study is not None and hash_codebook is not None:
        raise ValueError(
            "Specify an array codebook through either the EigenmodeStudy API or " "#array_codebook, not both."
        )
    if study is None and hash_spec is not None:
        study_type, csv_path = hash_spec
        study = Study.from_csv(study_type, csv_path)
    if hash_codebook is not None:
        if not isinstance(study, EigenmodeStudy):
            raise ValueError("#array_codebook requires an #study: eigenmode command.")
        study.codebook = ArrayCodebook.from_json(hash_codebook)
    if study is None:
        setattr(args, "study", None)
        return None

    if getattr(args, "taskfarm", False):
        raise ValueError("Studies do not yet support MPI task farming.")
    if getattr(args, "mpi", None) is not None:
        raise ValueError("Studies do not yet support MPI domain decomposition.")
    if isinstance(study, (PortStudy, EigenmodeStudy, PlaneWaveStudy, SourceStudy)) and getattr(
        args, "geometry_only", False
    ):
        raise ValueError(f"{type(study).__name__} requires a field solve and cannot use geometry_only.")
    scenes = getattr(args, "scenes", None)
    if scenes is not None and len(scenes) != 1:
        raise ValueError("A study requires exactly one reusable Scene.")

    count = len(study.cases)
    start = getattr(args, "i", None)
    if start is None:
        args.n = count
    else:
        if start <= 0 or start > count:
            raise ValueError(f"Study restart index i={start} is outside the {count} available cases.")
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


def _find_hash_array_codebook(inputfile: Optional[Union[str, Path]]) -> Optional[Path]:
    if inputfile is None:
        return None
    main_path = Path(inputfile).expanduser().resolve()
    found: list[Path] = []
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
                continue
            if stripped.startswith("#include_file:"):
                values = shlex.split(stripped.split(":", 1)[1])
                if len(values) != 1:
                    raise ValueError("#include_file requires exactly one parameter.")
                include = Path(values[0]).expanduser()
                if not include.is_absolute():
                    include = main_path.parent / include
                scan(include)
            elif stripped.startswith("#array_codebook:"):
                values = shlex.split(stripped.split(":", 1)[1])
                if len(values) != 1:
                    raise ValueError("#array_codebook requires exactly one JSON path.")
                codebook = Path(values[0]).expanduser()
                if not codebook.is_absolute():
                    codebook = main_path.parent / codebook
                found.append(codebook.resolve())

    scan(main_path)
    if len(found) > 1:
        raise ValueError("Only one #array_codebook command may be specified.")
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


def _parse_positive_int(value: str, path: Path, line: int, name: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"Study CSV '{path}', line {line}: {name} must be a positive integer.") from exc
    if result < 1:
        raise ValueError(f"Study CSV '{path}', line {line}: {name} must be a positive integer.")
    return result


def _parse_int(value: str, path: Path, line: int, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Study CSV '{path}', line {line}: {name} must be an integer.") from exc


def _safe_divide(numerator, denominator, complex_dtype):
    """Complex division retaining a separate algebraic validity mask."""

    numerator = np.asarray(numerator, dtype=complex_dtype)
    denominator = np.asarray(denominator, dtype=complex_dtype)
    magnitude = np.abs(denominator)
    finite = np.isfinite(denominator)
    scale = float(np.max(magnitude[finite], initial=0.0))
    threshold = np.finfo(np.empty((), dtype=complex_dtype).real.dtype).eps * scale
    valid = finite & (magnitude > threshold)
    result = np.full(numerator.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
    np.divide(numerator, denominator, out=result, where=valid)
    valid &= np.isfinite(result)
    return result, valid


def _jsonable(value: Any) -> Any:
    """Convert ordinary API parameter values to JSON-compatible builtins."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
