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

import math
from pathlib import Path

from .user_objects.cmds_singleuse import (
    DispersiveAveraging,
    Discretisation,
    Domain,
    DomainMode,
    MagneticAveraging,
    OMPThreads,
    OutputDir,
    PMLThickness,
    RxSteps,
    SrcSteps,
    TimeStepStabilityFactor,
    TimeWindow,
    Title,
)


def _split(command, value, count):
    tokens = value.split()
    if len(tokens) != count:
        noun = "parameter" if count == 1 else "parameters"
        number = "one" if count == 1 else str(count)
        raise ValueError(f"{command} requires exactly {number} {noun}; got {len(tokens)}")
    return tokens


def _numbers(command, tokens):
    values = []
    for index, token in enumerate(tokens, start=1):
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"{command} parameter {index} is not a number: {token!r}") from exc
    return values


def _integers(command, tokens):
    values = []
    for index, token in enumerate(tokens, start=1):
        try:
            values.append(int(token))
        except ValueError as exc:
            raise ValueError(f"{command} parameter {index} is not an integer: {token!r}") from exc
    return values


def _finite(command, values):
    for index, value in enumerate(values, start=1):
        if not math.isfinite(value):
            raise ValueError(f"{command} parameter {index} must be finite: {value!r}")


def process_singlecmds(singlecmds, *, input_dir=None):
    """Checks the validity of command parameters and creates instances of
        classes of parameters.

    Args:
        singlecmds: dict of commands that can only occur once in the model.
        input_dir: optional directory of the top-level input file, used to
            resolve relative hash-command output directories.

    Returns:
        scene_objects: list that holds objects in scene.
    """

    scene_objects = []

    cmd = "#title"
    if singlecmds[cmd] is not None:
        scene_objects.append(Title(name=str(singlecmds[cmd])))

    cmd = "#output_dir"
    if singlecmds[cmd] is not None:
        # Hash paths are relative to the top-level input, not the launch CWD.
        # Resolve here so Python OutputDir keeps its existing CWD semantics.
        directory = Path(singlecmds[cmd])
        if input_dir is not None and not directory.is_absolute():
            directory = Path(input_dir) / directory
        scene_objects.append(OutputDir(dir=str(directory)))

    # Number of threads for CPU-based (OpenMP) parallelised parts of code
    cmd = "#omp_threads"
    legacy_cmd = "#num_threads"
    if singlecmds[cmd] is not None and singlecmds[legacy_cmd] is not None:
        raise ValueError(f"{cmd} and its legacy alias {legacy_cmd} cannot both be specified")

    selected_cmd = cmd if singlecmds[cmd] is not None else legacy_cmd
    if singlecmds[selected_cmd] is not None:
        tokens = _split(selected_cmd, singlecmds[selected_cmd], 1)
        scene_objects.append(OMPThreads(n=_integers(selected_cmd, tokens)[0]))

    cmd = "#dx_dy_dz"
    if singlecmds[cmd] is not None:
        tokens = _split(cmd, singlecmds[cmd], 3)
        scene_objects.append(Discretisation(p1=tuple(_numbers(cmd, tokens))))

    cmd = "#domain_mode"
    if singlecmds[cmd] is not None:
        tokens = _split(cmd, singlecmds[cmd], 1)
        scene_objects.append(DomainMode(mode=tokens[0]))

    cmd = "#magnetic_averaging"
    if singlecmds[cmd] is not None:
        tokens = _split(cmd, singlecmds[cmd], 1)
        scene_objects.append(MagneticAveraging(mode=tokens[0]))

    cmd = "#dispersive_averaging"
    if singlecmds[cmd] is not None:
        tokens = _split(cmd, singlecmds[cmd], 1)
        if tokens[0].lower() not in ("y", "n"):
            raise ValueError(f"{cmd} requires 'y' or 'n'; got {tokens[0]!r}")
        scene_objects.append(DispersiveAveraging(enabled=tokens[0].lower() == "y"))

    cmd = "#domain"
    if singlecmds[cmd] is not None:
        tokens = _split(cmd, singlecmds[cmd], 3)
        scene_objects.append(Domain(p1=tuple(_numbers(cmd, tokens))))

    cmd = "#time_step_stability_factor"
    if singlecmds[cmd] is not None:
        tokens = _split(cmd, singlecmds[cmd], 1)
        scene_objects.append(TimeStepStabilityFactor(f=_numbers(cmd, tokens)[0]))

    cmd = "#time_window"
    if singlecmds[cmd] is not None:
        token = _split(cmd, singlecmds[cmd], 1)[0]
        try:
            iterations = int(token)
        except ValueError:
            try:
                time = float(token)
            except ValueError as exc:
                raise ValueError(
                    f"{cmd} requires seconds or an iteration count; got {token!r}"
                ) from exc
            scene_objects.append(TimeWindow(time=time))
        else:
            scene_objects.append(TimeWindow(iterations=iterations))

    cmd = "#pml_cells"
    if singlecmds[cmd] is not None:
        tokens = singlecmds[cmd].split()
        if len(tokens) not in (1, 6):
            raise ValueError(f"{cmd} requires exactly one or six parameters; got {len(tokens)}")
        thickness = tuple(_integers(cmd, tokens))
        if len(thickness) == 1:
            scene_objects.append(PMLThickness(thickness=thickness[0]))
        else:
            scene_objects.append(PMLThickness(thickness=thickness))

    cmd = "#src_steps"
    if singlecmds[cmd] is not None:
        tokens = _split(cmd, singlecmds[cmd], 3)
        step_size = tuple(_numbers(cmd, tokens))
        _finite(cmd, step_size)
        scene_objects.append(SrcSteps(p1=step_size))

    cmd = "#rx_steps"
    if singlecmds[cmd] is not None:
        tokens = _split(cmd, singlecmds[cmd], 3)
        step_size = tuple(_numbers(cmd, tokens))
        _finite(cmd, step_size)
        scene_objects.append(RxSteps(p1=step_size))

    return scene_objects
