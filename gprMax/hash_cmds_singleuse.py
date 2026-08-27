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

import logging

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

logger = logging.getLogger(__name__)


def process_singlecmds(singlecmds):
    """Checks the validity of command parameters and creates instances of
        classes of parameters.

    Args:
        singlecmds: dict of commands that can only occur once in the model.

    Returns:
        scene_objects: list that holds objects in scene.
    """

    scene_objects = []

    # Check validity of command parameters in order needed
    cmd = "#title"
    if singlecmds[cmd] is not None:
        title = Title(name=str(singlecmds[cmd]))
        scene_objects.append(title)

    cmd = "#output_dir"
    if singlecmds[cmd] is not None:
        output_dir = OutputDir(dir=singlecmds[cmd])
        scene_objects.append(output_dir)

    # Number of threads for CPU-based (OpenMP) parallelised parts of code
    cmd = "#omp_threads"
    legacy_cmd = "#num_threads"
    if singlecmds[cmd] is not None and singlecmds[legacy_cmd] is not None:
        message = f"{cmd} and its legacy alias {legacy_cmd} cannot both be specified"
        logger.error(message)
        raise ValueError(message)

    selected_cmd = cmd if singlecmds[cmd] is not None else legacy_cmd
    if singlecmds[selected_cmd] is not None:
        tmp = tuple(int(x) for x in singlecmds[selected_cmd].split())
        if len(tmp) != 1:
            logger.exception(
                f"{selected_cmd} requires exactly one parameter to specify the number of CPU OpenMP threads to use"
            )
            raise ValueError

        omp_threads = OMPThreads(n=tmp[0])
        scene_objects.append(omp_threads)

    cmd = "#dx_dy_dz"
    if singlecmds[cmd] is not None:
        tmp = [float(x) for x in singlecmds[cmd].split()]
        if len(tmp) != 3:
            logger.exception(f"{cmd} requires exactly three parameters")
            raise ValueError

        dl = (tmp[0], tmp[1], tmp[2])
        discretisation = Discretisation(p1=dl)
        scene_objects.append(discretisation)

    cmd = "#domain_mode"
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            logger.exception(f"{cmd} requires exactly one parameter, either 'TM', 'TE' or '3D'")
            raise ValueError

        domain_mode = DomainMode(mode=tmp[0])
        scene_objects.append(domain_mode)

    cmd = "#magnetic_averaging"
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            logger.exception(f"{cmd} requires exactly one parameter, either 'harmonic' or 'arithmetic'")
            raise ValueError

        magnetic_averaging = MagneticAveraging(mode=tmp[0])
        scene_objects.append(magnetic_averaging)

    cmd = "#dispersive_averaging"
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1 or tmp[0].lower() not in ("y", "n"):
            logger.exception(f"{cmd} requires exactly one parameter, either 'y' or 'n'")
            raise ValueError

        dispersive_averaging = DispersiveAveraging(enabled=tmp[0].lower() == "y")
        scene_objects.append(dispersive_averaging)

    cmd = "#domain"
    if singlecmds[cmd] is not None:
        tmp = [float(x) for x in singlecmds[cmd].split()]
        if len(tmp) != 3:
            logger.exception(f"{cmd} requires exactly three parameters")
            raise ValueError

        p1 = (tmp[0], tmp[1], tmp[2])
        domain = Domain(p1=p1)
        scene_objects.append(domain)

    cmd = "#time_step_stability_factor"
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            message = f"{cmd} requires exactly one parameter"
            logger.error(message)
            raise ValueError(message)

        tmp = tuple(float(x) for x in tmp)
        tsf = TimeStepStabilityFactor(f=tmp[0])
        scene_objects.append(tsf)

    cmd = "#time_window"
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            logger.exception(
                f"{cmd} requires exactly one parameter to specify the "
                f"time window. Either in seconds or number of iterations."
            )
            raise ValueError
        tmp = tmp[0].lower()

        # If number of iterations given
        try:
            tmp = int(tmp)
            tw = TimeWindow(iterations=tmp)
        # If real floating point value given
        except ValueError:
            tmp = float(tmp)
            tw = TimeWindow(time=tmp)

        scene_objects.append(tw)

    cmd = "#pml_cells"
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) not in [1, 6]:
            logger.exception(f"{cmd} requires either one or six parameter(s)")
            raise ValueError

        if len(tmp) == 1:
            pml_thickness = PMLThickness(thickness=int(tmp[0]))
        else:
            pml_thickness = PMLThickness(
                thickness=(
                    int(tmp[0]),
                    int(tmp[1]),
                    int(tmp[2]),
                    int(tmp[3]),
                    int(tmp[4]),
                    int(tmp[5]),
                )
            )

        scene_objects.append(pml_thickness)

    cmd = "#src_steps"
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 3:
            logger.exception(f"{cmd} requires exactly three parameters")
            raise ValueError

        p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
        src_steps = SrcSteps(p1=p1)
        scene_objects.append(src_steps)

    cmd = "#rx_steps"
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 3:
            logger.exception(f"{cmd} requires exactly three parameters")
            raise ValueError

        p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
        rx_steps = RxSteps(p1=p1)
        scene_objects.append(rx_steps)

    return scene_objects
