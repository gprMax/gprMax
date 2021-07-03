# Copyright (C) 2015-2021: The University of Edinburgh
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import logging

from .cmds_singleuse import (Discretisation, Domain, ExcitationFile,
                             OMPThreads, OutputDir, PMLCells, RxSteps,
                             SrcSteps, TimeStepStabilityFactor, TimeWindow,
                             Title)

logger = logging.getLogger(__name__)


def process_singlecmds(singlecmds):
    """
    Checks the validity of command parameters and creates instances of
        classes of parameters.

    Args:
        singlecmds (dict): Commands that can only occur once in the model.

    Returns:
        scene_objects (list): Holds objects in scene.
    """

    scene_objects = []

    # Check validity of command parameters in order needed
    cmd = '#title'
    if singlecmds[cmd] is not None:
        title = Title(name=str(singlecmds[cmd]))
        scene_objects.append(title)

    cmd = '#output_dir'
    if singlecmds[cmd] is not None:
        output_dir = OutputDir(dir=singlecmds[cmd])
        scene_objects.append(output_dir)

    # Number of threads for CPU-based (OpenMP) parallelised parts of code
    cmd = '#cpu_threads'
    if singlecmds[cmd] is not None:
        tmp = tuple(int(x) for x in singlecmds[cmd].split())
        if len(tmp) != 1:
            logger.exception(cmd + ' requires exactly one parameter to specify the number of CPU OpenMP threads to use')
            raise ValueError

        omp_threads = OMPThreads(n=tmp[0])
        scene_objects.append(omp_threads)

    cmd = '#dx_dy_dz'
    if singlecmds[cmd] is not None:
        tmp = [float(x) for x in singlecmds[cmd].split()]
        if len(tmp) != 3:
            logger.exception(cmd + ' requires exactly three parameters')
            raise ValueError

        dl = (tmp[0], tmp[1], tmp[2])
        discretisation = Discretisation(p1=dl)
        scene_objects.append(discretisation)

    cmd = '#domain'
    if singlecmds[cmd] is not None:
        tmp = [float(x) for x in singlecmds[cmd].split()]
        if len(tmp) != 3:
            logger.exception(cmd + ' requires exactly three parameters')
            raise ValueError

        p1 = (tmp[0], tmp[1], tmp[2])
        domain = Domain(p1=p1)
        domain_bounds = p1
        scene_objects.append(domain)

    cmd = '#time_step_stability_factor'
    if singlecmds[cmd] is not None:
        tmp = tuple(float(x) for x in singlecmds[cmd].split())
        tsf = TimeStepStabilityFactor(f=tmp[0])
        scene_objects.append(tsf)

    cmd = '#time_window'
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1:
            logger.exception(cmd + ' requires exactly one parameter to specify the time window. Either in seconds or number of iterations.')
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

    cmd = '#pml_cells'
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1 and len(tmp) != 6:
            logger.exception(cmd + ' requires either one or six parameter(s)')
            raise ValueError
        if len(tmp) == 1:
            pml_cells = PMLCells(thickness=int(tmp[0]))
        else:
            pml_cells = PMLCells(x0=int(tmp[0]),
                                 y0=int(tmp[1]),
                                 z0=int(tmp[2]),
                                 xmax=int(tmp[3]),
                                 ymax=int(tmp[4]),
                                 zmax=int(tmp[5]))

        scene_objects.append(pml_cells)

    cmd = '#src_steps'
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 3:
            logger.exception(cmd + ' requires exactly three parameters')
            raise ValueError

        p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
        src_steps = SrcSteps(p1=p1)
        scene_objects.append(src_steps)

    cmd = '#rx_steps'
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 3:
            logger.exception(cmd + ' requires exactly three parameters')
            raise ValueError

        p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
        rx_steps = RxSteps(p1=p1)
        scene_objects.append(rx_steps)

    # Excitation file for user-defined source waveforms
    cmd = '#excitation_file'
    if singlecmds[cmd] is not None:
        tmp = singlecmds[cmd].split()
        if len(tmp) != 1 and len(tmp) != 3:
            logger.exception(cmd + ' requires either one or three parameter(s)')
            raise ValueError

        if len(tmp) > 1:
            ex_file = ExcitationFile(filepath=tmp[0], kind=tmp[1], fill_value=tmp[2])
        else:
            ex_file = ExcitationFile(filepath=tmp[0])

        scene_objects.append(ex_file)

    return scene_objects, domain_bounds
