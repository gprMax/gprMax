# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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

from .cmds_multiuse import (
    PMLCFS,
    AddDebyeDispersion,
    AddDrudeDispersion,
    AddLorentzDispersion,
    ExcitationFile,
    GeometryObjectsWrite,
    GeometryView,
    HertzianDipole,
    MagneticDipole,
    Material,
    MaterialList,
    MaterialRange,
    Rx,
    RxArray,
    Snapshot,
    SoilPeplinski,
    TransmissionLine,
    VoltageSource,
    Waveform,
)

logger = logging.getLogger(__name__)


def process_multicmds(multicmds):
    """Checks the validity of command parameters and creates instances of
        classes of parameters.

    Args:
        multicmds: dict of commands that can have multiple instances in the model.

    Returns:
        scene_objects: list that holds objects in scene.
    """

    scene_objects = []

    cmdname = "#waveform"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 4:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly four parameters")
                raise ValueError

            waveform = Waveform(wave_type=tmp[0], amp=float(tmp[1]), freq=float(tmp[2]), id=tmp[3])
            scene_objects.append(waveform)

    cmdname = "#voltage_source"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) == 6:
                voltage_source = VoltageSource(
                    polarisation=tmp[0].lower(),
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    resistance=float(tmp[4]),
                    waveform_id=tmp[5],
                )
            elif len(tmp) == 8:
                voltage_source = VoltageSource(
                    polarisation=tmp[0].lower(),
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    resistance=float(tmp[4]),
                    waveform_id=tmp[5],
                    start=float(tmp[6]),
                    end=float(tmp[7]),
                )
            else:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least six parameters")
                raise ValueError

            scene_objects.append(voltage_source)

    cmdname = "#hertzian_dipole"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least five parameters")
                raise ValueError
            if len(tmp) == 5:
                hertzian_dipole = HertzianDipole(
                    polarisation=tmp[0], p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4]
                )
            elif len(tmp) == 7:
                hertzian_dipole = HertzianDipole(
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    waveform_id=tmp[4],
                    start=float(tmp[5]),
                    end=float(tmp[6]),
                )
            else:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters")
                raise ValueError

            scene_objects.append(hertzian_dipole)

    cmdname = "#magnetic_dipole"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least five parameters")
                raise ValueError
            if len(tmp) == 5:
                magnetic_dipole = MagneticDipole(
                    polarisation=tmp[0], p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4]
                )
            elif len(tmp) == 7:
                magnetic_dipole = MagneticDipole(
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    waveform_id=tmp[4],
                    start=float(tmp[5]),
                    end=float(tmp[6]),
                )
            else:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters")
                raise ValueError

            scene_objects.append(magnetic_dipole)

    cmdname = "#transmission_line"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least six parameters")
                raise ValueError

            if len(tmp) == 6:
                tl = TransmissionLine(
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    resistance=float(tmp[4]),
                    waveform_id=tmp[5],
                )
            elif len(tmp) == 8:
                tl = TransmissionLine(
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    resistance=float(tmp[4]),
                    waveform_id=tmp[5],
                    start=tmp[6],
                    end=tmp[7],
                )
            else:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters")
                raise ValueError

            scene_objects.append(tl)

    cmd = "#excitation_file"
    if multicmds[cmd] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) not in [1, 3]:
                logger.exception(f"{cmd} requires either one or three parameter(s)")
                raise ValueError

            if len(tmp) > 1:
                ex_file = ExcitationFile(filepath=tmp[0], kind=tmp[1], fill_value=tmp[2])
            else:
                ex_file = ExcitationFile(filepath=tmp[0])

            scene_objects.append(ex_file)

    cmdname = "#rx"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 3 and len(tmp) < 5:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " has an incorrect number of parameters")
                raise ValueError
            if len(tmp) == 3:
                rx = Rx(p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])))
            else:
                rx = Rx(p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])), id=tmp[3], outputs=" ".join(tmp[4:]))

            scene_objects.append(rx)

    cmdname = "#rx_array"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 9:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly nine parameters")
                raise ValueError

            p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
            p2 = (float(tmp[3]), float(tmp[4]), float(tmp[5]))
            dl = (float(tmp[6]), float(tmp[7]), float(tmp[8]))

            rx_array = RxArray(p1=p1, p2=p2, dl=dl)
            scene_objects.append(rx_array)

    cmdname = "#snapshot"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly eleven parameters")
                raise ValueError

            p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
            p2 = (float(tmp[3]), float(tmp[4]), float(tmp[5]))
            dl = (float(tmp[6]), float(tmp[7]), float(tmp[8]))
            filename = tmp[10]

            try:
                iterations = int(tmp[9])
                snapshot = Snapshot(p1=p1, p2=p2, dl=dl, iterations=iterations, filename=filename)

            except ValueError:
                time = float(tmp[9])
                snapshot = Snapshot(p1=p1, p2=p2, dl=dl, time=time, filename=filename)

            scene_objects.append(snapshot)

    cmdname = "#material"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 5:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly five parameters")
                raise ValueError

            material = Material(er=float(tmp[0]), se=float(tmp[1]), mr=float(tmp[2]), sm=float(tmp[3]), id=tmp[4])
            scene_objects.append(material)

    cmdname = "#add_dispersion_debye"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 4:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least four parameters")
                raise ValueError

            poles = int(tmp[0])
            er_delta = []
            tau = []
            material_ids = tmp[(2 * poles) + 1 : len(tmp)]

            for pole in range(1, 2 * poles, 2):
                er_delta.append(float(tmp[pole]))
                tau.append(float(tmp[pole + 1]))

            debye_dispersion = AddDebyeDispersion(poles=poles, er_delta=er_delta, tau=tau, material_ids=material_ids)
            scene_objects.append(debye_dispersion)

    cmdname = "#add_dispersion_lorentz"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 5:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least five parameters")
                raise ValueError

            poles = int(tmp[0])
            material_ids = tmp[(3 * poles) + 1 : len(tmp)]
            er_delta = []
            tau = []
            alpha = []

            for pole in range(1, 3 * poles, 3):
                er_delta.append(float(tmp[pole]))
                tau.append(float(tmp[pole + 1]))
                alpha.append(float(tmp[pole + 2]))

            lorentz_dispersion = AddLorentzDispersion(
                poles=poles, material_ids=material_ids, er_delta=er_delta, tau=tau, alpha=alpha
            )
            scene_objects.append(lorentz_dispersion)

    cmdname = "#add_dispersion_drude"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 5:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least five parameters")
                raise ValueError

            poles = int(tmp[0])
            material_ids = tmp[(3 * poles) + 1 : len(tmp)]
            tau = []
            alpha = []

            for pole in range(1, 2 * poles, 2):
                tau.append(float(tmp[pole]))
                alpha.append(float(tmp[pole + 1]))

            drude_dispersion = AddDrudeDispersion(poles=poles, material_ids=material_ids, tau=tau, alpha=alpha)
            scene_objects.append(drude_dispersion)

    cmdname = "#soil_peplinski"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 7:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at exactly seven parameters")
                raise ValueError
            soil = SoilPeplinski(
                sand_fraction=float(tmp[0]),
                clay_fraction=float(tmp[1]),
                bulk_density=float(tmp[2]),
                sand_density=float(tmp[3]),
                water_fraction_lower=float(tmp[4]),
                water_fraction_upper=float(tmp[5]),
                id=tmp[6],
            )
            scene_objects.append(soil)

    cmdname = "#geometry_view"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly eleven parameters")
                raise ValueError

            p1 = float(tmp[0]), float(tmp[1]), float(tmp[2])
            p2 = float(tmp[3]), float(tmp[4]), float(tmp[5])
            dl = float(tmp[6]), float(tmp[7]), float(tmp[8])

            geometry_view = GeometryView(p1=p1, p2=p2, dl=dl, filename=tmp[9], output_type=tmp[10])
            scene_objects.append(geometry_view)

    cmdname = "#geometry_objects_write"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 7:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly seven parameters")
                raise ValueError

            p1 = float(tmp[0]), float(tmp[1]), float(tmp[2])
            p2 = float(tmp[3]), float(tmp[4]), float(tmp[5])
            gow = GeometryObjectsWrite(p1=p1, p2=p2, filename=tmp[6])
            scene_objects.append(gow)

    cmdname = "#material_range"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 9:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at exactly nine parameters")
                raise ValueError
            material_range = MaterialRange(
                er_lower=float(tmp[0]),
                er_upper=float(tmp[1]),
                sigma_lower=float(tmp[2]),
                sigma_upper=float(tmp[3]),
                mr_lower=float(tmp[4]),
                mr_upper=float(tmp[5]),
                ro_lower=float(tmp[6]),
                ro_upper=float(tmp[7]),
                id=tmp[8],
            )
            scene_objects.append(material_range)

    cmdname = "#material_list"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 2:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least two parameters")
                raise ValueError

            tokens = len(tmp)
            lmats = []
            for iter in range(tokens - 1):
                lmats.append(tmp[iter])

            material_list = MaterialList(list_of_materials=lmats, id=tmp[tokens - 1])
            scene_objects.append(material_list)

    cmdname = "#pml_cfs"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 12:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly twelve parameters")
                raise ValueError

            pml_cfs = PMLCFS(
                alphascalingprofile=tmp[0],
                alphascalingdirection=tmp[1],
                alphamin=tmp[2],
                alphamax=tmp[3],
                kappascalingprofile=tmp[4],
                kappascalingdirection=tmp[5],
                kappamin=tmp[6],
                kappamax=tmp[7],
                sigmascalingprofile=tmp[8],
                sigmascalingdirection=tmp[9],
                sigmamin=tmp[10],
                sigmamax=tmp[11],
            )

            scene_objects.append(pml_cfs)

    return scene_objects
