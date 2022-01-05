# Copyright (C) 2015-2022: The University of Edinburgh
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
import os

os.path.join(os.path.dirname(__file__), '..', 'user_libs', 'DebyeFit')
from user_libs.DebyeFit import (HavriliakNegami, Jonscher, Crim, Rawdata)

from .cmds_multiuse import (PMLCFS, AddDebyeDispersion, AddDrudeDispersion,
                            AddLorentzDispersion, GeometryObjectsWrite,
                            GeometryView, HertzianDipole, MagneticDipole,
                            Material, Rx, RxArray, Snapshot, SoilPeplinski,
                            TransmissionLine, VoltageSource, Waveform)

logger = logging.getLogger(__name__)


def process_multicmds(multicmds):
    """
    Checks the validity of command parameters and creates instances of
        classes of parameters.

    Args:
        multicmds (dict): Commands that can have multiple instances in the model.

    Returns:
        scene_objects (list): Holds objects in scene.
    """

    scene_objects = []

    cmdname = '#waveform'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 4:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly four parameters')
                raise ValueError

            waveform = Waveform(wave_type=tmp[0], amp=float(tmp[1]), freq=float(tmp[2]), id=tmp[3])
            scene_objects.append(waveform)

    cmdname = '#voltage_source'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) == 6:
                voltage_source = VoltageSource(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), resistance=float(tmp[4]), waveform_id=tmp[5])
            elif len(tmp) == 8:
                voltage_source = VoltageSource(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), resistance=float(tmp[4]), waveform_id=tmp[5], start=float(tmp[6]), end=float(tmp[7]))
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least six parameters')
                raise ValueError

            scene_objects.append(voltage_source)

    cmdname = '#hertzian_dipole'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
                raise ValueError
            if len(tmp) == 5:
                hertzian_dipole = HertzianDipole(polarisation=tmp[0], p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4])
            elif len(tmp) == 7:
                hertzian_dipole = HertzianDipole(polarisation=tmp[0], p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4], start=float(tmp[5]), end=float(tmp[6]))
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' too many parameters')
                raise ValueError

            scene_objects.append(hertzian_dipole)

    cmdname = '#magnetic_dipole'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
                raise ValueError
            if len(tmp) == 5:
                magnetic_dipole = MagneticDipole(polarisation=tmp[0], p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4])
            elif len(tmp) == 7:
                magnetic_dipole = MagneticDipole(polarisation=tmp[0], p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4], start=float(tmp[5]), end=float(tmp[6]))
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' too many parameters')
                raise ValueError

            scene_objects.append(magnetic_dipole)

    cmdname = '#transmission_line'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least six parameters')
                raise ValueError

            if len(tmp) == 6:
                tl = TransmissionLine(polarisation=tmp[0], p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), resistance=float(tmp[4]), waveform_id=tmp[5])
            elif len(tmp) == 8:
                tl = TransmissionLine(polarisation=tmp[0], p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), resistance=float(tmp[4]), waveform_id=tmp[5], start=tmp[6], end=tmp[7])
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' too many parameters')
                raise ValueError

            scene_objects.append(tl)

    cmdname = '#rx'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 3 and len(tmp) < 5:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')
                raise ValueError
            if len(tmp) == 3:
                rx = Rx(p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])))
            else:
                rx = Rx(p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])), id=tmp[3], outputs=' '.join(tmp[4:]))

            scene_objects.append(rx)

    cmdname = '#rx_array'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 9:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly nine parameters')
                raise ValueError

            p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
            p2 = (float(tmp[3]), float(tmp[4]), float(tmp[5]))
            dl = (float(tmp[6]), float(tmp[7]), float(tmp[8]))

            rx_array = RxArray(p1=p1, p2=p2, dl=dl)
            scene_objects.append(rx_array)

    cmdname = '#snapshot'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly eleven parameters')
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

    cmdname = '#havriliak_negami'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 12 and len(tmp) != 13:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires either twelve or thirteen parameters')
                raise ValueError
            seed = None
            if len(tmp) == 13:
                seed = int(tmp[12])

            setup = HavriliakNegami(f_min=float(tmp[0]), f_max=float(tmp[1]),
                                    alpha=float(tmp[2]), beta=float(tmp[3]),
                                    e_inf=float(tmp[4]), de=float(tmp[5]), tau_0=float(tmp[6]),
                                    sigma=float(tmp[7]), mu=float(tmp[8]), mu_sigma=float(tmp[9]),
                                    number_of_debye_poles=int(tmp[10]), material_name=tmp[11],
                                    optimizer_options={'seed': seed})
            _, properties = setup.run()

            multicmds['#material'].append(properties[0].split(':')[1].strip(' \t\n'))
            multicmds['#add_dispersion_debye'].append(properties[1].split(':')[1].strip(' \t\n'))

    cmdname = '#jonscher'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 11 and len(tmp) != 12:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires either eleven or twelve parameters')
                raise ValueError
            seed = None
            if len(tmp) == 12:
                seed = int(tmp[11])

            setup = Jonscher(f_min=float(tmp[0]), f_max=float(tmp[1]),
                             e_inf=float(tmp[2]), a_p=float(tmp[3]),
                             omega_p=float(tmp[4]), n_p=float(tmp[5]),
                             sigma=float(tmp[6]), mu=float(tmp[7]), mu_sigma=float(tmp[8]),
                             number_of_debye_poles=int(tmp[9]), material_name=tmp[10],
                             optimizer_options={'seed': seed})
            _, properties = setup.run()

            multicmds['#material'].append(properties[0].split(':')[1].strip(' \t\n'))
            multicmds['#add_dispersion_debye'].append(properties[1].split(':')[1].strip(' \t\n'))

    cmdname = '#crim'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 10 and len(tmp) != 11:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires either ten or eleven parameters')
                raise ValueError
            seed = None
            if len(tmp) == 11:
                seed = int(tmp[10])

            if (tmp[3][0] != '[' and tmp[3][-1] != ']') or (tmp[4][0] != '[' and tmp[4][-1] != ']'):
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires list at 6th and 7th position')
                raise ValueError
            vol_frac = [float(i) for i in tmp[3].strip('[]').split(',')]
            material = [float(i) for i in tmp[4].strip('[]').split(',')]
            if len(material) % 3 != 0:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' each material requires three parameters: e_inf, de, tau_0')
                raise ValueError
            materials = [material[n:n+3] for n in range(0, len(material), 3)]

            setup = Crim(f_min=float(tmp[0]), f_max=float(tmp[1]), a=float(tmp[2]),
                         volumetric_fractions=vol_frac, materials=materials,
                         sigma=float(tmp[5]), mu=float(tmp[6]), mu_sigma=float(tmp[7]),
                         number_of_debye_poles=int(tmp[8]), material_name=tmp[9],
                         optimizer_options={'seed': seed})
            _, properties = setup.run()

            multicmds['#material'].append(properties[0].split(':')[1].strip(' \t\n'))
            multicmds['#add_dispersion_debye'].append(properties[1].split(':')[1].strip(' \t\n'))

    cmdname = '#raw_data'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 6 and len(tmp) != 7:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires either six or seven parameters')
                raise ValueError
            seed = None
            if len(tmp) == 7:
                seed = int(tmp[6])

            setup = Rawdata(filename=tmp[0], sigma=float(tmp[1]),
                            mu=float(tmp[2]), mu_sigma=float(tmp[3]),
                            number_of_debye_poles=int(tmp[4]), material_name=tmp[5],
                            optimizer_options={'seed': seed})
            _, properties = setup.run()

            multicmds['#material'].append(properties[0].split(':')[1].strip(' \t\n'))
            multicmds['#add_dispersion_debye'].append(properties[1].split(':')[1].strip(' \t\n'))

    cmdname = '#material'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 5:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly five parameters')
                raise ValueError

            material = Material(er=float(tmp[0]), se=float(tmp[1]), mr=float(tmp[2]), sm=float(tmp[3]), id=tmp[4])
            scene_objects.append(material)

    cmdname = '#add_dispersion_debye'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 4:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least four parameters')
                raise ValueError

            poles = int(tmp[0])
            er_delta = []
            tau = []
            material_ids = tmp[(2 * poles) + 1:len(tmp)]

            for pole in range(1, 2 * poles, 2):
                er_delta.append(float(tmp[pole]))
                tau.append(float(tmp[pole + 1]))

            debye_dispersion = AddDebyeDispersion(poles=poles, er_delta=er_delta, tau=tau, material_ids=material_ids)
            scene_objects.append(debye_dispersion)

    cmdname = '#add_dispersion_lorentz'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 5:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
                raise ValueError

            poles = int(tmp[0])
            material_ids = tmp[(3 * poles) + 1:len(tmp)]
            er_delta = []
            tau = []
            alpha = []

            for pole in range(1, 3 * poles, 3):
                er_delta.append(float(tmp[pole]))
                tau.append(float(tmp[pole + 1]))
                alpha.append(float(tmp[pole + 2]))

            lorentz_dispersion = AddLorentzDispersion(poles=poles, material_ids=material_ids, er_delta=er_delta, tau=tau, alpha=alpha)
            scene_objects.append(lorentz_dispersion)

    cmdname = '#add_dispersion_drude'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 5:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
                raise ValueError

            poles = int(tmp[0])
            material_ids = tmp[(3 * poles) + 1:len(tmp)]
            tau = []
            alpha = []

            for pole in range(1, 2 * poles, 2):
                tau.append(float(tmp[pole]))
                alpha.append(float(tmp[pole + 1]))

            drude_dispersion = AddDrudeDispersion(poles=poles, material_ids=material_ids, tau=tau, alpha=alpha)
            scene_objects.append(drude_dispersion)

    cmdname = '#soil_peplinski'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 7:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at exactly seven parameters')
                raise ValueError
            soil = SoilPeplinski(sand_fraction=float(tmp[0]),
                                 clay_fraction=float(tmp[1]),
                                 bulk_density=float(tmp[2]),
                                 sand_density=float(tmp[3]),
                                 water_fraction_lower=float(tmp[4]),
                                 water_fraction_upper=float(tmp[5]),
                                 id=tmp[6])
            scene_objects.append(soil)

    cmdname = '#geometry_view'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly eleven parameters')
                raise ValueError

            p1 = float(tmp[0]), float(tmp[1]), float(tmp[2])
            p2 = float(tmp[3]), float(tmp[4]), float(tmp[5])
            dl = float(tmp[6]), float(tmp[7]), float(tmp[8])

            geometry_view = GeometryView(p1=p1, p2=p2, dl=dl, filename=tmp[9], output_type=tmp[10])
            scene_objects.append(geometry_view)

    cmdname = '#geometry_objects_write'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 7:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly seven parameters')
                raise ValueError

            p1 = float(tmp[0]), float(tmp[1]), float(tmp[2])
            p2 = float(tmp[3]), float(tmp[4]), float(tmp[5])
            gow = GeometryObjectsWrite(p1=p1, p2=p2, filename=tmp[6])
            scene_objects.append(gow)

    return scene_objects
