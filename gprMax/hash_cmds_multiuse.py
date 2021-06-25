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
import numpy as np
from .random_create import (Rand_Create, Save_Params)
import gprMax.config as config
from .cmds_multiuse import (PMLCFS, AddDebyeDispersion, AddDrudeDispersion,
                            AddLorentzDispersion, GeometryObjectsWrite,
                            GeometryView, HertzianDipole, MagneticDipole,
                            Material, Rx, RxArray, Snapshot, SoilPeplinski,
                            TransmissionLine, VoltageSource, Waveform)

logger = logging.getLogger(__name__)


def process_multicmds(multicmds, hash_count_multiplecmds):
    """
    Checks the validity of command parameters and creates instances of
        classes of parameters.

    Args:
        multicmds (dict): Commands that can have multiple instances in the model.
        hash_count_multiplecmds (dict): Number of times a hash command is called in an input file.

    Returns:
        scene_objects (list): Holds objects in scene.
    """

    scene_objects = []

    cmdname = '#waveform'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) == 4:
                waveform = Waveform(wave_type=tmp[0], amp=float(tmp[1]), freq=float(tmp[2]), id=tmp[3])
            elif len(tmp) == 7:
                rand_params = []
                distr = tmp[0].lower()
                amp = Rand_Create(distr, float(tmp[2]), float(tmp[3]))
                freq = Rand_Create(distr, float(tmp[4]), float(tmp[5]))

                rand_params.extend((amp, freq))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                waveform = Waveform(wave_type=tmp[1], amp=amp, freq=freq, id=tmp[6])
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly four parameters (if single values entered) or seven parameters (if range of values entered)')
                raise ValueError

            scene_objects.append(waveform)

    cmdname = '#voltage_source'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least six parameters')
                raise ValueError
            elif len(tmp) == 6:
                voltage_source = VoltageSource(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), resistance=float(tmp[4]), waveform_id=tmp[5])
            elif len(tmp) == 8:
                voltage_source = VoltageSource(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), resistance=float(tmp[4]), waveform_id=tmp[5], start=float(tmp[6]), end=float(tmp[7])) 
            
            elif len(tmp) == 11:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[2]), float(tmp[3])), 
                      Rand_Create(distr, float(tmp[4]), float(tmp[5])), 
                      Rand_Create(distr, float(tmp[6]), float(tmp[7])))
                resistance = Rand_Create(distr, float(tmp[8]), float(tmp[9]))

                rand_params.extend((p1, resistance))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                voltage_source = VoltageSource(polarisation=tmp[1].lower(), p1=p1, resistance=resistance, waveform_id = tmp[10])

            elif len(tmp) == 15:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[2]), float(tmp[3])), 
                      Rand_Create(distr, float(tmp[4]), float(tmp[5])), 
                      Rand_Create(distr, float(tmp[6]), float(tmp[7])))
                resistance = Rand_Create(distr, float(tmp[8]), float(tmp[9]))
                start = Rand_Create(distr, float(tmp[11]), float(tmp[12])) 
                end = Rand_Create(distr, float(tmp[13]), float(tmp[14]))

                rand_params.extend((p1, resistance, start, end))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                voltage_source = VoltageSource(polarisation=tmp[1].lower(), p1=p1, resistance=resistance, waveform_id=tmp[10], start=start, end=end)
            
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')
                raise ValueError

            scene_objects.append(voltage_source)

    cmdname = '#hertzian_dipole'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()  
            if len(tmp) < 5:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
                raise ValueError
            elif len(tmp) == 5:
                hertzian_dipole = HertzianDipole(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4])
            elif len(tmp) == 7:
                hertzian_dipole = HertzianDipole(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4], start=float(tmp[5]), end=float(tmp[6]))   
            
            elif len(tmp) == 9:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[2]), float(tmp[3])), 
                      Rand_Create(distr, float(tmp[4]), float(tmp[5])), 
                      Rand_Create(distr, float(tmp[6]), float(tmp[7])))
                      
                rand_params.extend([p1])
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                hertzian_dipole = HertzianDipole(polarisation = tmp[1].lower(), p1=p1, waveform_id=tmp[8])
            
            elif len(tmp) == 13:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[2]), float(tmp[3])), 
                      Rand_Create(distr, float(tmp[4]), float(tmp[5])), 
                      Rand_Create(distr, float(tmp[6]), float(tmp[7])))
                start = Rand_Create(distr, float(tmp[9]), float(tmp[10]))
                end = Rand_Create(distr, float(tmp[11]), float(tmp[12]))

                rand_params.extend((p1, start, end))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')
                
                hertzian_dipole = HertzianDipole(polarisation = tmp[1].lower(), p1=p1, waveform_id = tmp[8], start=start, end=end)
            
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')
                raise ValueError

            scene_objects.append(hertzian_dipole)

    cmdname = '#magnetic_dipole'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least five parameters')
                raise ValueError
            elif len(tmp) == 5:
                magnetic_dipole = MagneticDipole(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4])
            elif len(tmp) == 7:
                magnetic_dipole = MagneticDipole(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), waveform_id=tmp[4], start=float(tmp[5]), end=float(tmp[6]))
            
            elif len(tmp) == 9:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[2]), float(tmp[3])), 
                      Rand_Create(distr, float(tmp[4]), float(tmp[5])), 
                      Rand_Create(distr, float(tmp[6]), float(tmp[7])))

                rand_params.extend([p1])
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                magnetic_dipole = MagneticDipole(polarisation = tmp[1].lower(), p1=p1, waveform_id=tmp[8])

            elif len(tmp) == 13:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[2]), float(tmp[3])), 
                      Rand_Create(distr, float(tmp[4]), float(tmp[5])), 
                      Rand_Create(distr, float(tmp[6]), float(tmp[7])))
                start = Rand_Create(distr, float(tmp[9]), float(tmp[10]))
                end = Rand_Create(distr, float(tmp[11]), float(tmp[12]))

                rand_params.extend((p1, start, end))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                magnetic_dipole = MagneticDipole(polarisation = tmp[1].lower(), p1=p1, waveform_id=tmp[8], start=start, end=end)
            
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')
                raise ValueError

            scene_objects.append(magnetic_dipole)

    cmdname = '#transmission_line'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least six parameters')
                raise ValueError
            elif len(tmp) == 6:
                tl = TransmissionLine(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), resistance=float(tmp[4]), waveform_id=tmp[5])
            elif len(tmp) == 8:
                tl = TransmissionLine(polarisation=tmp[0].lower(), p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])), resistance=float(tmp[4]), waveform_id=tmp[5], start=tmp[6], end=tmp[7])
            
            elif len(tmp) == 11:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[2]), float(tmp[3])), 
                      Rand_Create(distr, float(tmp[4]), float(tmp[5])), 
                      Rand_Create(distr, float(tmp[6]), float(tmp[7]))), 
                resistance = Rand_Create(distr, float(tmp[8]), float(tmp[9]))

                rand_params.extend((p1, resistance))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                tl = TransmissionLine(polarisation=tmp[1].lower(), p1=p1, resistance=resistance, waveform_id=tmp[10])

            elif len(tmp) == 15:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[2]), float(tmp[3])), 
                      Rand_Create(distr, float(tmp[4]), float(tmp[5])), 
                      Rand_Create(distr, float(tmp[6]), float(tmp[7]))), 
                resistance = Rand_Create(distr, float(tmp[8]), float(tmp[9]))
                start = Rand_Create(distr, float(tmp[11]), float(tmp[12])), 
                end = Rand_Create(distr, float(tmp[13]), float(tmp[14]))

                rand_params.extend((p1, resistance, start, end))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                tl = TransmissionLine(polarisation = tmp[1].lower(), p1=p1, resistance=resistance, waveform_id=tmp[10], start=start, end=end)

            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')
                raise ValueError

            scene_objects.append(tl)

    cmdname = '#rx'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            try:
                tmp[0] = float(tmp[0])
            except ValueError:
                tmp[0] = str(tmp[0])

            if isinstance(tmp[0], (int, float)):
                if len(tmp) != 3 and len(tmp) < 5:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')
                    raise ValueError
                if len(tmp) == 3:
                    rx = Rx(p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])))
                else:
                    rx = Rx(p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])), id=tmp[3], outputs=' '.join(tmp[4:]))
            
            elif isinstance(tmp[0], str):
                rand_params = []
                if len(tmp) != 7 and len(tmp) < 9:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect number of parameters')
                    raise ValueError
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[1]), float(tmp[2])), 
                      Rand_Create(distr, float(tmp[3]), float(tmp[4])), 
                      Rand_Create(distr, float(tmp[5]), float(tmp[6])))
                      
                rand_params.extend([p1])
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                if len(tmp) == 7:
                    rx = Rx(p1=p1)
                else:
                    rx = Rx(p1=p1, outputs=' '.join(tmp[7:]))

            scene_objects.append(rx)

    cmdname = '#rx_array'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) == 9:
                p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
                p2 = (float(tmp[3]), float(tmp[4]), float(tmp[5]))
                dl = (float(tmp[6]), float(tmp[7]), float(tmp[8]))
                
                rx_array = RxArray(p1=p1, p2=p2, dl=dl)

            elif len(tmp) == 19:
                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[1]), float(tmp[2])), 
                      Rand_Create(distr, float(tmp[3]), float(tmp[4])), 
                      Rand_Create(distr, float(tmp[5]), float(tmp[6])))
                p2 = (Rand_Create(distr, float(tmp[7]), float(tmp[8])),
                      Rand_Create(distr, float(tmp[9]), float(tmp[10])), 
                      Rand_Create(distr, float(tmp[11]), float(tmp[12])))
                dl = (Rand_Create(distr, float(tmp[13]), float(tmp[14])),
                      Rand_Create(distr, float(tmp[15]), float(tmp[16])), 
                      Rand_Create(distr, float(tmp[17]), float(tmp[18])))

                rand_params.extend((p1, p2, dl))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                rx_array = RxArray(p1=p1, p2=p2, dl=dl)
            
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly nine parameters (if single values entered) or nineteen parameters (if range of values entered)')
                raise ValueError

            scene_objects.append(rx_array)

    cmdname = '#snapshot'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) <= 11:
                if len(tmp) != 11:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly eleven parameters (if single values entered)')
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
            
            elif len(tmp) > 11:
                if len(tmp) != 21:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly twenty one parameters (if range of values entered)')
                    raise ValueError

                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[1]), float(tmp[2])), 
                      Rand_Create(distr, float(tmp[3]), float(tmp[4])), 
                      Rand_Create(distr, float(tmp[5]), float(tmp[6])))
                p2 = (Rand_Create(distr, float(tmp[7]), float(tmp[8])),
                      Rand_Create(distr, float(tmp[9]), float(tmp[10])), 
                      Rand_Create(distr, float(tmp[11]), float(tmp[12])))
                dl = (Rand_Create(distr, float(tmp[13]), float(tmp[14])),
                      Rand_Create(distr, float(tmp[15]), float(tmp[16])), 
                      Rand_Create(distr, float(tmp[17]), float(tmp[18])))
                filename = tmp[20]

                rand_params.extend((p1, p2, dl))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                try:
                    iterations = int(tmp[19])
                    snapshot = Snapshot(p1=p1, p2=p2, dl=dl, iterations=iterations, filename=filename)

                except ValueError:
                    time = float(tmp[19])
                    snapshot = Snapshot(p1=p1, p2=p2, dl=dl, time=time, filename=filename)

            scene_objects.append(snapshot)

    cmdname = '#material'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) == 5:
                material = Material(er=float(tmp[0]), se=float(tmp[1]), mr=float(tmp[2]), sm=float(tmp[3]), id=tmp[4])
            elif len(tmp) == 10:
                rand_params = []
                distr = tmp[0].lower()
                er = Rand_Create(distr, float(tmp[1]), float(tmp[2])), 
                se = Rand_Create(distr, float(tmp[3]), float(tmp[4])), 
                mr = Rand_Create(distr, float(tmp[5]), float(tmp[6])), 
                sm = Rand_Create(distr, float(tmp[7]), float(tmp[8]))

                rand_params.extend((er, se, mr, sm))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                material = Material(er=er, se=se, mr=mr, sm=sm, id = tmp[9])
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly five parameters (if single values entered) or ten parameters (if range of values entered)')
                raise ValueError

            scene_objects.append(material)

    cmdname = '#add_dispersion_debye'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            
            try:
                tmp[0] = float(tmp[0])
            except ValueError:
                tmp[0] = str(tmp[0])

            if isinstance(tmp[0], (int, float)):
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
            
            elif isinstance(tmp[0], str):
                if len(tmp) < 7:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least seven parameters')
                    raise ValueError

                rand_params = []
                distr = tmp[0].lower()
                poles = int(tmp[1])
                er_delta = []
                tau = []
                material_ids = tmp[(4 * poles) + 2:len(tmp)]

                for pole in range(2, 4 * poles, 4):
                    er_delta.append(Rand_Create(distr, float(tmp[pole]), float(tmp[pole+1])))
                    tau.append(Rand_Create(distr, float(tmp[pole+2]), float(tmp[pole+3])))
                
                rand_params.extend((er_delta, tau))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                debye_dispersion = AddDebyeDispersion(poles=poles, er_delta=er_delta, tau=tau, material_ids=material_ids)

            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect parameter')
                raise ValueError

            scene_objects.append(debye_dispersion)

    cmdname = '#add_dispersion_lorentz'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            try:
                tmp[0] = float(tmp[0])
            except ValueError:
                tmp[0] = str(tmp[0])

            if isinstance(tmp[0], (int, float)):
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
            
            elif isinstance(tmp[0], str):
                if len(tmp) < 9:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least nine parameters')
                    raise ValueError

                rand_params = []
                distr = tmp[0].lower()
                poles = int(tmp[1])
                material_ids = tmp[(6 * poles) + 2:len(tmp)]
                er_delta = []
                tau = []
                alpha = []

                for pole in range(2, 6 * poles, 6):
                    er_delta.append(Rand_Create(distr, float(tmp[pole]), float(tmp[pole+1])))
                    tau.append(Rand_Create(distr, float(tmp[pole+2]), float(tmp[pole+3])))
                    alpha.append(Rand_Create(distr, float(tmp[pole+4]), float(tmp[pole+5])))

                rand_params.extend((er_delta, tau, alpha))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                lorentz_dispersion = AddLorentzDispersion(poles=poles, material_ids=material_ids, er_delta=er_delta, tau=tau, alpha=alpha)

            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect parameter')
                raise ValueError
                
            scene_objects.append(lorentz_dispersion)

    cmdname = '#add_dispersion_drude'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            try:
                tmp[0] = float(tmp[0])
            except ValueError:
                tmp[0] = str(tmp[0])

            if isinstance(tmp[0], (int, float)):
                if len(tmp) < 4:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least four parameters')
                    raise ValueError

                poles = int(tmp[0])
                material_ids = tmp[(2 * poles) + 1:len(tmp)]
                tau = []
                alpha = []

                for pole in range(1, 2 * poles, 2):
                    tau.append(float(tmp[pole]))
                    alpha.append(float(tmp[pole + 1]))

                drude_dispersion = AddDrudeDispersion(poles=poles, material_ids=material_ids, tau=tau, alpha=alpha)
            
            elif isinstance(tmp[0], str):
                if len(tmp) < 7:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at least seven parameters')
                    raise ValueError

                rand_params = []
                distr = tmp[0].lower()
                poles = int(tmp[1])
                material_ids = tmp[(4 * poles) + 2:len(tmp)]
                tau = []
                alpha = []

                for pole in range(2, 4 * poles, 4):
                    tau.append(Rand_Create(distr, float(tmp[pole]), float(tmp[pole+1])))
                    alpha.append(Rand_Create(distr, float(tmp[pole+2]), float(tmp[pole+3])))

                rand_params.extend((tau, alpha))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                drude_dispersion = AddDrudeDispersion(poles=poles, material_ids=material_ids, tau=tau, alpha=alpha)  

            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' has an incorrect parameter')
                raise ValueError

            scene_objects.append(drude_dispersion)

    cmdname = '#soil_peplinski'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) == 7:
                soil = SoilPeplinski(sand_fraction = float(tmp[0]),
                                     clay_fraction = float(tmp[1]),
                                     bulk_density = float(tmp[2]),
                                     sand_density = float(tmp[3]),
                                     water_fraction_lower = float(tmp[4]),
                                     water_fraction_upper = float(tmp[5]),
                                     id = tmp[6])
            elif len(tmp) == 14:
                rand_params = []
                distr = tmp[0].lower()
                sand_fraction = Rand_Create(distr, float(tmp[1]), float(tmp[2])),
                clay_fraction = Rand_Create(distr, float(tmp[3]), float(tmp[4])),
                bulk_density = Rand_Create(distr, float(tmp[5]), float(tmp[6])),
                sand_density = Rand_Create(distr, float(tmp[7]), float(tmp[8])),
                water_fraction_lower = Rand_Create(distr, float(tmp[9]), float(tmp[10])),
                water_fraction_upper = Rand_Create(distr, float(tmp[11]), float(tmp[12]))

                rand_params.extend((sand_fraction, clay_fraction, bulk_density, sand_density, water_fraction_lower, water_fraction_upper))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                soil = SoilPeplinski(sand_fraction=sand_fraction, clay_fraction=clay_fraction, bulk_density=bulk_density, sand_density=sand_density, 
                                     water_fraction_lower=water_fraction_lower, water_fraction_upper=water_fraction_upper, id = tmp[13])
            else:
                logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires at exactly seven parameters (if single values entered) or fourteen parameters (if range of values entered)')
                raise ValueError
                
            scene_objects.append(soil)

    cmdname = '#geometry_view'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) <= 11:
                if len(tmp) != 11:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly eleven parameters')
                    raise ValueError

                p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
                p2 = (float(tmp[3]), float(tmp[4]), float(tmp[5]))
                dl = (float(tmp[6]), float(tmp[7]), float(tmp[8]))

                geometry_view = GeometryView(p1=p1, p2=p2, dl=dl, filename=tmp[9], output_type=tmp[10])

            elif len(tmp) > 11:
                if len(tmp) != 21:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly twenty one parameters')
                    raise ValueError

                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[1]), float(tmp[2])), 
                      Rand_Create(distr, float(tmp[3]), float(tmp[4])), 
                      Rand_Create(distr, float(tmp[5]), float(tmp[6])))
                p2 = (Rand_Create(distr, float(tmp[7]), float(tmp[8])),
                      Rand_Create(distr, float(tmp[9]), float(tmp[10])), 
                      Rand_Create(distr, float(tmp[11]), float(tmp[12])))
                dl = (Rand_Create(distr, float(tmp[13]), float(tmp[14])),
                      Rand_Create(distr, float(tmp[15]), float(tmp[16])), 
                      Rand_Create(distr, float(tmp[17]), float(tmp[18])))

                rand_params.extend((p1, p2, dl))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                geometry_view = GeometryView(p1=p1, p2=p2, dl=dl, filename=tmp[19], output_type=tmp[20])  

            scene_objects.append(geometry_view)

    cmdname = '#geometry_objects_write'
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) <= 7:
                if len(tmp) != 7:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly seven parameters')
                    raise ValueError

                p1 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
                p2 = (float(tmp[3]), float(tmp[4]), float(tmp[5]))
                gow = GeometryObjectsWrite(p1=p1, p2=p2, filename=tmp[6])

            elif len(tmp) > 7:
                if len(tmp) != 14:
                    logger.exception("'" + cmdname + ': ' + ' '.join(tmp) + "'" + ' requires exactly fourteen parameters')
                    raise ValueError

                rand_params = []
                distr = tmp[0].lower()
                p1 = (Rand_Create(distr, float(tmp[1]), float(tmp[2])), 
                      Rand_Create(distr, float(tmp[3]), float(tmp[4])), 
                      Rand_Create(distr, float(tmp[5]), float(tmp[6])))
                p2 = (Rand_Create(distr, float(tmp[7]), float(tmp[8])),
                      Rand_Create(distr, float(tmp[9]), float(tmp[10])), 
                      Rand_Create(distr, float(tmp[11]), float(tmp[12])))

                rand_params.extend((p1, p2))
                hash_count_multiplecmds[cmdname] += 1
                Save_Params(rand_params, str(config.get_model_config().output_file_path_ext_random) + '_' + cmdname + str(hash_count_multiplecmds[cmdname]) + '.csv')

                gow = GeometryObjectsWrite(p1=p1, p2=p2, filename=tmp[13])
                
            scene_objects.append(gow)

    return scene_objects
