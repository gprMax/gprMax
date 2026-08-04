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

import logging

import gprMax.config as config

from .user_objects.cmds_multiuse import (
    PMLCFS,
    AddDebyeDispersion,
    AddDrudeDispersion,
    AddLorentzDispersion,
    DiscretePlaneWaveAngles,
    DiscretePlaneWaveAxial,
    DiscretePlaneWaveVector,
    EigenmodeRx,
    EigenmodeSource,
    ExcitationFile,
    HertzianDipole,
    MagneticDipole,
    Material,
    MaterialList,
    MaterialRange,
    Rx,
    RxArray,
    SoilPeplinski,
    SymmetryBoundary,
    TransmissionLine,
    VoltageSource,
    Waveform,
)
from .user_objects.cmds_output import (
    GeometryObjectsWrite,
    GeometryView,
    KSIRAntennaPorts,
    KSIRFarField,
    KSIRFarFieldArray,
    KSIRFrequencyRx,
    KSIRFrequencyRxArray,
    KSIRFrequencyRxSpherical,
    KSIRFrequencyTransform,
    KSIRTimeRx,
    KSIRTimeRxArray,
    KSIRTimeRxSpherical,
    NTFFAntennaPorts,
    NTFFFarField,
    NTFFFarFieldArray,
    NTFFFrequencyTransform,
    NTFFSurface,
    NTFFTimeFarField,
    NTFFTimeFarFieldArray,
    RxPort,
    Snapshot,
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
            elif len(tmp) in (8, 9):
                voltage_source = VoltageSource(
                    polarisation=tmp[0].lower(),
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    resistance=float(tmp[4]),
                    waveform_id=tmp[5],
                    start=float(tmp[6]),
                    stop=float(tmp[7]),
                    reference_impedance=float(tmp[8]) if len(tmp) == 9 else None,
                )
            else:
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " requires six, eight, or nine parameters"
                )
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
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    waveform_id=tmp[4],
                )
            elif len(tmp) == 7:
                hertzian_dipole = HertzianDipole(
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    waveform_id=tmp[4],
                    start=float(tmp[5]),
                    stop=float(tmp[6]),
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
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    waveform_id=tmp[4],
                )
            elif len(tmp) == 7:
                magnetic_dipole = MagneticDipole(
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    waveform_id=tmp[4],
                    start=float(tmp[5]),
                    stop=float(tmp[6]),
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
                    stop=tmp[7],
                )
            else:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters")
                raise ValueError

            scene_objects.append(tl)

    cmdname = "#plane_wave_angles"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 10:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least ten parameters")
                raise ValueError

            if len(tmp) == 10:
                plWave = DiscretePlaneWaveAngles(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                    theta=float(tmp[6]),
                    phi=float(tmp[7]),
                    psi=float(tmp[8]),
                    waveform_id=tmp[9],
                )
            elif len(tmp) == 11:
                plWave = DiscretePlaneWaveAngles(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                    theta=float(tmp[6]),
                    phi=float(tmp[7]),
                    psi=float(tmp[8]),
                    waveform_id=tmp[9],
                    material_id=tmp[10],
                )
            elif len(tmp) == 13:
                plWave = DiscretePlaneWaveAngles(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                    theta=float(tmp[6]),
                    phi=float(tmp[7]),
                    psi=float(tmp[8]),
                    waveform_id=tmp[9],
                    material_id=tmp[10],
                    start=float(tmp[11]),
                    stop=float(tmp[12]),
                )
            else:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters")
                raise ValueError

            scene_objects.append(plWave)

    cmdname = "#plane_wave_vector"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 10:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least ten parameters")
                raise ValueError

            if len(tmp) == 11:
                plWave = DiscretePlaneWaveVector(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                    m_vec=(int(tmp[6]), int(tmp[7]), int(tmp[8])),
                    psi=float(tmp[9]),
                    waveform_id=tmp[10],
                )
            elif len(tmp) == 12:
                plWave = DiscretePlaneWaveVector(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                    m_vec=(int(tmp[6]), int(tmp[7]), int(tmp[8])),
                    psi=float(tmp[9]),
                    waveform_id=tmp[10],
                    material_id=tmp[11],
                )
            elif len(tmp) == 14:
                plWave = DiscretePlaneWaveVector(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                    m_vec=(int(tmp[6]), int(tmp[7]), int(tmp[8])),
                    psi=float(tmp[9]),
                    waveform_id=tmp[10],
                    material_id=tmp[11],
                    start=float(tmp[12]),
                    stop=float(tmp[13]),
                )
            else:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters")
                raise ValueError

            scene_objects.append(plWave)

    cmdname = "#plane_wave_axial"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 9:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least nine parameters")
                raise ValueError

            if len(tmp) == 9:
                plWave = DiscretePlaneWaveAxial(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                    psi=float(tmp[6]),
                    axis=tmp[7].lower(),
                    waveform_id=tmp[8],
                )
            elif len(tmp) == 11:
                plWave = DiscretePlaneWaveAxial(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                    psi=float(tmp[6]),
                    axis=tmp[7].lower(),
                    waveform_id=tmp[8],
                    start=float(tmp[9]),
                    stop=float(tmp[10]),
                )
            else:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters")
                raise ValueError

            scene_objects.append(plWave)

    eigenmode_source_cmds = multicmds["#eigenmode_source"] or []
    eigenmode_rx_cmds = multicmds["#eigenmode_rx"] or []
    if (eigenmode_source_cmds or eigenmode_rx_cmds) and len(eigenmode_source_cmds) != 1:
        raise ValueError(
            "Eigenmode ports require one and only one #eigenmode_source command; "
            f"found {len(eigenmode_source_cmds)}."
        )

    cmdname = "#eigenmode_source"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 14:
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires at least fourteen parameters: x0 y0 z0 x1 y1 z1 "
                    "direction excitation_mode[,mode_count] port_index "
                    "frequency [frequency ...] waveform_id "
                    "dft_start dft_stop dft_points"
                )
                raise ValueError

            p0 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
            p1 = (float(tmp[3]), float(tmp[4]), float(tmp[5]))
            mode = config.get_model_config().mode
            invariant_axis = "xyz".index(mode[-1]) if mode.startswith("2D") else None
            equal_axes = [axis for axis in range(3) if axis != invariant_axis and p0[axis] == p1[axis]]
            if len(equal_axes) != 1:
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " must have exactly one finite matching coordinate pair "
                    "for the source normal"
                )
                raise ValueError

            axis_names = ("x", "y", "z")
            normal_axis = equal_axes[0]
            transverse_axes = [axis for axis in range(3) if axis != normal_axis]
            transverse_p0 = [p0[axis] for axis in transverse_axes]
            transverse_p1 = [p1[axis] for axis in transverse_axes]
            transverse_lower = tuple(min(a, b) for a, b in zip(transverse_p0, transverse_p1))
            transverse_upper = tuple(max(a, b) for a, b in zip(transverse_p0, transverse_p1))

            try:
                mode_values = tuple(int(value) for value in tmp[7].split(","))
            except ValueError as exc:
                raise ValueError(
                    f"{cmdname} mode specification must be excitation_mode or " "excitation_mode,mode_count."
                ) from exc
            if len(mode_values) not in (1, 2):
                raise ValueError(f"{cmdname} mode specification must contain one or two integers.")
            excitation_mode = mode_values[0]
            mode_count = mode_values[-1]
            if excitation_mode < 1:
                raise ValueError(f"{cmdname} excitation_mode must be one or greater.")
            if mode_count < excitation_mode:
                raise ValueError(
                    f"{cmdname} mode_count must be at least excitation_mode " f"({excitation_mode}); got {mode_count}."
                )
            try:
                port_index = int(tmp[8])
            except ValueError as exc:
                raise ValueError(f"{cmdname} port_index must be an integer.") from exc
            if port_index < 1:
                raise ValueError(f"{cmdname} port_index must be one or greater.")

            frequency_tokens = []
            parameter_index = 9
            while parameter_index < len(tmp):
                try:
                    float(tmp[parameter_index])
                except ValueError:
                    break
                frequency_tokens.append(tmp[parameter_index])
                parameter_index += 1

            if not frequency_tokens or parameter_index >= len(tmp):
                raise ValueError(f"{cmdname} requires one or more frequencies followed by a waveform identifier.")

            frequencies = tuple(float(value) for value in frequency_tokens)
            waveform_id = tmp[parameter_index]
            tail = tmp[parameter_index + 1 :]
            if len(tail) not in (3, 4) or (len(tail) == 4 and tail[3].lower() not in ("y", "n")):
                raise ValueError(
                    f"{cmdname} requires dft_start dft_stop dft_points and accepts "
                    "an optional final y or n field-plot parameter."
                )
            dft_start, dft_stop = float(tail[0]), float(tail[1])
            dft_points = int(tail[2])
            plot_fields = None if len(tail) == 3 else tail[3].lower() == "y"
            kwargs = {
                "normal": axis_names[normal_axis],
                "direction": tmp[6],
                "p1": transverse_lower,
                "p2": transverse_upper,
                "w": p0[normal_axis],
                "mode_index": excitation_mode,
                "mode_count": mode_count,
                "port_index": port_index,
                "waveform_id": waveform_id,
                "dft_start": dft_start,
                "dft_stop": dft_stop,
                "dft_points": dft_points,
                "plot_fields": plot_fields,
            }
            if len(frequencies) == 1:
                kwargs["frequency"] = frequencies[0]
            else:
                kwargs["frequencies"] = frequencies
            eigenmode_source = EigenmodeSource(**kwargs)
            scene_objects.append(eigenmode_source)

    cmdname = "#eigenmode_rx"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 14:
                raise ValueError(
                    f"{cmdname} requires x0 y0 z0 x1 y1 z1 direction "
                    "mode_count port_index frequency [frequency ...] id "
                    "dft_start dft_stop dft_points [y|n]."
                )
            p0 = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
            p1 = (float(tmp[3]), float(tmp[4]), float(tmp[5]))
            mode = config.get_model_config().mode
            invariant_axis = "xyz".index(mode[-1]) if mode.startswith("2D") else None
            equal_axes = [axis for axis in range(3) if axis != invariant_axis and p0[axis] == p1[axis]]
            if len(equal_axes) != 1:
                raise ValueError(
                    f"{cmdname} must have exactly one finite matching coordinate pair " "for the receiver normal."
                )
            normal_axis = equal_axes[0]
            transverse_axes = [axis for axis in range(3) if axis != normal_axis]
            transverse_p0 = [p0[axis] for axis in transverse_axes]
            transverse_p1 = [p1[axis] for axis in transverse_axes]
            transverse_lower = tuple(min(a, b) for a, b in zip(transverse_p0, transverse_p1))
            transverse_upper = tuple(max(a, b) for a, b in zip(transverse_p0, transverse_p1))
            try:
                mode_count = int(tmp[7])
                port_index = int(tmp[8])
            except ValueError as exc:
                raise ValueError(f"{cmdname} mode_count and port_index must be integers.") from exc
            if mode_count < 1:
                raise ValueError(f"{cmdname} mode_count must be one or greater.")
            if port_index < 1:
                raise ValueError(f"{cmdname} port_index must be one or greater.")

            frequency_tokens = []
            parameter_index = 9
            while parameter_index < len(tmp):
                try:
                    float(tmp[parameter_index])
                except ValueError:
                    break
                frequency_tokens.append(tmp[parameter_index])
                parameter_index += 1
            if not frequency_tokens or parameter_index >= len(tmp):
                raise ValueError(f"{cmdname} requires modal frequencies followed by a receiver ID.")
            frequencies = tuple(float(value) for value in frequency_tokens)
            port_id = tmp[parameter_index]
            tail = tmp[parameter_index + 1 :]
            if len(tail) not in (3, 4) or (len(tail) == 4 and tail[3].lower() not in ("y", "n")):
                raise ValueError(
                    f"{cmdname} requires dft_start dft_stop dft_points and accepts "
                    "an optional final y or n field-plot parameter."
                )
            kwargs = {
                "normal": "xyz"[normal_axis],
                "direction": tmp[6],
                "p1": transverse_lower,
                "p2": transverse_upper,
                "w": p0[normal_axis],
                "mode_count": mode_count,
                "port_index": port_index,
                "id": port_id,
                "dft_start": float(tail[0]),
                "dft_stop": float(tail[1]),
                "dft_points": int(tail[2]),
                "plot_fields": None if len(tail) == 3 else tail[3].lower() == "y",
            }
            if len(frequencies) == 1:
                kwargs["frequency"] = frequencies[0]
            else:
                kwargs["frequencies"] = frequencies
            scene_objects.append(EigenmodeRx(**kwargs))

    cmdname = "#excitation_file"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) not in [1, 3]:
                logger.exception(f"{cmdname} requires either one or three parameter(s)")
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
                rx = Rx(
                    p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                    id=tmp[3],
                    outputs=tmp[4:],
                )

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

    cmdname = "#rx_port"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 3 or len(tokens) > 5:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires three coordinates, " "an optional ID and spectrum limit"
                )
            kwargs = {}
            if len(tokens) >= 4:
                kwargs["id"] = tokens[3]
            if len(tokens) >= 5:
                try:
                    kwargs["spectrum_limit"] = float(tokens[4])
                except ValueError:
                    kwargs["spectrum_limit"] = tokens[4].lower()
            scene_objects.append(
                RxPort(
                    p1=tuple(float(value) for value in tokens[:3]),
                    **kwargs,
                )
            )

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
            if "." in filename:
                fileext = "." + filename.split(".")[-1]
            else:
                fileext = None

            try:
                iterations = int(tmp[9])
                snapshot = Snapshot(p1=p1, p2=p2, dl=dl, iterations=iterations, filename=filename, fileext=fileext)

            except ValueError:
                time = float(tmp[9])
                snapshot = Snapshot(p1=p1, p2=p2, dl=dl, time=time, filename=filename, fileext=fileext)

            scene_objects.append(snapshot)

    cmdname = "#ntff_surface"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 7 or len(tokens) > 12:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires six coordinates, a surface "
                    "ID, and optionally one to five omitted Huygens faces"
                )
            scene_objects.append(
                NTFFSurface(
                    p1=tuple(float(value) for value in tokens[:3]),
                    p2=tuple(float(value) for value in tokens[3:6]),
                    id=tokens[6],
                    omit_faces=tuple(tokens[7:]),
                )
            )

    cmdname = "#ksir_frequency"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 3:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires a surface ID, transform ID, and one or more frequencies"
                )
            window = "rectangular"
            if tokens[-1].lower() in ("rectangular", "hann"):
                window = tokens.pop().lower()
            if len(tokens) < 3:
                raise ValueError(f"{cmdname} requires at least one frequency")
            scene_objects.append(
                KSIRFrequencyTransform(
                    surface_id=tokens[0],
                    id=tokens[1],
                    frequencies=tuple(float(value) for value in tokens[2:]),
                    window=window,
                )
            )

    cmdname = "#ntff_frequency"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 3:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires a surface ID, transform ID, and one or more frequencies"
                )
            window = "rectangular"
            if tokens[-1].lower() in ("rectangular", "hann"):
                window = tokens.pop().lower()
            if len(tokens) < 3:
                raise ValueError(f"{cmdname} requires at least one frequency")
            scene_objects.append(
                NTFFFrequencyTransform(
                    surface_id=tokens[0],
                    id=tokens[1],
                    frequencies=tuple(float(value) for value in tokens[2:]),
                    window=window,
                )
            )

    cmdname = "#ksir_antenna_ports"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 2:
                raise ValueError(f"'{cmdname}: {cmdinstance}' requires a transform ID and one or more port IDs")
            scene_objects.append(
                KSIRAntennaPorts(
                    transform_id=tokens[0],
                    port_ids=tuple(tokens[1:]),
                )
            )

    cmdname = "#ntff_antenna_ports"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 2:
                raise ValueError(f"'{cmdname}: {cmdinstance}' requires a transform ID and one or more port IDs")
            scene_objects.append(
                NTFFAntennaPorts(
                    transform_id=tokens[0],
                    port_ids=tuple(tokens[1:]),
                )
            )

    def parse_point_options(cmdname, cmdinstance, tokens, required, has_time=False):
        if len(tokens) < required:
            raise ValueError(f"'{cmdname}: {cmdinstance}' requires at least {required} parameters")
        extras = tokens[required:]
        time_origin = "simulation"
        if has_time and extras and extras[-1] in ("simulation", "first_arrival"):
            if len(extras) < 2:
                raise ValueError(f"{cmdname} requires the optional output ID before time_origin")
            time_origin = extras.pop()
        output_id = extras[0] if extras else None
        outputs = tuple(extras[1:]) if len(extras) > 1 else None
        kwargs = dict(id=output_id, outputs=outputs)
        if has_time:
            kwargs["time_origin"] = time_origin
        return kwargs

    cmdname = "#ksir_time_rx"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 4, True)
            scene_objects.append(
                KSIRTimeRx(
                    position=tuple(float(value) for value in tokens[:3]),
                    surface_id=tokens[3],
                    **kwargs,
                )
            )

    cmdname = "#ksir_time_rx_spherical"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 4, True)
            scene_objects.append(
                KSIRTimeRxSpherical(
                    float(tokens[0]),
                    float(tokens[1]),
                    float(tokens[2]),
                    tokens[3],
                    **kwargs,
                )
            )

    cmdname = "#ksir_time_rx_array"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 10, True)
            scene_objects.append(
                KSIRTimeRxArray(
                    tuple(float(value) for value in tokens[:3]),
                    tuple(float(value) for value in tokens[3:6]),
                    tuple(float(value) for value in tokens[6:9]),
                    tokens[9],
                    **kwargs,
                )
            )

    cmdname = "#ksir_frequency_rx"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 4)
            scene_objects.append(
                KSIRFrequencyRx(
                    position=tuple(float(value) for value in tokens[:3]),
                    transform_id=tokens[3],
                    **kwargs,
                )
            )

    cmdname = "#ksir_frequency_rx_spherical"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 4)
            scene_objects.append(
                KSIRFrequencyRxSpherical(
                    float(tokens[0]),
                    float(tokens[1]),
                    float(tokens[2]),
                    tokens[3],
                    **kwargs,
                )
            )

    cmdname = "#ksir_frequency_rx_array"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 10)
            scene_objects.append(
                KSIRFrequencyRxArray(
                    tuple(float(value) for value in tokens[:3]),
                    tuple(float(value) for value in tokens[3:6]),
                    tuple(float(value) for value in tokens[6:9]),
                    tokens[9],
                    **kwargs,
                )
            )

    cmdname = "#ksir_far_field"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 3)
            scene_objects.append(KSIRFarField(float(tokens[0]), float(tokens[1]), tokens[2], **kwargs))

    cmdname = "#ksir_far_field_array"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 7)
            scene_objects.append(
                KSIRFarFieldArray(
                    *(float(value) for value in tokens[:6]),
                    transform_id=tokens[6],
                    **kwargs,
                )
            )

    cmdname = "#ntff_far_field"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 3)
            scene_objects.append(NTFFFarField(float(tokens[0]), float(tokens[1]), tokens[2], **kwargs))

    cmdname = "#ntff_far_field_array"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 7)
            scene_objects.append(
                NTFFFarFieldArray(
                    *(float(value) for value in tokens[:6]),
                    transform_id=tokens[6],
                    **kwargs,
                )
            )

    cmdname = "#ntff_time_far_field"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 3)
            scene_objects.append(NTFFTimeFarField(float(tokens[0]), float(tokens[1]), tokens[2], **kwargs))

    cmdname = "#ntff_time_far_field_array"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            kwargs = parse_point_options(cmdname, cmdinstance, tokens, 7)
            scene_objects.append(
                NTFFTimeFarFieldArray(
                    *(float(value) for value in tokens[:6]),
                    surface_id=tokens[6],
                    **kwargs,
                )
            )

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
            omega = []
            delta = []

            for pole in range(1, 3 * poles, 3):
                er_delta.append(float(tmp[pole]))
                omega.append(float(tmp[pole + 1]))
                delta.append(float(tmp[pole + 2]))

            lorentz_dispersion = AddLorentzDispersion(
                poles=poles,
                material_ids=material_ids,
                er_delta=er_delta,
                omega=omega,
                delta=delta,
            )
            scene_objects.append(lorentz_dispersion)

    cmdname = "#add_dispersion_drude"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) < 4:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least four parameters")
                raise ValueError

            poles = int(tmp[0])
            material_ids = tmp[(2 * poles) + 1 : len(tmp)]
            omega = []
            alpha = []

            for pole in range(1, 2 * poles, 2):
                omega.append(float(tmp[pole]))
                alpha.append(float(tmp[pole + 1]))

            drude_dispersion = AddDrudeDispersion(poles=poles, material_ids=material_ids, omega=omega, alpha=alpha)
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

    cmdname = "#symmetry_boundary"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 2:
                logger.exception("'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly two parameters")
                raise ValueError

            symmetry_boundary = SymmetryBoundary(face=tmp[0].lower(), type=tmp[1].lower())
            scene_objects.append(symmetry_boundary)

    return scene_objects
