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

import numpy as np

import gprMax.config as config

from .user_objects.cmds_multiuse import (
    PMLCFS,
    AddDebyeDispersion,
    AddDrudeDispersion,
    AddLorentzDispersion,
    DiscretePlaneWaveAngles,
    DiscretePlaneWaveAxial,
    DiscretePlaneWaveVector,
    EigenmodeBand,
    EigenmodeExcitation,
    EigenmodePort,
    ExcitationFile,
    HertzianDipole,
    MagneticDipole,
    MagneticFrillSource,
    Material,
    MaterialDensity,
    MaterialFromDatabase,
    MaterialList,
    MaterialRange,
    NetworkExcitation,
    NetworkTerminal,
    PMLSlab,
    RationalNetwork,
    Rx,
    RxArray,
    SoilPeplinski,
    SymmetryBoundary,
    TransmissionLine,
    VirtualWaveguide,
    VoltageSource,
    Waveform,
)
from .user_objects.cmds_output import (
    SAR,
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
    NetworkPort,
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
from .user_objects.cmds_singleuse import PMLFormulation

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
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly four parameters"
                )
                raise ValueError

            waveform = Waveform(wave_type=tmp[0], amp=float(tmp[1]), freq=float(tmp[2]), id=tmp[3])
            scene_objects.append(waveform)

    cmdname = "#rational_network"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 4:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires an ID, G, C, pole count, "
                    "and four values per pole"
                )
            pole_count = int(tokens[3])
            if pole_count < 0:
                raise ValueError(f"{cmdname} pole count must be non-negative")
            required = 4 + 4 * pole_count
            if len(tokens) not in (required, required + 1):
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires {required} parameters, "
                    "or one additional y/n allow-active flag"
                )
            allow_active = False
            if len(tokens) == required + 1:
                if tokens[-1].lower() not in ("y", "n"):
                    raise ValueError(f"{cmdname} allow-active flag must be 'y' or 'n'")
                allow_active = tokens[-1].lower() == "y"
            poles = []
            residues = []
            for index in range(pole_count):
                offset = 4 + 4 * index
                poles.append(complex(float(tokens[offset]), float(tokens[offset + 1])))
                residues.append(complex(float(tokens[offset + 2]), float(tokens[offset + 3])))
            scene_objects.append(
                RationalNetwork(
                    id=tokens[0],
                    conductance=float(tokens[1]),
                    capacitance=float(tokens[2]),
                    poles=tuple(poles),
                    residues=tuple(residues),
                    allow_active=allow_active,
                )
            )

    cmdname = "#network_terminal"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) != 6:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires polarisation, three "
                    "coordinates, network ID, and terminal ID"
                )
            scene_objects.append(
                NetworkTerminal(
                    polarisation=tokens[0].lower(),
                    p1=tuple(float(value) for value in tokens[1:4]),
                    network_id=tokens[4],
                    id=tokens[5],
                )
            )

    cmdname = "#network_excitation"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) not in (2, 4):
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires terminal ID and waveform "
                    "ID, optionally followed by start and stop times"
                )
            scene_objects.append(
                NetworkExcitation(
                    terminal_id=tokens[0],
                    waveform_id=tokens[1],
                    start=float(tokens[2]) if len(tokens) == 4 else None,
                    stop=float(tokens[3]) if len(tokens) == 4 else None,
                )
            )

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
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires six, eight, or nine parameters"
                )
                raise ValueError

            scene_objects.append(voltage_source)

    cmdname = "#hertzian_dipole"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires at least five parameters"
                )
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
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters"
                )
                raise ValueError

            scene_objects.append(hertzian_dipole)

    cmdname = "#magnetic_dipole"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 5:
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires at least five parameters"
                )
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
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters"
                )
                raise ValueError

            scene_objects.append(magnetic_dipole)

    cmdname = "#transmission_line"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) < 6:
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least six parameters"
                )
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
                    start=float(tmp[6]),
                    stop=float(tmp[7]),
                )
            else:
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " too many parameters"
                )
                raise ValueError

            scene_objects.append(tl)

    cmdname = "#magnetic_frill_source"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) == 6:
                frill = MagneticFrillSource(
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    zcoax=float(tmp[4]),
                    waveform_id=tmp[5],
                )
            elif len(tmp) == 8:
                frill = MagneticFrillSource(
                    polarisation=tmp[0],
                    p1=(float(tmp[1]), float(tmp[2]), float(tmp[3])),
                    zcoax=float(tmp[4]),
                    waveform_id=tmp[5],
                    start=float(tmp[6]),
                    stop=float(tmp[7]),
                )
            else:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires six parameters, "
                    "or eight parameters when start and stop times are supplied"
                )

            scene_objects.append(frill)

    cmdname = "#plane_wave_angles"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) not in (10, 11, 13):
                raise ValueError(
                    f"'{cmdname}: {' '.join(tmp)}' requires ten parameters, 11 with a "
                    "material ID, or 13 with a material ID and start/stop times"
                )

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
            scene_objects.append(plWave)

    eigenmode_band_cmds = multicmds.get("#eigenmode_band") or []
    eigenmode_port_cmds = multicmds.get("#eigenmode_port") or []
    eigenmode_excitation_cmds = multicmds.get("#eigenmode_excitation") or []
    virtual_waveguide_cmds = multicmds.get("#virtual_waveguide") or []
    if eigenmode_port_cmds and len(eigenmode_band_cmds) != 1:
        raise ValueError(
            "Eigenmode ports require exactly one #eigenmode_band command; "
            f"found {len(eigenmode_band_cmds)}."
        )
    if eigenmode_port_cmds and len(eigenmode_excitation_cmds) > 1:
        raise ValueError(
            "Eigenmode ports allow at most one #eigenmode_excitation command; "
            f"found {len(eigenmode_excitation_cmds)}."
        )
    if eigenmode_excitation_cmds and not eigenmode_port_cmds:
        raise ValueError("#eigenmode_excitation requires at least one #eigenmode_port.")
    if virtual_waveguide_cmds and not eigenmode_port_cmds:
        raise ValueError("#virtual_waveguide requires at least one #eigenmode_port.")
    if eigenmode_port_cmds and not eigenmode_excitation_cmds:
        port_numbers = {int(command.split()[0]) for command in eigenmode_port_cmds}
        virtual_numbers = {int(command.split()[0]) for command in virtual_waveguide_cmds}
        if port_numbers != virtual_numbers:
            raise ValueError(
                "Eigenmode ports require exactly one #eigenmode_excitation, unless "
                "every port has a passive #virtual_waveguide."
            )

    for cmdinstance in eigenmode_band_cmds:
        tmp = cmdinstance.split()
        if len(tmp) != 4:
            raise ValueError("#eigenmode_band requires id fmin fmax points.")
        scene_objects.append(
            EigenmodeBand(
                id=tmp[0],
                fmin=float(tmp[1]),
                fmax=float(tmp[2]),
                points=int(tmp[3]),
            )
        )

    for cmdinstance in eigenmode_port_cmds:
        tmp = cmdinstance.split()
        if len(tmp) < 10:
            raise ValueError(
                "#eigenmode_port requires port x0 y0 z0 x1 y1 z1 direction "
                "modes anchors [anchor ...] [y|n]."
            )
        port = int(tmp[0])
        p1 = tuple(float(value) for value in tmp[1:4])
        p2 = tuple(float(value) for value in tmp[4:7])
        direction = tmp[7]
        try:
            modes = tuple(int(value) for value in tmp[8].split(","))
        except ValueError as exc:
            raise ValueError("#eigenmode_port modes must be comma-separated integers.") from exc
        tail = tmp[9:]
        plot_fields = None
        if tail[-1].lower() in ("y", "n"):
            plot_fields = tail[-1].lower() == "y"
            tail = tail[:-1]
        if tail == ["auto"]:
            anchors = "auto"
        else:
            if not tail or "auto" in (value.lower() for value in tail):
                raise ValueError("#eigenmode_port anchors must be auto or frequencies.")
            anchors = tuple(float(value) for value in tail)
        scene_objects.append(
            EigenmodePort(
                port=port,
                p1=p1,
                p2=p2,
                direction=direction,
                modes=modes,
                anchors=anchors,
                plot_fields=plot_fields,
            )
        )

    for cmdinstance in virtual_waveguide_cmds:
        tmp = cmdinstance.split()
        if len(tmp) not in (1, 2, 3, 4, 5):
            raise ValueError(
                "#virtual_waveguide requires port [length_cells] [pml_cells] "
                "[source_clearance_cells] [pml_profile]."
            )
        kwargs = {"port": int(tmp[0])}
        if len(tmp) >= 2:
            kwargs["length_cells"] = int(tmp[1])
        if len(tmp) >= 3:
            kwargs["pml_cells"] = int(tmp[2])
        if len(tmp) >= 4:
            kwargs["source_clearance_cells"] = int(tmp[3])
        if len(tmp) == 5 and tmp[4].lower() != "none":
            kwargs["pml_profile"] = tmp[4]
        scene_objects.append(VirtualWaveguide(**kwargs))

    for cmdinstance in eigenmode_excitation_cmds:
        tmp = cmdinstance.split()
        plot_waveform = None
        if tmp and tmp[-1].lower() in ("y", "n"):
            plot_waveform = tmp[-1].lower() == "y"
            tmp = tmp[:-1]
        if len(tmp) not in (2, 3, 4):
            raise ValueError(
                "#eigenmode_excitation requires port mode " "[auto|waveform_id] [amplitude] [y|n]."
            )
        kwargs = {"port": int(tmp[0]), "mode": int(tmp[1])}
        if len(tmp) >= 3:
            kwargs["waveform"] = tmp[2]
        if len(tmp) == 4:
            kwargs["amplitude"] = float(tmp[3])
        if plot_waveform is not None:
            kwargs["plot_waveform"] = plot_waveform
        scene_objects.append(EigenmodeExcitation(**kwargs))

    cmdname = "#plane_wave_vector"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) not in (11, 12, 14):
                raise ValueError(
                    f"'{cmdname}: {' '.join(tmp)}' requires 11 parameters, 12 with a "
                    "material ID, or 14 with a material ID and start/stop times"
                )

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
            scene_objects.append(plWave)

    cmdname = "#plane_wave_axial"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) not in (9, 11):
                raise ValueError(
                    f"'{cmdname}: {' '.join(tmp)}' requires nine parameters, or 11 "
                    "with start/stop times"
                )

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
            scene_objects.append(plWave)

    cmdname = "#excitation_file"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) not in [1, 3]:
                logger.exception(f"{cmdname} requires either one or three parameter(s)")
                raise ValueError

            if len(tmp) > 1:
                try:
                    kind = int(tmp[1])
                except ValueError:
                    kind = tmp[1]
                try:
                    fill_value = float(tmp[2])
                except ValueError:
                    fill_value = tmp[2]
                ex_file = ExcitationFile(filepath=tmp[0], kind=kind, fill_value=fill_value)
            else:
                ex_file = ExcitationFile(filepath=tmp[0])

            scene_objects.append(ex_file)

    cmdname = "#rx"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 3 and len(tmp) < 5:
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " has an incorrect number of parameters"
                )
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
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly nine parameters"
                )
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
                    f"'{cmdname}: {cmdinstance}' requires three coordinates, "
                    "an optional ID and spectrum limit"
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

    cmdname = "#network_port"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) not in (1, 2, 3):
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires a terminal ID and "
                    "optionally reference impedance and spectrum limit"
                )
            kwargs = {}
            if len(tokens) >= 2:
                kwargs["reference_impedance"] = float(tokens[1])
            if len(tokens) == 3:
                kwargs["spectrum_limit"] = (
                    "nyquist" if tokens[2].lower() == "nyquist" else float(tokens[2])
                )
            scene_objects.append(NetworkPort(terminal_id=tokens[0], **kwargs))

    cmdname = "#sar"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 8:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires start and stop frequencies, "
                    "number of points, waveform ID, normalisation, spectrum limit, "
                    "output ID, and at least one geometry tag"
                )
            start = float(tokens[0])
            stop = float(tokens[1])
            points = int(tokens[2])
            if points < 1:
                raise ValueError(f"'{cmdname}' number of frequency points must be positive")
            if points == 1 and start != stop:
                raise ValueError(
                    f"'{cmdname}' requires equal start/stop frequencies when points is one"
                )
            frequencies = np.linspace(start, stop, points)
            power_mode = tokens[4].lower() in ("incident_power", "accepted_power")
            if power_mode:
                if len(tokens) < 10:
                    raise ValueError(
                        f"'{cmdname}: {cmdinstance}' power normalisation requires "
                        "target power, port ID, spectrum limit, output ID, and a tag"
                    )
                target_power = float(tokens[5])
                port_id = tokens[6]
                spectrum_token = tokens[7]
                output_id = tokens[8]
                remaining = tokens[9:]
            else:
                target_power = None
                port_id = None
                spectrum_token = tokens[5]
                output_id = tokens[6]
                remaining = tokens[7:]
            averaging_masses = ()
            if remaining[0].lower() == "spatial_average":
                if len(remaining) < 4:
                    raise ValueError(
                        f"'{cmdname}: {cmdinstance}' spatial_average requires a positive "
                        "mass count, the masses in kg, and at least one geometry tag"
                    )
                mass_count = int(remaining[1])
                if mass_count < 1:
                    raise ValueError(
                        f"'{cmdname}: {cmdinstance}' spatial_average mass count must be positive"
                    )
                mass_end = 2 + mass_count
                if len(remaining) <= mass_end:
                    raise ValueError(
                        f"'{cmdname}: {cmdinstance}' spatial_average declares {mass_count} "
                        "mass value(s) but does not supply them and at least one geometry tag"
                    )
                averaging_masses = tuple(float(value) for value in remaining[2:mass_end])
                tags = remaining[mass_end:]
            else:
                tags = remaining
            spectrum_limit = (
                "nyquist" if spectrum_token.lower() == "nyquist" else float(spectrum_token)
            )
            scene_objects.append(
                SAR(
                    frequencies=frequencies,
                    waveform_id=tokens[3],
                    target_amplitude=1.0 if power_mode else float(tokens[4]),
                    normalisation=tokens[4] if power_mode else "waveform",
                    target_power=target_power,
                    port_id=port_id,
                    spectrum_limit=spectrum_limit,
                    id=output_id,
                    tags=tags,
                    averaging_masses=averaging_masses,
                )
            )

    cmdname = "#snapshot"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) != 11:
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires exactly eleven parameters"
                )
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
                snapshot = Snapshot(
                    p1=p1, p2=p2, dl=dl, iterations=iterations, filename=filename, fileext=fileext
                )

            except ValueError:
                time = float(tmp[9])
                snapshot = Snapshot(
                    p1=p1, p2=p2, dl=dl, time=time, filename=filename, fileext=fileext
                )

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
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires a transform ID and one or more port IDs"
                )
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
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires a transform ID and one or more port IDs"
                )
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
            scene_objects.append(
                KSIRFarField(float(tokens[0]), float(tokens[1]), tokens[2], **kwargs)
            )

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
            scene_objects.append(
                NTFFFarField(float(tokens[0]), float(tokens[1]), tokens[2], **kwargs)
            )

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
            scene_objects.append(
                NTFFTimeFarField(float(tokens[0]), float(tokens[1]), tokens[2], **kwargs)
            )

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
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly five parameters"
                )
                raise ValueError

            material = Material(
                er=float(tmp[0]), se=float(tmp[1]), mr=float(tmp[2]), sm=float(tmp[3]), id=tmp[4]
            )
            scene_objects.append(material)

    cmdname = "#material_from_database"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) not in (2, 3):
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires a database name, a material key, "
                    "and optionally a local material ID"
                )
            scene_objects.append(
                MaterialFromDatabase(
                    database=tokens[0],
                    material=tokens[1],
                    id=tokens[2] if len(tokens) == 3 else None,
                )
            )

    cmdname = "#material_density"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tokens = cmdinstance.split()
            if len(tokens) < 2:
                raise ValueError(
                    f"'{cmdname}: {cmdinstance}' requires a density in kg/m3 and at least "
                    "one material identifier"
                )
            scene_objects.append(
                MaterialDensity(
                    density=float(tokens[0]),
                    material_ids=tokens[1:],
                )
            )

    cmdname = "#add_dispersion_debye"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            poles = int(tmp[0]) if tmp else 0
            required = 2 + 2 * poles
            if poles <= 0 or len(tmp) < required:
                logger.exception(
                    f"'{cmdname}: {' '.join(tmp)}' requires a positive pole count, "
                    "two values per pole, and at least one material identifier"
                )
                raise ValueError

            er_delta = []
            tau = []
            material_ids = tmp[(2 * poles) + 1 : len(tmp)]

            for pole in range(1, 2 * poles, 2):
                er_delta.append(float(tmp[pole]))
                tau.append(float(tmp[pole + 1]))

            debye_dispersion = AddDebyeDispersion(
                poles=poles, er_delta=er_delta, tau=tau, material_ids=material_ids
            )
            scene_objects.append(debye_dispersion)

    cmdname = "#add_dispersion_lorentz"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            poles = int(tmp[0]) if tmp else 0
            required = 2 + 3 * poles
            if poles <= 0 or len(tmp) < required:
                logger.exception(
                    f"'{cmdname}: {' '.join(tmp)}' requires a positive pole count, "
                    "three values per pole, and at least one material identifier"
                )
                raise ValueError

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
            poles = int(tmp[0]) if tmp else 0
            required = 2 + 2 * poles
            if poles <= 0 or len(tmp) < required:
                logger.exception(
                    f"'{cmdname}: {' '.join(tmp)}' requires a positive pole count, "
                    "two values per pole, and at least one material identifier"
                )
                raise ValueError

            material_ids = tmp[(2 * poles) + 1 : len(tmp)]
            omega = []
            alpha = []

            for pole in range(1, 2 * poles, 2):
                omega.append(float(tmp[pole]))
                alpha.append(float(tmp[pole + 1]))

            drude_dispersion = AddDrudeDispersion(
                poles=poles, material_ids=material_ids, omega=omega, alpha=alpha
            )
            scene_objects.append(drude_dispersion)

    cmdname = "#soil_peplinski"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 7:
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires at exactly seven parameters"
                )
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
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires exactly eleven parameters"
                )
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
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires exactly seven parameters"
                )
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
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires at exactly nine parameters"
                )
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
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " requires at least two parameters"
                )
                raise ValueError

            tokens = len(tmp)
            lmats = []
            for iter in range(tokens - 1):
                lmats.append(tmp[iter])

            material_list = MaterialList(list_of_materials=lmats, id=tmp[tokens - 1])
            scene_objects.append(material_list)

    cmdname = "#pml_formulation"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()
            if len(tmp) not in (1, 2):
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "' requires one or two parameters"
                )
                raise ValueError
            scene_objects.append(
                PMLFormulation(formulation=tmp[0], id=tmp[1] if len(tmp) == 2 else None)
            )

    cmdname = "#pml_cfs"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) not in (12, 13):
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "'"
                    + " requires twelve or thirteen parameters"
                )
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
                profile_id=tmp[12] if len(tmp) == 13 else None,
            )

            scene_objects.append(pml_cfs)

    cmdname = "#pml_slab"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) not in (7, 8, 9):
                logger.exception(
                    "'"
                    + cmdname
                    + ": "
                    + " ".join(tmp)
                    + "' requires seven, eight, or nine parameters"
                )
                raise ValueError

            profile_id = tmp[7] if len(tmp) >= 8 else None
            if profile_id is not None and profile_id.lower() == "none":
                profile_id = None
            build_pec = True
            if len(tmp) == 9:
                if tmp[8].lower() == "y":
                    build_pec = True
                elif tmp[8].lower() == "n":
                    build_pec = False
                else:
                    raise ValueError(f"{cmdname} automatic PEC enclosure must be 'y' or 'n'.")

            pml_slab = PMLSlab(
                p1=(float(tmp[0]), float(tmp[1]), float(tmp[2])),
                p2=(float(tmp[3]), float(tmp[4]), float(tmp[5])),
                maximum_face=tmp[6].lower(),
                profile_id=profile_id,
                build_pec=build_pec,
            )
            scene_objects.append(pml_slab)

    cmdname = "#symmetry_boundary"
    if multicmds[cmdname] is not None:
        for cmdinstance in multicmds[cmdname]:
            tmp = cmdinstance.split()

            if len(tmp) != 2:
                logger.exception(
                    "'" + cmdname + ": " + " ".join(tmp) + "'" + " requires exactly two parameters"
                )
                raise ValueError

            symmetry_boundary = SymmetryBoundary(face=tmp[0].lower(), type=tmp[1].lower())
            scene_objects.append(symmetry_boundary)

    return scene_objects
