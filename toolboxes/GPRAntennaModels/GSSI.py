# Copyright (C) 2015-2026, Craig Warren, Sam Stadler, Ourania Patsia
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the following attributions:
#  For the antenna_like_GSSI_1500 http://dx.doi.org/10.1190/1.3548506 and http://dx.doi.org/10.1109/TGRS.2018.2869027
#  For the antenna_like_GSSI_400 https://doi.org/10.1109/TAP.2022.3142335
#  For the antenna_like_GSSI_2000 https://doi.org/10.1002/nsg.12280

from pathlib import Path

import gprMax


def antenna_like_GSSI_1500(x, y, z, resolution=0.001, **kwargs):
    """Inserts a description of an antenna similar to the GSSI 1.5GHz antenna.
        Can be used with 1mm (default) or 2mm spatial resolution. The external
        dimensions of the antenna are 170x108x45mm. One output point is defined
        between the arms of the receiver bowtie. The bowties are aligned with
        the y axis so the output is the y component of the electric field.

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the
                            antenna. Coordinates are relative to the geometric
                            centre of the antenna in the x-y plane and the
                            bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
        kwargs (dict): Optional variables, e.g. can be fed from an optimisation
                        process.

    Returns:
        scene_objects (list): All model objects that will be part of a scene.
    """

    # All model objects that will be returned by function
    scene_objects = []

    # Antenna geometry properties
    casesize = (0.170, 0.108, 0.043)
    casethickness = 0.002
    shieldthickness = 0.002
    foamsurroundthickness = 0.003
    pcbthickness = 0.002
    skidthickness = 0.004
    bowtiebase = 0.022
    bowtieheight = 0.014
    patchheight = 0.015

    x = x - (casesize[0] / 2)
    y = y - (casesize[1] / 2)

    # Coordinates of source excitation point in antenna
    tx = x + 0.114, y + 0.053, z + skidthickness

    if resolution == 0.001:
        dx = 0.001
        dy = 0.001
        dz = 0.001
    elif resolution == 0.002:
        dx = 0.002
        dy = 0.002
        dz = 0.002
        foamsurroundthickness = 0.002
        patchheight = 0.016
        tx = x + 0.114, y + 0.052, z + skidthickness
    else:
        raise ValueError(
            "This antenna module can only be used with a spatial discretisation of 1mm or 2mm"
        )

    # If using parameters from an optimisation
    if kwargs:
        required = {
            "absorber1Er",
            "absorber1sig",
            "absorber2Er",
            "absorber2sig",
            "pcbEr",
            "pcbsig",
            "hdpeEr",
            "hdpesig",
        }
        missing = sorted(required - kwargs.keys())
        if missing:
            raise ValueError(
                "Missing GSSI 1.5 GHz optimisation parameter(s): " + ", ".join(missing)
            )

        optstate = "Custom"
        excitationfreq = kwargs.get("excitationfreq", 1.71e9)
        sourceresistance = kwargs.get("sourceresistance", 195)
        absorber1Er = kwargs["absorber1Er"]
        absorber1sig = kwargs["absorber1sig"]
        absorber2Er = kwargs["absorber2Er"]
        absorber2sig = kwargs["absorber2sig"]
        pcbEr = kwargs["pcbEr"]
        pcbsig = kwargs["pcbsig"]
        hdpeEr = kwargs["hdpeEr"]
        hdpesig = kwargs["hdpesig"]
        absorber1 = gprMax.Material(er=absorber1Er, se=absorber1sig, mr=1, sm=0, id="absorber1")
        absorber2 = gprMax.Material(er=absorber2Er, se=absorber2sig, mr=1, sm=0, id="absorber2")
        pcb = gprMax.Material(er=pcbEr, se=pcbsig, mr=1, sm=0, id="pcb")
        hdpe = gprMax.Material(er=hdpeEr, se=hdpesig, mr=1, sm=0, id="hdpe")
        scene_objects.extend((absorber1, absorber2, pcb, hdpe))

    # Otherwise choose parameters for different optimisation models
    else:
        # Specify optimisation model
        optstate = ["WarrenThesis", "DebyeAbsorber", "GiannakisPaper"]
        optstate = optstate[0]

        if optstate == "WarrenThesis":
            # Original optimised values from http://hdl.handle.net/1842/4074
            excitationfreq = 1.71e9
            sourceresistance = 230  # Correction for old (< 123) GprMax3D bug (optimised to 4)
            rxres = 925  # Resistance at Rx bowtie
            absorber1 = gprMax.Material(er=1.58, se=0.428, mr=1, sm=0, id="absorber1")
            absorber2 = gprMax.Material(
                er=3, se=0, mr=1, sm=0, id="absorber2"
            )  # Foam modelled as PCB material
            pcb = gprMax.Material(er=3, se=0, mr=1, sm=0, id="pcb")
            hdpe = gprMax.Material(er=2.35, se=0, mr=1, sm=0, id="hdpe")
            rxres = gprMax.Material(er=3, se=(1 / rxres) * (dy / (dx * dz)), mr=1, sm=0, id="rxres")
            scene_objects.extend((absorber1, absorber2, pcb, hdpe, rxres))

        elif optstate == "DebyeAbsorber":
            # Same values as WarrenThesis but uses dispersive absorber properties for Eccosorb LS22
            excitationfreq = 1.71e9
            sourceresistance = 230  # Correction for old (< 123) GprMax3D bug (optimised to 4)
            rxres = 925  # Resistance at Rx bowtie
            absorber1 = gprMax.Material(er=1, se=0, mr=1, sm=0, id="absorber1")
            # Eccosorb LS22 3-pole Debye model (https://bitbucket.org/uoyaeg/aegboxts/wiki/Home)
            absorber1_disp = gprMax.AddDebyeDispersion(
                poles=3,
                er_delta=[3.7733, 3.14418, 20.2441],
                tau=[1.00723e-11, 1.55686e-10, 3.44129e-10],
                material_ids=["absorber1"],
            )
            absorber2 = gprMax.Material(
                er=3, se=0, mr=1, sm=0, id="absorber2"
            )  # Foam modelled as PCB material
            pcb = gprMax.Material(er=3, se=0, mr=1, sm=0, id="pcb")
            hdpe = gprMax.Material(er=2.35, se=0, mr=1, sm=0, id="hdpe")
            rxres = gprMax.Material(er=3, se=(1 / rxres) * (dy / (dx * dz)), mr=1, sm=0, id="rxres")
            scene_objects.extend((absorber1, absorber1_disp, absorber2, pcb, hdpe, rxres))

        elif optstate == "GiannakisPaper":
            # Further optimised values from https://doi.org/10.1109/TGRS.2018.2869027
            sourceresistance = 195
            absorber1 = gprMax.Material(er=3.96, se=0.31, mr=1, sm=0, id="absorber1")
            absorber2 = gprMax.Material(er=1.05, se=1.01, mr=1, sm=0, id="absorber2")
            pcb = gprMax.Material(er=1.37, se=0.0002, mr=1, sm=0, id="pcb")
            hdpe = gprMax.Material(er=1.99, se=0.013, mr=1, sm=0, id="hdpe")
            scene_objects.extend((absorber1, absorber2, pcb, hdpe))

    # Antenna geometry
    # Plastic case
    b1 = gprMax.Box(
        p1=(x, y, z + skidthickness),
        p2=(x + casesize[0], y + casesize[1], z + skidthickness + casesize[2]),
        material_id="hdpe",
    )
    b2 = gprMax.Box(
        p1=(x + casethickness, y + casethickness, z + skidthickness),
        p2=(
            x + casesize[0] - casethickness,
            y + casesize[1] - casethickness,
            z + skidthickness + casesize[2] - casethickness,
        ),
        material_id="free_space",
    )

    # Metallic enclosure
    b3 = gprMax.Box(
        p1=(x + 0.025, y + casethickness, z + skidthickness),
        p2=(
            x + casesize[0] - 0.025,
            y + casesize[1] - casethickness,
            z + skidthickness + 0.027,
        ),
        material_id="pec",
    )

    # Absorber material (absorber1) and foam (absorber2) around edge of absorber
    b4 = gprMax.Box(
        p1=(
            x + 0.025 + shieldthickness,
            y + casethickness + shieldthickness,
            z + skidthickness,
        ),
        p2=(
            x + 0.025 + shieldthickness + 0.057,
            y + casesize[1] - casethickness - shieldthickness,
            z + skidthickness + 0.027 - shieldthickness - 0.001,
        ),
        material_id="absorber2",
    )
    b5 = gprMax.Box(
        p1=(
            x + 0.025 + shieldthickness + foamsurroundthickness,
            y + casethickness + shieldthickness + foamsurroundthickness,
            z + skidthickness,
        ),
        p2=(
            x + 0.025 + shieldthickness + 0.057 - foamsurroundthickness,
            y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness,
            z + skidthickness + 0.027 - shieldthickness,
        ),
        material_id="absorber1",
    )
    b6 = gprMax.Box(
        p1=(x + 0.086, y + casethickness + shieldthickness, z + skidthickness),
        p2=(
            x + 0.086 + 0.057,
            y + casesize[1] - casethickness - shieldthickness,
            z + skidthickness + 0.027 - shieldthickness - 0.001,
        ),
        material_id="absorber2",
    )
    b7 = gprMax.Box(
        p1=(
            x + 0.086 + foamsurroundthickness,
            y + casethickness + shieldthickness + foamsurroundthickness,
            z + skidthickness,
        ),
        p2=(
            x + 0.086 + 0.057 - foamsurroundthickness,
            y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness,
            z + skidthickness + 0.027 - shieldthickness,
        ),
        material_id="absorber1",
    )

    # PCB
    b8 = gprMax.Box(
        p1=(
            x + 0.025 + shieldthickness + foamsurroundthickness,
            y + casethickness + shieldthickness + foamsurroundthickness,
            z + skidthickness,
        ),
        p2=(
            x + 0.086 - shieldthickness - foamsurroundthickness,
            y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness,
            z + skidthickness + pcbthickness,
        ),
        material_id="pcb",
    )
    b9 = gprMax.Box(
        p1=(
            x + 0.086 + foamsurroundthickness,
            y + casethickness + shieldthickness + foamsurroundthickness,
            z + skidthickness,
        ),
        p2=(
            x + 0.086 + 0.057 - foamsurroundthickness,
            y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness,
            z + skidthickness + pcbthickness,
        ),
        material_id="pcb",
    )

    scene_objects.extend((b1, b2, b3, b4, b5, b6, b7, b8, b9))

    # PCB components
    if resolution == 0.001:
        # Rx & Tx bowties
        a = 0
        b = 0
        while b < 13:
            p1 = gprMax.Plate(
                p1=(x + 0.045 + a * dx, y + 0.039 + b * dx, z + skidthickness),
                p2=(x + 0.065 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness),
                material_id="pec",
            )
            p2 = gprMax.Plate(
                p1=(x + 0.045 + a * dx, y + 0.067 - b * dx, z + skidthickness),
                p2=(x + 0.065 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness),
                material_id="pec",
            )
            p3 = gprMax.Plate(
                p1=(x + 0.104 + a * dx, y + 0.039 + b * dx, z + skidthickness),
                p2=(x + 0.124 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness),
                material_id="pec",
            )
            p4 = gprMax.Plate(
                p1=(x + 0.104 + a * dx, y + 0.067 - b * dx, z + skidthickness),
                p2=(x + 0.124 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness),
                material_id="pec",
            )
            scene_objects.extend((p1, p2, p3, p4))
            b += 1
            if a == 2 or a == 4 or a == 7:
                p5 = gprMax.Plate(
                    p1=(x + 0.045 + a * dx, y + 0.039 + b * dx, z + skidthickness),
                    p2=(x + 0.065 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness),
                    material_id="pec",
                )
                p6 = gprMax.Plate(
                    p1=(x + 0.045 + a * dx, y + 0.067 - b * dx, z + skidthickness),
                    p2=(x + 0.065 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness),
                    material_id="pec",
                )
                p7 = gprMax.Plate(
                    p1=(x + 0.104 + a * dx, y + 0.039 + b * dx, z + skidthickness),
                    p2=(x + 0.124 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness),
                    material_id="pec",
                )
                p8 = gprMax.Plate(
                    p1=(x + 0.104 + a * dx, y + 0.067 - b * dx, z + skidthickness),
                    p2=(x + 0.124 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness),
                    material_id="pec",
                )
                b += 1
                scene_objects.extend((p5, p6, p7, p8))
            a += 1
        # Rx extension section (upper y)
        p9 = gprMax.Plate(
            p1=(x + 0.044, y + 0.068, z + skidthickness),
            p2=(x + 0.044 + bowtiebase, y + 0.068 + patchheight, z + skidthickness),
            material_id="pec",
        )
        # Tx extension section (upper y)
        p10 = gprMax.Plate(
            p1=(x + 0.103, y + 0.068, z + skidthickness),
            p2=(x + 0.103 + bowtiebase, y + 0.068 + patchheight, z + skidthickness),
            material_id="pec",
        )
        scene_objects.extend((p9, p10))

        # Edges that represent wire between bowtie halves in 1mm model
        e1 = gprMax.Edge(
            p1=(tx[0] - 0.059, tx[1] - dy, tx[2]),
            p2=(tx[0] - 0.059, tx[1], tx[2]),
            material_id="pec",
        )
        e2 = gprMax.Edge(
            p1=(tx[0] - 0.059, tx[1] + dy, tx[2]),
            p2=(tx[0] - 0.059, tx[1] + 0.002, tx[2]),
            material_id="pec",
        )
        e3 = gprMax.Edge(p1=(tx[0], tx[1] - dy, tx[2]), p2=(tx[0], tx[1], tx[2]), material_id="pec")
        e4 = gprMax.Edge(
            p1=(tx[0], tx[1] + dz, tx[2]), p2=(tx[0], tx[1] + 0.002, tx[2]), material_id="pec"
        )
        scene_objects.extend((e1, e2, e3, e4))

    elif resolution == 0.002:
        # Rx & Tx bowties
        for a in range(0, 6):
            p1 = gprMax.Plate(
                p1=(x + 0.044 + a * dx, y + 0.040 + a * dx, z + skidthickness),
                p2=(x + 0.066 - a * dx, y + 0.040 + a * dx + dy, z + skidthickness),
                material_id="pec",
            )
            p2 = gprMax.Plate(
                p1=(x + 0.044 + a * dx, y + 0.064 - a * dx, z + skidthickness),
                p2=(x + 0.066 - a * dx, y + 0.064 - a * dx + dy, z + skidthickness),
                material_id="pec",
            )
            p3 = gprMax.Plate(
                p1=(x + 0.103 + a * dx, y + 0.040 + a * dx, z + skidthickness),
                p2=(x + 0.125 - a * dx, y + 0.040 + a * dx + dy, z + skidthickness),
                material_id="pec",
            )
            p4 = gprMax.Plate(
                p1=(x + 0.103 + a * dx, y + 0.064 - a * dx, z + skidthickness),
                p2=(x + 0.125 - a * dx, y + 0.064 - a * dx + dy, z + skidthickness),
                material_id="pec",
            )
            # Rx extension section (upper y)
            p5 = gprMax.Plate(
                p1=(x + 0.044, y + 0.066, z + skidthickness),
                p2=(x + 0.044 + bowtiebase, y + 0.066 + patchheight, z + skidthickness),
                material_id="pec",
            )
            # Tx extension section (upper y)
            p6 = gprMax.Plate(
                p1=(x + 0.103, y + 0.066, z + skidthickness),
                p2=(x + 0.103 + bowtiebase, y + 0.066 + patchheight, z + skidthickness),
                material_id="pec",
            )
            scene_objects.extend((p1, p2, p3, p4, p5, p6))

    # Rx extension section (lower y)
    p11 = gprMax.Plate(
        p1=(x + 0.044, y + 0.024, z + skidthickness),
        p2=(x + 0.044 + bowtiebase, y + 0.024 + patchheight, z + skidthickness),
        material_id="pec",
    )
    # Tx extension section (lower y)
    p12 = gprMax.Plate(
        p1=(x + 0.103, y + 0.024, z + skidthickness),
        p2=(x + 0.103 + bowtiebase, y + 0.024 + patchheight, z + skidthickness),
        material_id="pec",
    )
    scene_objects.extend((p11, p12))

    # Skid
    b10 = gprMax.Box(
        p1=(x, y, z), p2=(x + casesize[0], y + casesize[1], z + skidthickness), material_id="hdpe"
    )
    scene_objects.append(b10)

    # Geometry views
    gv1 = gprMax.GeometryView(
        p1=(x - dx, y - dy, z - dz),
        p2=(
            x + casesize[0] + dx,
            y + casesize[1] + dy,
            z + skidthickness + casesize[2] + dz,
        ),
        dl=(dx, dy, dz),
        filename="antenna_like_GSSI_1500",
        output_type="n",
    )
    gv2 = gprMax.GeometryView(
        p1=(x, y, z),
        p2=(x + casesize[0], y + casesize[1], z + 0.010),
        dl=(dx, dy, dz),
        filename="antenna_like_GSSI_1500_pcb",
        output_type="f",
    )
    # scene_objects.extend((gv1, gv2))

    # Excitation
    if optstate in ("WarrenThesis", "DebyeAbsorber", "Custom"):
        # Gaussian pulse
        w1 = gprMax.Waveform(wave_type="gaussian", amp=1, freq=excitationfreq, id="my_gaussian")
        vs1 = gprMax.VoltageSource(
            polarisation="y",
            p1=(tx[0], tx[1], tx[2]),
            resistance=sourceresistance,
            waveform_id="my_gaussian",
        )
        scene_objects.extend((w1, vs1))

    elif optstate == "GiannakisPaper":
        # Optimised custom pulse
        exc1 = gprMax.ExcitationFile(
            filepath=Path(__file__).with_name("GSSI_1500MHz_pulse.txt"),
            kind="linear",
            fill_value="extrapolate",
        )
        vs1 = gprMax.VoltageSource(
            polarisation="y",
            p1=(tx[0], tx[1], tx[2]),
            resistance=sourceresistance,
            waveform_id="my_pulse",
        )
        scene_objects.extend((exc1, vs1))

    # Output point - receiver bowtie
    if resolution == 0.001:
        if optstate == "WarrenThesis" or optstate == "DebyeAbsorber":
            e1 = gprMax.Edge(
                p1=(tx[0] - 0.059, tx[1], tx[2]),
                p2=(tx[0] - 0.059, tx[1] + dy, tx[2]),
                material_id="rxres",
            )
            scene_objects.append(e1)
        r1 = gprMax.Rx(p1=(tx[0] - 0.059, tx[1], tx[2]), id="rxbowtie", outputs=["Ey"])
        scene_objects.append(r1)

    elif resolution == 0.002:
        if optstate == "WarrenThesis" or optstate == "DebyeAbsorber":
            e1 = gprMax.Edge(
                p1=(tx[0] - 0.060, tx[1], tx[2]),
                p2=(tx[0] - 0.060, tx[1] + dy, tx[2]),
                material_id="rxres",
            )
            scene_objects.append(e1)
        r1 = gprMax.Rx(p1=(tx[0] - 0.060, tx[1], tx[2]), id="rxbowtie", outputs=["Ey"])
        scene_objects.append(r1)

    return scene_objects


def antenna_like_GSSI_2000(x, y, z, resolution=0.001):
    """Insert a model similar to the GSSI 2 GHz palm antenna.

    The model is based on the optimised unit No. 1 parameters reported by
    Patsia et al. (2024), https://doi.org/10.1002/nsg.12280. It must be used
    with a 1 mm cubic spatial discretisation. The nominal antenna dimensions
    are 86 x 86 x 68 mm; in the discretised model the skid projects 2 mm
    beyond each side of the case in the y direction. A Gaussian voltage source
    excites the transmitter and an ``Ey`` receiver is placed across the
    resistively loaded receiver gap.

    Args:
        x, y, z (float): Coordinates of the geometric centre of the antenna
            in the x-y plane and the bottom of the antenna skid in z.
        resolution (float): Spatial resolution of the antenna model. Only
            1 mm is supported.

    Returns:
        scene_objects (list): All objects required to add the antenna to a
            scene.
    """

    if resolution != 0.001:
        raise ValueError("This antenna module can only be used with a spatial discretisation of 1 mm")

    dx = dy = dz = resolution
    scene_objects = []

    # The original model used the lower x-y corner and had the skid at the
    # high-z face. Translate to the toolbox convention (x-y centre and bottom
    # of skid) by reflecting only the local z coordinate. This is a rigid
    # transformation and does not alter the antenna geometry.
    x0 = x - 0.043
    y0 = y - 0.043
    legacy_zmax = 0.067

    def point(p):
        return (x0 + p[0], y0 + p[1], z + legacy_zmax - p[2])

    def ordered_points(p1, p2):
        q1 = point(p1)
        q2 = point(p2)
        return tuple(min(a, b) for a, b in zip(q1, q2)), tuple(max(a, b) for a, b in zip(q1, q2))

    def add_box(p1, p2, material_id):
        q1, q2 = ordered_points(p1, p2)
        scene_objects.append(gprMax.Box(p1=q1, p2=q2, material_id=material_id))

    def add_plate(p1, p2, material_id="pec"):
        q1, q2 = ordered_points(p1, p2)
        scene_objects.append(gprMax.Plate(p1=q1, p2=q2, material_id=material_id))

    def add_edge(p1, p2, material_id):
        q1, q2 = ordered_points(p1, p2)
        scene_objects.append(gprMax.Edge(p1=q1, p2=q2, material_id=material_id))

    # Optimised material and equivalent-edge parameters for unit No. 1. The
    # absorber and receiver resistances are converted to bulk conductivity
    # exactly as in the original 1 mm model.
    absorber1_resistance = 920.0
    absorber2_resistance = 790.0
    source_resistance = 560.0
    receiver_resistance = 200008.0107
    edge_to_bulk = dy / (dx * dz)

    material_properties = (
        ("gssi2000_rxres", 1.0560, (1 / receiver_resistance) * edge_to_bulk),
        ("gssi2000_plastic", 6.10, 0.0029),
        ("gssi2000_skid", 2.6792, 0.0050),
        ("gssi2000_pcb", 1.5220, 0.0231),
        ("gssi2000_divider", 1.0143, 45685957),
        ("gssi2000_case_inner", 1.0, 45355559),
        ("gssi2000_plastic_inner", 1.1758, 0.0017),
        ("gssi2000_gasket", 1.0, 100000000),
        ("gssi2000_absorber1", 1.10091, (1 / absorber1_resistance) * edge_to_bulk),
        ("gssi2000_absorber2", 1.073, (1 / absorber2_resistance) * edge_to_bulk),
    )
    scene_objects.extend(
        gprMax.Material(er=er, se=se, mr=1, sm=0, id=material_id) for material_id, er, se in material_properties
    )

    # Case and skid. Coordinates below are local coordinates from the legacy
    # model and are transformed by the helpers above.
    add_box((0, -0.002, 0.064), (0.086, 0.088, 0.067), "gssi2000_skid")
    add_box((0.001, -0.001, -0.001), (0.085, 0.087, 0.066), "free_space")
    add_box((0.001, -0.001, -0.001), (0.085, 0.087, 0.066), "gssi2000_plastic")
    add_box((0.006, 0.040, 0.065), (0.039, 0.046, 0.066), "free_space")
    add_box((0.047, 0.040, 0.065), (0.080, 0.046, 0.066), "free_space")
    add_box((0.002, 0, 0), (0.084, 0.086, 0.066), "free_space")
    add_box((0.002, 0, 0.020), (0.084, 0.086, 0.063), "gssi2000_case_inner")
    add_box((0.003, 0.001, 0.021), (0.083, 0.085, 0.063), "free_space")
    add_plate((0.001, -0.001, 0.066), (0.085, 0.087, 0.066), "gssi2000_plastic")

    ydist = 0.007

    def add_bowtie(shape_xoffset, feed_x):
        zbowtie = 0.064
        add_plate(
            (0.008 + shape_xoffset, 0.010, zbowtie),
            (0.032 + shape_xoffset, 0.030, zbowtie),
        )
        add_plate(
            (0.017 + shape_xoffset, 0.023, zbowtie),
            (0.023 + shape_xoffset, 0.041, zbowtie),
        )
        add_plate(
            (0.015 + shape_xoffset, 0.023, zbowtie),
            (0.017 + shape_xoffset, 0.040, zbowtie),
        )
        add_plate(
            (0.023 + shape_xoffset, 0.023, zbowtie),
            (0.025 + shape_xoffset, 0.040, zbowtie),
        )
        add_plate(
            (0.013 + shape_xoffset, 0.023, zbowtie),
            (0.015 + shape_xoffset, 0.039, zbowtie),
        )
        add_plate(
            (0.025 + shape_xoffset, 0.023, zbowtie),
            (0.027 + shape_xoffset, 0.039, zbowtie),
        )

        for i in range(1, 6):
            upper_y = 0.033 - i * 0.001 + ydist - 0.001
            if i == 5:
                upper_y -= 0.001
            add_plate(
                (0.013 - i * 0.001 + shape_xoffset, 0.030, zbowtie),
                (0.014 - i * 0.001 + shape_xoffset, upper_y, zbowtie),
            )
            add_plate(
                (0.026 + i * 0.001 + shape_xoffset, 0.030, zbowtie),
                (0.027 + i * 0.001 + shape_xoffset, upper_y, zbowtie),
            )

        add_plate(
            (0.008 + shape_xoffset, 0.055, zbowtie),
            (0.032 + shape_xoffset, 0.075, zbowtie),
        )
        for i in range(1, 6):
            lower_y = 0.046 + i * 0.001
            if i == 5:
                lower_y += 0.001
            add_plate(
                (0.013 - i * 0.001 + shape_xoffset, lower_y, zbowtie),
                (0.014 - i * 0.001 + shape_xoffset, 0.055, zbowtie),
            )
            add_plate(
                (0.026 + i * 0.001 + shape_xoffset, lower_y, zbowtie),
                (0.027 + i * 0.001 + shape_xoffset, 0.055, zbowtie),
            )

        add_plate(
            (0.017 + shape_xoffset, 0.044, zbowtie),
            (0.023 + shape_xoffset, 0.055, zbowtie),
        )
        add_plate(
            (0.015 + shape_xoffset, 0.045, zbowtie),
            (0.017 + shape_xoffset, 0.055, zbowtie),
        )
        add_plate(
            (0.023 + shape_xoffset, 0.045, zbowtie),
            (0.025 + shape_xoffset, 0.055, zbowtie),
        )
        add_plate(
            (0.013 + shape_xoffset, 0.046, zbowtie),
            (0.015 + shape_xoffset, 0.055, zbowtie),
        )
        add_plate(
            (0.025 + shape_xoffset, 0.046, zbowtie),
            (0.027 + shape_xoffset, 0.055, zbowtie),
        )
        add_plate((feed_x - 0.001, 0.040, zbowtie), (feed_x + 0.001, 0.042, zbowtie))
        add_plate((feed_x - 0.001, 0.043, zbowtie), (feed_x + 0.001, 0.044, zbowtie))

    # Transmitter and receiver bowties.
    add_bowtie(shape_xoffset=0.003, feed_x=0.023)
    add_bowtie(shape_xoffset=0.043, feed_x=0.063)

    # PCBs and absorbers.
    add_box((0.006, 0.005, 0.063), (0.039, 0.081, 0.064), "gssi2000_pcb")
    add_box((0.047, 0.005, 0.063), (0.080, 0.081, 0.064), "gssi2000_pcb")
    add_box((0.003, 0.001, 0.058), (0.042, 0.085, 0.063), "gssi2000_absorber1")
    add_box((0.044, 0.001, 0.058), (0.083, 0.085, 0.063), "gssi2000_absorber1")
    add_box((0.003, 0.001, 0.052), (0.042, 0.085, 0.058), "gssi2000_absorber2")
    add_box((0.044, 0.001, 0.052), (0.083, 0.085, 0.058), "gssi2000_absorber2")

    # Plastic shells and air gaps within the absorbers.
    add_box((0.012, 0.032, 0.052), (0.036, 0.045, 0.063), "free_space")
    add_box((0.051, 0.032, 0.052), (0.075, 0.045, 0.063), "free_space")
    add_box((0.012, 0.032, 0.052), (0.036, 0.045, 0.063), "gssi2000_plastic_inner")
    add_box((0.051, 0.032, 0.052), (0.075, 0.045, 0.063), "gssi2000_plastic_inner")
    add_box((0.013, 0.033, 0.053), (0.035, 0.044, 0.063), "free_space")
    add_box((0.052, 0.033, 0.053), (0.074, 0.044, 0.063), "free_space")

    # EMI gaskets.
    add_box((0.009, 0.030, 0.051), (0.035, 0.056, 0.052), "gssi2000_gasket")
    add_box((0.051, 0.026, 0.051), (0.077, 0.056, 0.052), "gssi2000_gasket")

    # Resistively loaded receiver gap.
    add_edge((0.063, 0.042, 0.064), (0.063, 0.043, 0.064), "free_space")
    add_edge((0.063, 0.042, 0.064), (0.063, 0.043, 0.064), "gssi2000_rxres")

    # Metallic divider and its apertures.
    add_box((0.041, 0.001, 0.020), (0.045, 0.085, 0.064), "free_space")
    add_box((0.041, 0.001, 0.020), (0.045, 0.085, 0.064), "gssi2000_divider")
    add_box((0.041, 0.020, 0.043), (0.045, 0.045, 0.063), "free_space")
    add_box((0.041, 0.047, 0.043), (0.045, 0.070, 0.063), "free_space")

    source_point = point((0.023, 0.042, 0.064))
    receiver_point = point((0.063, 0.042, 0.064))
    waveform = gprMax.Waveform(wave_type="gaussian", amp=-1, freq=2.12e9, id="gssi2000_gaussian")
    source = gprMax.VoltageSource(
        polarisation="y",
        p1=source_point,
        resistance=source_resistance,
        waveform_id="gssi2000_gaussian",
    )
    receiver = gprMax.Rx(p1=receiver_point, id="gssi2000_rxbowtie", outputs=["Ey"])
    scene_objects.extend((waveform, source, receiver))

    return scene_objects


def antenna_like_GSSI_400(x, y, z, resolution=0.002, **kwargs):
    """Inserts a description of an antenna similar to the GSSI 400MHz antenna.
        This model represents an update to the previous model of the GSSI 400MHz
        antenna and was created and optimised by Stadler et al. (2022)
        in: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9686638.
        Can be used with 2mm spatial resolution.
        The external dimensions of the antenna are 300x300x178mm.
        One output point is defined between the arms of the receiver bowtie.
        The bowties are aligned with the y axis so the output is the y component
        of the electric field.

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the
                            antenna. Coordinates are relative to the geometric
                            centre of the antenna in the x-y plane and the
                            bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
        kwargs (dict): Optional variables, e.g. can be fed from an optimisation
                        process.

    Returns:
        scene_objects (list): All model objects that will be part of a scene.
    """

    # All model objects that will be returned by function
    scene_objects = []

    # Antenna geometry properties
    casesize = (0.3, 0.3, 0.178)  # original
    casethickness = 0.002
    shieldthickness = 0.002
    pcbthickness = 0.002
    bowtiebase = 0.06
    bowtieheight = 0.06  # original 0.056
    patchheight = 0.06  # original 0.056
    metalboxheight = 0.089
    metalmiddleplateheight = 0.11

    smooth_dec = "yes"  # choose to use dielectric smoothing or not
    src_type = "GSSI_400MHz_pulse"  # (or voltage_source)
    pcber = 6.401200848809589
    hdper = 1.0
    skidthickness = 0.01

    if resolution != 0.002:
        raise ValueError("The GSSI 400 MHz antenna model requires a 2 mm resolution")

    # If using parameters from an optimisation
    if kwargs:
        required = {"excitationfreq", "sourceresistance", "absorberEr", "absorbersig"}
        missing = sorted(required - kwargs.keys())
        if missing:
            raise ValueError(
                "Missing GSSI 400 MHz optimisation parameter(s): " + ", ".join(missing)
            )
        excitationfreq = kwargs["excitationfreq"]
        sourceresistance = kwargs["sourceresistance"]
        receiverresistance = sourceresistance
        absorberEr = kwargs["absorberEr"]
        absorbersig = kwargs["absorbersig"]

    # Otherwise choose pre-set optimised parameters
    else:
        excitationfreq = 3.5e8  # Hz, only used with voltage_source
        sourceresistance = 257.97407389585214  # Ohms
        receiverresistance = 288.92728542970417  # Ohms
        absorberEr = 2.42966922703319
        absorbersig = 0.03839822151712033  # S/m

    x = x - (casesize[0] / 2)
    y = y - (casesize[1] / 2)

    # Coordinates of source excitation point in antenna
    tx = x + 0.01 + 0.005 + 0.056, y + casethickness + 0.005 + 0.143, z + skidthickness

    dx = 0.002
    dy = 0.002
    dz = 0.002
    foamsurroundthickness = 0.002
    metalboxheight = 0.088
    tx = (
        x + 0.01 + 0.004 + 0.056,
        y + casethickness + 0.005 + 0.143 - 0.002,
        z + skidthickness - 0.002,
    )

    # Material definitions
    absorber = gprMax.Material(er=absorberEr, se=absorbersig, mr=1, sm=0, id="absorber")
    pcb = gprMax.Material(er=pcber, se=0, mr=1, sm=0, id="pcb")
    hdpe = gprMax.Material(er=hdper, se=0, mr=1, sm=0, id="hdpe")
    scene_objects.extend((absorber, pcb, hdpe))

    # Antenna geometry
    if smooth_dec == "yes":
        # Plastic case
        b1 = gprMax.Box(
            p1=(x, y, z + skidthickness - 0.002),
            p2=(x + casesize[0], y + casesize[1], z + casesize[2]),
            material_id="hdpe",
        )
        b2 = gprMax.Box(
            p1=(x + casethickness, y + casethickness, z + skidthickness - 0.002),
            p2=(
                x + casesize[0] - casethickness,
                y + casesize[1] - casethickness,
                z + casesize[2] - casethickness,
            ),
            material_id="free_space",
        )

        # Metallic enclosure
        b3 = gprMax.Box(
            p1=(
                x + casethickness,
                y + casethickness,
                z + skidthickness + (metalmiddleplateheight - metalboxheight),
            ),
            p2=(
                x + casesize[0] - casethickness,
                y + casesize[1] - casethickness,
                z + skidthickness + (metalmiddleplateheight - metalboxheight) + metalboxheight,
            ),
            material_id="pec",
        )

        # Absorber, and foam (modelled as PCB material) around edge of absorber
        b4 = gprMax.Box(
            p1=(x + casethickness, y + casethickness, z + skidthickness),
            p2=(
                x + casesize[0] - casethickness,
                y + casesize[1] - casethickness,
                z + skidthickness + (metalmiddleplateheight - metalboxheight),
            ),
            material_id="absorber",
        )
        b5 = gprMax.Box(
            p1=(
                x + casethickness + shieldthickness,
                y + casethickness + shieldthickness,
                z + skidthickness + (metalmiddleplateheight - metalboxheight),
            ),
            p2=(
                x + casesize[0] - casethickness - shieldthickness,
                y + casesize[1] - casethickness - shieldthickness,
                z + skidthickness - shieldthickness + metalmiddleplateheight,
            ),
            material_id="absorber",
        )
        scene_objects.extend((b1, b2, b3, b4, b5))

        # PCB
        b6 = gprMax.Box(
            p1=(
                x + 0.01 + 0.005 + 0.017,
                y + casethickness + 0.005 + 0.021,
                z + skidthickness - 0.002,
            ),
            p2=(
                x + 0.01 + 0.005 + 0.033 + bowtiebase,
                y + casethickness + 0.006 + 0.202 + patchheight,
                z + skidthickness + pcbthickness - 0.002,
            ),
            material_id="pcb",
        )
        b7 = gprMax.Box(
            p1=(
                x + 0.01 + 0.005 + 0.179,
                y + casethickness + 0.005 + 0.021,
                z + skidthickness - 0.002,
            ),
            p2=(
                x + 0.01 + 0.005 + 0.195 + bowtiebase,
                y + casethickness + 0.006 + 0.202 + patchheight,
                z + skidthickness + pcbthickness - 0.002,
            ),
            material_id="pcb",
        )
        scene_objects.extend((b6, b7))

    elif smooth_dec == "no":
        # Plastic case
        b8 = gprMax.Box(
            p1=(x, y, z + skidthickness - 0.002),
            p2=(x + casesize[0], y + casesize[1], z + casesize[2]),
            material_id="hdpe",
            averaging="n",
        )
        b9 = gprMax.Box(
            p1=(x + casethickness, y + casethickness, z + skidthickness - 0.002),
            p2=(
                x + casesize[0] - casethickness,
                y + casesize[1] - casethickness,
                z + casesize[2] - casethickness,
            ),
            material_id="free_space",
            averaging="n",
        )

        # Metallic enclosure
        b10 = gprMax.Box(
            p1=(
                x + casethickness,
                y + casethickness,
                z + skidthickness + (metalmiddleplateheight - metalboxheight),
            ),
            p2=(
                x + casesize[0] - casethickness,
                y + casesize[1] - casethickness,
                z + skidthickness + (metalmiddleplateheight - metalboxheight) + metalboxheight,
            ),
            material_id="pec",
        )

        # Absorber, and foam (modelled as PCB material) around edge of absorber
        b11 = gprMax.Box(
            p1=(x + casethickness, y + casethickness, z + skidthickness),
            p2=(
                x + casesize[0] - casethickness,
                y + casesize[1] - casethickness,
                z + skidthickness + (metalmiddleplateheight - metalboxheight),
            ),
            material_id="absorber",
            averaging="n",
        )
        b12 = gprMax.Box(
            p1=(
                x + casethickness + shieldthickness,
                y + casethickness + shieldthickness,
                z + skidthickness + (metalmiddleplateheight - metalboxheight),
            ),
            p2=(
                x + casesize[0] - casethickness - shieldthickness,
                y + casesize[1] - casethickness - shieldthickness,
                z + skidthickness - shieldthickness + metalmiddleplateheight,
            ),
            material_id="absorber",
            averaging="n",
        )
        scene_objects.extend((b8, b9, b10, b11, b12))

        # PCB
        b13 = gprMax.Box(
            p1=(
                x + 0.01 + 0.005 + 0.017,
                y + casethickness + 0.005 + 0.021,
                z + skidthickness - 0.002,
            ),
            p2=(
                x + 0.01 + 0.005 + 0.033 + bowtiebase,
                y + casethickness + 0.006 + 0.202 + patchheight,
                z + skidthickness + pcbthickness - 0.002,
            ),
            material_id="pcb",
            averaging="n",
        )
        b14 = gprMax.Box(
            p1=(
                x + 0.01 + 0.005 + 0.179,
                y + casethickness + 0.005 + 0.021,
                z + skidthickness,
            ),
            p2=(
                x + 0.01 + 0.005 + 0.195 + bowtiebase,
                y + casethickness + 0.006 + 0.202 + patchheight,
                z + skidthickness + pcbthickness,
            ),
            material_id="pcb",
            averaging="n",
        )
        scene_objects.extend((b13, b14))

    # PCB components
    # My own bowties with triangle commands
    # "left" side
    # extension plates
    p1 = gprMax.Plate(
        p1=(
            x + 0.01 + 0.005 + 0.025,
            y + casethickness + 0.005 + 0.021,
            z + skidthickness - 0.002,
        ),
        p2=(
            x + 0.01 + 0.005 + 0.025 + bowtiebase,
            y + casethickness + 0.005 + 0.021 + patchheight,
            z + skidthickness - 0.002,
        ),
        material_id="pec",
    )
    p2 = gprMax.Plate(
        p1=(
            x + 0.01 + 0.005 + 0.025,
            y + casethickness + 0.005 + 0.203,
            z + skidthickness - 0.002,
        ),
        p2=(
            x + 0.01 + 0.005 + 0.025 + bowtiebase,
            y + casethickness + 0.005 + 0.203 + patchheight,
            z + skidthickness - 0.002,
        ),
        material_id="pec",
    )
    # triangles
    t1 = gprMax.Triangle(
        p1=(
            x + 0.01 + 0.005 + 0.025,
            y + casethickness + 0.005 + 0.081,
            z + skidthickness - 0.002,
        ),
        p2=(
            x + 0.01 + 0.005 + 0.025 + bowtiebase,
            y + casethickness + 0.005 + 0.081,
            z + skidthickness - 0.002,
        ),
        p3=(
            x + 0.01 + 0.005 + 0.025 + (bowtiebase / 2),
            y + casethickness + 0.005 + 0.081 + bowtieheight,
            z + skidthickness - 0.002,
        ),
        thickness=0,
        material_id="pec",
    )
    t2 = gprMax.Triangle(
        p1=(
            x + 0.01 + 0.005 + 0.025,
            y + casethickness + 0.005 + 0.203,
            z + skidthickness - 0.002,
        ),
        p2=(
            x + 0.01 + 0.005 + 0.025 + bowtiebase,
            y + casethickness + 0.005 + 0.203,
            z + skidthickness - 0.002,
        ),
        p3=(
            x + 0.01 + 0.005 + 0.025 + (bowtiebase / 2),
            y + casethickness + 0.005 + 0.203 - bowtieheight,
            z + skidthickness - 0.002,
        ),
        thickness=0,
        material_id="pec",
    )
    # "right" side
    p3 = gprMax.Plate(
        p1=(
            x + 0.01 + 0.005 + 0.187,
            y + casethickness + 0.005 + 0.021,
            z + skidthickness - 0.002,
        ),
        p2=(
            x + 0.01 + 0.005 + 0.187 + bowtiebase,
            y + casethickness + 0.005 + 0.021 + patchheight,
            z + skidthickness - 0.002,
        ),
        material_id="pec",
    )
    p4 = gprMax.Plate(
        p1=(
            x + 0.01 + 0.005 + 0.187,
            y + casethickness + 0.005 + 0.203,
            z + skidthickness - 0.002,
        ),
        p2=(
            x + 0.01 + 0.005 + 0.187 + bowtiebase,
            y + casethickness + 0.005 + 0.203 + patchheight,
            z + skidthickness - 0.002,
        ),
        material_id="pec",
    )
    # triangles
    t3 = gprMax.Triangle(
        p1=(
            x + 0.01 + 0.005 + 0.187,
            y + casethickness + 0.005 + 0.081,
            z + skidthickness - 0.002,
        ),
        p2=(
            x + 0.01 + 0.005 + 0.187 + bowtiebase,
            y + casethickness + 0.005 + 0.081,
            z + skidthickness - 0.002,
        ),
        p3=(
            x + 0.01 + 0.005 + 0.187 + (bowtiebase / 2),
            y + casethickness + 0.005 + 0.081 + bowtieheight,
            z + skidthickness - 0.002,
        ),
        thickness=0,
        material_id="pec",
    )
    t4 = gprMax.Triangle(
        p1=(
            x + 0.01 + 0.005 + 0.187,
            y + casethickness + 0.005 + 0.203,
            z + skidthickness - 0.002,
        ),
        p2=(
            x + 0.01 + 0.005 + 0.187 + bowtiebase,
            y + casethickness + 0.005 + 0.203,
            z + skidthickness - 0.002,
        ),
        p3=(
            x + 0.01 + 0.005 + 0.187 + (bowtiebase / 2),
            y + casethickness + 0.005 + 0.203 - bowtieheight,
            z + skidthickness - 0.002,
        ),
        thickness=0,
        material_id="pec",
    )

    # Edges that represent wire between bowtie halves in 2mm model
    e1 = gprMax.Edge(
        p1=(tx[0] + 0.162, tx[1] - dy, tx[2]), p2=(tx[0] + 0.162, tx[1], tx[2]), material_id="pec"
    )
    e2 = gprMax.Edge(
        p1=(tx[0] + 0.162, tx[1] + dy, tx[2]),
        p2=(tx[0] + 0.162, tx[1] + 2 * dy, tx[2]),
        material_id="pec",
    )
    e3 = gprMax.Edge(p1=(tx[0], tx[1] - dy, tx[2]), p2=(tx[0], tx[1], tx[2]), material_id="pec")
    e4 = gprMax.Edge(
        p1=(tx[0], tx[1] + dy, tx[2]), p2=(tx[0], tx[1] + 2 * dy, tx[2]), material_id="pec"
    )
    scene_objects.extend((p1, p2, t1, t2, p3, p4, t3, t4, e1, e2, e3, e4))

    # Metallic plate extension
    b15 = gprMax.Box(
        p1=(x + (casesize[0] / 2), y + casethickness, z + skidthickness),
        p2=(
            x + (casesize[0] / 2) + shieldthickness,
            y + casesize[1] - casethickness,
            z + skidthickness + metalmiddleplateheight,
        ),
        material_id="pec",
    )

    # Skid
    if smooth_dec == "yes":
        b16 = gprMax.Box(
            p1=(x, y, z),
            p2=(x + casesize[0], y + casesize[1], z + skidthickness - 0.002),
            material_id="hdpe",
        )
    elif smooth_dec == "no":
        b16 = gprMax.Box(
            p1=(x, y, z),
            p2=(x + casesize[0], y + casesize[1], z + skidthickness - 0.002),
            material_id="hdpe",
            averaging="n",
        )
    scene_objects.extend((b15, b16))

    # Source
    if src_type == "voltage_source":
        w1 = gprMax.Waveform(wave_type="gaussian", amp=1, freq=excitationfreq, id="my_gaussian")
        vs1 = gprMax.VoltageSource(
            polarisation="y",
            p1=(tx[0], tx[1], tx[2]),
            resistance=sourceresistance,
            waveform_id="my_gaussian",
        )
        scene_objects.extend((w1, vs1))
    elif src_type == "transmission_line":
        w1 = gprMax.Waveform(wave_type="gaussian", amp=1, freq=excitationfreq, id="my_gaussian")
        tl1 = gprMax.TransmissionLine(
            polarisation="y",
            p1=(tx[0], tx[1], tx[2]),
            resistance=sourceresistance,
            waveform_id="my_gaussian",
        )
        scene_objects.extend((w1, tl1))
    else:
        # Optimised custom pulse
        exc1 = gprMax.ExcitationFile(
            filepath=Path(__file__).with_name("GSSI_400MHz_pulse.txt"),
            kind="linear",
            fill_value="extrapolate",
        )
        vs1 = gprMax.VoltageSource(
            polarisation="y",
            p1=(tx[0], tx[1], tx[2]),
            resistance=sourceresistance,
            waveform_id="my_pulse",
        )
        scene_objects.extend((exc1, vs1))

    # Receiver
    if src_type == "transmission_line":
        # Zero waveform to use with transmission line at receiver output
        w2 = gprMax.Waveform(wave_type="gaussian", amp=0, freq=excitationfreq, id="my_zero_wave")
        tl2 = gprMax.TransmissionLine(
            polarisation="y",
            p1=(tx[0] + 0.162, tx[1], tx[2]),
            resistance=receiverresistance,
            waveform_id="my_zero_wave",
        )
        scene_objects.extend((w2, tl2))
    else:
        r1 = gprMax.Rx(p1=(tx[0] + 0.162, tx[1], tx[2]), id="rxbowtie", outputs=["Ey"])
        scene_objects.append(r1)

    # Geometry views
    gv1 = gprMax.GeometryView(
        p1=(x - dx, y - dy, z - dz),
        p2=(
            x + casesize[0] + dx,
            y + casesize[1] + dy,
            z + skidthickness + casesize[2] + dz,
        ),
        dl=(dx, dy, dz),
        filename="antenna_like_GSSI_400",
        output_type="n",
    )
    gv2 = gprMax.GeometryView(
        p1=(x, y, z),
        p2=(x + casesize[0], y + casesize[1], z + 0.010),
        dl=(dx, dy, dz),
        filename="antenna_like_GSSI_400_pcb",
        output_type="f",
    )
    # scene_objects.extend((gv1, gv2))

    return scene_objects
