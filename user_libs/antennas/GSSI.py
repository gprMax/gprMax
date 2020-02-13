# Copyright (C) 2015-2020, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import gprMax
from gprMax.exceptions import CmdInputError


def antenna_like_GSSI_1500(x, y, z, resolution=0.001):
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
        raise CmdInputError('This antenna module can only be used with a spatial discretisation of 1mm or 2mm')

    # Specify optimisation state of antenna model
    optstate = ['WarrenThesis', 'DebyeAbsorber', 'GiannakisPaper']
    optstate = optstate[0]

    if optstate == 'WarrenThesis':
        # Original optimised values from http://hdl.handle.net/1842/4074
        excitationfreq = 1.71e9
        sourceresistance = 230  # Correction for old (< 123) GprMax3D bug (optimised to 4)
        rxres = 925  # Resistance at Rx bowtie
        absorber1 = gprMax.Material(er=1.58, se=0.428, mr=1, sm=0, id='absorber1')
        absorber2 = gprMax.Material(er=3, se=0, mr=1, sm=0, id='absorber2') # Foam modelled as PCB material
        pcb = gprMax.Material(er=3, se=0, mr=1, sm=0, id='pcb')
        hdpe = gprMax.Material(er=2.35, se=0, mr=1, sm=0, id='hdpe')
        rxres = gprMax.Material(er=3, se=(1 / rxres) * (dy / (dx * dz)), mr=1, sm=0, id='rxres')
        scene_objects.extend((absorber1, absorber2, pcb, hdpe, rxres))

    elif optstate == 'DebyeAbsorber':
        # Same values as WarrenThesis but uses dispersive absorber properties for Eccosorb LS22
        excitationfreq = 1.71e9
        sourceresistance = 230  # Correction for old (< 123) GprMax3D bug (optimised to 4)
        rxres = 925  # Resistance at Rx bowtie
        absorber1 = gprMax.Material(er=1, se=0, mr=1, sm=0, id='absorber1')
        # Eccosorb LS22 3-pole Debye model (https://bitbucket.org/uoyaeg/aegboxts/wiki/Home)
        absorber1_disp = gprMax.AddDebyeDispersion(poles=3, er_delta=[3.7733, 3.14418, 20.2441],
                                                   tau=[1.00723e-11, 1.55686e-10, 3.44129e-10],
                                                   material_ids=['absorber1'])
        absorber2 = gprMax.Material(er=3, se=0, mr=1, sm=0, id='absorber2') # Foam modelled as PCB material
        pcb = gprMax.Material(er=3, se=0, mr=1, sm=0, id='pcb')
        hdpe = gprMax.Material(er=2.35, se=0, mr=1, sm=0, id='hdpe')
        rxres = gprMax.Material(er=3, se=(1 / rxres) * (dy / (dx * dz)), mr=1, sm=0, id='rxres')
        scene_objects.extend((absorber1, absorber1_disp, absorber2, pcb, hdpe, rxres))

    elif optstate == 'GiannakisPaper':
        # Further optimised values from https://doi.org/10.1109/TGRS.2018.2869027
        sourceresistance = 195
        absorber1 = gprMax.Material(er=3.96, se=0.31, mr=1, sm=0, id='absorber1')
        absorber2 = gprMax.Material(er=1.05, se=1.01, mr=1, sm=0, id='absorber2')
        pcb = gprMax.Material(er=1.37, se=0.0002, mr=1, sm=0, id='pcb')
        hdpe = gprMax.Material(er=1.99, se=0.013, mr=1, sm=0, id='hdpe')
        scene_objects.extend((absorber1, absorber2, pcb, hdpe))

    # Antenna geometry
    # Plastic case
    b1 = gprMax.Box(p1=(x, y, z + skidthickness),
                    p2=(x + casesize[0], y + casesize[1], z + skidthickness + casesize[2]),
                    material_id='hdpe')
    b2 = gprMax.Box(p1=(x + casethickness, y + casethickness, z + skidthickness),
                    p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                        z + skidthickness + casesize[2] - casethickness),
                    material_id='free_space')

    # Metallic enclosure
    b3 = gprMax.Box(p1=(x + 0.025, y + casethickness, z + skidthickness),
                    p2=(x + casesize[0] - 0.025, y + casesize[1] - casethickness,
                        z + skidthickness + 0.027), material_id='pec')

    # Absorber material (absorber1) and foam (absorber2) around edge of absorber
    b4 = gprMax.Box(p1=(x + 0.025 + shieldthickness, y + casethickness + shieldthickness,
                        z + skidthickness), p2=(x + 0.025 + shieldthickness + 0.057,
                        y + casesize[1] - casethickness - shieldthickness,
                        z + skidthickness + 0.027 - shieldthickness - 0.001),
                        material_id='absorber2')
    b5 = gprMax.Box(p1=(x + 0.025 + shieldthickness + foamsurroundthickness,
                        y + casethickness + shieldthickness + foamsurroundthickness,
                        z + skidthickness),
                    p2=(x + 0.025 + shieldthickness + 0.057 - foamsurroundthickness,
                        y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness,
                        z + skidthickness + 0.027 - shieldthickness),
                    material_id='absorber1')
    b6 = gprMax.Box(p1=(x + 0.086, y + casethickness + shieldthickness, z + skidthickness),
                    p2=(x + 0.086 + 0.057, y + casesize[1] - casethickness - shieldthickness,
                        z + skidthickness + 0.027 - shieldthickness - 0.001),
                    material_id='absorber2')
    b7 = gprMax.Box(p1=(x + 0.086 + foamsurroundthickness,
                        y + casethickness + shieldthickness + foamsurroundthickness,
                        z + skidthickness),
                    p2=(x + 0.086 + 0.057 - foamsurroundthickness,
                        y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness,
                        z + skidthickness + 0.027 - shieldthickness),
                    material_id='absorber1')

    # PCB
    b8 = gprMax.Box(p1=(x + 0.025 + shieldthickness + foamsurroundthickness,
                        y + casethickness + shieldthickness + foamsurroundthickness,
                        z + skidthickness),
                    p2=(x + 0.086 - shieldthickness - foamsurroundthickness,
                        y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness,
                        z + skidthickness + pcbthickness), material_id='pcb')
    b9 = gprMax.Box(p1=(x + 0.086 + foamsurroundthickness,
                        y + casethickness + shieldthickness + foamsurroundthickness,
                        z + skidthickness),
                    p2=(x + 0.086 + 0.057 - foamsurroundthickness,
                        y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness,
                        z + skidthickness + pcbthickness), material_id='pcb')

    scene_objects.extend((b1, b2, b3, b4, b5, b6, b7, b8, b9))

    # PCB components
    if resolution == 0.001:
        # Rx & Tx bowties
        a = 0
        b = 0
        while b < 13:
            p1 = gprMax.Plate(p1=(x + 0.045 + a * dx, y + 0.039 + b * dx, z + skidthickness),
                              p2=(x + 0.065 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness),
                              material_id='pec')
            p2 = gprMax.Plate(p1=(x + 0.045 + a * dx, y + 0.067 - b * dx, z + skidthickness),
                              p2=(x + 0.065 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness),
                              material_id='pec')
            p3 = gprMax.Plate(p1=(x + 0.104 + a * dx, y + 0.039 + b * dx, z + skidthickness),
                              p2=(x + 0.124 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness),
                              material_id='pec')
            p4 = gprMax.Plate(p1=(x + 0.104 + a * dx, y + 0.067 - b * dx, z + skidthickness),
                              p2=(x + 0.124 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness),
                              material_id='pec')
            scene_objects.extend((p1, p2, p3, p4))
            b += 1
            if a == 2 or a == 4 or a == 7:
                p5 = gprMax.Plate(p1=(x + 0.045 + a * dx, y + 0.039 + b * dx, z + skidthickness),
                                  p2=(x + 0.065 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness),
                                  material_id='pec')
                p6 = gprMax.Plate(p1=(x + 0.045 + a * dx, y + 0.067 - b * dx, z + skidthickness),
                                  p2=(x + 0.065 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness),
                                  material_id='pec')
                p7 = gprMax.Plate(p1=(x + 0.104 + a * dx, y + 0.039 + b * dx, z + skidthickness),
                                  p2=(x + 0.124 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness),
                                  material_id='pec')
                p8 = gprMax.Plate(p1=(x + 0.104 + a * dx, y + 0.067 - b * dx, z + skidthickness),
                                  p2=(x + 0.124 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness),
                                  material_id='pec')
                b += 1
                scene_objects.extend((p5, p6, p7, p8))
            a += 1
        # Rx extension section (upper y)
        p9 = gprMax.Plate(p1=(x + 0.044, y + 0.068, z + skidthickness),
                          p2=(x + 0.044 + bowtiebase, y + 0.068 + patchheight, z + skidthickness),
                          material_id='pec')
        # Tx extension section (upper y)
        p10 = gprMax.Plate(p1=(x + 0.103, y + 0.068, z + skidthickness),
                           p2=(x + 0.103 + bowtiebase, y + 0.068 + patchheight, z + skidthickness),
                           material_id='pec')
        scene_objects.extend((p9, p10))

        # Edges that represent wire between bowtie halves in 1mm model
        e1 = gprMax.Edge(p1=(tx[0] - 0.059, tx[1] - dy, tx[2]),
                         p2=(tx[0] - 0.059, tx[1], tx[2]), material_id='pec')
        e2 = gprMax.Edge(p1=(tx[0] - 0.059, tx[1] + dy, tx[2]),
                         p2=(tx[0] - 0.059, tx[1] + 0.002, tx[2]), material_id='pec')
        e3 = gprMax.Edge(p1=(tx[0], tx[1] - dy, tx[2]),
                         p2=(tx[0], tx[1], tx[2]), material_id='pec')
        e4 = gprMax.Edge(p1=(tx[0], tx[1] + dz, tx[2]),
                         p2=(tx[0], tx[1] + 0.002, tx[2]), material_id='pec')
        scene_objects.extend((e1, e2, e3, e4))

    elif resolution == 0.002:
        # Rx & Tx bowties
        for a in range(0, 6):
            p1 = gprMax.Plate(p1=(x + 0.044 + a * dx, y + 0.040 + a * dx, z + skidthickness),
                              p2=(x + 0.066 - a * dx, y + 0.040 + a * dx + dy, z + skidthickness),
                              material_id='pec')
            p2 = gprMax.Plate(p1=(x + 0.044 + a * dx, y + 0.064 - a * dx, z + skidthickness),
                              p2=(x + 0.066 - a * dx, y + 0.064 - a * dx + dy, z + skidthickness),
                              material_id='pec')
            p3 = gprMax.Plate(p1=(x + 0.103 + a * dx, y + 0.040 + a * dx, z + skidthickness),
                              p2=(x + 0.125 - a * dx, y + 0.040 + a * dx + dy, z + skidthickness),
                              material_id='pec')
            p4 = gprMax.Plate(p1=(x + 0.103 + a * dx, y + 0.064 - a * dx, z + skidthickness),
                              p2=(x + 0.125 - a * dx, y + 0.064 - a * dx + dy, z + skidthickness),
                              material_id='pec')
            # Rx extension section (upper y)
            p5 = gprMax.Plate(p1=(x + 0.044, y + 0.066, z + skidthickness),
                              p2=(x + 0.044 + bowtiebase, y + 0.066 + patchheight, z + skidthickness),
                              material_id='pec')
            # Tx extension section (upper y)
            p6 = gprMax.Plate(p1=(x + 0.103, y + 0.066, z + skidthickness),
                              p2=(x + 0.103 + bowtiebase, y + 0.066 + patchheight, z + skidthickness),
                              material_id='pec')
            scene_objects.extend((p1, p2, p3, p4, p5, p6))

    # Rx extension section (lower y)
    p11 = gprMax.Plate(p1=(x + 0.044, y + 0.024, z + skidthickness),
                       p2=(x + 0.044 + bowtiebase, y + 0.024 + patchheight, z + skidthickness),
                       material_id='pec')
    # Tx extension section (lower y)
    p12 = gprMax.Plate(p1=(x + 0.103, y + 0.024, z + skidthickness),
                       p2=(x + 0.103 + bowtiebase, y + 0.024 + patchheight, z + skidthickness),
                       material_id='pec')
    scene_objects.extend((p11, p12))

    # Skid
    b10 = gprMax.Box(p1=(x, y, z), p2=(x + casesize[0], y + casesize[1], z + skidthickness),
                     material_id='hdpe')
    scene_objects.append(b10)

    # Geometry views
    gv1 = gprMax.GeometryView(p1=(x - dx, y - dy, z - dz), p2=(x + casesize[0] + dx,
                              y + casesize[1] + dy, z + skidthickness + casesize[2] + dz),
                              dl=(dx, dy, dz), filename='antenna_like_GSSI_1500',
                              output_type='n')
    gv2 = gprMax.GeometryView(p1=(x, y, z), p2=(x + casesize[0], y + casesize[1], z + 0.010),
                              dl=(dx, dy, dz), filename='antenna_like_GSSI_1500_pcb',
                              output_type='f')
    # scene_objects.extend((gv1, gv2))

    # Excitation
    if optstate == 'WarrenThesis' or optstate == 'DebyeAbsorber':
        # Gaussian pulse
        w1 = gprMax.Waveform(wave_type='gaussian', amp=1, freq=excitationfreq, id='my_gaussian')
        vs1 = gprMax.VoltageSource(polarisation='y', p1=(tx[0], tx[1], tx[2]), resistance=sourceresistance, waveform_id='my_gaussian')
        scene_objects.extend((w1, vs1))

    elif optstate == 'GiannakisPaper':
        # Optimised custom pulse
        exc1 = gprMax.ExcitationFile(filepath='../user_libs/antennas/GSSI1p5optpulse.txt', kind='linear', fill_value='extrapolate')
        vs1 = gprMax.VoltageSource(polarisation='y', p1=(tx[0], tx[1], tx[2]), resistance=sourceresistance, waveform_id='GSSI1p5optpulse')
        scene_objects.extend((exc1, vs1))

    # Output point - receiver bowtie
    if resolution == 0.001:
        if optstate == 'WarrenThesis' or optstate == 'DebyeAbsorber':
            e1 = gprMax.Edge(p1=(tx[0] - 0.059, tx[1], tx[2]),
                             p2=(tx[0] - 0.059, tx[1] + dy, tx[2]),
                             material_id='rxres')
            scene_objects.append(e1)
        r1 = gprMax.Rx(p1=(tx[0] - 0.059, tx[1], tx[2]), id='rxbowtie',
                       outputs='Ey')
        scene_objects.append(r1)

    elif resolution == 0.002:
        if optstate == 'WarrenThesis' or optstate == 'DebyeAbsorber':
            e1 = gprMax.Edge(p1=(tx[0] - 0.060, tx[1], tx[2]),
                             p2=(tx[0] - 0.060, tx[1] + dy, tx[2]),
                             material_id='rxres')
            scene_objects.append(e1)
        r1 = gprMax.Rx(p1=(tx[0] - 0.060, tx[1], tx[2]), id='rxbowtie',
                       outputs='Ey')
        scene_objects.append(r1)

    return scene_objects


def antenna_like_GSSI_400(x, y, z, resolution=0.001):
    """Inserts a description of an antenna similar to the GSSI 400MHz antenna.
        Can be used with 0.5mm, 1mm (default) or 2mm spatial resolution.
        The external dimensions of the antenna are 300x300x178mm.
        One output point is defined between the arms of the receiver bowtie.
        The bowties are aligned with the y axis so the output is the y component
        of the electric field.

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.

    Returns:
        scene_objects (list): All model objects that will be part of a scene.
    """

    # All model objects that will be returned by function
    scene_objects = []

    # Antenna geometry properties
    casesize = (0.3, 0.3, 0.178) # original
    casethickness = 0.002
    shieldthickness = 0.002
    foamsurroundthickness = 0.003
    pcbthickness = 0.002
    bowtiebase = 0.06
    bowtieheight = 0.06 # original 0.056
    patchheight = 0.06 # original 0.056
    metalboxheight = 0.089
    metalmiddleplateheight = 0.11

    smooth_dec = 'yes' # choose to use dielectric smoothing or not
    src_type = 'voltage_source' # # source type. "voltage_source" or "transmission_line"
    excitationfreq = 0.39239891e9 # GHz
    sourceresistance = 111.59927 # Ohms
    receiverresistance = sourceresistance # Ohms
    absorberEr = 1.1
    absorbersig = 0.062034689
    pcber = 2.35
    hdper = 2.35
    skidthickness = 0.01

    x = x - (casesize[0] / 2)
    y = y - (casesize[1] / 2)

    # Coordinates of source excitation point in antenna
    tx = x + 0.01 + 0.005 + 0.056, y + casethickness + 0.005 + 0.143, z + skidthickness

    if resolution == 0.0005:
        dx = 0.0005
        dy = 0.0005
        dz = 0.0005
        tx = x + 0.01 + 0.005 + 0.056, y + casethickness + 0.005 + 0.1435, z + skidthickness
    elif resolution == 0.001:
        dx = 0.001
        dy = 0.001
        dz = 0.001
    elif resolution == 0.002:
        dx = 0.002
        dy = 0.002
        dz = 0.002
        foamsurroundthickness = 0.002
        metalboxheight = 0.088
        tx = x + 0.01 + 0.004 + 0.056, y + casethickness + 0.005 + 0.143 - 0.002, z + skidthickness
    else:
        raise CmdInputError('This antenna module can only be used with a spatial discretisation of 0.5mm, 1mm, 2mm')

    # Material definitions
    absorber = gprMax.Material(er=absorberEr, se=absorbersig, mr=1, sm=0, id='absorber')
    pcb = gprMax.Material(er=pcber, se=0, mr=1, sm=0, id='pcb')
    hdpe = gprMax.Material(er=hdper, se=0, mr=1, sm=0, id='hdpe')
    scene_objects.extend((absorber, pcb, hdpe))

    # Antenna geometry
    if smooth_dec == 'yes':
        # Plastic case
        b1 = gprMax.Box(p1=(x, y, z + skidthickness - 0.002),
                        p2=(x + casesize[0], y + casesize[1], z + casesize[2]),
                        material_id='hdpe')
        b2 = gprMax.Box(p1=(x + casethickness, y + casethickness, z + skidthickness - 0.002),
                        p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                        z + casesize[2] - casethickness), material_id='free_space')

        # Metallic enclosure
        b3 = gprMax.Box(p1=(x + casethickness, y + casethickness, z + skidthickness +
                        (metalmiddleplateheight - metalboxheight)),
                        p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                        z + skidthickness + (metalmiddleplateheight - metalboxheight) +
                        metalboxheight), material_id='pec')

        # Absorber, and foam (modelled as PCB material) around edge of absorber
        b4 = gprMax.Box(p1=(x + casethickness, y + casethickness, z + skidthickness),
                        p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                        z + skidthickness + (metalmiddleplateheight - metalboxheight)),
                        material_id='absorber')
        b5 = gprMax.Box(p1=(x + casethickness + shieldthickness, y + casethickness +
                        shieldthickness, z + skidthickness + (metalmiddleplateheight - metalboxheight)),
                        p2=(x + casesize[0] - casethickness - shieldthickness,
                        y + casesize[1] - casethickness - shieldthickness,
                        z + skidthickness - shieldthickness + metalmiddleplateheight),
                        material_id='absorber')
        scene_objects.extend((b1, b2, b3, b4, b5))

        # PCB
        if resolution == 0.0005:
            b6 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.018, y + casethickness +
                            0.005 + 0.0235, z + skidthickness),
                            p2=(x + 0.01 + 0.005 + 0.034 + bowtiebase,
                            y + casethickness + 0.005 + 0.204 + patchheight,
                            z + skidthickness + pcbthickness), material_id='pcb')
            b7 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.178, y + casethickness +
                            0.005 + 0.0235, z + skidthickness),
                            p2=(x + 0.01 + 0.005 + 0.194 + bowtiebase,
                            y + casethickness + 0.005 + 0.204 + patchheight,
                            z + skidthickness + pcbthickness), material_id='pcb')
            scene_objects.extend((b6, b7))

        elif resolution == 0.001:
            b6 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.018, y + casethickness + 0.005 + 0.023,
                            z + skidthickness), p2=(x + 0.01 + 0.005 + 0.034 + bowtiebase,
                            y + casethickness + 0.005 + 0.204 + patchheight,
                            z + skidthickness + pcbthickness), material_id='pcb')
            b7 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.178, y + casethickness + 0.005 + 0.023,
                            z + skidthickness), p2=(x + 0.01 + 0.005 + 0.194 + bowtiebase,
                            y + casethickness + 0.005 + 0.204 + patchheight,
                            z + skidthickness + pcbthickness), material_id='pcb')
            scene_objects.extend((b6, b7))

        elif resolution == 0.002:
            b6 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.017, y + casethickness + 0.005 + 0.021,
                            z + skidthickness), p2=(x + 0.01 + 0.005 + 0.033 + bowtiebase,
                            y + casethickness + 0.006 + 0.202 + patchheight,
                            z + skidthickness + pcbthickness), material_id='pcb')
            b7 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.179, y + casethickness + 0.005 + 0.021,
                            z + skidthickness), p2=(x + 0.01 + 0.005 + 0.195 + bowtiebase,
                            y + casethickness + 0.006 + 0.202 + patchheight,
                            z + skidthickness + pcbthickness), material_id='pcb')
            scene_objects.extend((b6, b7))

    elif smooth_dec == 'no':
        # Plastic case
        b8 = gprMax.Box(p1=(x, y, z + skidthickness - 0.002),
                        p2=(x + casesize[0], y + casesize[1], z + casesize[2]),
                        material_id='hdpe', averaging='n')
        b9 = gprMax.Box(p1=(x + casethickness, y + casethickness, z + skidthickness - 0.002),
                        p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                        z + casesize[2] - casethickness), material_id='free_space',
                        averaging='n')

        # Metallic enclosure
        b10 = gprMax.Box(p1=(x + casethickness, y + casethickness,
                         z + skidthickness + (metalmiddleplateheight - metalboxheight)),
                         p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                         z + skidthickness + (metalmiddleplateheight - metalboxheight) +
                         metalboxheight), material_id='pec')

        # Absorber, and foam (modelled as PCB material) around edge of absorber
        b11 = gprMax.Box(p1=(x + casethickness, y + casethickness, z + skidthickness),
                         p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                         z + skidthickness + (metalmiddleplateheight - metalboxheight)),
                         material_id='absorber', averaging='n')
        b12 = gprMax.Box(p1=(x + casethickness + shieldthickness,
                         y + casethickness + shieldthickness,
                         z + skidthickness + (metalmiddleplateheight - metalboxheight)),
                         p2=(x + casesize[0] - casethickness - shieldthickness,
                         y + casesize[1] - casethickness - shieldthickness,
                         z + skidthickness - shieldthickness + metalmiddleplateheight),
                         material_id='absorber', averaging='n')
        scene_objects.extend((b8, b9, b10, b11, b12))

        # PCB
        if resolution == 0.0005:
            b13 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.018,
                             y + casethickness + 0.005 + 0.0235, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.034 + bowtiebase,
                             y + casethickness + 0.005 + 0.204 + patchheight,
                             z + skidthickness + pcbthickness), material_id='pcb',
                             averaging='n')
            b14 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.178,
                             y + casethickness + 0.005 + 0.0235, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.194 + bowtiebase,
                             y + casethickness + 0.005 + 0.204 + patchheight,
                             z + skidthickness + pcbthickness), material_id='pcb',
                             averaging='n')
            scene_objects.extend((b13, b14))

        elif resolution == 0.001:
            b13 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.018,
                             y + casethickness + 0.005 + 0.023, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.034 + bowtiebase,
                             y + casethickness + 0.005 + 0.204 + patchheight,
                             z + skidthickness + pcbthickness), material_id='pcb',
                             averaging='n')
            b14 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.178,
                             y + casethickness + 0.005 + 0.023, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.194 + bowtiebase,
                             y + casethickness + 0.005 + 0.204 + patchheight,
                             z + skidthickness + pcbthickness), material_id='pcb',
                             averaging='n')
            scene_objects.extend((b13, b14))

        elif resolution == 0.002:
            b13 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.017,
                             y + casethickness + 0.005 + 0.021, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.033 + bowtiebase,
                             y + casethickness + 0.006 + 0.202 + patchheight,
                             z + skidthickness + pcbthickness), material_id='pcb',
                             averaging='n')
            b14 = gprMax.Box(p1=(x + 0.01 + 0.005 + 0.179,
                             y + casethickness + 0.005 + 0.021, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.195 + bowtiebase,
                             y + casethickness + 0.006 + 0.202 + patchheight,
                             z + skidthickness + pcbthickness), material_id='pcb',
                             averaging='n')
            scene_objects.extend((b13, b14))

    # PCB components
    # My own bowties with triangle commands
    if resolution == 0.0005:
        # "left" side
        # extension plates
        p1 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.026,
                          y + casethickness + 0.005 + 0.0235, z + skidthickness),
                          p2=(x + 0.01 + 0.005 + 0.026 + bowtiebase,
                          y + casethickness + 0.005 + 0.0235 + patchheight,
                          z + skidthickness), material_id='pec')
        p2 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.026,
                          y + casethickness + 0.005 + 0.204, z + skidthickness),
                          p2=(x + 0.01 + 0.005 + 0.026 + bowtiebase,
                          y + casethickness + 0.005 + 0.204 + patchheight,
                          z + skidthickness), material_id='pec')
        # triangles
        t1 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.026,
                             y + casethickness + 0.005 + 0.0835, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.026 + bowtiebase,
                             y + casethickness + 0.005 + 0.0835, z + skidthickness),
                             p3=(x + 0.01 + 0.005 + 0.026 + (bowtiebase/2),
                             y + casethickness + 0.005 + 0.0835 + bowtieheight,
                             z + skidthickness), thickness=0, material_id='pec')
        t2 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.026,
                             y + casethickness + 0.005 + 0.204, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.026 + bowtiebase,
                             y + casethickness + 0.005 + 0.204, z + skidthickness),
                             p3=(x + 0.01 + 0.005 + 0.026 + (bowtiebase/2),
                             y + casethickness + 0.005 + 0.204 - bowtieheight,
                             z + skidthickness), thickness=0, material_id='pec')

        # "right" side
        p3 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.186,
                          y + casethickness + 0.005 + 0.0235, z + skidthickness),
                          p2=(x + 0.01 + 0.005 + 0.186 + bowtiebase,
                          y + casethickness + 0.005 + 0.0235 + patchheight,
                          z + skidthickness), material_id='pec')
        p4 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.186,
                          y + casethickness + 0.005 + 0.204, z + skidthickness),
                          p2=(x + 0.01 + 0.005 + 0.186 + bowtiebase,
                          y + casethickness + 0.005 + 0.204 + patchheight,
                          z + skidthickness), material_id='pec')
        # triangles
        t3 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.186,
                             y + casethickness + 0.005 + 0.0835, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.186 + bowtiebase,
                             y + casethickness + 0.005 + 0.0835, z + skidthickness),
                             p3=(x + 0.01 + 0.005 + 0.186 + (bowtiebase/2),
                             y + casethickness + 0.005 + 0.0835 + bowtieheight,
                             z + skidthickness), thickness=0, material_ID='pec')
        t4 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.186,
                             y + casethickness + 0.005 + 0.204, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.186 + bowtiebase,
                             y + casethickness + 0.005 + 0.204, z + skidthickness),
                             p3=(x + 0.01 + 0.005 + 0.186 + (bowtiebase/2),
                             y + casethickness + 0.005 + 0.204 - bowtieheight,
                             z + skidthickness), thickness=0, material_id='pec')

        # Edges that represent wire between bowtie halves in 1mm model
        e1 = gprMax.Edge(p1=(tx[0] + 0.16, tx[1] - dy, tx[2]),
                         p2=(tx[0] + 0.16, tx[1], tx[2]), material_id='pec')
        e2 = gprMax.Edge(p1=(tx[0] + 0.16, tx[1] + dy, tx[2]),
                         p2=(tx[0] + 0.16, tx[1] + 2*dy, tx[2]), material_id='pec')
        e3 = gprMax.Edge(p1=(tx[0], tx[1] - dy, tx[2]),
                         p2=(tx[0], tx[1], tx[2]), material_id='pec')
        e4 = gprMax.Edge(p1=(tx[0], tx[1] + dy, tx[2]),
                         p2=(tx[0], tx[1] + 2*dy, tx[2]), material_id='pec')
        scene_objects.extend((p1, p2, t1, t2, p3, p4, t3, t4, e1, e2, e3, e4))

    elif resolution == 0.001:
        # "left" side
        # extension plates
        p1 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.026,
                          y + casethickness + 0.005 + 0.023, z + skidthickness),
                          p2=(x + 0.01 + 0.005 + 0.026 + bowtiebase,
                          y + casethickness + 0.005 + 0.023 + patchheight,
                          z + skidthickness), material_id='pec')
        p2 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.026,
                          y + casethickness + 0.005 + 0.204, z + skidthickness),
                          p2=(x + 0.01 + 0.005 + 0.026 + bowtiebase,
                          y + casethickness + 0.005 + 0.204 + patchheight,
                          z + skidthickness), material_id='pec')
        # triangles
        t1 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.026,
                             y + casethickness + 0.005 + 0.083, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.026 + bowtiebase,
                             y + casethickness + 0.005 + 0.083, z + skidthickness),
                             p3=(x + 0.01 + 0.005 + 0.026 + (bowtiebase/2),
                             y + casethickness + 0.005 + 0.083 + bowtieheight,
                             z + skidthickness), thickness=0, material_id='pec')
        t2 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.026,
                             y + casethickness + 0.005 + 0.204, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.026 + bowtiebase,
                             y + casethickness + 0.005 + 0.204, z + skidthickness),
                             p3=(x + 0.01 + 0.005 + 0.026 + (bowtiebase/2),
                             y + casethickness + 0.005 + 0.204 - bowtieheight,
                             z + skidthickness), thickness=0, material_id='pec')

        # "right" side
        p3 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.186,
                          y + casethickness + 0.005 + 0.023, z + skidthickness),
                          p2=(x + 0.01 + 0.005 + 0.186 + bowtiebase,
                          y + casethickness + 0.005 + 0.023 + patchheight,
                          z + skidthickness), material_id='pec')
        p4 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.186,
                          y + casethickness + 0.005 + 0.204, z + skidthickness),
                          p2=(x + 0.01 + 0.005 + 0.186 + bowtiebase,
                          y + casethickness + 0.005 + 0.204 + patchheight,
                          z + skidthickness), material_id='pec')
        # triangles
        t3 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.186,
                             y + casethickness + 0.005 + 0.083, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.186 + bowtiebase,
                             y + casethickness + 0.005 + 0.083, z + skidthickness),
                             p3=(x + 0.01 + 0.005 + 0.186 + (bowtiebase/2),
                             y + casethickness + 0.005 + 0.083 + bowtieheight,
                             z + skidthickness), thickness=0, material_id='pec')
        t4 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.186,
                             y + casethickness + 0.005 + 0.204, z + skidthickness),
                             p2=(x + 0.01 + 0.005 + 0.186 + bowtiebase,
                             y + casethickness + 0.005 + 0.204, z + skidthickness),
                             p3=(x + 0.01 + 0.005 + 0.186 + (bowtiebase/2),
                             y + casethickness + 0.005 + 0.204 - bowtieheight,
                             z + skidthickness), thickness=0, material_id='pec')

        # Edges that represent wire between bowtie halves in 1mm model
        e1 = gprMax.Edge(p1=(tx[0] + 0.16, tx[1] - dy, tx[2]),
                         p2=(tx[0] + 0.16, tx[1], tx[2]), material_id='pec')
        e2 = gprMax.Edge(p1=(tx[0] + 0.16, tx[1] + dy, tx[2]),
                         p2=(tx[0] + 0.16, tx[1] + 2*dy, tx[2]), material_id='pec')
        e3 = gprMax.Edge(p1=(tx[0], tx[1] - dy, tx[2]),
                         p2=(tx[0], tx[1], tx[2]), material_id='pec')
        e4 = gprMax.Edge(p1=(tx[0], tx[1] + dy, tx[2]),
                         p2=(tx[0], tx[1] + 2*dy, tx[2]), material_id='pec')
        scene_objects.extend((p1, p2, t1, t2, p3, p4, t3, t4, e1, e2, e3, e4))

    elif resolution == 0.002:
            # "left" side
            # extension plates
            p1 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.025,
                              y + casethickness + 0.005 + 0.021, z + skidthickness),
                              p2=(x + 0.01 + 0.005 + 0.025 + bowtiebase,
                              y + casethickness + 0.005 + 0.021 + patchheight,
                              z + skidthickness), material_id='pec')
            p2 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.025,
                              y + casethickness + 0.005 + 0.203, z + skidthickness),
                              p2=(x + 0.01 + 0.005 + 0.025 + bowtiebase,
                              y + casethickness + 0.005 + 0.203 + patchheight,
                              z + skidthickness), material_id='pec')
            # triangles
            t1 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.025,
                                 y + casethickness + 0.005 + 0.081, z + skidthickness),
                                 p2=(x + 0.01 + 0.005 + 0.025 + bowtiebase,
                                 y + casethickness + 0.005 + 0.081, z + skidthickness),
                                 p3=(x + 0.01 + 0.005 + 0.025 + (bowtiebase/2),
                                 y + casethickness + 0.005 + 0.081 + bowtieheight,
                                 z + skidthickness), thickness=0, material_id='pec')
            t2 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.025,
                                 y + casethickness + 0.005 + 0.203, z + skidthickness),
                                 p2=(x + 0.01 + 0.005 + 0.025 + bowtiebase,
                                 y + casethickness + 0.005 + 0.203, z + skidthickness),
                                 p3=(x + 0.01 + 0.005 + 0.025 + (bowtiebase/2),
                                 y + casethickness + 0.005 + 0.203 - bowtieheight,
                                 z + skidthickness), thickness=0, material_id='pec')
            # "right" side
            p3 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.187,
                              y + casethickness + 0.005 + 0.021, z + skidthickness),
                              p2=(x + 0.01 + 0.005 + 0.187 + bowtiebase,
                              y + casethickness + 0.005 + 0.021 + patchheight,
                              z + skidthickness), material_id='pec')
            p4 = gprMax.Plate(p1=(x + 0.01 + 0.005 + 0.187,
                              y + casethickness + 0.005 + 0.203, z + skidthickness),
                              p2=(x + 0.01 + 0.005 + 0.187 + bowtiebase,
                              y + casethickness + 0.005 + 0.203 + patchheight,
                              z + skidthickness), material_id='pec')
            # triangles
            t3 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.187,
                                 y + casethickness + 0.005 + 0.081, z + skidthickness),
                                 p2=(x + 0.01 + 0.005 + 0.187 + bowtiebase,
                                 y + casethickness + 0.005 + 0.081, z + skidthickness),
                                 p3=(x + 0.01 + 0.005 + 0.187 + (bowtiebase/2),
                                 y + casethickness + 0.005 + 0.081 + bowtieheight,
                                 z + skidthickness), thickness=0, material_id='pec')
            t4 = gprMax.Triangle(p1=(x + 0.01 + 0.005 + 0.187,
                                 y + casethickness + 0.005 + 0.203, z + skidthickness),
                                 p2=(x + 0.01 + 0.005 + 0.187 + bowtiebase,
                                 y + casethickness + 0.005 + 0.203, z + skidthickness),
                                 p3=(x + 0.01 + 0.005 + 0.187 + (bowtiebase/2),
                                 y + casethickness + 0.005 + 0.203 - bowtieheight,
                                 z + skidthickness), thickness=0, material_id='pec')

            # Edges that represent wire between bowtie halves in 2mm model
            e1 = gprMax.Edge(p1=(tx[0] + 0.162, tx[1] - dy, tx[2]),
                             p2=(tx[0] + 0.162, tx[1], tx[2]), material_id='pec')
            e2 = gprMax.Edge(p1=(tx[0] + 0.162, tx[1] + dy, tx[2]),
                             p2=(tx[0] + 0.162, tx[1] + 2*dy, tx[2]), material_id='pec')
            e3 = gprMax.Edge(p1=(tx[0], tx[1] - dy, tx[2]),
                             p2=(tx[0], tx[1], tx[2]), material_id='pec')
            e4 = gprMax.Edge(p1=(tx[0], tx[1] + dy, tx[2]),
                             p2=(tx[0], tx[1] + 2*dy, tx[2]), material_id='pec')
            scene_objects.extend((p1, p2, t1, t2, p3, p4, t3, t4, e1, e2, e3, e4))

    # Metallic plate extension
    b15 = gprMax.Box(p1=(x + (casesize[0] / 2), y + casethickness, z + skidthickness),
                     p2=(x + (casesize[0] / 2) + shieldthickness,
                     y + casesize[1] - casethickness, z + skidthickness + metalmiddleplateheight),
                     material_id='pec')

    # Skid
    if smooth_dec == 'yes':
        b16 = gprMax.Box(p1=(x, y, z),
                         p2=(x + casesize[0], y + casesize[1], z + skidthickness - 0.002),
                         material_id='hdpe')
    elif smooth_dec == 'no':
        b16 = gprMax.Box(p1=(x, y, z),
                         p2=(x + casesize[0], y + casesize[1], z + skidthickness - 0.002),
                         material_id='hdpe', averaging='n')
    scene_objects.extend((b15, b16))

    # Source
    # Excitation - Gaussian pulse
    w1 = gprMax.Waveform(wave_type='gaussian', amp=1, freq=excitationfreq, id='my_gaussian')
    scene_objects.append(w1)

    if src_type == 'voltage_source':
        vs1 = gprMax.VoltageSource(polarisation='y', p1=(tx[0], tx[1], tx[2]),
                                   resistance=sourceresistance, waveform_id='my_gaussian')
        scene_objects.append(vs1)
    elif src_type == 'transmission_line':
        tl1 = gprMax.TransmissionLine(polarisation='y', p1=(tx[0], tx[1], tx[2]),
                                      resistance=sourceresistance, waveform_id='my_gaussian')
        scene_objects.append(tl1)

    # Receiver
    # Zero waveform to use with transmission line at receiver output
    w2 = gprMax.Waveform(wave_type='gaussian', amp=0, freq=excitationfreq, id='my_zero_wave')
    scene_objects.append(w2)

    if resolution == 0.001 or resolution == 0.0005:
        if src_type == 'transmission_line':
            tl2 = gprMax.TransmissionLine(polarisation='y', p1=(tx[0] + 0.16, tx[1], tx[2]),
                                          resistance=receiverresistance, waveform_id='my_zero_wave')
            scene_objects.append(tl2)
        elif src_type == 'voltage_source':
            r1 = gprMax.Rx(p1=(tx[0] + 0.16, tx[1], tx[2]), id='rxbowtie', outputs='Ey')
            scene_objects.append(r1)

    elif resolution == 0.002:
        if src_type == 'transmission_line':
            tl2 = gprMax.TransmissionLine(polarisation='y', p1=(tx[0] + 0.162, tx[1], tx[2]),
                                          resistance=receiverresistance, waveform_id='my_zero_wave')
            scene_objects.append(tl2)
        elif src_type == 'voltage_source':
            r1 = gprMax.Rx(p1=(tx[0] + 0.162, tx[1], tx[2]), id='rxbowtie', outputs='Ey')
            scene_objects.append(r1)

    # Geometry views
    gv1 = gprMax.GeometryView(p1=(x - dx, y - dy, z - dz), p2=(x + casesize[0] + dx,
                              y + casesize[1] + dy, z + skidthickness + casesize[2] + dz),
                              dl=(dx, dy, dz), filename='antenna_like_GSSI_400',
                              output_type='n')
    gv2 = gprMax.GeometryView(p1=(x, y, z), p2=(x + casesize[0], y + casesize[1], z + 0.010),
                              dl=(dx, dy, dz), filename='antenna_like_GSSI_400_pcb',
                              output_type='f')
    # scene_objects.extend((gv1, gv2))

    return scene_objects
