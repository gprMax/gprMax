# Copyright (C) 2015-2023, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506


from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import *


def antenna_like_GSSI_1500(x, y, z, resolution=0.001, rotate90=False):
    """Inserts a description of an antenna similar to the GSSI 1.5GHz antenna. Can be used with 1mm (default) or 2mm spatial resolution. The external dimensions of the antenna are 170x108x45mm. One output point is defined between the arms of the receiver bowtie. The bowties are aligned with the y axis so the output is the y component of the electric field (x component if the antenna is rotated 90 degrees).

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
        rotate90 (bool): Rotate model 90 degrees CCW in xy plane.
    """

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

    # Set origin for rotation to geometric centre of antenna in x-y plane if required, and set output component for receiver
    if rotate90:
        rotate90origin = (x, y)
        output = 'Ex'
    else:
        rotate90origin = ()
        output = 'Ey'

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
        material(1.58, 0.428, 1, 0, 'absorber1')
        material(3, 0, 1, 0, 'absorber2') # Foam modelled as PCB material
        material(3, 0, 1, 0, 'pcb')
        material(2.35, 0, 1, 0, 'hdpe')
        material(3, (1 / rxres) * (dy / (dx * dz)), 1, 0, 'rxres')

    elif optstate == 'DebyeAbsorber':
        # Same values as WarrenThesis but uses dispersive absorber properties for Eccosorb LS22
        excitationfreq = 1.71e9
        sourceresistance = 230  # Correction for old (< 123) GprMax3D bug (optimised to 4)
        rxres = 925  # Resistance at Rx bowtie
        material(1, 0, 1, 0, 'absorber1')
        print('#add_dispersion_debye: 3 3.7733 1.00723e-11 3.14418 1.55686e-10 20.2441 3.44129e-10 absorber1') # Eccosorb LS22 3-pole Debye model (https://bitbucket.org/uoyaeg/aegboxts/wiki/Home)
        material(3, 0, 1, 0, 'absorber2') # Foam modelled as PCB material
        material(3, 0, 1, 0, 'pcb')
        material(2.35, 0, 1, 0, 'hdpe')
        material(3, (1 / rxres) * (dy / (dx * dz)), 1, 0, 'rxres')

    elif optstate == 'GiannakisPaper':
        # Further optimised values from https://doi.org/10.1109/TGRS.2018.2869027
        sourceresistance = 195
        material(3.96, 0.31, 1, 0, 'absorber1')
        material(1.05, 1.01, 1, 0, 'absorber2')
        material(1.37, 0.0002, 1, 0, 'pcb')
        material(1.99, 0.013, 1, 0, 'hdpe')

    # Antenna geometry
    # Plastic case
    box(x, y, z + skidthickness, x + casesize[0], y + casesize[1], z + skidthickness + casesize[2], 'hdpe', rotate90origin=rotate90origin)
    box(x + casethickness, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness, 'free_space', rotate90origin=rotate90origin)

    # Metallic enclosure
    box(x + 0.025, y + casethickness, z + skidthickness, x + casesize[0] - 0.025, y + casesize[1] - casethickness, z + skidthickness + 0.027, 'pec', rotate90origin=rotate90origin)

    # Absorber material (absorber1) and foam (absorber2) around edge of absorber
    box(x + 0.025 + shieldthickness, y + casethickness + shieldthickness, z + skidthickness, x + 0.025 + shieldthickness + 0.057, y + casesize[1] - casethickness - shieldthickness, z + skidthickness + 0.027 - shieldthickness - 0.001, 'absorber2', rotate90origin=rotate90origin)
    box(x + 0.025 + shieldthickness + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.025 + shieldthickness + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + 0.027 - shieldthickness, 'absorber1', rotate90origin=rotate90origin)
    box(x + 0.086, y + casethickness + shieldthickness, z + skidthickness, x + 0.086 + 0.057, y + casesize[1] - casethickness - shieldthickness, z + skidthickness + 0.027 - shieldthickness - 0.001, 'absorber2', rotate90origin=rotate90origin)
    box(x + 0.086 + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + 0.027 - shieldthickness, 'absorber1', rotate90origin=rotate90origin)

    # PCB
    box(x + 0.025 + shieldthickness + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 - shieldthickness - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)
    box(x + 0.086 + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)

    # PCB components
    if resolution == 0.001:
        # Rx & Tx bowties
        a = 0
        b = 0
        while b < 13:
            plate(x + 0.045 + a * dx, y + 0.039 + b * dx, z + skidthickness, x + 0.065 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.045 + a * dx, y + 0.067 - b * dx, z + skidthickness, x + 0.065 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.104 + a * dx, y + 0.039 + b * dx, z + skidthickness, x + 0.124 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.104 + a * dx, y + 0.067 - b * dx, z + skidthickness, x + 0.124 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            b += 1
            if a == 2 or a == 4 or a == 7:
                plate(x + 0.045 + a * dx, y + 0.039 + b * dx, z + skidthickness, x + 0.065 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
                plate(x + 0.045 + a * dx, y + 0.067 - b * dx, z + skidthickness, x + 0.065 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
                plate(x + 0.104 + a * dx, y + 0.039 + b * dx, z + skidthickness, x + 0.124 - a * dx, y + 0.039 + b * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
                plate(x + 0.104 + a * dx, y + 0.067 - b * dx, z + skidthickness, x + 0.124 - a * dx, y + 0.067 - b * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
                b += 1
            a += 1
        # Rx extension section (upper y)
        plate(x + 0.044, y + 0.068, z + skidthickness, x + 0.044 + bowtiebase, y + 0.068 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin)
        # Tx extension section (upper y)
        plate(x + 0.103, y + 0.068, z + skidthickness, x + 0.103 + bowtiebase, y + 0.068 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin)

        # Edges that represent wire between bowtie halves in 1mm model
        edge(tx[0] - 0.059, tx[1] - dy, tx[2], tx[0] - 0.059, tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0] - 0.059, tx[1] + dy, tx[2], tx[0] - 0.059, tx[1] + 0.002, tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] - dy, tx[2], tx[0], tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] + dz, tx[2], tx[0], tx[1] + 0.002, tx[2], 'pec', rotate90origin=rotate90origin)

    elif resolution == 0.002:
        # Rx & Tx bowties
        for a in range(0, 6):
            plate(x + 0.044 + a * dx, y + 0.040 + a * dx, z + skidthickness, x + 0.066 - a * dx, y + 0.040 + a * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.044 + a * dx, y + 0.064 - a * dx, z + skidthickness, x + 0.066 - a * dx, y + 0.064 - a * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.103 + a * dx, y + 0.040 + a * dx, z + skidthickness, x + 0.125 - a * dx, y + 0.040 + a * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.103 + a * dx, y + 0.064 - a * dx, z + skidthickness, x + 0.125 - a * dx, y + 0.064 - a * dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            # Rx extension section (upper y)
            plate(x + 0.044, y + 0.066, z + skidthickness, x + 0.044 + bowtiebase, y + 0.066 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            # Tx extension section (upper y)
            plate(x + 0.103, y + 0.066, z + skidthickness, x + 0.103 + bowtiebase, y + 0.066 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin)

    # Rx extension section (lower y)
    plate(x + 0.044, y + 0.024, z + skidthickness, x + 0.044 + bowtiebase, y + 0.024 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin)
    # Tx extension section (lower y)
    plate(x + 0.103, y + 0.024, z + skidthickness, x + 0.103 + bowtiebase, y + 0.024 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin)

    # Skid
    box(x, y, z, x + casesize[0], y + casesize[1], z + skidthickness, 'hdpe', rotate90origin=rotate90origin)

    # Geometry views
    # geometry_view(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + skidthickness + casesize[2] + dz, dx, dy, dz, 'antenna_like_GSSI_1500')
    # geometry_view(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz, 'antenna_like_GSSI_1500_pcb', type='f')

    # Excitation
    if optstate == 'WarrenThesis' or optstate == 'DebyeAbsorber':
        # Gaussian pulse
        print('#waveform: gaussian 1 {} myGaussian'.format(excitationfreq))
        voltage_source('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

    elif optstate == 'GiannakisPaper':
        # Optimised custom pulse
        print('#excitation_file: ../user_libs/antennas/GSSI1p5optpulse.txt linear extrapolate')
        voltage_source('y', tx[0], tx[1], tx[2], sourceresistance, 'GSSI1p5optpulse', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

    # Output point - receiver bowtie
    if resolution == 0.001:
        if optstate == 'WarrenThesis' or optstate == 'DebyeAbsorber':
            edge(tx[0] - 0.059, tx[1], tx[2], tx[0] - 0.059, tx[1] + dy, tx[2], 'rxres', rotate90origin=rotate90origin)
        rx(tx[0] - 0.059, tx[1], tx[2], identifier='rxbowtie', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
    elif resolution == 0.002:
        if optstate == 'WarrenThesis' or optstate == 'DebyeAbsorber':
            edge(tx[0] - 0.060, tx[1], tx[2], tx[0] - 0.060, tx[1] + dy, tx[2], 'rxres', rotate90origin=rotate90origin)
        rx(tx[0] - 0.060, tx[1], tx[2], identifier='rxbowtie', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)


def antenna_like_GSSI_400(x, y, z, resolution=0.001, rotate90=False):
    """Inserts a description of an antenna similar to the GSSI 400MHz antenna. Can be used with 0.5mm, 1mm (default) or 2mm spatial resolution. The external dimensions of the antenna are 300x300x178mm. One output point is defined between the arms of the receiver bowtie. The bowties are aligned with the y axis so the output is the y component of the electric field.

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
        rotate90 (bool): Rotate model 90 degrees CCW in xy plane.
    """

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

    # Set origin for rotation to geometric centre of antenna in x-y plane if required, and set output component for receiver
    if rotate90:
        rotate90origin = (x, y)
        output = 'Ex'
    else:
        rotate90origin = ()
        output = 'Ey'

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
    material(absorberEr, absorbersig, 1, 0, 'absorber')
    material(pcber, 0, 1, 0, 'pcb')
    material(hdper, 0, 1, 0, 'hdpe')

    # Antenna geometry
    if smooth_dec == 'yes':
        # Plastic case
        box(x, y, z + skidthickness - 0.002, x + casesize[0], y + casesize[1], z + casesize[2], 'hdpe', rotate90origin=rotate90origin) # new new (0.300 x 0.300 x 0.170)
        box(x + casethickness, y + casethickness, z + skidthickness - 0.002, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + casesize[2] - casethickness, 'free_space', rotate90origin=rotate90origin) # (0.296 x 0.296 x 0.168)

        # Metallic enclosure
        box(x + casethickness, y + casethickness, z + skidthickness + (metalmiddleplateheight - metalboxheight), x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + (metalmiddleplateheight - metalboxheight) + metalboxheight, 'pec', rotate90origin=rotate90origin) # new (0.296 x 0.296 x 0.088)

        # Absorber, and foam (modelled as PCB material) around edge of absorber
        box(x + casethickness, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + (metalmiddleplateheight - metalboxheight), 'absorber', rotate90origin=rotate90origin)	# new 4 (0.296 x 0.296 x 0.022)
        box(x + casethickness + shieldthickness, y + casethickness + shieldthickness, z + skidthickness + (metalmiddleplateheight - metalboxheight), x + casesize[0] - casethickness - shieldthickness, y + casesize[1] - casethickness - shieldthickness, z + skidthickness - shieldthickness + metalmiddleplateheight, 'absorber', rotate90origin=rotate90origin)	# new 4 (0.292 x 0.292 x 0.086)

        # PCB
        if resolution == 0.0005:
            box(x + 0.01 + 0.005 + 0.018, y + casethickness + 0.005 + 0.0235, z + skidthickness, x + 0.01 + 0.005 + 0.034 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)    # new
            box(x + 0.01 + 0.005 + 0.178, y + casethickness + 0.005 + 0.0235, z + skidthickness, x + 0.01 + 0.005 + 0.194 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)   # new

        elif resolution == 0.001:
            box(x + 0.01 + 0.005 + 0.018, y + casethickness + 0.005 + 0.023, z + skidthickness, x + 0.01 + 0.005 + 0.034 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)    # new
            box(x + 0.01 + 0.005 + 0.178, y + casethickness + 0.005 + 0.023, z + skidthickness, x + 0.01 + 0.005 + 0.194 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)   # new

        elif resolution == 0.002:
            box(x + 0.01 + 0.005 + 0.017, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.033 + bowtiebase, y + casethickness + 0.006 + 0.202 + patchheight, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)    # new (for use with edges)
            box(x + 0.01 + 0.005 + 0.179, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.195 + bowtiebase, y + casethickness + 0.006 + 0.202 + patchheight, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)   # new (for use with edges)

    elif smooth_dec == 'no':
        # Plastic case
        box(x, y, z + skidthickness - 0.002, x + casesize[0], y + casesize[1], z + casesize[2], 'hdpe', 'n', rotate90origin=rotate90origin) # new new (0.300 x 0.300 x 0.170)
        box(x + casethickness, y + casethickness, z + skidthickness - 0.002, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + casesize[2] - casethickness, 'free_space', 'n', rotate90origin=rotate90origin) # (0.296 x 0.296 x 0.168)

        # Metallic enclosure
        box(x + casethickness, y + casethickness, z + skidthickness + (metalmiddleplateheight - metalboxheight), x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + (metalmiddleplateheight - metalboxheight) + metalboxheight, 'pec', rotate90origin=rotate90origin) # new (0.296 x 0.296 x 0.088)

        # Absorber, and foam (modelled as PCB material) around edge of absorber
        box(x + casethickness, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + (metalmiddleplateheight - metalboxheight), 'absorber', 'n', rotate90origin=rotate90origin)	# new 4 (0.296 x 0.296 x 0.022)
        box(x + casethickness + shieldthickness, y + casethickness + shieldthickness, z + skidthickness + (metalmiddleplateheight - metalboxheight), x + casesize[0] - casethickness - shieldthickness, y + casesize[1] - casethickness - shieldthickness, z + skidthickness - shieldthickness + metalmiddleplateheight, 'absorber', 'n', rotate90origin=rotate90origin)	# new 4 (0.292 x 0.292 x 0.086)

        # PCB
        if resolution == 0.0005:
            box(x + 0.01 + 0.005 + 0.018, y + casethickness + 0.005 + 0.0235, z + skidthickness, x + 0.01 + 0.005 + 0.034 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + pcbthickness, 'pcb', 'n', rotate90origin=rotate90origin)    # new
            box(x + 0.01 + 0.005 + 0.178, y + casethickness + 0.005 + 0.0235, z + skidthickness, x + 0.01 + 0.005 + 0.194 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + pcbthickness, 'pcb', 'n', rotate90origin=rotate90origin)   # new

        elif resolution == 0.001:
            box(x + 0.01 + 0.005 + 0.018, y + casethickness + 0.005 + 0.023, z + skidthickness, x + 0.01 + 0.005 + 0.034 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + pcbthickness, 'pcb', 'n', rotate90origin=rotate90origin)    # new
            box(x + 0.01 + 0.005 + 0.178, y + casethickness + 0.005 + 0.023, z + skidthickness, x + 0.01 + 0.005 + 0.194 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + pcbthickness, 'pcb', 'n', rotate90origin=rotate90origin)   # new

        elif resolution == 0.002:
            box(x + 0.01 + 0.005 + 0.017, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.033 + bowtiebase, y + casethickness + 0.006 + 0.202 + patchheight, z + skidthickness + pcbthickness, 'pcb', 'n', rotate90origin=rotate90origin)    # new (for use with edges)
            box(x + 0.01 + 0.005 + 0.179, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.195 + bowtiebase, y + casethickness + 0.006 + 0.202 + patchheight, z + skidthickness + pcbthickness, 'pcb', 'n', rotate90origin=rotate90origin)   # new (for use with edges)

    # PCB components
    # My own bowties with triangle commands
    if resolution == 0.0005:
        # "left" side
        # extension plates
        plate(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.0235, z + skidthickness, x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.0235 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new
        # box(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.0235, z + skidthickness, x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.0235 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new new

        plate(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.204, z + skidthickness, x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new
        # box(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.204, z + skidthickness, x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new new

        # triangles
        # triangle(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.0835, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.0835, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.026 + (bowtiebase/2), y + casethickness + 0.005 + 0.0835 + bowtieheight, z + skidthickness,
                 # 0.002,'pec', rotate90origin=rotate90origin)
        triangle(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.0835, z + skidthickness,
                 x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.0835, z + skidthickness,
                 x + 0.01 + 0.005 + 0.026 + (bowtiebase/2), y + casethickness + 0.005 + 0.0835 + bowtieheight, z + skidthickness,
                 0,'pec', rotate90origin=rotate90origin) # new
        # triangle(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.026 + (bowtiebase/2), y + casethickness + 0.005 + 0.204 - bowtieheight, z + skidthickness,
                 # 0.002,'pec', rotate90origin=rotate90origin)
        triangle(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 x + 0.01 + 0.005 + 0.026 + (bowtiebase/2), y + casethickness + 0.005 + 0.204 - bowtieheight, z + skidthickness,
                 0,'pec', rotate90origin=rotate90origin) # new

        # "right" side
        plate(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.0235, z + skidthickness, x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.0235 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new
        # box(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.0235, z + skidthickness, x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.0235 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new new

        plate(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.204, z + skidthickness, x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new
        # box(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.204, z + skidthickness, x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new

        # triangles
        # triangle(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.0835, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.0835, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.186 + (bowtiebase/2), y + casethickness + 0.005 + 0.0835 + bowtieheight, z + skidthickness,
                 # 0.002,'pec', rotate90origin=rotate90origin)
        triangle(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.0835, z + skidthickness,
                 x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.0835, z + skidthickness,
                 x + 0.01 + 0.005 + 0.186 + (bowtiebase/2), y + casethickness + 0.005 + 0.0835 + bowtieheight, z + skidthickness,
                 0,'pec', rotate90origin=rotate90origin)
        # triangle(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.186 + (bowtiebase/2), y + casethickness + 0.005 + 0.204 - bowtieheight, z + skidthickness,
                 # 0.002,'pec', rotate90origin=rotate90origin)
        triangle(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 x + 0.01 + 0.005 + 0.186 + (bowtiebase/2), y + casethickness + 0.005 + 0.204 - bowtieheight, z + skidthickness,
                 0,'pec', rotate90origin=rotate90origin)

        # Edges that represent wire between bowtie halves in 1mm model
        edge(tx[0] + 0.16, tx[1] - dy, tx[2], tx[0] + 0.16, tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0] + 0.16, tx[1] + dy, tx[2], tx[0] + 0.16, tx[1] + 2*dy, tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] - dy, tx[2], tx[0], tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] + dy, tx[2], tx[0], tx[1] + 2*dy, tx[2], 'pec', rotate90origin=rotate90origin)

    elif resolution == 0.001:
        # "left" side
        # extension plates
        plate(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.023, z + skidthickness, x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.023 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new
        # box(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.023, z + skidthickness, x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.023 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new

        plate(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.204, z + skidthickness, x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new
        # box(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.204, z + skidthickness, x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new

        # triangles
        # triangle(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.083, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.083, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.026 + (bowtiebase/2), y + casethickness + 0.005 + 0.083 + bowtieheight, z + skidthickness,
                 # 0.002,'pec', rotate90origin=rotate90origin)
        triangle(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.083, z + skidthickness,
                 x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.083, z + skidthickness,
                 x + 0.01 + 0.005 + 0.026 + (bowtiebase/2), y + casethickness + 0.005 + 0.083 + bowtieheight, z + skidthickness,
                 0,'pec', rotate90origin=rotate90origin) # new
        # triangle(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.026 + (bowtiebase/2), y + casethickness + 0.005 + 0.204 - bowtieheight, z + skidthickness,
                 # 0.002,'pec', rotate90origin=rotate90origin)
        triangle(x + 0.01 + 0.005 + 0.026, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 x + 0.01 + 0.005 + 0.026 + bowtiebase, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 x + 0.01 + 0.005 + 0.026 + (bowtiebase/2), y + casethickness + 0.005 + 0.204 - bowtieheight, z + skidthickness,
                 0,'pec', rotate90origin=rotate90origin) # new

        # "right" side
        plate(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.023, z + skidthickness, x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.023 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new
        # box(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.023, z + skidthickness, x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.023 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new

        plate(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.204, z + skidthickness, x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new
        # box(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.204, z + skidthickness, x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.204 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new

        # triangles
        # triangle(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.083, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.083, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.186 + (bowtiebase/2), y + casethickness + 0.005 + 0.083 + bowtieheight, z + skidthickness,
                 # 0.002,'pec', rotate90origin=rotate90origin)
        triangle(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.083, z + skidthickness,
                 x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.083, z + skidthickness,
                 x + 0.01 + 0.005 + 0.186 + (bowtiebase/2), y + casethickness + 0.005 + 0.083 + bowtieheight, z + skidthickness,
                 0,'pec', rotate90origin=rotate90origin)
        # triangle(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 # x + 0.01 + 0.005 + 0.186 + (bowtiebase/2), y + casethickness + 0.005 + 0.204 - bowtieheight, z + skidthickness,
                 # 0.002,'pec', rotate90origin=rotate90origin)
        triangle(x + 0.01 + 0.005 + 0.186, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 x + 0.01 + 0.005 + 0.186 + bowtiebase, y + casethickness + 0.005 + 0.204, z + skidthickness,
                 x + 0.01 + 0.005 + 0.186 + (bowtiebase/2), y + casethickness + 0.005 + 0.204 - bowtieheight, z + skidthickness,
                 0,'pec', rotate90origin=rotate90origin)

        # Edges that represent wire between bowtie halves in 1mm model
        edge(tx[0] + 0.16, tx[1] - dy, tx[2], tx[0] + 0.16, tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0] + 0.16, tx[1] + dy, tx[2], tx[0] + 0.16, tx[1] + 2*dy, tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] - dy, tx[2], tx[0], tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] + dy, tx[2], tx[0], tx[1] + 2*dy, tx[2], 'pec', rotate90origin=rotate90origin)

    elif resolution == 0.002:
            # "left" side
            # extension plates
            plate(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.021 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # box(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.021 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            plate(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.203, z + skidthickness, x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.203 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # box(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.203, z + skidthickness, x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.203 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new (for use with edges)

            # triangles
            # triangle(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.025 + (bowtiebase/2), y + casethickness + 0.005 + 0.081 + bowtieheight, z + skidthickness,
                     # 0.002,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            triangle(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     x + 0.01 + 0.005 + 0.025 + (bowtiebase/2), y + casethickness + 0.005 + 0.081 + bowtieheight, z + skidthickness,
                     0,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # triangle(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.025 + (bowtiebase/2), y + casethickness + 0.005 + 0.203 - bowtieheight, z + skidthickness,
                     # 0.002,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            triangle(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     x + 0.01 + 0.005 + 0.025 + (bowtiebase/2), y + casethickness + 0.005 + 0.203 - bowtieheight, z + skidthickness,
                     0,'pec', rotate90origin=rotate90origin) # new (for use with edges)

            # "right" side
            plate(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.021 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # box(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.021 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            plate(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.203, z + skidthickness, x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.203 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # box(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.203, z + skidthickness, x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.203 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new (for use with edges)

            # triangles
            # triangle(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.187 + (bowtiebase/2), y + casethickness + 0.005 + 0.081 + bowtieheight, z + skidthickness,
                     # 0.002,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            triangle(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     x + 0.01 + 0.005 + 0.187 + (bowtiebase/2), y + casethickness + 0.005 + 0.081 + bowtieheight, z + skidthickness,
                     0,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # triangle(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.187 + (bowtiebase/2), y + casethickness + 0.005 + 0.203 - bowtieheight, z + skidthickness,
                     # 0.002,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            triangle(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     x + 0.01 + 0.005 + 0.187 + (bowtiebase/2), y + casethickness + 0.005 + 0.203 - bowtieheight, z + skidthickness,
                     0,'pec', rotate90origin=rotate90origin) # new (for use with edges)

            # Edges that represent wire between bowtie halves in 2mm model
            edge(tx[0] + 0.162, tx[1] - dy, tx[2], tx[0] + 0.162, tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
            edge(tx[0] + 0.162, tx[1] + dy, tx[2], tx[0] + 0.162, tx[1] + 2*dy, tx[2], 'pec', rotate90origin=rotate90origin)
            edge(tx[0], tx[1] - dy, tx[2], tx[0], tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
            edge(tx[0], tx[1] + dy, tx[2], tx[0], tx[1] + 2*dy, tx[2], 'pec', rotate90origin=rotate90origin)

            # "left" side
            # extension plates
            plate(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.021 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # box(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.021 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            plate(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.203, z + skidthickness, x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.203 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # box(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.203, z + skidthickness, x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.203 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new (for use with edges)

            # triangles
            # triangle(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.025 + (bowtiebase/2), y + casethickness + 0.005 + 0.081 + bowtieheight, z + skidthickness,
                     # 0.002,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            triangle(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     x + 0.01 + 0.005 + 0.025 + (bowtiebase/2), y + casethickness + 0.005 + 0.081 + bowtieheight, z + skidthickness,
                     0,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # triangle(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.025 + (bowtiebase/2), y + casethickness + 0.005 + 0.203 - bowtieheight, z + skidthickness,
                     # 0.002,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            triangle(x + 0.01 + 0.005 + 0.025, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     x + 0.01 + 0.005 + 0.025 + bowtiebase, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     x + 0.01 + 0.005 + 0.025 + (bowtiebase/2), y + casethickness + 0.005 + 0.203 - bowtieheight, z + skidthickness,
                     0,'pec', rotate90origin=rotate90origin) # new (for use with edges)

            # "right" side
            plate(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.021 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # box(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.021, z + skidthickness, x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.021 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            plate(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.203, z + skidthickness, x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.203 + patchheight, z + skidthickness, 'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # box(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.203, z + skidthickness, x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.203 + patchheight, z + skidthickness + 0.002, 'pec', rotate90origin=rotate90origin) # new (for use with edges)

            # triangles
            # triangle(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.187 + (bowtiebase/2), y + casethickness + 0.005 + 0.081 + bowtieheight, z + skidthickness,
                     # 0.002,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            triangle(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.081, z + skidthickness,
                     x + 0.01 + 0.005 + 0.187 + (bowtiebase/2), y + casethickness + 0.005 + 0.081 + bowtieheight, z + skidthickness,
                     0,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            # triangle(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     # x + 0.01 + 0.005 + 0.187 + (bowtiebase/2), y + casethickness + 0.005 + 0.203 - bowtieheight, z + skidthickness,
                     # 0.002,'pec', rotate90origin=rotate90origin) # new (for use with edges)
            triangle(x + 0.01 + 0.005 + 0.187, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     x + 0.01 + 0.005 + 0.187 + bowtiebase, y + casethickness + 0.005 + 0.203, z + skidthickness,
                     x + 0.01 + 0.005 + 0.187 + (bowtiebase/2), y + casethickness + 0.005 + 0.203 - bowtieheight, z + skidthickness,
                     0,'pec', rotate90origin=rotate90origin) # new (for use with edges)

            # Edges that represent wire between bowtie halves in 2mm model
            edge(tx[0] + 0.162, tx[1] - dy, tx[2], tx[0] + 0.162, tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
            edge(tx[0] + 0.162, tx[1] + dy, tx[2], tx[0] + 0.162, tx[1] + 2*dy, tx[2], 'pec', rotate90origin=rotate90origin)
            edge(tx[0], tx[1] - dy, tx[2], tx[0], tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
            edge(tx[0], tx[1] + dy, tx[2], tx[0], tx[1] + 2*dy, tx[2], 'pec', rotate90origin=rotate90origin)

    # metallic plate extension
    box(x + (casesize[0] / 2), y + casethickness, z + skidthickness, x + (casesize[0] / 2) + shieldthickness, y + casesize[1] - casethickness, z + skidthickness + metalmiddleplateheight, 'pec', rotate90origin=rotate90origin) # new

    if smooth_dec == 'yes':
        # Skid
        box(x, y, z, x + casesize[0], y + casesize[1], z + skidthickness - 0.002, 'hdpe', rotate90origin=rotate90origin) # new

    elif smooth_dec == 'no':
        # Skid
        box(x, y, z, x + casesize[0], y + casesize[1], z + skidthickness - 0.002, 'hdpe', 'n', rotate90origin=rotate90origin) # new

    # Excitation - Gaussian pulse
    print('#waveform: gaussian 1 {} myGaussian'.format(excitationfreq))
    if src_type == 'voltage_source':
        voltage_source('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
    elif src_type == 'transmission_line':
        transmission_line('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

    # Output point - receiver bowtie
    print('#waveform: gaussian 0 4e8 my_zero_src')

    if resolution == 0.001 or resolution == 0.0005:
        if src_type == 'transmission_line':
            transmission_line('y', tx[0] + 0.16, tx[1], tx[2], receiverresistance, 'my_zero_src', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
        elif src_type == 'voltage_source':
            voltage_source('y', tx[0] + 0.16, tx[1], tx[2], receiverresistance, 'my_zero_src', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
        rx(tx[0] + 0.16, tx[1], tx[2], identifier='rx1', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

    elif resolution == 0.002:
        if src_type == 'transmission_line':
            transmission_line('y', tx[0] + 0.162, tx[1], tx[2], receiverresistance, 'my_zero_src', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
        elif src_type == 'voltage_source':
            rx(tx[0] + 0.162, tx[1], tx[2], identifier='rx1', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

    # Geometry views
    # geometry_view(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + skidthickness + casesize[2] + dz, dx, dy, dz, 'antenna_like_GSSI_400')
    # geometry_view(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz, 'antenna_like_GSSI_400_pcb', type='f')
