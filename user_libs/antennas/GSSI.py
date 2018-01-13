# Copyright (C) 2015-2018, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import os

from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import *

userlibdir = os.path.dirname(os.path.abspath(__file__))


def antenna_like_GSSI_1500(x, y, z, resolution=0.001, rotate90=False, **kwargs):
    """Inserts a description of an antenna similar to the GSSI 1.5GHz antenna. Can be used with 1mm (default) or 2mm spatial resolution. The external dimensions of the antenna are 170x108x45mm. One output point is defined between the arms of the receiever bowtie. The bowties are aligned with the y axis so the output is the y component of the electric field (x component if the antenna is rotated 90 degrees).

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
        rotate90 (bool): Rotate model 90 degrees CCW in xy plane.
        kwargs (dict): Optional variables, e.g. can be fed from an optimisation process.
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

    # Unknown properties
    if kwargs:
        excitationfreq = kwargs['excitationfreq']
        sourceresistance = kwargs['sourceresistance']
        absorberEr = kwargs['absorberEr']
        absorbersig = kwargs['absorbersig']
        rxres = 50
    else:
        # excitationfreq = 1.5e9 # GHz
        # sourceresistance = 50 # Ohms
        # absorberEr = 1.7
        # absorbersig = 0.59

        # Values from http://hdl.handle.net/1842/4074
        excitationfreq = 1.71e9
        # sourceresistance = 4
        sourceresistance = 230  # Correction for old (< 123) GprMax3D bug
        # absorberEr = 1.58
        # absorbersig = 0.428
        rxres = 925  # Resistance at Rx bowtie

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

    # Material definitions
    # material(absorberEr, absorbersig, 1, 0, 'absorber')
    material(1, 0, 1, 0, 'absorber')
    print('#add_dispersion_debye: 3 3.7733 1.00723e-11 3.14418 1.55686e-10 20.2441 3.44129e-10 absorber') # Eccosorb LS22 3-pole Debye model (https://bitbucket.org/uoyaeg/aegboxts/wiki/Home)
    material(3, 0, 1, 0, 'pcb')
    material(2.35, 0, 1, 0, 'hdpe')
    material(3, (1 / rxres) * (dy / (dx * dz)), 1, 0, 'rxres')

    # Antenna geometry
    # Plastic case
    box(x, y, z + skidthickness, x + casesize[0], y + casesize[1], z + skidthickness + casesize[2], 'hdpe', rotate90origin=rotate90origin)
    box(x + casethickness, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness, 'free_space', rotate90origin=rotate90origin)

    # Metallic enclosure
    box(x + 0.025, y + casethickness, z + skidthickness, x + casesize[0] - 0.025, y + casesize[1] - casethickness, z + skidthickness + 0.027, 'pec', rotate90origin=rotate90origin)

    # Absorber material, and foam (modelled as PCB material) around edge of absorber
    box(x + 0.025 + shieldthickness, y + casethickness + shieldthickness, z + skidthickness, x + 0.025 + shieldthickness + 0.057, y + casesize[1] - casethickness - shieldthickness, z + skidthickness + 0.027 - shieldthickness - 0.001, 'pcb', rotate90origin=rotate90origin)
    box(x + 0.025 + shieldthickness + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.025 + shieldthickness + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + 0.027 - shieldthickness, 'absorber', rotate90origin=rotate90origin)
    box(x + 0.086, y + casethickness + shieldthickness, z + skidthickness, x + 0.086 + 0.057, y + casesize[1] - casethickness - shieldthickness, z + skidthickness + 0.027 - shieldthickness - 0.001, 'pcb', rotate90origin=rotate90origin)
    box(x + 0.086 + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + 0.027 - shieldthickness, 'absorber', rotate90origin=rotate90origin)

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

    # Excitation - custom pulse
    # print('#excitation_file: {}'.format(os.path.join(userlibdir, 'GSSIgausspulse1.txt')))
    # print('#transmission_line: y {} {} {} {} GSSIgausspulse1'.format(tx[0], tx[1], tx[2], sourceresistance))

    # Excitation - Gaussian pulse
    print('#waveform: gaussian 1 {} myGaussian'.format(excitationfreq))
    # transmission_line('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
    voltage_source('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

    # Output point - receiver bowtie
    if resolution == 0.001:
        edge(tx[0] - 0.059, tx[1], tx[2], tx[0] - 0.059, tx[1] + dy, tx[2], 'rxres', rotate90origin=rotate90origin)
        rx(tx[0] - 0.059, tx[1], tx[2], identifier='rxbowtie', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
    elif resolution == 0.002:
        edge(tx[0] - 0.060, tx[1], tx[2], tx[0] - 0.060, tx[1] + dy, tx[2], 'rxres', rotate90origin=rotate90origin)
        rx(tx[0] - 0.060, tx[1], tx[2], identifier='rxbowtie', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
