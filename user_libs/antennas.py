# Copyright (C) 2015-2017, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import os

from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import *

moduledirectory = os.path.dirname(os.path.abspath(__file__))

def antenna_like_GSSI_1500(x, y, z, resolution=0.001, rotate90=False, **kwargs):
    """Inserts a description of an antenna similar to the GSSI 1.5GHz antenna. Can be used with 1mm (default) or 2mm spatial resolution. The external dimensions of the antenna are 170x108x45mm. One output point is defined between the arms of the receiever bowtie. The bowties are aligned with the y axis so the output is the y component of the electric field.
        
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
    
    # Set origin for rotation to geometric centre of antenna in x-y plane if required
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
        #excitationfreq = 1.5e9 # GHz
        #sourceresistance = 50 # Ohms
        #absorberEr = 1.7
        #absorbersig = 0.59
        
        # Values from http://hdl.handle.net/1842/4074
        excitationfreq = 1.71e9
        #sourceresistance = 4
        sourceresistance = 230 # Correction for old (< 123) GprMax3D bug
        absorberEr = 1.58
        absorbersig = 0.428
        rxres = 925 # Resistance at Rx bowtie
    
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
        tx = x + 0.112, y + 0.052, z + skidthickness
    else:
        raise CmdInputError('This antenna module can only be used with a spatial discretisation of 1mm or 2mm')

    # Material definitions
    material(absorberEr, absorbersig, 1, 0, 'absorber')
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
            plate(x + 0.045 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.045 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.104 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.104 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            b += 1
            if a == 2 or a == 4 or a == 7:
                plate(x + 0.045 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
                plate(x + 0.045 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
                plate(x + 0.104 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
                plate(x + 0.104 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
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
        for a in range(0,6):
            plate(x + 0.044 + a*dx, y + 0.040 + a*dx, z + skidthickness, x + 0.066 - a*dx, y + 0.040 + a*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.044 + a*dx, y + 0.064 - a*dx, z + skidthickness, x + 0.066 - a*dx, y + 0.064 - a*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.103 + a*dx, y + 0.040 + a*dx, z + skidthickness, x + 0.125 - a*dx, y + 0.040 + a*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
            plate(x + 0.103 + a*dx, y + 0.064 - a*dx, z + skidthickness, x + 0.125 - a*dx, y + 0.064 - a*dx + dy, z + skidthickness, 'pec', rotate90origin=rotate90origin)
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
    #geometry_view(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + skidthickness + casesize[2] + dz, dx, dy, dz, 'antenna_like_GSSI_1500')
    #geometry_view(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz, 'antenna_like_GSSI_1500_pcb', type='f')
    
    # Excitation - custom pulse
    #print('#excitation_file: {}'.format(os.path.join(moduledirectory, 'GSSIgausspulse1.txt')))
    #print('#transmission_line: y {} {} {} {} GSSIgausspulse1'.format(tx[0], tx[1], tx[2], sourceresistance))

    # Excitation - Gaussian pulse
    print('#waveform: gaussian 1 {} myGaussian'.format(excitationfreq))
    transmission_line('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
    
    # Output point - receiver bowtie
    if resolution == 0.001:
        edge(tx[0] - 0.059, tx[1], tx[2], tx[0] - 0.059, tx[1] + dy, tx[2], 'rxres', rotate90origin=rotate90origin)
        rx(tx[0] - 0.059, tx[1], tx[2], identifier='rxbowtie', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
    elif resolution == 0.002:
        edge(tx[0] - 0.058, tx[1], tx[2], tx[0] - 0.058, tx[1] + dy, tx[2], 'rxres', rotate90origin=rotate90origin)
        rx(tx[0] - 0.058, tx[1], tx[2], identifier='rxbowtie', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)


def antenna_like_MALA_1200(x, y, z, resolution=0.001, rotate90=False, **kwargs):
    """Inserts a description of an antenna similar to the MALA 1.2GHz antenna. Can be used with 1mm (default) or 2mm spatial resolution. The external dimensions of the antenna are 184x109x46mm. One output point is defined between the arms of the receiever bowtie. The bowties are aligned with the y axis so the output is the y component of the electric field.
        
    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
        rotate90 (bool): Rotate model 90 degrees CCW in xy plane.
        kwargs (dict): Optional variables, e.g. can be fed from an optimisation process.
    """
    
    # Antenna geometry properties
    casesize = (0.184, 0.109, 0.040)
    casethickness = 0.002
    cavitysize = (0.062, 0.062, 0.037)
    cavitythickness = 0.001
    pcbthickness = 0.002
    polypropylenethickness = 0.003;
    hdpethickness = 0.003;
    skidthickness = 0.006
    bowtieheight = 0.025
    
    # Set origin for rotation to geometric centre of antenna in x-y plane if required
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
    else:
        # Values from http://hdl.handle.net/1842/4074
        excitationfreq = 0.978e9
        sourceresistance = 1000
        absorberEr = 6.49
        absorbersig = 0.252
    
    x = x - (casesize[0] / 2)
    y = y - (casesize[1] / 2)
    
    # Coordinates of source excitation point in antenna
    tx = x + 0.063, y + 0.052, z + skidthickness
    
    if resolution == 0.001:
        dx = 0.001
        dy = 0.001
        dz = 0.001
    elif resolution == 0.002:
        dx = 0.002
        dy = 0.002
        dz = 0.002
        cavitysize = (0.062, 0.062, 0.036)
        cavitythickness = 0.002
        polypropylenethickness = 0.002;
        hdpethickness = 0.004;
        bowtieheight = 0.024
        tx = x + 0.062, y + 0.052, z + skidthickness
    else:
        raise CmdInputError('This antenna module can only be used with a spatial resolution of 1mm or 2mm')
    
    # SMD resistors - 3 on each Tx & Rx bowtie arm
    txres = 470 # Ohms
    txrescellupper = txres / 3 # Resistor over 3 cells
    txsigupper = ((1 / txrescellupper) * (dy / (dx * dz))) / 2 # Divide by number of parallel edges per resistor
    txrescelllower = txres / 4 # Resistor over 4 cells
    txsiglower = ((1 / txrescelllower) * (dy / (dx * dz))) / 2 # Divide by number of parallel edges per resistor
    rxres = 150 # Ohms
    rxrescellupper = rxres / 3 # Resistor over 3 cells
    rxsigupper = ((1 / rxrescellupper) * (dy / (dx * dz))) / 2 # Divide by number of parallel edges per resistor
    rxrescelllower = rxres / 4 # Resistor over 4 cells
    rxsiglower = ((1 / rxrescelllower) * (dy / (dx * dz))) / 2 # Divide by number of parallel edges per resistor
    
    # Material definitions
    material(absorberEr, absorbersig, 1, 0, 'absorber')
    material(3, 0, 1, 0, 'pcb')
    material(2.35, 0, 1, 0, 'hdpe')
    material(2.26, 0, 1, 0, 'polypropylene')
    material(3, txsiglower, 1, 0, 'txreslower')
    material(3, txsigupper, 1, 0, 'txresupper')
    material(3, rxsiglower, 1, 0, 'rxreslower')
    material(3, rxsigupper, 1, 0, 'rxresupper')
    
    # Antenna geometry
    # Shield - metallic enclosure
    box(x, y, z + skidthickness, x + casesize[0], y + casesize[1], z + skidthickness + casesize[2], 'pec', rotate90origin=rotate90origin)
    box(x + 0.020, y + casethickness, z + skidthickness, x + 0.100, y + casesize[1] - casethickness, z + skidthickness + casethickness, 'free_space', rotate90origin=rotate90origin)
    box(x + 0.100, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casethickness, 'free_space', rotate90origin=rotate90origin)
    
    # Absorber material
    box(x + 0.020, y + casethickness, z + skidthickness, x + 0.100, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness, 'absorber', rotate90origin=rotate90origin)
    box(x + 0.100, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness, 'absorber', rotate90origin=rotate90origin)
    
    # Shield - cylindrical sections
    cylinder(x + 0.055, y + casesize[1] - 0.008, z + skidthickness, x + 0.055, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.008, 'pec', rotate90origin=rotate90origin)
    cylinder(x + 0.055, y + 0.008, z + skidthickness, x + 0.055, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.008, 'pec', rotate90origin=rotate90origin)
    cylinder(x + 0.147, y + casesize[1] - 0.008, z + skidthickness, x + 0.147, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.008, 'pec', rotate90origin=rotate90origin)
    cylinder(x + 0.147, y + 0.008, z + skidthickness, x + 0.147, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.008, 'pec', rotate90origin=rotate90origin)
    cylinder(x + 0.055, y + casesize[1] - 0.008, z + skidthickness, x + 0.055, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.007, 'free_space', rotate90origin=rotate90origin)
    cylinder(x + 0.055, y + 0.008, z + skidthickness, x + 0.055, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.007, 'free_space', rotate90origin=rotate90origin)
    cylinder(x + 0.147, y + casesize[1] - 0.008, z + skidthickness, x + 0.147, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.007, 'free_space', rotate90origin=rotate90origin)
    cylinder(x + 0.147, y + 0.008, z + skidthickness, x + 0.147, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.007, 'free_space', rotate90origin=rotate90origin)
    box(x + 0.054, y + casesize[1] - 0.016, z + skidthickness, x + 0.056, y + casesize[1] - 0.014, z + skidthickness + casesize[2] - casethickness, 'free_space', rotate90origin=rotate90origin)
    box(x + 0.054, y + 0.014, z + skidthickness, x + 0.056, y + 0.016, z + skidthickness + casesize[2] - casethickness, 'free_space', rotate90origin=rotate90origin)
    box(x + 0.146, y + casesize[1] - 0.016, z + skidthickness, x + 0.148, y + casesize[1] - 0.014, z + skidthickness + casesize[2] - casethickness, 'free_space', rotate90origin=rotate90origin)
    box(x + 0.146, y + 0.014, z + skidthickness, x + 0.148, y + 0.016, z + skidthickness + casesize[2] - casethickness, 'free_space', rotate90origin=rotate90origin)
    
    # PCB
    box(x + 0.020, y + 0.018, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - 0.018, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)
    
    # Shield - Tx & Rx cavities
    box(x + 0.032, y + 0.022, z + skidthickness, x + 0.032 + cavitysize[0], y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2], 'pec', rotate90origin=rotate90origin)
    box(x + 0.032 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.032 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + cavitysize[2], 'absorber', rotate90origin=rotate90origin)
    box(x + 0.108, y + 0.022, z + skidthickness, x + 0.108 + cavitysize[0], y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2], 'pec', rotate90origin=rotate90origin)
    box(x + 0.108 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.108 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + cavitysize[2], 'free_space', rotate90origin=rotate90origin)
    
    # Shield - Tx & Rx cavities - joining strips
    box(x + 0.032 + cavitysize[0], y + 0.022 + cavitysize[1] - 0.006, z + skidthickness + cavitysize[2] - casethickness, x + 0.108, y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2], 'pec', rotate90origin=rotate90origin)
    box(x + 0.032 + cavitysize[0], y + 0.022, z + skidthickness + cavitysize[2] - casethickness, x + 0.108, y + 0.022 + 0.006, z + skidthickness + cavitysize[2], 'pec', rotate90origin=rotate90origin)
    
    # PCB - replace bits chopped by TX & Rx cavities
    box(x + 0.032 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.032 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)
    box(x + 0.108 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.108 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + pcbthickness, 'pcb', rotate90origin=rotate90origin)
    
    # PCB components
    # Tx bowtie
    if resolution == 0.001:
        triangle(tx[0], tx[1] - 0.001, tx[2], tx[0] - 0.026, tx[1] - bowtieheight - 0.001, tx[2], tx[0] + 0.026, tx[1] - bowtieheight - 0.001, tx[2], 0, 'pec', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] - 0.001, tx[2], tx[0], tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
        triangle(tx[0], tx[1] + 0.002, tx[2], tx[0] - 0.026, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.026, tx[1] + bowtieheight + 0.002, tx[2], 0, 'pec', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] + 0.001, tx[2], tx[0], tx[1] + 0.002, tx[2], 'pec', rotate90origin=rotate90origin)
    elif resolution == 0.002:
        triangle(tx[0], tx[1], tx[2], tx[0] - 0.026, tx[1] - bowtieheight, tx[2], tx[0] + 0.026, tx[1] - bowtieheight, tx[2], 0, 'pec', rotate90origin=rotate90origin)
        triangle(tx[0], tx[1] + 0.002, tx[2], tx[0] - 0.026, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.026, tx[1] + bowtieheight + 0.002, tx[2], 0, 'pec', rotate90origin=rotate90origin)
    
    # Rx bowtie
    if resolution == 0.001:
        triangle(tx[0] + 0.076, tx[1] - 0.001, tx[2], tx[0] + 0.076 - 0.026, tx[1] - bowtieheight - 0.001, tx[2], tx[0] + 0.076 + 0.026, tx[1] - bowtieheight - 0.001, tx[2], 0, 'pec', rotate90origin=rotate90origin)
        edge(tx[0] + 0.076, tx[1] - 0.001, tx[2], tx[0] + 0.076, tx[1], tx[2], 'pec', rotate90origin=rotate90origin)
        triangle(tx[0] + 0.076, tx[1] + 0.002, tx[2], tx[0] + 0.076 - 0.026, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076 + 0.026, tx[1] + bowtieheight + 0.002, tx[2], 0, 'pec', rotate90origin=rotate90origin)
        edge(tx[0] + 0.076, tx[1] + 0.001, tx[2], tx[0] + 0.076, tx[1] + 0.002, tx[2], 'pec', rotate90origin=rotate90origin)
    elif resolution == 0.002:
        triangle(tx[0] + 0.076, tx[1], tx[2], tx[0] + 0.076 - 0.026, tx[1] - bowtieheight, tx[2], tx[0] + 0.076 + 0.026, tx[1] - bowtieheight, tx[2], 0, 'pec', rotate90origin=rotate90origin)
        triangle(tx[0] + 0.076, tx[1] + 0.002, tx[2], tx[0] + 0.076 - 0.026, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076 + 0.026, tx[1] + bowtieheight + 0.002, tx[2], 0, 'pec', rotate90origin=rotate90origin)
    
    # Tx surface mount resistors (lower y coordinate)
    if resolution == 0.001:
        edge(tx[0] - 0.023, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023, tx[1] - bowtieheight - dy, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0] - 0.023 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx, tx[1] - bowtieheight - dy, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] - bowtieheight - 0.004, tx[2], tx[0], tx[1] - bowtieheight - dy, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0] + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx, tx[1] - bowtieheight - dy, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.022, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022, tx[1] - bowtieheight - dy, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.022 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + dx, tx[1] - bowtieheight - dy, tx[2], 'txreslower', rotate90origin=rotate90origin)
    elif resolution == 0.002:
        edge(tx[0] - 0.023, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023, tx[1] - bowtieheight, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0] - 0.023 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx, tx[1] - bowtieheight, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] - bowtieheight - 0.004, tx[2], tx[0], tx[1] - bowtieheight, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0] + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx, tx[1] - bowtieheight, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.020, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020, tx[1] - bowtieheight, tx[2], 'txreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.020 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + dx, tx[1] - bowtieheight, tx[2], 'txreslower', rotate90origin=rotate90origin)
    
    # Tx surface mount resistors (upper y coordinate)
    if resolution == 0.001:
        edge(tx[0] - 0.023, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] + bowtieheight + 0.002, tx[2], tx[0], tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0] + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.022, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.022 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
    elif resolution == 0.002:
        edge(tx[0] - 0.023, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0], tx[1] + bowtieheight + 0.002, tx[2], tx[0], tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0] + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.020, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.020 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper', rotate90origin=rotate90origin)
    
    # Rx surface mount resistors (lower y coordinate)
    if resolution == 0.001:
        edge(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.022 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.022 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + dx + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower', rotate90origin=rotate90origin)
    elif resolution == 0.002:
        edge(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.020 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower', rotate90origin=rotate90origin)
        edge(tx[0] + 0.020 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + dx + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower', rotate90origin=rotate90origin)
    
    # Rx surface mount resistors (upper y coordinate)
    if resolution == 0.001:
        edge(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.022 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.022 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
    elif resolution == 0.002:
        edge(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.020 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
        edge(tx[0] + 0.020 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper', rotate90origin=rotate90origin)
    
    # Skid
    box(x, y, z, x + casesize[0], y + casesize[1], z + polypropylenethickness, 'polypropylene', rotate90origin=rotate90origin)
    box(x, y, z + polypropylenethickness, x + casesize[0], y + casesize[1], z + polypropylenethickness + hdpethickness, 'hdpe', rotate90origin=rotate90origin)

    # Geometry views
    #geometry_view(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + casesize[2] + skidthickness + dz, dx, dy, dz, 'antenna_like_MALA_1200')
    #geometry_view(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz, 'antenna_like_MALA_1200_pcb', type='f')
    
    # Excitation
    print('#waveform: gaussian 1.0 {} myGaussian'.format(excitationfreq))
    voltage_source('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
    
    # Output point - receiver bowtie
    rx(tx[0] + 0.076, tx[1], tx[2], identifier='rxbowtie', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

