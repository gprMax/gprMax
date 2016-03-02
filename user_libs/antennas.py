# Copyright (C) 2015-2016, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import os

from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import *

moduledirectory = os.path.dirname(os.path.abspath(__file__))

def antenna_like_GSSI_1500(x, y, z, resolution=0.001, **kwargs):
    """Inserts a description of an antenna similar to the GSSI 1.5GHz antenna. Can be used with 1mm (default) or 2mm spatial resolution. The external dimensions of the antenna are 170mm x 108mm x 45mm. One output point is defined between the arms of the receiever bowtie. The bowties are aligned with the y axis so the output is the y component of the electric field.
        
    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
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
    
    # Unknown properties
    if kwargs:
        excitationfreq = kwargs['excitationfreq']
        sourceresistance = kwargs['sourceresistance']
        absorberEr = kwargs['absorberEr']
        absorbersig = kwargs['absorbersig']
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
    box(x, y, z + skidthickness, x + casesize[0], y + casesize[1], z + skidthickness + casesize[2], 'hdpe')
    box(x + casethickness, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness, 'free_space')
    
    # Metallic enclosure
    box(x + 0.025, y + casethickness, z + skidthickness, x + casesize[0] - 0.025, y + casesize[1] - casethickness, z + skidthickness + 0.027, 'pec')
    
    # Absorber material, and foam (modelled as PCB material) around edge of absorber
    box(x + 0.025 + shieldthickness, y + casethickness + shieldthickness, z + skidthickness, x + 0.025 + shieldthickness + 0.057, y + casesize[1] - casethickness - shieldthickness, z + skidthickness + 0.027 - shieldthickness - 0.001, 'pcb')
    box(x + 0.025 + shieldthickness + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.025 + shieldthickness + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + 0.027 - shieldthickness, 'absorber')
    box(x + 0.086, y + casethickness + shieldthickness, z + skidthickness, x + 0.086 + 0.057, y + casesize[1] - casethickness - shieldthickness, z + skidthickness + 0.027 - shieldthickness - 0.001, 'pcb')
    box(x + 0.086 + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + 0.027 - shieldthickness, 'absorber')
    
    # PCB
    box(x + 0.025 + shieldthickness + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 - shieldthickness - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + pcbthickness, 'pcb')
    box(x + 0.086 + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + pcbthickness, 'pcb')
    
    # PCB components
    if resolution == 0.001:
        # Rx & Tx bowties
        a = 0
        b = 0
        while b < 13:
            plate(x + 0.045 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness, 'pec')
            plate(x + 0.045 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness, 'pec')
            plate(x + 0.104 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness, 'pec')
            plate(x + 0.104 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness, 'pec')
            b += 1
            if a == 2 or a == 4 or a == 7:
                plate(x + 0.045 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness, 'pec')
                plate(x + 0.045 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness, 'pec')
                plate(x + 0.104 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness, 'pec')
                plate(x + 0.104 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness, 'pec')
                b += 1
            a += 1
        # Rx extension section (upper y)
        plate(x + 0.044, y + 0.068, z + skidthickness, x + 0.044 + bowtiebase, y + 0.068 + patchheight, z + skidthickness, 'pec')
        # Tx extension section (upper y)
        plate(x + 0.103, y + 0.068, z + skidthickness, x + 0.103 + bowtiebase, y + 0.068 + patchheight, z + skidthickness, 'pec')
        
        # Edges that represent wire between bowtie halves in 1mm model
        edge(tx[0] - 0.059, tx[1] - dy, tx[2], tx[0] - 0.059, tx[1], tx[2], 'pec')
        edge(tx[0] - 0.059, tx[1] + dy, tx[2], tx[0] - 0.059, tx[1] + 0.002, tx[2], 'pec')
        edge(tx[0], tx[1] - dy, tx[2], tx[0], tx[1], tx[2], 'pec')
        edge(tx[0], tx[1] + dz, tx[2], tx[0], tx[1] + 0.002, tx[2], 'pec')

    elif resolution == 0.002:
    # Rx & Tx bowties
        for a in range(0,6):
            plate(x + 0.044 + a*dx, y + 0.040 + a*dx, z + skidthickness, x + 0.066 - a*dx, y + 0.040 + a*dx + dy, z + skidthickness, 'pec')
            plate(x + 0.044 + a*dx, y + 0.064 - a*dx, z + skidthickness, x + 0.066 - a*dx, y + 0.064 - a*dx + dy, z + skidthickness, 'pec')
            plate(x + 0.103 + a*dx, y + 0.040 + a*dx, z + skidthickness, x + 0.125 - a*dx, y + 0.040 + a*dx + dy, z + skidthickness, 'pec')
            plate(x + 0.103 + a*dx, y + 0.064 - a*dx, z + skidthickness, x + 0.125 - a*dx, y + 0.064 - a*dx + dy, z + skidthickness, 'pec')
            # Rx extension section (upper y)
            plate(x + 0.044, y + 0.066, z + skidthickness, x + 0.044 + bowtiebase, y + 0.066 + patchheight, z + skidthickness, 'pec')
            # Tx extension section (upper y)
            plate(x + 0.103, y + 0.066, z + skidthickness, x + 0.103 + bowtiebase, y + 0.066 + patchheight, z + skidthickness, 'pec')

    # Rx extension section (lower y)
    plate(x + 0.044, y + 0.024, z + skidthickness, x + 0.044 + bowtiebase, y + 0.024 + patchheight, z + skidthickness, 'pec')
    # Tx extension section (lower y)
    plate(x + 0.103, y + 0.024, z + skidthickness, x + 0.103 + bowtiebase, y + 0.024 + patchheight, z + skidthickness, 'pec')

    # Skid
    box(x, y, z, x + casesize[0], y + casesize[1], z + skidthickness, 'hdpe')
    
    # Geometry views
    #geometry_view(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + skidthickness + casesize[2] + dz, dx, dy, dz, 'antenna_like_GSSI_1500')
    #geometry_view(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz, 'antenna_like_GSSI_1500_pcb', type='f')
    
    # Excitation - custom pulse
    #print('#excitation_file: {}'.format(os.path.join(moduledirectory, 'GSSIgausspulse1.txt')))
    #print('#transmission_line: y {} {} {} {} GSSIgausspulse1'.format(tx[0], tx[1], tx[2], sourceresistance))

    # Excitation - Gaussian pulse
    print('#waveform: gaussian 1 {} myGaussian'.format(excitationfreq))
    print('#transmission_line: y {} {} {} {} myGaussian'.format(tx[0], tx[1], tx[2], sourceresistance))
    
    # Output point - transmitter bowtie
    #print('#rx: {} {} {}'.format(tx[0], tx[1], tx[2]))
    # Output point - receiver bowtie
    edge(tx[0] - 0.059, tx[1], tx[2], tx[0] - 0.059, tx[1] + dy, tx[2], 'rxres')
    print('#rx: {} {} {} rxbowtie Ey'.format(tx[0] - 0.059, tx[1], tx[2]))



def antenna_like_MALA_1200(x, y, z, resolution=0.001, **kwargs):
    """Inserts a description of an antenna similar to the MALA 1.2GHz antenna. Can be used with 1mm (default) or 2mm spatial resolution. The external dimensions of the antenna are 184mm x 109mm x 46mm. One output point is defined between the arms of the receiever bowtie. The bowties are aligned with the y axis so the output is the y component of the electric field.
        
    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
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
    box(x, y, z + skidthickness, x + casesize[0], y + casesize[1], z + skidthickness + casesize[2], 'pec')
    box(x + 0.020, y + casethickness, z + skidthickness, x + 0.100, y + casesize[1] - casethickness, z + skidthickness + casethickness, 'free_space')
    box(x + 0.100, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casethickness, 'free_space')
    
    # Absorber material
    box(x + 0.020, y + casethickness, z + skidthickness, x + 0.100, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness, 'absorber')
    box(x + 0.100, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness, 'absorber')
    
    # Shield - cylindrical sections
    cylinder(x + 0.055, y + casesize[1] - 0.008, z + skidthickness, x + 0.055, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.008, 'pec')
    cylinder(x + 0.055, y + 0.008, z + skidthickness, x + 0.055, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.008, 'pec')
    cylinder(x + 0.147, y + casesize[1] - 0.008, z + skidthickness, x + 0.147, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.008, 'pec')
    cylinder(x + 0.147, y + 0.008, z + skidthickness, x + 0.147, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.008, 'pec')
    cylinder(x + 0.055, y + casesize[1] - 0.008, z + skidthickness, x + 0.055, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.007, 'free_space')
    cylinder(x + 0.055, y + 0.008, z + skidthickness, x + 0.055, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.007, 'free_space')
    cylinder(x + 0.147, y + casesize[1] - 0.008, z + skidthickness, x + 0.147, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.007, 'free_space')
    cylinder(x + 0.147, y + 0.008, z + skidthickness, x + 0.147, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.007, 'free_space')
    box(x + 0.054, y + casesize[1] - 0.016, z + skidthickness, x + 0.056, y + casesize[1] - 0.014, z + skidthickness + casesize[2] - casethickness, 'free_space')
    box(x + 0.054, y + 0.014, z + skidthickness, x + 0.056, y + 0.016, z + skidthickness + casesize[2] - casethickness, 'free_space')
    box(x + 0.146, y + casesize[1] - 0.016, z + skidthickness, x + 0.148, y + casesize[1] - 0.014, z + skidthickness + casesize[2] - casethickness, 'free_space')
    box(x + 0.146, y + 0.014, z + skidthickness, x + 0.148, y + 0.016, z + skidthickness + casesize[2] - casethickness, 'free_space')
    
    # PCB
    box(x + 0.020, y + 0.018, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - 0.018, z + skidthickness + pcbthickness, 'pcb')
    
    # Shield - Tx & Rx cavities
    box(x + 0.032, y + 0.022, z + skidthickness, x + 0.032 + cavitysize[0], y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2], 'pec')
    box(x + 0.032 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.032 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + cavitysize[2], 'absorber')
    box(x + 0.108, y + 0.022, z + skidthickness, x + 0.108 + cavitysize[0], y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2], 'pec')
    box(x + 0.108 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.108 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + cavitysize[2], 'free_space')
    
    # Shield - Tx & Rx cavities - joining strips
    box(x + 0.032 + cavitysize[0], y + 0.022 + cavitysize[1] - 0.006, z + skidthickness + cavitysize[2] - casethickness, x + 0.108, y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2], 'pec')
    box(x + 0.032 + cavitysize[0], y + 0.022, z + skidthickness + cavitysize[2] - casethickness, x + 0.108, y + 0.022 + 0.006, z + skidthickness + cavitysize[2], 'pec')
    
    # PCB - replace bits chopped by TX & Rx cavities
    box(x + 0.032 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.032 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + pcbthickness, 'pcb')
    box(x + 0.108 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.108 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + pcbthickness, 'pcb')
    
    # PCB components
    # Tx bowtie
    triangle(tx[0], tx[1] - 0.001, tx[2], tx[0] - 0.026, tx[1] - bowtieheight - 0.001, tx[2], tx[0] + 0.026, tx[1] - bowtieheight - 0.001, tx[2], 0, 'pec')
    edge(tx[0], tx[1] - 0.001, tx[2], tx[0], tx[1], tx[2], 'pec')
    triangle(tx[0], tx[1] + 0.002, tx[2], tx[0] - 0.026, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.026, tx[1] + bowtieheight + 0.002, tx[2], 0, 'pec')
    edge(tx[0], tx[1] + 0.001, tx[2], tx[0], tx[1] + 0.002, tx[2], 'pec')
    
    # Rx bowtie
    triangle(tx[0] + 0.076, tx[1] - 0.001, tx[2], tx[0] + 0.076 - 0.026, tx[1] - bowtieheight - 0.001, tx[2], tx[0] + 0.076 + 0.026, tx[1] - bowtieheight - 0.001, tx[2], 0, 'pec')
    edge(tx[0] + 0.076, tx[1] - 0.001, tx[2], tx[0] + 0.076, tx[1], tx[2], 'pec')
    triangle(tx[0] + 0.076, tx[1] + 0.002, tx[2], tx[0] + 0.076 - 0.026, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076 + 0.026, tx[1] + bowtieheight + 0.002, tx[2], 0, 'pec')
    edge(tx[0] + 0.076, tx[1] + 0.001, tx[2], tx[0] + 0.076, tx[1] + 0.002, tx[2], 'pec')
    
    # Tx surface mount resistors (lower y coordinate)
    if resolution == 0.001:
        edge(tx[0] - 0.023, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023, tx[1] - bowtieheight - dy, tx[2], 'txreslower')
        edge(tx[0] - 0.023 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx, tx[1] - bowtieheight - dy, tx[2], 'txreslower')
        edge(tx[0], tx[1] - bowtieheight - 0.004, tx[2], tx[0], tx[1] - bowtieheight - dy, tx[2], 'txreslower')
        edge(tx[0] + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx, tx[1] - bowtieheight - dy, tx[2], 'txreslower')
        edge(tx[0] + 0.022, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022, tx[1] - bowtieheight - dy, tx[2], 'txreslower')
        edge(tx[0] + 0.022 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + dx, tx[1] - bowtieheight - dy, tx[2], 'txreslower')
    elif resolution == 0.002:
        edge(tx[0] - 0.023, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023, tx[1] - bowtieheight, tx[2], 'txreslower')
        edge(tx[0] - 0.023 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx, tx[1] - bowtieheight, tx[2], 'txreslower')
        edge(tx[0], tx[1] - bowtieheight - 0.004, tx[2], tx[0], tx[1] - bowtieheight, tx[2], 'txreslower')
        edge(tx[0] + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx, tx[1] - bowtieheight, tx[2], 'txreslower')
        edge(tx[0] + 0.020, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020, tx[1] - bowtieheight, tx[2], 'txreslower')
        edge(tx[0] + 0.020 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + dx, tx[1] - bowtieheight, tx[2], 'txreslower')
    
    # Tx surface mount resistors (upper y coordinate)
    if resolution == 0.001:
        edge(tx[0] - 0.023, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0], tx[1] + bowtieheight + 0.002, tx[2], tx[0], tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0] + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0] + 0.022, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0] + 0.022 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
    elif resolution == 0.002:
        edge(tx[0] - 0.023, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0], tx[1] + bowtieheight + 0.002, tx[2], tx[0], tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0] + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0] + 0.020, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
        edge(tx[0] + 0.020 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + dx, tx[1] + bowtieheight + 0.006, tx[2], 'txresupper')
    
    # Rx surface mount resistors (lower y coordinate)
    if resolution == 0.001:
        edge(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower')
        edge(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower')
        edge(tx[0] + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower')
        edge(tx[0] + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower')
        edge(tx[0] + 0.022 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower')
        edge(tx[0] + 0.022 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + dx + 0.076, tx[1] - bowtieheight - dy, tx[2], 'rxreslower')
    elif resolution == 0.002:
        edge(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower')
        edge(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower')
        edge(tx[0] + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower')
        edge(tx[0] + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower')
        edge(tx[0] + 0.020 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower')
        edge(tx[0] + 0.020 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + dx + 0.076, tx[1] - bowtieheight, tx[2], 'rxreslower')
    
    # Rx surface mount resistors (upper y coordinate)
    if resolution == 0.001:
        edge(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] + 0.022 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] + 0.022 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
    elif resolution == 0.002:
        edge(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] + 0.020 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
        edge(tx[0] + 0.020 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2], 'rxresupper')
    
    # Skid
    box(x, y, z, x + casesize[0], y + casesize[1], z + polypropylenethickness, 'polypropylene')
    box(x, y, z + polypropylenethickness, x + casesize[0], y + casesize[1], z + polypropylenethickness + hdpethickness, 'hdpe')

    # Geometry views
    #geometry_view(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + casesize[2] + skidthickness + dz, dx, dy, dz, 'antenna_like_MALA_1200')
    #geometry_view(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz, 'antenna_like_MALA_1200_pcb', type='f')
    
    # Excitation
    print('#waveform: gaussian 1.0 {} myGaussian'.format(excitationfreq))
    print('#voltage_source: y {} {} {} {} myGaussian'.format(tx[0], tx[1], tx[2], sourceresistance))
    
    # Output point - transmitter bowtie
    #print('#rx: {} {} {}'.format(tx[0], tx[1], tx[2]))
    # Output point - receiver bowtie
    print('#rx: {} {} {} rxbowtie Ey'.format(tx[0] + 0.076, tx[1], tx[2]))

