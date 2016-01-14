# Copyright (C) 2015, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import os

from gprMax.exceptions import CmdInputError

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
    if 'kwargs' in locals():
        excitationfreq = kwargs['excitationfreq']
        sourceresistance = kwargs['sourceresistance']
        absorberEr = kwargs['absorberEr']
        absorbersig = kwargs['absorbersig']
    else:
        excitationfreq = 1.5e9 # GHz
        # excitationfreq = 1.71e9 # Value from http://hdl.handle.net/1842/4074
        sourceresistance = 50 # Ohms
        # sourceresistance = 4 # Value from http://hdl.handle.net/1842/4074
        absorberEr = 1.7
        # absorberEr = 1.58 # Value from http://hdl.handle.net/1842/4074
        absorbersig = 0.59
        # absorbersig = 0.428 # Value from http://hdl.handle.net/1842/4074
    
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
    print('#material: {:.2f} {:.3f} 1 0 absorber'.format(absorberEr, absorbersig))
    print('#material: 3 0 1 0 pcb')
    print('#material: 2.35 0 1 0 hdpe')

    # Antenna geometry
    # Plastic case
    print('#box: {} {} {} {} {} {} hdpe'.format(x, y, z + skidthickness, x + casesize[0], y + casesize[1], z + skidthickness + casesize[2]))
    print('#box: {} {} {} {} {} {} free_space'.format(x + casethickness, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness))
    
    # Metallic enclosure
    print('#box: {} {} {} {} {} {} pec'.format(x + 0.025, y + casethickness, z + skidthickness, x + casesize[0] - 0.025, y + casesize[1] - casethickness, z + skidthickness + 0.027))
    
    # Absorber material, and foam (modelled as PCB material) around edge of absorber
    print('#box: {} {} {} {} {} {} pcb'.format(x + 0.025 + shieldthickness, y + casethickness + shieldthickness, z + skidthickness, x + 0.025 + shieldthickness + 0.057, y + casesize[1] - casethickness - shieldthickness, z + skidthickness + 0.027 - shieldthickness - 0.001))
    print('#box: {} {} {} {} {} {} absorber'.format(x + 0.025 + shieldthickness + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.025 + shieldthickness + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + 0.027 - shieldthickness))
    print('#box: {} {} {} {} {} {} pcb'.format(x + 0.086, y + casethickness + shieldthickness, z + skidthickness, x + 0.086 + 0.057, y + casesize[1] - casethickness - shieldthickness, z + skidthickness + 0.027 - shieldthickness - 0.001))
    print('#box: {} {} {} {} {} {} absorber'.format(x + 0.086 + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + 0.027 - shieldthickness))
    
    # PCB
    print('#box: {} {} {} {} {} {} pcb'.format(x + 0.025 + shieldthickness + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 - shieldthickness - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + pcbthickness))
    print('#box: {} {} {} {} {} {} pcb'.format(x + 0.086 + foamsurroundthickness, y + casethickness + shieldthickness + foamsurroundthickness, z + skidthickness, x + 0.086 + 0.057 - foamsurroundthickness, y + casesize[1] - casethickness - shieldthickness - foamsurroundthickness, z + skidthickness + pcbthickness))
    
    # PCB components
    if resolution == 0.001:
        # Rx & Tx bowties
        a = 0
        b = 0
        while b < 13:
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.045 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness))
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.045 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness))
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.104 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness))
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.104 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness))
            b += 1
            if a == 2 or a == 4 or a == 7:
                print('#plate: {} {} {} {} {} {} pec'.format(x + 0.045 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness))
                print('#plate: {} {} {} {} {} {} pec'.format(x + 0.045 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.065 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness))
                print('#plate: {} {} {} {} {} {} pec'.format(x + 0.104 + a*dx, y + 0.039 + b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.039 + b*dx + dy, z + skidthickness))
                print('#plate: {} {} {} {} {} {} pec'.format(x + 0.104 + a*dx, y + 0.067 - b*dx, z + skidthickness, x + 0.124 - a*dx, y + 0.067 - b*dx + dy, z + skidthickness))
                b += 1
            a += 1
        # Rx extension section (upper y)
        print('#plate: {} {} {} {} {} {} pec'.format(x + 0.044, y + 0.068, z + skidthickness, x + 0.044 + bowtiebase, y + 0.068 + patchheight, z + skidthickness))
        # Tx extension section (upper y)
        print('#plate: {} {} {} {} {} {} pec'.format(x + 0.103, y + 0.068, z + skidthickness, x + 0.103 + bowtiebase, y + 0.068 + patchheight, z + skidthickness))
        
        # Edges that represent wire between bowtie halves in 1mm model
        print('#edge: {} {} {} {} {} {} pec'.format(tx[0] - 0.059, tx[1] - dy, tx[2], tx[0] - 0.059, tx[1], tx[2]))
        print('#edge: {} {} {} {} {} {} pec'.format(tx[0] - 0.059, tx[1] + dy, tx[2], tx[0] - 0.059, tx[1] + 0.002, tx[2]))
        print('#edge: {} {} {} {} {} {} pec'.format(tx[0], tx[1] - dy, tx[2], tx[0], tx[1], tx[2]))
        print('#edge: {} {} {} {} {} {} pec'.format(tx[0], tx[1] + dz, tx[2], tx[0], tx[1] + 0.002, tx[2]))

    elif resolution == 0.002:
    # Rx & Tx bowties
        for a in range(0,6):
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.044 + a*dx, y + 0.040 + a*dx, z + skidthickness, x + 0.066 - a*dx, y + 0.040 + a*dx + dy, z + skidthickness))
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.044 + a*dx, y + 0.064 - a*dx, z + skidthickness, x + 0.066 - a*dx, y + 0.064 - a*dx + dy, z + skidthickness))
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.103 + a*dx, y + 0.040 + a*dx, z + skidthickness, x + 0.125 - a*dx, y + 0.040 + a*dx + dy, z + skidthickness))
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.103 + a*dx, y + 0.064 - a*dx, z + skidthickness, x + 0.125 - a*dx, y + 0.064 - a*dx + dy, z + skidthickness))
            # Rx extension section (upper y)
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.044, y + 0.066, z + skidthickness, x + 0.044 + bowtiebase, y + 0.066 + patchheight, z + skidthickness))
            # Tx extension section (upper y)
            print('#plate: {} {} {} {} {} {} pec'.format(x + 0.103, y + 0.066, z + skidthickness, x + 0.103 + bowtiebase, y + 0.066 + patchheight, z + skidthickness))

    # Rx extension section (lower y)
    print('#plate: {} {} {} {} {} {} pec'.format(x + 0.044, y + 0.024, z + skidthickness, x + 0.044 + bowtiebase, y + 0.024 + patchheight, z + skidthickness))
    # Tx extension section (lower y)
    print('#plate: {} {} {} {} {} {} pec'.format(x + 0.103, y + 0.024, z + skidthickness, x + 0.103 + bowtiebase, y + 0.024 + patchheight, z + skidthickness))

    # Skid
    print('#box: {} {} {} {} {} {} hdpe'.format(x, y, z, x + casesize[0], y + casesize[1], z + skidthickness))
    
    # Geometry views
    #print('#geometry_view: {} {} {} {} {} {} {} {} {} antenna_like_GSSI_1500 n'.format(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + skidthickness + casesize[2] + dz, dx, dy, dz))
    #print('#geometry_view: {} {} {} {} {} {} {} {} {} antenna_like_GSSI_1500_pcb f'.format(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz))
    
    # Excitation - custom pulse
#    print('#excitation_file: {}'.format(os.path.join(moduledirectory, 'GSSIgausspulse.txt')))
#    print('#transmission_line: y {} {} {} {} GSSIgausspulse'.format(tx[0], tx[1], tx[2], sourceresistance))

    # Excitation - Gaussian pulse
    print('#waveform: gaussian 1 {} myGaussian'.format(excitationfreq))
    print('#voltage_source: y {} {} {} {} myGaussian'.format(tx[0], tx[1], tx[2], sourceresistance))
    
    # Output point - transmitter bowtie
    #print('#rx: {} {} {}'.format(tx[0], tx[1], tx[2]))
    # Output point - receiver bowtie
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
    if 'kwargs' in locals():
        excitationfreq = kwargs['excitationfreq']
        sourceresistance = kwargs['sourceresistance']
        absorberEr = kwargs['absorberEr']
        absorbersig = kwargs['absorbersig']
    else:
        excitationfreq = 0.978e9 # GHz
        sourceresistance = 1000 # Ohms
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
    print('#material: {:.2f} {:.3f} 1 0 absorber'.format(absorberEr, absorbersig))
    print('#material: 3 0 1 0 pcb')
    print('#material: 2.35 0 1 0 hdpe')
    print('#material: 2.26 0 1 0 polypropylene')
    print('#material: 3 {:.3f} 1 0 txreslower'.format(txsiglower))
    print('#material: 3 {:.3f} 1 0 txresupper'.format(txsigupper))
    print('#material: 3 {:.3f} 1 0 rxreslower'.format(rxsiglower))
    print('#material: 3 {:.3f} 1 0 rxresupper'.format(rxsigupper))
    
    # Antenna geometry
    # Shield - metallic enclosure
    print('#box: {} {} {} {} {} {} pec'.format(x, y, z + skidthickness, x + casesize[0], y + casesize[1], z + skidthickness + casesize[2]))
    print('#box: {} {} {} {} {} {} free_space'.format(x + 0.020, y + casethickness, z + skidthickness, x + 0.100, y + casesize[1] - casethickness, z + skidthickness + casethickness))
    print('#box: {} {} {} {} {} {} free_space'.format(x + 0.100, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casethickness))
    
    # Absorber material
    print('#box: {} {} {} {} {} {} absorber'.format(x + 0.020, y + casethickness, z + skidthickness, x + 0.100, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness))
    print('#box: {} {} {} {} {} {} absorber'.format(x + 0.100, y + casethickness, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - casethickness, z + skidthickness + casesize[2] - casethickness))
    
    # Shield - cylindrical sections
    print('#cylinder: {} {} {} {} {} {} {} pec'.format(x + 0.055, y + casesize[1] - 0.008, z + skidthickness, x + 0.055, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.008))
    print('#cylinder: {} {} {} {} {} {} {} pec'.format(x + 0.055, y + 0.008, z + skidthickness, x + 0.055, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.008))
    print('#cylinder: {} {} {} {} {} {} {} pec'.format(x + 0.147, y + casesize[1] - 0.008, z + skidthickness, x + 0.147, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.008))
    print('#cylinder: {} {} {} {} {} {} {} pec'.format(x + 0.147, y + 0.008, z + skidthickness, x + 0.147, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.008))
    print('#cylinder: {} {} {} {} {} {} {} free_space'.format(x + 0.055, y + casesize[1] - 0.008, z + skidthickness, x + 0.055, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.007))
    print('#cylinder: {} {} {} {} {} {} {} free_space'.format(x + 0.055, y + 0.008, z + skidthickness, x + 0.055, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.007))
    print('#cylinder: {} {} {} {} {} {} {} free_space'.format(x + 0.147, y + casesize[1] - 0.008, z + skidthickness, x + 0.147, y + casesize[1] - 0.008, z + skidthickness + casesize[2] - casethickness, 0.007))
    print('#cylinder: {} {} {} {} {} {} {} free_space'.format(x + 0.147, y + 0.008, z + skidthickness, x + 0.147, y + 0.008, z + skidthickness + casesize[2] - casethickness, 0.007))
    print('#box: {} {} {} {} {} {} free_space'.format(x + 0.054, y + casesize[1] - 0.016, z + skidthickness, x + 0.056, y + casesize[1] - 0.014, z + skidthickness + casesize[2] - casethickness))
    print('#box: {} {} {} {} {} {} free_space'.format(x + 0.054, y + 0.014, z + skidthickness, x + 0.056, y + 0.016, z + skidthickness + casesize[2] - casethickness))
    print('#box: {} {} {} {} {} {} free_space'.format(x + 0.146, y + casesize[1] - 0.016, z + skidthickness, x + 0.148, y + casesize[1] - 0.014, z + skidthickness + casesize[2] - casethickness))
    print('#box: {} {} {} {} {} {} free_space'.format(x + 0.146, y + 0.014, z + skidthickness, x + 0.148, y + 0.016, z + skidthickness + casesize[2] - casethickness))
    
    # PCB
    print('#box: {} {} {} {} {} {} pcb'.format(x + 0.020, y + 0.018, z + skidthickness, x + casesize[0] - casethickness, y + casesize[1] - 0.018, z + skidthickness + pcbthickness))
    
    # Shield - Tx & Rx cavities
    print('#box: {} {} {} {} {} {} pec'.format(x + 0.032, y + 0.022, z + skidthickness, x + 0.032 + cavitysize[0], y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2]))
    print('#box: {} {} {} {} {} {} absorber'.format(x + 0.032 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.032 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + cavitysize[2]))
    print('#box: {} {} {} {} {} {} pec'.format(x + 0.108, y + 0.022, z + skidthickness, x + 0.108 + cavitysize[0], y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2]))
    print('#box: {} {} {} {} {} {} free_space'.format(x + 0.108 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.108 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + cavitysize[2]))
    
    # Shield - Tx & Rx cavities - joining strips
    print('#box: {} {} {} {} {} {} pec'.format(x + 0.032 + cavitysize[0], y + 0.022 + cavitysize[1] - 0.006, z + skidthickness + cavitysize[2] - casethickness, x + 0.108, y + 0.022 + cavitysize[1], z + skidthickness + cavitysize[2]))
    print('#box: {} {} {} {} {} {} pec'.format(x + 0.032 + cavitysize[0], y + 0.022, z + skidthickness + cavitysize[2] - casethickness, x + 0.108, y + 0.022 + 0.006, z + skidthickness + cavitysize[2]))
    
    # PCB - replace bits chopped by TX & Rx cavities
    print('#box: {} {} {} {} {} {} pcb'.format(x + 0.032 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.032 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + pcbthickness))
    print('#box: {} {} {} {} {} {} pcb'.format(x + 0.108 + cavitythickness, y + 0.022 + cavitythickness, z + skidthickness, x + 0.108 + cavitysize[0] - cavitythickness, y + 0.022 + cavitysize[1] - cavitythickness, z + skidthickness + pcbthickness))
    
    # PCB components
    # Tx bowtie
    print('#triangle: {} {} {} {} {} {} {} {} {} 0 pec'.format(tx[0], tx[1] - 0.001, tx[2], tx[0] - 0.026, tx[1] - bowtieheight - 0.001, tx[2], tx[0] + 0.026, tx[1] - bowtieheight - 0.001, tx[2]))
    print('#edge: {} {} {} {} {} {} pec'.format(tx[0], tx[1] - 0.001, tx[2], tx[0], tx[1], tx[2]))
    print('#triangle: {} {} {} {} {} {} {} {} {} 0 pec'.format(tx[0], tx[1] + 0.002, tx[2], tx[0] - 0.026, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.026, tx[1] + bowtieheight + 0.002, tx[2]))
    print('#edge: {} {} {} {} {} {} pec'.format(tx[0], tx[1] + 0.001, tx[2], tx[0], tx[1] + 0.002, tx[2]))
    
    # Rx bowtie
    print('#triangle: {} {} {} {} {} {} {} {} {} 0 pec'.format(tx[0] + 0.076, tx[1] - 0.001, tx[2], tx[0] + 0.076 - 0.026, tx[1] - bowtieheight - 0.001, tx[2], tx[0] + 0.076 + 0.026, tx[1] - bowtieheight - 0.001, tx[2]))
    print('#edge: {} {} {} {} {} {} pec'.format(tx[0] + 0.076, tx[1] - 0.001, tx[2], tx[0] + 0.076, tx[1], tx[2]))
    print('#triangle: {} {} {} {} {} {} {} {} {} 0 pec'.format(tx[0] + 0.076, tx[1] + 0.002, tx[2], tx[0] + 0.076 - 0.026, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076 + 0.026, tx[1] + bowtieheight + 0.002, tx[2]))
    print('#edge: {} {} {} {} {} {} pec'.format(tx[0] + 0.076, tx[1] + 0.001, tx[2], tx[0] + 0.076, tx[1] + 0.002, tx[2]))
    
    # Tx surface mount resistors (lower y coordinate)
    if resolution == 0.001:
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] - 0.023, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] - 0.023 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0], tx[1] - bowtieheight - 0.004, tx[2], tx[0], tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] + 0.022, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] + 0.022 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + dx, tx[1] - bowtieheight - dy, tx[2]))
    elif resolution == 0.002:
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] - 0.023, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] - 0.023 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0], tx[1] - bowtieheight - 0.004, tx[2], tx[0], tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] + 0.020, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} txreslower'.format(tx[0] + 0.020 + dx, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + dx, tx[1] - bowtieheight, tx[2]))
    
    # Tx surface mount resistors (upper y coordinate)
    if resolution == 0.001:
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] - 0.023, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0], tx[1] + bowtieheight + 0.002, tx[2], tx[0], tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] + 0.022, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] + 0.022 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + dx, tx[1] + bowtieheight + 0.006, tx[2]))
    elif resolution == 0.002:
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] - 0.023, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0], tx[1] + bowtieheight + 0.002, tx[2], tx[0], tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] + 0.020, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} txresupper'.format(tx[0] + 0.020 + dx, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + dx, tx[1] + bowtieheight + 0.006, tx[2]))
    
    # Rx surface mount resistors (lower y coordinate)
    if resolution == 0.001:
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.076, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx + 0.076, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] + 0.022 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + 0.076, tx[1] - bowtieheight - dy, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] + 0.022 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.022 + dx + 0.076, tx[1] - bowtieheight - dy, tx[2]))
    elif resolution == 0.002:
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + 0.076, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.076, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + dx + 0.076, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] + 0.020 + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + 0.076, tx[1] - bowtieheight, tx[2]))
        print('#edge: {} {} {} {} {} {} rxreslower'.format(tx[0] + 0.020 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2], tx[0] + 0.020 + dx + 0.076, tx[1] - bowtieheight, tx[2]))
    
    # Rx surface mount resistors (upper y coordinate)
    if resolution == 0.001:
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] + 0.022 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] + 0.022 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.022 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
    elif resolution == 0.002:
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] + 0.020 + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
        print('#edge: {} {} {} {} {} {} rxresupper'.format(tx[0] + 0.020 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2], tx[0] + 0.020 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]))
    
    # Skid
    print('#box: {} {} {} {} {} {} polypropylene'.format(x, y, z, x + casesize[0], y + casesize[1], z + polypropylenethickness))
    print('#box: {} {} {} {} {} {} hdpe'.format(x, y, z + polypropylenethickness, x + casesize[0], y + casesize[1], z + polypropylenethickness + hdpethickness))

    # Geometry views
    #print('#geometry_view: {} {} {} {} {} {} {} {} {} antenna_like_MALA_1200 n'.format(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + casesize[2] + skidthickness + dz, dx, dy, dz))
    #print('#geometry_view: {} {} {} {} {} {} {} {} {} antenna_like_MALA_1200_pcb f'.format(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz))
    
    # Excitation
    print('#waveform: gaussian 1.0 {} myGaussian'.format(excitationfreq))
    print('#voltage_source: y {} {} {} {} myGaussian'.format(tx[0], tx[1], tx[2], sourceresistance))
    
    # Output point - transmitter bowtie
    #print('#rx: {} {} {}'.format(tx[0], tx[1], tx[2]))
    # Output point - receiver bowtie
    print('#rx: {} {} {} rxbowtie Ey'.format(tx[0] + 0.076, tx[1], tx[2]))

