# Copyright (C) 2015-2023, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import os

from gprMax.exceptions import CmdInputError
from gprMax.input_cmd_funcs import *

userlibdir = os.path.dirname(os.path.abspath(__file__))


def antenna_like_MALA_1200(x, y, z, resolution=0.001, rotate90=False):
    """Inserts a description of an antenna similar to the MALA 1.2GHz antenna. Can be used with 1mm (default) or 2mm spatial resolution. The external dimensions of the antenna are 184x109x46mm. One output point is defined between the arms of the receiver bowtie. The bowties are aligned with the y axis so the output is the y component of the electric field (x component if the antenna is rotated 90 degrees).

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.
        rotate90 (bool): Rotate model 90 degrees CCW in xy plane.
    """

    # Antenna geometry properties
    casesize = (0.184, 0.109, 0.040)
    casethickness = 0.002
    cavitysize = (0.062, 0.062, 0.037)
    cavitythickness = 0.001
    pcbthickness = 0.002
    polypropylenethickness = 0.003
    hdpethickness = 0.003
    skidthickness = 0.006
    bowtieheight = 0.025

    # Set origin for rotation to geometric centre of antenna in x-y plane if required, and set output component for receiver
    if rotate90:
        rotate90origin = (x, y)
        output = 'Ex'
    else:
        rotate90origin = ()
        output = 'Ey'

    # Original optimised values from http://hdl.handle.net/1842/4074
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
        polypropylenethickness = 0.002
        hdpethickness = 0.004
        bowtieheight = 0.024
        tx = x + 0.062, y + 0.052, z + skidthickness
    else:
        raise CmdInputError('This antenna module can only be used with a spatial resolution of 1mm or 2mm')

    # SMD resistors - 3 on each Tx & Rx bowtie arm
    txres = 470  # Ohms
    txrescellupper = txres / 3  # Resistor over 3 cells
    txsigupper = ((1 / txrescellupper) * (dy / (dx * dz))) / 2  # Divide by number of parallel edges per resistor
    txrescelllower = txres / 4  # Resistor over 4 cells
    txsiglower = ((1 / txrescelllower) * (dy / (dx * dz))) / 2  # Divide by number of parallel edges per resistor
    rxres = 150  # Ohms
    rxrescellupper = rxres / 3  # Resistor over 3 cells
    rxsigupper = ((1 / rxrescellupper) * (dy / (dx * dz))) / 2  # Divide by number of parallel edges per resistor
    rxrescelllower = rxres / 4  # Resistor over 4 cells
    rxsiglower = ((1 / rxrescelllower) * (dy / (dx * dz))) / 2  # Divide by number of parallel edges per resistor

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
    # geometry_view(x - dx, y - dy, z - dz, x + casesize[0] + dx, y + casesize[1] + dy, z + casesize[2] + skidthickness + dz, dx, dy, dz, 'antenna_like_MALA_1200')
    # geometry_view(x, y, z, x + casesize[0], y + casesize[1], z + 0.010, dx, dy, dz, 'antenna_like_MALA_1200_pcb', type='f')

    # Excitation
    print('#waveform: gaussian 1.0 {} myGaussian'.format(excitationfreq))
    voltage_source('y', tx[0], tx[1], tx[2], sourceresistance, 'myGaussian', dxdy=(resolution, resolution), rotate90origin=rotate90origin)

    # Output point - receiver bowtie
    rx(tx[0] + 0.076, tx[1], tx[2], identifier='rxbowtie', to_save=[output], polarisation='y', dxdy=(resolution, resolution), rotate90origin=rotate90origin)
