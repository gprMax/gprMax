# Copyright (C) 2015-2020, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import logging

import gprMax

logger = logging.getLogger(__name__)


def antenna_like_MALA_1200(x, y, z, resolution=0.001):
    """Inserts a description of an antenna similar to the MALA 1.2GHz antenna.
        Can be used with 1mm (default) or 2mm spatial resolution.
        The external dimensions of the antenna are 184x109x46mm.
        One output point is defined between the arms of the receiver bowtie.
        The bowties are aligned with the y axis so the output is the y component
        of the electric field (x component if the antenna is rotated 90 degrees).

    Args:
        x, y, z (float): Coordinates of a location in the model to insert the antenna. Coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction.
        resolution (float): Spatial resolution for the antenna model.

    Returns:
        scene_objects (list): All model objects that will be part of a scene.
    """

    # All model objects that will be returned by function
    scene_objects = []

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
        logger.exception('This antenna module can only be used with a spatial resolution of 1mm or 2mm')
        raise ValueError

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
    absorber = gprMax.Material(er=absorberEr, se=absorbersig, mr=1, sm=0, id='absorber')
    pcb = gprMax.Material(er=3, se=0, mr=1, sm=0, id='pcb')
    hdpe = gprMax.Material(er=2.35, se=0, mr=1, sm=0, id='hdpe')
    polypropylene = gprMax.Material(er=2.26, se=0, mr=1, sm=0, id='polypropylene')
    txreslower = gprMax.Material(er=3, se=txsiglower, mr=1, sm=0, id='txreslower')
    txresupper = gprMax.Material(er=3, se=txsigupper, mr=1, sm=0, id='txresupper')
    rxreslower = gprMax.Material(er=3, se=rxsiglower, mr=1, sm=0, id='rxreslower')
    rxresupper = gprMax.Material(er=3, se=rxsigupper, mr=1, sm=0, id='rxresupper')
    scene_objects.extend((absorber, pcb, hdpe, polypropylene, txreslower, txresupper,
                        rxreslower, rxresupper))

    # Antenna geometry
    # Shield - metallic enclosure
    b1 = gprMax.Box(p1=(x, y, z + skidthickness),
                    p2=(x + casesize[0], y + casesize[1], z + skidthickness + casesize[2]),
                    material_id='pec')
    b2 = gprMax.Box(p1=(x + 0.020, y + casethickness, z + skidthickness),
                    p2=(x + 0.100, y + casesize[1] - casethickness,
                    z + skidthickness + casethickness), material_id='free_space')
    b3 = gprMax.Box(p1=(x + 0.100, y + casethickness, z + skidthickness),
                    p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                    z + skidthickness + casethickness), material_id='free_space')

    # Absorber material
    b4 = gprMax.Box(p1=(x + 0.020, y + casethickness, z + skidthickness),
                    p2=(x + 0.100, y + casesize[1] - casethickness,
                    z + skidthickness + casesize[2] - casethickness),
                    material_id='absorber')
    b5 = gprMax.Box(p1=(x + 0.100, y + casethickness, z + skidthickness),
                    p2=(x + casesize[0] - casethickness, y + casesize[1] - casethickness,
                    z + skidthickness + casesize[2] - casethickness),
                    material_id='absorber')
    scene_objects.extend((b1, b2, b3, b4, b5))

    # Shield - cylindrical sections
    c1 = gprMax.Cylinder(p1=(x + 0.055, y + casesize[1] - 0.008, z + skidthickness),
                         p2=(x + 0.055, y + casesize[1] - 0.008,
                         z + skidthickness + casesize[2] - casethickness),
                         r=0.008, material_id='pec')
    c2 = gprMax.Cylinder(p1=(x + 0.055, y + 0.008, z + skidthickness),
                         p2=(x + 0.055, y + 0.008,
                         z + skidthickness + casesize[2] - casethickness),
                         r=0.008, material_id='pec')
    c3 = gprMax.Cylinder(p1=(x + 0.147, y + casesize[1] - 0.008, z + skidthickness),
                         p2=(x + 0.147, y + casesize[1] - 0.008,
                         z + skidthickness + casesize[2] - casethickness),
                         r=0.008, material_id='pec')
    c4 = gprMax.Cylinder(p1=(x + 0.147, y + 0.008, z + skidthickness),
                         p2=(x + 0.147, y + 0.008,
                         z + skidthickness + casesize[2] - casethickness),
                         r=0.008, material_id='pec')
    c5 = gprMax.Cylinder(p1=(x + 0.055, y + casesize[1] - 0.008, z + skidthickness),
                         p2=(x + 0.055, y + casesize[1] - 0.008,
                         z + skidthickness + casesize[2] - casethickness),
                         r=0.007, material_id='free_space')
    c6 = gprMax.Cylinder(p1=(x + 0.055, y + 0.008, z + skidthickness),
                         p2=(x + 0.055, y + 0.008,
                         z + skidthickness + casesize[2] - casethickness),
                         r=0.007, material_id='free_space')
    c7 = gprMax.Cylinder(p1=(x + 0.147, y + casesize[1] - 0.008, z + skidthickness),
                         p2=(x + 0.147, y + casesize[1] - 0.008,
                         z + skidthickness + casesize[2] - casethickness),
                         r=0.007, material_id='free_space')
    c8 = gprMax.Cylinder(p1=(x + 0.147, y + 0.008, z + skidthickness),
                         p2=(x + 0.147, y + 0.008,
                         z + skidthickness + casesize[2] - casethickness),
                         r=0.007, material_id='free_space')
    b6 = gprMax.Box(p1=(x + 0.054, y + casesize[1] - 0.016, z + skidthickness),
                    p2=(x + 0.056, y + casesize[1] - 0.014,
                    z + skidthickness + casesize[2] - casethickness),
                    material_id='free_space')
    b7 = gprMax.Box(p1=(x + 0.054, y + 0.014, z + skidthickness),
                    p2=(x + 0.056, y + 0.016,
                    z + skidthickness + casesize[2] - casethickness),
                    material_id='free_space')
    b8 = gprMax.Box(p1=(x + 0.146, y + casesize[1] - 0.016, z + skidthickness),
                    p2=(x + 0.148, y + casesize[1] - 0.014,
                    z + skidthickness + casesize[2] - casethickness),
                    material_id='free_space')
    b9 = gprMax.Box(p1=(x + 0.146, y + 0.014, z + skidthickness),
                    p2=(x + 0.148, y + 0.016,
                    z + skidthickness + casesize[2] - casethickness),
                    material_id='free_space')
    scene_objects.extend((c1, c2, c3, c4, c5, c6, c7, c8, b6, b7, b8, b9))

    # PCB
    b10 = gprMax.Box(p1=(x + 0.020, y + 0.018, z + skidthickness),
                     p2=(x + casesize[0] - casethickness, y + casesize[1] - 0.018,
                     z + skidthickness + pcbthickness), material_id='pcb')

    # Shield - Tx & Rx cavities
    b11 = gprMax.Box(p1=(x + 0.032, y + 0.022, z + skidthickness),
                     p2=(x + 0.032 + cavitysize[0], y + 0.022 + cavitysize[1],
                     z + skidthickness + cavitysize[2]), material_id='pec')
    b12 = gprMax.Box(p1=(x + 0.032 + cavitythickness, y + 0.022 + cavitythickness,
                     z + skidthickness), p2=(x + 0.032 + cavitysize[0] - cavitythickness,
                     y + 0.022 + cavitysize[1] - cavitythickness,
                     z + skidthickness + cavitysize[2]), material_id='absorber')
    b13 = gprMax.Box(p1=(x + 0.108, y + 0.022, z + skidthickness),
                     p2=(x + 0.108 + cavitysize[0], y + 0.022 + cavitysize[1],
                     z + skidthickness + cavitysize[2]), material_id='pec')
    b14 = gprMax.Box(p1=(x + 0.108 + cavitythickness, y + 0.022 + cavitythickness,
                     z + skidthickness), p2=(x + 0.108 + cavitysize[0] - cavitythickness,
                     y + 0.022 + cavitysize[1] - cavitythickness,
                     z + skidthickness + cavitysize[2]), material_id='free_space')

    # Shield - Tx & Rx cavities - joining strips
    b15 = gprMax.Box(p1=(x + 0.032 + cavitysize[0], y + 0.022 + cavitysize[1] - 0.006,
                     z + skidthickness + cavitysize[2] - casethickness),
                     p2=(x + 0.108, y + 0.022 + cavitysize[1],
                     z + skidthickness + cavitysize[2]), material_id='pec')
    b16 = gprMax.Box(p1=(x + 0.032 + cavitysize[0], y + 0.022,
                     z + skidthickness + cavitysize[2] - casethickness),
                     p2=(x + 0.108, y + 0.022 + 0.006,
                     z + skidthickness + cavitysize[2]), material_id='pec')

    # PCB - replace bits chopped by TX & Rx cavities
    b17 = gprMax.Box(p1=(x + 0.032 + cavitythickness, y + 0.022 + cavitythickness,
                     z + skidthickness), p2=(x + 0.032 + cavitysize[0] - cavitythickness,
                     y + 0.022 + cavitysize[1] - cavitythickness,
                     z + skidthickness + pcbthickness), material_id='pcb')
    b18 = gprMax.Box(p1=(x + 0.108 + cavitythickness, y + 0.022 + cavitythickness,
                     z + skidthickness), p2=(x + 0.108 + cavitysize[0] - cavitythickness,
                     y + 0.022 + cavitysize[1] - cavitythickness,
                     z + skidthickness + pcbthickness), material_id='pcb')
    scene_objects.extend((b10, b11, b12, b13, b14, b15, b16, b17, b18))

    # PCB components
    # Tx bowtie
    if resolution == 0.001:
        t1 = gprMax.Triangle(p1=(tx[0], tx[1] - 0.001, tx[2]),
                             p2=(tx[0] - 0.026, tx[1] - bowtieheight - 0.001, tx[2]),
                             p3=(tx[0] + 0.026, tx[1] - bowtieheight - 0.001, tx[2]),
                             thickness=0, material_id='pec')
        e1 = gprMax.Edge(p1=(tx[0], tx[1] - 0.001, tx[2]),
                         p2=(tx[0], tx[1], tx[2]), material_id='pec')
        t2 = gprMax.Triangle(p1=(tx[0], tx[1] + 0.002, tx[2]),
                             p2=(tx[0] - 0.026, tx[1] + bowtieheight + 0.002, tx[2]),
                             p3=(tx[0] + 0.026, tx[1] + bowtieheight + 0.002, tx[2]),
                             thickness=0, material_id='pec')
        e2 = gprMax.Edge(p1=(tx[0], tx[1] + 0.001, tx[2]),
                         p2=(tx[0], tx[1] + 0.002, tx[2]), material_id='pec')
        scene_objects.extend((t1, t2, e1, e2))
    elif resolution == 0.002:
        t1 = gprMax.Triangle(p1=(tx[0], tx[1], tx[2]),
                             p2=(tx[0] - 0.026, tx[1] - bowtieheight, tx[2]),
                             p3=(tx[0] + 0.026, tx[1] - bowtieheight, tx[2]),
                             thickness=0, material_id='pec')
        t2 = gprMax.Triangle(p1=(tx[0], tx[1] + 0.002, tx[2]),
                             p2=(tx[0] - 0.026, tx[1] + bowtieheight + 0.002, tx[2]),
                             p3=(tx[0] + 0.026, tx[1] + bowtieheight + 0.002, tx[2]),
                             thickness=0, material_id='pec')
        scene_objects.extend((t1, t2))

    # Rx bowtie
    if resolution == 0.001:
        t3 = gprMax.Triangle(p1=(tx[0] + 0.076, tx[1] - 0.001, tx[2]),
                             p2=(tx[0] + 0.076 - 0.026, tx[1] - bowtieheight - 0.001, tx[2]),
                             p3=(tx[0] + 0.076 + 0.026, tx[1] - bowtieheight - 0.001, tx[2]),
                             thickness=0, material_id='pec')
        e3 = gprMax.Edge(p1=(tx[0] + 0.076, tx[1] - 0.001, tx[2]),
                         p2=(tx[0] + 0.076, tx[1], tx[2]), material_id='pec')
        t4 = gprMax.Triangle(p1=(tx[0] + 0.076, tx[1] + 0.002, tx[2]),
                             p2=(tx[0] + 0.076 - 0.026, tx[1] + bowtieheight + 0.002, tx[2]),
                             p3=(tx[0] + 0.076 + 0.026, tx[1] + bowtieheight + 0.002, tx[2]),
                             thickness=0, material_id='pec')
        e4 = gprMax.Edge(p1=(tx[0] + 0.076, tx[1] + 0.001, tx[2]),
                         p2=(tx[0] + 0.076, tx[1] + 0.002, tx[2]), material_id='pec')
        scene_objects.extend((t3, e3, t4, e4))
    elif resolution == 0.002:
        t3 = gprMax.Triangle(p1=(tx[0] + 0.076, tx[1], tx[2]),
                             p2=(tx[0] + 0.076 - 0.026, tx[1] - bowtieheight, tx[2]),
                             p3=(tx[0] + 0.076 + 0.026, tx[1] - bowtieheight, tx[2]),
                             thickness=0, material_id='pec')
        t4 = gprMax.Triangle(p1=(tx[0] + 0.076, tx[1] + 0.002, tx[2]),
                             p2=(tx[0] + 0.076 - 0.026, tx[1] + bowtieheight + 0.002, tx[2]),
                             p3=(tx[0] + 0.076 + 0.026, tx[1] + bowtieheight + 0.002, tx[2]),
                             thickness=0, material_id='pec')
        scene_objects.extend((t3, t4))

    # Tx surface mount resistors (lower y coordinate)
    if resolution == 0.001:
        e5 = gprMax.Edge(p1=(tx[0] - 0.023, tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0] - 0.023, tx[1] - bowtieheight - dy, tx[2]),
                         material_id='txreslower')
        e6 = gprMax.Edge(p1=(tx[0] - 0.023 + dx, tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0] - 0.023 + dx, tx[1] - bowtieheight - dy, tx[2]),
                         material_id='txreslower')
        e7 = gprMax.Edge(p1=(tx[0], tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0], tx[1] - bowtieheight - dy, tx[2]),
                         material_id='txreslower')
        e8 = gprMax.Edge(p1=(tx[0] + dx, tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0] + dx, tx[1] - bowtieheight - dy, tx[2]),
                         material_id='txreslower')
        e9 = gprMax.Edge(p1=(tx[0] + 0.022, tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0] + 0.022, tx[1] - bowtieheight - dy, tx[2]),
                         material_id='txreslower')
        e10 = gprMax.Edge(p1=(tx[0] + 0.022 + dx, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + 0.022 + dx, tx[1] - bowtieheight - dy, tx[2]),
                          material_id='txreslower')
        scene_objects.extend((e5, e6, e7, e8, e9, e10))
    elif resolution == 0.002:
        e5 = gprMax.Edge(p1=(tx[0] - 0.023, tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0] - 0.023, tx[1] - bowtieheight, tx[2]),
                         material_id='txreslower')
        e6 = gprMax.Edge(p1=(tx[0] - 0.023 + dx, tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0] - 0.023 + dx, tx[1] - bowtieheight, tx[2]),
                         material_id='txreslower')
        e7 = gprMax.Edge(p1=(tx[0], tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0], tx[1] - bowtieheight, tx[2]),
                         material_id='txreslower')
        e8 = gprMax.Edge(p1=(tx[0] + dx, tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0] + dx, tx[1] - bowtieheight, tx[2]),
                         material_id='txreslower')
        e9 = gprMax.Edge(p1=(tx[0] + 0.020, tx[1] - bowtieheight - 0.004, tx[2]),
                         p2=(tx[0] + 0.020, tx[1] - bowtieheight, tx[2]),
                         material_id='txreslower')
        e10 = gprMax.Edge(p1=(tx[0] + 0.020 + dx, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + 0.020 + dx, tx[1] - bowtieheight, tx[2]),
                          material_id='txreslower')
        scene_objects.extend((e5, e6, e7, e8, e9, e10))

    # Tx surface mount resistors (upper y coordinate)
    if resolution == 0.001:
        e11 = gprMax.Edge(p1=(tx[0] - 0.023, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] - 0.023, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e12 = gprMax.Edge(p1=(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e13 = gprMax.Edge(p1=(tx[0], tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0], tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e14 = gprMax.Edge(p1=(tx[0] + dx, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + dx, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e15 = gprMax.Edge(p1=(tx[0] + 0.022, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.022, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e16 = gprMax.Edge(p1=(tx[0] + 0.022 + dx, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.022 + dx, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        scene_objects.extend((e11, e12, e13, e14, e15, e16))
    elif resolution == 0.002:
        e11 = gprMax.Edge(p1=(tx[0] - 0.023, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] - 0.023, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e12 = gprMax.Edge(p1=(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] - 0.023 + dx, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e13 = gprMax.Edge(p1=(tx[0], tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0], tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e14 = gprMax.Edge(p1=(tx[0] + dx, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + dx, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e15 = gprMax.Edge(p1=(tx[0] + 0.020, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.020, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        e16 = gprMax.Edge(p1=(tx[0] + 0.020 + dx, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.020 + dx, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='txresupper')
        scene_objects.extend((e11, e12, e13, e14, e15, e16))

    # Rx surface mount resistors (lower y coordinate)
    if resolution == 0.001:
        e17 = gprMax.Edge(p1=(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - dy, tx[2]),
                          material_id='rxreslower')
        e18 = gprMax.Edge(p1=(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - dy, tx[2]),
                          material_id='rxreslower')
        e19 = gprMax.Edge(p1=(tx[0] + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + 0.076, tx[1] - bowtieheight - dy, tx[2]),
                          material_id='rxreslower')
        e20 = gprMax.Edge(p1=(tx[0] + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + dx + 0.076, tx[1] - bowtieheight - dy, tx[2]),
                          material_id='rxreslower')
        e21 = gprMax.Edge(p1=(tx[0] + 0.022 + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + 0.022 + 0.076, tx[1] - bowtieheight - dy, tx[2]),
                          material_id='rxreslower')
        e22 = gprMax.Edge(p1=(tx[0] + 0.022 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + 0.022 + dx + 0.076, tx[1] - bowtieheight - dy, tx[2]),
                          material_id='rxreslower')
        scene_objects.extend((e17, e18, e19, e20, e21, e22))
    elif resolution == 0.002:
        e17 = gprMax.Edge(p1=(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] - 0.023 + 0.076, tx[1] - bowtieheight, tx[2]),
                          material_id='rxreslower')
        e18 = gprMax.Edge(p1=(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] - 0.023 + dx + 0.076, tx[1] - bowtieheight, tx[2]),
                          material_id='rxreslower')
        e19 = gprMax.Edge(p1=(tx[0] + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + 0.076, tx[1] - bowtieheight, tx[2]),
                          material_id='rxreslower')
        e20 = gprMax.Edge(p1=(tx[0] + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + dx + 0.076, tx[1] - bowtieheight, tx[2]),
                          material_id='rxreslower')
        e21 = gprMax.Edge(p1=(tx[0] + 0.020 + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + 0.020 + 0.076, tx[1] - bowtieheight, tx[2]),
                          material_id='rxreslower')
        e22 = gprMax.Edge(p1=(tx[0] + 0.020 + dx + 0.076, tx[1] - bowtieheight - 0.004, tx[2]),
                          p2=(tx[0] + 0.020 + dx + 0.076, tx[1] - bowtieheight, tx[2]),
                          material_id='rxreslower')
        scene_objects.extend((e17, e18, e19, e20, e21, e22))

    # Rx surface mount resistors (upper y coordinate)
    if resolution == 0.001:
        e23 = gprMax.Edge(p1=(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e24 = gprMax.Edge(p1=(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e25 = gprMax.Edge(p1=(tx[0] + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e26 = gprMax.Edge(p1=(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e27 = gprMax.Edge(p1=(tx[0] + 0.022 + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.022 + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e28 = gprMax.Edge(p1=(tx[0] + 0.022 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.022 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        scene_objects.extend((e23, e24, e25, e26, e27, e28))
    elif resolution == 0.002:
        e23 = gprMax.Edge(p1=(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] - 0.023 + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e24 = gprMax.Edge(p1=(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] - 0.023 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e25 = gprMax.Edge(p1=(tx[0] + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e26 = gprMax.Edge(p1=(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e27 = gprMax.Edge(p1=(tx[0] + 0.020 + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.020 + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        e28 = gprMax.Edge(p1=(tx[0] + 0.020 + dx + 0.076, tx[1] + bowtieheight + 0.002, tx[2]),
                          p2=(tx[0] + 0.020 + dx + 0.076, tx[1] + bowtieheight + 0.006, tx[2]),
                          material_id='rxresupper')
        scene_objects.extend((e23, e24, e25, e26, e27, e28))

    # Skid
    b19 = gprMax.Box(p1=(x, y, z), p2=(x + casesize[0], y + casesize[1],
                    z + polypropylenethickness), material_id='polypropylene')
    b20 = gprMax.Box(p1=(x, y, z + polypropylenethickness),
                    p2=(x + casesize[0], y + casesize[1],
                    z + polypropylenethickness + hdpethickness),
                    material_id='hdpe')
    scene_objects.extend((b19, b20))

    # Excitation
    w2 = gprMax.Waveform(wave_type='gaussian', amp=1, freq=excitationfreq, id='my_gaussian')
    scene_objects.append(w2)
    vs1 = gprMax.VoltageSource(polarisation='y', p1=(tx[0], tx[1], tx[2]),
                               resistance=sourceresistance, waveform_id='my_gaussian')
    scene_objects.append(vs1)

    # Output point - receiver bowtie
    r1 = gprMax.Rx(p1=(tx[0] + 0.076, tx[1], tx[2]), id='rxbowtie', outputs='Ey')
    scene_objects.append(r1)

    # Geometry views
    gv1 = gprMax.GeometryView(p1=(x - dx, y - dy, z - dz), p2=(x + casesize[0] + dx,
                              y + casesize[1] + dy, z + skidthickness + casesize[2] + dz),
                              dl=(dx, dy, dz), filename='antenna_like_MALA_1200',
                              output_type='n')
    gv2 = gprMax.GeometryView(p1=(x, y, z), p2=(x + casesize[0], y + casesize[1], z + 0.010),
                              dl=(dx, dy, dz), filename='antenna_like_MALA_1200_pcb',
                              output_type='f')
    scene_objects.extend((gv1, gv2))

    return scene_objects
