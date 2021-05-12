# Copyright (C) 2015-2021, John Hartley
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.

from pathlib import Path
import sys
import gprMax
from user_libs.antennas.GSSI import antenna_like_GSSI_400

import numpy as np

# file path step
fn = Path(__file__)
parts = fn.parts

ratio = 5
dl = 5e-3

# cells
# default number of pml cells
pml_cells = 10
# distance between model and pml cells
pml_gap = 15
# number of cells between the Inner Surface and the Outer Surface of the sub-grid
is_os_gap = 4
# size of the sub-gridded region
sub_gridded_region = 3
# domain size
extent = 3 + 2 * (pml_cells + pml_gap + 4)

# domain extent
x = dl * extent
y = x
z = x

tw = 1e-9

scene = gprMax.Scene()

title_gpr = gprMax.Title(name=fn.name)
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
domain = gprMax.Domain(p1=(x, y, z))
time_window = gprMax.TimeWindow(time=tw)

scene.add(domain)
scene.add(title_gpr)
scene.add(dxdydz)
scene.add(time_window)

# plastic box in sub grid
material = gprMax.Material(er=3, mr=1, se=0, sm=0, id='plastic')
scene.add(material)

plastic_box = gprMax.Box(p1=(20*dl, 20*dl, 20*dl), p2=(40*dl, 40*dl, 30*dl), material_id='plastic')
scene.add(plastic_box)

pec_box = gprMax.Box(p1=(20*dl, 20*dl, 30*dl), p2=(40*dl, 40*dl, 40*dl), material_id='pec')
scene.add(pec_box)

pec_plate = gprMax.Plate(p1=(20*dl, 20*dl, 20*dl), p2=(40*dl, 40*dl, 20*dl), material_id='pec')
scene.add(pec_plate)

# create a geometry view of the main grid and the sub grid stitched together
gv1 = gprMax.GeometryView(p1=(0, 0, 0),
                         p2=domain.props.p1,
                         dl=dl,
                         filename=fn.with_suffix('').parts[-1],
                         output_type='f')

# create a geometry view of the main grid and the sub grid stitched together
gv2 = gprMax.GeometryView(p1=(0, 0, 0),
                         p2=domain.props.p1,
                         dl=dl,
                         filename=fn.with_suffix('').parts[-1],
                         output_type='n')


scene.add(gv1)
scene.add(gv2)

gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile=fn)
