# Copyright (C) 2015-2021, John Hartley
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.

from pathlib import Path

import gprMax
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_400

import numpy as np

# file path step
fn = Path(__file__)
parts = fn.parts

# subgrid Discretisation is 1 mm in x, y, z directions. This allows us
# to model the geometry of the antenna
dl_s = 1e-3
# subgridding ratio. This must always be an odd integer multiple. In this case
# the main grid discrestisation is 9e-3 m.
ratio = 9
dl = dl_s * 9

# estimated return time for signal to propagate 1 metre and back
tw = 2 / 3e8 * (np.sqrt(3.2) + np.sqrt(9))

# domain extent
x = 3
y = 1
z = 2

scene = gprMax.Scene()

title_gpr = gprMax.Title(name=fn.name)
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
domain = gprMax.Domain(p1=(x, y, z))
time_window = gprMax.TimeWindow(time=tw)

scene.add(domain)
scene.add(title_gpr)
scene.add(dxdydz)
scene.add(time_window)

# half space material
halfspace_m = gprMax.Material(er=3.2, se=0.397e-3, mr=1, sm=0, id='soil')
scene.add(halfspace_m)

antenna_case = (0.3, 0.3, 0.178)

bounding_box = 2 * dl

# pml + boundary_cells + is_os + subgrid+boundary + half antenna
x0 = (10 + 15 + 5 + 2) * dl + antenna_case[0] / 2

#antenna_p = (x / 2, y / 2, z - 30 * dl - bounding_box - antenna_case[2])

# Position of antenna base
antenna_p = (x / 2, y / 2, 170 * dl)

sg_x0 = antenna_p[0] - antenna_case[0] / 2 - bounding_box
sg_y0 = antenna_p[1] - antenna_case[1] / 2 - bounding_box
sg_z0 = antenna_p[2] - bounding_box

sg_x1 = antenna_p[0] + antenna_case[0] / 2 + bounding_box
sg_y1 = antenna_p[1] + antenna_case[1] / 2 + bounding_box
sg_z1 = antenna_p[2] + antenna_case[2] + bounding_box

sg = gprMax.SubGridHSG(p1=[sg_x0, sg_y0, sg_z0], p2=[sg_x1, sg_y1, sg_z1], ratio=ratio, id='subgrid1')
scene.add(sg)

# half space box in main grid
halfspace = gprMax.Box(p1=(0, 0, 0), p2=(x, y, antenna_p[2]), material_id='soil')
scene.add(halfspace)

# position half space box in the subgrid. The halfspace has to be positioned
# manually because it traverses the grid. Grid traversal requires that objects extend
# beyond the OS. Turning off autotranslate allows you to place objects beyond the OS.

# PML seperation from the OS
ps = ratio // 2 + 2
# number of pml cells in the subgrid
pc = 6
# is os seperation
isos = 3 * ratio

h = antenna_p[2] - sg_z0 + (ps + pc + isos) * dl_s

# half space box
halfspace = gprMax.Box(p1=(0, 0, 0), p2=(411 * dl_s, 411 * dl_s, h), material_id='soil')
# position the box using local coordinates3e8 / 400e6
halfspace.autotranslate = False
sg.add(halfspace)

# Import the antenna model and add components to subgrid
gssi_objects = antenna_like_GSSI_400(*antenna_p, resolution=dl_s)
for obj in gssi_objects:
    sg.add(obj)

# half space box
halfspace = gprMax.Box(p1=(0, 0, 0), p2=(x, y, antenna_p[2]), material_id='soil')
scene.add(halfspace)

for i in range(1, 51):
    snap = gprMax.Snapshot(p1=(0, y / 2, 0),
                           p2=(x, y / 2 + dl, z),
                           dl=(dl, dl, dl),
                           filename=Path(*parts[:-1], f'{parts[-1]}_{str(i)}').name,
                           time=i * tw / 50,)
    scene.add(snap)

# create a geometry view of the main grid and the sub grid stitched together
gv1 = gprMax.GeometryView(p1=(sg_x0, sg_y0, sg_z0), p2=(sg_x1, sg_y1, sg_z1),
                         dl=(dl_s, dl_s, dl_s),
                         filename=fn.with_suffix('').parts[-1] + '_sg',
                         output_type='n')
#sg.add(gv1)

gv3 = gprMax.GeometryView(p1=(sg_x0, sg_y0, 1.512), p2=(sg_x1, sg_y1, 1.513),
                         dl=(dl_s, dl_s, dl_s),
                         filename=fn.with_suffix('').parts[-1] + '_sg',
                         output_type='f')
sg.add(gv3)

gv2 = gprMax.GeometryView(p1=(0, 0, 0),
                         p2=domain.props.p1,
                         dl=dl,
                         filename=fn.with_suffix('').parts[-1],
                         output_type='n')
scene.add(gv2)

# half space material
layer_m = gprMax.Material(er=9, se=0.397e-3, mr=1, sm=0, id='soil_2')
scene.add(layer_m)

fb = gprMax.FractalBox(p1=(0, 0, 0), p2=(3, 1, 1), frac_dim=1.5, weighting=(1, 1, 1), n_materials=1, mixing_model_id='soil_2', id='fbox', seed=1)
scene.add(fb)

sr = gprMax.AddSurfaceRoughness(p1=(0, 0, 1), p2=(3, 1, 1), frac_dim=1.5, weighting=(1, 1), limits=(0.4, 1.2), fractal_box_id='fbox', seed=1)
scene.add(sr)

gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=fn, subgrid=True, autotranslate=True)
