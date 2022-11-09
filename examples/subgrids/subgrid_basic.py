# Copyright (C) 2015-2021, John Hartley
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.

from pathlib import Path
import gprMax

# file path step
fn = Path(__file__)
parts = fn.parts

# Subgrid Discretisation in x, y, z directions.
dl_s = 1e-3

# Subgridding ratio. This must always be an odd integer multiple. 
ratio = 5
dl = dl_s * ratio

# Cells
# Default number of PML cells
pml_cells = 10
# Distance between model and PML cells
pml_gap = 15
# Number of cells between the Inner Surface and the Outer Surface of the sub-grid
is_os_gap = 4
# Size of the sub-gridded region
sub_gridded_region = 3
# Domain size
extent = sub_gridded_region + 2 * (pml_cells + pml_gap + is_os_gap)

# Domain extent
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

sg_x0 = (pml_cells + pml_gap + is_os_gap) * dl
sg_y0 = sg_x0
sg_z0 = sg_x0

sg_x1 = sg_x0 + sub_gridded_region * dl
sg_y1 = sg_x1
sg_z1 = sg_x1

sg_p0 = [sg_x0, sg_y0, sg_z0]
sg_p1 = [sg_x1, sg_y1, sg_z1]

sg = gprMax.SubGridHSG(p1=sg_p0, p2=sg_p1, ratio=ratio, id='mysubgrid')
scene.add(sg)

# Plastic box in sub grid
material = gprMax.Material(er=3, mr=1, se=0, sm=0, id='plastic')
scene.add(material)
plastic_box = gprMax.Box(p1=(30*dl, 30*dl, 30*dl), p2=(31*dl, 31*dl, 31*dl), material_id='plastic')
sg.add(plastic_box)

# Create a geometry view of the sub grid only. This command currently exports the entire subgrid regardless of p1, p2
gv_sg_normal = gprMax.GeometryView(p1=sg_p0,
                         p2=sg_p1,
                         dl=(1e-3, 1e-3, 1e-3),
                         filename=fn.with_suffix('').parts[-1] + '_subgrid_normal',
                         output_type='n')

# Add the subgrid geometry view to the sub grid object 
sg.add(gv_sg_normal)

gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile=fn, subgrid=True, autotranslate=True)
