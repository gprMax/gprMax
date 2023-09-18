"""GPR antenna model (like a GSSI 400MHz antenna) over layered media with a
    rough subsurface interface.

This example model demonstrates how to use subgrids at a more advanced level -
    combining use of an imported antenna model and rough subsurface interface.

The geometry is 3D (required for any use of subgrids) and is of a 2 layered
subsurface. The top layer in a sandy soil and the bottom layer a soil with
higher permittivity (both have some simple conductive loss). There is a rough
interface between the soil layers. A GPR antenna model (like a GSSI 400MHz
antenna) is imported and placed on the surface of the layered media. The antenna
is meshed using a subgrid with a fine spatial discretisation (1mm), and a
courser spatial discretisation (9mm) is used in the rest of the model (main
grid).
"""

from pathlib import Path

import numpy as np

import gprMax
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_400

# File path - used later to specify name of output files
fn = Path(__file__)
parts = fn.parts

# Subgrid spatial discretisation in x, y, z directions
dl_sg = 1e-3

# Subgrid ratio - must always be an odd integer multiple
ratio = 9
dl = dl_sg * ratio

# Domain extent
x = 3
y = 1
z = 2

# Time window
# Estimated two way travel time over 1 metre in material with highest
# permittivity, slowest velocity.
tw = 2 / 3e8 * (np.sqrt(3.2) + np.sqrt(9))

scene = gprMax.Scene()

title = gprMax.Title(name=fn.name)
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
domain = gprMax.Domain(p1=(x, y, z))
time_window = gprMax.TimeWindow(time=tw)

scene.add(title)
scene.add(dxdydz)
scene.add(domain)
scene.add(time_window)

# Dimensions of antenna case
antenna_case = (0.3, 0.3, 0.178)

# Position of antenna
antenna_p = (x / 2, y / 2, 170 * dl)

# Extra distance surrounding antenna for subgrid
bounding_box = 2 * dl

# Subgrid extent
sg_x0 = antenna_p[0] - antenna_case[0] / 2 - bounding_box
sg_y0 = antenna_p[1] - antenna_case[1] / 2 - bounding_box
sg_z0 = antenna_p[2] - bounding_box
sg_x1 = antenna_p[0] + antenna_case[0] / 2 + bounding_box
sg_y1 = antenna_p[1] + antenna_case[1] / 2 + bounding_box
sg_z1 = antenna_p[2] + antenna_case[2] + bounding_box

# Create subgrid
sg = gprMax.SubGridHSG(p1=[sg_x0, sg_y0, sg_z0], p2=[sg_x1, sg_y1, sg_z1], ratio=ratio, id="sg")
scene.add(sg)

# Create and add a box of homogeneous material to main grid - sandy_soil
sandy_soil = gprMax.Material(er=3.2, se=0.397e-3, mr=1, sm=0, id="sandy_soil")
scene.add(sandy_soil)
b1 = gprMax.Box(p1=(0, 0, 0), p2=(x, y, antenna_p[2]), material_id="sandy_soil")
scene.add(b1)

# Position box of sandy_soil in the subgrid.
#   It has to be positioned manually because it traverses the main grid/subgrid
#   interface. Grid traversal is when objects extend beyond the outer surface.
#   Setting autotranslate to false allows you to place objects beyond the outer
#   surface.

# PML separation from the outer surface
ps = ratio // 2 + 2
# Number of PML cells in the subgrid
pc = 6
# Inner surface/outer surface separation
isos = 3 * ratio

# Calculate maximum z-coordinate (height) for box of sandy_soil in subgrid
h = antenna_p[2] - sg_z0 + (ps + pc + isos) * dl_sg

# Create and add a box of homogeneous material to subgrid - sandy_soil
sg.add(sandy_soil)
b2 = gprMax.Box(p1=(0, 0, 0), p2=(411 * dl_sg, 411 * dl_sg, h), material_id="sandy_soil")
# Set autotranslate for the box object to false
b2.autotranslate = False
sg.add(b2)

# Import antenna model and add components to subgrid
gssi_objects = antenna_like_GSSI_400(*antenna_p, resolution=dl_sg)
for obj in gssi_objects:
    sg.add(obj)

# Create and add a homogeneous material with a rough surface
soil = gprMax.Material(er=9, se=0.397e-3, mr=1, sm=0, id="soil")
scene.add(soil)

fb = gprMax.FractalBox(
    p1=(0, 0, 0),
    p2=(3, 1, 1),
    frac_dim=1.5,
    weighting=(1, 1, 1),
    n_materials=1,
    mixing_model_id="soil",
    id="fbox",
    seed=1,
)
scene.add(fb)

rough_surf = gprMax.AddSurfaceRoughness(
    p1=(0, 0, 1), p2=(3, 1, 1), frac_dim=1.5, weighting=(1, 1), limits=(0.4, 1.2), fractal_box_id="fbox", seed=1
)
scene.add(rough_surf)

# Create some snapshots and geometry views
for i in range(1, 51):
    snap = gprMax.Snapshot(
        p1=(0, y / 2, 0),
        p2=(x, y / 2 + dl, z),
        dl=(dl, dl, dl),
        filename=Path(*parts[:-1], f"{parts[-1]}_{str(i)}").name,
        time=i * tw / 50,
    )
    scene.add(snap)

gvsg = gprMax.GeometryView(
    p1=(sg_x0, sg_y0, sg_z0),
    p2=(sg_x1, sg_y1, sg_z1),
    dl=(dl_sg, dl_sg, dl_sg),
    filename=fn.with_suffix("").parts[-1] + "_sg",
    output_type="n",
)
sg.add(gvsg)

gv1 = gprMax.GeometryView(
    p1=(0, 0, 0), p2=domain.props.p1, dl=dl, filename=fn.with_suffix("").parts[-1], output_type="n"
)
scene.add(gv1)

gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile=fn, subgrid=True, autotranslate=True)
