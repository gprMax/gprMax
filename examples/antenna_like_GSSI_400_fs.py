"""An antenna model similar to a GSSI 400MHz antenna in free space

This example model demonstrates how to use one of the built-in antenna models.

The geometry is 3D and the domain filled with freespace (the default). The
antenna model method is imported from its toolbox and the objects that build the
antenna are iteratively added to the scene. The antenna can be rotated if
desired, by rotating the objects that it is built from before they are added to
the scene.
"""

from pathlib import Path

import gprMax
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_400

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.002

# Domain
x = 0.340
y = 0.340
z = 0.318

scene = gprMax.Scene()

title = gprMax.Title(name=fn.with_suffix("").name)
domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=6e-9)

scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)

# Import antenna model and add to model
ant_pos = (0.170, 0.170, 0.100)
gssi_objects = antenna_like_GSSI_400(ant_pos[0], ant_pos[1], ant_pos[2], resolution=dl)
for obj in gssi_objects:
    scene.add(obj)

gv1 = gprMax.GeometryView(
    p1=(0, 0, 0), p2=(x, y, z), dl=(dl, dl, dl), filename="antenna_like_GSSI_400", output_type="n"
)
gv2 = gprMax.GeometryView(
    p1=(ant_pos[0] - 0.150 / 2, ant_pos[1] - 0.150 / 2, ant_pos[2] - 0.050),
    p2=(ant_pos[0] + 0.150 / 2, ant_pos[1] + 0.150 / 2, ant_pos[2] + 0.010),
    dl=(dl, dl, dl),
    filename="antenna_like_GSSI_400_pcb",
    output_type="f",
)
scene.add(gv1)
scene.add(gv2)

# Run model
gprMax.run(scenes=[scene], geometry_only=False, outputfile=fn, gpu=None)
