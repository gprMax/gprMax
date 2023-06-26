"""An antenna model similar to a MALA 1.2GHz antenna in free space

This example model demonstrates how to use one of the built-in antenna models.

The geometry is 3D and the domain filled with freespace (the default). The
antenna model method is imported from its toolbox and the objects that build the
antenna are iteratively added to the scene.
"""

from pathlib import Path

import gprMax
from toolboxes.GPRAntennaModels.MALA import antenna_like_MALA_1200

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

# Domain
x = 0.264
y = 0.189
z = 0.220

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
ant_pos = (0.132, 0.095, 0.100)
mala_objects = antenna_like_MALA_1200(ant_pos[0], ant_pos[1], ant_pos[2], resolution=dl)
for obj in mala_objects:
    scene.add(obj)

# Run model
gprMax.run(scenes=[scene], geometry_only=False, outputfile=fn, gpu=None)
