from pathlib import Path

import gprMax
from user_libs.antennas.GSSI import antenna_like_GSSI_1500

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

# Domain
x = 0.250
y = 0.188
z = 0.220

scene = gprMax.Scene()

title = gprMax.Title(name=fn.with_suffix('').name)
domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=6e-9)

scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)

# Import antenna model and add to model
gssi_objects = antenna_like_GSSI_1500(0.125, 0.094, 0.100, resolution=dl)
for obj in gssi_objects:
    scene.add(obj)

# Run model
gprMax.run(scenes=[scene], geometry_only=False, outputfile=fn, gpu=[0])
