from pathlib import Path

import gprMax
from user_libs.GPRAntennaModels.GSSI import antenna_like_GSSI_400

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

# Domain
x = 0.380
y = 0.380
z = 0.360

scene = gprMax.Scene()

title = gprMax.Title(name=fn.with_suffix('').name)
domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=12e-9)

scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)

# Import antenna model and add to model
gssi_objects = antenna_like_GSSI_400(0.190, 0.190, 0.140, resolution=dl)
for obj in gssi_objects:
    scene.add(obj)

# Run model
gprMax.run(scenes=[scene], geometry_only=False, outputfile=fn)
