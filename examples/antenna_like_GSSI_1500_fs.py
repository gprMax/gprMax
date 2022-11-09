from pathlib import Path

import gprMax
from user_libs.GPRAntennaModels.GSSI import antenna_like_GSSI_1500

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

# Domain
x = 0.250
y = 0.250
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
ant_pos = (0.125, 0.094, 0.100)
gssi_objects = antenna_like_GSSI_1500(ant_pos[0], ant_pos[1], ant_pos[2], 
                                      resolution=dl)
for obj in gssi_objects:
    obj.rotate('z', 90, origin=(ant_pos[0], ant_pos[1], ant_pos[2]))
    scene.add(obj)

gv1 = gprMax.GeometryView(p1=(0, 0, 0), p2=(x, y, z),
                          dl=(dl, dl, dl), filename='antenna_like_GSSI_1500',
                          output_type='n')
gv2 = gprMax.GeometryView(p1=(ant_pos[0] - 0.170/2, ant_pos[1] - 0.108/2, ant_pos[2] - 0.050), 
                          p2=(ant_pos[0] + 0.170/2, ant_pos[1] + 0.108/2, ant_pos[2] + 0.010),
                          dl=(dl, dl, dl), filename='antenna_like_GSSI_1500_pcb',
                          output_type='f')
scene.add(gv1)
scene.add(gv2)

# Run model
gprMax.run(scenes=[scene], geometry_only=True, outputfile=fn, gpu=None)
