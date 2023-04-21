from pathlib import Path

import gprMax
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

# Domain
x = 0.480
y = 0.148
z = 0.235

# Create list of scenes (A-scans) that comprise B-scan:
scenes = []

scans = 2

for i in range(1, scans):
    scene = gprMax.Scene()

    title = gprMax.Title(name=fn.with_suffix('').name)
    domain = gprMax.Domain(p1=(x, y, z))
    dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
    time_window = gprMax.TimeWindow(time=6e-9)

    scene.add(title)
    scene.add(domain)
    scene.add(dxdydz)
    scene.add(time_window)

    mat = gprMax.Material(er=6, se=0, mr=1, sm=0, id='half_space')
    c1 = gprMax.Cylinder(p1=(0.240, 0, 0.080), p2=(0.240, 0.148, 0.080), r=0.010, 
                        material_id='pec')

    
    scene.add(mat)
    scene.add(c1)

    # Import antenna model and add to model
    ant_pos = (0.125, 0.094, 0.100)
    gssi_objects = antenna_like_GSSI_1500(0.105 + i * 0.005, 0.074, 0.170, 
                                        resolution=dl)
    for obj in gssi_objects:
        scene.add(obj)

    gv1 = gprMax.GeometryView(p1=(0, 0, 0), p2=(x, y, z), dl=(dl, dl, dl), 
                              filename=fn.with_suffix('').name,
                              output_type='n')
    # scene.add(gv1)

    scenes.append(scene)

# Run model
gprMax.run(scenes=scenes, n=scans, geometry_only=False, outputfile=fn, gpu=None)
