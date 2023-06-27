"""B-scan using a antenna model.

This example model demonstrates how to create a B-scan using an antenna model.

The key part of this example is the concept of moving the antenna model in steps
within the model domain to create multiple A-scans that build up the B-scan.
Each A-scan requires its own scene and a list of scenes it built up using a for
loop. A different scene is required for each A-scan because the model geometry is
changing, as the antenna geometry must be moved to a new position. This is in
contrast to simpler models that may use a Hertzian dipole source which has no
associated 'geometry' and can be moved within a model without having to change
the scene. When the all the scenes are created, the list of scenes is then passed
to gprMax to run, noting the number of times 'n' gprMax is run corresponds to the
number of scenes, i.e. A-scans.
"""

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

for i in range(1, 55):
    scene = gprMax.Scene()

    title = gprMax.Title(name=fn.with_suffix("").name)
    domain = gprMax.Domain(p1=(x, y, z))
    dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
    time_window = gprMax.TimeWindow(time=6e-9)

    scene.add(title)
    scene.add(domain)
    scene.add(dxdydz)
    scene.add(time_window)

    mat = gprMax.Material(er=6, se=0, mr=1, sm=0, id="half_space")
    c1 = gprMax.Cylinder(p1=(0.240, 0, 0.080), p2=(0.240, 0.148, 0.080), r=0.010, material_id="pec")
    scene.add(mat)
    scene.add(c1)

    # Import antenna model and add to model
    ant_pos = (0.125, 0.094, 0.100)
    gssi_objects = antenna_like_GSSI_1500(0.105 + i * 0.005, 0.074, 0.170, resolution=dl)
    for obj in gssi_objects:
        scene.add(obj)

    gv1 = gprMax.GeometryView(
        p1=(0, 0, 0), p2=(x, y, z), dl=(dl, dl, dl), filename=fn.with_suffix("").name, output_type="n"
    )
    # scene.add(gv1)

    scenes.append(scene)

# Run model
gprMax.run(scenes=scenes, n=len(scenes), geometry_only=False, outputfile=fn, gpu=None)
