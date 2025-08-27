from pathlib import Path

import numpy as np

import gprMax
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

scene = gprMax.Scene()

title = gprMax.Title(name=fn.with_suffix("").name)
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
pml = gprMax.PMLThickness(thickness=14)
scene.add(title)
scene.add(dxdydz)
scene.add(pml)

timewindow = 4.5e-9  # For 0.3m max
radii = np.linspace(0.1, 0.3, 20)
theta = np.linspace(3, 357, 60)

fs = np.array([0.040, 0.040, 0.040])
domain_size = np.array([2 * fs[0] + 0.170, 2 * fs[1] + 2 * radii[-1], 2 * fs[2] + 2 * radii[-1]])
domain = gprMax.Domain(p1=(domain_size[0], domain_size[1], domain_size[2]))
time_window = gprMax.TimeWindow(time=timewindow)
scene.add(domain)
scene.add(time_window)

antennaposition = np.array([domain_size[0] / 2, fs[1] + radii[-1], fs[2] + radii[-1]])
gssi_objects = antenna_like_GSSI_1500(antennaposition[0], antennaposition[1], antennaposition[2])
for obj in gssi_objects:
    scene.add(obj)

## Can introduce soil model
# soil = gprMax.SoilPeplinski(sand_fraction=0.5, clay_fraction=0.5,
#                             bulk_density=2.0, sand_density=2.66,
#                             water_fraction_lower=0.001,
#                             water_fraction_upper=0.25,
#                             id='mySoil')
# scene.add(soil)
# fbox = gprMax.FractalBox(p1=(0, 0, 0), p2=(domain_size[0], domain_size[1], fs[2] + radii[-1]),
#                          frac_dim=1.5, weighting=[1, 1, 1], n_materials=50,
#                          mixing_model_id=soil.id, id='mySoilBox')
# scene.add(fbox)

mat = gprMax.Material(er=5, se=0, mr=1, sm=0, id="er5")
scene.add(mat)
box = gprMax.Box(p1=(0, 0, 0), p2=(domain_size[0], domain_size[1], fs[2] + radii[-1]), material_id="er5")
scene.add(box)

## Save the position of the antenna to file for use when processing results
np.savetxt(fn.with_suffix("").name + "_rxsorigin.txt", antennaposition, fmt="%f")

## Generate receiver points for pattern
for radius in range(len(radii)):
    ## E-plane circle (yz plane, x=0, phi=pi/2,3pi/2)
    x = radii[radius] * np.sin(theta * np.pi / 180) * np.cos(90 * np.pi / 180)
    y = radii[radius] * np.sin(theta * np.pi / 180) * np.sin(90 * np.pi / 180)
    z = radii[radius] * np.cos(theta * np.pi / 180)
    for rxpt in range(len(theta)):
        rx = gprMax.Rx(p1=(x[rxpt] + antennaposition[0], y[rxpt] + antennaposition[1], z[rxpt] + antennaposition[2]))
        scene.add(rx)

gv1 = gprMax.GeometryView(
    p1=(0, 0, 0),
    p2=(domain_size[0], domain_size[1], domain_size[2]),
    dl=(dl, dl, dl),
    filename="antenna_like_GSSI_1500_patterns",
    output_type="n",
)
scene.add(gv1)

gprMax.run(scenes=[scene], geometry_only=True, outputfile=fn, gpu=None)
