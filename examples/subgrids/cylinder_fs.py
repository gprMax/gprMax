"""Cylinder in freespace

This example model demonstrates how to use subgrids at a basic level.

The geometry is 3D (required for any use of subgrids) and is of a water-filled
cylindrical object in freespace. The subgrid encloses the cylinderical object
using a fine spatial discretisation (1mm), and a courser spatial discretisation
(5mm) is used in the rest of the model (main grid). A simple Hertzian dipole
source is used with a waveform shaped as the first derivative of a gaussian.
"""

from pathlib import Path
import gprMax
from gprMax.materials import calculate_water_properties

# File path - used later to specify name of output files
fn = Path(__file__)
parts = fn.parts

# Subgrid spatial discretisation in x, y, z directions
dl_sg = 1e-3

# Subgrid ratio - must always be an odd integer multiple
ratio = 5
dl = dl_sg * ratio

# Domain extent
x = 0.500
y = 0.500
z = 0.500

# Time window
tw = 6e-9

scene = gprMax.Scene()

title = gprMax.Title(name=fn.name)
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
domain = gprMax.Domain(p1=(x, y, z))
time_window = gprMax.TimeWindow(time=tw)

wf = gprMax.Waveform(wave_type="gaussiandot", amp=1, freq=1.5e9, id="mypulse")
hd = gprMax.HertzianDipole(polarisation="z", p1=(0.205, 0.400, 0.250), waveform_id="mypulse")
rx = gprMax.Rx(p1=(0.245, 0.400, 0.250))

scene.add(title)
scene.add(dxdydz)
scene.add(domain)
scene.add(time_window)
scene.add(wf)
scene.add(hd)
scene.add(rx)

# Cylinder parameters
c1 = (0.225, 0.250, 0.100)
c2 = (0.225, 0.250, 0.400)
r = 0.010
sg1 = (c1[0] - r, c1[1] - r, c1[2])
sg2 = (c2[0] + r, c2[1] + r, c2[2])

# Create subgrid
subgrid = gprMax.SubGridHSG(p1=sg1, p2=sg2, ratio=ratio, id="sg")
scene.add(subgrid)

# Create water material
eri, er, tau, sig = calculate_water_properties()
water = gprMax.Material(er=eri, se=sig, mr=1, sm=0, id="water")
subgrid.add(water)
water = gprMax.AddDebyeDispersion(poles=1, er_delta=[er - eri], tau=[tau], material_ids=["water"])
subgrid.add(water)

# Add cylinder to subgrid
cylinder = gprMax.Cylinder(p1=c1, p2=c2, r=r, material_id="water")
subgrid.add(cylinder)

# Create some geometry views for both subgrid and main grid
gvsg = gprMax.GeometryView(
    p1=sg1, p2=sg2, dl=(dl_sg, dl_sg, dl_sg), filename=fn.with_suffix("").parts[-1] + "_sg", output_type="n"
)
subgrid.add(gvsg)

gv1 = gprMax.GeometryView(
    p1=(0, 0, 0), p2=(x, y, z), dl=(dl, dl, dl), filename=fn.with_suffix("").parts[-1], output_type="n"
)
scene.add(gv1)

# Create some snapshots of entire domain
for i in range(5):
    s = gprMax.Snapshot(
        p1=(0, 0, 0),
        p2=(x, y, z),
        dl=(dl, dl, dl),
        time=(i + 0.5) * 1e-9,
        filename=fn.with_suffix("").parts[-1] + "_" + str(i + 1),
    )
    scene.add(s)

gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile=fn, subgrid=True, autotranslate=True)
