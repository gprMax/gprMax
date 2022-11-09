from pathlib import Path

import gprMax

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

# Domain
x = 0.148
y = 0.128
z = 0.148

scene = gprMax.Scene()

title = gprMax.Title(name=fn.with_suffix('').name)
domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=10e-9)

scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)

go = gprMax.GeometryObjectsRead(p1=(0.020, 0.020, 0.020), geofile='stl/Stanford_Bunny.h5', matfile= 'materials.txt')

gv = gprMax.GeometryView(p1=(0, 0, 0), p2=domain.props.p1,
                         dl=(dl, dl, dl), filename=fn.with_suffix('').name,
                         output_type='n')
                         
scene.add(go)
scene.add(gv)

gprMax.run(scenes=[scene], geometry_only=True, outputfile=fn)