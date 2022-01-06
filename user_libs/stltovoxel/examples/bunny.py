from pathlib import Path

import gprMax

fn = Path(__file__)

title = gprMax.Title(name=fn.with_suffix('').name)
domain = gprMax.Domain(p1=(0.148, 0.128, 0.148))
dl = 0.001
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=10e-9)

g_read = gprMax.GeometryObjectsRead(p1=(0.020, 0.020, 0.020), geofile='bunny.h5', matfile= 'materials.txt')

gv = gprMax.GeometryView(p1=(0, 0, 0),
                         p2=domain.props.p1,
                         dl=(dl, dl, dl),
                         filename=fn.with_suffix('').name,
                         output_type='n')
                         
scene = gprMax.Scene()
scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)
scene.add(g_read)
scene.add(gv)

gprMax.run(scenes=[scene], geometry_only=True, n=1, outputfile=fn)
