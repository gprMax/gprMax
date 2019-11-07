from pathlib import Path

import gprMax

fn = Path(__file__)

title = gprMax.Title(name=fn.with_suffix('').name)
domain = gprMax.Domain(p1=(0.050, 0.050, 0.200))
dxdydz = gprMax.Discretisation(p1=(0.001, 0.001, 0.001))
time_window = gprMax.TimeWindow(time=10e-9)

waveform = gprMax.Waveform(wave_type='gaussian', amp=1, freq=1e9, id='mypulse')
transmission_line = gprMax.TransmissionLine(polarisation='z',
                                            p1=(0.025, 0.025, 0.100),
                                            resistance=73,
                                            waveform_id='mypulse')

## 150mm length
e1 = gprMax.Edge(p1=(0.025, 0.025, 0.025),
                 p2=(0.025, 0.025, 0.175),
                 material_id='pec')

## 1mm gap at centre of dipole
e2 = gprMax.Edge(p1=(0.025, 0.025, 0.100),
                 p2=(0.025, 0.025, 0.100),
                 material_id='free_space')

gv = gprMax.GeometryView(p1=(0.020, 0.020, 0.020),
                         p2=(0.030, 0.030, 0.180),
                         dl=(0.001, 0.001, 0.001),
                         filename=fn.with_suffix('').name,
                         output_type='n')

# create a scene
scene = gprMax.Scene()
# add the simulation objects to the scene
scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)
scene.add(waveform)
scene.add(transmission_line)
scene.add(e1)
scene.add(e2)
scene.add(gv)

# run the simulation
gprMax.run(scenes=[scene], n=1, outputfile=fn)
