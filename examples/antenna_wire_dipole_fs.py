"""An half-wavelength wire dipole antenna in freespace

This example model demonstrates how to a the transmission line source, which
allows s-parameters and input impedances to be calculated after the simulation
has run. 

Both API and hash commands are given next to each other as a comparison of the
two different methods that can be used to build a model.
"""

from pathlib import Path

import gprMax

fn = Path(__file__)

title = gprMax.Title(name=fn.with_suffix('').name)
#title: antenna_wire_dipole_fs

domain = gprMax.Domain(p1=(0.050, 0.050, 0.200))
#domain: 0.050 0.050 0.200

dxdydz = gprMax.Discretisation(p1=(0.001, 0.001, 0.001))
#dx_dy_dz: 0.001 0.001 0.001

time_window = gprMax.TimeWindow(time=10e-9)
#time_window: 10e-9

waveform = gprMax.Waveform(wave_type='gaussian', amp=1, freq=1e9, id='mypulse')
#waveform: gaussian 1 1e9 mypulse

transmission_line = gprMax.TransmissionLine(polarisation='z',
                                            p1=(0.025, 0.025, 0.100),
                                            resistance=73,
                                            waveform_id='mypulse')
#transmission_line: z 0.025 0.025 0.100 73 mypulse

## 150mm length wire
e1 = gprMax.Edge(p1=(0.025, 0.025, 0.025),
                 p2=(0.025, 0.025, 0.175),
                 material_id='pec')
#edge: 0.025 0.025 0.025 0.025 0.025 0.175 pec

## 1mm gap at centre of dipole
e2 = gprMax.Edge(p1=(0.025, 0.025, 0.100),
                 p2=(0.025, 0.025, 0.101),
                 material_id='free_space')
#edge: 0.025 0.025 0.100 0.025 0.025 0.101 free_space

gv = gprMax.GeometryView(p1=(0.020, 0.020, 0.020),
                         p2=(0.030, 0.030, 0.180),
                         dl=(0.001, 0.001, 0.001),
                         filename=fn.with_suffix('').name,
                         output_type='n')
#geometry_view: 0.020 0.020 0.020 0.030 0.030 0.180 0.001 0.001 0.001 antenna_wire_dipole_fs f

# Create a scene
scene = gprMax.Scene()

# Add the simulation objects to the scene
scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)
scene.add(waveform)
scene.add(transmission_line)
scene.add(e1)
scene.add(e2)
scene.add(gv)

# Run the simulation
gprMax.run(scenes=[scene], n=1, outputfile=fn)