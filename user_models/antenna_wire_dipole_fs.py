# import the gprMax module
import gprMax



# Create simulation objects. The equivalent input commands are given.

#title: Wire antenna - half-wavelength dipole in free-space
title = gprMax.Title(name="Wire antenna - half-wavelength dipole in free-space")
#domain: 0.050 0.050 0.200
domain = gprMax.Domain(p1=(0.050, 0.050, 0.200))
#dx_dy_dz: 0.001 0.001 0.001
dxdydz = gprMax.Discretisation(p1=(0.001, 0.001, 0.001))
#time_window: 60e-9
time_window = gprMax.TimeWindow(time=10e-9)
#waveform: gaussian 1 1e9 mypulse
waveform = gprMax.Waveform(wave_type='gaussian', amp=1, freq=1e9, id='mypulse')
#transmission_line: z 0.025 0.025 0.100 73 mypulse
transmission_line = gprMax.TransmissionLine(polarisation='z',
                                            p1=(0.025, 0.025, 0.100),
                                            resistance=73,
                                            waveform_id='mypulse')
## 150mm length
#edge: 0.025 0.025 0.025 0.025 0.025 0.175 pec
e1 = gprMax.Edge(p1=(0.025, 0.025, 0.025),
                 p2=(0.025, 0.025, 0.175),
                 material_id='pec')

## 1mm gap at centre of dipole
#edge: 0.025 0.025 0.100 0.025 0.025 0.101 free_space
e2 = gprMax.Edge(p1=(0.025, 0.025, 0.100),
                 p2=(0.025, 0.025, 0.100),
                 material_id='free_space')

#geometry_view: 0.020 0.020 0.020 0.030 0.030 0.180 0.001 0.001 0.001 antenna_wire_dipole_fs f
gv = gprMax.GeometryView(p1=(0.020, 0.020, 0.020),
                         p2=(0.030, 0.030, 0.180),
                         dl=(0.001, 0.001, 0.001),
                         filename='antenna_wire_dipole_fs',
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
gprMax.run(scenes=[scene], n=1, outputfile='mysimulation')
