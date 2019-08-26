import gprMax

# single objects
dxdydz = gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3))
messages = gprMax.Messages(yn='y')
tw = gprMax.TimeWindow(time=6e-9)
domain = gprMax.Domain(p1=(0.15, 0.15, 0.1))
title = gprMax.Title(name='Heterogeneous soil using a stochastic distribution of dielectric properties given by a mixing model from Peplinski')
waveform1 = gprMax.Waveform(wave_type='ricker', amp=1, freq=1.5e9, id='my_ricker')
dipole = gprMax.HertzianDipole(p1=(0.045, 0.075, 0.085), polarisation='y', waveform_id='my_ricker')


sp = gprMax.SoilPeplinski(sand_fraction=0.5,
                     clay_fraction=0.5,
                     bulk_density=2.0,
                     sand_density=2.66,
                     water_fraction_lower=0.001,
                     water_fraction_upper=0.25,
                     id='my_soil')

fb = gprMax.FractalBox(p1=(0, 0, 0), p2=(0.15, 0.15, 0.070), frac_dim=1.5, weighting=[1, 1, 1], n_materials=50, mixing_model_id='my_soil', id='my_soil_box')
asf = gprMax.AddSurfaceRoughness(p1=(0, 0, 0.070), p2=(0.15, 0.15, 0.070), frac_dim=1.5, weighting=[1, 1], limits=[0.065, 0.080], fractal_box_id='my_soil_box')
gv = gprMax.GeometryView(p1=(0, 0, 0), p2=(0.15, 0.15, 0.1), dl=(0.001, 0.001, 0.001), filename='heterogeneous_soil', output_type='n')


rx = gprMax.Rx(p1=(0.045, 0.075 + 10e-3, 0.085))

scene = gprMax.Scene()
scene.add(dxdydz)
scene.add(messages)
scene.add(tw)
scene.add(domain)
scene.add(title)
scene.add(waveform1)
scene.add(dipole)
scene.add(sp)
scene.add(fb)
scene.add(asf)
scene.add(gv)
scene.add(rx)

gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile='mysimulation')
