from pathlib import Path

import gprMax
import numpy as np

# File path for output
fn = Path(__file__)
parts = fn.parts

# Discretisation
dl = 0.001

# Domain
x = 0.201
y = 0.276
z = 0.176

domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(iterations=2100)
tssf = gprMax.TimeStepStabilityFactor(f=0.99)

waveform = gprMax.Waveform(wave_type='gaussiandotnorm', amp=1, freq=9.42e9, id='mypulse')
hertzian_dipole = gprMax.HertzianDipole(polarisation='z',
                                        p1=(0.088, 0.088, 0.089),
                                        waveform_id='mypulse')
rx = gprMax.Rx(p1=(0.113, 0.189, 0.088))

plate = gprMax.Plate(p1=(0.088, 0.088, 0.088), 
                     p2=(0.113, 0.188, 0.088), material_id='pec')

gv1 = gprMax.GeometryView(
    p1=(0, 0, 0),
    p2=(x, y, z),
    dl=(dl, dl, dl),
    filename=Path(*parts[:-1], f'{parts[-1]}_n'),
    output_type='n',
)
gv2 = gprMax.GeometryView(
    p1=(0, 0, 0),
    p2=(x, y, z),
    dl=(dl, dl, dl),
    filename=Path(*parts[:-1], f'{parts[-1]}_f'),
    output_type='f',
)

pml = gprMax.PMLProps(formulation='HORIPML', thickness=10)

# Parameters from http://dx.doi.org/10.1109/TAP.2018.2823864
pml_cfs = gprMax.PMLCFS(alphascalingprofile='constant', 
                        alphascalingdirection='forward', 
                        alphamin=0.05, alphamax=0.05,
                        kappascalingprofile='quartic', 
                        kappascalingdirection='forward', 
                        kappamin=1, kappamax=8, 
                        sigmascalingprofile='quartic', 
                        sigmascalingdirection='forward', 
                        sigmamin=0, 
                        sigmamax=1.1 * ((4 + 1) / (150 * np.pi * dl)))

scene = gprMax.Scene()
title = gprMax.Title(name=fn.with_suffix('').name + '_ref')
scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)
scene.add(tssf)
scene.add(waveform)
scene.add(hertzian_dipole)
scene.add(rx)
# scene.add(gv1)
# scene.add(gv2)

scene.add(pml)
scene.add(pml_cfs)

# Run model
gprMax.run(scenes=[scene], geometry_only=False, outputfile=fn)
