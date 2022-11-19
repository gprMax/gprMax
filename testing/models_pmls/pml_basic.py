from pathlib import Path

import gprMax

# File path for output
fn = Path(__file__)

# Discretisation
dl = 0.001

# Domain
x = 0.100
y = 0.100
z = 0.100

domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=3e-9)

waveform = gprMax.Waveform(wave_type='gaussian', amp=1, freq=1e9, id='mypulse')
hertzian_dipole = gprMax.HertzianDipole(polarisation='z',
                                        p1=(0.050, 0.050, 0.050),
                                        waveform_id='mypulse')
rx = gprMax.Rx(p1=(0.070, 0.070, 0.070))

# PML cases
thick = 10 # thickness
cases = {'off': {'x0': 0, 'y0': 0, 'z0': 0, 'xmax': 0, 'ymax': 0, 'zmax':0},
         'x0': {'x0': thick, 'y0': 0, 'z0': 0, 'xmax': 0, 'ymax': 0, 'zmax':0}, 
         'y0': {'x0': 0, 'y0': thick, 'z0': 0, 'xmax': 0, 'ymax': 0, 'zmax':0},
         'z0': {'x0': 0, 'y0': 0, 'z0': thick, 'xmax': 0, 'ymax': 0, 'zmax':0},
         'xmax': {'x0': 0, 'y0': 0, 'z0': 0, 'xmax': thick, 'ymax': 0, 'zmax':0},
         'ymax': {'x0': 0, 'y0': 0, 'z0': 0, 'xmax': 0, 'ymax': thick, 'zmax':0},
         'zmax': {'x0': 0, 'y0': 0, 'z0': 0, 'xmax': 0, 'ymax': 0, 'zmax': thick}}

# PML formulation
pml_type = gprMax.PMLProps(formulation='HORIPML')

## Built-in 1st order PML
pml_cfs = gprMax.PMLCFS(alphascalingprofile='constant', 
                         alphascalingdirection='forward', 
                         alphamin=0, alphamax=0,
                         kappascalingprofile='constant', 
                         kappascalingdirection='forward', 
                         kappamin=1, kappamax=1, 
                         sigmascalingprofile='quartic', 
                         sigmascalingdirection='forward', 
                         sigmamin=0, sigmamax=None)

## PMLs from http://dx.doi.org/10.1109/TAP.2011.2180344
## Standard PML
# pml_cfs = gprMax.PMLCFS(alphascalingprofile='constant', 
#                          alphascalingdirection='forward', 
#                          alphamin=0, alphamax=0,
#                          kappascalingprofile='quartic', 
#                          kappascalingdirection='forward', 
#                          kappamin=1, kappamax=11, 
#                          sigmascalingprofile='quartic', 
#                          sigmascalingdirection='forward', 
#                          sigmamin=0, sigmamax=7.427)

## CFS PML
# pml_cfs = gprMax.PMLCFS(alphascalingprofile='constant', 
#                          alphascalingdirection='forward', 
#                          alphamin=0.05, alphamax=0.05,
#                          kappascalingprofile='quartic', 
#                          kappascalingdirection='forward', 
#                          kappamin=1, kappamax=7, 
#                          sigmascalingprofile='quartic', 
#                          sigmascalingdirection='forward', 
#                          sigmamin=0, sigmamax=11.671)

## 2nd order RIPML
# pml_cfs1 = gprMax.PMLCFS(alphascalingprofile='constant', 
#                          alphascalingdirection='forward', 
#                          alphamin=0, alphamax=0,
#                          kappascalingprofile='constant', 
#                          kappascalingdirection='forward', 
#                          kappamin=1, kappamax=1, 
#                          sigmascalingprofile='sextic', 
#                          sigmascalingdirection='forward', 
#                          sigmamin=0, sigmamax=0.5836)
# pml_cfs2 = gprMax.PMLCFS(alphascalingprofile='constant', 
#                          alphascalingdirection='forward', 
#                          alphamin=0.05, alphamax=0.05,
#                          kappascalingprofile='cubic', 
#                          kappascalingdirection='forward', 
#                          kappamin=1, kappamax=8, 
#                          sigmascalingprofile='quadratic', 
#                          sigmascalingdirection='forward', 
#                          sigmamin=0, sigmamax=5.8357)

scenes = []
for k, v in cases.items():
    scene = gprMax.Scene()
    title = gprMax.Title(name=fn.with_suffix('').name + '_' + k)
    scene.add(title)
    scene.add(domain)
    scene.add(dxdydz)
    scene.add(time_window)
    scene.add(waveform)
    scene.add(hertzian_dipole)
    scene.add(rx)

    pml = gprMax.PMLProps(formulation='HORIPML', **v)
    scene.add(pml)
    scene.add(pml_cfs)

    scenes.append(scene)

# Run model
gprMax.run(scenes=scenes, n=len(cases), geometry_only=False, outputfile=fn)