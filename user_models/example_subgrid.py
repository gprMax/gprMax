import gprMax

scene = gprMax.Scene()

# global configuration
# objects within the inner surface can be positioned automatically
# in this configuration coordinates in the main grid Space
# are converted to subgrid local space automatically
# if objects or required in the region between the inner Surface
# and the outer surface autotranslate must be set to False
# and the local position must be set explicitly. Addionally,
# in this instance, objects which occur within the IS OS gap
# must be positioned in the subgrid and the main grid.
gprMax.config.general['autotranslate'] = True

# Points are specified as iterables
p1 = (1, 1, 1)

# single objects
dxdydz = gprMax.Discretisation(p1=(1e-2, 1e-2, 1e-2))
messages = gprMax.Messages(yn='y')
tw = gprMax.TimeWindow(iterations=250)
domain = gprMax.Domain(p1=p1)
title = gprMax.Title(name='mysimulation')
waveform = gprMax.Waveform(wave_type='gaussiandotnorm', amp=1, freq=400e6, id='mywaveform')

# position in subgrid automatically
dipole = gprMax.HertzianDipole(p1=(0.5, 0.5, 0.5), polarisation='y', waveform_id='mywaveform')

# equivalent position when autotranslate is false
#rx = gprMax.Rx(p1=(50 * 1e-2 / 5, 50 * 1e-2 / 5, 50 * 1e-2 / 5))

# rx in the subgrid
# subgrid data is stored in an additional output file called
# outputfile_<subgrid_id>.out
rx1 = gprMax.Rx(p1=(0.52, 0.5, 0.5))

# rx in the main grid
rx2 = gprMax.Rx(p1=(0.2, 0.5, 0.5))

# set the position of the Inner Surface
subgrid = gprMax.SubGridHSG(p1=(45e-2, 45e-2, 45e-2),
                            p2=(55e-2, 55e-2, 55e-2),
                            ratio=3,
                            id='subgrid1',
                            filter=False)
                            #pml_separation=6,
                            #subgrid_pml_thickness=10)

scene = gprMax.Scene()
scene.add(dxdydz)
scene.add(messages)
scene.add(tw)
scene.add(domain)
scene.add(title)
scene.add(subgrid)
scene.add(rx2)

subgrid.add(waveform)
subgrid.add(dipole)
subgrid.add(rx1)

#gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile='mysimulation', geometry_fixed=False, subgrid=True)
gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile='mysimulation', geometry_fixed=False, subgrid=True)
