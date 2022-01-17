from pathlib import Path

import gprMax
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.io import loadmat

# Load B-scan data to be migrated
matfile = Path(str(Path(__file__).parent.resolve()), 'bgr_6.mat')
matcontents = loadmat(str(matfile))
data = matcontents['data']

# Specify trace interval (metres), sampling interval (seconds),
#  and create time vector for B-scan data
trac_int = 0.005
samp_int = 1.7578e-11
time = np.linspace(0, samp_int * data.shape[1], data.shape[1])

# Specify velocity/permittivity of B-scan data
v = 0.12e9
er = (c / v)**2

# Reverse B-scan data to use as sources for RTM model
data = np.flipud(data)

# Title and file path for FDTD model output
modeltitle = 'rtm_model'
fn = Path(__file__)
fn = Path(fn.parent, modeltitle)

# FDTD discretisation
dl = trac_int

# FDTD domain - 2D
pml_cells = 10
extra_cells = 10 # Allow some extra cells after PML before placing sources
x = (data.shape[0] + 2 * pml_cells + 2 * extra_cells) * dl
y = (data.shape[1] + 2 * pml_cells + 2 * extra_cells) * dl
z = trac_int

# FDTD time window
timewindow = (data.shape[1] - 1) * samp_int

# Build FDTD model
scene = gprMax.Scene()

title = gprMax.Title(name=modeltitle)
domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=timewindow)

scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)

# Specify materials and geometry for FDTD
# N.B Permittivity should be 4 x permittivity from B-scan, i.e 2 x velocity
mat1 = gprMax.Material(er=4*er, se=0, mr=1, sm=0, id='mat1')
b1 = gprMax.Box(p1=(0, 0, 0), p2=(domain.props.p1[0], 
                                  domain.props.p1[1] - (pml_cells + extra_cells) * dl, 
                                  domain.props.p1[2]), material_id='mat1')

# Specify waveforms and sources from reversed B-scan data
data = np.transpose(data) # Transpose to match shape of time vector

for i in range(data.shape[1]):
    wv = gprMax.Waveform(wave_type='user', 
                         user_values=data[:,i], user_time=time,
                         id='mypulse' + str(i + 1))
    scene.add(wv)
    src = gprMax.HertzianDipole(polarisation='z',
                                p1=((pml_cells + extra_cells) * dl + i * trac_int,
                                    domain.props.p1[1] - (pml_cells + extra_cells) * dl, 0),
                                waveform_id='mypulse' + str(i + 1))
    scene.add(src)

gv = gprMax.GeometryView(p1=(0, 0, 0), p2=domain.props.p1, dl=(dl, dl, dl),
                        filename=fn.with_suffix('').parts[-1],
                        output_type='n')

# Snapshot of at end of time window will be RTM result
fileext = '.h5' # Can also be '.vti' for a VTK format
snap = gprMax.Snapshot(p1=(0, 0, 0), p2=domain.props.p1, dl=(dl, dl, dl),
                        filename=fn.with_suffix('').parts[-1] + '_rtm_result',
                        fileext=fileext, time=timewindow)

scene.add(mat1)
scene.add(b1)
scene.add(gv)
scene.add(snap)

# Run FDTD model
gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile=fn)

# Open RTM results file
filename = Path(str(fn) + '_snaps', fn.with_suffix('').parts[-1] + '_rtm_result' + fileext)
fieldcomponent = 'Ez'
f = h5py.File(filename, 'r')
outputdata = f[fieldcomponent]
outputdata = np.array(outputdata)
time = f.attrs['time']
f.close()

# Manipulation/processing of outputdata
outputdata = outputdata.squeeze()
outputdata = outputdata.transpose()

# Plot RTM result
fig = plt.figure(num=str(filename), figsize=(20, 10), facecolor='w', edgecolor='w')
plt.imshow(outputdata, extent=[0, outputdata.shape[1], time, 0],
            interpolation='nearest', aspect='auto', cmap='gray',
            vmin=-np.amax(np.abs(outputdata)), vmax=np.amax(np.abs(outputdata)))
plt.xlabel('Trace number')
plt.ylabel('Time [s]')

# Grid properties
ax = fig.gca()
ax.grid(which='both', axis='both', linestyle='-.')

cb = plt.colorbar()
cb.set_label(fieldcomponent + ' [V/m]')

# Save a PDF/PNG of the figure
# fig.savefig(filename.with_suffix('.pdf'), dpi=None, format='pdf',
#             bbox_inches='tight', pad_inches=0.1)
# fig.savefig(filename.with_suffix('.png'), dpi=150, format='png',
#             bbox_inches='tight', pad_inches=0.1)

plt.show()
