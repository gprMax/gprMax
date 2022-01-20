from pathlib import Path

import gprMax
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.io import loadmat


# Title and file path for FDTD model output
modeltitle = 'bgr_6'
fn = Path(__file__)
fn = Path(fn.parent, modeltitle)

# Load B-scan data to be migrated
matfile = Path(str(Path(__file__).parent.resolve()), modeltitle + '.mat')
matcontents = loadmat(str(matfile))
data = matcontents['data']
data = np.transpose(data) # Transpose to rows: samples, cols: traces

# Specify trace interval, sampling interval, & create time vector for B-scan data
trac_int = 0.005 # metres
samp_int = 1.7578e-11 # seconds
maxtime = samp_int * data.shape[0]
time = np.linspace(0, maxtime, data.shape[0])

# Specify velocity/permittivity of B-scan data
v = 0.12e9

# Depth used for calculating FDTD z-dimension
depth = v * maxtime / 2

# FDTD discretisation, 2D domain dims, and time window
dl = 0.001 # metres
pml_cells = 10
extra_cells = 10 # Allow some extra cells after PML before placing sources
x_cells = data.shape[1] + 2 * pml_cells + 2 * extra_cells
x = x_cells * trac_int
y_cells = 1
y = y_cells * dl
z_cells = int(np.ceil(depth / dl) + 2 * pml_cells + 2 * extra_cells)
z = z_cells * dl

# Can build FDTD model from:
# 1. Matrix of velocity/permittivity, then write to file, and import
# 2. Directly using geometry primitives, i.e. boxes, etc...

# Option 1:
# Holds permittivity field to import into FDTD model
# er = np.ones((x_cells, y_cells, z_cells - (pml_cells + extra_cells)))
# er_value = np.around(4 * (c / v)**2, decimals=2) # 4xEr as velocity doubled
# er = er * er_value
# mat_ers = np.unique(er)

# Write materials text file
# with open(fn.with_suffix('.txt'), 'w') as fmaterials:
#     for i, mat_er in enumerate(mat_ers):
#         er[er==mat_er] = i
#         fmaterials.write(f'#material: {mat_er} 0 1 0 mat{i}\n')

# # Write permittivity HDF5 file
# with h5py.File(fn.with_suffix('.h5'), 'w') as fdata:
#     fdata.attrs['Title'] = modeltitle
#     fdata.attrs['dx_dy_dz'] = (dl, dl, dl)
#     fdata['/data'] = er.astype('int16')

# Build FDTD model
scene = gprMax.Scene()

title = gprMax.Title(name=modeltitle)
domain = gprMax.Domain(p1=(x, y, z))
dxdydz = gprMax.Discretisation(p1=(dl, dl, dl))
time_window = gprMax.TimeWindow(time=maxtime)

scene.add(title)
scene.add(domain)
scene.add(dxdydz)
scene.add(time_window)

# Option 1
# go = gprMax.GeometryObjectsRead(p1=(0, 0, 0), geofile=fn.with_suffix('.h5'), 
#                                 matfile=fn.with_suffix('.txt'))
# scene.add(go)

# Option 2
mat = gprMax.Material(er=np.around(4 * (c / v)**2, decimals=2), se=0, mr=1,
                      sm=0, id='mat1')
scene.add(mat)
b1 = gprMax.Box(p1=(0, 0, 0), p2=(domain.props.p1[0], dl, 
                domain.props.p1[2] - (pml_cells + extra_cells) * dl), 
                material_id='mat1')
scene.add(b1)

# Specify waveforms and sources from reversed B-scan data
for i in range(data.shape[1]):
    wv = gprMax.Waveform(wave_type='user', 
                         user_values=np.flipud(data[:,i]), user_time=time,
                         kind='linear', fill_value='extrapolate',
                         id='mypulse' + str(i + 1))
    scene.add(wv)
    src = gprMax.HertzianDipole(polarisation='y',
                                p1=((pml_cells + extra_cells) * dl + i * trac_int,
                                    0, domain.props.p1[2] - (pml_cells + extra_cells) * dl),
                                waveform_id='mypulse' + str(i + 1))
    scene.add(src)

gv = gprMax.GeometryView(p1=(0, 0, 0), p2=domain.props.p1, dl=(dl, dl, dl),
                        filename=fn.with_suffix('').parts[-1],
                        output_type='n')

# Snapshot at end of time window is RTM result
fileext = '.h5' # Can also be '.vti' for a VTK format
snap = gprMax.Snapshot(p1=((pml_cells + extra_cells) * dl, 
                           0, 
                           (pml_cells + extra_cells) * dl), 
                       p2=(domain.props.p1[0] - (pml_cells + extra_cells) * dl,
                           dl, 
                           domain.props.p1[2] - (pml_cells + extra_cells) * dl),
                       dl=(dl, dl, dl),
                       filename=fn.with_suffix('').parts[-1] + '_rtm_result',
                       fileext=fileext, time=maxtime)

scene.add(gv)
scene.add(snap)

# Run FDTD model
gprMax.run(scenes=[scene], n=1, geometry_only=False, outputfile=fn)

# Open RTM results file
filename = Path(str(fn) + '_snaps', fn.with_suffix('').parts[-1] + '_rtm_result' + fileext)
fieldcomponent = 'Ey'
with h5py.File(filename, 'r') as f:
    outputdata = f[fieldcomponent]
    outputdata = np.array(outputdata)
    time = f.attrs['time']

# Manipulation/processing of outputdata
outputdata = outputdata.squeeze()
outputdata = outputdata.transpose()

# Plot RTM result
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, num=str(filename), 
                               figsize=(15, 10), facecolor='w', edgecolor='w')
orig_plt = ax1.imshow(data, extent=[0, data.shape[1] * trac_int, 
                      (data.shape[0] * samp_int) / 2 * v, 0], interpolation='nearest', 
                      aspect='auto', cmap='viridis', vmin=-np.amax(np.abs(data)),
                      vmax=np.amax(np.abs(data)))
ax1.set_xlabel('Distance [m]')
ax1.set_ylabel('Depth [m]')
ax1.title.set_text('Original')
ax1.grid(which='both', axis='both', linestyle='-.')
cb1 = plt.colorbar(orig_plt, ax=ax1)
cb1.set_label(fieldcomponent + ' [V/m]')

rtm_plt = ax2.imshow(np.flipud(outputdata), 
                     extent=[0, outputdata.shape[1] * dl, depth, 0], 
                     interpolation='nearest', aspect='auto', cmap='viridis',
                     vmin=-np.amax(np.abs(outputdata)), 
                     vmax=np.amax(np.abs(outputdata)))
ax2.set_xlabel('Distance [m]')
ax2.set_ylabel('Depth [m]')
ax2.title.set_text('RTM')
ax2.grid(which='both', axis='both', linestyle='-.')
cb2 = plt.colorbar(rtm_plt, ax=ax2)
cb2.set_label(fieldcomponent + ' [V/m]')

# Save a PDF/PNG of the figure
# fig.savefig(filename.with_suffix('.pdf'), dpi=None, format='pdf',
#             bbox_inches='tight', pad_inches=0.1)
# fig.savefig(filename.with_suffix('.png'), dpi=150, format='png',
#             bbox_inches='tight', pad_inches=0.1)

plt.show()