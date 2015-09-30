import sys, os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tools.plot_fields import plot_Ascan
from tests.analytical_solutions import hertzian_dipole_fs

"""Compare field outputs
    
Usage:
    cd gprMax
    python -m tests.test_compare_analytical path_to_model_output
    
"""

modelfile = sys.argv[1]
path = '/rxs/rx1/'
# Key refers to subplot location
fields = {0: 'Ex', 1: 'Ey', 2: 'Ez', 3: 'Hx', 4: 'Hy', 5: 'Hz'}
plotorder = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2, 5: 5}

# Model results
f = h5py.File(modelfile, 'r')
# Get model/file attributes
floattype = f[path + 'Ex'].dtype
iterations = f.attrs['Iterations']
dt = f.attrs['dt']
dxdydz = f.attrs['dx, dy, dz']
model = np.zeros((iterations, 6), dtype=floattype)
time = np.arange(0, dt * iterations, dt) / 1e-9
rxpos = f[path + 'Position']
txpos = f['/txs/tx1/Position']
rxposrelative = ((rxpos[0] - txpos[0]), (rxpos[1] - txpos[1]), (rxpos[2] - txpos[2]))
# Read fields
for ID, name in fields.items():
    model[:,ID] = f[path + str(name)][:]
f.close()

# Analytical solution of a dipole in free space
analytical = hertzian_dipole_fs(iterations * dt, dt, dxdydz, rxposrelative)

# Differences
threshold = 1e-4 # Threshold, below which ignore differences
diffs = np.zeros((iterations, 6), dtype=floattype)
for ID, name in fields.items():
    max = np.amax(np.abs(analytical[:,ID]))
    if max < threshold:
        diffs[:,ID] = 0
        diffsum = 0
        print('Detected differences of less than {} when comparing {} field component, therefore set as zero.'.format(threshold, fields[ID]))
    else:
        diffs[:,ID] = (np.abs(analytical[:,ID] - model[:,ID]) / max) * 100
        diffsum = (np.sum(np.abs(analytical[:,ID] - model[:,ID])) / np.sum(np.abs(analytical[:,ID]))) * 100
    print('Total differences in field component {}: {:.1f}%'.format(name, diffsum))

# Plot model
fig1, plt1 = plot_Ascan(modelfile + ' versus analytical solution', time, model[:,0], model[:,1], model[:,2], model[:,3], model[:,4], model[:,5])

# Add analytical solution and set legend
for index, ax in enumerate(fig1.axes):
    if index in [0, 2, 4]:
        ax.plot(time, analytical[:,plotorder[index]], 'r', label='analytical', lw=2, ls='--')
    else:
        ax.plot(time, analytical[:,plotorder[index]], label='analytical', lw=2, ls='--')
    ax.set_xlim(0, time[-1])
    handles, existlabels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Model', 'Analytical'])

# Plots of differences
fig2, plt2 = plot_Ascan('Deltas: ' + modelfile + ' versus analytical solution', time, diffs[:,0], diffs[:,1], diffs[:,2], diffs[:,3], diffs[:,4], diffs[:,5])
[ax.set_xlim(0, time[-1]) for ax in fig2.axes]
[ax.set_ylim(0, np.ceil(np.amax(np.abs(diffs)))) for ax in fig2.axes]
ylabels = ['$E_x$', '$H_x$', '$E_y$', '$H_y$', '$E_z$', '$H_z$']
ylabels = [ylabel + ', percentage difference [%]' for ylabel in ylabels]
[ax.set_ylabel(ylabels[index]) for index, ax in enumerate(fig2.axes)]

# Show/print plots
savename = os.path.abspath(os.path.dirname(modelfile)) + os.sep + os.path.splitext(os.path.split(modelfile)[1])[0] + '_vs_analytical'
#fig1.savefig(savename + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
#fig2.savefig(savename + '_diffs.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt1.show()
plt2.show()