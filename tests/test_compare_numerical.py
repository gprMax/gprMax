import sys, os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tools.plot_fields import plot_Ascan

"""Compare field outputs
    
Usage:
    cd gprMax
    python -m tests.test_compare_numerical path_to_new_file path_to_old_file
    
"""

newfile = sys.argv[1]
oldfile = sys.argv[2]
path = '/rxs/rx1/'
# Key refers to subplot location
fields = {0: 'Ex', 2: 'Ey', 4: 'Ez', 1: 'Hx', 3: 'Hy', 5: 'Hz'}
plotorder = list(fields.keys())

# New results
f = h5py.File(newfile, 'r')
floattype = f[path + 'Ex'].dtype
new = np.zeros((f.attrs['Iterations'], 6), dtype=floattype)
timenew = np.zeros((f.attrs['Iterations']), dtype=floattype)
timenew = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt']) / 1e-9
for ID, name in fields.items():
    new[:,ID] = f[path + str(name)][:]
f.close()

# Old results
f = h5py.File(oldfile, 'r')
old = np.zeros((f.attrs['Iterations'], 6), dtype=floattype)
timeold = np.zeros((f.attrs['Iterations']), dtype=floattype)
timeold = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt']) / 1e-9
for ID, name in fields.items():
    old[:,ID] = f[path + str(name)][:]
f.close()

# Differences
# In case there is any difference in the number of iterations, take the smaller
timesmallest = np.amin((timeold.shape, timenew.shape))
fieldssmallest = np.amin((old.shape[0], new.shape[0]))

threshold = 1e-4 # Threshold, below which ignore differences
diffs = np.zeros((fieldssmallest, 6), dtype=floattype)
for ID, name in fields.items():
    max = np.amax(np.abs(new[:fieldssmallest,ID]))
    if max < threshold:
        diffs[:,ID] = 0
        diffsum = 0
        print('Detected differences of less than {} when comparing {} field component, therefore set as zero.'.format(threshold, fields[ID]))
    else:
        diffs[:,ID] = (np.abs(new[:fieldssmallest,ID] - old[:fieldssmallest,ID]) / max) * 100
        diffsum = (np.sum(np.abs(new[:fieldssmallest,ID] - old[:fieldssmallest,ID])) / np.sum(np.abs(new[:fieldssmallest,ID]))) * 100
    print('Total differences in field component {}: {:.1f}%'.format(name, diffsum))

# Plot new
fig1, plt1 = plot_Ascan(newfile + ' versus ' + oldfile, timenew, new[:,0], new[:,2], new[:,4], new[:,1], new[:,3], new[:,5])

# Add old and set legend
for index, ax in enumerate(fig1.axes):
    if plotorder[index] in [0, 2, 4]:
        ax.plot(timeold, old[:,plotorder[index]], 'r', label='old', lw=2, ls='--')
    else:
        ax.plot(timeold, old[:,plotorder[index]], label='old', lw=2, ls='--')
    ax.set_xlim(0, timeold[-1])
    handles, existlabels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Model (new code)', 'Model (old C code)'])

# Plots of differences
fig2, plt2 = plot_Ascan('Deltas: ' + newfile + ' versus ' + oldfile, timenew[:timesmallest], diffs[:,0], diffs[:,2], diffs[:,4], diffs[:,1], diffs[:,3], diffs[:,5])
[ax.set_xlim(0, timenew[timesmallest - 1]) for ax in fig2.axes]
[ax.set_ylim(0, np.ceil(np.amax(np.abs(diffs)))) for ax in fig2.axes]
ylabels = ['$E_x$', '$H_x$', '$E_y$', '$H_y$', '$E_z$', '$H_z$']
ylabels = [ylabel + ', percentage difference [%]' for ylabel in ylabels]
[ax.set_ylabel(ylabels[index]) for index, ax in enumerate(fig2.axes)]

# Show/print plots
savename = os.path.abspath(os.path.dirname(newfile)) + os.sep + os.path.splitext(os.path.split(newfile)[1])[0] + '_vs_' + os.path.splitext(os.path.split(oldfile)[1])[0]
#fig1.savefig(savename + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
#fig2.savefig(savename + '_diffs.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt1.show()
plt2.show()


