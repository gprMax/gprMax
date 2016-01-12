# Copyright (C) 2015-2016: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import sys, os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""Compare field outputs
    
Usage:
    cd gprMax
    python -m tests.test_compare_experimental path_to_model_output path_to_real_output
    
"""

modelfile = sys.argv[1]
realfile = sys.argv[2]
path = '/rxs/rx1/'
# Key refers to subplot location
fields = {0: 'Ex', 1: 'Ey', 2: 'Ez', 3: 'Hx', 4: 'Hy', 5: 'Hz'}
plotorder = list(fields.keys())

# Model results
f = h5py.File(modelfile, 'r')
floattype = f[path + 'Ex'].dtype
model = np.zeros((f.attrs['Iterations'], 6), dtype=floattype)
timemodel = np.zeros((f.attrs['Iterations']), dtype=floattype)
timemodel = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt']) / 1e-9
for ID, name in fields.items():
    model[:,ID] = f[path + str(name)][:] * -1
    model[:,ID] = model[:,ID] / np.amax(np.abs(model[:,ID]))
f.close()

# Select model field of interest and find max
modelmax = np.where(np.abs(model[:,1]) == 1)[0][0]

# Real results
with open(realfile, 'r') as f:
    real = np.loadtxt(f)
real[:,1] = real[:,1] / np.amax(np.abs(real[:,1]))
realmax = np.where(np.abs(real[:,1]) == 1)[0][0]

difftime = - (timemodel[modelmax] - real[realmax,0])

# Plot modelled and real data
fig, ax = plt.subplots(num=modelfile + ' versus ' + realfile, figsize=(20, 10), facecolor='w', edgecolor='w')
ax.plot(timemodel + difftime, model[:,1], 'r', lw=2, label='Model')
ax.plot(real[:,0], real[:,1], 'r', ls='--', lw=2, label='Experiment')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Amplitude')
ax.set_xlim([0, timemodel[-1]])
ax.set_ylim([-1, 1])
ax.legend()
[label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 )) for label in ax.get_xticklabels() + ax.get_yticklabels()]
ax.grid()

# Show/print plots
savename = os.path.abspath(os.path.dirname(modelfile)) + os.sep + os.path.splitext(os.path.split(modelfile)[1])[0] + '_vs_' + os.path.splitext(os.path.split(realfile)[1])[0]
#fig.savefig(savename + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()