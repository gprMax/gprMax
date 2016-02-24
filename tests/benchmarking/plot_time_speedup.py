import os, platform, psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax._version import __version__

moduledirectory = os.path.dirname(os.path.abspath(__file__))

# Machine identifier
platformID = platform.platform()
#machineID = 'MacPro1,1'
#machineIDlong = machineID + ' (2006); 2 x 2.66 GHz Quad-Core Intel Xeon; Mac OS X 10.11.3'
machineID = 'iMac15,1'
machineIDlong = machineID + ' (Retina 5K, 27-inch, Late 2014); 4GHz Intel Core i7; Mac OS X 10.11.3'

# Load results
results = np.load(os.path.join(moduledirectory, machineID + '.npz'))

# Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colors = ['#5CB7C6', '#E60D30', '#A21797', '#A3B347']

fig, ax = plt.subplots(num=machineIDlong, figsize=(20, 10), facecolor='w', edgecolor='w')
fig.suptitle(machineIDlong)
gs = gridspec.GridSpec(1, 2, hspace=0.5)
ax = plt.subplot(gs[0, 0])
ax.plot(results['threads'], results['bench1'], color=colors[1], marker='.', ms=10, lw=2, label='1e6 cells (v' + __version__ + ')')
ax.plot(results['threads'], results['bench1c'], color=colors[0], marker='.', ms=10, lw=2, label='1e6 cells (v2)')
ax.plot(results['threads'], results['bench2'], color=colors[1], marker='.', ms=10, lw=2, ls='--', label='3.375e6 cells (v' + __version__ + ')')
ax.plot(results['threads'], results['bench2c'], color=colors[0], marker='.', ms=10, lw=2, ls='--', label='3.375e6 cells (v2)')

ax.set_xlabel('Number of threads')
ax.set_ylabel('Time [s]')
ax.grid()

legend = ax.legend(loc=1)
frame = legend.get_frame()
frame.set_edgecolor('white')

ax.set_xlim([0, results['threads'][-1] * 1.1])
ax.set_xticks(results['threads'])
ax.set_ylim(0, top=ax.get_ylim()[1] * 1.1)

ax = plt.subplot(gs[0, 1])
ax.plot(results['threads'], results['bench1'][1] / results['bench1'], color=colors[1], marker='.', ms=10, lw=2, label='1e6 cells (v' + __version__ + ')')
ax.plot(results['threads'], results['bench1c'][1] / results['bench1c'], color=colors[0], marker='.', ms=10, lw=2, label='1e6 cells (v2)')
ax.plot(results['threads'], results['bench2'][1] / results['bench2'], color=colors[1], marker='.', ms=10, lw=2, ls='--', label='3.375e6 cells (v' + __version__ + ')')
ax.plot(results['threads'], results['bench2c'][1] / results['bench2c'], color=colors[0], marker='.', ms=10, lw=2, ls='--', label='3.375e6 cells (v2)')

ax.set_xlabel('Number of threads')
ax.set_ylabel('Speed-up factor')
ax.grid()

legend = ax.legend(loc=1)
frame = legend.get_frame()
frame.set_edgecolor('white')

ax.set_xlim([0, results['threads'][-1] * 1.1])
ax.set_xticks(results['threads'])
ax.set_ylim(bottom=1, top=ax.get_ylim()[1] * 1.1)

# Save a pdf of the plot
fig.savefig(os.path.join(moduledirectory, machineID + '.png'), dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()


