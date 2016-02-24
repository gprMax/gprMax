import os, platform, psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

moduledirectory = os.path.dirname(os.path.abspath(__file__))

# Machine identifier
platformID = platform.platform()
platformlongID = 'iMac (Retina 5K, 27-inch, Late 2014); 4GHz Intel Core i7; Mac OS X 10.11.3'

# Nmber of physical CPU cores on machine
phycores = psutil.cpu_count(logical=False)

# Number of threads (0 signifies serial compiled code)
threads = np.array([0, 1, 2, 4])

# 100 x 100 x 100 cell model execution times (seconds)
bench1 = np.array([40, 48, 37, 32])
bench1c = np.array([76, 77, 46, 32])

# 150 x 150 x 150 cell model execution times (seconds)
bench2 = np.array([108, 133, 93, 75])
bench2c = np.array([220, 220, 132, 94])

# Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colors = ['#5CB7C6', '#E60D30', '#A21797', '#A3B347']

fig, ax = plt.subplots(num=platformID, figsize=(20, 10), facecolor='w', edgecolor='w')
fig.suptitle(platformlongID)
gs = gridspec.GridSpec(1, 2, hspace=0.5)
ax = plt.subplot(gs[0, 0])
ax.plot(threads, bench1, color=colors[1], marker='.', ms=10, lw=2, label='1e6 cells (gprMax v3b21)')
ax.plot(threads, bench1c, color=colors[0], marker='.', ms=10, lw=2, label='1e6 cells (gprMax v2)')
ax.plot(threads, bench2, color=colors[1], marker='.', ms=10, lw=2, ls='--', label='3.375e6 cells (gprMax v3b21)')
ax.plot(threads, bench2c, color=colors[0], marker='.', ms=10, lw=2, ls='--', label='3.375e6 cells (gprMax v2)')

ax.set_xlabel('Number of threads')
ax.set_ylabel('Time [s]')
ax.grid()

legend = ax.legend(loc=1)
frame = legend.get_frame()
frame.set_edgecolor('white')

ax.set_xlim([0, phycores])
ax.set_xticks(threads)
ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

ax = plt.subplot(gs[0, 1])
ax.plot(threads, bench1[1] / bench1, color=colors[1], marker='.', ms=10, lw=2, label='1e6 cells (gprMax v3b21)')
ax.plot(threads, bench1c[1] / bench1c, color=colors[0], marker='.', ms=10, lw=2, label='1e6 cells (gprMax v2)')
ax.plot(threads, bench2[1] / bench2, color=colors[1], marker='.', ms=10, lw=2, ls='--', label='3.375e6 cells (gprMax v3b21)')
ax.plot(threads, bench2c[1] / bench2c, color=colors[0], marker='.', ms=10, lw=2, ls='--', label='3.375e6 cells (gprMax v2)')

ax.set_xlabel('Number of threads')
ax.set_ylabel('Speed-up factor')
ax.grid()

legend = ax.legend(loc=1)
frame = legend.get_frame()
frame.set_edgecolor('white')

ax.set_xlim([0, phycores])
ax.set_xticks(threads)
ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

# Save a pdf of the plot
fig.savefig(os.path.join(moduledirectory, platformID + '.png'), dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()


