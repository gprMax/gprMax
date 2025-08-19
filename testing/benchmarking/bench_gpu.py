import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height != 0:
            ax.text(rect.get_x() + rect.get_width()/2, height,
                    '%d' % int(height),
                    ha='center', va='bottom', fontsize=10, rotation=90)

# List of results to plot
cells = dict()
perf = dict()

# Cells multiplied by iterations divided by 1e6
cells_default = np.array([100**3, 150**3, 200**3, 300**3, 400**3, 450**3]) * 1559

# Load data - CPU
# basepath = '/Users/cwarren/Dropbox (Northumbria University)/Research/Journal-papers/Published/2018-CPC-gprMax_GPU/benchmarking-results/'
# CPU_GPU_f32 = dict(np.load(basepath + 'cpu_gpu_perf_f32.npz'))
# CPU_f32 = np.array(CPU_GPU_f32['cpucellspersec'][:-1] / 1e6)
# perf['2 x Intel(R) Xeon(R) E5-2640 v4 (2.4 GHz, 20 cores)'] = CPU_f32
# cells['2 x Intel(R) Xeon(R) E5-2640 v4 (2.4 GHz, 20 cores)'] = np.arange(1, len(cells_default) + 1)
#CPU_f64 = dict(np.load('basepath + 'cpu_gpu_perf_f64.npz'))
#CPU_iMac = dict(np.load('basepath + 'cpu_perf_iMac15,1.npz'))

# Load data - GPUs
# Tesla K40c
#K40_f32 = np.array(CPU_GPU_f32['gpucellspersec'][1,:-1] / 1e6)

# Tesla K80
# K80_f32 = dict(np.load(basepath + 'gpu_TeslaK80_perf_f32.npz'))
# K80_f32 = np.array(K80_f32['gpucellspersec'][0,:] / 1e6)
#K80_f64 = dict(np.load(basepath + 'gpu_TeslaK80_perf_f64.npz'))
# perf['Tesla K80'] = K80_f32
# cells['Tesla K80'] = np.arange(1, len(cells_default) + 1)

# GeForce GTX 1080 Ti
# GTX1080_f32 = dict(np.load(basepath + 'cpu_gpu_GTX1080_perf_f32.npz'))
# GTX1080_f32 = np.array(GTX1080_f32['gpucellspersec'][0,:-1] / 1e6)
#GTX1080_f64  = dict(np.load(basepath + 'cpu_gpu_GTX1080_perf_f64.npz'))
# perf['GeForce GTX 1080 Ti'] = GTX1080_f32
# cells['GeForce GTX 1080 Ti'] = np.arange(1, len(cells_default) + 1)

# TITAN X
# TITAN_X_f32 = np.array(CPU_GPU_f32['gpucellspersec'][0,:][:-1] / 1e6)
# perf['TITAN X'] = TITAN_X_f32
# cells['TITAN X'] = np.arange(1, len(cells_default) + 1)

# GeForce RTX 2080 Ti
with np.errstate(divide='ignore'):
    RTX2080_f32 = np.true_divide(cells_default, np.array([0.660652, 1.609686, 3.352049, 10.190110, 24.348148, 34.579480]) * 1e6)
RTX2080_f32[RTX2080_f32 == np.inf] = 0
perf['GeForce RTX 2080 Ti'] = RTX2080_f32
cells['GeForce RTX 2080 Ti'] = np.arange(1, len(cells_default) + 1)

# Quadro RTX 8000
RTX8000 = np.array([2592, 3409, 3875, 4284, 4267, 4225, 4262, 4376, 4461, 4478])
perf['Quadro RTX 8000'] = RTX8000
cells['Quadro RTX 8000'] = np.arange(1, 11)

# TITAN RTX
TITAN_RTX_f32 = np.array([2732, 3997, 4391, 4369, 4370, 4469, 4561])
perf['TITAN RTX'] = TITAN_RTX_f32
cells['TITAN RTX'] = np.array([1, 3, 4, 5, 7, 8, 9])

# TESLA P100
# P100_f32 = dict(np.load(basepath + 'gpu_TeslaP100_perf_f32.npz'))
# P100_f32 = np.array(P100_f32['gpucellspersec'] / 1e6)
#P100_f64 = dict(np.load(basepath + 'gpu_TeslaP100_perf_f64.npz'))
# perf['Tesla P100'] = P100_f32
# cells['Tesla P100'] = np.arange(1, len(cells_default) + 1)

# TESLA V100
#V100_f32 = np.array([1.717608475334649, 4.367920599789287, 5.097342297141094, 5.769181843483034, 5.988364069530498, 5.974646317241049]) * 1e9
with np.errstate(divide='ignore'):
    V100_oswald_f32 = np.true_divide(cells_default, np.array([0.700, 1.143, 2.387, 7.286, 16.721, 23.870]) * 1e6)
V100_oswald_f32[V100_oswald_f32 == np.inf] = 0
perf['Tesla V100'] = V100_oswald_f32
cells['Tesla V100'] = np.arange(1, len(cells_default) + 1)

# A100
A100_f32 = np.divide(np.array([100**3, 150**3, 200**3, 300**3, 400**3, 450**3, 500**3, 600**3, 700**3]) * 1559, 
                     np.array([0.638363, 0.719807, 1.416151, 4.007607, 8.803767, 12.211772, 16.443219, 27.781816, 44.187797]) * 1e6)
perf['A100'] = A100_f32
cells['A100'] = np.arange(1, 10)

# AMD Radeon7900XTX
with np.errstate(divide='ignore'):
    cells_amd = np.array([100**3, 150**3, 200**3, 300**3, 400**3, 450**3, 500**3, 600**3, 700**3])* 1559
    Radeon7900XTX_f32 = np.true_divide(cells_amd, np.array([4.2170, 4.1862, 4.1710, 7.9477, 17.1864, 23.8735, 32.1639, 53.4792, 83.2350]) * 1e6)
Radeon7900XTX_f32[Radeon7900XTX_f32 == np.inf] = 0
perf['AMD Radeon7900XTX'] = Radeon7900XTX_f32
cells['AMD Radeon7900XTX'] = np.arange(1, len(cells_amd) + 1)

# Create/setup plot figure
# Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colorIDs = ["#004996", "#5d6200", "#9600af", "#e09c60", "#4c78ff", "#602e0e", "#e685fb", "#a60042", "#835e8b"]
colors = itertools.cycle(colorIDs)
fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='w')
ax.set_axisbelow(True)
ax.grid(color=(0.75,0.75,0.75), linestyle='dashed')

width = 0.08 # the width of the bars
handles = []
# offset = - (len(perf) - 1) / 2 # offset used to position centre of each bar

# Calculate offsets (number of bars at each cell location)
offset_base = np.array([9, 8, 9, 9, 9, 8, 2, 2, 2, 1])
offsets = - ((offset_base - 1) / 2) * width

# Create bar plot
cells_max = 0
for (k1,v1), (k2,v2) in zip(cells.items(), perf.items()):
    if len(v1) > cells_max:
        cells_max =len(v1)
    x = v1 + offsets[v1 - 1]
    bar = ax.bar(x, v2, width, color=next(colors), edgecolor='none', label=k2)
    handles.append(bar)
    offsets[v1 - 1] += width

# Set plot options
ax.set_xlabel('Side length of cubic domain [cells]', fontsize=16)
ax.set_ylabel('Performance [Mcells/s]', fontsize=16)

legend = ax.legend(handles=handles, loc=2, fontsize=14)
frame = legend.get_frame()
frame.set_edgecolor('white')
frame.set_alpha(0)

ax.set_yticks(np.arange(0, 13500, 500))
ax.set_xticks(range(1, cells_max + 1))
cells_labels = ['100', '150', '200', '300', '400', '450', '500', '600', '700', '750']
ax.set_xticklabels(cells_labels[:cells_max])
for handle in handles:
    autolabel(handle)

##########################
# Save a png of the plot #
##########################
#fig.savefig('/Users/cwarren/Desktop/cpu_gpu_perf.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
#fig.savefig('/Users/cwarren/Desktop/cpu_gpu_perf_web.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.savefig('cpu_gpu_perf.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)