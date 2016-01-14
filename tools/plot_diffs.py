# Copyright (C) 2015, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import os, argparse
import h5py
import numpy as np
np.seterr(divide='ignore')
import matplotlib.pyplot as plt

"""Plots the differences (in dB) between a response and a reference response."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots the differences (in dB) between a response and a reference response.', usage='cd gprMax; python -m tools.plot_diffs refoutputfile outputfile')
parser.add_argument('refoutputfile', help='name of output file including path containing reference response')
parser.add_argument('outputfile', help='name of output file including path')
args = parser.parse_args()

# Load (from gprMax output file) the reference response
f = h5py.File(args.refoutputfile, 'r')
tmp = f['/rxs/rx1/']
fieldname = list(tmp.keys())[0]
refresp = np.array(tmp[fieldname])

# Load (from gprMax output file) the response
f = h5py.File(args.outputfile, 'r')
tmp = f['/rxs/rx1/']
fieldname = list(tmp.keys())[0]
modelresp = np.array(tmp[fieldname])

# Calculate differences
diffdB = np.abs(modelresp - refresp) / np.amax(np.abs(refresp))
diffdB = 20 * np.log10(diffdB)
print(np.abs(np.sum(diffdB[-np.isneginf(diffdB)])) / len(diffdB[-np.isneginf(diffdB)]))

# Plot differences
fig, ax = plt.subplots(subplot_kw=dict(xlabel='Iterations', ylabel='Error [dB]'), num=args.outputfile, figsize=(20, 10), facecolor='w', edgecolor='w')
ax.plot(diffdB, 'r', lw=2)
ax.grid()
plt.show()












    