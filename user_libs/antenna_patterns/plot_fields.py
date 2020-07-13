# Copyright (C) 2016, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1016/j.sigpro.2016.04.010

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from gprMax.constants import c, z0


# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot field patterns from a simulation with receivers positioned in circles around an antenna. This module should be used after the field pattern data has been processed and stored using the initial_save.py module.', usage='cd gprMax; python -m user_libs.antenna_patterns.plot_fields numpyfile')
parser.add_argument('numpyfile', help='name of numpy file including path')
# parser.add_argument('hertzian', help='name of numpy file including path')
args = parser.parse_args()
patterns = np.load(args.numpyfile)
# hertzian = np.load(args.hertzian)

########################################
# User configurable parameters

# Pattern type (E or H)
type = 'H'

# Relative permittivity of half-space for homogeneous materials (set to None for inhomogeneous)
epsr = 5

# Observation radii and angles
radii = np.linspace(0.1, 0.3, 20)
theta = np.linspace(3, 357, 60)
theta = np.deg2rad(np.append(theta, theta[0]))  # Append start value to close circle

# Centre frequency of modelled antenna
f = 1.5e9  # GSSI 1.5GHz antenna model

# Largest dimension of antenna transmitting element
D = 0.060  # GSSI 1.5GHz antenna model

# Minimum value for plotting energy and ring steps (dB)
min = -72
step = 12
########################################

# Critical angle and velocity
if epsr:
    mr = 1
    z1 = np.sqrt(mr / epsr) * z0
    v1 = c / np.sqrt(epsr)
    thetac = np.round(np.rad2deg(np.arcsin(v1 / c)))
    wavelength = v1 / f

# Print some useful information
print('Centre frequency: {} GHz'.format(f / 1e9))
if epsr:
    print('Critical angle for Er {} is {} degrees'.format(epsr, thetac))
    print('Wavelength: {:.3f} m'.format(wavelength))
    print('Observation distance(s) from {:.3f} m ({:.1f} wavelengths) to {:.3f} m ({:.1f} wavelengths)'.format(radii[0], radii[0] / wavelength, radii[-1], radii[-1] / wavelength))
    print('Theoretical boundary between reactive & radiating near-field (0.62*sqrt((D^3/wavelength): {:.3f} m'.format(0.62 * np.sqrt((D**3) / wavelength)))
    print('Theoretical boundary between radiating near-field & far-field (2*D^2/wavelength): {:.3f} m'.format((2 * D**2) / wavelength))

# Setup figure
fig = plt.figure(num=args.numpyfile, figsize=(8, 8), facecolor='w', edgecolor='w')
ax = plt.subplot(111, polar=True)
cmap = plt.cm.get_cmap('rainbow')
ax.set_prop_cycle('color', [cmap(i) for i in np.linspace(0, 1, len(radii))])

# Critical angle window and air/subsurface interface lines
if epsr:
    ax.plot([0, np.deg2rad(180 - thetac)], [min, 0], color='0.7', lw=2)
    ax.plot([0, np.deg2rad(180 + thetac)], [min, 0], color='0.7', lw=2)
ax.plot([np.deg2rad(270), np.deg2rad(90)], [0, 0], color='0.7', lw=2)
ax.annotate('Air', xy=(np.deg2rad(270), 0), xytext=(8, 8), textcoords='offset points')
ax.annotate('Ground', xy=(np.deg2rad(270), 0), xytext=(8, -15), textcoords='offset points')

# Plot patterns
for patt in range(0, len(radii)):
    pattplot = np.append(patterns[patt, :], patterns[patt, 0])  # Append start value to close circle
    pattplot = pattplot / np.max(np.max(patterns))  # Normalise, based on set of patterns

    # Calculate power (ignore warning from taking a log of any zero values)
    with np.errstate(divide='ignore'):
        power = 10 * np.log10(pattplot)
    # Replace any NaNs or Infs from zero division
    power[np.invert(np.isfinite(power))] = 0
    
    ax.plot(theta, power, label='{:.2f}m'.format(radii[patt]), marker='.', ms=6, lw=1.5)

# Add Hertzian dipole plot
# hertzplot1 = np.append(hertzian[0, :], hertzian[0, 0]) # Append start value to close circle
# hertzplot1 = hertzplot1 / np.max(np.max(hertzian))
# ax.plot(theta, 10 * np.log10(hertzplot1), label='Inf. dipole, 0.1m', color='black', ls='-.', lw=3)
# hertzplot2 = np.append(hertzian[-1, :], hertzian[-1, 0]) # Append start value to close circle
# hertzplot2 = hertzplot2 / np.max(np.max(hertzian))
# ax.plot(theta, 10 * np.log10(hertzplot2), label='Inf. dipole, 0.58m', color='black', ls='--', lw=3)

# Theta axis options
ax.set_theta_zero_location('N')
ax.set_theta_direction('clockwise')
ax.set_thetagrids(np.arange(0, 360, 30))

# Radial axis options
ax.set_rmax(0)
ax.set_rlabel_position(45)
ax.set_yticks(np.arange(min, step, step))
yticks = ax.get_yticks().tolist()
yticks[-1] = '0 dB'
ax.set_yticklabels(yticks)

# Grid and legend
ax.grid(True)
handles, existlabels = ax.get_legend_handles_labels()
leg = ax.legend([handles[0], handles[-1]], [existlabels[0], existlabels[-1]], ncol=2, loc=(0.27, -0.12), frameon=False)  # Plot just first and last legend entries
# leg = ax.legend([handles[0], handles[-3], handles[-2], handles[-1]], [existlabels[0], existlabels[-3], existlabels[-2], existlabels[-1]], ncol=4, loc=(-0.13,-0.12), frameon=False)
[legobj.set_linewidth(2) for legobj in leg.legendHandles]

# Save a pdf of the plot
savename = os.path.splitext(args.numpyfile)[0] + '.pdf'
fig.savefig(savename, dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
# savename = os.path.splitext(args.numpyfile)[0] + '.png'
# fig.savefig(savename, dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()
