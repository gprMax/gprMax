# Copyright (C) 2016, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1016/j.sigpro.2016.04.010

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from gprMax.constants import c, z0


# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate and store (in a Numpy file) field patterns from a simulation with receivers positioned in circles around an antenna.', usage='cd gprMax; python -m user_libs.antenna_patterns.initial_save outputfile')
parser.add_argument('outputfile', help='name of gprMax output file including path')
args = parser.parse_args()
outputfile = args.outputfile

########################################
# User configurable parameters

# Pattern type (E or H)
type = 'H'

# Antenna (true if using full antenna model; false for a theoretical Hertzian dipole
antenna = True

# Relative permittivity of half-space for homogeneous materials (set to None for inhomogeneous)
epsr = 5

# Observation radii and angles
radii = np.linspace(0.1, 0.3, 20)
theta = np.linspace(3, 357, 60) * (np.pi / 180)

# Scaling of time-domain field pattern values by material impedance
impscaling = False

# Centre frequency of modelled antenna
f = 1.5e9  # GSSI 1.5GHz antenna model

# Largest dimension of antenna transmitting element
D = 0.060  # GSSI 1.5GHz antenna model

# Traces to plot for sanity checking
traceno = np.s_[:]  # All traces
########################################

# Critical angle and velocity
if epsr:
    mr = 1
    z1 = np.sqrt(mr / epsr) * z0
    v1 = c / np.sqrt(epsr)
    thetac = np.round(np.arcsin(v1 / c) * (180 / np.pi))
    wavelength = v1 / f

# Print some useful information
print('Centre frequency: {} GHz'.format(f / 1e9))
if epsr:
    print('Critical angle for Er {} is {} degrees'.format(epsr, thetac))
    print('Wavelength: {:.3f} m'.format(wavelength))
    print('Observation distance(s) from {:.3f} m ({:.1f} wavelengths) to {:.3f} m ({:.1f} wavelengths)'.format(radii[0], radii[0] / wavelength, radii[-1], radii[-1] / wavelength))
    print('Theoretical boundary between reactive & radiating near-field (0.62*sqrt((D^3/wavelength): {:.3f} m'.format(0.62 * np.sqrt((D**3) / wavelength)))
    print('Theoretical boundary between radiating near-field & far-field (2*D^2/wavelength): {:.3f} m'.format((2 * D**2) / wavelength))

# Load text file with coordinates of pattern origin
origin = np.loadtxt(os.path.splitext(outputfile)[0] + '_rxsorigin.txt')

# Load output file and read some header information
f = h5py.File(outputfile, 'r')
iterations = f.attrs['Iterations']
dt = f.attrs['dt']
nrx = f.attrs['nrx']
if antenna:
    nrx = nrx - 1  # Ignore first receiver point with full antenna model
    start = 2
else:
    start = 1
time = np.arange(0, dt * iterations, dt)
time = time / 1E-9
fs = 1 / dt  # Sampling frequency

# Initialise arrays to store fields
coords = np.zeros((nrx, 3), dtype=np.float32)
Ex = np.zeros((iterations, nrx), dtype=np.float32)
Ey = np.zeros((iterations, nrx), dtype=np.float32)
Ez = np.zeros((iterations, nrx), dtype=np.float32)
Hx = np.zeros((iterations, nrx), dtype=np.float32)
Hy = np.zeros((iterations, nrx), dtype=np.float32)
Hz = np.zeros((iterations, nrx), dtype=np.float32)
Er = np.zeros((iterations, nrx), dtype=np.float32)
Etheta = np.zeros((iterations, nrx), dtype=np.float32)
Ephi = np.zeros((iterations, nrx), dtype=np.float32)
Hr = np.zeros((iterations, nrx), dtype=np.float32)
Htheta = np.zeros((iterations, nrx), dtype=np.float32)
Hphi = np.zeros((iterations, nrx), dtype=np.float32)
Ethetasum = np.zeros(len(theta), dtype=np.float32)
Hthetasum = np.zeros(len(theta), dtype=np.float32)
patternsave = np.zeros((len(radii), len(theta)), dtype=np.float32)

for rx in range(0, nrx):
    path = '/rxs/rx' + str(rx + start) + '/'
    position = f[path].attrs['Position'][:]
    coords[rx, :] = position - origin
    Ex[:, rx] = f[path + 'Ex'][:]
    Ey[:, rx] = f[path + 'Ey'][:]
    Ez[:, rx] = f[path + 'Ez'][:]
    Hx[:, rx] = f[path + 'Hx'][:]
    Hy[:, rx] = f[path + 'Hy'][:]
    Hz[:, rx] = f[path + 'Hz'][:]
f.close()

# Plot traces for sanity checking
# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(num=outputfile, nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), figsize=(20, 10), facecolor='w', edgecolor='w')
# ax1.plot(time, Ex[:, traceno],'r', lw=2)
# ax1.set_ylabel('$E_x$, field strength [V/m]')
# ax3.plot(time, Ey[:, traceno],'r', lw=2)
# ax3.set_ylabel('$E_y$, field strength [V/m]')
# ax5.plot(time, Ez[:, traceno],'r', lw=2)
# ax5.set_ylabel('$E_z$, field strength [V/m]')
# ax2.plot(time, Hx[:, traceno],'b', lw=2)
# ax2.set_ylabel('$H_x$, field strength [A/m]')
# ax4.plot(time, Hy[:, traceno],'b', lw=2)
# ax4.set_ylabel('$H_y$, field strength [A/m]')
# ax6.plot(time, Hz[:, traceno],'b', lw=2)
# ax6.set_ylabel('$H_z$, field strength [A/m]')
# Turn on grid
# [ax.grid() for ax in fig.axes]
# plt.show()

# Calculate fields for patterns
rxstart = 0  # Index for rx points
for radius in range(0, len(radii)):
    index = 0
    # Observation points
    for pt in range(rxstart, rxstart + len(theta)):
        # Cartesian to spherical coordinate transform coefficients from (Kraus,1991,Electromagnetics,p.34)
        r1 = coords[pt, 0] / np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2 + coords[pt, 2]**2)
        r2 = coords[pt, 1] / np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2 + coords[pt, 2]**2)
        r3 = coords[pt, 2] / np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2 + coords[pt, 2]**2)
        theta1 = (coords[pt, 0] * coords[pt, 2]) / (np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2) * np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2 + coords[pt, 2]**2))
        theta2 = (coords[pt, 1] * coords[pt, 2]) / (np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2) * np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2 + coords[pt, 2]**2))
        theta3 = -(np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2) / np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2 + coords[pt, 2]**2))
        phi1 = -(coords[pt, 1] / np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2))
        phi2 = coords[pt, 0] / np.sqrt(coords[pt, 0]**2 + coords[pt, 1]**2)
        phi3 = 0

        # Fields in spherical coordinates
        Er[:, index] = Ex[:, pt] * r1 + Ey[:, pt] * r2 + Ez[:, pt] * r3
        Etheta[:, index] = Ex[:, pt] * theta1 + Ey[:, pt] * theta2 + Ez[:, pt] * theta3
        Ephi[:, index] = Ex[:, pt] * phi1 + Ey[:, pt] * phi2 + Ez[:, pt] * phi3
        Hr[:, index] = Hx[:, pt] * r1 + Hy[:, pt] * r2 + Hz[:, pt] * r3
        Htheta[:, index] = Hx[:, pt] * theta1 + Hy[:, pt] * theta2 + Hz[:, pt] * theta3
        Hphi[:, index] = Hx[:, pt] * phi1 + Hy[:, pt] * phi2 + Hz[:, pt] * phi3

        # Calculate metric for time-domain field pattern values
        if impscaling and coords[pt, 2] < 0:
            Ethetasum[index] = np.sum(Etheta[:, index]**2) / z1
            Hthetasum[index] = np.sum(Htheta[:, index]**2) / z1
        else:
            Ethetasum[index] = np.sum(Etheta[:, index]**2) / z0
            Hthetasum[index] = np.sum(Htheta[:, index]**2) / z0

        index += 1

    if type == 'H':
        # Flip H-plane patterns as rx points are written CCW but always plotted CW
        patternsave[radius, :] = Hthetasum[::-1]
    elif type == 'E':
        patternsave[radius, :] = Ethetasum

    rxstart += len(theta)

# Save pattern to numpy file
np.save(os.path.splitext(outputfile)[0], patternsave)
print('Written Numpy file: {}.npy'.format(os.path.splitext(outputfile)[0]))
