# Copyright (C) 2015: The University of Edinburgh
#            Authors: Craig Warren and Antonis Giannopoulos
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

import os, argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

"""Plots electric and magnetic fields from all receiver points in the given output file. Each receiver point is plotted in a new figure window."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots electric and magnetic fields from all receiver points in the given output file. Each receiver point is plotted in a new figure window.', usage='cd gprMax; python -m tools.plot_Ascan outputfile')
parser.add_argument('outputfile', help='name of output file including path')
args = parser.parse_args()

file = args.outputfile
f = h5py.File(file, 'r')
nrx = f.attrs['nrx']
time = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt'])
time = time / 1e-9

for rx in range(1, nrx + 1):
    path = '/rxs/rx' + str(rx) + '/'
    Ex = f[path + 'Ex'][:]
    Ey = f[path + 'Ey'][:]
    Ez = f[path + 'Ez'][:]
    Hx = f[path + 'Hx'][:]
    Hy = f[path + 'Hy'][:]
    Hz = f[path + 'Hz'][:]
    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
    ax1.plot(time, Ex,'r', lw=2, label='Ex')
    ax3.plot(time, Ey,'r', lw=2, label='Ey')
    ax5.plot(time, Ez,'r', lw=2, label='Ez')
    ax2.plot(time, Hx,'b', lw=2, label='Hx')
    ax4.plot(time, Hy,'b', lw=2, label='Hy')
    ax6.plot(time, Hz,'b', lw=2, label='Hz')

    # Set ylabels
    ylabels = ['$E_x$, field strength [V/m]', '$H_x$, field strength [A/m]', '$E_y$, field strength [V/m]', '$H_y$, field strength [A/m]', '$E_z$, field strength [V/m]', '$H_z$, field strength [A/m]']
    [ax.set_ylabel(ylabels[index]) for index, ax in enumerate(fig.axes)]

    # Turn on grid
    [ax.grid() for ax in fig.axes]

    # Save a PDF of the figure
    #fig.savefig(os.path.splitext(os.path.abspath(file))[0] + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
f.close()