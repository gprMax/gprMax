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

from gprMax.exceptions import CmdInputError

"""Plots electric and magnetic fields from all receiver points in the given output file. Each receiver point is plotted in a new figure window."""

# Fields that can be plotted
fieldslist = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots electric and magnetic fields from all receiver points in the given output file. Each receiver point is plotted in a new figure window.', usage='cd gprMax; python -m tools.plot_Ascan outputfile')
parser.add_argument('outputfile', help='name of output file including path')
parser.add_argument('--fields', help='list of fields to be plotted, i.e. Ex Ey Ez', default=fieldslist, nargs='+')
args = parser.parse_args()

file = args.outputfile
f = h5py.File(file, 'r')
nrx = f.attrs['nrx']
time = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt'])
time = time / 1e-9

# Check for valid field names
for field in args.fields:
    if field not in fieldslist:
        raise CmdInputError('{} not allowed. Options are: Ex Ey Ez Hx Hy Hz'.format(field))

for rx in range(1, nrx + 1):
    path = '/rxs/rx' + str(rx) + '/'
    
    # If only a single field is required, create one subplot
    if len(args.fields) == 1:
        fielddata = f[path + args.fields[0]][:]
        if 'E' in args.fields[0]:
            fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel=args.fields[0] + ', field strength [V/m]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
            ax.plot(time, fielddata,'r', lw=2, label=args.fields[0])
            ax.grid()
        elif 'H' in args.fields[0]:
            fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel=args.fields[0] + ', field strength [A/m]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
            ax.plot(time, fielddata,'b', lw=2, label=args.fields[0])
            ax.grid()

    # If multiple fields are required, created all six subplots and populate only the specified ones
    else:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
        for field in args.fields:
            fielddata = f[path + field][:]
            if field == 'Ex':
                ax1.plot(time, fielddata,'r', lw=2, label=field)
                ax1.set_ylabel('$E_x$, field strength [V/m]')
            elif field == 'Ey':
                ax3.plot(time, fielddata,'r', lw=2, label=field)
                ax3.set_ylabel('$E_y$, field strength [V/m]')
            elif field == 'Ez':
                ax5.plot(time, fielddata,'r', lw=2, label=field)
                ax5.set_ylabel('$E_z$, field strength [V/m]')
            elif field == 'Hx':
                ax2.plot(time, fielddata,'b', lw=2, label=field)
                ax2.set_ylabel('$H_x$, field strength [A/m]')
            elif field == 'Hy':
                ax4.plot(time, fielddata,'b', lw=2, label=field)
                ax4.set_ylabel('$H_y$, field strength [A/m]')
            elif field == 'Hz':
                ax6.plot(time, fielddata,'b', lw=2, label=field)
                ax6.set_ylabel('$H_z$, field strength [A/m]')
        # Turn on grid
        [ax.grid() for ax in fig.axes]

    # Save a PDF of the figure
    #fig.savefig(os.path.splitext(os.path.abspath(file))[0] + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
f.close()