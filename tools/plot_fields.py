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

import numpy as np
import matplotlib.pyplot as plt

def plot_Ascan(figurename, time, Ex, Ey, Ez, Hx, Hy, Hz):
    """Plot electric and magnetic fields.
    
    Args:
        figurename (str): Name of figure for titlebar
        time (float): Array containing time
        Ex, Ey, Ez, Hx, Hy, Hz (float): Arrays containing electric and magnetic field values
    
    """
    
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), num=figurename, figsize=(20, 10), facecolor='w', edgecolor='w')
    ax1.plot(time, Ex,'r', lw=2, label='Ex')
    ax3.plot(time, Ey,'r', lw=2, label='Ey')
    ax5.plot(time, Ez,'r', lw=2, label='Ez')
    ax2.plot(time, Hx,'b', lw=2, label='Hx')
    ax4.plot(time, Hy,'b', lw=2, label='Hy')
    ax6.plot(time, Hz,'b', lw=2, label='Hz')
    
    # Make subplots close to each other
#    fig.subplots_adjust(hspace=0)

    # Set ylabels
    ylabels = ['$E_x$, field strength [V/m]', '$H_x$, field strength [A/m]', '$E_y$, field strength [V/m]', '$H_y$, field strength [A/m]', '$E_z$, field strength [V/m]', '$H_z$, field strength [A/m]']
    [ax.set_ylabel(ylabels[index]) for index, ax in enumerate(fig.axes)]

    # Turn on grid
    [ax.grid() for ax in fig.axes]

    # Hide x ticks for all but bottom plots
#    plt.setp([ax.get_xticklabels() for ax in fig.axes[:-2]], visible=False)

    return fig, plt


