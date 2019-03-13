# Copyright (C) 2015-2019: The University of Edinburgh
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

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError
from .outputfiles_merge import get_output_data


def scaleoutput(outputdata,scalestring,dt):
    """Scales B-scan data to plot

    Args:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        scalestring (string): 'gsc' or 'agc' argument passed to plot_Bscan script
        dt (float): Temporal resolution of the model.
    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data with scaling applied

    AGC (Automated gain control) - dividing by local mean to obtain a desired rms for a local window
    GSC (Geometrical Spreading Compensation) - multiply by t^2 array to approximate
        amplitude loss with time


    """
    if scalestring=='agc':
        outp=[]
        meanval=np.mean(np.abs(outputdata))
        if meanval==0:
            print('rms value=0, do you have any data?')
        for trace in outputdata.T:
            outp.append(np.divide(1*trace,np.abs(trace)+meanval))
        outputdata=np.asarray(outp).T
    if scalestring=='gsc':
        tarray=np.linspace(0,dt*outputdata.shape[0],outputdata.shape[0])**2
        outputdata*=tarray[:, np.newaxis]
    return outputdata

def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent,scale=None):
    """Creates a plot (with matplotlib) of the B-scan.

    Args:
        filename (string): Filename (including path) of output file.
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.


    Returns:
        plt (object): matplotlib plot object.
    """

    if args.scale:
        outputdata=scaleoutput(outputdata,args.scale,dt)

    (path, filename) = os.path.split(filename)
    fig = plt.figure(num=filename + ' - rx' + str(rxnumber), figsize=(20, 10), facecolor='w', edgecolor='w')
    plt.imshow(outputdata, extent=[0, outputdata.shape[1], outputdata.shape[0] * dt, 0], interpolation='nearest', aspect='auto', cmap='seismic', vmin=-np.amax(np.abs(outputdata)), vmax=np.amax(np.abs(outputdata)))
    plt.xlabel('Trace number')
    plt.ylabel('Time [s]')
    # plt.title('{}'.format(filename))

    # Grid properties
    ax = fig.gca()
    ax.grid(which='both', axis='both', linestyle='-.')

    cb = plt.colorbar()
    if 'E' in rxcomponent:
        cb.set_label('Field strength [V/m]')
    elif 'H' in rxcomponent:
        cb.set_label('Field strength [A/m]')
    elif 'I' in rxcomponent:
        cb.set_label('Current [A]')

    # Save a PDF/PNG of the figure
    # savefile = os.path.splitext(filename)[0]
    # fig.savefig(path + os.sep + savefile + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    # fig.savefig(path + os.sep + savefile + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    return plt


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plots a B-scan image.', usage='cd gprMax; python -m tools.plot_Bscan outputfile output [--scale]')
    parser.add_argument('outputfile', help='name of output file including path')
    parser.add_argument('rx_component', help='name of output component to be plotted', choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz'])
    parser.add_argument('--scale', help='name of gain control to plot', choices=['agc', 'gsc'])
    args = parser.parse_args()

    # Open output file and read number of outputs (receivers)
    f = h5py.File(args.outputfile, 'r')
    nrx = f.attrs['nrx']
    f.close()

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(args.outputfile))
    elif nrx == 1:
        rx=1
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
        if outputdata.ndim==2:
            plthandle = mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component,args.scale)
        else:
            print('Throw an error')

    elif nrx > 1:
        bscan=[]
        for rx in np.arange(1,nrx,1):
            outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
            bscan.append(outputdata)
        bscan=np.asarray(bscan).T
        plthandle = mpl_plot(args.outputfile, bscan, dt, rx, args.rx_component,args.scale)


    plthandle.show()
