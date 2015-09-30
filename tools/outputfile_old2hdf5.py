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

import os, struct, argparse
import numpy as np

from gprMax.grid import FDTDGrid
from gprMax.receivers import Rx
from gprMax.fields_output import prepare_output_file, write_output

"""Converts old output file to new HDF5 format."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Converts old output file to new HDF5 format.', usage='cd gprMax; python -m tools.outputfile_old2hdf5 outputfile')
parser.add_argument('outputfile', help='name of output file including path')
args = parser.parse_args()

outputfile = args.outputfile
G = FDTDGrid()

print("Reading: '{}'".format(outputfile))

with open(outputfile, 'rb') as f:
    # Get information from file header
    f.read(2)
    filetype, = struct.unpack('h', f.read(2))
    myshort, = struct.unpack('h', f.read(2))
    myfloat, = struct.unpack('h', f.read(2))
    titlelength, = struct.unpack('h', f.read(2))
    sourcelength, = struct.unpack('h', f.read(2))
    medialength, = struct.unpack('h', f.read(2))
    reserved, = struct.unpack('h', f.read(2))
    G.title = ''
    for c in range(titlelength):
        tmp, = struct.unpack('c', f.read(1))
        G.title += tmp.decode('utf-8')
    G.iterations, = struct.unpack('f', f.read(4))
    G.iterations = int(G.iterations)
    G.dx, = struct.unpack('f', f.read(4))
    G.dy, = struct.unpack('f', f.read(4))
    G.dz, = struct.unpack('f', f.read(4))
    G.dt, = struct.unpack('f', f.read(4))
    nsteps, = struct.unpack('h', f.read(2))
    G.txstepx, = struct.unpack('h', f.read(2))
    G.txstepy, = struct.unpack('h', f.read(2))
    G.txstepz, = struct.unpack('h', f.read(2))
    G.rxstepx, = struct.unpack('h', f.read(2))
    G.rxstepy, = struct.unpack('h', f.read(2))
    G.rxstepz, = struct.unpack('h', f.read(2))
    ntx, = struct.unpack('h', f.read(2))
    nrx, = struct.unpack('h', f.read(2))
    nrxbox, = struct.unpack('h', f.read(2))

    # Display some basic information
    print('Model title: {}'.format(G.title))
    print('Spatial discretisation: {:.3f} x {:.3f} x {:.3f} m'.format(G.dx, G.dy, G.dz))
    print('Time step: {:.3e} secs'.format(G.dt))
    print('Time window: {:.3e} secs ({} iterations)'.format(G.iterations * G.dt, G.iterations))

    # txs
    for tx in range(ntx):
        polarisation, = struct.unpack('c', f.read(1))
        x, = struct.unpack('h', f.read(2))
        y, = struct.unpack('h', f.read(2))
        z, = struct.unpack('h', f.read(2))
        for c in range(sourcelength):
            tmp, = struct.unpack('c', f.read(1))
        start, = struct.unpack('f', f.read(4))
        stop, = struct.unpack('f', f.read(4))
        # Only want transmitter position information so store in a Rx class for ease
        t = Rx(positionx=x, positiony=y, positionz=z)
        G.txs.append(t)

    # rxs
    for r in range(nrx):
        x, = struct.unpack('h', f.read(2))
        y, = struct.unpack('h', f.read(2))
        z, = struct.unpack('h', f.read(2))
        r = Rx(positionx=x, positiony=y, positionz=z)
        G.rxs.append(r)

    # rxboxes
    for rxbox in range(nrxbox):
        nrxs, = struct.unpack('h', f.read(2))
        for rx in range(nrxs):
            x, = struct.unpack('h', f.read(2))
            y, = struct.unpack('h', f.read(2))
            z, = struct.unpack('h', f.read(2))
            r = Rx(positionx=x, positiony=y, positionz=z)
            G.rxs.append(r)

    # Fields
    fieldsdata = np.fromfile(f, dtype=np.float32)
    ex = np.reshape(fieldsdata[0::9], (len(G.rxs), G.iterations, nsteps), order='F')
    ey = np.reshape(fieldsdata[1::9], (len(G.rxs), G.iterations, nsteps), order='F')
    ez = np.reshape(fieldsdata[2::9], (len(G.rxs), G.iterations, nsteps), order='F')
    hx = np.reshape(fieldsdata[3::9], (len(G.rxs), G.iterations, nsteps), order='F')
    hy = np.reshape(fieldsdata[4::9], (len(G.rxs), G.iterations, nsteps), order='F')
    hz = np.reshape(fieldsdata[5::9], (len(G.rxs), G.iterations, nsteps), order='F')
    ix = np.reshape(fieldsdata[6::9], (len(G.rxs), G.iterations, nsteps), order='F')
    iy = np.reshape(fieldsdata[7::9], (len(G.rxs), G.iterations, nsteps), order='F')
    iz = np.reshape(fieldsdata[8::9], (len(G.rxs), G.iterations, nsteps), order='F')

    if nsteps == 1:
        ex = np.transpose(ex)
        ey = np.transpose(ey)
        ez = np.transpose(ez)
        hx = np.transpose(hx)
        hy = np.transpose(hy)
        hz = np.transpose(hz)
        ix = np.transpose(ix)
        iy = np.transpose(iy)
        iz = np.transpose(iz)
    else:
        for i in range(len(G.rxs)):
            ex[:,i,:] = ex[i,:,:]
            ey[:,i,:] = ey[i,:,:]
            ez[:,i,:] = ez[i,:,:]
            hx[:,i,:] = hx[i,:,:]
            hy[:,i,:] = hy[i,:,:]
            hz[:,i,:] = hz[i,:,:]
            ix[:,i,:] = ix[i,:,:]
            iy[:,i,:] = iy[i,:,:]
            iz[:,i,:] = iz[i,:,:]

    # Remove any singleton dimensions
    ex = np.squeeze(ex)
    ey = np.squeeze(ey)
    ez = np.squeeze(ez)
    hx = np.squeeze(hx)
    hy = np.squeeze(hy)
    hz = np.squeeze(hz)
    ix = np.squeeze(ix)
    iy = np.squeeze(iy)
    iz = np.squeeze(iz)

# Create new HDF5 outputfile
newoutputfile = os.path.splitext(outputfile)
newoutputfile = newoutputfile[0] + '_hdf5.out'
f = prepare_output_file(newoutputfile, G)
write_output(f, np.s_[:], ex, ey, ez, hx, hy, hz, G)

print("Written: '{}'".format(newoutputfile))

