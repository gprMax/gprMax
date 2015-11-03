import numpy as np
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

import h5py

from gprMax.constants import floattype


def prepare_output_file(outputfile, G):
    """Prepares an output file in HDF5 format for writing.
        
    Args:
        outputfile (str): Name of the output file.
        G (class): Grid class instance - holds essential parameters describing the model.
        
    Returns:
        f (file object): File object for the file to be written to. 
    """

    f = h5py.File(outputfile, 'w')
    f.attrs['Title'] = G.title
    f.attrs['Iterations'] = G.iterations
    f.attrs['dx, dy, dz'] = (G.dx, G.dy, G.dz)
    f.attrs['dt'] = G.dt
    f.attrs['txsteps'] = (G.txstepx, G.txstepy, G.txstepz)
    f.attrs['rxsteps'] = (G.rxstepx, G.rxstepy, G.rxstepz)
    f.attrs['ntx'] = len(G.voltagesources) + len(G.hertziandipoles) + len(G.magneticdipoles)
    f.attrs['nrx'] = len(G.rxs)

    # Create groups for txs, rxs
    txs = f.create_group('/txs')
    rxs = f.create_group('/rxs')
    
    # Add positional data for txs
    if G.txs: # G.txs will be populated only if this is being used for converting old style output file to HDF5 format
        txlist = G.txs
    else:
        txlist = G.voltagesources + G.hertziandipoles + G.magneticdipoles
    for txindex, tx in enumerate(txlist):
        tmp = f.create_group('/txs/tx' + str(txindex + 1))
        tmp['Position'] = (tx.positionx * G.dx, tx.positiony * G.dy, tx.positionz * G.dz)
    
    # Add positional data for rxs
    for rxindex, rx in enumerate(G.rxs):
        tmp = f.create_group('/rxs/rx' + str(rxindex + 1))
        tmp['Position'] = (rx.positionx * G.dx, rx.positiony * G.dy, rx.positionz * G.dz)
        tmp['Ex'] = np.zeros(G.iterations, dtype=floattype)
        tmp['Ey'] = np.zeros(G.iterations, dtype=floattype)
        tmp['Ez'] = np.zeros(G.iterations, dtype=floattype)
        tmp['Hx'] = np.zeros(G.iterations, dtype=floattype)
        tmp['Hy'] = np.zeros(G.iterations, dtype=floattype)
        tmp['Hz'] = np.zeros(G.iterations, dtype=floattype)

    return f


def write_output(f, timestep, Ex, Ey, Ez, Hx, Hy, Hz, G):
    """Writes field component values to an output file in HDF5 format.
        
    Args:
        f (file object): File object for the file to be written to.
        timestep (int): Current iteration number.
        Ex, Ey, Ez, Hx, Hy, Hz (memory view): Current electric and magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    # Normal field writing from main
    if type(timestep) is not slice:
        # For each rx, write field component values at current timestep
        for rxindex, rx in enumerate(G.rxs):
            f['/rxs/rx' + str(rxindex + 1) + '/Ex'][timestep] = Ex[rx.positionx, rx.positiony, rx.positionz]
            f['/rxs/rx' + str(rxindex + 1) + '/Ey'][timestep] = Ey[rx.positionx, rx.positiony, rx.positionz]
            f['/rxs/rx' + str(rxindex + 1) + '/Ez'][timestep] = Ez[rx.positionx, rx.positiony, rx.positionz]
            f['/rxs/rx' + str(rxindex + 1) + '/Hx'][timestep] = Hx[rx.positionx, rx.positiony, rx.positionz]
            f['/rxs/rx' + str(rxindex + 1) + '/Hy'][timestep] = Hy[rx.positionx, rx.positiony, rx.positionz]
            f['/rxs/rx' + str(rxindex + 1) + '/Hz'][timestep] = Hz[rx.positionx, rx.positiony, rx.positionz]

    # Field writing when converting old style output file to HDF5 format
    else:
        if len(G.rxs) == 1:
            f['/rxs/rx1/Ex'][timestep] = Ex
            f['/rxs/rx1/Ey'][timestep] = Ey
            f['/rxs/rx1/Ez'][timestep] = Ez
            f['/rxs/rx1/Hx'][timestep] = Hx
            f['/rxs/rx1/Hy'][timestep] = Hy
            f['/rxs/rx1/Hz'][timestep] = Hz
        else:
            for rxindex, rx in enumerate(G.rxs):
                f['/rxs/rx' + str(rxindex + 1) + '/Ex'][timestep] = Ex[:, rxindex]
                f['/rxs/rx' + str(rxindex + 1) + '/Ey'][timestep] = Ey[:, rxindex]
                f['/rxs/rx' + str(rxindex + 1) + '/Ez'][timestep] = Ez[:, rxindex]
                f['/rxs/rx' + str(rxindex + 1) + '/Hx'][timestep] = Hx[:, rxindex]
                f['/rxs/rx' + str(rxindex + 1) + '/Hy'][timestep] = Hy[:, rxindex]
                f['/rxs/rx' + str(rxindex + 1) + '/Hz'][timestep] = Hz[:, rxindex]


