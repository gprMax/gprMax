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
import numpy as np

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
    f.attrs['nx, ny, nz'] = (G.nx, G.ny, G.nz)
    f.attrs['dx, dy, dz'] = (G.dx, G.dy, G.dz)
    f.attrs['dt'] = G.dt
    nsrc = len(G.voltagesources + G.hertziandipoles + G.magneticdipoles + G.transmissionlines)
    f.attrs['nsrc'] = nsrc
    f.attrs['nrx'] = len(G.rxs)
    f.attrs['srcsteps'] = (G.srcstepx, G.srcstepy, G.srcstepz)
    f.attrs['rxsteps'] = (G.rxstepx, G.rxstepy, G.rxstepz)
    

    # Create group for sources (except transmission lines); add type and positional data attributes
    if G.txs: # G.txs will be populated only if this is being used for converting old style output file to HDF5 format
        srclist = G.txs
    else:
        srclist = G.voltagesources + G.hertziandipoles + G.magneticdipoles

    for srcindex, src in enumerate(srclist):
        tmp = f.create_group('/srcs/src' + str(srcindex + 1))
        tmp.attrs['Type'] = type(src).__name__
        tmp.attrs['Position'] = (src.positionx * G.dx, src.positiony * G.dy, src.positionz * G.dz)
    
    # Create group for transmission lines; add positional data, line resistance and line discretisation attributes; initialise arrays for line voltages and currents
    if G.transmissionlines:
        for tlindex, tl in enumerate(G.transmissionlines):
            waveform = next(x for x in G.waveforms if x.ID == tl.waveformID)
            tmp = f.create_group('/tls/tl' + str(tlindex + 1))
            tmp.attrs['Position'] = (tl.positionx * G.dx, tl.positiony * G.dy, tl.positionz * G.dz)
            tmp.attrs['Resistance'] = tl.resistance
            tmp.attrs['dl'] = tl.dl
            Vinc = np.zeros(G.iterations, dtype=floattype)
            for timestep in range(G.iterations):
                Vinc[timestep] = waveform.amp * waveform.calculate_value(timestep * G.dt, G.dt)
            tmp['Vinc'] = Vinc
            tmp['Vscat'] = np.zeros(G.iterations, dtype=floattype)
            tmp['Iscat'] = np.zeros(G.iterations, dtype=floattype)
            tmp['Vtotal'] = np.zeros(G.iterations, dtype=floattype)
            tmp['Itotal'] = np.zeros(G.iterations, dtype=floattype)
    
    # Create group and add positional data and initialise field component arrays for receivers
    for rxindex, rx in enumerate(G.rxs):
        tmp = f.create_group('/rxs/rx' + str(rxindex + 1))
        if rx.ID:
            tmp.attrs['Name'] = rx.ID
        tmp.attrs['Position'] = (rx.positionx * G.dx, rx.positiony * G.dy, rx.positionz * G.dz)
        if 'Ex' in rx.outputs:
            tmp['Ex'] = np.zeros(G.iterations, dtype=floattype)
        if 'Ey' in rx.outputs:
            tmp['Ey'] = np.zeros(G.iterations, dtype=floattype)
        if 'Ez' in rx.outputs:
            tmp['Ez'] = np.zeros(G.iterations, dtype=floattype)
        if 'Hx' in rx.outputs:
            tmp['Hx'] = np.zeros(G.iterations, dtype=floattype)
        if 'Hy' in rx.outputs:
            tmp['Hy'] = np.zeros(G.iterations, dtype=floattype)
        if 'Hz' in rx.outputs:
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
            if 'Ex' in rx.outputs:
                f['/rxs/rx' + str(rxindex + 1) + '/Ex'][timestep] = Ex[rx.positionx, rx.positiony, rx.positionz]
            if 'Ey' in rx.outputs:
                f['/rxs/rx' + str(rxindex + 1) + '/Ey'][timestep] = Ey[rx.positionx, rx.positiony, rx.positionz]
            if 'Ez' in rx.outputs:
                f['/rxs/rx' + str(rxindex + 1) + '/Ez'][timestep] = Ez[rx.positionx, rx.positiony, rx.positionz]
            if 'Hx' in rx.outputs:
                f['/rxs/rx' + str(rxindex + 1) + '/Hx'][timestep] = Hx[rx.positionx, rx.positiony, rx.positionz]
            if 'Hy' in rx.outputs:
                f['/rxs/rx' + str(rxindex + 1) + '/Hy'][timestep] = Hy[rx.positionx, rx.positiony, rx.positionz]
            if 'Hz' in rx.outputs:
                f['/rxs/rx' + str(rxindex + 1) + '/Hz'][timestep] = Hz[rx.positionx, rx.positiony, rx.positionz]

        if G.transmissionlines:
            for tlindex, tl in enumerate(G.transmissionlines):
                f['/tls/tl' + str(tlindex + 1) + '/Vscat'][timestep] = tl.voltage[tl.srcpos - 1]
                f['/tls/tl' + str(tlindex + 1) + '/Iscat'][timestep] = tl.current[tl.srcpos - 1]
                f['/tls/tl' + str(tlindex + 1) + '/Vtotal'][timestep] = tl.voltage[-2]
                f['/tls/tl' + str(tlindex + 1) + '/Itotal'][timestep] = tl.current[-2]

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

