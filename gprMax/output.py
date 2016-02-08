# Copyright (C) 2015-2016: The University of Edinburgh
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

import h5py
import numpy as np

from gprMax.constants import floattype
from gprMax.grid import Ix, Iy, Iz


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
    srclist = G.voltagesources + G.hertziandipoles + G.magneticdipoles
    for srcindex, src in enumerate(srclist):
        grp = f.create_group('/srcs/src' + str(srcindex + 1))
        grp.attrs['Type'] = type(src).__name__
        grp.attrs['Position'] = (src.positionx * G.dx, src.positiony * G.dy, src.positionz * G.dz)
    
    # Create group for transmission lines; add positional data, line resistance and line discretisation attributes; initialise arrays for line voltages and currents
    if G.transmissionlines:
        for tlindex, tl in enumerate(G.transmissionlines):
            grp = f.create_group('/tls/tl' + str(tlindex + 1))
            grp.attrs['Position'] = (tl.positionx * G.dx, tl.positiony * G.dy, tl.positionz * G.dz)
            grp.attrs['Resistance'] = tl.resistance
            grp.attrs['dl'] = tl.dl
            # Save incident voltage and current
            grp['Vinc'] = tl.Vinc
            grp['Iinc'] = tl.Iinc
            grp.create_dataset('Vtotal', (G.iterations, ), dtype=floattype)
            grp.create_dataset('Itotal', (G.iterations, ), dtype=floattype)
    
    # Create group and add positional data and initialise field component arrays for receivers
    for rxindex, rx in enumerate(G.rxs):
        grp = f.create_group('/rxs/rx' + str(rxindex + 1))
        if rx.ID:
            grp.attrs['Name'] = rx.ID
        grp.attrs['Position'] = (rx.positionx * G.dx, rx.positiony * G.dy, rx.positionz * G.dz)
        for output in rx.outputs:
            grp.create_dataset(output, (G.iterations, ), dtype=floattype)

    return f


def write_output(f, timestep, Ex, Ey, Ez, Hx, Hy, Hz, G):
    """Writes field component values to an output file in HDF5 format.
        
    Args:
        f (file object): File object for the file to be written to.
        timestep (int): Current iteration number.
        Ex, Ey, Ez, Hx, Hy, Hz (memory view): Current electric and magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

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
        if 'Ix' in rx.outputs:
            f['/rxs/rx' + str(rxindex + 1) + '/Ix'][timestep] = Ix(rx.positionx, rx.positiony, rx.positionz, G.Hy, G.Hz, G)
        if 'Iy' in rx.outputs:
            f['/rxs/rx' + str(rxindex + 1) + '/Iy'][timestep] = Iy(rx.positionx, rx.positiony, rx.positionz, G.Hx, G.Hz, G)
        if 'Iz' in rx.outputs:
            f['/rxs/rx' + str(rxindex + 1) + '/Iz'][timestep] = Iz(rx.positionx, rx.positiony, rx.positionz, G.Hx, G.Hy, G)

    if G.transmissionlines:
        for tlindex, tl in enumerate(G.transmissionlines):
            f['/tls/tl' + str(tlindex + 1) + '/Vtotal'][timestep] = tl.voltage[tl.antpos]
            f['/tls/tl' + str(tlindex + 1) + '/Itotal'][timestep] = tl.current[tl.antpos]

