# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import logging

import h5py

from ._version import __version__

logger = logging.getLogger(__name__)


def store_outputs(G):
    """Stores field component values for every receiver and transmission line.

    Args:
        G: FDTDGrid class describing a grid in a model.
    """

    iteration = G.iteration
    Ex, Ey, Ez, Hx, Hy, Hz = G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz

    for rx in G.rxs:
        for output in rx.outputs:
            # Store electric or magnetic field components
            if 'I' not in output:
                field = locals()[output]
                rx.outputs[output][iteration] = field[rx.xcoord, rx.ycoord, rx.zcoord]
            # Store current component
            else:
                func = globals()[output]
                rx.outputs[output][iteration] = func(rx.xcoord, rx.ycoord, rx.zcoord,
                                                     Hx, Hy, Hz, G)

    for tl in G.transmissionlines:
        tl.Vtotal[iteration] = tl.voltage[tl.antpos]
        tl.Itotal[iteration] = tl.current[tl.antpos]


def write_hdf5_outputfile(outputfile, G):
    """Writes an output file in HDF5 (.h5) format.

    Args:
        outputfile: string of the name of the output file.
        G: FDTDGrid class describing a grid in a model.
    """

    # Check for any receivers in subgrids
    sg_rxs = [True for sg in G.subgrids if sg.rxs]

    # Create output file and write top-level meta data
    if G.rxs or sg_rxs:
        f = h5py.File(outputfile, 'w')
        f.attrs['gprMax'] = __version__
        f.attrs['Title'] = G.title

    # Write meta data and data for main grid
    if G.rxs:
        write_hd5_data(f, G)

    # Write meta data and data for any subgrids
    if sg_rxs:
        for sg in G.subgrids:
            grp = f.create_group('/subgrids/' + sg.name)
            write_hd5_data(grp, sg, is_subgrid=True)

    if G.rxs or sg_rxs:
        logger.basic(f'Written output file: {outputfile.name}')


def write_hd5_data(basegrp, G, is_subgrid=False):
    """Writes grid meta data and data to HDF5 group.

    Args:
        basegrp: dict of HDF5 group.
        G: FDTDGrid class describing a grid in a model.
        is_subgrid: boolean for grid instance the main grid or a subgrid.
    """

    # Write meta data for grid
    basegrp.attrs['Iterations'] = G.iterations
    basegrp.attrs['nx_ny_nz'] = (G.nx, G.ny, G.nz)
    basegrp.attrs['dx_dy_dz'] = (G.dx, G.dy, G.dz)
    basegrp.attrs['dt'] = G.dt
    nsrc = len(G.voltagesources + G.hertziandipoles + G.magneticdipoles + G.transmissionlines)
    basegrp.attrs['nsrc'] = nsrc
    basegrp.attrs['nrx'] = len(G.rxs)
    basegrp.attrs['srcsteps'] = G.srcsteps
    basegrp.attrs['rxsteps'] = G.rxsteps

    if is_subgrid:
        # Write additional meta data about subgrid
        basegrp.attrs['is_os_sep'] = G.is_os_sep
        basegrp.attrs['pml_separation'] = G.pml_separation
        basegrp.attrs['subgrid_pml_thickness'] = G.pml['thickness']['x0']
        basegrp.attrs['filter'] = G.filter
        basegrp.attrs['ratio'] = G.ratio
        basegrp.attrs['interpolation'] = G.interpolation

    # Create group for sources (except transmission lines); add type and positional data attributes
    srclist = G.voltagesources + G.hertziandipoles + G.magneticdipoles
    for srcindex, src in enumerate(srclist):
        grp = basegrp.create_group('srcs/src' + str(srcindex + 1))
        grp.attrs['Type'] = type(src).__name__
        grp.attrs['Position'] = (src.xcoord * G.dx, src.ycoord * G.dy, src.zcoord * G.dz)

    # Create group for transmission lines; add positional data, line resistance and
    # line discretisation attributes; write arrays for line voltages and currents
    for tlindex, tl in enumerate(G.transmissionlines):
        grp = basegrp.create_group('tls/tl' + str(tlindex + 1))
        grp.attrs['Position'] = (tl.xcoord * G.dx, tl.ycoord * G.dy, tl.zcoord * G.dz)
        grp.attrs['Resistance'] = tl.resistance
        grp.attrs['dl'] = tl.dl
        # Save incident voltage and current
        grp['Vinc'] = tl.Vinc
        grp['Iinc'] = tl.Iinc
        # Save total voltage and current
        basegrp['tls/tl' + str(tlindex + 1) + '/Vtotal'] = tl.Vtotal
        basegrp['tls/tl' + str(tlindex + 1) + '/Itotal'] = tl.Itotal

    # Create group, add positional data and write field component arrays for receivers
    for rxindex, rx in enumerate(G.rxs):
        grp = basegrp.create_group('rxs/rx' + str(rxindex + 1))
        if rx.ID:
            grp.attrs['Name'] = rx.ID
        grp.attrs['Position'] = (rx.xcoord * G.dx, rx.ycoord * G.dy, rx.zcoord * G.dz)

        for output in rx.outputs:
            basegrp['rxs/rx' + str(rxindex + 1) + '/' + output] = rx.outputs[output]


def Ix(x, y, z, Hx, Hy, Hz, G):
    """Calculates the x-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if y == 0 or z == 0:
        Ix = 0
    else:
        Ix = G.dy * (Hy[x, y, z - 1] - Hy[x, y, z]) + G.dz * (Hz[x, y, z] - Hz[x, y - 1, z])

    return Ix


def Iy(x, y, z, Hx, Hy, Hz, G):
    """Calculates the y-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if x == 0 or z == 0:
        Iy = 0
    else:
        Iy = G.dx * (Hx[x, y, z] - Hx[x, y, z - 1]) + G.dz * (Hz[x - 1, y, z] - Hz[x, y, z])

    return Iy


def Iz(x, y, z, Hx, Hy, Hz, G):
    """Calculates the z-component of current at a grid position.

    Args:
        x, y, z: floats for coordinates of position in grid.
        Hx, Hy, Hz: numpy array of magnetic field values.
        G: FDTDGrid class describing a grid in a model.
    """

    if x == 0 or y == 0:
        Iz = 0
    else:
        Iz = G.dx * (Hx[x, y - 1, z] - Hx[x, y, z]) + G.dy * (Hy[x, y, z] - Hy[x - 1, y, z])

    return Iz
