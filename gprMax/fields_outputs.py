# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
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
from string import Template

import h5py

from ._version import __version__

logger = logging.getLogger(__name__)


def store_outputs(G):
    """Stores field component values for every receiver and transmission line.

    Args:
        iteration (int): Current iteration number.
        Ex, Ey, Ez, Hx, Hy, Hz (memory view): Current electric and magnetic
                                                field values.
        G (FDTDGrid): Parameters describing a grid in a model.
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


knl_template_store_outputs = Template("""

// Macros for converting subscripts to linear index:
#define INDEX2D_RXCOORDS(m, n) (m)*($NY_RXCOORDS)+(n)
#define INDEX3D_RXS(i, j, k) (i)*($NY_RXS)*($NZ_RXS)+(j)*($NZ_RXS)+(k)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)

//////////////////////////////////////////////////////
// Stores field component values for every receiver //
//////////////////////////////////////////////////////

__global__ void store_outputs(int NRX, int iteration, const int* __restrict__ rxcoords, $REAL *rxs, const $REAL* __restrict__ Ex, const $REAL* __restrict__ Ey, const $REAL* __restrict__ Ez, const $REAL* __restrict__ Hx, const $REAL* __restrict__ Hy, const $REAL* __restrict__ Hz) {

    //  This function stores field component values for every receiver in the model.
    //
    //  Args:
    //      NRX: Total number of receivers in the model
    //      rxs: Array to store field components for receivers - rows are field components; columns are iterations; pages are receivers
    //      E, H: Access to field component arrays

    // Obtain the linear index corresponding to the current thread and use for each receiver
    int rx = blockIdx.x * blockDim.x + threadIdx.x;

    int i, j, k;

    if (rx < NRX) {
        i = rxcoords[INDEX2D_RXCOORDS(rx,0)];
        j = rxcoords[INDEX2D_RXCOORDS(rx,1)];
        k = rxcoords[INDEX2D_RXCOORDS(rx,2)];
        rxs[INDEX3D_RXS(0,iteration,rx)] = Ex[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(1,iteration,rx)] = Ey[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(2,iteration,rx)] = Ez[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(3,iteration,rx)] = Hx[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(4,iteration,rx)] = Hy[INDEX3D_FIELDS(i,j,k)];
        rxs[INDEX3D_RXS(5,iteration,rx)] = Hz[INDEX3D_FIELDS(i,j,k)];
    }
}

""")


def write_hdf5_outputfile(outputfile, G):
    """Write an output file in HDF5 (.h5) format.

    Args:
        outputfile (str): Name of the output file.
        G (FDTDGrid): Parameters describing a grid in a model.
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
        write_grid(f, G)

    # Write meta data and data for any subgrids
    if sg_rxs:
        for sg in G.subgrids:
            grp = f.create_group('/subgrids/' + sg.name)
            write_grid(grp, sg, is_subgrid=True)

    if G.rxs or sg_rxs:
        logger.basic(f'Written output file: {outputfile.name}')


def write_grid(basegrp, G, is_subgrid=False):
    """Write grid meta data and data to HDF5 group.

    Args:
        basegrp (dict): HDF5 group.
        G (FDTDGrid): Parameters describing a grid in a model.
        is_subgrid (bool): Is grid instance the main grid or a subgrid.
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
        basegrp.attrs['subgrid_pml_thickness'] = G.pmlthickness['x0']
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
