# Copyright (C) 2015-2023: The University of Edinburgh
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

from string import Template

import h5py

from gprMax._version import __version__
from gprMax.grid import Ix, Iy, Iz

# Field component mapping for efficient lookups
_FIELD_MAP = {
    'Ex': 0, 'Ey': 1, 'Ez': 2,
    'Hx': 3, 'Hy': 4, 'Hz': 5
}

# Current function mapping
_CURRENT_FUNC_MAP = {
    'Ix': Ix,
    'Iy': Iy,
    'Iz': Iz
}


def _prepare_receiver_cache(G):
    """Pre-compute receiver data structure for faster access during simulation.
    
    Creates a cache that stores receiver coordinates and output arrays
    in a format optimized for fast iteration, avoiding repeated attribute
    lookups during the hot loop.
    
    Args:
        G (class): Grid class instance - holds essential parameters describing the model.
        
    Returns:
        list: List of receiver cache entries, where each entry is a list of
              output tuples: (output_type, output_array, x, y, z, func_or_idx)
              - output_type: 'field' or 'current'
              - output_array: numpy array to store the output
              - x, y, z: receiver coordinates
              - func_or_idx: function (for current) or field index (for field)
    """
    rx_cache = []
    
    for rx in G.rxs:
        # Cache coordinates once
        x, y, z = rx.xcoord, rx.ycoord, rx.zcoord
        receiver_outputs = []
        
        for output_name, output_array in rx.outputs.items():
            if output_name in _FIELD_MAP:
                # Field output: store type, array, coords, and field index
                receiver_outputs.append(('field', output_array, x, y, z, _FIELD_MAP[output_name]))
            elif output_name in _CURRENT_FUNC_MAP:
                # Current output: store type, array, coords, and function
                receiver_outputs.append(('current', output_array, x, y, z, _CURRENT_FUNC_MAP[output_name]))
            else:
                raise ValueError(f"Unknown output type: {output_name}")
        
        rx_cache.append(receiver_outputs)
    
    return rx_cache


def store_outputs(iteration, Ex, Ey, Ez, Hx, Hy, Hz, G):
    """Stores field component values for every receiver and transmission line.

    Args:
        iteration (int): Current iteration number.
        Ex, Ey, Ez, Hx, Hy, Hz (memory view): Current electric and magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    # Create field tuple once for efficient access
    fields = (Ex, Ey, Ez, Hx, Hy, Hz)
    
    # Use cached receiver data if available (optimization #1), otherwise fall back to standard method
    if hasattr(G, '_rx_cache') and G._rx_cache is not None:
        # Use pre-computed cache for maximum performance
        for receiver_outputs in G._rx_cache:
            for output_type, output_array, x, y, z, func_or_idx in receiver_outputs:
                if output_type == 'field':
                    # Direct field component access using cached index
                    output_array[iteration] = fields[func_or_idx][x, y, z]
                else:  # output_type == 'current'
                    # Call cached function reference
                    output_array[iteration] = func_or_idx(x, y, z, Hx, Hy, Hz, G)
    else:
        # Standard method (backward compatible)
        rxs = G.rxs
        transmissionlines = G.transmissionlines

        for rx in rxs:
            # Cache coordinates once per receiver to avoid repeated attribute access
            x, y, z = rx.xcoord, rx.ycoord, rx.zcoord
            outputs = rx.outputs  # Cache dictionary reference
            
            for output in outputs:
                # Store electric or magnetic field components
                if output in _FIELD_MAP:
                    field_idx = _FIELD_MAP[output]
                    outputs[output][iteration] = fields[field_idx][x, y, z]
                # Store current component
                elif output in _CURRENT_FUNC_MAP:
                    func = _CURRENT_FUNC_MAP[output]
                    outputs[output][iteration] = func(x, y, z, Hx, Hy, Hz, G)
                else:
                    raise ValueError(f"Unknown output type: {output}")

    # Process transmission lines
    transmissionlines = G.transmissionlines
    for tl in transmissionlines:
        tl.Vtotal[iteration] = tl.voltage[tl.antpos]
        tl.Itotal[iteration] = tl.current[tl.antpos]


kernel_template_store_outputs = Template("""

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
    """Write an output file in HDF5 format.

    Args:
        outputfile (str): Name of the output file.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    f = h5py.File(outputfile, 'w')
    
    # Pre-compute source list once to avoid repeated concatenations
    srclist = list(G.voltagesources) + list(G.hertziandipoles) + list(G.magneticdipoles)
    nsrc = len(srclist) + len(G.transmissionlines)
    
    # Cache frequently used attributes
    dx, dy, dz = G.dx, G.dy, G.dz
    
    f.attrs['gprMax'] = __version__
    f.attrs['Title'] = G.title
    f.attrs['Iterations'] = G.iterations
    f.attrs['nx_ny_nz'] = (G.nx, G.ny, G.nz)
    f.attrs['dx_dy_dz'] = (dx, dy, dz)
    f.attrs['dt'] = G.dt
    f.attrs['nsrc'] = nsrc
    f.attrs['nrx'] = len(G.rxs)
    f.attrs['srcsteps'] = G.srcsteps
    f.attrs['rxsteps'] = G.rxsteps

    # Create group for sources (except transmission lines); add type and positional data attributes
    for srcindex, src in enumerate(srclist, start=1):
        grp = f.create_group(f'/srcs/src{srcindex}')
        grp.attrs['Type'] = type(src).__name__
        grp.attrs['Position'] = (src.xcoord * dx, src.ycoord * dy, src.zcoord * dz)

    # Create group for transmission lines; add positional data, line resistance and
    # line discretisation attributes; write arrays for line voltages and currents
    for tlindex, tl in enumerate(G.transmissionlines, start=1):
        grp = f.create_group(f'/tls/tl{tlindex}')
        grp.attrs['Position'] = (tl.xcoord * dx, tl.ycoord * dy, tl.zcoord * dz)
        grp.attrs['Resistance'] = tl.resistance
        grp.attrs['dl'] = tl.dl
        # Save incident voltage and current
        grp['Vinc'] = tl.Vinc
        grp['Iinc'] = tl.Iinc
        # Save total voltage and current
        f[f'/tls/tl{tlindex}/Vtotal'] = tl.Vtotal
        f[f'/tls/tl{tlindex}/Itotal'] = tl.Itotal

    # Create group, add positional data and write field component arrays for receivers
    for rxindex, rx in enumerate(G.rxs, start=1):
        grp = f.create_group(f'/rxs/rx{rxindex}')
        if rx.ID:
            grp.attrs['Name'] = rx.ID
        grp.attrs['Position'] = (rx.xcoord * dx, rx.ycoord * dy, rx.zcoord * dz)

        for output in rx.outputs:
            f[f'/rxs/rx{rxindex}/{output}'] = rx.outputs[output]
