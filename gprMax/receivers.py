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

from collections import OrderedDict

import numpy as np

import gprMax.config as config


class Rx:
    """Receiver output points."""

    allowableoutputs = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz']
    gpu_allowableoutputs = allowableoutputs[:-3]
    defaultoutputs = allowableoutputs[:-3]
    maxnumoutputs = 0

    def __init__(self):

        self.ID = None
        self.outputs = OrderedDict()
        self.xcoord = None
        self.ycoord = None
        self.zcoord = None
        self.xcoordorigin = None
        self.ycoordorigin = None
        self.zcoordorigin = None


def gpu_initialise_rx_arrays(G):
    """Initialise arrays on GPU for receiver coordinates and to store field
        components for receivers.

    Args:
        G (FDTDGrid): Holds essential parameters describing the model.
    """

    import pycuda.gpuarray as gpuarray

    # Array to store receiver coordinates on GPU
    rxcoords = np.zeros((len(G.rxs), 3), dtype=np.int32)
    for i, rx in enumerate(G.rxs):
        rxcoords[i, 0] = rx.xcoord
        rxcoords[i, 1] = rx.ycoord
        rxcoords[i, 2] = rx.zcoord

    # Array to store field components for receivers on GPU - rows are field components;
    # columns are iterations; pages are receivers
    rxs = np.zeros((Rx.maxnumoutputs, G.iterations, len(G.rxs)),
                   dtype=config.dtypes['float_or_double'])

    # Copy arrays to GPU
    rxcoords_gpu = gpuarray.to_gpu(rxcoords)
    rxs_gpu = gpuarray.to_gpu(rxs)

    return rxcoords_gpu, rxs_gpu


def gpu_get_rx_array(rxs_gpu, rxcoords_gpu, G):
    """Copy output from receivers array used on GPU back to receiver objects.

    Args:
        rxs_gpu (float): numpy array of receiver data from GPU - rows are field
                            components; columns are iterations; pages are receivers.
        rxcoords_gpu (float): numpy array of receiver coordinates from GPU.
        G (FDTDGrid): Holds essential parameters describing the model.
    """

    for rx in G.rxs:
        for rxgpu in range(len(G.rxs)):
            if rx.xcoord == rxcoords_gpu[rxgpu, 0] and \
               rx.ycoord == rxcoords_gpu[rxgpu, 1] and \
               rx.zcoord == rxcoords_gpu[rxgpu, 2]:
                for k in rx.outputs.items():
                    rx.outputs[k] = rxs_gpu[Rx.gpu_allowableoutputs.index(k), :, rxgpu]
