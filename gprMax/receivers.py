# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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

import numpy as np

import gprMax.config as config


class Rx:
    """Receiver output points."""

    allowableoutputs = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"]
    defaultoutputs = allowableoutputs[:-3]
    allowableoutputs_dev = allowableoutputs[:-3]

    def __init__(self):
        self.ID = None
        self.outputs = {}
        self.xcoord: int
        self.ycoord: int
        self.zcoord: int
        self.xcoordorigin: int
        self.ycoordorigin: int
        self.zcoordorigin: int


def htod_rx_arrays(G, queue=None):
    """Initialise arrays on compute device for receiver coordinates and to store field
        components for receivers.

    Args:
        G: FDTDGrid class describing a grid in a model.
        queue: pyopencl queue.

    Returns:
        rxcoords_dev: int array of receiver coordinates on compute device.
        rxs_dev: float array of receiver data on compute device - rows are field
                    components; columns are iterations; pages are receivers.
    """

    # Array to store receiver coordinates on compute device
    rxcoords = np.zeros((len(G.rxs), 3), dtype=np.int32)
    for i, rx in enumerate(G.rxs):
        rxcoords[i, 0] = rx.xcoord
        rxcoords[i, 1] = rx.ycoord
        rxcoords[i, 2] = rx.zcoord

    # Array to store field components for receivers on compute device -
    #   rows are field components; columns are iterations; pages are receivers
    rxs = np.zeros(
        (len(Rx.allowableoutputs_dev), G.iterations, len(G.rxs)),
        dtype=config.sim_config.dtypes["float_or_double"],
    )

    # Copy arrays to compute device
    if config.sim_config.general["solver"] == "cuda":
        import pycuda.gpuarray as gpuarray

        rxcoords_dev = gpuarray.to_gpu(rxcoords)
        rxs_dev = gpuarray.to_gpu(rxs)

    elif config.sim_config.general["solver"] == "opencl":
        import pyopencl.array as clarray

        rxcoords_dev = clarray.to_device(queue, rxcoords)
        rxs_dev = clarray.to_device(queue, rxs)

    return rxcoords_dev, rxs_dev


def dtoh_rx_array(rxs_dev, rxcoords_dev, G):
    """Copy output from receivers array used on compute device back to receiver
        objects.

    Args:
        rxs_dev: float array of receiver data on compute device - rows are field
                    components; columns are iterations; pages are receivers.
        rxcoords_dev: int array of receiver coordinates on compute device.
        G: FDTDGrid class describing a grid in a model.

    """

    for rx in G.rxs:
        for rxd in range(len(G.rxs)):
            if (
                rx.xcoord == rxcoords_dev[rxd, 0]
                and rx.ycoord == rxcoords_dev[rxd, 1]
                and rx.zcoord == rxcoords_dev[rxd, 2]
            ):
                for output in rx.outputs.keys():
                    rx.outputs[output] = rxs_dev[Rx.allowableoutputs_dev.index(output), :, rxd]
