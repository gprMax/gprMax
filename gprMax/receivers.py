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
        self.xcoord = None
        self.ycoord = None
        self.zcoord = None
        self.xcoordorigin = None
        self.ycoordorigin = None
        self.zcoordorigin = None

def htod_rx_arrays(G, queue=None, dev=None):
    """Initialise arrays on compute device for receiver coordinates and to store field
        components for receivers.

    Args:
        G: FDTDGrid class describing a grid in a model.
        queue: pyopencl queue.
        dev: Apple Metal device object.

    Returns:
        rxcoords_dev: MTLBuffer for receiver coordinates on compute device.
        rxs_dev: MTLBuffer for receiver data on compute device.
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
        (len(Rx.allowableoutputs_dev), G.iterations, len(G.rxs)), dtype=config.sim_config.dtypes["float_or_double"]
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

    elif config.sim_config.general["solver"] == "metal":
        # Create Metal buffers for receiver coordinates and field components
        rxcoords_dev = dev.newBufferWithBytes_length_options_(rxcoords, 
                                                              rxcoords.nbytes, 
                                                              0)  # 0 for default options
        rxs_dev = dev.newBufferWithBytes_length_options_(rxs, 
                                                         rxs.nbytes, 
                                                         0)  # 0 for default options

    return rxcoords_dev, rxs_dev

def dtoh_rx_array(rxs_dev, rxcoords_dev, G, dev):
    """Copy output from receivers array used on compute device back to receiver objects.

    Args:
        rxs_dev: MTLBuffer for receiver data on compute device - rows are field components; columns are iterations; pages are receivers.
        rxcoords_dev: MTLBuffer for receiver coordinates on compute device.
        G: FDTDGrid class describing a grid in a model.
        dev: Apple Metal device object.
    """

    if config.sim_config.general["solver"] == "metal":
        # Create NumPy arrays to hold the data
        rxcoords_np = np.empty_like(np.zeros((len(G.rxs), 3), dtype=np.int32))
        rxs_np = np.empty_like(np.zeros(
            (len(Rx.allowableoutputs_dev), G.iterations, len(G.rxs)), dtype=config.sim_config.dtypes["float_or_double"]
        ))

        # Create a Metal command queue
        command_queue = dev.newCommandQueue()

        # Create a command buffer
        command_buffer = command_queue.commandBuffer()

        # Create a blit encoder
        blit_encoder = command_buffer.blitCommandEncoder()

        # Copy data from Metal buffers to NumPy arrays
        blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(rxcoords_dev, 
                                                                                  0, 
                                                                                  rxcoords_np.data, 
                                                                                  0, 
                                                                                  rxcoords_np.nbytes)
        blit_encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(rxs_dev, 
                                                                                  0, 
                                                                                  rxs_np.data, 
                                                                                  0, 
                                                                                  rxs_np.nbytes)
        blit_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Release Metal resources 
        blit_encoder.release()
        command_buffer.release()
        command_queue.release()

        # Use the data in NumPy arrays
        rxcoords_dev = rxcoords_np
        rxs_dev = rxs_np
    else:
        rxs_dev = rxs_dev.get()
        rxcoords_dev = rxcoords_dev.get()

    for rx in G.rxs:
        for rxd in range(len(G.rxs)):
            if (
                rx.xcoord == rxcoords_dev[rxd, 0]
                and rx.ycoord == rxcoords_dev[rxd, 1]
                and rx.zcoord == rxcoords_dev[rxd, 2]
            ):
                for output in rx.outputs.keys():
                    rx.outputs[output] = rxs_dev[Rx.allowableoutputs_dev.index(output), :, rxd]
