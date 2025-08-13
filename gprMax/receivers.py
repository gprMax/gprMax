# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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
import numpy as np

import gprMax.config as config

logger = logging.getLogger(__name__)


class Rx:
    """Receiver output points."""

    allowableoutputs = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"]
    defaultoutputs = allowableoutputs[:-3]
    allowableoutputs_dev = allowableoutputs[:-3]

    def __init__(self):
        self.ID: str
        self.outputs = {}
        self.coord = np.zeros(3, dtype=np.int32)
        self.coordorigin = np.zeros(3, dtype=np.int32)

    @property
    def xcoord(self) -> int:
        return self.coord[0]

    @xcoord.setter
    def xcoord(self, value: int):
        self.coord[0] = value

    @property
    def ycoord(self) -> int:
        return self.coord[1]

    @ycoord.setter
    def ycoord(self, value: int):
        self.coord[1] = value

    @property
    def zcoord(self) -> int:
        return self.coord[2]

    @zcoord.setter
    def zcoord(self, value: int):
        self.coord[2] = value

    @property
    def xcoordorigin(self) -> int:
        return self.coordorigin[0]

    @xcoordorigin.setter
    def xcoordorigin(self, value: int):
        self.coordorigin[0] = value

    @property
    def ycoordorigin(self) -> int:
        return self.coordorigin[1]

    @ycoordorigin.setter
    def ycoordorigin(self, value: int):
        self.coordorigin[1] = value

    @property
    def zcoordorigin(self) -> int:
        return self.coordorigin[2]

    @zcoordorigin.setter
    def zcoordorigin(self, value: int):
        self.coordorigin[2] = value


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

    elif config.sim_config.general["solver"] == "metal":
        # Create Metal buffers for receiver coordinates and field components
        rxcoords_dev = dev.newBufferWithBytes_length_options_(
            rxcoords, rxcoords.nbytes, 0
        )  # 0 for default options
        rxs_dev = dev.newBufferWithBytes_length_options_(
            rxs, rxs.nbytes, 0
        )  # 0 for default options

    return rxcoords_dev, rxs_dev


def dtoh_rx_array(rxs_dev, rxcoords_dev, G):
    """Copy output from receivers array used on compute device back to receiver objects.

    Args:
        rxs_dev: MTLBuffer for receiver data on compute device.
        rxcoords_dev: MTLBuffer for receiver coordinates on compute device.
        G: FDTDGrid class describing a grid in a model.
    """

    if config.sim_config.general["solver"] == "metal":
        # For Metal, we need to read the data back from the GPU buffers
        # Create NumPy arrays to hold the data
        rxcoords_shape = (len(G.rxs), 3)
        rxs_shape = (len(Rx.allowableoutputs_dev), G.iterations, len(G.rxs))

        # Initialize arrays
        rxcoords_np = np.zeros(rxcoords_shape, dtype=np.int32)
        rxs_np = np.zeros(rxs_shape, dtype=config.sim_config.dtypes["float_or_double"])

        # Copy receiver coordinates from the model (these should be correct)
        for i, rx in enumerate(G.rxs):
            rxcoords_np[i] = [rx.xcoord, rx.ycoord, rx.zcoord]

        # Copy data from Metal GPU buffers to numpy arrays
        try:
            # Copy receiver field data from Metal buffer
            # The Metal buffer contains data in the same format as the numpy array
            # rxs_shape: (6 field components, iterations, num_receivers)
            expected_rxs_bytes = rxs_np.nbytes

            if rxs_dev.length() == expected_rxs_bytes:
                # For Metal buffers, use the correct API to read data
                buffer_size = rxs_dev.length()
                rxs_buffer = rxs_dev.contents().as_buffer(buffer_size)
                rxs_np = (
                    np.frombuffer(
                        rxs_buffer, dtype=config.sim_config.dtypes["float_or_double"]
                    )
                    .reshape(rxs_shape)
                    .copy()
                )

                logger.debug(
                    f"Successfully copied {rxs_np.size} elements from Metal GPU buffer to CPU"
                )
            else:
                logger.debug(
                    f"Metal rxs buffer size mismatch: expected {expected_rxs_bytes}, got {rxs_dev.length()}"
                )

        except Exception as e:
            logger.exception(f"Error copying Metal buffer data: {e}, using zero-filled arrays as fallback")

        # Use the numpy arrays
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
                    rx.outputs[output] = rxs_dev[
                        Rx.allowableoutputs_dev.index(output), :, rxd
                    ]
