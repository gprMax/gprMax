# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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

import argparse
import logging
import os

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Cursor(object):
    """Get RGB(A) value of pixel at x,y coordinate of button press and store in a list."""

    def __init__(self, im, materials):
        """
        Args:
            im (ndarray): Pixels of the image.
            materials (list): To store selected RGB(A) values of selected pixels.
        """
        self.im = im
        self.materials = materials
        plt.connect("button_press_event", self)

    def __call__(self, event):
        """
        Args:
            event (MouseEvent): matplotlib mouse event.
        """
        if not event.dblclick:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                pixel = self.im[int(y), int(x)]
                pixel = np.floor(pixel * 255).astype(
                    np.int16
                )  # Convert pixel values from float (0-1) to integer (0-255)
                match = pixel_match(materials, pixel)
                if match is False:
                    logger.info(
                        f"x, y: {int(x)} {int(y)} px; RGB: {pixel[:-1]}; material ID: {len(self.materials)}"
                    )
                    materials.append(pixel)


def pixel_match(pixellist, pixeltest):
    """Checks if the RGB(A) value of a pixel already exists in a list of pixel values.

    Args:
        pixellist (list): List of numpy arrays of pixels to test against.
        pixeltest (ndarray): RGB(A) value of test pixel.

    Returns:
        match (boolean): True if pixel is matched in pixel list or False if not found.
    """
    match = False
    for pixel in pixellist:
        if np.all(pixel == pixeltest):
            match = True
            break
    return match


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert a PNG image to a HDF5 file that can be used to import geometry (#geometry_objects_read) into a 2D gprMax model. Colours from the image are selected which correspond to a list of materials that should be supplied in a separate text file.",
        usage="python convert_png2h5.py imagefile dx dy dz",
    )
    parser.add_argument("imagefile", help="name of image file including path")
    parser.add_argument(
        "dxdydz",
        type=float,
        action="append",
        nargs=3,
        help="spatial resolution of model, e.g. dx dy dz",
    )
    parser.add_argument(
        "-zcells",
        default=1,
        type=int,
        help="number of cells for domain in z-direction (infinite direction)",
    )
    args = parser.parse_args()

    # Open image file
    im = mpimg.imread(args.imagefile)

    # Store image data to use for creating geometry
    imdata = np.rot90(im, k=3)  # Rotate 90CW
    imdata = np.floor(imdata * 255).astype(
        np.int16
    )  # Convert pixel values from float (0-1) to integer (0-255)

    logger.info(f"Reading PNG image file: {os.path.split(args.imagefile)[1]}")
    logger.info(
        " 1. Select discrete material colours by clicking on parts of the image.\n 2. When all materials have been selected close the image."
    )

    # List to hold selected RGB values from image
    materials = []

    # Plot image and record rgb values from mouse clicks
    fig = plt.figure(num=os.path.split(args.imagefile)[1], facecolor="w", edgecolor="w")
    im = np.flipud(im)  # Flip image for viewing with origin in lower left
    plt.imshow(im, interpolation="nearest", aspect="equal", origin="lower")
    Cursor(im, materials)
    plt.show()

    # Format spatial resolution into tuple
    dx_dy_dz = (args.dxdydz[0][0], args.dxdydz[0][1], args.dxdydz[0][2])

    # Filename for geometry (HDF5) file
    hdf5file = f"{os.path.splitext(args.imagefile)[0]}.h5"

    # Array to store geometry data (initialised as background, i.e. -1)
    data = np.ones((imdata.shape[0], imdata.shape[1], args.zcells), dtype=np.int16) * -1

    # Write geometry (HDF5) file
    with h5py.File(hdf5file, "w") as fout:
        # Add attribute with name 'dx_dy_dz' for spatial resolution
        fout.attrs["dx_dy_dz"] = dx_dy_dz

        # Use a boolean mask to match selected pixel values with position in image
        for i, material in enumerate(materials):
            mask = np.all(imdata == material, axis=-1)
            data[mask, :] = i

        # Write data to file
        fout.create_dataset("data", data=data)

    logger.info(f"Written HDF5 file: {os.path.split(hdf5file)[1]}")
