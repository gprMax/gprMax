# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Convert discrete colours in a PNG image to reusable gprMax geometry.

The generated voxel-only HDF5 file uses the current geometry-object schema:
compact integer material indices in ``/data``, stable keys in
``/material_keys``, and the companion database identity in root attributes.
Constitutive properties are written as null values in an adjacent JSON
material-database template because they cannot be inferred from image colour.
The user must complete those values before importing the geometry.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from gprMax.material_database import make_database_id
from toolboxes.GeometryImport.common import (
    write_geometry_hdf5,
    write_null_material_database,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PNGGeometryResult:
    """Files and mappings produced by :func:`convert_png`."""

    geometry_file: Path
    material_database_file: Path
    material_database_id: str
    material_keys: tuple[str, ...]
    material_names: tuple[str, ...]
    selected_colours: tuple[tuple[int, ...], ...]
    shape: tuple[int, int, int]


class Cursor:
    """Collect unique RGB(A) values selected in an image."""

    def __init__(self, im, materials):
        """
        Args:
            im (ndarray): Pixels of the image.
            materials (list): Store for selected RGB(A) pixel values.
        """

        self.im = im
        self.materials = materials
        plt.connect("button_press_event", self)

    def __call__(self, event):
        """Record a material colour from a single mouse click."""

        if not event.dblclick:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                pixel = _to_uint8(np.asarray(self.im[int(y), int(x)]))
                if not pixel_match(self.materials, pixel):
                    logger.info(
                        "x, y: %d %d px; colour: %s; material index: %d",
                        int(x),
                        int(y),
                        tuple(int(value) for value in pixel),
                        len(self.materials),
                    )
                    self.materials.append(pixel)


def _to_uint8(values: np.ndarray) -> np.ndarray:
    """Normalise image channels to the 8-bit values shown to the user."""

    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("PNG image contains NaN or infinite channel values")
        array = np.floor(np.clip(array, 0, 1) * 255)
    elif np.issubdtype(array.dtype, np.integer):
        if array.size and (array.min() < 0 or array.max() > 255):
            raise ValueError("PNG integer channels must be in the range 0 to 255")
    else:
        raise ValueError("PNG image channels must be numeric")
    return array.astype(np.uint8)


def pixel_match(pixellist, pixeltest):
    """Return whether an RGB(A) pixel already occurs in a selected list."""

    return any(np.array_equal(pixel, pixeltest) for pixel in pixellist)


def _validate_image(image: np.ndarray) -> np.ndarray:
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("image must contain RGB or RGBA colour channels")
    return _to_uint8(pixels)


def _validate_colours(
    materials: Sequence[Sequence[int] | np.ndarray],
    channels: int,
) -> tuple[tuple[int, ...], ...]:
    if not materials:
        raise ValueError("at least one material colour must be selected")
    colours = []
    for material in materials:
        colour = _to_uint8(np.asarray(material))
        if colour.shape != (channels,):
            raise ValueError(f"selected colours must contain {channels} channels to match the PNG image")
        value = tuple(int(item) for item in colour)
        if value in colours:
            raise ValueError(f"material colour {value} was selected more than once")
        colours.append(value)
    return tuple(colours)


def _colour_name(colour: tuple[int, ...]) -> str:
    prefix = "rgb" if len(colour) == 3 else "rgba"
    return prefix + "_" + "_".join(str(value) for value in colour)


def convert_png(
    imagefile: str | os.PathLike[str],
    spacing: Sequence[float],
    materials: Sequence[Sequence[int] | np.ndarray],
    *,
    zcells: int = 1,
    output_file: str | os.PathLike[str] | None = None,
) -> PNGGeometryResult:
    """Convert selected PNG colours to a modern HDF5/JSON geometry pair.

    Unselected pixels are stored as ``-1`` and therefore leave the existing
    model material unchanged when imported. Image rows are mapped to gprMax
    x/y cells using the historical clockwise rotation of this utility.
    """

    source = Path(imagefile)
    if not source.is_file():
        raise FileNotFoundError(f"PNG image file does not exist: {source}")
    dxyz = tuple(float(value) for value in spacing)
    if len(dxyz) != 3 or not np.isfinite(dxyz).all() or any(value <= 0 for value in dxyz):
        raise ValueError("spacing must contain three positive finite values in metres")
    if not isinstance(zcells, int) or isinstance(zcells, bool) or zcells < 1:
        raise ValueError("zcells must be a positive integer")

    image = _validate_image(mpimg.imread(source))
    colours = _validate_colours(materials, image.shape[2])
    # Image arrays are row-major (y, x); gprMax geometry is indexed (x, y).
    image_xy = np.rot90(image, k=3)
    data = np.full((image_xy.shape[0], image_xy.shape[1], zcells), -1, dtype=np.int16)
    for material_index, colour in enumerate(colours):
        mask = np.all(image_xy == np.asarray(colour, dtype=np.uint8), axis=-1)
        data[mask, :] = material_index

    geometry_file = Path(output_file) if output_file is not None else source.with_suffix(".h5")
    if geometry_file.suffix.lower() != ".h5":
        geometry_file = geometry_file.with_suffix(".h5")
    geometry_file.parent.mkdir(parents=True, exist_ok=True)
    database_id = make_database_id(f"{geometry_file.stem}_materials", prefix="geometry")
    database_file = geometry_file.with_name(f"{database_id}.json")
    material_names = tuple(_colour_name(colour) for colour in colours)
    metadata = tuple(
        {
            "source_png": source.name,
            "selected_colour": list(colour),
            "colour_space": "RGB" if len(colour) == 3 else "RGBA",
        }
        for colour in colours
    )

    # Validate/preserve an existing editable database before replacing the
    # reproducible HDF5 array. This prevents a changed colour selection from
    # silently detaching an already edited database from its geometry.
    material_keys = write_null_material_database(
        database_file,
        database_id,
        material_names,
        source=f"PNG image {source.name}",
        metadata=metadata,
    )
    write_geometry_hdf5(
        geometry_file,
        data,
        dxyz,
        material_keys=material_keys,
        material_database=database_id,
    )

    logger.info("Written geometry object file: %s", geometry_file)
    logger.info("Written/preserved editable material database: %s", database_file)
    logger.info(
        "Complete the null constitutive values in %s before importing this geometry.",
        database_file,
    )
    return PNGGeometryResult(
        geometry_file=geometry_file,
        material_database_file=database_file,
        material_database_id=database_id,
        material_keys=material_keys,
        material_names=material_names,
        selected_colours=colours,
        shape=tuple(int(value) for value in data.shape),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=(
            "Convert selected colours in a PNG image to a current gprMax "
            "geometry HDF5 file and editable JSON material database."
        ),
        usage="python -m toolboxes.Utilities.convert_png2h5 imagefile dx dy dz [options]",
    )
    parser.add_argument("imagefile", help="PNG filename including path")
    parser.add_argument(
        "dxdydz",
        type=float,
        nargs=3,
        metavar=("dx", "dy", "dz"),
        help="spatial resolution in metres",
    )
    parser.add_argument(
        "-zcells",
        "--zcells",
        default=1,
        type=int,
        help="number of cells in the z (invariant) direction (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=None,
        help="geometry HDF5 destination (default: image name with .h5)",
    )
    args = parser.parse_args(argv)
    if args.zcells < 1:
        parser.error("zcells must be greater than zero")
    if any(value <= 0 for value in args.dxdydz):
        parser.error("dx, dy, and dz must be greater than zero")

    try:
        image = _validate_image(mpimg.imread(args.imagefile))
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    logger.info("Reading PNG image file: %s", Path(args.imagefile).name)
    logger.info(
        " 1. Select discrete material colours with a single click.\n"
        " 2. Close the image after selecting every required material."
    )
    materials: list[np.ndarray] = []
    figure = plt.figure(num=Path(args.imagefile).name, facecolor="w", edgecolor="w")
    displayed = np.flipud(image)
    plt.imshow(displayed, interpolation="nearest", aspect="equal", origin="lower")
    Cursor(displayed, materials)
    plt.show()
    plt.close(figure)

    try:
        result = convert_png(
            args.imagefile,
            args.dxdydz,
            materials,
            zcells=args.zcells,
            output_file=args.output_file,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    logger.info(
        "Import with material_database=%r after editing the JSON properties.",
        result.material_database_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
