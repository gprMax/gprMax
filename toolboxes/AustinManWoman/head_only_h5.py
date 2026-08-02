import argparse
import logging
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)


def extract_head(filename, outputfile=None):
    """Extract the upper eighth of an AustinMan/AustinWoman geometry file."""
    filename = Path(filename)
    outputfile = (
        Path(outputfile)
        if outputfile is not None
        else filename.with_name(f"{filename.stem}_head.h5")
    )

    with h5py.File(filename, "r") as source:
        if "data" not in source or "dx_dy_dz" not in source.attrs:
            raise ValueError(f"{filename} is not a gprMax geometry file with data and dx_dy_dz")
        data = source["data"]
        if data.ndim != 3:
            raise ValueError(f"Expected three-dimensional geometry data in {filename}")

        first_head_plane = 7 * (data.shape[2] // 8)
        logger.info(
            "Dimensions of head model: %d x %d x %d cells",
            data.shape[0],
            data.shape[1],
            data.shape[2] - first_head_plane,
        )

        with h5py.File(outputfile, "w") as destination:
            destination.attrs["dx_dy_dz"] = source.attrs["dx_dy_dz"]
            destination.create_dataset(
                "data",
                data=data[:, :, first_head_plane:],
                dtype=data.dtype,
            )

    return outputfile


def main():
    parser = argparse.ArgumentParser(
        description="Writes a HDF5 file of an AustinMan or AustinWoman head.",
        usage="python -m toolboxes.AustinManWoman.head_only_h5 filename",
    )
    parser.add_argument(
        "filename",
        help="HDF5 file containing the full AustinMan or AustinWoman model",
    )
    parser.add_argument("-o", "--output", help="optional output HDF5 filename")
    args = parser.parse_args()

    outputfile = extract_head(args.filename, args.output)
    logger.info("Written head geometry file: %s", outputfile)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()
