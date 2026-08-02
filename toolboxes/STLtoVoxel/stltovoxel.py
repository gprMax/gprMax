import argparse
import logging
from pathlib import Path

import h5py

from .convert import convert_files

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Allows the user to convert a STL files to voxelized mesh.",
        usage="python -m toolboxes.STLtoVoxel.stltovoxel stlfilename -dxdydz VALUE",
    )
    parser.add_argument(
        "stlfiles",
        help="can be the filename of a single STL file, or the path to folder containing multiple STL files",
    )
    parser.add_argument(
        "-dxdydz",
        type=float,
        required=True,
        help="discretisation to use in voxelisation process",
    )
    args = parser.parse_args()

    input_path = Path(args.stlfiles)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.stl"))
        filename_hdf5 = input_path / f"{input_path.name}_geo.h5"
    elif input_path.is_file():
        files = [input_path]
        filename_hdf5 = input_path.with_name(f"{input_path.stem}_geo.h5")
    else:
        parser.error(f"STL file or directory does not exist: {input_path}")

    if not files:
        parser.error(f"No STL files found in: {input_path}")

    dxdydz = (args.dxdydz, args.dxdydz, args.dxdydz)

    newline = "\n\t"
    logger.info(f"\nConverting STL file(s): {newline.join(map(str, files))}")
    model_array = convert_files(files, dxdydz)
    logger.info(
        f"Number of voxels: {model_array.shape[0]} x {model_array.shape[1]} x {model_array.shape[2]}"
    )
    logger.info(f"Spatial discretisation: {dxdydz[0]} x {dxdydz[1]} x {dxdydz[2]}m")

    # Write HDF5 file for gprMax using voxels
    with h5py.File(filename_hdf5, "w") as f:
        f.create_dataset("data", data=model_array)
        f.attrs["dx_dy_dz"] = (dxdydz[0], dxdydz[1], dxdydz[2])
    logger.info(f"Written geometry object file: {filename_hdf5}")


if __name__ == "__main__":
    main()
