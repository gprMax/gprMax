import argparse
import glob
import logging
import os

import h5py

from .convert import convert_files

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", level=logging.INFO)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Allows the user to convert a STL files to voxelized mesh.",
        usage="cd gprMax; python -m toolboxes.STLtoVoxel.stltovoxel stlfilename -matindex -dxdydz",
    )
    parser.add_argument("stlfiles", help="can be the filename of a single STL file, or the path to folder containing multiple STL files")
    parser.add_argument(
        "-dxdydz", type=float, required=True, help="discretisation to use in voxelisation process"
    )
    args = parser.parse_args()

    if os.path.isdir(args.stlfiles):
        path = args.stlfiles
        files = sorted(glob.glob(path + "/*.stl"))
        filename_hdf5 = os.path.join(path, os.path.basename(path) + "_geo.h5")
        filename_mats = os.path.join(path, os.path.basename(path) + "_mats.txt")
    elif os.path.isfile(args.stlfiles):
        path = os.path.dirname(args.stlfiles)
        files = args.stlfiles
        filename_hdf5 = os.path.join(path, os.path.split(os.path.basename(path))[0] + "_geo.h5")
        filename_mats = os.path.join(path, os.path.split(os.path.basename(path))[0] + "_mats.txt")

    dxdydz = (args.dxdydz, args.dxdydz, args.dxdydz)

    newline = "\n\t"
    logger.info(f"\nConverting STL file(s): {newline.join(files)}")
    model_array = convert_files(files, dxdydz)
    logger.info(f"Number of voxels: {model_array.shape[0]} x {model_array.shape[1]} x {model_array.shape[2]}")
    logger.info(f"Spatial discretisation: {dxdydz[0]} x {dxdydz[1]} x {dxdydz[2]}m")

    # Write HDF5 file for gprMax using voxels
    with h5py.File(filename_hdf5, "w") as f:
        f.create_dataset("data", data=model_array)
        f.attrs["dx_dy_dz"] = (dxdydz[0], dxdydz[1], dxdydz[2])
    logger.info(f"Written geometry object file: {filename_hdf5}")

    # Write materials file for gprMax
    # with open(filename_mats, 'w') as f:
    #     for i, file in enumerate(files):
    #         name = os.path.splitext(os.path.basename(file))[0]
    #         f.write(f"#material: {i + 1} 0 1 0 {name}" + "\n")
    # logger.info(f"Written materials file: {filename_mats}")