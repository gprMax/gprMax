import argparse
import logging
from pathlib import Path

import h5py

from .convert import convert_file

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s', level=logging.INFO)

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Allows the user to convert a STL files to voxelized mesh.', usage='cd gprMax; python -m user_libs.stltovoxel.stltovoxel stlfilename matindex dx_dy_dz')
    parser.add_argument('stlfilename', help='name of STL file to convert including path')
    parser.add_argument('-matindex', type=int, required=True,
                        help='index of material to extract from STL file')
    parser.add_argument('-dxdydz', nargs='+', type=float, required=True,
                        help='discretisation to use in voxelisation process')
    args = parser.parse_args()

    filename_stl = Path(args.stlfilename)
    dxdydz = tuple(args.dxdydz)

    logger.info(f'\nConverting STL file: {filename_stl.name}')
    model_array = convert_file(filename_stl, dxdydz)
    model_array[model_array==0] = -1
    model_array[model_array==1] = args.matindex
    logger.info(f'Number of voxels: {model_array.shape[0]} x {model_array.shape[1]} x {model_array.shape[2]}')
    logger.info(f'Spatial discretisation: {dxdydz[0]} x {dxdydz[1]} x {dxdydz[2]}m')
    
    # Write HDF5 file for gprMax using voxels
    filename_hdf5 = filename_stl.with_suffix('.h5')
    with h5py.File(filename_hdf5, 'w') as f:
        f.create_dataset('data', data=model_array)
        f.attrs['dx_dy_dz'] = (dxdydz[0], dxdydz[1], dxdydz[2])

    logger.info(f'Written geometry object file: {filename_hdf5.name}')