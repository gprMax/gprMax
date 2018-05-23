import argparse
import os

import h5py

# Parse command line arguments
parser = argparse.ArgumentParser(description='Writes a HDF5 file of AustinMan or AustinWoman head only.', usage='python head_only_hdf5 filename')
parser.add_argument('filename', help='name and path to (HDF5) file containing AustinMan or AustinWoman model')
args = parser.parse_args()

# Read full body HDF5 file
f = h5py.File(args.filename, 'r')
dx_dy_dz = f.attrs['dx_dy_dz']
data = f['/data'][:, :, :]

# Define head as last 1/8 of total body height
nzhead = 7 * int(data.shape[2] / 8)

print('Dimensions of head model: {:g} x {:g} x {:g} cells'.format(data.shape[0], data.shape[1], data.shape[2] - nzhead))

# Write HDF5 file
headfile = os.path.splitext(args.filename)[0] + '_head.h5'
f = h5py.File(headfile, 'w')
f.attrs['dx_dy_dz'] = dx_dy_dz
f['/data'] = data[:, :, nzhead:data.shape[2]]
