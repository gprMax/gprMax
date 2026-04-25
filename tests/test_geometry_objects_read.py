import os
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np

from gprMax.exceptions import CmdInputError
from gprMax.input_cmds_geometry import process_geometrycmds
from gprMax.materials import Material


def builtin_materials():
    pec = Material(0, 'pec')
    pec.se = float('inf')
    pec.type = 'builtin'
    pec.averagable = False
    free_space = Material(1, 'free_space')
    free_space.type = 'builtin'
    return [pec, free_space]


class GeometryObjectsReadTest(unittest.TestCase):

    def test_generated_geometry_reports_missing_material_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            geofile = os.path.join(tmpdir, 'geometry.h5')
            matfile = os.path.join(tmpdir, 'materials.txt')

            with h5py.File(geofile, 'w') as f:
                f.attrs['dx_dy_dz'] = (1.0, 1.0, 1.0)
                f['/data'] = np.zeros((1, 1, 1), dtype=np.int16)
                f['/rigidE'] = np.zeros((6, 1, 1, 1), dtype=np.int8)
                f['/rigidH'] = np.zeros((6, 1, 1, 1), dtype=np.int8)
                f['/ID'] = np.ones((6, 1, 1, 1), dtype=np.uint32)

            with open(matfile, 'w') as f:
                f.write('#material: 1 inf 1 0 pec\n')

            grid = SimpleNamespace(
                dx=1.0,
                dy=1.0,
                dz=1.0,
                inputdirectory=tmpdir,
                materials=builtin_materials(),
                progressbars=True,
                messages=False,
                solid=np.zeros((1, 1, 1), dtype=np.int16),
                rigidE=np.zeros((6, 1, 1, 1), dtype=np.int8),
                rigidH=np.zeros((6, 1, 1, 1), dtype=np.int8),
                ID=np.zeros((6, 1, 1, 1), dtype=np.uint32),
            )

            command = '#geometry_objects_read: 0 0 0 geometry.h5 materials.txt'

            with self.assertRaisesRegex(CmdInputError, 'materials file'):
                process_geometrycmds([command], grid)


if __name__ == '__main__':
    unittest.main()
