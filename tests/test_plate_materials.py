"""
Tests for issue #158 - Magnetic material properties & plate objects.

Verifies that #plate only sets electric material properties (rigidE + ID[0-2]),
and does NOT modify magnetic material properties (rigidH or ID[3-5]).
This matches the behaviour of #edge and is the correct fix for issue #158.
"""

import inspect
import unittest

import numpy as np

from gprMax.geometry_primitives_ext import build_face_xy
from gprMax.geometry_primitives_ext import build_face_xz
from gprMax.geometry_primitives_ext import build_face_yz


# Grid dimensions for test arrays – large enough to avoid boundary clamping
NX, NY, NZ = 10, 10, 10

# Default free-space material numeric ID (matches Grid initialisation)
FREE_SPACE_ID = 1

# Non-free-space material IDs used in tests
MAT_ID_A = 5
MAT_ID_B = 7


def _make_arrays():
    """Return freshly initialised (rigidE, rigidH, ID) numpy arrays."""
    rigidE = np.zeros((12, NX, NY, NZ), dtype=np.int8)
    rigidH = np.zeros((6, NX, NY, NZ), dtype=np.int8)
    ID = np.ones((6, NX + 1, NY + 1, NZ + 1), dtype=np.uint32)
    return rigidE, rigidH, ID


class TestBuildFaceYZSignature(unittest.TestCase):
    """build_face_yz must NOT accept a rigidH parameter (issue #158 fix)."""

    def test_no_rigidH_param(self):
        sig = inspect.signature(build_face_yz)
        self.assertNotIn('rigidH', sig.parameters,
                         "build_face_yz must not have a rigidH parameter after issue #158 fix")

    def test_accepts_seven_args(self):
        rigidE, _, ID = _make_arrays()
        # Should NOT raise – correct 7-argument call
        build_face_yz(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, ID)

    def test_rejects_eight_args(self):
        rigidE, rigidH, ID = _make_arrays()
        # Passing 8 args (old API with rigidH) must fail
        with self.assertRaises(TypeError):
            build_face_yz(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, rigidH, ID)


class TestBuildFaceXZSignature(unittest.TestCase):
    """build_face_xz must NOT accept a rigidH parameter."""

    def test_no_rigidH_param(self):
        sig = inspect.signature(build_face_xz)
        self.assertNotIn('rigidH', sig.parameters)

    def test_accepts_seven_args(self):
        rigidE, _, ID = _make_arrays()
        build_face_xz(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, ID)

    def test_rejects_eight_args(self):
        rigidE, rigidH, ID = _make_arrays()
        with self.assertRaises(TypeError):
            build_face_xz(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, rigidH, ID)


class TestBuildFaceXYSignature(unittest.TestCase):
    """build_face_xy must NOT accept a rigidH parameter."""

    def test_no_rigidH_param(self):
        sig = inspect.signature(build_face_xy)
        self.assertNotIn('rigidH', sig.parameters)

    def test_accepts_seven_args(self):
        rigidE, _, ID = _make_arrays()
        build_face_xy(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, ID)

    def test_rejects_eight_args(self):
        rigidE, rigidH, ID = _make_arrays()
        with self.assertRaises(TypeError):
            build_face_xy(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, rigidH, ID)


class TestBuildFaceYZElectricOnly(unittest.TestCase):
    """build_face_yz must only set electric (Ey, Ez) IDs; magnetic IDs untouched."""

    def setUp(self):
        self.rigidE, self.rigidH, self.ID = _make_arrays()
        self.i, self.j, self.k = 4, 4, 4

    def _call(self, numIDy=MAT_ID_A, numIDz=MAT_ID_B):
        build_face_yz(self.i, self.j, self.k, numIDy, numIDz, self.rigidE, self.ID)

    # --- Electric IDs (ID[1]=Ey, ID[2]=Ez) must be set ---
    def test_ey_id_set(self):
        self._call()
        self.assertEqual(self.ID[1, self.i, self.j, self.k], MAT_ID_A)

    def test_ez_id_set(self):
        self._call()
        self.assertEqual(self.ID[2, self.i, self.j, self.k], MAT_ID_B)

    def test_ey_id_adjacent_k_set(self):
        self._call()
        self.assertEqual(self.ID[1, self.i, self.j, self.k + 1], MAT_ID_A)

    def test_ez_id_adjacent_j_set(self):
        self._call()
        self.assertEqual(self.ID[2, self.i, self.j + 1, self.k], MAT_ID_B)

    # --- Magnetic IDs (ID[3]=Hx, ID[4]=Hy, ID[5]=Hz) must stay FREE_SPACE ---
    def test_hx_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[3] == FREE_SPACE_ID),
                        "Hx IDs were modified by build_face_yz (issue #158)")

    def test_hy_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[4] == FREE_SPACE_ID),
                        "Hy IDs were modified by build_face_yz (issue #158)")

    def test_hz_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[5] == FREE_SPACE_ID),
                        "Hz IDs were modified by build_face_yz (issue #158)")

    # --- rigidE must be marked rigid for Ey/Ez ---
    def test_rigidE_modified(self):
        self._call()
        self.assertTrue(np.any(self.rigidE != 0),
                        "build_face_yz did not set any rigidE values")

    # --- rigidH (independent array) must remain untouched ---
    def test_rigidH_not_modified(self):
        # rigidH is a separate array not passed to the function
        rigidH_copy = self.rigidH.copy()
        self._call()
        np.testing.assert_array_equal(self.rigidH, rigidH_copy,
                                      err_msg="build_face_yz must not modify rigidH (issue #158)")

    # --- Isotropic case: numIDy == numIDz ---
    def test_isotropic(self):
        build_face_yz(self.i, self.j, self.k, MAT_ID_A, MAT_ID_A, self.rigidE, self.ID)
        self.assertEqual(self.ID[1, self.i, self.j, self.k], MAT_ID_A)
        self.assertEqual(self.ID[2, self.i, self.j, self.k], MAT_ID_A)
        self.assertTrue(np.all(self.ID[3] == FREE_SPACE_ID))
        self.assertTrue(np.all(self.ID[4] == FREE_SPACE_ID))
        self.assertTrue(np.all(self.ID[5] == FREE_SPACE_ID))


class TestBuildFaceXZElectricOnly(unittest.TestCase):
    """build_face_xz must only set electric (Ex, Ez) IDs; magnetic IDs untouched."""

    def setUp(self):
        self.rigidE, self.rigidH, self.ID = _make_arrays()
        self.i, self.j, self.k = 4, 4, 4

    def _call(self, numIDx=MAT_ID_A, numIDz=MAT_ID_B):
        build_face_xz(self.i, self.j, self.k, numIDx, numIDz, self.rigidE, self.ID)

    def test_ex_id_set(self):
        self._call()
        self.assertEqual(self.ID[0, self.i, self.j, self.k], MAT_ID_A)

    def test_ez_id_set(self):
        self._call()
        self.assertEqual(self.ID[2, self.i, self.j, self.k], MAT_ID_B)

    def test_ex_id_adjacent_k_set(self):
        self._call()
        self.assertEqual(self.ID[0, self.i, self.j, self.k + 1], MAT_ID_A)

    def test_ez_id_adjacent_i_set(self):
        self._call()
        self.assertEqual(self.ID[2, self.i + 1, self.j, self.k], MAT_ID_B)

    def test_hx_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[3] == FREE_SPACE_ID))

    def test_hy_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[4] == FREE_SPACE_ID))

    def test_hz_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[5] == FREE_SPACE_ID))

    def test_rigidE_modified(self):
        self._call()
        self.assertTrue(np.any(self.rigidE != 0))

    def test_rigidH_not_modified(self):
        rigidH_copy = self.rigidH.copy()
        self._call()
        np.testing.assert_array_equal(self.rigidH, rigidH_copy)

    def test_isotropic(self):
        build_face_xz(self.i, self.j, self.k, MAT_ID_A, MAT_ID_A, self.rigidE, self.ID)
        self.assertEqual(self.ID[0, self.i, self.j, self.k], MAT_ID_A)
        self.assertEqual(self.ID[2, self.i, self.j, self.k], MAT_ID_A)
        self.assertTrue(np.all(self.ID[3] == FREE_SPACE_ID))
        self.assertTrue(np.all(self.ID[4] == FREE_SPACE_ID))
        self.assertTrue(np.all(self.ID[5] == FREE_SPACE_ID))


class TestBuildFaceXYElectricOnly(unittest.TestCase):
    """build_face_xy must only set electric (Ex, Ey) IDs; magnetic IDs untouched."""

    def setUp(self):
        self.rigidE, self.rigidH, self.ID = _make_arrays()
        self.i, self.j, self.k = 4, 4, 4

    def _call(self, numIDx=MAT_ID_A, numIDy=MAT_ID_B):
        build_face_xy(self.i, self.j, self.k, numIDx, numIDy, self.rigidE, self.ID)

    def test_ex_id_set(self):
        self._call()
        self.assertEqual(self.ID[0, self.i, self.j, self.k], MAT_ID_A)

    def test_ey_id_set(self):
        self._call()
        self.assertEqual(self.ID[1, self.i, self.j, self.k], MAT_ID_B)

    def test_ex_id_adjacent_j_set(self):
        self._call()
        self.assertEqual(self.ID[0, self.i, self.j + 1, self.k], MAT_ID_A)

    def test_ey_id_adjacent_i_set(self):
        self._call()
        self.assertEqual(self.ID[1, self.i + 1, self.j, self.k], MAT_ID_B)

    def test_hx_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[3] == FREE_SPACE_ID))

    def test_hy_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[4] == FREE_SPACE_ID))

    def test_hz_id_unchanged(self):
        self._call()
        self.assertTrue(np.all(self.ID[5] == FREE_SPACE_ID))

    def test_rigidE_modified(self):
        self._call()
        self.assertTrue(np.any(self.rigidE != 0))

    def test_rigidH_not_modified(self):
        rigidH_copy = self.rigidH.copy()
        self._call()
        np.testing.assert_array_equal(self.rigidH, rigidH_copy)

    def test_isotropic(self):
        build_face_xy(self.i, self.j, self.k, MAT_ID_A, MAT_ID_A, self.rigidE, self.ID)
        self.assertEqual(self.ID[0, self.i, self.j, self.k], MAT_ID_A)
        self.assertEqual(self.ID[1, self.i, self.j, self.k], MAT_ID_A)
        self.assertTrue(np.all(self.ID[3] == FREE_SPACE_ID))
        self.assertTrue(np.all(self.ID[4] == FREE_SPACE_ID))
        self.assertTrue(np.all(self.ID[5] == FREE_SPACE_ID))


class TestBuildFaceConsistency(unittest.TestCase):
    """Cross-orientation consistency: each build_face_* variant only touches its own components."""

    def test_yz_does_not_set_ex_id(self):
        rigidE, _, ID = _make_arrays()
        build_face_yz(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, ID)
        # Ex (ID[0]) must remain free_space everywhere
        self.assertTrue(np.all(ID[0] == FREE_SPACE_ID))

    def test_xz_does_not_set_ey_id(self):
        rigidE, _, ID = _make_arrays()
        build_face_xz(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, ID)
        # Ey (ID[1]) must remain free_space everywhere
        self.assertTrue(np.all(ID[1] == FREE_SPACE_ID))

    def test_xy_does_not_set_ez_id(self):
        rigidE, _, ID = _make_arrays()
        build_face_xy(3, 3, 3, MAT_ID_A, MAT_ID_B, rigidE, ID)
        # Ez (ID[2]) must remain free_space everywhere
        self.assertTrue(np.all(ID[2] == FREE_SPACE_ID))

    def test_multiple_cells_yz(self):
        """Calling build_face_yz for a 2x2 patch sets all cells correctly."""
        rigidE, _, ID = _make_arrays()
        for j in range(3, 5):
            for k in range(3, 5):
                build_face_yz(2, j, k, MAT_ID_A, MAT_ID_B, rigidE, ID)
        for j in range(3, 5):
            for k in range(3, 5):
                self.assertEqual(ID[1, 2, j, k], MAT_ID_A)
                self.assertEqual(ID[2, 2, j, k], MAT_ID_B)
        self.assertTrue(np.all(ID[3] == FREE_SPACE_ID))
        self.assertTrue(np.all(ID[4] == FREE_SPACE_ID))
        self.assertTrue(np.all(ID[5] == FREE_SPACE_ID))


if __name__ == '__main__':
    unittest.main()
