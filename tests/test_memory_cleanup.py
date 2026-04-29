# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

"""Tests for memory cleanup between model runs.

These tests verify that class-level state is reset and large arrays are released
when running multiple scenes to prevent memory accumulation.

Usage:
    cd gprMax
    python -m pytest tests/test_memory_cleanup.py -v
"""

import numpy as np
import pytest

from gprMax.constants import floattype
from gprMax.grid import FDTDGrid
from gprMax.materials import Material
from gprMax.snapshots import Snapshot


class TestFDTDGridCleanup:
    """Tests for FDTDGrid.cleanup() method."""

    def _create_grid_with_arrays(self):
        """Helper to create a minimal FDTDGrid with allocated arrays."""
        G = FDTDGrid.__new__(FDTDGrid)
        # Minimal grid dimensions
        G.nx = 10
        G.ny = 10
        G.nz = 10

        # Allocate field arrays (the primary memory consumers)
        G.Ex = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=floattype)
        G.Ey = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=floattype)
        G.Ez = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=floattype)
        G.Hx = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=floattype)
        G.Hy = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=floattype)
        G.Hz = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=floattype)

        # Geometry arrays
        G.solid = np.zeros((G.nx + 1, G.ny + 1, G.nz + 1), dtype=np.uint32)
        G.rigidE = np.zeros((18, G.nx + 1, G.ny + 1, G.nz + 1), dtype=np.int8)
        G.rigidH = np.zeros((18, G.nx + 1, G.ny + 1, G.nz + 1), dtype=np.int8)
        G.ID = np.zeros((6, G.nx + 1, G.ny + 1, G.nz + 1), dtype=np.uint32)

        # Update coefficients
        G.updatecoeffsE = np.zeros((2, 5), dtype=floattype)
        G.updatecoeffsH = np.zeros((2, 5), dtype=floattype)

        # Object lists
        G.pmls = [object(), object()]  # Simulate PML objects
        G.rxs = [object()]
        G.snapshots = []
        G.materials = [object(), object()]
        G.voltagesources = []
        G.hertziandipoles = [object()]
        G.magneticdipoles = []
        G.transmissionlines = []
        G.waveforms = [object()]
        G.geometryviews = []
        G.geometryobjectswrite = []

        return G

    def test_cleanup_removes_field_arrays(self):
        """Verify field arrays are deleted after cleanup."""
        G = self._create_grid_with_arrays()

        G.cleanup()

        for attr in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
            assert not hasattr(G, attr), f'{attr} still exists after cleanup'

    def test_cleanup_removes_geometry_arrays(self):
        """Verify geometry arrays are deleted after cleanup."""
        G = self._create_grid_with_arrays()

        G.cleanup()

        for attr in ('solid', 'rigidE', 'rigidH', 'ID'):
            assert not hasattr(G, attr), f'{attr} still exists after cleanup'

    def test_cleanup_removes_update_coefficients(self):
        """Verify update coefficient arrays are deleted after cleanup."""
        G = self._create_grid_with_arrays()

        G.cleanup()

        for attr in ('updatecoeffsE', 'updatecoeffsH'):
            assert not hasattr(G, attr), f'{attr} still exists after cleanup'

    def test_cleanup_clears_object_lists(self):
        """Verify object lists are emptied (not removed) after cleanup."""
        G = self._create_grid_with_arrays()

        G.cleanup()

        for listattr in ('pmls', 'rxs', 'snapshots', 'materials',
                         'voltagesources', 'hertziandipoles',
                         'magneticdipoles', 'transmissionlines',
                         'waveforms'):
            lst = getattr(G, listattr, None)
            assert lst is not None, f'{listattr} was deleted instead of cleared'
            assert len(lst) == 0, f'{listattr} is not empty after cleanup'

    def test_cleanup_handles_missing_attrs(self):
        """Verify cleanup doesn't fail if some attributes are missing."""
        G = FDTDGrid.__new__(FDTDGrid)
        G.pmls = []
        G.rxs = []
        G.snapshots = []
        G.materials = []
        G.voltagesources = []
        G.hertziandipoles = []
        G.magneticdipoles = []
        G.transmissionlines = []
        G.waveforms = []
        G.geometryviews = []
        G.geometryobjectswrite = []

        # Should not raise even if field arrays were never allocated
        G.cleanup()

    def test_cleanup_handles_dispersive_arrays(self):
        """Verify dispersive material arrays are cleaned up."""
        G = self._create_grid_with_arrays()
        G.updatecoeffsdispersive = np.zeros((2, 3), dtype=floattype)
        G.Tx = np.zeros((10, 10, 10), dtype=floattype)
        G.Ty = np.zeros((10, 10, 10), dtype=floattype)
        G.Tz = np.zeros((10, 10, 10), dtype=floattype)

        G.cleanup()

        for attr in ('updatecoeffsdispersive', 'Tx', 'Ty', 'Tz'):
            assert not hasattr(G, attr), f'{attr} still exists after cleanup'


class TestMaterialResetClassState:
    """Tests for Material.reset_class_state() classmethod."""

    def test_reset_maxpoles(self):
        """Verify maxpoles is reset to 0."""
        Material.maxpoles = 5
        Material.reset_class_state()
        assert Material.maxpoles == 0

    def test_reset_preserves_constants(self):
        """Verify physical constants are not affected by reset."""
        original_waterer = Material.waterer
        original_grasser = Material.grasser

        Material.maxpoles = 3
        Material.reset_class_state()

        assert Material.waterer == original_waterer
        assert Material.grasser == original_grasser

    def test_reset_idempotent(self):
        """Verify calling reset multiple times has the same effect."""
        Material.maxpoles = 10
        Material.reset_class_state()
        Material.reset_class_state()
        assert Material.maxpoles == 0


class TestSnapshotResetClassState:
    """Tests for Snapshot.reset_class_state() classmethod."""

    def test_reset_dimensions(self):
        """Verify dimension trackers are reset to 0."""
        Snapshot.nx_max = 100
        Snapshot.ny_max = 200
        Snapshot.nz_max = 300

        Snapshot.reset_class_state()

        assert Snapshot.nx_max == 0
        assert Snapshot.ny_max == 0
        assert Snapshot.nz_max == 0

    def test_reset_bpg(self):
        """Verify GPU blocks per grid is reset to None."""
        Snapshot.bpg = (64, 1, 1)

        Snapshot.reset_class_state()

        assert Snapshot.bpg is None

    def test_reset_idempotent(self):
        """Verify calling reset multiple times has the same effect."""
        Snapshot.nx_max = 50
        Snapshot.ny_max = 60
        Snapshot.nz_max = 70
        Snapshot.bpg = (32, 1, 1)

        Snapshot.reset_class_state()
        Snapshot.reset_class_state()

        assert Snapshot.nx_max == 0
        assert Snapshot.ny_max == 0
        assert Snapshot.nz_max == 0
        assert Snapshot.bpg is None
