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
"""
Tests for GradedIndexMaterial - a material with permittivity grading.
"""
import numpy as np
import pytest

import gprMax
import gprMax.config as config
from gprMax.materials import GradedIndexMaterial


class TestGradedIndexMaterialInit:
    """Test the GradedIndexMaterial constructor and basic properties."""

    def test_basic_initialization(self):
        """Test basic creation of GradedIndexMaterial."""
        material = GradedIndexMaterial(3, "graded-mat")
        assert material.ID == "graded-mat"
        assert material.numID == 3
        assert material.er_min == 1.0
        assert material.er_max == 2.0
        assert material.extent_x == 1.0
        assert material.type == "graded-index"

    def test_custom_parameters(self):
        """Test material with custom parameters."""
        material = GradedIndexMaterial(5, "custom-mat", er_min=3.0, er_max=7.0, extent_x=2.0)
        assert material.er_min == 3.0
        assert material.er_max == 7.0
        assert material.extent_x == 2.0


class TestGradedIndexMaterialCalculateEr:
    """Test the calculate_er method."""

    def test_er_at_min_position(self, monkeypatch):
        """Test permittivity at minimum position (x=0)."""
        material = GradedIndexMaterial(3, "pos-0-mat", er_min=2.0, er_max=5.0, extent_x=10.0)
        er = material.calculate_er(1e9, x_position=0.0)
        assert er.real == pytest.approx(2.0)  # er_min at x=0

    def test_er_at_max_position(self, monkeypatch):
        """Test permittivity at maximum position (x=extent_x)."""
        material = GradedIndexMaterial(3, "pos-max-mat", er_min=2.0, er_max=5.0, extent_x=10.0)
        er = material.calculate_er(1e9, x_position=10.0)
        assert er.real == pytest.approx(5.0)  # er_max at x=extent_x

    def test_er_at_intermediate_position(self, monkeypatch):
        """Test permittivity at intermediate position."""
        material = GradedIndexMaterial(3, "pos-intermediate-mat", er_min=1.0, er_max=4.0, extent_x=10.0)
        er = material.calculate_er(1e9, x_position=5.0)
        assert er.real == pytest.approx(2.5)  # halfway between er_min and er_max

    def test_er_independent_of_frequency(self, monkeypatch):
        """Test that permittivity is independent of frequency (no dispersion)."""
        material = GradedIndexMaterial(3, "no-disp-mat", er_min=2.0, er_max=6.0, extent_x=5.0)
        er_1GHz = material.calculate_er(1e9)
        er_10GHz = material.calculate_er(10e9)
        assert er_1GHz == er_10GHz  # Same permittivity at all frequencies


class TestGradedIndexMaterialIsValid:
    """Test the is_valid property."""

    def test_valid_graded_material(self):
        """Test valid graded index material."""
        material = GradedIndexMaterial(3, "valid-mat", er_min=1.0, er_max=5.0, extent_x=10.0)
        assert material.is_valid is True

    def test_invalid_er_min_negative(self):
        """Test material with negative er_min is invalid."""
        material = GradedIndexMaterial(3, "neg-min-mat", er_min=-1.0, er_max=5.0, extent_x=10.0)
        assert material.is_valid is False

    def test_invalid_er_max_less_than_min(self):
        """Test material with er_max < er_min is invalid."""
        material = GradedIndexMaterial(3, "inverted-mat", er_min=5.0, er_max=2.0, extent_x=10.0)
        assert material.is_valid is False


class TestGradedIndexMaterialReprAndCopy:
    """Test string representation and copy method."""

    def test_repr(self):
        """Test __repr__ method."""
        material = GradedIndexMaterial(3, "repr-mat", er_min=2.0, er_max=8.0, extent_x=5.0)
        repr_str = repr(material)
        assert "GradedIndexMaterial" in repr_str
        assert "ID='repr-mat'" in repr_str or "ID=" in repr_str

    def test_copy(self):
        """Test copy method creates independent copy."""
        material = GradedIndexMaterial(3, "copy-mat", er_min=2.0, er_max=8.0, extent_x=5.0)

        copied = material.copy()
        assert copied.ID == material.ID
        assert copied.numID == material.numID
        assert copied.er_min == material.er_min
        assert copied.er_max == material.er_max
        assert copied.extent_x == material.extent_x

        # Modify original and check copy is independent
        material.er_min = 1.0
        assert copied.er_min == 2.0  # Copy should be unchanged


class TestGradedIndexMaterialIntegration:
    """Integration-style tests."""

    def test_full_workflow(self, monkeypatch):
        """Test full workflow: init -> calculate_er -> copy."""
        material = GradedIndexMaterial(3, "workflow-mat", er_min=2.0, er_max=8.0, extent_x=10.0)

        # Step 1: Calculate permittivity at different positions
        er_0 = material.calculate_er(1e9, x_position=0.0)
        er_5 = material.calculate_er(1e9, x_position=5.0)
        er_10 = material.calculate_er(1e9, x_position=10.0)

        assert er_0.real == pytest.approx(2.0)  # er_min
        assert er_10.real == pytest.approx(8.0)  # er_max
        assert er_5.real == pytest.approx(5.0)  # halfway

        # Step 2: Verify is_valid
        assert material.is_valid is True

        # Step 3: Copy the material
        copied = material.copy()
        assert copied.er_min == material.er_min
        assert copied.er_max == material.er_max

        # Modify original and check copy is independent
        material.er_min = 1.0
        assert copied.er_min == 2.0  # Copy should be unchanged


def test_graded_material_import():
    """Test that GradedIndexMaterial can be imported from gprMax.materials."""
    from gprMax.materials import GradedIndexMaterial
    assert GradedIndexMaterial is not None


def test_graded_is_subclass_of_material():
    """Test that GradedIndexMaterial is a subclass of Material."""
    from gprMax.materials import GradedIndexMaterial, Material
    assert issubclass(GradedIndexMaterial, Material)


def test_graded_material_inherits_properties():
    """Test that GradedIndexMaterial inherits Material properties."""
    material = GradedIndexMaterial(3, "test")
    # Inherited from Material
    assert hasattr(material, 'numID')
    assert hasattr(material, 'ID')
    assert hasattr(material, 'type')
    assert hasattr(material, 'er')
    assert hasattr(material, 'se')
    assert hasattr(material, 'mr')
    assert hasattr(material, 'sm')
    # Inherited from DispersiveMaterial (Material base)
    assert hasattr(material, 'poles')
    assert hasattr(material, 'deltaer')
    assert hasattr(material, 'tau')
    assert hasattr(material, 'alpha')
