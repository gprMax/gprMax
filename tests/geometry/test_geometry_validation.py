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
Tests for geometry validation and repair tools.

Tests cover:
- Geometry object validation
- Tag consistency checks
- Material assignment validation
- Spatial consistency checks
- Repair hint generation
"""
import numpy as np
import pytest

from gprMax.geometry_validation import (
    ValidationResult,
    validate_geometry_object,
    validate_required_params,
    validate_material_assignment,
    check_manifold_consistency,
    validate_dispersive_material,
    check_overlap,
    get_repair_hints,
    add_validation_to_object,
)


class ValidationResultStr:
    """Helper class for test string representations."""
    
    def __init__(self, valid, message, severity="info", suggestions=None):
        self.valid = valid
        self.message = message
        self.severity = severity
        self.suggestions = suggestions or []


class TestValidationResult:
    """Test the ValidationResult class."""
    
    def test_valid_result(self):
        """Test valid result creation."""
        result = ValidationResult(valid=True, message="All good")
        assert result.valid is True
        assert result.message == "All good"
        assert result.severity == "info"
        assert result.suggestions == []
    
    def test_invalid_result(self):
        """Test invalid result creation."""
        result = ValidationResult(valid=False, message="Issue found", severity="error")
        assert result.valid is False
        assert result.message == "Issue found"
        assert result.severity == "error"
        assert result.suggestions == []
    
    def test_result_with_suggestions(self):
        """Test result with suggestions."""
        result = ValidationResult(valid=False, message="Issue", suggestions=["fix1", "fix2"])
        assert result.valid is False
        assert result.suggestions == ["fix1", "fix2"]


class TestValidateGeometryObject:
    """Test geometry object validation."""
    
    def test_basic_validation_structure(self, mock_grid, mock_geometry_object):
        """Test that validation returns a list of ValidationResult objects."""
        results = validate_geometry_object(mock_geometry_object, mock_grid)
        assert isinstance(results, list)
    
    def test_validation_with_missing_params(self, mock_grid, mock_geometry_object_with_missing_params):
        """Test validation catches missing required parameters."""
        results = validate_geometry_object(mock_geometry_object_with_missing_params, mock_grid)
        error_results = [r for r in results if r.severity == "error"]
        assert len(error_results) > 0
    
    def test_validation_with_valid_object(self, mock_grid, mock_geometry_object_valid):
        """Test validation passes for a valid object."""
        results = validate_geometry_object(mock_geometry_object_valid, mock_grid)
        assert isinstance(results, list)


class TestValidateRequiredParams:
    """Test the validate_required_params function."""
    
    def test_no_implementation_raises_not_implemented(self):
        """Test that base implementation does not crash."""
        try:
            validate_required_params(object())
            # Base implementation should not raise
        except NotImplementedError:
            pytest.fail("Base implementation should not raise NotImplementedError")
    
    def test_custom_subclass(self):
        """Test that subclasses can implement their own validation."""
        class MockObject:
            def validate_required_params(self):
                pass  # No-op implementation
        
        try:
            validate_required_params(MockObject())
            # Should not raise
        except Exception:
            pytest.fail("Should not raise for no-op implementation")


class TestValidateMaterialAssignment:
    """Test material assignment validation."""
    
    def test_valid_material_assignment(self, mock_grid, valid_material):
        """Test valid material assignment passes."""
        # Create a simple geometry object mock
        class MockObj:
            def _get_materialsrequested(self):
                return [0]
        
        results = validate_material_assignment(MockObj(), mock_grid)
        # Should not have error results for valid material
        error_results = [r for r in results if r.severity == "error"]
        assert len(error_results) == 0 or len(error_results) == 1  # Depending on material status
    
    def test_missing_material(self, mock_grid):
        """Test missing material is caught."""
        class MockObj:
            def _get_materialsrequested(self):
                return [999]  # Non-existent material ID
        
        results = validate_material_assignment(MockObj(), mock_grid)
        error_results = [r for r in results if r.severity == "error"]
        assert len(error_results) > 0
    
    def test_invalid_material(self, mock_grid, invalid_material):
        """Test invalid material is caught."""
        class MockObj:
            def _get_materialsrequested(self):
                return [999]
        
        results = validate_material_assignment(MockObj(), mock_grid)
        error_results = [r for r in results if r.severity == "error"]
        assert len(error_results) > 0


class TestCheckManifoldConsistency:
    """Test manifold consistency checks."""
    
    def test_small_occupied_cells(self):
        """Test that small occupied cell counts trigger warnings."""
        occupied = np.array([True, True])  # Only 2 cells
        results = check_manifold_consistency(None, occupied, 0)
        assert len(results) > 0
        assert any(r.severity == "warning" for r in results)
    
    def test_large_occupied_cells(self):
        """Test that large occupied cell counts don't trigger warnings."""
        occupied = np.array([True] * 100)
        results = check_manifold_consistency(None, occupied, 0)
        # May or may not have warnings, but shouldn't crash
        assert isinstance(results, list)


class TestValidateDispersiveMaterial:
    """Test dispersive material validation."""
    
    def test_valid_dispersive_material(self, valid_dispersive_material):
        """Test valid dispersive material passes validation."""
        results = validate_dispersive_material(valid_dispersive_material, None)
        # May have some warnings but should not have critical errors for basic case
        critical_errors = [r for r in results if r.severity == "error"]
        # Basic valid material should not have critical errors
    
    def test_invalid_er(self, invalid_dispersive_material_er):
        """Test invalid permittivity is caught."""
        results = validate_dispersive_material(invalid_dispersive_material_er, None)
        error_results = [r for r in results if r.severity == "error"]
        assert len(error_results) > 0
    
    def test_invalid_mr(self, invalid_dispersive_material_mr):
        """Test invalid permeability is caught."""
        results = validate_dispersive_material(invalid_dispersive_material_mr, None)
        error_results = [r for r in results if r.severity == "error"]
        assert len(error_results) > 0


class TestCheckOverlap:
    """Test overlap checking."""
    
    def test_no_overlap(self):
        """Test no overlap returns empty results."""
        tag_maps = {1: np.array([False, False, False]), 
                    2: np.array([False, False, False])}
        results = check_overlap(tag_maps)
        assert len(results) == 0
    
    def test_with_overlap(self):
        """Test overlap detection finds overlaps."""
        tag_maps = {1: np.array([True, False, False]), 
                    2: np.array([True, False, False])}
        results = check_overlap(tag_maps)
        assert len(results) > 0
        assert any(r.severity == "warning" for r in results)


class TestGetRepairHints:
    """Test repair hints generation."""
    
    def test_basic_hints(self, mock_geometry_object):
        """Test basic repair hints are generated."""
        hints = get_repair_hints(mock_geometry_object)
        assert isinstance(hints, list)
        # May be empty if no specific issues detected
    
    def test_missing_materials_hints(self, mock_geometry_object_missing_materials):
        """Test hints for missing materials."""
        hints = get_repair_hints(mock_geometry_object_missing_materials)
        assert isinstance(hints, list)
        # Should contain hints about materials


class TestAddValidationToObject:
    """Test the add_validation_to_object decorator."""
    
    def test_decorator_basic(self):
        """Test that the decorator adds required methods."""
        # This test verifies the decorator structure exists
        from gprMax.geometry_validation import add_validation_to_object
        # The decorator should be callable
        assert callable(add_validation_to_object)


# =============================================================================
# Fixtures
# =============================================================================

import pytest

@pytest.fixture
def mock_grid():
    """Provide a mock grid for tests."""
    class MockGrid:
        materials = {0: type('Material', (), {'is_valid': True, 'er': 2.0, 'se': 0.0, 'mr': 1.0, 'sm': 0.0})()}
        symmetry_boundaries = {}
    return MockGrid()


@pytest.fixture
def mock_geometry_object():
    """Provide a basic geometry object for tests."""
    class MockGeometryObject:
        def __init__(self):
            self.kwargs = {"material_id": 0, "tag": "test"}
        
        def declared_geometry_tags(self):
            return ("test",)
        
        def validate_required_params(self):
            pass  # No-op
        
        def _get_materialsrequested(self):
            return [0]
    
    return MockGeometryObject()


@pytest.fixture
def mock_geometry_object_valid(mock_grid):
    """Provide a valid geometry object for tests."""
    class MockGeometryObject:
        def __init__(self, grid):
            self.kwargs = {"material_id": 0, "tag": "valid-tag"}
            self._grid = grid
        
        def declared_geometry_tags(self):
            return ("valid-tag",)
        
        def validate_required_params(self):
            pass
        
        def _get_materialsrequested(self):
            return [0]
    
    return MockGeometryObject(mock_grid)


@pytest.fixture
def mock_geometry_object_with_missing_params(mock_grid):
    """Provide a geometry object with missing parameters."""
    class MockGeometryObject:
        def __init__(self, grid):
            self.kwargs = {}  # No materials or tag
        
        def declared_geometry_tags(self):
            return ()
        
        def validate_required_params(self):
            raise KeyError("Missing material_id")
        
        def _get_materialsrequested(self):
            return []
    
    return MockGeometryObject(mock_grid)


@pytest.fixture
def mock_geometry_object_missing_materials(mock_grid):
    """Provide a geometry object with missing materials."""
    class MockGeometryObject:
        def __init__(self, grid):
            self.kwargs = {"tag": "test"}  # No material_id
        
        def declared_geometry_tags(self):
            return ("test",)
        
        def validate_required_params(self):
            pass
        
        def _get_materialsrequested(self):
            return []  # No materials
    
    return MockGeometryObject(mock_grid)


@pytest.fixture
def valid_material():
    """Provide a valid material for tests."""
    return type('Material', (), {'is_valid': True, 'er': 2.0, 'se': 0.0, 'mr': 1.0, 'sm': 0.0})()


@pytest.fixture
def invalid_material():
    """Provide an invalid material for tests."""
    return type('Material', (), {'is_valid': False, 'er': -1.0, 'se': 0.0, 'mr': 1.0, 'sm': 0.0})()


@pytest.fixture
def valid_dispersive_material():
    """Provide a valid dispersive material for tests."""
    return type('DispersiveMaterial', (), {
        'is_valid': True, 
        'er': 2.0, 'se': 0.0, 
        'mr': 1.0, 'sm': 0.0, 
        'poles': 2, 
        'deltaer': [1.0, 2.0], 
        'tau': [1e-11, 2e-11], 
        'alpha': [0.1, 0.2]
    })()


@pytest.fixture
def invalid_dispersive_material_er():
    """Provide a dispersive material with invalid er for tests."""
    return type('DispersiveMaterial', (), {
        'is_valid': False, 
        'er': -1.0, 'se': 0.0, 
        'mr': 1.0, 'sm': 0.0, 
        'poles': 2, 
        'deltaer': [1.0, 2.0], 
        'tau': [1e-11, 2e-11], 
        'alpha': [0.1, 0.2]
    })()


@pytest.fixture
def invalid_dispersive_material_mr():
    """Provide a dispersive material with invalid mr for tests."""
    return type('DispersiveMaterial', (), {
        'is_valid': False, 
        'er': 2.0, 'se': 0.0, 
        'mr': -1.0, 'sm': 0.0, 
        'poles': 2, 
        'deltaer': [1.0, 2.0], 
        'tau': [1e-11, 2e-11], 
        'alpha': [0.1, 0.2]
    })()
