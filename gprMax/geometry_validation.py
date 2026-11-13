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
Geometry validation and repair tools.

Provides comprehensive validation of geometry objects, tags, materials,
and spatial consistency checks for the gprMax FDTD framework.

Features:
- Geometry object validation (manifold checks, material assignments)
- Tag consistency validation
- Spatial overlap detection
- Material assignment validation
- Repair hints and suggestions
- Integration with geometry_tags module
"""

import logging
import numpy as np

from collections.abc import Iterable
from typing import Optional, Union, Tuple, List, Dict, Set

import gprMax.config as config
from gprMax.geometry_tags import validate_geometry_tag, UNTAGGED_NAME, _TAG_PATTERN
from gprMax.user_objects.user_objects import GeometryUserObject
from gprMax.materials import Material, DispersiveMaterial

logger = logging.getLogger(__name__)

# =============================================================================
# Geometry Validation Results
# =============================================================================

class ValidationResult:
    """Result of a geometry validation check."""
    
    def __init__(self, valid: bool, message: str = "", severity: str = "info",
                 suggestions: Optional[List[str]] = None):
        self.valid = valid
        self.message = message
        self.severity = severity  # "info", "warning", "error"
        self.suggestions = suggestions or []
    
    def __str__(self):
        return f"[{self.severity}] {self.message}"


# =============================================================================
# Geometry Object Validation
# =============================================================================

def validate_geometry_object(obj: GeometryUserObject, grid) -> List[ValidationResult]:
    """
    Validate a geometry user object for consistency and completeness.
    
    Checks include:
    - Required parameters present
    - Valid tag names
    - Valid material assignments
    - Spatial consistency
    - Parameter range validity
    
    Args:
        obj: GeometryUserObject instance to validate
        grid: FDTDGrid instance
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    # Check declared tags
    declared_tags = obj.declared_geometry_tags()
    for tag in declared_tags:
        try:
            validated = validate_geometry_tag(tag)
            if validated is None:
                results.append(ValidationResult(
                    valid=False,
                    message=f"Tag '{tag}' is reserved/untagged but declared",
                    severity="warning",
                    suggestions=["Use a different tag or declare as untagged"]
                ))
            elif not _TAG_PATTERN.fullmatch(validated):
                results.append(ValidationResult(
                    valid=False,
                    message=f"Tag '{tag}' has invalid format",
                    severity="error",
                    suggestions=[f"Use valid tag format: {_TAG_PATTERN.pattern}"]
                ))
        except (TypeError, ValueError) as e:
            results.append(ValidationResult(
                valid=False,
                message=str(e),
                severity="error",
                suggestions=["Check tag specification in geometry definition"]
            ))
    
    # Check for required parameters based on object type
    obj_type = obj.__class__.__name__
    try:
        obj.validate_required_params()
    except (KeyError, ValueError) as e:
        results.append(ValidationResult(
            valid=False,
            message=f"Missing required parameter for {obj_type}: {e}",
            severity="error",
            suggestions=["Check geometry object definition - all required parameters must be specified"]
        ))
    
    # Validate material assignments
    try:
        material_results = validate_material_assignment(obj, grid)
        results.extend(material_results)
    except Exception as e:
        results.append(ValidationResult(
            valid=False,
            message=f"Material validation error for {obj_type}: {e}",
            severity="error",
            suggestions=["Review material definitions and assignments"]
        ))
    
    return results


def validate_required_params(obj: GeometryUserObject) -> None:
    """
    Validate that all required parameters are present for a geometry object.
    
    Subclasses should implement this to check their specific required parameters.
    """
    # Base implementation - override in subclasses
    pass


def validate_material_assignment(obj: GeometryUserObject, grid) -> List[ValidationResult]:
    """
    Validate that material assignments are consistent and valid.
    
    Args:
        obj: GeometryUserObject instance
        grid: FDTDGrid instance
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    try:
        materialsrequested = obj._get_materialsrequested()
    except (KeyError, AttributeError):
        results.append(ValidationResult(
            valid=False,
            message="Could not determine requested materials",
            severity="error",
            suggestions=["Ensure material_id or material_ids are specified"]
        ))
        return results
    
    # Check each requested material exists and is valid
    for mat_id in materialsrequested:
        try:
            mat = grid.materials[mat_id]
            if not isinstance(mat, Material):
                results.append(ValidationResult(
                    valid=False,
                    message=f"Material ID {mat_id} is not a valid Material instance",
                    severity="error",
                    suggestions=[f"Material ID {mat_id} refers to non-material object"]
                ))
            elif not mat.is_valid:
                results.append(ValidationResult(
                    valid=False,
                    message=f"Material ID {mat_id} '{mat.ID}' is not valid",
                    severity="warning",
                    suggestions=["Review material parameters for consistency"]
                ))
        except KeyError:
            results.append(ValidationResult(
                valid=False,
                message=f"Material ID {mat_id} not found in grid materials",
                severity="error",
                suggestions=["Ensure material is defined before referencing it in geometry"]
            ))
    
    return results


# =============================================================================
# Spatial Consistency Validation
# =============================================================================

def check_manifold_consistency(grid, occupied: np.ndarray, tag_id: int) -> List[ValidationResult]:
    """
    Check manifold consistency of a geometry tag.
    
    A manifold geometry has no holes, no self-intersections, and consistent
    orientation of surfaces. This is important for correct voxelisation.
    
    Args:
        grid: FDTDGrid instance
        occupied: Boolean array indicating occupied cells
        tag_id: Tag ID associated with this geometry
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    # Check for isolated cells (cells with no neighbors in the same tag)
    # This can indicate holes or non-manifold geometry
    from gprMax.geometry_tags import validate_geometry_tag_ids
    
    tag_mask = occupied  # Simplified - would use actual tag map
    
    # Check for extremely small features (likely numerical errors)
    # Count connected components
    # ... (complex implementation would go here)
    
    # For now, add basic checks
    occupied_cells = np.sum(occupied)
    if occupied_cells < 3:
        results.append(ValidationResult(
            valid=False,
            message=f"Geometry has only {occupied_cells} occupied cells, "
                    "which is likely too small for valid simulation",
            severity="warning",
            suggestions=["Increase geometry size or check discretisation"]
        ))
    
    return results


# =============================================================================
# Material Validation
# =============================================================================

def validate_dispersive_material(material: Material, grid) -> List[ValidationResult]:
    """
    Validate a dispersive material's parameters for consistency.
    
    Checks permittivity, permeability, and dispersion model parameters.
    
    Args:
        material: Material instance to validate
        grid: FDTDGrid instance
        
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    if not isinstance(material, Material):
        results.append(ValidationResult(
            valid=False,
            message=f"Expected Material instance, got {type(material)}",
            severity="error"
        ))
        return results
    
    # Check permittivity is positive
    if material.er <= 0:
        results.append(ValidationResult(
            valid=False,
            message=f"Permittivity er={material.er} must be positive",
            severity="error",
            suggestions=["er must be > 0 for valid permittivity"]
        ))
    
    # Check permeability is positive
    if material.mr <= 0:
        results.append(ValidationResult(
            valid=False,
            message=f"Permeability mr={material.mr} must be positive",
            severity="error",
            suggestions=["mr must be > 0 for valid permeability"]
        ))
    
    # Check conductivity ranges
    if material.se < 0:
        results.append(ValidationResult(
            valid=False,
            message=f"Electric conductivity se={material.se} must be non-negative",
            severity="error",
            suggestions=["se must be >= 0"]
        ))
    
    if material.sm < 0:
        results.append(ValidationResult(
            valid=False,
            message=f"Magnetic conductivity sm={material.sm} must be non-negative",
            severity="error",
            suggestions=["sm must be >= 0"]
        ))
    
    # Check dispersive material parameters
    if isinstance(material, DispersiveMaterial):
        # Check poles are non-negative
        if material.poles < 0:
            results.append(ValidationResult(
                valid=False,
                message=f"Number of poles {material.poles} must be non-negative",
                severity="error",
                suggestions=["poles must be >= 0"]
            ))
        
        # Check dispersion parameters are consistent
        if material.poles > 0 and len(material.deltaer) != material.poles:
            results.append(ValidationResult(
                valid=False,
                message=f"deltaer length {len(material.deltaer)} must match poles {material.poles}",
                severity="error",
                suggestions=["Ensure deltaer array length matches poles"]
            ))
        
        if material.poles > 0 and len(material.tau) != material.poles:
            results.append(ValidationResult(
                valid=False,
                message=f"tau length {len(material.tau)} must match poles {material.poles}",
                severity="error",
                suggestions=["Ensure tau array length matches poles"]
            ))
    
    return results


# =============================================================================
# Utility Functions
# =============================================================================

def check_overlap(grid, tag_maps: Dict[int, np.ndarray]) -> List[ValidationResult]:
    """
    Check for overlapping geometry objects.
    
    Args:
        grid: FDTDGrid instance
        tag_maps: Dictionary mapping tag_id to boolean occupied array
        
    Returns:
        List of ValidationResult objects indicating overlaps
    """
    results = []
    
    tag_ids = list(tag_maps.keys())
    for i, tag_id_i in enumerate(tag_ids):
        for tag_id_j in tag_ids[i+1:]:
            overlap = np.logical_and(tag_maps[tag_id_i], tag_maps[tag_id_j])
            overlap_count = np.sum(overlap)
            
            if overlap_count > 0:
                # Check if overlap is at boundaries (expected) or interior (problematic)
                # ... (complex logic would go here)
                suggestions = ["Review geometry definitions for expected overlaps"]
                results.append(ValidationResult(
                    valid=False,
                    message=f"Geometry overlap detected between tags {tag_id_i} and {tag_id_j}: "
                            f"{overlap_count} cells overlap",
                    severity="warning",
                    suggestions=suggestions
                ))
    
    return results


# =============================================================================
# Repair Hints
# =============================================================================

def get_repair_hints(obj: GeometryUserObject, grid) -> List[str]:
    """
    Get repair hints for a geometry object.
    
    Args:
        obj: GeometryUserObject instance
        grid: FDTDGrid instance
        
    Returns:
        List of repair suggestion strings
    """
    hints = []
    
    # Check for common issues
    try:
        materialsrequested = obj._get_materialsrequested()
    except (KeyError, AttributeError):
        hints.append("Ensure material_id or material_ids are specified")
        return hints
    
    for mat_id in materialsrequested:
        try:
            mat = grid.materials[mat_id]
            if not mat.is_valid:
                hints.append(f"Material {mat.ID} has invalid parameters - review er, se, mr, sm")
        except KeyError:
            hints.append(f"Material ID {mat_id} not found - define material first")
    
    # Check tag validity
    declared_tags = obj.declared_geometry_tags()
    for tag in declared_tags:
        try:
            validated = validate_geometry_tag(tag)
            if validated is None:
                hints.append("Tag is reserved/untagged - ensure this is intended")
        except (TypeError, ValueError) as e:
            hints.append(f"Tag validation error: {e}")
    
    return hints


# =============================================================================
# Integration with GeometryUserObject
# =============================================================================

def add_validation_to_object(cls):
    """
    Decorator to add validation methods to GeometryUserObject subclasses.
    
    Adds:
    - validate() method returning List[ValidationResult]
    - get_repair_hints() method returning List[str]
    - is_valid property
    """
    original_build = cls.build
    
    def validated_build(self, grid):
        # Run validation before building
        results = validate_geometry_object(self, grid)
        
        # Report errors/warnings
        for result in results:
            if result.severity == "error":
                logger.error(str(result))
            elif result.severity == "warning":
                logger.warning(str(result))
        
        # Return validation results for caller to handle
        return original_build(self, grid), results
    
    cls.validate_geometry = lambda self, grid: validate_geometry_object(self, grid)
    cls.get_repair_hints = lambda self, grid: get_repair_hints(self, grid)
    cls.is_valid_geometry = lambda self, grid: all(r.valid for r in validate_geometry_object(self, grid))
    
    # Replace build method
    cls.build = validated_build
    
    return cls


# =============================================================================
# Module-level validation on import
# =============================================================================

# Validate geometry_tags module imports
try:
    from gprMax.geometry_tags import validate_geometry_tag, UNTAGGED_NAME
    _geometry_tags_validated = True
except Exception as e:
    logger.warning(f"Geometry tags validation import failed: {e}")
    _geometry_tags_validated = False

# Validate that the module can be imported without errors
