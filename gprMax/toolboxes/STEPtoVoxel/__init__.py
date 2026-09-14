# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""STEP CAD assembly voxelisation for gprMax.

OpenCascade is imported only when a STEP file is inspected or converted, so
the optional toolbox does not affect normal gprMax imports.
"""

from .converter import (
    ConversionConfig,
    ConversionResult,
    convert_step,
    inspect_step,
    write_material_template,
)
from .markers import CADMarker, load_markers
from .visualisation import translate_reference_geometry

__all__ = [
    "ConversionConfig",
    "ConversionResult",
    "CADMarker",
    "convert_step",
    "inspect_step",
    "load_markers",
    "translate_reference_geometry",
    "write_material_template",
]
