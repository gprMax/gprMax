# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Utilities for gprMax material databases and legacy geometry migration."""

from gprMax.material_database import validate_material_database

from .convert_geometry import convert_geometry

__all__ = ["convert_geometry", "validate_material_database"]
