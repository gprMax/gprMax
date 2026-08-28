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

import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.mixins import MpiMixin
from reframe_tests.tests.standard_tests import GprMaxRegressionTest

"""Reframe regression tests for each gprMax source
"""


@rfm.simple_test
class TestDispersiveMaterials(GprMaxRegressionTest):
    tags = {"test", "serial", "hertzian_dipole", "waveform", "material", "dispersive", "box"}
    sourcesdir = "src/material_tests"
    model = parameter(["hertzian_dipole_dispersive"])


"""Test MPI Functionality
"""


@rfm.simple_test
class TestDispersiveMaterialsMpi(MpiMixin, TestDispersiveMaterials):
    tags = {"test", "mpi", "hertzian_dipole", "waveform", "material", "dispersive", "box"}
    mpi_layout = parameter([[3, 3, 3]])
    test_dependency = TestDispersiveMaterials
