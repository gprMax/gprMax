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

"""Reframe regression tests for models defining geometry
"""


@rfm.simple_test
class TestSingleCellPml(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "box", "pml"}
    sourcesdir = "src/pml_tests"
    model = parameter(["single_cell_pml_2d"])


@rfm.simple_test
class TestInternalPmlSlab(GprMaxRegressionTest):
    """Internal MRIPML slab with a profile crossing the x midpoint."""

    tags = {"test", "serial", "pml", "internal_pml", "mripml"}
    sourcesdir = "src/pml_tests"
    model = parameter(["internal_pml_slab"])


"""Test MPI Functionality
"""


@rfm.simple_test
class TestSingleCellPmlMpi(MpiMixin, TestSingleCellPml):
    tags = {"test", "mpi", "geometery", "box", "pml"}
    mpi_layout = parameter([[2, 2, 1], [3, 3, 1]])
    test_dependency = TestSingleCellPml


@rfm.simple_test
class TestInternalPmlSlabMpi(MpiMixin, TestInternalPmlSlab):
    """Compare normal and transverse MPI slab partitions with serial."""

    tags = {"test", "mpi", "pml", "internal_pml", "mripml"}
    mpi_layout = parameter([[3, 1, 1], [1, 2, 1], [1, 1, 2], [2, 2, 1]])
    test_dependency = TestInternalPmlSlab
