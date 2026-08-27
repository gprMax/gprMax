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

"""Reframe regression tests for 2D models (TMx, TMy, and TMz)
"""


@rfm.simple_test
class Test2DModelXY(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole"}
    sourcesdir = "src/2d_tests"
    model = parameter(["2D_EzHxHy"])


@rfm.simple_test
class Test2DModelXZ(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole"}
    sourcesdir = "src/2d_tests"
    model = parameter(["2D_EyHxHz"])


@rfm.simple_test
class Test2DModelYZ(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole"}
    sourcesdir = "src/2d_tests"
    model = parameter(["2D_ExHyHz"])


"""Test MPI Functionality
"""


@rfm.simple_test
class Test2DModelXYMpi(MpiMixin, Test2DModelXY):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole"}
    mpi_layout = parameter([[4, 4, 1]])
    test_dependency = Test2DModelXY


@rfm.simple_test
class Test2DModelXZMpi(MpiMixin, Test2DModelXZ):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole"}
    mpi_layout = parameter([[4, 1, 4]])
    test_dependency = Test2DModelXZ


@rfm.simple_test
class Test2DModelYZMpi(MpiMixin, Test2DModelYZ):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole"}
    mpi_layout = parameter([[1, 4, 4]])
    test_dependency = Test2DModelYZ
