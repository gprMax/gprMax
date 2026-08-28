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

from reframe_tests.tests.mixins import GeometryOnlyMixin, MpiMixin, ReceiverMixin
from reframe_tests.tests.standard_tests import (
    GprMaxGeometryObjectsReadTest,
    GprMaxGeometryObjectsReadWriteTest,
    GprMaxGeometryObjectsWriteTest,
)


@rfm.simple_test
class TestGeometryObject(ReceiverMixin, GprMaxGeometryObjectsWriteTest):
    tags = {"test", "serial", "geometry only", "geometry object"}
    sourcesdir = "src/geometry_object_tests"
    model = parameter(["geometry_object_write"])
    geometry_objects_write = ["partial_volume", "full_volume"]


@rfm.simple_test
class TestGeometryObjectMPI(MpiMixin, TestGeometryObject):
    tags = {"test", "mpi", "geometry only", "geometry object"}
    mpi_layout = parameter([[2, 2, 2], [4, 4, 1]])
    test_dependency = TestGeometryObject


@rfm.simple_test
class TestGeometryObjectReadFullVolume(ReceiverMixin, GprMaxGeometryObjectsReadTest):
    tags = {"test", "serial", "geometry only", "geometry object"}
    sourcesdir = "src/geometry_object_tests"
    model = parameter(["geometry_object_read_full_volume"])
    geometry_objects_read = {"full_volume": "full_volume_read"}
    test_dependency = TestGeometryObject


@rfm.simple_test
class TestGeometryObjectReadFullVolumeMPI(MpiMixin, TestGeometryObjectReadFullVolume):
    tags = {"test", "mpi", "geometry only", "geometry object"}
    mpi_layout = parameter([[2, 2, 2], [4, 4, 1]])
    test_dependency = TestGeometryObject


@rfm.simple_test
class TestGeometryObjectReadWrite(GeometryOnlyMixin, GprMaxGeometryObjectsReadWriteTest):
    tags = {"test", "serial", "geometry only", "geometry object"}
    sourcesdir = "src/geometry_object_tests"
    model = parameter(["geometry_object_read_write"])
    geometry_objects_read = {
        "full_volume": "full_volume_read",
    }
    geometry_objects_write = ["partial_volume", "full_volume"]
    test_dependency = TestGeometryObject


@rfm.simple_test
class TestGeometryObjectReadWriteMPI(MpiMixin, TestGeometryObjectReadWrite):
    tags = {"test", "mpi", "geometry only", "geometry object"}
    mpi_layout = parameter([[2, 2, 2], [4, 4, 1]])
    test_dependency = TestGeometryObject


# TODO: This test fails in the serial implementation due to the geometry
# object being positioned such that it overflows the grid
# @rfm.simple_test
class TestGeometryObjectMove(GeometryOnlyMixin, GprMaxGeometryObjectsReadWriteTest):
    tags = {"test", "serial", "geometry only", "geometry object"}
    sourcesdir = "src/geometry_object_tests"
    model = parameter(["geometry_object_move"])
    geometry_objects_read = {
        "full_volume": "full_volume_read",
    }
    geometry_objects_write = ["partial_volume"]
    test_dependency = TestGeometryObject


@rfm.simple_test
class TestGeometryObjectMoveMPI(MpiMixin, TestGeometryObjectMove):
    tags = {"test", "mpi", "geometry only", "geometry object"}
    mpi_layout = parameter([[2, 2, 2], [4, 3, 1]])
    test_dependency = TestGeometryObject
