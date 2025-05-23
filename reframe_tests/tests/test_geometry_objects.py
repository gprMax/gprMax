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
