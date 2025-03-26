import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.mixins import MpiMixin
from reframe_tests.tests.standard_tests import GprMaxGeometryObjectTest


@rfm.simple_test
class TestGeometryObject(GprMaxGeometryObjectTest):
    tags = {"test", "serial", "geometry only", "geometry object"}
    sourcesdir = "src/geometry_object_tests"
    model = parameter(["geometry_object_write"])
    geometry_objects = ["partial_volume", "full_volume"]


@rfm.simple_test
class TestGeometryObjectMPI(MpiMixin, TestGeometryObject):
    tags = {"test", "mpi", "geometry only", "geometry object"}
    mpi_layout = parameter([[2, 2, 2], [4, 4, 1]])
    test_dependency = TestGeometryObject
