import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.mixins import MpiMixin
from reframe_tests.tests.standard_tests import GprMaxGeometryViewTest


@rfm.simple_test
class TestGeometryView(GprMaxGeometryViewTest):
    tags = {"test", "serial", "geometry only", "geometry view"}
    sourcesdir = "src/geometry_view_tests"
    model = parameter(["geometry_view_voxel", "geometry_view_fine"])
    geometry_views = ["partial_volume", "full_volume"]


@rfm.simple_test
class TestGeometryViewMPI(MpiMixin, TestGeometryView):
    tags = {"test", "mpi", "geometry only", "geometry view"}
    mpi_layout = parameter([[2, 2, 2], [4, 4, 1]])
    test_dependency = TestGeometryView
