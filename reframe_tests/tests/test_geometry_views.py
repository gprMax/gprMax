import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.mixins import MpiMixin
from reframe_tests.tests.standard_tests import GprMaxGeometryViewTest


@rfm.simple_test
class TestGeometryViewVoxel(GprMaxGeometryViewTest):
    tags = {"test", "serial", "geometry only", "geometry view"}
    sourcesdir = "src/geometry_view_tests"
    model = parameter(["geometry_view_voxel"])
    geometry_views = ["partial_volume", "full_volume", "z_plane_48", "z_plane_49", "z_plane_50"]


@rfm.simple_test
class TestGeometryViewFine(GprMaxGeometryViewTest):
    tags = {"test", "serial", "geometry only", "geometry view"}
    sourcesdir = "src/geometry_view_tests"
    model = parameter(["geometry_view_fine"])
    geometry_views = ["partial_volume", "z_plane_48", "z_plane_49", "z_plane_50"]


@rfm.simple_test
class TestGeometryViewVoxelMPI(MpiMixin, TestGeometryViewVoxel):
    tags = {"test", "mpi", "geometry only", "geometry view"}
    mpi_layout = parameter([[2, 2, 2], [4, 4, 1]])
    test_dependency = TestGeometryViewVoxel


# Fails as the VTKHDF file format for unstructured grids uses seperate
# partitions for each MPI rank. This means the internal structure of the
# file is dependant on the number MPI ranks and just directly comparing
# the files does not check correctness.
@rfm.simple_test
class TestGeometryViewFineMPI(MpiMixin, TestGeometryViewFine):
    tags = {"test", "mpi", "geometry only", "geometry view"}
    mpi_layout = parameter([[2, 2, 2], [1, 4, 4]])
    test_dependency = TestGeometryViewFine
