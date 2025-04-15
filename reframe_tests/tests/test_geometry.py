import reframe as rfm
from reframe.core.builtins import parameter, run_before

from reframe_tests.tests.mixins import AntennaModelMixin, MpiMixin
from reframe_tests.tests.standard_tests import GprMaxRegressionTest

"""Reframe regression tests for models defining geometry
"""


@rfm.simple_test
class TestBoxGeometryDefaultPml(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "box"}
    sourcesdir = "src/geometry_tests/box_geometry"
    model = parameter(
        [
            "box_full_model",
            "box_half_model",
            "box_rigid",
            "box_single_rank",
            "box_outside_pml",
            "box_single_rank_outside_pml",
        ]
    )


@rfm.simple_test
class TestBoxGeometryNoPml(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "box"}
    sourcesdir = "src/geometry_tests/box_geometry"
    model = parameter(["box_full_model", "box_half_model", "box_single_rank"])

    @run_before("run")
    def add_gprmax_commands(self):
        self.prerun_cmds.append(f"echo '#pml_cells: 0' >> {self.input_file}")


@rfm.simple_test
class TestConeGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "cone"}
    sourcesdir = "src/geometry_tests/cone_geometry"
    model = parameter(
        ["full_cone", "small_cone", "non_axis_aligned_cone", "overtall_cone", "rigid_cone"]
    )


@rfm.simple_test
class TestCylinderGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "cylinder"}
    sourcesdir = "src/geometry_tests/cylinder_geometry"
    model = parameter(
        [
            "cylinder_full",
            "cylinder_small",
            "cylinder_non_axis_aligned",
            "cylinder_overtall",
            "cylinder_rigid",
        ]
    )


@rfm.simple_test
class TestCylindricalSectorGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "cylindrical", "sector", "cylindrical_sector"}
    sourcesdir = "src/geometry_tests/cylindrical_sector_geometry"
    model = parameter(
        [
            "cylindrical_sector_x_full",
            "cylindrical_sector_x_half",
            "cylindrical_sector_y_small",
            "cylindrical_sector_z_outside_boundary",
            "cylindrical_sector_z_rigid",
        ]
    )


@rfm.simple_test
class TestEdgeGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometry", "edge"}
    sourcesdir = "src/geometry_tests/edge_geometry"
    model = parameter(["edge_y_full_width", "edge_z_small"])


@rfm.simple_test
class TestEdgeGeometryAntennaModel(AntennaModelMixin, GprMaxRegressionTest):
    tags = {"test", "serial", "geometry", "edge", "transmission_line", "waveform", "antenna"}
    sourcesdir = "src/geometry_tests/edge_geometry"
    model = parameter(["antenna_wire_dipole_fs"])


@rfm.simple_test
class TestEllipsoidGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "ellipsoid"}
    sourcesdir = "src/geometry_tests/ellipsoid_geometry"
    model = parameter(
        ["ellipsoid_full", "ellipsoid_small", "ellipsoid_outside_boundary", "ellipsoid_rigid"]
    )


@rfm.simple_test
class TestFractalBoxGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "fractal", "box", "fractal_box"}
    sourcesdir = "src/geometry_tests/fractal_box_geometry"
    model = parameter(
        ["fractal_box_full", "fractal_box_half", "fractal_box_small", "fractal_box_weighted"]
    )


# TODO: Add Mixin class to enable testing that invalid geometry throws an error
@rfm.simple_test
class TestPlateGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "plate"}
    sourcesdir = "src/geometry_tests/plate_geometry"
    model = parameter(["plate_x_full", "plate_z_full", "plate_y_small"])


@rfm.simple_test
class TestSphereGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "sphere"}
    sourcesdir = "src/geometry_tests/sphere_geometry"
    model = parameter(["sphere_full", "sphere_small", "sphere_outside_boundary", "sphere_rigid"])


@rfm.simple_test
class TestTriangleGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "triangle"}
    sourcesdir = "src/geometry_tests/triangle_geometry"
    model = parameter(["triangle_x_full", "triangle_y_small", "triangle_z_rigid"])


"""Test MPI Functionality
"""


@rfm.simple_test
class TestBoxGeometryDefaultPmlMpi(MpiMixin, TestBoxGeometryDefaultPml):
    tags = {"test", "mpi", "geometery", "box"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestBoxGeometryDefaultPml


@rfm.simple_test
class TestBoxGeometryNoPmlMpi(MpiMixin, TestBoxGeometryNoPml):
    tags = {"test", "mpi", "geometery", "box"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestBoxGeometryNoPml


@rfm.simple_test
class TestConeGeometryMpi(MpiMixin, TestConeGeometry):
    tags = {"test", "mpi", "geometery", "cone"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestConeGeometry


@rfm.simple_test
class TestCylinderGeometryMpi(MpiMixin, TestCylinderGeometry):
    tags = {"test", "mpi", "geometery", "cylindrical", "sector", "cylindrical_sector"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestCylinderGeometry


@rfm.simple_test
class TestCylindricalSectorGeometryMpi(MpiMixin, TestCylindricalSectorGeometry):
    tags = {"test", "mpi", "geometery", "cylinder"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestCylindricalSectorGeometry


@rfm.simple_test
class TestEdgeGeometryMpi(MpiMixin, TestEdgeGeometry):
    tags = {"test", "mpi", "geometry", "edge"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestEdgeGeometry


@rfm.simple_test
class TestEdgeGeometryAntennaModelMpi(MpiMixin, TestEdgeGeometryAntennaModel):
    tags = {"test", "mpi", "geometry", "edge", "transmission_line", "waveform", "antenna"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestEdgeGeometryAntennaModel


@rfm.simple_test
class TestEllipsoidGeometryMpi(MpiMixin, TestEllipsoidGeometry):
    tags = {"test", "mpi", "geometery", "ellipsoid"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestEllipsoidGeometry


@rfm.simple_test
class TestFractalBoxGeometryMpi(MpiMixin, TestFractalBoxGeometry):
    tags = {"test", "mpi", "geometery", "fractal", "box", "fractal_box"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestFractalBoxGeometry


@rfm.simple_test
class TestPlateGeometryMpi(MpiMixin, TestPlateGeometry):
    tags = {"test", "mpi", "geometery", "plate"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestPlateGeometry


@rfm.simple_test
class TestSphereGeometryMpi(MpiMixin, TestSphereGeometry):
    tags = {"test", "mpi", "geometery", "sphere"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestSphereGeometry


@rfm.simple_test
class TestTriangleGeometryMpi(MpiMixin, TestTriangleGeometry):
    tags = {"test", "mpi", "geometery", "triangle"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    test_dependency = TestTriangleGeometry
