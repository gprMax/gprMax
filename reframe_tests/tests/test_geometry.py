import reframe as rfm
from reframe.core.builtins import parameter, run_before

from reframe_tests.tests.base_tests import GprMaxMPIRegressionTest, GprMaxRegressionTest

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
class TestEdgeGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometry", "edge", "transmission_line", "waveform", "antenna"}
    sourcesdir = "src/geometry_tests/edge_geometry"
    model = parameter(["antenna_wire_dipole_fs"])
    is_antenna_model = True


"""Test MPI Functionality
"""


@rfm.simple_test
class TestBoxGeometryDefaultPmlMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "geometery", "box"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    serial_dependency = TestBoxGeometryDefaultPml
    model = serial_dependency.model


@rfm.simple_test
class TestBoxGeometryNoPmlMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "geometery", "box"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    serial_dependency = TestBoxGeometryNoPml
    model = serial_dependency.model

    @run_before("run")
    def add_gprmax_commands(self):
        self.prerun_cmds.append(f"echo '#pml_cells: 0' >> {self.input_file}")


@rfm.simple_test
class TestEdgeGeometryMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "geometry", "edge", "transmission_line", "waveform", "antenna"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestEdgeGeometry
    model = serial_dependency.model
    is_antenna_model = True
