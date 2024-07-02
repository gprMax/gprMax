import reframe as rfm
from base_tests import (
    GprMaxAPIRegressionTest,
    GprMaxBScanRegressionTest,
    GprMaxMPIRegressionTest,
    GprMaxRegressionTest,
    GprMaxTaskfarmRegressionTest,
)
from reframe.core.builtins import parameter, run_after, run_before

"""ReFrame tests for basic functionality

    Usage:
        cd gprMax/reframe_tests
        reframe -C configuraiton/{CONFIG_FILE} -c reframe_tests.py -c base_tests.py -r
"""


@rfm.simple_test
class TestAscan(GprMaxRegressionTest):
    tags = {
        "test",
        "serial",
        "ascan",
        "2d",
        "hertzian_dipole",
        "waveform",
        "material",
        "box",
        "cylinder",
    }
    model = parameter(["cylinder_Ascan_2D"])


@rfm.simple_test
class TestBscan(GprMaxBScanRegressionTest):
    tags = {
        "test",
        "serial",
        "bscan",
        "steps",
        "waveform",
        "hertzian_dipole",
        "material",
        "box",
        "cylinder",
    }
    model = parameter(["cylinder_Bscan_2D"])
    num_models = parameter([64])


@rfm.simple_test
class TestSingleNodeTaskfarm(GprMaxTaskfarmRegressionTest):
    tags = {
        "test",
        "mpi",
        "taskfarm",
        "steps",
        "waveform",
        "hertzian_dipole",
        "material",
        "box",
        "cylinder",
    }
    num_tasks = 8
    num_tasks_per_node = 8
    serial_dependency = TestBscan
    model = serial_dependency.model
    num_models = serial_dependency.num_models


@rfm.simple_test
class TestMultiNodeTaskfarm(GprMaxTaskfarmRegressionTest):
    tags = {
        "test",
        "mpi",
        "taskfarm",
        "steps",
        "waveform",
        "hertzian_dipole",
        "material",
        "box",
        "cylinder",
    }
    num_tasks = 32
    num_tasks_per_node = 8
    serial_dependency = TestBscan
    model = serial_dependency.model
    num_models = serial_dependency.num_models


@rfm.simple_test
class Test2DModelXY(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole"}
    model = parameter(["2D_EzHxHy"])


@rfm.simple_test
class Test2DModelXZ(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole"}
    model = parameter(["2D_EyHxHz"])


@rfm.simple_test
class Test2DModelYZ(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole"}
    model = parameter(["2D_ExHyHz"])


@rfm.simple_test
class TestHertzianDipoleSource(GprMaxRegressionTest):
    tags = {"test", "serial", "hertzian_dipole", "waveform"}
    model = parameter(["hertzian_dipole_fs"])


@rfm.simple_test
class TestMagneticDipoleSource(GprMaxRegressionTest):
    tags = {"test", "serial", "magnetic_dipole", "waveform"}
    model = parameter(["magnetic_dipole_fs"])


@rfm.simple_test
class TestDispersiveMaterials(GprMaxRegressionTest):
    tags = {"test", "serial", "hertzian_dipole", "waveform", "material", "dispersive", "box"}
    model = parameter(["hertzian_dipole_dispersive"])


@rfm.simple_test
class TestTransmissionLineSource(GprMaxRegressionTest):
    tags = {"test", "serial", "transmission_line", "waveform"}
    model = parameter(["transmission_line_fs"])


@rfm.simple_test
class TestEdgeGeometry(GprMaxRegressionTest):
    tags = {"test", "serial", "geometry", "edge", "transmission_line", "waveform", "antenna"}
    model = parameter(["antenna_wire_dipole_fs"])
    is_antenna_model = True


@rfm.simple_test
class TestSubgrids(GprMaxAPIRegressionTest):
    tags = {
        "test",
        "api",
        "serial",
        "subgrid",
        "hertzian_dipole",
        "waveform",
        "material",
        "dispersive",
        "cylinder",
    }
    model = parameter(["cylinder_fs"])


@rfm.simple_test
class TestSubgridsWithAntennaModel(GprMaxAPIRegressionTest):
    tags = {
        "test",
        "api",
        "serial",
        "subgrid",
        "antenna",
        "material",
        "box",
        "fractal_box",
        "add_surface_roughness",
    }
    model = parameter(["gssi_400_over_fractal_subsurface"])
    is_antenna_model = True

    @run_after("init")
    def skip_test(self):
        self.skip_if(self.current_system.name == "archer2", "Takes ~1hr 30m on ARCHER2")


@rfm.simple_test
class Test2DModelXYMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole"}
    mpi_layout = parameter([[4, 4, 1]])
    serial_dependency = Test2DModelXY
    model = serial_dependency.model


@rfm.simple_test
class Test2DModelXZMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole"}
    mpi_layout = parameter([[4, 1, 4]])
    serial_dependency = Test2DModelXZ
    model = serial_dependency.model


@rfm.simple_test
class Test2DModelYZMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole"}
    mpi_layout = parameter([[1, 4, 4]])
    serial_dependency = Test2DModelYZ
    model = serial_dependency.model


@rfm.simple_test
class TestHertzianDipoleSourceMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "hertzian_dipole", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestHertzianDipoleSource
    model = serial_dependency.model


@rfm.simple_test
class TestMagneticDipoleSourceMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "magnetic_dipole", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestMagneticDipoleSource
    model = serial_dependency.model


@rfm.simple_test
class TestDispersiveMaterialsMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "hertzian_dipole", "waveform", "material", "dispersive", "box"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestDispersiveMaterials
    model = serial_dependency.model


@rfm.simple_test
class TestTransmissionLineSourceMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "transmission_line", "waveform"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestTransmissionLineSource
    model = serial_dependency.model


@rfm.simple_test
class TestEdgeGeometryMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "geometry", "edge", "transmission_line", "waveform", "antenna"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestEdgeGeometry
    model = serial_dependency.model
    is_antenna_model = True


@rfm.simple_test
class TestBoxGeometryNoPml(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "box"}
    sourcesdir = "src/box_geometry_tests"
    model = parameter(["box_full_model", "box_half_model"])

    @run_before("run")
    def add_gprmax_commands(self):
        self.prerun_cmds.append(f"echo '#pml_cells: 0' >> {self.input_file}")


@rfm.simple_test
class TestBoxGeometryDefaultPml(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "box"}
    sourcesdir = "src/box_geometry_tests"
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
class TestBoxGeometryNoPmlMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "geometery", "box"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    serial_dependency = TestBoxGeometryNoPml
    model = serial_dependency.model
    sourcesdir = "src/box_geometry_tests"

    @run_before("run")
    def add_gprmax_commands(self):
        self.prerun_cmds.append(f"echo '#pml_cells: 0' >> {self.input_file}")


@rfm.simple_test
class TestBoxGeometryDefaultPmlMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "geometery", "box"}
    mpi_layout = parameter([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    serial_dependency = TestBoxGeometryDefaultPml
    model = serial_dependency.model
    sourcesdir = "src/box_geometry_tests"


@rfm.simple_test
class TestSingleCellPml(GprMaxRegressionTest):
    tags = {"test", "serial", "geometery", "box", "pml"}
    sourcesdir = "src/pml_tests"
    model = parameter(["single_cell_pml_2d"])


@rfm.simple_test
class TestSingleCellPmlMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "geometery", "box", "pml"}
    sourcesdir = "src/pml_tests"
    mpi_layout = parameter([[2, 2, 1], [3, 3, 1]])
    serial_dependency = TestSingleCellPml
    model = serial_dependency.model
