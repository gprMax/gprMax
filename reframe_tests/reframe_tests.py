import reframe as rfm
from base_tests import (
    GprMaxAPIRegressionTest,
    GprMaxBScanRegressionTest,
    GprMaxMPIRegressionTest,
    GprMaxRegressionTest,
    GprMaxTaskfarmRegressionTest,
)
from reframe.core.builtins import parameter, run_after

"""ReFrame tests for basic functionality

    Usage:
        cd gprMax/reframe_tests
        reframe -C configuraiton/{CONFIG_FILE} -c reframe_tests.py -c base_tests.py -r
"""


@rfm.simple_test
class TestBscan(GprMaxBScanRegressionTest):
    tags = {"test", "bscan"}

    model = "cylinder_Bscan_2D"
    num_models = 64


@rfm.simple_test
class TestSingleNodeTaskfarm(GprMaxTaskfarmRegressionTest):
    tags = {"test", "mpi", "taskfarm"}

    model = "cylinder_Bscan_2D"
    num_tasks = 8
    num_tasks_per_node = 8
    num_models = 64
    serial_dependecy = TestBscan


@rfm.simple_test
class TestMultiNodeTaskfarm(GprMaxTaskfarmRegressionTest):
    tags = {"test", "mpi", "taskfarm"}

    model = "cylinder_Bscan_2D"
    num_tasks = 32
    num_tasks_per_node = 8
    num_models = 64
    serial_dependecy = TestBscan


@rfm.simple_test
class Test2DModelXY(GprMaxRegressionTest):
    tags = {"test", "serial", "2d"}

    model = "2D_EzHxHy"


@rfm.simple_test
class Test2DModelYZ(GprMaxRegressionTest):
    tags = {"test", "serial", "2d"}

    model = "2D_EzHxHy"


@rfm.simple_test
class BasicModelsTest(GprMaxRegressionTest):
    tags = {"test", "serial", "regression"}

    def __init__(self):
        super().__init__()
        # List of available basic test models
        self.model = parameter(
            [
                "2D_ExHyHz",
                "2D_EyHxHz",
                "2D_EzHxHy",
                "2D_ExHyHz_hs",
                "cylinder_Ascan_2D",
                "hertzian_dipole_fs",
                "hertzian_dipole_hs",
                "hertzian_dipole_dispersive",
                "magnetic_dipole_fs",
                "magnetic_dipole_hs",
            ]
        )


@rfm.simple_test
class AntennaModelsTest(GprMaxRegressionTest):
    tags = {"test", "serial", "regression", "antenna"}

    # List of available antenna test models
    model = "antenna_wire_dipole_fs"


@rfm.simple_test
class SubgridTest(GprMaxAPIRegressionTest):
    tags = {"test", "api", "serial", "regression", "subgrid"}

    def __init__(self):
        super().__init__()
        # List of available subgrid test models
        self.model = parameter(
            [
                "cylinder_fs",
                # "gssi_400_over_fractal_subsurface",  # Takes ~1hr 30m on ARCHER2
            ]
        )


@rfm.simple_test
class MPIBasicModelsTest(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "regression"}

    num_tasks_per_node = 4
    mpi_layout = [2, 2, 2]

    serial_dependency = BasicModelsTest

    def __init__(self):
        super().__init__()
        # List of available basic test models
        self.model = parameter(
            [
                "2D_ExHyHz",
                "2D_EyHxHz",
                "2D_EzHxHy",
                "2D_ExHyHz_hs",
                "cylinder_Ascan_2D",
                "hertzian_dipole_fs",
                "hertzian_dipole_hs",
                "hertzian_dipole_dispersive",
                "magnetic_dipole_fs",
                "magnetic_dipole_hs",
            ]
        )


@rfm.simple_test
class TestBoxGeometryNoPml(GprMaxRegressionTest):
    sourcesdir = "src/box_geometry_tests"

    def __init__(self):
        super().__init__()
        self.model = parameter(
            [
                "box_full_model",
                "box_half_model",
            ]
        )

    @run_after("init", always_last=True)
    def add_gprmax_commands(self):
        self.prerun_cmds.append(f"echo '#pml_cells: 0' >> {self.input_file}")


@rfm.simple_test
class TestBoxGeometryDefaultPml(GprMaxRegressionTest):
    sourcesdir = "src/box_geometry_tests"

    def __init__(self):
        super().__init__()
        self.model = parameter(
            [
                "box_full_model",
                "box_half_model",
            ]
        )


@rfm.simple_test
class TestBoxGeometryNoPmlMpi(GprMaxMPIRegressionTest):
    mpi_layout = [2, 2, 2]

    serial_dependency = TestBoxGeometryNoPml

    def __init__(self):
        super().__init__()
        self.model = parameter(
            [
                "box_full_model",
                "box_half_model",
            ]
        )

    @run_after("init", always_last=True)
    def add_gprmax_commands(self):
        self.prerun_cmds.append(f"echo '#pml_cells: 0' >> {self.input_file}")


@rfm.simple_test
class TestBoxGeometryDefaultPmlMpi(GprMaxMPIRegressionTest):
    mpi_layout = [2, 2, 2]

    serial_dependency = TestBoxGeometryDefaultPml

    def __init__(self):
        super().__init__()
        self.model = parameter(
            [
                "box_full_model",
                "box_half_model",
            ]
        )
