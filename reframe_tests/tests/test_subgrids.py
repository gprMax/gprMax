import reframe as rfm
from reframe.core.builtins import parameter, run_after

from reframe_tests.tests.base_tests import GprMaxAPIRegressionTest

"""Reframe regression tests for subgrids
"""


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
    sourcesdir = "src/subgrid_tests"
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
    sourcesdir = "src/subgrid_tests"
    model = parameter(["gssi_400_over_fractal_subsurface"])
    is_antenna_model = True

    @run_after("init")
    def skip_test(self):
        self.skip_if(self.current_system.name == "archer2", "Takes ~1hr 30m on ARCHER2")
