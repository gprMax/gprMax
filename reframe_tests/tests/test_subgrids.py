import reframe as rfm
from reframe.core.builtins import parameter, run_after

from reframe_tests.tests.mixins import AntennaModelMixin, PythonApiMixin
from reframe_tests.tests.standard_tests import GprMaxRegressionTest

"""Reframe regression tests for subgrids
"""


@rfm.simple_test
class TestSubgrids(PythonApiMixin, GprMaxRegressionTest):
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
class TestSubgridsWithAntennaModel(AntennaModelMixin, PythonApiMixin, GprMaxRegressionTest):
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

    @run_after("init")
    def skip_test(self):
        self.skip_if(self.current_system.name == "archer2", "Takes ~1hr 30m on ARCHER2")
