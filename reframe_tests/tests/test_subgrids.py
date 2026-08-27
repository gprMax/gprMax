# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

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
