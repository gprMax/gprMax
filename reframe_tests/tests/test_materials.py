import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.base_tests import GprMaxMPIRegressionTest, GprMaxRegressionTest

"""Reframe regression tests for each gprMax source
"""


@rfm.simple_test
class TestDispersiveMaterials(GprMaxRegressionTest):
    tags = {"test", "serial", "hertzian_dipole", "waveform", "material", "dispersive", "box"}
    sourcesdir = "src/material_tests"
    model = parameter(["hertzian_dipole_dispersive"])


"""Test MPI Functionality
"""


@rfm.simple_test
class TestDispersiveMaterialsMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "hertzian_dipole", "waveform", "material", "dispersive", "box"}
    mpi_layout = parameter([[3, 3, 3]])
    serial_dependency = TestDispersiveMaterials
    model = serial_dependency.model
