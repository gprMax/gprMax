import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.mixins import MpiMixin
from reframe_tests.tests.standard_tests import GprMaxRegressionTest

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
class TestDispersiveMaterialsMpi(MpiMixin, TestDispersiveMaterials):
    tags = {"test", "mpi", "hertzian_dipole", "waveform", "material", "dispersive", "box"}
    mpi_layout = parameter([[3, 3, 3]])
    test_dependency = "TestDispersiveMaterials"
