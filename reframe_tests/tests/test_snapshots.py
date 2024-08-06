import reframe as rfm
from reframe.core.builtins import parameter

from reframe_tests.tests.base_tests import GprMaxMPIRegressionTest, GprMaxRegressionTest


@rfm.simple_test
class Test2DSnapshot(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole", "snapshot"}
    sourcesdir = "src/snapshot_tests"
    model = parameter(["whole_domain_2d"])
    has_receiver_output = False
    snapshots = ["snapshot_0.h5", "snapshot_1.h5", "snapshot_2.h5", "snapshot_3.h5"]


@rfm.simple_test
class TestSnapshot(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole", "snapshot"}
    sourcesdir = "src/snapshot_tests"
    model = parameter(["whole_domain"])
    has_receiver_output = False
    snapshots = ["snapshot_0.h5", "snapshot_1.h5", "snapshot_2.h5", "snapshot_3.h5"]


@rfm.simple_test
class Test2DSliceSnapshot(GprMaxRegressionTest):
    tags = {"test", "serial", "2d", "waveform", "hertzian_dipole", "snapshot"}
    sourcesdir = "src/snapshot_tests"
    model = parameter(["2d_slices"])
    has_receiver_output = False
    snapshots = [
        "snapshot_x_05.h5",
        "snapshot_x_35.h5",
        "snapshot_x_65.h5",
        # "snapshot_x_95.h5",  # Fails due to out-of-bounds error
        "snapshot_y_15.h5",
        "snapshot_y_45.h5",
        "snapshot_y_75.h5",
        "snapshot_z_25.h5",
        "snapshot_z_55.h5",
        "snapshot_z_85.h5",
    ]


"""Test MPI Functionality
"""


@rfm.simple_test
class Test2DSnapshotMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole", "snapshot"}
    mpi_layout = parameter([[2, 2, 1], [3, 3, 1], [4, 4, 1]])
    serial_dependency = Test2DSnapshot
    model = serial_dependency.model


@rfm.simple_test
class TestSnapshotMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole", "snapshot"}
    mpi_layout = parameter(
        [
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
            [3, 1, 1],
            [1, 3, 1],
            [1, 1, 3],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
        ]
    )
    serial_dependency = TestSnapshot
    model = serial_dependency.model


@rfm.simple_test
class Test2DSliceSnapshotMpi(GprMaxMPIRegressionTest):
    tags = {"test", "mpi", "2d", "waveform", "hertzian_dipole", "snapshot"}
    mpi_layout = parameter(
        [
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
            [3, 1, 1],
            [1, 3, 1],
            [1, 1, 3],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
        ]
    )
    serial_dependency = Test2DSliceSnapshot
    model = serial_dependency.model
