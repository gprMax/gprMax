from pathlib import Path

import reframe.utility.typecheck as typ
from numpy import prod
from reframe import RegressionMixin
from reframe.core.builtins import parameter, required, run_after, variable
from typing_extensions import TYPE_CHECKING

from reframe_tests.tests.base_tests import GprMaxBaseTest
from reframe_tests.tests.regression_checks import (
    ReceiverRegressionCheck,
    RegressionCheck,
    SnapshotRegressionCheck,
)

if TYPE_CHECKING:
    GprMaxMixin = GprMaxBaseTest
else:
    GprMaxMixin = RegressionMixin


class ReceiverMixin(GprMaxMixin):
    number_of_receivers = variable(int, value=-1)

    @run_after("setup")
    def add_receiver_regression_checks(self):
        reference_file = self.build_reference_filepath(self.output_file)

        if self.number_of_receivers > 0:
            for i in range(self.number_of_receivers):
                regression_check = ReceiverRegressionCheck(
                    self.output_file, reference_file, f"r{i}"
                )
                self.regression_checks.append(regression_check)
        else:
            regression_check = RegressionCheck(self.output_file, reference_file)
            self.regression_checks.append(regression_check)


class SnapshotMixin(GprMaxMixin):
    snapshots = variable(typ.List[str], value=[])

    def build_snapshot_filepath(self, snapshot: str) -> Path:
        return Path(f"{self.model}_snaps", snapshot).with_suffix(".h5")

    @run_after("setup")
    def add_snapshot_regression_checks(self):
        has_specified_snapshots = len(self.snapshots) > 0
        valid_test_dependency = self.test_dependency is not None and issubclass(
            self.test_dependency, SnapshotMixin
        )

        self.skip_if(
            not valid_test_dependency and not has_specified_snapshots,
            f"Must provide either a list of snapshots, or a test dependency that inherits from SnapshotMixin.",
        )
        self.skip_if(
            valid_test_dependency and has_specified_snapshots,
            f"Cannot provide both a list of snapshots, and a test dependency that inherits from SnapshotMixin.",
        )

        if valid_test_dependency:
            target = self.get_test_dependency()
            assert isinstance(target, SnapshotMixin)
            self.snapshots = target.snapshots

        for snapshot in self.snapshots:
            snapshot_file = self.build_snapshot_filepath(snapshot)
            reference_file = self.build_reference_filepath(snapshot)
            regression_check = SnapshotRegressionCheck(snapshot_file, reference_file)
            self.regression_checks.append(regression_check)


class PythonApiMixin(GprMaxMixin):
    executable = "time -p python"

    @run_after("setup")
    def set_python_input_file(self):
        """Input files for API tests will be python files"""
        self.input_file = self.input_file.with_suffix(".py")


class MpiMixin(GprMaxMixin):
    mpi_layout = parameter()

    @run_after("setup")
    def configure_mpi_tasks(self):
        """Add MPI specific commandline arguments"""
        self.num_tasks = int(prod(self.mpi_layout))
        self.executable_opts += ["--mpi", *map(str, self.mpi_layout)]


class BScanMixin(GprMaxMixin):
    num_models = parameter()

    @run_after("setup")
    def setup_bscan_test(self):
        """Add B-Scan specific commandline arguments and postrun cmds"""
        self.executable_opts += ["-n", str(self.num_models)]


class TaskfarmMixin(GprMaxMixin):
    extra_executable_opts = ["-taskfarm"]

    num_tasks = required

    @run_after("setup")
    def add_taskfarm_flag(self):
        """Add taskfarm specific commandline arguments"""
        self.executable_opts += ["-taskfarm"]


class AntennaModelMixin(GprMaxMixin):
    pass
