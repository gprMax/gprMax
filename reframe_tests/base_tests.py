"""ReFrame base classes for GprMax tests"""
import os
from pathlib import Path
from shutil import copyfile

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import (
    performance_function,
    require_deps,
    run_after,
    run_before,
    sanity_function,
    variable,
)
from reframe.utility import udeps
from utilities.deferrable import path_join

GPRMAX_ROOT_DIR = Path(__file__).parent.parent.resolve()
PATH_TO_PYENV = os.path.join(".venv", "bin", "activate")


@rfm.simple_test
class CreatePyenvTest(rfm.RunOnlyRegressionTest):
    valid_systems = ["generic", "archer2:login"]
    valid_prog_environs = ["builtin", "PrgEnv-gnu"]
    modules = ["cray-python"]

    prerun_cmds = [
        "python -m venv --system-site-packages --prompt gprMax .venv",
        f"source {PATH_TO_PYENV}",
        "CC=cc CXX=CC FC=ftn python -m pip install --upgrade pip",
        f"CC=cc CXX=CC FC=ftn python -m pip install -r {os.path.join(GPRMAX_ROOT_DIR, 'requirements.txt')}",
    ]
    executable = f"CC=cc CXX=CC FC=ftn python -m pip install -e {GPRMAX_ROOT_DIR}"

    @sanity_function
    def check_requirements_installed(self):
        """
        Check packages successfully installed from requirements.txt
        Check gprMax installed successfully and no other errors thrown
        """
        return (
            sn.assert_found(
                r"(Successfully installed pip)|(Requirement already satisfied: pip.*\n(?!Collecting pip))",
                self.stdout,
                "Failed to update pip",
            )
            and sn.assert_found(
                r"Successfully installed (?!(gprMax)|(pip))",
                self.stdout,
                "Failed to install requirements",
            )
            and sn.assert_found(
                r"Successfully installed gprMax", self.stdout, "Failed to install gprMax"
            )
            and sn.assert_not_found(r"finished with status 'error'", self.stdout)
            and sn.assert_not_found(r"(ERROR|error):", self.stderr)
        )


class GprMaxBaseTest(rfm.RunOnlyRegressionTest):
    valid_systems = ["archer2:compute"]
    valid_prog_environs = ["PrgEnv-gnu"]
    modules = ["cray-python"]
    executable = "time -p python -m gprMax --log-level 25"
    exclusive_access = True

    @run_after("init")
    def setup_env_vars(self):
        """Set OMP_NUM_THREADS environment variable from num_cpus_per_task"""
        self.env_vars["OMP_NUM_THREADS"] = self.num_cpus_per_task

        if self.current_system.name == "archer2":
            # Avoid inheriting slurm memory environment variables from any previous slurm job (i.e. the reframe job)
            self.prerun_cmds.append("unset SLURM_MEM_PER_NODE")
            self.prerun_cmds.append("unset SLURM_MEM_PER_CPU")

            # Set the matplotlib cache to the work filesystem
            self.env_vars["MPLCONFIGDIR"] = "${HOME/home/work}/.config/matplotlib"

    # TODO: Change CreatePyenvTest to a fixture instead of a test dependency
    @run_after("init")
    def inject_dependencies(self):
        """Test depends on the Python virtual environment building correctly"""
        self.depends_on("CreatePyenvTest", udeps.by_env)

    @require_deps
    def get_pyenv_path(self, CreatePyenvTest):
        """Add prerun command to load the built Python environment"""
        path_to_pyenv = os.path.join(CreatePyenvTest(part="login").stagedir, PATH_TO_PYENV)
        self.prerun_cmds.append(f"source {path_to_pyenv}")

    @sanity_function
    def test_simulation_complete(self):
        """Check simulation completed successfully"""
        # TODO: Check for correctness/regression rather than just completing
        return sn.assert_not_found(
            r"(?i)error",
            self.stderr,
            f"An error occured. See '{path_join(self.stagedir, self.stderr)}' for details.",
        ) and sn.assert_found(
            r"=== Simulation completed in ", self.stdout, "Simulation did not complete"
        )

    @performance_function("s", perf_key="run_time")
    def extract_run_time(self):
        """Extract total runtime from the last task to complete"""
        return sn.extractsingle(
            r"real\s+(?P<run_time>\S+)", self.stderr, "run_time", float, self.num_tasks - 1
        )

    @performance_function("s", perf_key="simulation_time")
    def extract_simulation_time(self):
        """Extract simulation time reported by gprMax"""

        # sn.extractall throws an error if a group has value None.
        # Therefore have to handle the < 1 min, >= 1 min and >= 1 hour cases separately.
        timeframe = sn.extractsingle(
            r"=== Simulation completed in \S+ (?P<timeframe>hour|minute|second)",
            self.stdout,
            "timeframe",
        )
        if timeframe == "hour":
            simulation_time = sn.extractall(
                r"=== Simulation completed in (?P<hours>\S+) hours?, (?P<minutes>\S+) minutes? and (?P<seconds>\S+) seconds? =*",
                self.stdout,
                ["hours", "minutes", "seconds"],
                float,
            )
            hours = simulation_time[0][0]
            minutes = simulation_time[0][1]
            seconds = simulation_time[0][2]
        elif timeframe == "minute":
            hours = 0
            simulation_time = sn.extractall(
                r"=== Simulation completed in (?P<minutes>\S+) minutes? and (?P<seconds>\S+) seconds? =*",
                self.stdout,
                ["minutes", "seconds"],
                float,
            )
            minutes = simulation_time[0][0]
            seconds = simulation_time[0][1]
        else:
            hours = 0
            minutes = 0
            seconds = sn.extractsingle(
                r"=== Simulation completed in (?P<seconds>\S+) seconds? =*",
                self.stdout,
                "seconds",
                float,
            )
        return hours * 3600 + minutes * 60 + seconds


class GprMaxRegressionTest(GprMaxBaseTest):
    input_file = variable(str)
    output_file = variable(str)

    h5diff_header = f"{'=' * 10} h5diff output {'=' * 10}"

    @run_before("run", always_last=True)
    def setup_regression_check(self):
        """Build reference file path and add h5diff command to run after the test"""
        self.modules.append("cray-hdf5")
        self.reference_file = Path("regression_checks", self.unique_name).with_suffix(".h5")
        self.reference_file = os.path.abspath(self.reference_file)
        if os.path.exists(self.reference_file):
            self.postrun_cmds.append(f"echo {self.h5diff_header}")
            self.postrun_cmds.append(f"h5diff {self.output_file} {self.reference_file}")

    @sanity_function
    def regression_check(self):
        """
        Perform regression check by checking for the h5diff output.
        Create reference file from the test output if it does not exist.
        """
        if sn.path_exists(self.reference_file):
            h5diff_output = sn.extractsingle(
                f"{self.h5diff_header}\n(?P<h5diff>[\S\s]*)", self.stdout, "h5diff"
            )
            return (
                self.test_simulation_complete()
                and sn.assert_found(self.h5diff_header, self.stdout, "Failed to find h5diff header")
                and sn.assert_false(
                    h5diff_output,
                    (
                        f"Found h5diff output (see '{path_join(self.stagedir, self.stdout)}')\n"
                        f"For more details run: 'h5diff {os.path.abspath(self.output_file)} {self.reference_file}'\n"
                        f"To re-create regression file, delete '{self.reference_file}' and rerun the test."
                    ),
                )
            )
        else:
            copyfile(self.output_file, self.reference_file)
            return sn.assert_true(
                False, f"No reference file exists. Creating... '{self.reference_file}'"
            )


class GprMaxMpiTest(GprMaxBaseTest):
    pass
