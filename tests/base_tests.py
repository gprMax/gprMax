"""ReFrame base classes for GprMax tests"""
import os
import pathlib

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import performance_function, require_deps, run_after, sanity_function
from reframe.utility import udeps

GPRMAX_ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
PATH_TO_PYENV = os.path.join(".venv", "bin", "activate")


@rfm.simple_test
class CreatePyenvTest(rfm.RunOnlyRegressionTest):
    valid_systems = ["generic", "archer2:login"]
    valid_prog_environs = ["builtin", "PrgEnv-cray"]
    modules = ["cray-python"]

    prerun_cmds = [
        "python -m venv --system-site-packages --prompt gprMax .venv",
        f"source {PATH_TO_PYENV}",
        f"pip install -r {os.path.join(GPRMAX_ROOT_DIR, 'requirements.txt')}",
    ]
    executable = f"pip install -e {GPRMAX_ROOT_DIR}"

    @sanity_function
    def check_requirements_installed(self):
        """
        Check packages successfully installed from requirements.txt
        Check gprMax installed successfully and no other errors thrown
        """
        return (
            sn.assert_found(r"Successfully installed (?!gprMax)", self.stdout, "Failed to install requirements")
            and sn.assert_found(r"Successfully installed gprMax", self.stdout, "Failed to install gprMax")
            and sn.assert_not_found(r"finished with status 'error'", self.stdout)
            and sn.assert_not_found(r"ERROR:", self.stderr)
        )


class GprmaxBaseTest(rfm.RunOnlyRegressionTest):
    valid_systems = ["archer2:compute"]
    valid_prog_environs = ["PrgEnv-cray"]
    executable = "time -p python -m gprMax --log-level 25"
    postrun_cmds = [
        "sacct --format=JobID,State,Submit,Start,End,Elapsed,NodeList,ReqMem,MaxRSS,MaxVMSize --units=M -j $SLURM_JOBID"
    ]
    exclusive_access = True

    @run_after("init")
    def setup_env_vars(self):
        """Set OMP_NUM_THREADS environment variable from num_cpus_per_task"""
        self.env_vars["OMP_NUM_THREADS"] = self.num_cpus_per_task

        # Avoid inheriting slurm memory environment variables from any previous slurm job (i.e. the reframe job)
        self.prerun_cmds.append("unset SLURM_MEM_PER_NODE")
        self.prerun_cmds.append("unset SLURM_MEM_PER_CPU")

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
        return sn.assert_found(r"=== Simulation completed in ", self.stdout)

    @performance_function("s", perf_key="run_time")
    def extract_run_time(self):
        """Extract total runtime"""
        return sn.extractsingle(r"real\s+(?P<run_time>\S+)", self.stderr, "run_time", float)

    @performance_function("s", perf_key="simulation_time")
    def extract_simulation_time(self):
        """Extract simulation time reported by gprMax"""

        # sn.extractall throws an error if a group has value None.
        # Therefore have to handle the < 1 min and >= 1 min cases separately.
        if (
            sn.extractsingle(r"=== Simulation completed in \S+ (?P<case>minute|seconds)", self.stdout, "case")
            == "minute"
        ):
            simulation_time = sn.extractall(
                r"=== Simulation completed in (?P<minutes>\S+) minutes? and (?P<seconds>\S+) seconds =*",
                self.stdout,
                ["minutes", "seconds"],
                float,
            )
            minutes = simulation_time[0][0]
            seconds = simulation_time[0][1]
        else:
            minutes = 0
            seconds = sn.extractsingle(
                r"=== Simulation completed in (?P<seconds>\S+) seconds =*", self.stdout, "seconds", float
            )
        return minutes * 60 + seconds
