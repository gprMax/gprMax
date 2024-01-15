"""ReFrame base classes for GprMax tests"""
import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.utility import udeps

from configuration.user_config import GPRMAX_ROOT_DIR


PATH_TO_PYENV = os.path.join(".venv", "bin", "activate")

@rfm.simple_test
class CreatePyenvTest(rfm.RunOnlyRegressionTest):
    valid_systems = ["generic", "archer2:login"]
    valid_prog_environs = ["builtin", "PrgEnv-cray"]
    modules = ["cray-python"]

    prerun_cmds = [
        "python -m venv --system-site-packages --prompt gprMax .venv",
        f"source {PATH_TO_PYENV}",
        f"pip install -r {os.path.join(GPRMAX_ROOT_DIR, 'requirements.txt')}"
    ]
    executable = f"pip install -e {GPRMAX_ROOT_DIR}"

    @sanity_function
    def check_requirements_installed(self):
        return sn.assert_found(r"Successfully installed (?!gprMax)", self.stdout, "Failed to install requirements") \
        and sn.assert_found(r"Successfully installed gprMax", self.stdout, "Failed to install gprMax") \
        and sn.assert_not_found(r"finished with status 'error'", self.stdout) \
        and sn.assert_not_found(r"ERROR:", self.stderr)


class GprmaxBaseTest(rfm.RunOnlyRegressionTest):
    valid_systems = ["archer2:compute"]
    valid_prog_environs = ["PrgEnv-cray"]
    executable = "python -m gprMax --log-level 25"
    exclusive_access = True
    
    @run_after("init")
    def setup_omp(self):
        self.env_vars = {
            "OMP_NUM_THREADS": str(self.num_cpus_per_task)
        }

    @run_after("init")
    def inject_dependencies(self):
        self.depends_on("CreatePyenvTest", udeps.by_env)

    @require_deps
    def set_sourcesdir(self, CreatePyenvTest):
        path_to_pyenv = os.path.join(CreatePyenvTest(part="login").stagedir, PATH_TO_PYENV)
        self.prerun_cmds = [f"source {path_to_pyenv}"]
    
    @sanity_function
    def test_simulation_complete(self):
        return sn.assert_found(r"=== Simulation completed in ", self.stdout)