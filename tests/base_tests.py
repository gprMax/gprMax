"""ReFrame base classes for GprMax tests"""
import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.utility import udeps


@rfm.simple_test
class CreatePyenvTest(rfm.RunOnlyRegressionTest):
    valid_systems = ["archer2:compute"]
    valid_prog_environs = ["PrgEnv-cray"]
    modules = ["cray-python"]

    # DOES NOT CURRENTLY WORK!!!
    prerun_cmds = [
        "python -m venv --system-site-packages --prompt gprMax .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt"
    ]
    executable = "pip install -e ."
    keep_files = ["requirements.txt"]

    @sanity_function
    def test_requirements_installed(self):
        return sn.assert_found(r'Successfully installed ', self.stdout) and sn.assert_not_found(r'ERROR', self.stdout)


class GprmaxBaseTest(rfm.RunOnlyRegressionTest):
    valid_systems = ["archer2:compute"]
    valid_prog_environs = ["PrgEnv-cray"]
    executable = "python -m gprMax --log-level 25"
    exclusive_access = True
    prerun_cmds = ["source .venv/bin/activate"]
    
    @run_after("init")
    def setup_omp(self):
        self.env_vars = {
            "OMP_NUM_THREADS": str(self.num_cpus_per_task)
        }

    @run_after("init")
    def inject_dependencies(self):
        self.depends_on("CreatePyenvTest", udeps.fully)

    @require_deps
    def set_sourcedir(self, CreatePyenvTest):
        self.sourcesdir = ['src', CreatePyenvTest(part="archer2:compute", environ="PrgEnv-cray").stagedir]
    
    @sanity_function
    def test_simulation_complete(self):
        return sn.assert_found(r'=== Simulation completed in ', self.stdout)