from pathlib import Path

import reframe as rfm
from reframe.core.builtins import parameter

from base_tests import GprmaxBaseTest


"""ReFrame tests for taskfarm functionality

    Usage:
        cd gprMax/tests
        reframe -C configuraiton/{CONFIG_FILE} -c test_mpi.py -r
"""


@rfm.simple_test
class BScanTest(GprmaxBaseTest):

    executable_opts = "cylinder_Bscan_2D.in -n 64 -mpi".split()
    num_tasks = 8
    num_cpus_per_task = 16


@rfm.simple_test
class BasicModelsTest(GprmaxBaseTest):

    # List of available basic test models
    model = parameter([
        "2D_ExHyHz"
        "2D_EyHxHz",
        "2D_EzHxHy",
        "cylinder_Ascan_2D",
        "hertzian_dipole_fs",
        "hertzian_dipole_hs",
        "hertzian_dipole_dispersive",
        "magnetic_dipole_fs",
    ])
    num_cpus_per_task = 16

    @run_after("init")
    def set_model(self):
        self.executable_opts = f"{self.model}.in -o {self.model}.h5".split()
        self.keep_files = [f"{self.model}.in", f"{self.model}.h5"]
