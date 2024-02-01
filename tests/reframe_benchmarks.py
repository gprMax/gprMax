import reframe as rfm
from base_tests import GprMaxBaseTest
from reframe.core.builtins import parameter, run_after

"""ReFrame tests for performance benchmarking

    Usage:
        cd gprMax/tests
        reframe -C configuraiton/{CONFIG_FILE} -c reframe_benchmarks.py -c base_tests.py -r
"""


@rfm.simple_test
class SingleNodeBenchmark(GprMaxBaseTest):
    tags = {"benchmark", "single node", "openmp"}

    num_tasks = 1
    omp_threads = parameter([1, 2, 4, 8, 16, 32, 64, 128])
    domain = parameter([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cpu_freq = parameter([2000000, 2250000])
    time_limit = "8h"

    @run_after("init")
    def setup_env_vars(self):
        self.num_cpus_per_task = self.omp_threads
        self.env_vars["SLURM_CPU_FREQ_REQ"] = self.cpu_freq
        super().setup_env_vars()

    @run_after("init")
    def set_model_file(self):
        input_file = f"benchmark_model_{self.domain}.in"
        self.executable_opts = [input_file]
        self.keep_files = [input_file]
