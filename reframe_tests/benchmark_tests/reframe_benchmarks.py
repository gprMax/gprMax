import os
from pathlib import Path

import numpy as np
from primePy import primes
from reframe import simple_test
from reframe.core.builtins import parameter, run_after

from reframe_tests.tests.base_tests import GprMaxMPIRegressionTest, GprMaxRegressionTest

"""ReFrame tests for performance benchmarking

    Usage:
        cd gprMax/reframe_tests
        reframe -C configuraiton/{CONFIG_FILE} -c reframe_benchmarks.py -c base_tests.py -r
"""


def calculate_mpi_decomposition(number: int):
    factors: list[int] = primes.factors(number)
    if len(factors) < 3:
        factors += [1] * (3 - len(factors))
    elif len(factors) > 3:
        base = factors[-3:]
        factors = factors[:-3]
        for factor in reversed(factors):  # Use the largest factors first
            min_index = np.argmin(base)
            base[min_index] *= factor
        factors = base

    return sorted(factors)


@simple_test
class SingleNodeBenchmark(GprMaxRegressionTest):
    tags = {"benchmark", "single node", "openmp"}

    omp_threads = parameter([1, 2, 4, 8, 16, 32, 64, 128])
    # domain = parameter([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cpu_freq = parameter([2000000, 2250000])
    time_limit = "8h"
    sourcesdir = "src"
    model = parameter(
        [
            "benchmark_model_10",
            "benchmark_model_15",
            "benchmark_model_20",
            "benchmark_model_30",
            "benchmark_model_40",
            "benchmark_model_50",
            "benchmark_model_60",
            "benchmark_model_70",
            "benchmark_model_80",
        ]
    )

    @run_after("init")
    def setup_env_vars(self):
        self.num_cpus_per_task = self.omp_threads
        self.env_vars["SLURM_CPU_FREQ_REQ"] = self.cpu_freq
        super().setup_env_vars()


@simple_test
class SingleNodeMPIBenchmark(GprMaxRegressionTest):
    tags = {"benchmark", "mpi", "openmp", "single node"}
    mpi_tasks = parameter([1, 2, 4, 8, 16, 32, 64, 128, 256])
    # domain = parameter([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cpu_freq = parameter([2000000, 2250000])
    model = parameter(["benchmark_model_40"])
    sourcesdir = "src"
    time_limit = "1h"

    @run_after("setup")
    def setup_env_vars(self):
        cpus_per_node = self.current_partition.processor.num_cpus
        self.skip_if(
            cpus_per_node < self.mpi_tasks,
            f"Insufficient CPUs per node ({cpus_per_node}) to run test with at least {self.mpi_tasks} processors",
        )

        self.num_cpus_per_task = cpus_per_node // self.mpi_tasks
        self.num_tasks = cpus_per_node // self.num_cpus_per_task
        self.num_tasks_per_node = self.num_tasks
        self.extra_executable_opts = [
            "--mpi",
            *map(str, calculate_mpi_decomposition(self.num_tasks)),
        ]

        self.env_vars["SLURM_CPU_FREQ_REQ"] = self.cpu_freq
        super().setup_env_vars()


@simple_test
class MPIStrongScalingBenchmark(GprMaxRegressionTest):
    tags = {"benchmark", "mpi", "openmp"}

    num_nodes = parameter([1, 2, 4, 8, 16])
    # domain = parameter([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cpu_freq = parameter([2000000, 2250000])
    time_limit = "8h"
    sourcesdir = "src"
    model = parameter(["benchmark_model_40"])

    # serial_dependency = SingleNodeBenchmark
    # mpi_layout = parameter([[1, 1, 1]])  # parameter([[2, 2, 2], [4, 4, 4], [6, 6, 6]])

    def build_reference_filepath(self, suffix: str = "") -> str:
        filename = (
            f"MPIWeakScalingBenchmark_{suffix}" if len(suffix) > 0 else "MPIWeakScalingBenchmark"
        )
        reference_file = Path("regression_checks", filename).with_suffix(".h5")
        return os.path.abspath(reference_file)

    @run_after("setup")
    def setup_env_vars(self):
        cpus_per_node = self.current_partition.processor.num_cpus

        self.num_cpus_per_task = 16
        self.num_tasks_per_node = cpus_per_node // self.num_cpus_per_task
        self.num_tasks = self.num_tasks_per_node * self.num_nodes
        self.extra_executable_opts = [
            "--mpi",
            *map(str, calculate_mpi_decomposition(self.num_tasks)),
        ]

        self.env_vars["SLURM_CPU_FREQ_REQ"] = self.cpu_freq
        super().setup_env_vars()


@simple_test
class MPIWeakScalingBenchmark(GprMaxRegressionTest):
    tags = {"benchmark", "mpi", "openmp"}

    num_nodes = parameter([1, 2, 4, 8, 16])
    # domain = parameter([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    cpu_freq = parameter([2000000, 2250000])
    time_limit = "8h"
    sourcesdir = "src"
    model = parameter(["benchmark_model_40"])

    def build_reference_filepath(self, suffix: str = "") -> str:
        filename = (
            f"MPIStrongScalingBenchmark_{suffix}_{self.num_nodes}"
            if len(suffix) > 0
            else f"MPIStrongScalingBenchmark_{self.num_nodes}"
        )
        reference_file = Path("regression_checks", filename).with_suffix(".h5")
        return os.path.abspath(reference_file)

    @run_after("setup")
    def setup_env_vars(self):
        cpus_per_node = self.current_partition.processor.num_cpus

        self.num_cpus_per_task = 16
        self.num_tasks_per_node = cpus_per_node // self.num_cpus_per_task
        self.num_tasks = self.num_tasks_per_node * self.num_nodes
        size = 0.4
        scale_factor = calculate_mpi_decomposition(self.num_nodes)
        self.prerun_cmds.append(
            f'sed -i "s/#domain: 0.4 0.4 0.4/#domain: {size * scale_factor[0]} {size * scale_factor[1]} {size * scale_factor[2]}/g" {self.model}.in'
        )
        self.extra_executable_opts = [
            "--mpi",
            *map(str, calculate_mpi_decomposition(self.num_tasks)),
        ]

        self.env_vars["SLURM_CPU_FREQ_REQ"] = self.cpu_freq
        super().setup_env_vars()
