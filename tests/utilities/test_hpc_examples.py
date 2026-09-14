"""Check executable advice in the shipped Grid Engine examples."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.unit
HPC = Path(__file__).resolve().parents[2] / "gprMax" / "toolboxes" / "Utilities" / "HPC"
BASH = shutil.which("bash")


@pytest.mark.skipif(BASH is None, reason="Bash is needed to check the scheduler examples")
@pytest.mark.parametrize("filename", ["gprmax_omp.sh", "gprmax_omp_jobarray.sh", "gprmax_omp_taskfarm.sh"])
def test_grid_engine_examples_are_valid_bash_with_the_release_environment(filename):
    script = HPC / filename
    assert "conda activate gprMax-v4" in script.read_text()
    subprocess.run([BASH, "-n", str(script)], check=True, capture_output=True, timeout=10)


@pytest.mark.skipif(BASH is None, reason="Bash is needed to expand scheduler task variables")
@pytest.mark.parametrize("task_id", ["1", "2", "10"])
def test_job_array_runs_one_model_with_a_task_specific_output(task_id):
    script = (HPC / "gprmax_omp_jobarray.sh").read_text()
    assert re.search(r"^#\$ -t 1-10$", script, re.MULTILINE)
    command = next(line for line in script.splitlines() if line.startswith("python -m gprMax "))
    # Capture the actual expanded solver arguments; do not load site modules
    # or submit a scheduler job in a unit test.
    capture = "python() { printf '%s\\n' \"$@\"; }\n" + command
    result = subprocess.run(
        [BASH, "-c", capture],
        env=dict(os.environ, SGE_TASK_ID=task_id),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.splitlines() == [
        "-m",
        "gprMax",
        "mymodel.in",
        "-n",
        "1",
        "-i",
        task_id,
        "-o",
        f"mymodel_{task_id}",
    ]
