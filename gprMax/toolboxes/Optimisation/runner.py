"""Launch and supervise one isolated gprMax worker subprocess.

LocalExecutor receives a request from Campaign or a pool. It allocates
the attempt directory, writes request.json, sets CPU/device/environment
settings and launches python -m gprMax.toolboxes.Optimisation._worker.

The worker builds/runs the model and writes worker_result.json.
This parent verifies its exit status and output hash, then commits
result.json and returns RunResult. stdout.log/stderr.log remain available
when a build, solve, timeout or launch fails. No objective is computed here.
"""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ._storage import SCHEMA_VERSION, sha256, write_json
from .models import RunResult


def _stop(process):
    """Terminate the worker process group on POSIX and reap it so child solvers cannot linger."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait()


@dataclass(frozen=True)
class LocalExecutor:
    """Settings for one fresh solver subprocess per attempt.

    python selects the interpreter, pythonpath adds import roots and
    cpu_threads bounds OpenMP/BLAS work. timeout is wall-clock seconds for
    the child process. solver selects cpu/cuda/opencl/metal; device is a
    visible accelerator ID and precision is single or double where supported.
    A LocalPool constructs several such executors with distinct allocations.
    """

    python: str = sys.executable
    cpu_threads: int = 1
    timeout: float = 300.0
    precision: str = "single"
    pythonpath: tuple[Path, ...] = ()
    solver: str = "cpu"
    device: int | None = None

    def __post_init__(self):
        """Validate the interpreter, timeout, thread budget and solver/device combination."""
        if (
            isinstance(self.cpu_threads, bool)
            or not isinstance(self.cpu_threads, int)
            or self.cpu_threads < 1
        ):
            raise ValueError("cpu_threads must be a positive integer")
        if isinstance(self.timeout, bool) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if self.precision not in ("single", "double"):
            raise ValueError("precision must be single or double")
        if self.solver not in ("cpu", "cuda", "opencl", "metal"):
            raise ValueError("solver must be cpu, cuda, opencl or metal")
        if self.solver == "cpu" and self.device is not None:
            raise ValueError("CPU execution does not take a device")
        if self.solver != "cpu" and (
            isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0
        ):
            raise ValueError("Accelerator execution requires a nonnegative visible device ID")
        if self.solver == "metal" and (self.device != 0 or self.precision != "single"):
            raise ValueError("This gprMax Metal backend supports device 0 and single precision")
        interpreter = Path(self.python).expanduser().absolute()
        if not interpreter.is_file():
            raise ValueError(f"Python interpreter does not exist: {interpreter}")
        object.__setattr__(self, "python", str(interpreter))
        object.__setattr__(
            self, "pythonpath", tuple(Path(p).expanduser().resolve() for p in self.pythonpath)
        )

    def to_dict(self):
        """Describe execution settings for the worker request and reproducibility record."""
        return {
            "python": self.python,
            "cpu_threads": self.cpu_threads,
            "timeout": self.timeout,
            "precision": self.precision,
            "pythonpath": [str(p) for p in self.pythonpath],
            "solver": self.solver,
            "device": self.device,
        }

    def run(self, request, directory, *, cancel_event=None):
        """Launch one request, wait within the timeout and return its committed RunResult.

        directory is a new attempt directory. cancel_event is used by local
        pools to stop child processes on interruption. Failed outputs are never
        presented as complete merely because an output.h5 happens to exist.
        """
        request = dict(request, execution=self.to_dict())
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "request.json", request)
        env = os.environ.copy()
        # A farm worker launches an independent solver, outside its MPI world.
        for name in tuple(env):
            if name.startswith(("OMPI_", "PMI_", "PMIX_", "MPI_LOCAL")):
                env.pop(name)
        # The directory containing gprMax, both in a checkout and site-packages.
        package_root = Path(__file__).resolve().parents[3]
        paths = [str(package_root), *(str(p) for p in self.pythonpath)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Honour explicit caller caches: rebuilding a font cache for every
        # candidate can cost much more than a small FDTD solve.
        env.setdefault("MPLCONFIGDIR", str(directory / ".cache" / "matplotlib"))
        env.setdefault("XDG_CACHE_HOME", str(directory / ".cache"))
        env["MPLBACKEND"] = "Agg"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = str(self.cpu_threads)
        started = time.perf_counter()
        process = None
        failure = None
        cancelled = False
        write_json(
            directory / "state.json",
            {"status": "running", "host": socket.gethostname(), "execution": self.to_dict()},
        )
        write_json(
            directory.parent / "task.json",
            {
                "status": "running",
                "host": socket.gethostname(),
                "execution": self.to_dict(),
                "attempt": directory.name,
            },
        )
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open(
            "w"
        ) as stderr:
            try:
                process = subprocess.Popen(
                    [
                        self.python,
                        "-m",
                        "gprMax.toolboxes.Optimisation._worker",
                        str(directory / "request.json"),
                    ],
                    cwd=directory,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=os.name == "posix",
                )
                deadline = time.monotonic() + self.timeout
                while process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        _stop(process)
                        failure = {"kind": "cancelled", "message": "Execution pool cancelled"}
                        cancelled = True
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(process.args, self.timeout)
                    try:
                        process.wait(timeout=min(0.1, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            except subprocess.TimeoutExpired:
                _stop(process)
                failure = {"kind": "timeout", "message": f"Exceeded {self.timeout} seconds"}
            except KeyboardInterrupt:
                if process is not None:
                    _stop(process)
                cancelled = True
                failure = {"kind": "cancelled", "message": "Interrupted by user"}
            except OSError as exc:
                failure = {"kind": "launch_error", "message": str(exc)}
        # A child can exit before writing a record. Combine whatever it committed
        # with the parent's timeout/return-code evidence before declaring success.
        try:
            record = json.loads((directory / "worker_result.json").read_text())
        except (OSError, ValueError):
            record = {}
        record.update(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": request["candidate_id"],
                "scenario_id": request["scenario"]["id"],
                "returncode": process.returncode if process else None,
            }
        )
        record.setdefault("timings", {})["worker_wall_seconds"] = time.perf_counter() - started
        record["execution"] = self.to_dict()
        record["host"] = socket.gethostname()
        record["run_attempts"] = 1
        if not failure and process.returncode != 0:
            failure = record.get("failure") or {
                "kind": "worker_exit",
                "message": f"Worker exited {process.returncode}",
            }
        if not failure and record.get("status") != "complete":
            failure = record.get("failure") or {
                "kind": "missing_result",
                "message": "Worker did not commit completion",
            }
        if not failure:
            output = directory / "output.h5"
            artifact = record.get("artifacts", {}).get("output", {})
            if not output.is_file() or sha256(output) != artifact.get("sha256"):
                failure = {
                    "kind": "invalid_output",
                    "message": "Output is missing or failed integrity verification",
                }
        if failure:
            record.update(status="cancelled" if cancelled else "failed", failure=failure)
            record.pop("artifacts", None)
        write_json(directory / "result.json", record)
        write_json(
            directory / "state.json", {"status": record["status"], "host": socket.gethostname()}
        )
        if cancelled and cancel_event is None:
            raise KeyboardInterrupt
        return RunResult(request["candidate_id"], request["scenario"]["id"], directory, record)
