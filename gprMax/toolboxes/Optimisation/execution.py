"""Schedule independent simulation tasks on local or MPI workers.

RunTask holds one request plus its attempt directory. LocalPool assigns
slots to CPU workers or selected GPUs; MPIPool uses an existing MPI
allocation with rank 0 coordinating. Each slot launches LocalExecutor,
which runs an isolated _worker subprocess. No Scene/solver state is shared.

_dispatch keeps only one task per slot in flight, reports completed
results to the coordinator, and returns results in submission order.
Parallelism is across candidate/scenario simulations, not across
individual parameters or pieces of one FDTD model.
"""

import json
import os
import socket
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ._storage import json_copy, write_json
from .models import RunResult
from .runner import LocalExecutor


@dataclass(frozen=True)
class RunTask:
    """One candidate/scenario request and its initial attempt directory.

    request is JSON-compatible data produced by Campaign._prepare. Retries
    may use a new sibling directory while retaining this task's identity.
    """

    request: dict
    directory: Path

    def __post_init__(self):
        """Detach the request and make the output directory absolute before dispatch."""
        object.__setattr__(self, "request", json_copy(self.request))
        object.__setattr__(self, "directory", Path(self.directory).resolve())


class ExecutionBackend(Protocol):
    """Minimal execution service used by Campaign.

    map returns one RunResult per task in input order, including failures
    and cancellations. on_result callbacks run on the coordinator as results
    arrive. to_dict describes the configuration saved with the campaign.
    """

    def to_dict(self) -> dict:
        """Return JSON-compatible resource and execution settings."""
        ...

    def map(self, tasks, *, stop_on_failure=False, on_result=None) -> tuple[RunResult, ...]:
        """Execute tasks, report completion centrally and preserve input ordering."""
        ...


def _integer(value, name):
    """Validate a positive worker/thread count without accepting a boolean."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _retry_limit(value):
    """Validate how many extra attempts may follow the first attempt."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    return value


def _run_attempts(executor, task, max_retries, cancel_event=None):
    """Retry infrastructure failures only, keeping each attempt's own artifacts."""
    initial = (
        int(task.directory.name.split("-")[-1]) if task.directory.name.startswith("attempt-") else 1
    )
    directories = []
    # One initial execution plus the allowed retries. Each directory preserves
    # its own logs; only timeout, process exit and launch errors qualify for retry.
    for index in range(max_retries + 1):
        directory = (
            task.directory
            if index == 0
            else task.directory.with_name(f"attempt-{initial + index:04d}")
        )
        result = executor.run(task.request, directory, cancel_event=cancel_event)
        directories.append(str(directory))
        retryable = (result.record.get("failure") or {}).get("kind") in (
            "timeout",
            "worker_exit",
            "launch_error",
        )
        if result.status != "failed" or not retryable:
            break
    record = dict(result.record, run_attempts=len(directories), attempt_directories=directories)
    # Custom executor fixtures need not create artifacts; real workers do.
    if result.directory.exists():
        write_json(result.directory / "result.json", record)
    return RunResult(result.candidate_id, result.scenario_id, result.directory, record)


def _unsolved(task, kind="batch_aborted", message="Earlier task failed", *, attempted=False):
    """Write an explicit result for work that did not produce a solver result.

    attempted=False represents queued work cancelled before launch and costs
    zero attempts. Executor errors after submission use attempted=True.
    """
    task.directory.mkdir(parents=True, exist_ok=True)
    record = {
        "candidate_id": task.request["candidate_id"],
        "scenario_id": task.request["scenario"]["id"],
        "status": "failed" if attempted else "cancelled",
        "run_attempts": int(attempted),
        "failure": {"kind": kind, "message": message},
    }
    write_json(task.directory / "request.json", task.request)
    write_json(task.directory / "result.json", record)
    write_json(task.directory / "state.json", {"status": record["status"]})
    return RunResult(record["candidate_id"], record["scenario_id"], task.directory, record)


def _dispatch(tasks, slots, submit, *, stop_on_failure, on_result):
    """Keep one task per slot running and return results in submission order.

    pending maps each Future to (task index, slot). Completion order can
    differ from input order; results[index] preserves the association.
    On stop_on_failure, running tasks finish and queued tasks are cancelled.
    """
    tasks, slots = tuple(tasks), list(slots)
    results, pending = [None] * len(tasks), {}
    next_index, stopped = 0, False
    while next_index < len(tasks) or pending:
        while slots and next_index < len(tasks) and not stopped:
            index, next_index = next_index, next_index + 1
            slot = slots.pop(0)
            pending[submit(slot, tasks[index])] = (index, slot)
        if not pending:
            break
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            index, slot = pending.pop(future)
            try:
                result = future.result()
                task = tasks[index]
                if (
                    not isinstance(result, RunResult)
                    or result.candidate_id != task.request["candidate_id"]
                    or result.scenario_id != task.request["scenario"]["id"]
                    or result.directory.parent != task.directory.parent
                ):
                    raise ValueError("Executor returned feedback for a different task")
            except Exception as exc:
                result = _unsolved(tasks[index], "executor_error", str(exc), attempted=True)
            results[index] = result
            if on_result:
                on_result(result)
            stopped |= stop_on_failure and result.status != "complete"
            # Release this CPU/GPU allocation only after its result is collected.
            # The next loop can then start another simulation on the same slot.
            slots.append(slot)
    for index in range(next_index, len(tasks)):
        results[index] = _unsolved(tasks[index])
        if on_result:
            on_result(results[index])
    return tuple(results)


class LocalPool:
    """Run independent CPU/GPU simulations with bounded concurrency.

    CPU pools use workers and cpu_threads_per_worker. GPU pools require
    distinct visible device IDs and one worker per selected device. Their
    CPU thread budgets still apply to preparation and supporting work.
    Timeout/launch/worker-exit failures can be retried with max_retries;
    model or objective errors are not silently retried.
    """

    def __init__(
        self,
        *,
        solver="cpu",
        workers=None,
        devices=None,
        cpu_threads_per_worker=1,
        precision="single",
        timeout=300,
        python=sys.executable,
        pythonpath=(),
        allow_oversubscription=False,
        max_retries=0,
    ):
        """Validate resources and construct one serial subprocess executor per worker slot."""
        self.max_attempts = _retry_limit(max_retries) + 1
        devices = tuple(devices or ())
        if solver == "cpu":
            if devices:
                raise ValueError("CPU pools do not take GPU devices")
            self.workers = _integer(1 if workers is None else workers, "workers")
            assigned = [None] * self.workers
        else:
            if not devices or len(set(devices)) != len(devices):
                raise ValueError("Select a nonempty list of distinct visible GPU device IDs")
            self.workers = len(devices) if workers is None else _integer(workers, "workers")
            if self.workers != len(devices):
                raise ValueError("Use one worker per selected GPU")
            assigned = devices
        cpu_threads_per_worker = _integer(cpu_threads_per_worker, "cpu_threads_per_worker")
        available = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        if not allow_oversubscription and self.workers * cpu_threads_per_worker > available:
            raise ValueError(
                f"Requested CPU threads exceed {available} visible CPUs; allocate fewer or explicitly allow oversubscription"
            )
        self.executors = tuple(
            LocalExecutor(
                python=python,
                solver=solver,
                device=device,
                cpu_threads=cpu_threads_per_worker,
                precision=precision,
                timeout=timeout,
                pythonpath=tuple(pythonpath),
            )
            for device in assigned
        )
        self.configuration = {
            "backend": "local",
            "solver": solver,
            "workers": self.workers,
            "devices": list(devices),
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "precision": precision,
            "timeout": timeout,
            "python": self.executors[0].python,
            "pythonpath": [str(p) for p in self.executors[0].pythonpath],
            "allow_oversubscription": allow_oversubscription,
            "max_retries": max_retries,
        }

    def to_dict(self):
        """Return the allocation saved in campaign/request records."""
        return json_copy(self.configuration)

    def map(self, tasks, *, stop_on_failure=False, on_result=None):
        """Use coordinating threads to keep isolated solver subprocesses busy.

        Threads schedule processes; they do not run shared gprMax Scenes.
        A raised interruption sets cancellation so local child solvers stop.
        """
        cancelled = threading.Event()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:

            def submit(index, task):
                """Associate an available slot with its executor and retry policy."""
                return pool.submit(
                    _run_attempts, self.executors[index], task, self.max_attempts - 1, cancelled
                )

            try:
                return _dispatch(
                    tasks,
                    range(self.workers),
                    submit,
                    stop_on_failure=stop_on_failure,
                    on_result=on_result,
                )
            except BaseException:
                cancelled.set()
                raise

    def run(self, request, directory):
        """Adapt one request to the same pool machinery used for a batch."""
        return self.map((RunTask(request, directory),))[0]

    def __enter__(self):
        """Allow the same with-block calling pattern as MPIPool; resources are created by map."""
        return self

    def __exit__(self, *args):
        """Propagate exceptions; each map call already closes its coordinating thread pool."""
        return False


def _mpi_run(task, configuration, assignments):
    """Execute on an MPI worker using that rank's allocated visible device.

    LocalExecutor starts a separate solver process and strips inherited MPI
    launcher variables. The MPI rank transports tasks/results; it does not
    become a rank in a distributed FDTD simulation.
    """
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    device = assignments[rank]
    executor = LocalExecutor(
        python=configuration["python"],
        solver=configuration["solver"],
        device=device,
        cpu_threads=configuration["cpu_threads_per_worker"],
        precision=configuration["precision"],
        timeout=configuration["timeout"],
        pythonpath=tuple(Path(p) for p in configuration["pythonpath"]),
    )
    result = _run_attempts(executor, task, configuration["max_retries"])
    record = dict(result.record, mpi_rank=rank)
    write_json(result.directory / "result.json", record)
    return RunResult(result.candidate_id, result.scenario_id, result.directory, record)


class MPIPool:
    """Pool over an existing MPI allocation, with one coordinator rank.

    Enter on every rank: ``with MPIPool(...) as execution:``. Worker ranks yield
    None after servicing the coordinator. Shared paths and the same Python
    environment must be accessible on every node. No dynamic MPI spawn.
    """

    def __init__(
        self,
        *,
        solver="cpu",
        cpu_threads_per_worker=1,
        precision="single",
        timeout=300,
        python=sys.executable,
        pythonpath=(),
        device_by_rank=None,
        max_retries=0,
    ):
        """Record shared execution settings; the collective with-block establishes the MPI pool."""
        self.max_attempts = _retry_limit(max_retries) + 1
        _integer(cpu_threads_per_worker, "cpu_threads_per_worker")
        LocalExecutor(
            python=python,
            cpu_threads=cpu_threads_per_worker,
            precision=precision,
            timeout=timeout,
            solver=solver,
            device=None if solver == "cpu" else 0,
        )
        self.configuration = {
            "backend": "mpi",
            "solver": solver,
            "cpu_threads_per_worker": cpu_threads_per_worker,
            "precision": precision,
            "timeout": timeout,
            "python": str(Path(python).resolve()),
            "pythonpath": [str(Path(p).resolve()) for p in pythonpath],
            "max_retries": max_retries,
        }
        self.explicit_devices = (
            None if device_by_rank is None else {int(k): v for k, v in device_by_rank.items()}
        )
        self._pool = None

    def __enter__(self):
        """Join all ranks, validate device assignments and enter MPICommExecutor.

        Rank 0 receives this pool and may create a campaign. Other ranks service
        work, then receive None, so example code guards with execution is not None.
        Device IDs are interpreted in each worker's visible GPU namespace.
        """
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor

        comm = MPI.COMM_WORLD
        if comm.size < 2:
            raise ValueError("MPIPool requires mpiexec with a coordinator and at least one worker")
        # A visible device number is local to a host and its visibility settings.
        # Use both when detecting accidental GPU sharing between MPI workers.
        hosts = comm.allgather((socket.gethostname(), os.environ.get("CUDA_VISIBLE_DEVICES")))
        self.assignments = {}
        for rank in range(1, comm.size):
            if self.configuration["solver"] == "cpu":
                device = None
            else:
                if self.explicit_devices is None or set(self.explicit_devices) != set(
                    range(1, comm.size)
                ):
                    raise ValueError(
                        "GPU MPI pools require device_by_rank for every worker, using each rank's visible device IDs"
                    )
                device = self.explicit_devices[rank]
                LocalExecutor(
                    solver=self.configuration["solver"],
                    device=device,
                    precision=self.configuration["precision"],
                )
            self.assignments[rank] = device
        # Detect accidental sharing within a host/visibility namespace.
        if self.configuration["solver"] != "cpu":
            allocation = [(hosts[r][0], hosts[r][1], d) for r, d in self.assignments.items()]
            if len(set(allocation)) != len(allocation):
                raise ValueError(
                    "MPI workers would share a GPU; assign distinct devices or scheduler-isolated visibility"
                )
        self.workers = comm.size - 1
        self.configuration.update(workers=self.workers, device_by_rank=self.assignments)
        self._context = MPICommExecutor(comm, root=0)
        self._pool = self._context.__enter__()
        return self if self._pool is not None else None

    def __exit__(self, *args):
        """Close the collective MPI executor and discard the active pool handle."""
        try:
            return self._context.__exit__(*args)
        finally:
            self._pool = None

    def to_dict(self):
        """Return shared settings plus worker/device assignments once the pool is entered."""
        return json_copy(self.configuration)

    def map(self, tasks, *, stop_on_failure=False, on_result=None):
        """Submit work from the coordinator only, preserving the common ordered-result contract."""
        if self._pool is None:
            raise RuntimeError(
                "Enter MPIPool collectively and create the campaign only on the coordinator"
            )

        def submit(slot, task):
            """Submit to the MPI pool; the receiving rank chooses its device in _mpi_run."""
            return self._pool.submit(_mpi_run, task, self.configuration, self.assignments)

        return _dispatch(
            tasks, range(self.workers), submit, stop_on_failure=stop_on_failure, on_result=on_result
        )

    def run(self, request, directory):
        """Run one request through the entered MPI pool and return its sole result."""
        return self.map((RunTask(request, directory),))[0]


def execution_from_profile(filename):
    """Read a JSON execution configuration. MPI profiles must be used in a with block."""
    specification = json.loads(Path(filename).read_text())
    backend = specification.pop("backend")
    if backend == "local":
        return LocalPool(**specification)
    if backend == "mpi":
        return MPIPool(**specification)
    raise ValueError("Profile backend must be local or mpi")
