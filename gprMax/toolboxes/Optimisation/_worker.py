"""Private subprocess entry point that builds and runs one gprMax Scene.

runner.LocalExecutor writes request.json and invokes this module. The
request describes parameters, scenario, builder, dependencies, seed and
execution settings. These are framework records, not user model commands.

main validates the request, calls the builder, runs gprMax and commits
worker_result.json. It does not calculate the optimisation criterion;
processing.py invokes the user's objective after the parent verifies
the result. The current artifact contract covers the main output.h5.
Ordinary model authors edit their callbacks, not this module.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import MappingProxyType

from ._storage import SCHEMA_VERSION, json_copy, sha256, write_json
from .models import BuildContext, PreparedModel, Scenario, check_reference


def _git_identity(root):
    """Record the solver checkout commit and tracked-change flag when available.

    An installed package without .git can still run. This best-effort
    provenance does not inventory untracked files or replace dependency hashes.
    """
    if not (root / ".git").exists():
        return None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        return {"commit": commit, "tracked_changes": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return None


def main(request_path):
    """Consume request.json and return 0 for a committed successful simulation.

    The phase label identifies where an exception occurred (request/import/
    build/solve/output). Worker provenance and timings accompany the result.
    The parent runner performs its own exit-status/hash checks before exposing
    a completed RunResult to objective processing.
    """
    request_path = Path(request_path).resolve()
    workdir = request_path.parent
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "worker_pid": os.getpid(),
        "timings": {},
        "provenance": {"python": sys.executable, "python_version": platform.python_version()},
    }
    phase = "request"
    try:
        # 1. Read the description produced by Campaign; verify declared files
        # before importing the builder or spending time preparing a model.
        request = json.loads(request_path.read_text())
        if request["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported request schema")
        for asset in request["dependencies"]:
            if sha256(asset["path"]) != asset["sha256"]:
                raise ValueError(f"Declared dependency changed: {asset['path']}")
        phase = "import"
        check_reference(request["builder"])
        module_name, name = request["builder"].split(":")
        builder = getattr(importlib.import_module(module_name), name)
        if not callable(builder):
            raise TypeError("The model builder must be callable")
        source_path = inspect.getsourcefile(builder)
        result["provenance"]["builder"] = {
            "reference": request["builder"],
            "source": source_path,
            "sha256": sha256(source_path) if source_path else None,
        }
        import h5py
        import numpy as np

        import gprMax

        result["provenance"]["gprMax"] = {
            "version": gprMax.__version__,
            "module": gprMax.__file__,
            "git": _git_identity(Path(gprMax.__file__).resolve().parent.parent),
        }
        result["provenance"]["numpy"] = np.__version__
        result["provenance"]["h5py"] = h5py.__version__
        random.seed(request["seed"])
        np.random.seed(request["seed"])
        context = BuildContext(
            workdir, workdir / "output", request["seed"], request["execution"]["cpu_threads"]
        )
        scenario = Scenario(**request["scenario"])
        phase = "build"
        started = time.perf_counter()
        # 2. Build one candidate/scenario. The simple API reaches the user
        # build_model(parameters) through the adapter in _simple.py.
        prepared = builder(MappingProxyType(request["parameters"]), scenario, context)
        result["timings"]["builder_seconds"] = time.perf_counter() - started
        if not isinstance(prepared, PreparedModel) or not isinstance(prepared.scene, gprMax.Scene):
            raise TypeError("Builder must return PreparedModel containing a fresh gprMax.Scene")
        # Output location and thread budget are execution contracts, not model parameters.
        threads = [x for x in prepared.scene.single_use_objects if isinstance(x, gprMax.OMPThreads)]
        if len(threads) > 1 or (threads and threads[0].omp_threads != context.cpu_threads):
            raise ValueError("Model OMPThreads conflicts with the allocated CPU thread budget")
        if not threads:
            prepared.scene.add(gprMax.OMPThreads(context.cpu_threads))
        if any(isinstance(x, gprMax.OutputDir) for x in prepared.scene.single_use_objects):
            raise ValueError("Do not use OutputDir; use the builder context output location")
        result["effective_parameters"] = json_copy(dict(prepared.effective_parameters))
        result["model_metadata"] = json_copy(dict(prepared.metadata))
        phase = "solve"
        started = time.perf_counter()
        # 3. Translate the allocated backend into gprMax run arguments.
        # The model supplies physics; the executor supplies device/thread resources.
        solver = request["execution"].get("solver", "cpu")
        accelerator = (
            {}
            if solver == "cpu"
            else {
                {"cuda": "gpu", "opencl": "opencl", "metal": "metal"}[solver]: [
                    request["execution"]["device"]
                ],
                "gpu_precision": request["execution"]["precision"],
            }
        )
        gprMax.run(
            scenes=[prepared.scene],
            n=1,
            outputfile=context.output_stem,
            geometry_fixed=False,
            cpu_precision=request["execution"]["precision"],
            hide_progress_bars=True,
            log_level=30,
            **accelerator,
        )
        result["timings"]["solver_call_seconds"] = time.perf_counter() - started
        phase = "output"
        # 4. Verify the main saved output and describe the artifact for the parent.
        # Do not choose S11, a receiver component or an objective in the worker.
        output = workdir / "output.h5"
        if not output.is_file():
            raise FileNotFoundError(
                "The model did not produce an HDF5 result. Configure the saved outputs you need "
                "(for example, an Rx or a port) in your model-building function."
            )
        with h5py.File(output, "r") as handle:
            dt = float(handle.attrs["dt"])
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("Output contains an invalid time step")
            result["output_summary"] = {
                "dt": dt,
                "shape": [int(x) for x in handle.attrs["nx_ny_nz"]],
                "groups": list(handle.keys()),
            }
        result["artifacts"] = {
            "output": {
                "path": "output.h5",
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            }
        }
        result["status"] = "complete"
    except Exception as exc:
        result["failure"] = {
            "kind": f"{phase}_error",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        traceback.print_exc()
    write_json(workdir / "worker_result.json", result)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
