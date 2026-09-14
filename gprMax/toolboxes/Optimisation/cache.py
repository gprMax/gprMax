"""Optional reuse of exactly matching deterministic simulation requests.

A cache hit reuses the main HDF5 result and simulation record, not an
objective score. Processing can still apply a different objective.
Matching includes proposed physical parameters, scenario, dependencies,
seed, execution settings and a user-selected solver/environment namespace.
It does not infer equivalent geometry after rounding model dimensions.
Campaign also uses these keys to coalesce duplicates within one batch.
"""

import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

from ._storage import json_copy, sha256, write_json
from .models import RunResult


class SimulationCache:
    """The namespace identifies the user's solver build and Python environment.

    Change namespace after changing the solver/build/environment. Builder source,
    declared dependencies, parameters, scenario, seed and execution configuration
    are additionally included in every key. Only successful HDF5 runs are reused.
    """

    def __init__(self, directory, *, namespace):
        """Create/open a cache under an explicit solver-build/environment namespace."""
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(
                "A cache namespace identifying the solver build/environment is required"
            )
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace

    def to_dict(self):
        """Return cache configuration for the saved campaign definition."""
        return {
            "directory": str(self.directory),
            "namespace": self.namespace,
            "matching": "exact_parameters",
            "deterministic_runs_only": True,
        }

    def key(self, request):
        """Fingerprint an execution request independently of its candidate directory ID.

        Identical designs proposed twice may reuse output. Builder source and
        declared inputs still participate in identity; the namespace must change
        when the solver build/environment changes outside those recorded files.
        """
        module = request["builder"].split(":")[0]
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None or not Path(spec.origin).is_file():
            raise ValueError(
                "Cached builders need a source file and declared helper/data dependencies"
            )
        # Candidate numbering changes between proposals but does not change the
        # simulation. Keep all physical/execution settings in the comparison.
        identity = {k: v for k, v in request.items() if k not in ("candidate_id", "schema_version")}
        identity.update(namespace=self.namespace, builder_sha256=sha256(spec.origin))
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def restore(self, key, task):
        """Copy a verified cached HDF5 into the new task directory, or return None.

        Rewrite candidate/scenario identity for this task and set run_attempts=0.
        Original timings remain labelled as reused timings, not a new solve.
        Missing/corrupt entries act as misses so the task can be simulated.
        """
        entry = self.directory / key
        try:
            record = json.loads((entry / "result.json").read_text())
            digest = record["artifacts"]["output"]["sha256"]
            if record["status"] != "complete" or sha256(entry / "output.h5") != digest:
                return None
        except (OSError, ValueError, KeyError):
            return None
        task.directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(entry / "output.h5", task.directory / "output.h5")
        record = json_copy(record)
        record.update(
            candidate_id=task.request["candidate_id"],
            scenario_id=task.request["scenario"]["id"],
            run_attempts=0,
            cache={"key": key, "entry": str(entry), "hit": True},
        )
        record["artifacts"]["output"]["path"] = "output.h5"
        record["timings"] = {"reused_solver_timings": record.get("timings", {})}
        write_json(task.directory / "request.json", task.request)
        write_json(task.directory / "result.json", record)
        write_json(task.directory / "state.json", {"status": "complete", "cache_hit": True})
        return RunResult(record["candidate_id"], record["scenario_id"], task.directory, record)

    def store(self, key, result):
        """Publish a verified successful HDF5 and result record under the exact key.

        A temporary directory keeps incomplete copies invisible. If another
        coordinator has already published this key, its entry is retained.
        Separate solver sidecars are not part of this cache contract yet.
        """
        if result.status != "complete" or result.output_file is None:
            return
        if sha256(result.output_file) != result.record["artifacts"]["output"]["sha256"]:
            raise ValueError("Cannot cache a simulation with an invalid output hash")
        final = self.directory / key
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.directory))
        try:
            shutil.copy2(result.output_file, temporary / "output.h5")
            record = json_copy(dict(result.record))
            record["artifacts"]["output"]["path"] = "output.h5"
            write_json(temporary / "result.json", record)
            try:
                temporary.rename(final)
            except OSError:
                if not final.is_dir():
                    raise
                # Another coordinator may have committed the same key first.
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
