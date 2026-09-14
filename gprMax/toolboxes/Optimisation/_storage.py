"""Small persistence helpers for the toolbox's JSON and artifact records.

JSON is used for parameters, settings and progress; numerical output
arrays remain in HDF5/NPZ files. Content hashes link a result to the
exact file that was produced. Atomic replacement prevents readers from
observing half-written JSON. These helpers do not define physical units
or any required objective fields.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path

SCHEMA_VERSION = 1


def json_copy(value):
    """Copy a value through JSON to validate the cross-process record format.

    This also detaches mutable dictionaries/lists from the caller. NaN and
    infinity are rejected: nonfinite solver data belongs in arrays with
    validity information, not ambiguous JSON settings or objective records.
    """
    return json.loads(json.dumps(value, allow_nan=False))


def write_json(path, value):
    """Commit one JSON record by writing and syncing a temporary sibling first.

    os.replace makes the new record visible in one step. The destination's
    parent must exist; this helper replaces a record, not a campaign folder.
    """
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    """Return a file-content digest while reading in bounded chunks rather than loading a large artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
