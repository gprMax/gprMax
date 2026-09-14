"""Save and restore native optimiser state at a batch boundary.

run_optimisation saves before ask so an interrupted batch can be proposed
again and matched to its recorded candidates. Python/NumPy random states
are included alongside library state. This differs from the JSON audit
trail, which records outcomes without restoring the optimiser itself.

Only load checkpoints from your own campaigns: native state uses pickle.
The checksum detects corruption; it does not authenticate a file.
Adapters with external side effects need their own recovery integration.
"""

import hashlib
import os
import pickle
import random
import uuid
from pathlib import Path


def save_checkpoint(directory, state):
    """Atomically save adapter/coordinator state, random states and a payload checksum."""
    import numpy as np

    try:
        import cloudpickle
    except ImportError as exc:
        raise ImportError("Checkpointing requires the optional cloudpickle package") from exc

    state = dict(state, python_random=random.getstate(), numpy_random=np.random.get_state())
    payload = cloudpickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    path = Path(directory) / "checkpoint.bin"
    temporary = path.with_name(f".checkpoint-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(hashlib.sha256(payload).digest())
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(directory):
    """Check payload integrity, restore random generators and return the saved session state."""
    import numpy as np

    data = (Path(directory) / "checkpoint.bin").read_bytes()
    if hashlib.sha256(data[32:]).digest() != data[:32]:
        raise ValueError("Optimiser checkpoint integrity verification failed")
    state = pickle.loads(data[32:])
    random.setstate(state.pop("python_random"))
    np.random.set_state(state.pop("numpy_random"))
    return state
