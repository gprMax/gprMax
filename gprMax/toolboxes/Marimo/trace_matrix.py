"""
trace_matrix.py: stacks single-trace gprMax HDF5 files into a
position x time matrix — the shared core behind both the B-scan heatmap
and 3D surface views.

No marimo dependency, same design principle as h5_reader.py: pure logic,
testable with plain pytest, importable from any dashboard cell that needs
it. Used by bscan_dashboard.py's live-polling path (one new trace per
tick) and its load-files path (all traces at once).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from gprMax.toolboxes.Marimo.h5_reader import (
    FileData,
    get_time_axis,
    get_trace,
    list_components,
    list_receivers,
)
from gprMax.toolboxes.Utilities.receiver_identity import ReceiverIdentity, match_receiver

ProcessedTrace = dict[str, Any]
# success: {"ok": True, "component", "receiver", "array", "time_ns", "x",
#           "known_components"}
# failure: {"ok": False, "reason", "known_components"}

StackResult = dict[str, Any]
# {"matrix": np.ndarray | None, "positions_x": list[float],
#  "time_ns": np.ndarray | None, "component": str | None,
#  "receiver": str | None, "all_positions_physical": bool,
#  "warnings": list[str]}


def process_trace(
    file_data: FileData,
    preferred_component: str | None,
    expected_len: int | None,
    preferred_receiver: str | None = None,
    expected_time_ns: np.ndarray | None = None,
    expected_identity: ReceiverIdentity | None = None,
) -> ProcessedTrace:
    """Validate and extract one column from a loaded single-trace file.

    `expected_len`, if given, must match the trace's sample count — this
    is what catches a run where dt or iteration count drifted mid-sequence.
    `preferred_receiver` selects a file-local key; a missing explicit key is
    rejected. `expected_identity` anchors later traces to the first selection,
    even when another file assigns that receiver a different key.
    """
    rxs = list_receivers(file_data)
    rx = preferred_receiver if preferred_receiver is not None else (rxs[0] if rxs else "rx1")
    receiver_infos = file_data.get("receivers", {})
    names = Counter(info.get("name", "") for info in receiver_infos.values())
    identities = {
        f"rxs/{key}": info.get(
            "identity",
            ReceiverIdentity(
                path=f"rxs/{key}",
                name=info.get("name", ""),
                name_unique=bool(info.get("name")) and names[info.get("name")] == 1,
                count=len(receiver_infos),
            ),
        )
        for key, info in receiver_infos.items()
    }
    if expected_identity is not None:
        try:
            rx = match_receiver(expected_identity, identities).path.split("/")[-1]
        except ValueError as error:
            return {"ok": False, "reason": str(error), "known_components": []}
    if rx not in rxs:
        return {"ok": False, "reason": f"receiver {rx!r} is missing", "known_components": []}
    comps = list_components(file_data, rx)

    if not comps:
        return {"ok": False, "reason": "no field components found", "known_components": comps}
    if expected_identity is not None and preferred_component not in comps:
        return {"ok": False, "reason": f"component {preferred_component!r} is missing", "known_components": comps}

    comp = (
        preferred_component
        if preferred_component in comps
        else ("Ez" if "Ez" in comps else comps[0])
    )
    arr = get_trace(file_data, comp, rx)
    time_ns = get_time_axis(file_data, unit="ns", receiver=rx, component=comp)

    if expected_len is not None and len(arr) != expected_len:
        return {
            "ok": False,
            "reason": f"{len(arr)} samples vs expected {expected_len}",
            "known_components": comps,
        }

    if expected_time_ns is not None and not np.allclose(
        time_ns, expected_time_ns, rtol=1e-9, atol=1e-15
    ):
        return {
            "ok": False,
            "reason": "sample times differ from the first accepted trace",
            "known_components": comps,
        }

    sources = file_data.get("sources", {})
    x = sources["src1"]["position"][0] if "src1" in sources else None

    return {
        "ok": True,
        "component": comp,
        "receiver": rx,
        "identity": identities.get(f"rxs/{rx}"),
        "array": arr,
        "time_ns": time_ns,
        "x": x,
        "known_components": comps,
    }


def stack_traces(
    file_datas: list[FileData],
    preferred_component: str | None = None,
    preferred_receiver: str | None = None,
) -> StackResult:
    """Stack already-loaded single-trace files into a matrix, in the order
    given. Ordering is the caller's decision — by trace number for a live
    sequential run, by physical position for a hand-picked file set.

    Skips traces with no usable component or a sample-count mismatch
    against the first accepted trace, recording why in `warnings` rather
    than raising, since one bad file shouldn't kill the whole assembly.
    """
    cols: list[np.ndarray] = []
    xpos: list[float] = []
    warnings: list[str] = []
    time_ns = None
    component = preferred_component
    receiver = None
    all_physical = True
    identity = None

    for i, fdata in enumerate(file_datas):
        expected_len = len(cols[0]) if cols else None
        result = process_trace(
            fdata,
            component,
            expected_len,
            preferred_receiver,
            expected_time_ns=time_ns,
            expected_identity=identity,
        )

        if not result["ok"]:
            warnings.append(f"trace {i}: {result['reason']} — skipped")
            continue

        component = result["component"]
        if receiver is None:
            receiver = result["receiver"]
            identity = result["identity"]
        if time_ns is None:
            time_ns = result["time_ns"]

        x = result["x"]
        if x is None:
            all_physical = False

        cols.append(result["array"])
        xpos.append(x if x is not None else float(i))

    return {
        "matrix": np.column_stack(cols) if cols else None,
        "positions_x": xpos,
        "time_ns": time_ns,
        "component": component,
        "receiver": receiver,
        "all_positions_physical": all_physical,
        "warnings": warnings,
    }
