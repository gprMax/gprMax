"""
Live progress display for gprMax simulations.

Progress comes from parsing gprMax's tqdm output on stdout (not stderr -
checked). No callback API exists for this in gprMax itself.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import subprocess
    import sys
    import threading
    from collections import deque
    from pathlib import Path

    import marimo as mo

    return Path, deque, mo, re, subprocess, sys, threading


@app.cell
def _(deque, threading):
    # background thread writes here directly, not through mo.state
    run_lock = threading.Lock()
    run_state = {
        "proc": None,
        "current": 0,
        "total": 0,
        "running": False,
        "returncode": None,
        "last_lines": deque(maxlen=5),
    }
    return run_lock, run_state


@app.cell
def _(mo):
    get_progress, set_progress = mo.state(
        {
            "current": 0,
            "total": 0,
            "running": False,
            "returncode": None,
            "last_lines": [],
        }
    )
    return get_progress, set_progress


@app.cell
def _(mo):
    input_browser = mo.ui.file_browser(filetypes=[".in"], label="", multiple=True)
    run_button = mo.ui.run_button(label="▶  Run simulation")
    stop_button = mo.ui.run_button(label="■  Stop")

    mo.output.replace(
        mo.vstack(
            [
                mo.md("# gprMax Simulation Progress Tracker"),
                mo.md(
                    "Launches a gprMax simulation and shows live progress "
                    "parsed from its tqdm output."
                ),
                input_browser,
                mo.hstack([run_button, stop_button], gap="1rem", justify="start"),
            ],
            gap="0.4rem",
        )
    )
    return input_browser, run_button, stop_button


@app.cell
def _(
    Path,
    input_browser,
    mo,
    re,
    run_button,
    run_lock,
    run_state,
    stop_button,
    subprocess,
    sys,
    threading,
):
    # requires tqdm's trailing bracket so prose like "Model 1/1" doesn't match
    progress_pattern = re.compile(r"(\d+)/(\d+)\s*\[")

    def launch(in_path: str) -> None:
        cmd = [sys.executable, "-m", "gprMax", in_path, "--show-progress-bars"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            with run_lock:
                run_state["running"] = False
                run_state["returncode"] = -1
                run_state["last_lines"].append(f"Failed to launch: {e}")
            return

        with run_lock:
            run_state["proc"] = proc

        for line in proc.stdout:  # not communicate() — that blocks until exit
            line = line.rstrip("\n")
            if not line:
                continue
            with run_lock:
                run_state["last_lines"].append(line)
            match = progress_pattern.search(line)
            if match:
                with run_lock:
                    run_state["current"] = int(match.group(1))
                    run_state["total"] = int(match.group(2))

        proc.wait()
        with run_lock:
            run_state["running"] = False
            run_state["returncode"] = proc.returncode
            run_state["proc"] = None

    if run_button.value and not run_state["running"]:
        if not input_browser.value:
            mo.output.replace(
                mo.callout(mo.md("Select a `.in` file first."), kind="warn")
            )
        else:
            selected = input_browser.value[0].path
            if not Path(selected).exists():
                mo.output.replace(
                    mo.callout(mo.md(f"File not found: `{selected}`"), kind="danger")
                )
            else:
                with run_lock:
                    run_state["current"] = 0
                    run_state["total"] = 0
                    run_state["running"] = True
                    run_state["returncode"] = None
                    run_state["proc"] = None
                    run_state["last_lines"].clear()
                threading.Thread(target=launch, args=(selected,), daemon=True).start()

    if stop_button.value and run_state["running"]:
        with run_lock:
            proc = run_state["proc"]
        if proc is not None:
            proc.terminate()
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(default_interval="0.2s")
    mo.output.replace(refresh)
    return (refresh,)


@app.cell
def _(get_progress, mo, refresh, run_lock, run_state, set_progress):
    refresh  # dependency only

    with run_lock:
        snapshot = {
            "current": run_state["current"],
            "total": run_state["total"],
            "running": run_state["running"],
            "returncode": run_state["returncode"],
            "last_lines": list(run_state["last_lines"]),
        }
    # Only write when something changed. Setting identical state five times
    # a second keeps the reactive graph busy for the life of the notebook.
    if snapshot != get_progress():
        set_progress(snapshot)
    p = get_progress()

    if not p["running"] and p["returncode"] is None:
        mo.stop(
            True,
            mo.callout(
                mo.md("No simulation running. Select a file and click **Run**."),
                kind="neutral",
            ),
        )

    pct = p["current"] / p["total"] if p["total"] else 0.0

    if p["running"]:
        bar = mo.Html(
            f'<div style="background:#e5e5e5;border-radius:4px;height:10px;'
            f'width:100%;overflow:hidden;">'
            f'<div style="background:#1f77b4;height:100%;'
            f'width:{pct * 100:.1f}%;transition:width 0.2s;"></div></div>'
        )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(f"**Running** — {p['current']} / {p['total']} iterations"),
                    bar,
                ],
                gap="0.3rem",
            )
        )
    elif p["returncode"] == 0:
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"**Done.** {p['current']} / {p['total']} iterations completed."
                ),
                kind="success",
            )
        )
    elif p["returncode"] is not None and p["returncode"] < 0:
        mo.output.replace(mo.callout(mo.md("Simulation stopped."), kind="neutral"))
    else:
        tail = "\n".join(f"`{l}`" for l in p["last_lines"])
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"**gprMax exited with code {p['returncode']}.**\n\n"
                    f"Last output:\n\n{tail}"
                ),
                kind="danger",
            )
        )
    return


if __name__ == "__main__":
    app.run()