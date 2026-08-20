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
    import numpy as np
    import plotly.graph_objects as go

    from toolboxes.Marimo.h5_reader import (
        format_metadata_text,
        get_time_axis,
        get_trace,
        get_unit_label,
        list_components,
        list_receivers,
        load_file,
    )
    from toolboxes.Marimo.hyperbola import C_M_PER_NS, ricker_delay, travel_time
    from toolboxes.Marimo.processing import apply_gain, gain_label

    # The model is fixed apart from what the sliders control. These are the
    # dimensions of the standard 2D examples and are not worth exposing.
    DOMAIN_X, DOMAIN_Y, DOMAIN_Z = 0.240, 0.210, 0.002
    DX = 0.002
    SURFACE_Y = 0.170          # top of the half_space box
    TARGET_X = 0.120           # cylinder always sits mid-domain

    return (
        C_M_PER_NS,
        DOMAIN_X,
        DOMAIN_Y,
        DOMAIN_Z,
        DX,
        Path,
        SURFACE_Y,
        TARGET_X,
        apply_gain,
        deque,
        format_metadata_text,
        gain_label,
        get_time_axis,
        get_trace,
        get_unit_label,
        go,
        list_components,
        list_receivers,
        load_file,
        mo,
        np,
        re,
        ricker_delay,
        subprocess,
        sys,
        threading,
        travel_time,
    )


# ── Step 1: Model parameters ──────────────────────────────────────────────
# Widget creation only. Their values are read in the next cell, which builds
# the input file.
@app.cell
def _(Path, mo):
    permittivity = mo.ui.slider(
        start=1.0, stop=20.0, step=0.5, value=6.0,
        label="Soil relative permittivity", debounce=True,
    )
    frequency = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.5,
        label="Antenna centre frequency (GHz)", debounce=True,
    )
    src_x = mo.ui.slider(
        start=0.020, stop=0.200, step=0.002, value=0.040,
        label="Source x (m)", debounce=True,
    )
    separation = mo.ui.slider(
        start=0.010, stop=0.100, step=0.002, value=0.040,
        label="Tx-Rx separation (m)", debounce=True,
    )
    target_depth = mo.ui.slider(
        start=0.020, stop=0.150, step=0.002, value=0.090,
        label="Target depth (m)", debounce=True,
    )
    target_radius = mo.ui.slider(
        start=0.004, stop=0.030, step=0.002, value=0.010,
        label="Target radius (m)", debounce=True,
    )
    time_window = mo.ui.slider(
        start=1.0, stop=12.0, step=0.5, value=3.0,
        label="Time window (ns)", debounce=True,
    )

    out_dir = mo.ui.text(
        value=str(Path.cwd() / "recipe_output"),
        label="Output directory",
        full_width=True,
    )
    model_name = mo.ui.text(value="recipe_ascan", label="Model name")

    mo.output.replace(
        mo.vstack(
            [
                mo.md("# A-scan End-to-End Workflow"),
                mo.md(
                    "One notebook from parameters to waveform: set up a model, write "
                    "the gprMax input file, run the solver while watching its progress, "
                    "then read the output back and inspect it. The reflection arrival "
                    "is predicted from the geometry before the solver runs, so the "
                    "result can be checked against what the physics says it should be."
                ),
                mo.md("---"),
                mo.md("### Step 1 — Model parameters"),
                mo.md(
                    "_A metal cylinder buried in a dielectric half-space, the same "
                    "arrangement as `examples/cylinder_Ascan_2D.in`. The domain is "
                    "0.240 x 0.210 m at 2 mm resolution with the surface at "
                    "y = 0.170 m and the target at x = 0.120 m._"
                ),
                mo.hstack([permittivity, frequency, time_window], gap="1.5rem", justify="start"),
                mo.hstack([src_x, separation], gap="1.5rem", justify="start"),
                mo.hstack([target_depth, target_radius], gap="1.5rem", justify="start"),
                mo.md("**Where to write the model and its output**"),
                out_dir,
                model_name,
            ],
            gap="0.4rem",
        )
    )
    return (
        frequency,
        model_name,
        out_dir,
        permittivity,
        separation,
        src_x,
        target_depth,
        target_radius,
        time_window,
    )


# ── Step 2: Input file and predicted arrivals ─────────────────────────────
@app.cell
def _(
    C_M_PER_NS,
    DOMAIN_X,
    DOMAIN_Y,
    DOMAIN_Z,
    DX,
    SURFACE_Y,
    TARGET_X,
    frequency,
    mo,
    permittivity,
    ricker_delay,
    separation,
    src_x,
    target_depth,
    target_radius,
    time_window,
    travel_time,
):
    _eps = permittivity.value
    _freq = frequency.value
    _depth = target_depth.value
    _radius = target_radius.value
    _sep = separation.value
    _sx = src_x.value
    _rx = round(_sx + _sep, 3)
    _cyl_y = round(SURFACE_Y - _depth, 3)
    _tw = time_window.value

    in_text = "\n".join(
        [
            "#title: A-scan from a metal cylinder buried in a dielectric half-space",
            f"#domain: {DOMAIN_X:.3f} {DOMAIN_Y:.3f} {DOMAIN_Z:.3f}",
            f"#dx_dy_dz: {DX:.3f} {DX:.3f} {DX:.3f}",
            f"#time_window: {_tw:g}e-9",
            "",
            f"#material: {_eps:g} 0 1 0 half_space",
            "",
            f"#waveform: ricker 1 {_freq:g}e9 my_ricker",
            f"#hertzian_dipole: z {_sx:.3f} {SURFACE_Y:.3f} 0 my_ricker",
            f"#rx: {_rx:.3f} {SURFACE_Y:.3f} 0",
            "",
            f"#box: 0 0 0 {DOMAIN_X:.3f} {SURFACE_Y:.3f} {DOMAIN_Z:.3f} half_space",
            f"#cylinder: {TARGET_X:.3f} {_cyl_y:.3f} 0 {TARGET_X:.3f} {_cyl_y:.3f} "
            f"{DOMAIN_Z:.3f} {_radius:.3f} pec",
        ]
    )

    # Predicted from the geometry, before anything is run.
    _delay = ricker_delay(_freq * 1e9)
    predicted = {
        "delay": _delay,
        "direct": _sep / C_M_PER_NS + _delay,
        "reflection": float(
            travel_time(_sx, TARGET_X, _depth, _eps, _sep, _radius, _delay)
        ),
        "window": _tw,
    }

    _problems = []
    if _depth <= _radius:
        _problems.append(
            f"The target is {_depth * 1000:.0f} mm deep with a "
            f"{_radius * 1000:.0f} mm radius, so it breaks the surface."
        )
    if _rx > DOMAIN_X:
        _problems.append(f"The receiver at x = {_rx:.3f} m is outside the domain.")
    if predicted["reflection"] > _tw:
        _problems.append(
            f"The reflection is predicted at {predicted['reflection']:.3f} ns but the "
            f"time window is only {_tw:g} ns, so it will not be recorded. Lengthen "
            "the window, reduce the depth, or reduce the permittivity."
        )

    _rows = [
        f"| Source pulse leaves at | {_delay:.3f} ns (ricker delay, sqrt(2)/f) |",
        f"| Direct wave expected near | {predicted['direct']:.3f} ns |",
        f"| Cylinder reflection expected at | {predicted['reflection']:.3f} ns |",
        f"| Record length | {_tw:g} ns |",
    ]

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 2 — Input file"),
                mo.md(f"```text\n{in_text}\n```"),
                mo.md("**Predicted from the geometry, before running anything**"),
                mo.md("| | |\n|---|---|\n" + "\n".join(_rows)),
                *(
                    [mo.callout(mo.md("\n\n".join(_problems)), kind="warn")]
                    if _problems
                    else []
                ),
                mo.md("---"),
            ],
            gap="0.4rem",
        )
    )
    return in_text, predicted


# ── Run state: plain dict, written by the background thread ───────────────
# Not mo.state. The reader thread cannot call a marimo setter safely, so it
# writes here under a lock and a timer cell mirrors it into reactive state.
@app.cell
def _(deque, threading):
    run_lock = threading.Lock()
    run_state = {
        "proc": None,
        "current": 0,
        "total": 0,
        "running": False,
        "returncode": None,
        "last_lines": deque(maxlen=6),
        "input_path": None,
    }
    return run_lock, run_state


# ── Progress mirror: isolated state, no UI dependencies ───────────────────
@app.cell
def _(mo):
    get_progress, set_progress = mo.state(
        {
            "current": 0,
            "total": 0,
            "running": False,
            "returncode": None,
            "last_lines": [],
            "input_path": None,
        }
    )
    return get_progress, set_progress


# ── Step 3: Run controls ──────────────────────────────────────────────────
@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="▶  Write and run")
    stop_button = mo.ui.run_button(label="■  Stop")
    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 3 — Run the simulation"),
                mo.md(
                    "_Writes the input file above, then launches "
                    "`python -m gprMax` on it. A 3 ns 2D model of this size takes a "
                    "few seconds._"
                ),
                mo.hstack([run_button, stop_button], gap="1rem", justify="start"),
            ],
            gap="0.4rem",
        )
    )
    return run_button, stop_button


# ── Launch handler ────────────────────────────────────────────────────────
@app.cell
def _(
    Path,
    in_text,
    mo,
    model_name,
    out_dir,
    re,
    run_button,
    run_lock,
    run_state,
    stop_button,
    subprocess,
    sys,
    threading,
):
    # Requires tqdm's trailing bracket. A bare \d+/\d+ also matches gprMax's
    # own "Model 1/1" and "Writing geometry view file 1/1" status lines.
    progress_pattern = re.compile(r"(\d+)/(\d+)\s*\[")

    def launch(in_path: str) -> None:
        cmd = [sys.executable, "-m", "gprMax", in_path, "--show-progress-bars"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # tqdm writes to stdout here, checked
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

        for line in proc.stdout:  # not communicate(), which blocks until exit
            line = line.rstrip("\n")
            if not line:
                continue
            with run_lock:
                run_state["last_lines"].append(line)
            m = progress_pattern.search(line)
            if m:
                with run_lock:
                    run_state["current"] = int(m.group(1))
                    run_state["total"] = int(m.group(2))

        proc.wait()
        with run_lock:
            run_state["running"] = False
            run_state["returncode"] = proc.returncode
            run_state["proc"] = None

    if run_button.value and not run_state["running"]:
        _stem = (model_name.value or "recipe_ascan").strip()
        _dir = Path(out_dir.value).expanduser()
        try:
            _dir.mkdir(parents=True, exist_ok=True)
            _path = _dir / f"{_stem}.in"
            _path.write_text(in_text + "\n")
        except OSError as _e:
            mo.output.replace(
                mo.callout(
                    mo.md(f"**Could not write the input file.** `{_e}`"), kind="danger"
                )
            )
        else:
            with run_lock:
                run_state["current"] = 0
                run_state["total"] = 0
                run_state["running"] = True
                run_state["returncode"] = None
                run_state["proc"] = None
                run_state["input_path"] = str(_path)
                run_state["last_lines"].clear()
            threading.Thread(target=launch, args=(str(_path),), daemon=True).start()
            mo.output.replace(mo.md(f"_Wrote `{_path}`_"))

    if stop_button.value and run_state["running"]:
        with run_lock:
            _proc = run_state["proc"]
        if _proc is not None:
            _proc.terminate()
    return


# ── Timer ─────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    refresh = mo.ui.refresh(default_interval="0.2s")
    mo.output.replace(refresh)
    return (refresh,)


# ── Progress display ──────────────────────────────────────────────────────
@app.cell
def _(get_progress, mo, refresh, run_lock, run_state, set_progress):
    refresh  # timer dependency only

    with run_lock:
        _snapshot = {
            "current": run_state["current"],
            "total": run_state["total"],
            "running": run_state["running"],
            "returncode": run_state["returncode"],
            "last_lines": list(run_state["last_lines"]),
            "input_path": run_state["input_path"],
        }

    # Only write when something changed. Setting identical state on every tick
    # would re-run the loader below forever, re-reading the output file four
    # times a second for the life of the notebook.
    if _snapshot != get_progress():
        set_progress(_snapshot)

    _p = get_progress()

    if not _p["running"] and _p["returncode"] is None:
        mo.stop(
            True,
            mo.callout(
                mo.md("No simulation has been run yet. Press **Write and run**."),
                kind="neutral",
            ),
        )

    _pct = _p["current"] / _p["total"] if _p["total"] else 0.0

    if _p["running"]:
        _bar = mo.Html(
            f'<div style="background:#e5e5e5;border-radius:4px;height:10px;'
            f'width:100%;overflow:hidden;">'
            f'<div style="background:#1f77b4;height:100%;'
            f'width:{_pct * 100:.1f}%;transition:width 0.2s;"></div></div>'
        )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(f"**Running** — {_p['current']} / {_p['total']} iterations"),
                    _bar,
                ],
                gap="0.3rem",
            )
        )
    elif _p["returncode"] == 0:
        mo.output.replace(
            mo.callout(
                mo.md(f"**Done.** {_p['current']} / {_p['total']} iterations completed."),
                kind="success",
            )
        )
    elif _p["returncode"] is not None and _p["returncode"] < 0:
        mo.output.replace(mo.callout(mo.md("Simulation stopped."), kind="neutral"))
    else:
        _tail = "\n".join(f"`{line}`" for line in _p["last_lines"])
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"**gprMax exited with code {_p['returncode']}.**\n\n"
                    f"Last output:\n\n{_tail}"
                ),
                kind="danger",
            )
        )
    return


# ── Step 4: Read the output back ──────────────────────────────────────────
@app.cell
def _(Path, format_metadata_text, get_progress, load_file, mo):
    _p = get_progress()
    if _p["returncode"] != 0 or _p["input_path"] is None:
        mo.stop(True, mo.md(""))

    # gprMax names the output after the input file, but whether a single model
    # writes <stem>.h5 or <stem>1.h5 depends on how it was invoked, so match
    # the pattern rather than assuming either.
    _in = Path(_p["input_path"])
    _candidates = sorted(
        _in.parent.glob(f"{_in.stem}*.h5"), key=lambda q: q.stat().st_mtime, reverse=True
    )
    if not _candidates:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"**The run finished but no output file matching "
                    f"`{_in.stem}*.h5` is in `{_in.parent}`.**"
                ),
                kind="danger",
            ),
        )

    output_path = _candidates[0]
    result = load_file(output_path)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 4 — Output"),
                mo.callout(mo.md(format_metadata_text(result)), kind="info"),
                mo.md(f"_Read from `{output_path}`_"),
            ],
            gap="0.4rem",
        )
    )
    return output_path, result


# ── Component picker, built from what is actually in the file ─────────────
@app.cell
def _(list_components, list_receivers, mo, result):
    _rxs = list_receivers(result)
    _rx = _rxs[0] if _rxs else "rx1"
    _comps = list_components(result, _rx) or ["Ez"]

    component = mo.ui.dropdown(
        options=_comps,
        value="Ez" if "Ez" in _comps else _comps[0],
        label="Component",
    )
    mo.output.replace(
        mo.vstack([mo.md("**Inspect the result**"), component], gap="0.3rem")
    )
    return component


# ── Display options, independent of the data ──────────────────────────────
@app.cell
def _(mo):
    use_gain = mo.ui.checkbox(label="Apply SEC gain", value=False)
    show_predicted = mo.ui.checkbox(label="Mark the predicted arrivals", value=True)
    mo.output.replace(
        mo.hstack([use_gain, show_predicted], gap="1.5rem", justify="start")
    )
    return show_predicted, use_gain


# ── Step 5: Waveform ──────────────────────────────────────────────────────
@app.cell
def _(
    apply_gain,
    component,
    gain_label,
    get_time_axis,
    get_trace,
    get_unit_label,
    go,
    list_receivers,
    mo,
    np,
    predicted,
    result,
    show_predicted,
    use_gain,
):
    _rxs = list_receivers(result)
    _rx = _rxs[0] if _rxs else "rx1"
    _comp = component.value
    _raw = get_trace(result, _comp, _rx)
    _time = get_time_axis(result, unit="ns")
    _unit = get_unit_label(_comp)

    if not np.any(_raw):
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"**`{_comp}` is all zeros.** In a 2D TMz model only Ez, Hx and Hy "
                    "are excited, so the other components are stored but empty."
                ),
                kind="warn",
            ),
        )

    _title = f"{_comp} at {_rx}"
    _arr = _raw
    if use_gain.value:
        _arr, _ = apply_gain(
            _raw, _time, "sec", factor=0.5, power=1.0,
            start_ns=0.15 * float(_time[-1]), max_gain=50,
        )
        _title += "  ·  " + gain_label("sec", 0.5, 1.0, 0.15 * float(_time[-1]))

    _fig = go.Figure(
        go.Scatter(
            x=_time, y=_arr, mode="lines", name=_comp,
            line=dict(color="#1f77b4", width=1.5),
            hovertemplate="%{x:.4f} ns<br>%{y:.5g} " + _unit + "<extra></extra>",
        )
    )

    if show_predicted.value:
        for _label, _t, _colour in (
            ("direct", predicted["direct"], "#7f7f7f"),
            ("reflection", predicted["reflection"], "#d62728"),
        ):
            if _t <= float(_time[-1]):
                _fig.add_vline(
                    x=_t, line=dict(color=_colour, width=1, dash="dot"),
                    annotation_text=f"{_label} {_t:.3f} ns",
                    annotation_position="top",
                )

    _axis = dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.07)",
        showline=True,
        linecolor="rgba(0,0,0,0.18)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.18)",
    )
    _fig.update_layout(
        title=dict(text=_title, x=0.0, xanchor="left"),
        xaxis=dict(title="Time (ns)", **_axis),
        yaxis=dict(title=f"Field strength [{_unit}]", **_axis),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=420,
        margin=dict(l=72, r=32, t=55, b=55),
        hovermode="x unified",
    )

    _peak_t = float(_time[int(np.argmax(np.abs(_arr)))])
    _record_end = float(_time[-1])
    _refl_off = predicted["reflection"] > _record_end
    _rows = [
        f"| Largest response at | {_peak_t:.4f} ns |",
        f"| Predicted direct wave | {predicted['direct']:.4f} ns |",
        f"| Predicted reflection | {predicted['reflection']:.4f} ns"
        + (f" — past the end of the {_record_end:.3f} ns record |" if _refl_off else " |"),
    ]

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 5 — Waveform"),
                mo.ui.plotly(_fig, config={"displaylogo": False}),
                mo.md("| | |\n|---|---|\n" + "\n".join(_rows)),
                *(
                    [
                        mo.callout(
                            mo.md(
                                f"**No reflection in this trace, and there should not "
                                f"be.** The geometry puts it at "
                                f"{predicted['reflection']:.3f} ns, past the end of "
                                f"the {_record_end:.3f} ns record, so it is not marked "
                                "on the plot. Lengthen the time window in Step 1 and "
                                "run again to record it."
                            ),
                            kind="warn",
                        )
                    ]
                    if _refl_off
                    else []
                ),
                mo.md(
                    "_The dotted markers are predicted from the geometry alone, not "
                    "fitted to this trace. They should land on the two events. A gap "
                    "of a few percent is expected: the prediction is the arrival of an "
                    "idealised impulse, the trace is a finite-bandwidth wavelet through "
                    "a dispersive grid._"
                ),
            ],
            gap="0.5rem",
        )
    )
    return


# ── Where to go next ──────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "**Where to go next**\n\n"
            "- Change the permittivity and run again. The reflection moves later as "
            "the material slows the wave down, and the predicted marker moves with "
            "it. Push the permittivity or the depth far enough and the reflection "
            "leaves the record entirely, which the warning in Step 2 catches before "
            "the solver wastes time on it.\n"
            "- Open the same output file in `ascan_dashboard.py` to overlay several "
            "components, compare runs, switch to the frequency domain, or export the "
            "figure.\n"
            "- Run a sweep of source positions instead of one, and open the results "
            "in `recipes/velocity_permittivity.py` to recover the permittivity from "
            "the hyperbola rather than being told it."
        ),
        kind="neutral",
    )
    return


if __name__ == "__main__":
    app.run()
