import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import io
    import re
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from toolboxes.Marimo.h5_reader import (
        format_metadata_text,
        get_unit_label,
        list_components,
        list_receivers,
        load_file,
    )
    from toolboxes.Marimo.processing import (
        GAIN_KINDS,
        apply_gain,
        gain_label,
        remove_mean_trace,
    )
    from toolboxes.Marimo.trace_matrix import process_trace, stack_traces

    # gprMax per-trace B-scan naming: <base><N>.h5, e.g. cylinder_Bscan_2D1.h5
    # .. cylinder_Bscan_2D60.h5 for a -n 60 run. \d+ is greedy, so multi-digit
    # trace numbers parse correctly.
    TRACE_FILE_RE = re.compile(r"^(.*?)(\d+)\.h5$")

    def discover_bases(directory: str) -> dict[str, int]:
        """Group .h5 trace files in `directory` by inferred base name."""
        d = Path(directory)
        if not directory or not d.is_dir():
            return {}
        bases: dict[str, int] = {}
        for p in sorted(d.glob("*.h5")):
            if "merged" in p.name.lower():
                continue
            m = TRACE_FILE_RE.match(p.name)
            if m:
                bases[m.group(1)] = bases.get(m.group(1), 0) + 1
        return bases

    def find_trace_files(directory: str, base: str) -> list[tuple[int, Path]]:
        """Trace files for `base` in `directory`, sorted by trace number."""
        d = Path(directory)
        if not directory or not base or not d.is_dir():
            return []
        found = []
        for p in d.glob(f"{base}*.h5"):
            if "merged" in p.name.lower():
                continue
            m = TRACE_FILE_RE.match(p.name)
            if m and m.group(1) == base:
                found.append((int(m.group(2)), p))
        found.sort(key=lambda t: t[0])
        return found

    def apply_processing(matrix, time_ns, processing):
        """Background removal then gain, in that order. Returns
        (display_matrix, gain_curve, summary, background).

        Order is not arbitrary: the GPR literature is consistent that
        background removal comes before gain, because gaining first
        amplifies the stationary direct wave along with everything else and
        leaves a much larger residual for the subtraction to cancel.
        Display normalisation happens last, inside build_bscan_figure.

        `summary` is the one-line description that goes into the plot title
        and the CSV header, so an exported figure records what was done.
        """
        proc = processing or {}
        kind = proc.get("kind", "none")
        window = proc.get("background_window")
        parts = []

        display = matrix
        background = None

        if proc.get("remove_background"):
            display, background = remove_mean_trace(display, window)
            parts.append(
                "background removed (all traces)"
                if window is None
                else f"background removed ({window}-trace window)"
            )

        curve = None
        if kind != "none":
            # Start is carried as a percentage so the control cell has no
            # dependency on the data. A live B-scan rebuilds its time-window
            # slider on every arriving trace, and an ns-valued start slider
            # would be rebuilt with it, resetting gain mid-run.
            window_ns = float(time_ns[-1]) if len(time_ns) else 0.0
            start_ns = (proc.get("start_pct", 0.0) / 100.0) * window_ns
            max_gain = proc.get("max_gain")
            display, curve = apply_gain(
                display,
                time_ns,
                kind,
                factor=proc.get("factor", 1.0),
                power=proc.get("power", 1.0),
                start_ns=start_ns,
                max_gain=max_gain,
            )
            # Only name the clamp when it actually bound. Reporting a limit
            # the gain never reached puts a misleading number in a title.
            bound = max_gain is not None and float(np.max(curve)) >= max_gain
            parts.append(
                gain_label(
                    kind,
                    proc.get("factor", 1.0),
                    proc.get("power", 1.0),
                    start_ns,
                    max_gain if bound else None,
                )
            )

        return display, curve, "  ·  ".join(parts), background

    def build_gain_curve_figure(time_ns, curve, bg, font_size, time_range, line_colour="#7f7f7f"):
        """Small standalone panel for the applied gain curve.

        Its own figure rather than an extra axis on the radargram: a heatmap
        has no spare y-axis, and a 3D surface has none at all. Time runs
        horizontally here even though the radargram draws it vertically,
        which keeps this panel identical to the one in ascan_dashboard.py.
        """
        is_dark = bg == "#0e1117"
        fc = "#e8e8e8" if is_dark else "#1a1a1a"
        lc = "rgba(255,255,255,0.25)" if is_dark else "rgba(0,0,0,0.18)"
        gc = "rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.07)"

        fig = go.Figure(
            data=go.Scatter(
                x=time_ns,
                y=curve,
                mode="lines",
                showlegend=False,
                line=dict(color=line_colour, width=1.5),
                hovertemplate="%{x:.4f} ns<br>x%{y:.4g}<extra>gain</extra>",
            )
        )
        lo, hi = time_range if time_range is not None else (float(time_ns[0]), float(time_ns[-1]))
        fig.update_layout(
            xaxis=dict(
                title="Time (ns)",
                range=[lo, hi],
                showgrid=True,
                gridcolor=gc,
                showline=True,
                linecolor=lc,
                tickfont=dict(size=font_size - 2, color=fc),
                title_font=dict(size=font_size - 1, color=fc),
            ),
            yaxis=dict(
                title="Gain (x)",
                type="log" if float(np.min(curve)) > 0 else "linear",
                showgrid=True,
                gridcolor=gc,
                showline=True,
                linecolor=lc,
                tickfont=dict(size=font_size - 2, color=fc),
                title_font=dict(size=font_size - 1, color=fc),
            ),
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            height=180,
            margin=dict(l=72, r=32, t=10, b=44),
        )
        return fig

    def build_bscan_figure(matrix, x, y, x_title, unit, colourscale, bg, font_size, view_mode, title_text, time_range=None, normalize=False):
        """Heatmap or 3D Surface from an assembled (time x position) matrix."""
        if normalize:
            peak = np.max(np.abs(matrix), axis=0)
            peak[peak == 0] = 1.0
            matrix = matrix / peak
            unit = "normalized"

        is_dark = bg == "#0e1117"
        fc = "#e8e8e8" if is_dark else "#1a1a1a"
        lc = "rgba(255,255,255,0.25)" if is_dark else "rgba(0,0,0,0.18)"
        amp = float(np.max(np.abs(matrix))) or 1.0

        # Time axis always drawn with early time at top (GPR convention).
        # Plotly's "reversed" autorange doesn't combine cleanly with an
        # explicit zoom range, so this always sets an explicit descending
        # range instead of relying on autorange.
        lo, hi = time_range if time_range is not None else (float(np.min(y)), float(np.max(y)))
        time_axis_range = [hi, lo]

        if view_mode == "3D Surface":
            fig = go.Figure(
                data=go.Surface(
                    z=matrix,
                    x=x,
                    y=y,
                    colorscale=colourscale,
                    cmin=-amp,
                    cmax=amp,
                    colorbar=dict(title=dict(text=unit, font=dict(color=fc)), tickfont=dict(color=fc)),
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title=x_title, color=fc),
                    yaxis=dict(title="Time (ns)", range=time_axis_range, color=fc),
                    zaxis=dict(title=unit, color=fc),
                ),
                height=600,
            )
        else:
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=x,
                    y=y,
                    colorscale=colourscale,
                    zmid=0,
                    zmin=-amp,
                    zmax=amp,
                    colorbar=dict(title=dict(text=unit, font=dict(color=fc)), tickfont=dict(color=fc)),
                )
            )
            fig.update_layout(
                xaxis=dict(
                    title=x_title,
                    tickfont=dict(size=font_size - 1, color=fc),
                    title_font=dict(size=font_size, color=fc),
                    showline=True,
                    linecolor=lc,
                ),
                yaxis=dict(
                    title="Time (ns)",
                    range=time_axis_range,
                    tickfont=dict(size=font_size - 1, color=fc),
                    title_font=dict(size=font_size, color=fc),
                    showline=True,
                    linecolor=lc,
                ),
                height=550,
                margin=dict(l=72, r=32, t=55, b=55),
            )

        fig.update_layout(
            title=dict(text=title_text, font=dict(size=font_size + 1, color=fc), x=0.0, xanchor="left"),
            paper_bgcolor=bg,
            plot_bgcolor=bg,
        )
        return fig

    def render_bscan_section(matrix, positions_x, time_ns, component, all_positions_physical, index_label, colourscale, bg, font_size, view_mode, normalize, title_text, csv_filename, time_range=None, sample_file=None, processing=None):
        """Metadata banner + plot + CSV/SVG/PDF export. Shared by both the
        live and load-files renderer cells so the two don't drift apart.

        Processing lives here rather than in the two renderer cells for the
        same reason: one implementation, so background removal and gain
        cannot behave differently between live monitoring and load-files.
        """
        if matrix is None:
            return mo.callout(mo.md("No traces to show yet."), kind="neutral")

        display, gain_curve, proc_summary, _background = apply_processing(
            matrix, time_ns, processing
        )
        proc = processing or {}
        unit = get_unit_label(component or "Ez")
        if proc.get("kind", "none") != "none":
            unit += " (gained)"
        if proc_summary:
            title_text = f"{title_text}  ·  {proc_summary}"

        x_title = (
            "Source x-position (m)"
            if all_positions_physical
            else f"{index_label} (source position unavailable in one or more files)"
        )
        fig = build_bscan_figure(
            display, positions_x, time_ns, x_title, unit,
            colourscale, bg, font_size, view_mode, title_text,
            time_range=time_range, normalize=normalize,
        )

        # CSV always exports the full, unzoomed, un-normalized, unprocessed
        # matrix — same convention ascan_dashboard.py uses. The processing
        # line is written whether or not any processing is active, so the
        # file format never changes depending on a UI setting. Commas inside
        # the summary become semicolons to keep it one unquoted field.
        try:
            buf = io.StringIO()
            buf.write(
                f"# processing: {(proc_summary or 'none').replace(',', ';')}. "
                f"Values below are raw.\n"
            )
            writer = csv.writer(buf)
            writer.writerow(["time_ns"] + [f"x={p:.4f}" for p in positions_x])
            for i in range(len(time_ns)):
                writer.writerow([f"{time_ns[i]:.8g}"] + [f"{matrix[i, j]:.10g}" for j in range(matrix.shape[1])])
            csv_btn = mo.download(
                data=buf.getvalue().encode("utf-8"), filename=csv_filename,
                label="Download CSV (full raw matrix)", mimetype="text/csv",
            )
        except Exception as e:
            csv_btn = mo.md(f"_CSV export error: `{e}`_")

        stem = csv_filename.rsplit(".", 1)[0]

        # Lazy on purpose: mo.download accepts a zero-arg callable and only
        # calls it when the button is clicked. kaleido spins up headless
        # Chrome per export, which is slow — computing it eagerly here
        # meant every poll tick (every 1-2s with live polling on) paid that
        # cost twice for a download nobody asked for yet. That was the
        # actual cause of the "reloads every couple seconds" slowness.
        def _make_svg():
            import plotly.io as pio
            return pio.to_image(fig, format="svg", width=1200, height=600, scale=2)

        def _make_pdf():
            import plotly.io as pio
            return pio.to_image(fig, format="pdf", width=1200, height=600, scale=2)

        svg_btn = mo.download(data=_make_svg, filename=f"{stem}.svg", label="Download SVG", mimetype="image/svg+xml")
        pdf_btn = mo.download(data=_make_pdf, filename=f"{stem}.pdf", label="Download PDF", mimetype="application/pdf")

        elements = []
        if sample_file is not None:
            elements.append(mo.callout(mo.md(format_metadata_text(sample_file)), kind="info"))
        elements += [
            mo.md("### Radargram"),
            mo.ui.plotly(
                fig,
                config={
                    "toImageButtonOptions": {"format": "svg", "filename": stem, "height": 600, "width": 1200, "scale": 2},
                    "displaylogo": False,
                },
            ),
        ]

        if gain_curve is not None and proc.get("show_curve"):
            elements += [
                mo.md(f"**Applied gain curve** — {proc_summary}"),
                mo.ui.plotly(
                    build_gain_curve_figure(time_ns, gain_curve, bg, font_size, time_range),
                    config={"displaylogo": False, "displayModeBar": False},
                ),
            ]

        notes = []
        if proc_summary:
            notes.append(
                "_Radargram and image exports show processed amplitudes. "
                "CSV stays raw, with the processing recorded on its first line._"
            )
        notes.append(
            "_SVG/PDF need `plotly_get_chrome` run once. If either "
            "fails, the camera icon in the plot toolbar above always "
            "works and needs no setup._"
        )

        elements += [
            mo.md("**Export**"),
            mo.hstack([csv_btn, svg_btn, pdf_btn], gap="1rem", justify="start"),
            mo.md("  \n".join(notes)),
        ]
        return mo.vstack(elements, gap="0.5rem")

    return (
        GAIN_KINDS,
        Path,
        discover_bases,
        find_trace_files,
        list_components,
        list_receivers,
        load_file,
        mo,
        np,
        process_trace,
        render_bscan_section,
        stack_traces,
        time,
    )


# ══ PART 1 — LIVE MONITORING ═══════════════════════════════════════════════
# Watches a directory while a B-scan is actively running.


# ── Step 1: Output directory ────────────────────────────────────────────
# Widget creation and reading its .value must be in separate cells in this
# marimo version, so the base-name scan happens in the next cell.
@app.cell
def _(mo):
    directory_input = mo.ui.text(
        label="",
        placeholder="/absolute/path/to/gprMax/output/directory",
        full_width=True,
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md("# gprMax B-Scan Dashboard"),
                mo.md(
                    "Two ways to build a radargram: watch a B-scan while "
                    "it's still running (Part 1), or assemble one from "
                    "A-scan files you already have (Part 2)."
                ),
                mo.md("---"),
                mo.md("## Part 1 — Live monitoring"),
                mo.md(
                    "Watches a directory while gprMax writes one closed "
                    "`.h5` file per trace, building the radargram as each "
                    "one completes. Safe to point at a directory before "
                    "the simulation starts."
                ),
                mo.md("### Step 1 — Output directory"),
                directory_input,
            ],
            gap="0.4rem",
        )
    )
    return (directory_input,)


# ── Step 2: Base filename ───────────────────────────────────────────────
@app.cell
def _(Path, directory_input, discover_bases, mo):
    _dir = directory_input.value
    _bases = discover_bases(_dir)

    if _dir and not Path(_dir).is_dir():
        _status = mo.callout(mo.md(f"`{_dir}` is not a directory."), kind="danger")
        base_selector = mo.ui.dropdown(options=["—"], value="—", label="Base filename")
    elif not _bases:
        _status = mo.md(
            "_Waiting for `.h5` trace files matching gprMax's per-trace "
            "naming (`<basename><N>.h5`) to appear in this directory._"
            if _dir
            else ""
        )
        base_selector = mo.ui.dropdown(options=["—"], value="—", label="Base filename")
    else:
        _options = {
            f"{b}   ({n} file{'s' if n != 1 else ''} so far)": b
            for b, n in sorted(_bases.items())
        }
        base_selector = mo.ui.dropdown(
            options=_options, value=list(_options.keys())[0], label="Base filename"
        )
        _status = mo.md(f"_Detected {len(_bases)} distinct base name(s)._")

    mo.output.replace(
        mo.vstack([mo.md("### Step 2 — Base filename"), _status, base_selector], gap="0.4rem")
    )
    return (base_selector,)


# ── State: isolated, no UI params — must only ever run once. Sharing this
# cell with a widget caused a state-reset bug in ascan_dashboard.py
# (Component 3) — don't repeat that here.
@app.cell
def _(mo):
    get_bscan_state, set_bscan_state = mo.state(
        {
            "seen": set(),
            "matrix_cols": [],
            "positions_x": [],
            "trace_nums": [],
            "time_ns": None,
            "component": None,
            "receiver": None,
            "base_name": None,
            "known_components": ["Ez"],
            "all_positions_physical": True,
            "warnings": [],
            "last_poll": None,
            "sample_file": None,
        }
    )
    return get_bscan_state, set_bscan_state


# ── Step 3: Polling ──────────────────────────────────────────────────────
@app.cell
def _(mo):
    live_toggle = mo.ui.switch(label="Live polling", value=True)
    refresh = mo.ui.refresh(options=["1s", "2s", "5s", "10s", "30s"], default_interval="2s")
    poll_now_button = mo.ui.run_button(label="⟳  Poll now")
    reset_button = mo.ui.run_button(label="↺  Reset accumulated data")

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 3 — Polling"),
                mo.hstack([live_toggle, refresh, poll_now_button, reset_button], gap="1.5rem", justify="start"),
                mo.md(
                    "_Each tick only reads unseen files and appends them — "
                    "never rebuilds from scratch. Changing base filename, "
                    "field component, or receiver resets accumulated data "
                    "automatically._"
                ),
            ],
            gap="0.4rem",
        )
    )
    return live_toggle, poll_now_button, refresh, reset_button


# ── Step 4: Field component & receiver ──────────────────────────────────
# Independent of get_bscan_state() on purpose. The poll cell below writes
# that state and depends on these widgets — reading state here to build
# the options caused an infinite loop (poll writes -> this reruns -> new
# widgets -> poll reruns -> ...). Peeking at the file directly instead
# means this only reruns when directory/base actually change.
@app.cell
def _(base_selector, directory_input, find_trace_files, list_components, list_receivers, load_file, mo):
    _known_comps = ["Ez"]
    _known_rx = ["rx1"]
    _files = find_trace_files(directory_input.value, base_selector.value)

    if _files:
        try:
            _fdata = load_file(_files[0][1])
            _known_rx = list_receivers(_fdata) or _known_rx
            _known_comps = list_components(_fdata, _known_rx[0]) or _known_comps
        except (FileNotFoundError, OSError, KeyError):
            pass  # file may still be mid-write — the poll cell retries it

    _default_comp = "Ez" if "Ez" in _known_comps else _known_comps[0]
    component_selector = mo.ui.dropdown(options=_known_comps, value=_default_comp, label="Field component")
    receiver_selector = mo.ui.dropdown(options=_known_rx, value=_known_rx[0], label="Receiver")

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 4 — Field component & receiver"),
                mo.md("_Read from the first trace file found, not assumed._"),
                mo.hstack([component_selector, receiver_selector], gap="2rem", justify="start"),
            ],
            gap="0.4rem",
        )
    )
    return component_selector, receiver_selector


# ── Poll / load ──────────────────────────────────────────────────────────
@app.cell
def _(
    Path,
    base_selector,
    component_selector,
    directory_input,
    find_trace_files,
    get_bscan_state,
    live_toggle,
    load_file,
    mo,
    poll_now_button,
    process_trace,
    receiver_selector,
    refresh,
    reset_button,
    set_bscan_state,
    time,
):
    _ = refresh.value  # reactive dependency on the polling timer

    _prev = get_bscan_state()
    _base = base_selector.value
    _current_component = component_selector.value
    _current_receiver = receiver_selector.value

    if not directory_input.value or _base in (None, "—"):
        mo.stop(
            True,
            mo.callout(
                mo.md("Enter a directory and select a base filename above to start watching."),
                kind="neutral",
            ),
        )

    _needs_reset = (
        reset_button.value
        or _prev["base_name"] != _base
        or _prev["component"] != _current_component
        or _prev["receiver"] != _current_receiver
    )

    # Fresh dict for set_bscan_state() rather than mutating _prev in place —
    # matches the append-a-new-list pattern in ascan_dashboard.py's handler.
    if _needs_reset:
        _seen, _cols, _xpos, _nums = set(), [], [], []
        _time_ns, _all_physical, _warnings, _sample_file = None, True, [], None
    else:
        _seen = set(_prev["seen"])
        _cols = list(_prev["matrix_cols"])
        _xpos = list(_prev["positions_x"])
        _nums = list(_prev["trace_nums"])
        _time_ns = _prev["time_ns"]
        _all_physical = _prev["all_positions_physical"]
        _warnings = list(_prev["warnings"])
        _sample_file = _prev["sample_file"]

    _known_components = _prev["known_components"]
    _new_warnings = []
    _new_trace_count = 0
    _should_scan = live_toggle.value or poll_now_button.value or _needs_reset

    if _should_scan and Path(directory_input.value).is_dir():
        for _num, _path in find_trace_files(directory_input.value, _base):
            _key = str(_path.resolve())
            if _key in _seen:
                continue

            try:
                _fdata = load_file(_path)
            except (FileNotFoundError, OSError, KeyError):
                continue  # still being written — retry next tick, don't mark seen

            _result = process_trace(_fdata, _current_component, len(_cols[0]) if _cols else None, _current_receiver)
            if _result["known_components"]:
                _known_components = _result["known_components"]

            if not _result["ok"]:
                _new_warnings.append(f"{_path.name}: {_result['reason']} — skipped")
                _seen.add(_key)
                continue

            if _time_ns is None:
                _time_ns = _result["time_ns"]
            if _sample_file is None:
                _sample_file = _fdata
            if _result["x"] is None:
                _all_physical = False

            _cols.append(_result["array"])
            _xpos.append(_result["x"] if _result["x"] is not None else float(_num))
            _nums.append(_num)
            _seen.add(_key)
            _new_trace_count += 1

    _scan_time = time.strftime("%H:%M:%S") if _should_scan else _prev["last_poll"]
    _all_warnings = (_warnings + _new_warnings)[-10:]

    # Only touch shared state when something real happened — a reset, a
    # newly loaded trace, or a new warning. An empty poll (the common case
    # once caught up) leaves state untouched, so the time-window slider and
    # renderer downstream don't rebuild every tick for no reason. The
    # status line below still updates the displayed poll time every tick
    # regardless, since that's local to this cell's own output and doesn't
    # trigger anything downstream.
    _changed = _needs_reset or _new_trace_count > 0 or bool(_new_warnings)
    if _changed:
        set_bscan_state(
            {
                "seen": _seen,
                "matrix_cols": _cols,
                "positions_x": _xpos,
                "trace_nums": _nums,
                "time_ns": _time_ns,
                "component": _current_component,
                "receiver": _current_receiver,
                "base_name": _base,
                "known_components": _known_components,
                "all_positions_physical": _all_physical,
                "warnings": _all_warnings,
                "last_poll": _scan_time,
                "sample_file": _sample_file,
            }
        )

    _parts = [f"**{len(_cols)} trace(s) loaded**"]
    if _current_component:
        _parts.append(f"component `{_current_component}`")
    if _current_receiver:
        _parts.append(f"receiver `{_current_receiver}`")
    _parts.append(f"last poll {_scan_time or '—'}")
    _summary = [mo.md("  ·  ".join(_parts))]
    if _all_warnings:
        _summary.append(
            mo.callout(
                mo.md("**Skipped files:**\n\n" + "\n".join(f"- {w}" for w in _all_warnings)),
                kind="warn",
            )
        )
    mo.output.replace(mo.vstack(_summary, gap="0.3rem"))
    return


# ── Step 5: Time window ─────────────────────────────────────────────────
@app.cell
def _(get_bscan_state, mo):
    _state = get_bscan_state()
    if not _state["matrix_cols"]:
        mo.stop(True, mo.md(""))

    _max_ns = float(_state["time_ns"][-1])
    live_time_range = mo.ui.range_slider(
        start=0.0, stop=_max_ns, step=round(_max_ns / 600, 4),
        value=[0.0, _max_ns], label="Time window (ns)", full_width=True, debounce=True,
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 5 — Time window"),
                mo.md("_Zooms the plot only — CSV export always contains the full unzoomed data._"),
                live_time_range,
            ],
            gap="0.4rem",
        )
    )
    return (live_time_range,)


# ── Step 6: Appearance ───────────────────────────────────────────────────
@app.cell
def _(mo):
    live_view_mode = mo.ui.dropdown(options=["Heatmap", "3D Surface"], value="Heatmap", label="View")
    live_colourscale = mo.ui.dropdown(
        options=["RdBu", "Viridis", "Greys", "Turbo"], value="RdBu", label="Colourscale"
    )
    live_bg = mo.ui.dropdown(
        options={"Light": "#ffffff", "Paper white": "#f8f9fa", "Dark": "#0e1117"},
        value="Light",
        label="Background",
    )
    live_font_size = mo.ui.slider(start=10, stop=22, step=1, value=13, label="Font size", debounce=True)
    live_normalize = mo.ui.checkbox(label="Normalize each trace (view only — CSV stays raw)", value=False)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 6 — Appearance"),
                mo.hstack([live_view_mode, live_colourscale, live_bg, live_font_size], gap="1.5rem", justify="start"),
                live_normalize,
            ],
            gap="0.5rem",
        )
    )
    return live_bg, live_colourscale, live_font_size, live_normalize, live_view_mode


# ── Step 6b: Processing controls ─────────────────────────────────────────
# Widget creation only. live_gain_kind's .value drives the parameter cell
# below, and reading it here would raise at runtime.
@app.cell
def _(GAIN_KINDS, mo):
    live_gain_kind = mo.ui.dropdown(
        options={_meta["label"]: _k for _k, _meta in GAIN_KINDS.items()},
        value="None",
        label="Gain function",
    )
    live_remove_bg = mo.ui.checkbox(label="Remove background (mean trace)", value=False)
    live_bg_window = mo.ui.slider(
        start=0, stop=51, step=1, value=0,
        label="Background window (traces, 0 = all)", debounce=True,
    )
    live_show_curve = mo.ui.checkbox(label="Show gain curve", value=True)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 6b — Processing"),
                mo.md(
                    "_Background removal runs first, then gain, then display "
                    "normalisation, matching standard GPR processing order. "
                    "CSV export stays raw throughout._"
                ),
                mo.hstack([live_remove_bg, live_bg_window], gap="1.5rem", justify="start"),
                mo.hstack([live_gain_kind, live_show_curve], gap="1.5rem", justify="start"),
            ],
            gap="0.4rem",
        )
    )
    return live_bg_window, live_gain_kind, live_remove_bg, live_show_curve


# ── Step 6c: Gain parameters ─────────────────────────────────────────────
# Depends on live_gain_kind alone. Anything derived from the accumulated
# state would rebuild these widgets on every arriving trace and reset the
# gain settings in the middle of a live run, which is why the gain start is
# a percentage of the window rather than an absolute time.
@app.cell
def _(GAIN_KINDS, live_gain_kind, mo):
    _FACTOR_SLIDERS = {
        "constant": dict(start=0.1, stop=20.0, step=0.1, value=2.0, label="Factor (x)"),
        "linear": dict(start=0.0, stop=10.0, step=0.1, value=1.0, label="Slope (per ns)"),
        "exponential": dict(start=0.0, stop=5.0, step=0.05, value=1.0, label="Rate a (per ns)"),
        "db": dict(start=0.0, stop=60.0, step=0.5, value=6.0, label="Gain (dB per ns)"),
        "sec": dict(start=0.0, stop=5.0, step=0.05, value=0.5, label="Rate a (per ns)"),
    }

    _kind = live_gain_kind.value

    if _kind == "none":
        live_gain_factor = None
        live_gain_power = None
        live_gain_start = None
        live_gain_clamp = None
        live_gain_max = None
        mo.output.replace(mo.md(""))
    else:
        _cfg = _FACTOR_SLIDERS.get(_kind)
        live_gain_factor = mo.ui.slider(**_cfg, debounce=True) if _cfg else None
        live_gain_power = (
            mo.ui.slider(start=0.0, stop=4.0, step=0.1, value=1.0, label="Exponent b", debounce=True)
            if GAIN_KINDS[_kind]["uses_power"]
            else None
        )
        live_gain_start = mo.ui.slider(
            start=0.0, stop=100.0, step=0.5, value=0.0,
            label="Gain start (% of window)", debounce=True,
        )
        live_gain_clamp = mo.ui.checkbox(label="Limit maximum gain", value=True)
        live_gain_max = mo.ui.slider(
            start=1, stop=1000, step=1, value=100, label="Maximum gain (x)", debounce=True
        )
        _params = [w for w in (live_gain_factor, live_gain_power, live_gain_start) if w is not None]
        mo.output.replace(
            mo.vstack(
                [
                    mo.hstack(_params, gap="1.5rem", justify="start"),
                    mo.hstack([live_gain_clamp, live_gain_max], gap="1.5rem", justify="start"),
                ],
                gap="0.4rem",
            )
        )
    return live_gain_clamp, live_gain_factor, live_gain_max, live_gain_power, live_gain_start


# ── Live renderer + export ──────────────────────────────────────────────
@app.cell
def _(
    get_bscan_state,
    live_bg,
    live_bg_window,
    live_colourscale,
    live_font_size,
    live_gain_clamp,
    live_gain_factor,
    live_gain_kind,
    live_gain_max,
    live_gain_power,
    live_gain_start,
    live_normalize,
    live_remove_bg,
    live_show_curve,
    live_time_range,
    live_view_mode,
    mo,
    np,
    render_bscan_section,
):
    _state = get_bscan_state()
    _cols = _state["matrix_cols"]

    if not _cols:
        mo.stop(
            True,
            mo.callout(
                mo.md("No traces loaded yet. Point Step 1 at a directory containing gprMax B-scan output."),
                kind="neutral",
            ),
        )

    mo.output.replace(
        render_bscan_section(
            matrix=np.column_stack(_cols),
            positions_x=_state["positions_x"],
            time_ns=_state["time_ns"],
            component=_state["component"],
            all_positions_physical=_state["all_positions_physical"],
            index_label="Trace index",
            colourscale=live_colourscale.value,
            bg=live_bg.value,
            font_size=live_font_size.value,
            view_mode=live_view_mode.value,
            normalize=live_normalize.value,
            title_text=f"B-scan radargram — {_state['component']} ({len(_cols)} traces)",
            csv_filename="gprmax_bscan_live.csv",
            time_range=tuple(live_time_range.value),
            sample_file=_state["sample_file"],
            processing=dict(
                kind=live_gain_kind.value,
                factor=live_gain_factor.value if live_gain_factor is not None else 1.0,
                power=live_gain_power.value if live_gain_power is not None else 1.0,
                start_pct=live_gain_start.value if live_gain_start is not None else 0.0,
                max_gain=(
                    live_gain_max.value
                    if (live_gain_clamp is not None and live_gain_clamp.value)
                    else None
                ),
                remove_background=live_remove_bg.value,
                background_window=(None if live_bg_window.value == 0 else live_bg_window.value),
                show_curve=live_show_curve.value,
            ),
        )
    )
    return


# ══ PART 2 — LOAD FILES ═════════════════════════════════════════════════════
# Assembles a radargram from A-scan files you already have — not
# necessarily from a live-running B-scan, and not necessarily numbered
# sequentially. Ordered by source x-position when available, falling back
# to selection order. Independent of Part 1's state entirely.


# ── Step 7: Pick files ───────────────────────────────────────────────────
@app.cell
def _(mo):
    file_picker = mo.ui.file_browser(filetypes=[".h5"], multiple=True, label="")
    mo.output.replace(
        mo.vstack(
            [
                mo.md("## Part 2 — Load files"),
                mo.md(
                    "Pick any set of single-trace `.h5` files — a finished "
                    "B-scan, or hand-picked A-scans from different runs — "
                    "and assemble them into a radargram or 3D surface."
                ),
                mo.md("### Step 7 — Select files"),
                file_picker,
            ],
            gap="0.4rem",
        )
    )
    return (file_picker,)


# ── Step 8: Field component & receiver ──────────────────────────────────
# Same independent-peek pattern as Part 1's Step 4, for the same reason:
# this must not depend on the assemble cell's output, since the assemble
# cell depends on these widgets.
@app.cell
def _(file_picker, list_components, list_receivers, load_file, mo):
    _known_comps = ["Ez"]
    _known_rx = ["rx1"]

    if file_picker.value:
        try:
            _fdata = load_file(file_picker.value[0].path)
            _known_rx = list_receivers(_fdata) or _known_rx
            _known_comps = list_components(_fdata, _known_rx[0]) or _known_comps
        except (FileNotFoundError, OSError, KeyError):
            pass

    _default_comp = "Ez" if "Ez" in _known_comps else _known_comps[0]
    load_component_selector = mo.ui.dropdown(options=_known_comps, value=_default_comp, label="Field component")
    load_receiver_selector = mo.ui.dropdown(options=_known_rx, value=_known_rx[0], label="Receiver")

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 8 — Field component & receiver"),
                mo.md("_Read from the first selected file, not assumed._"),
                mo.hstack([load_component_selector, load_receiver_selector], gap="2rem", justify="start"),
            ],
            gap="0.4rem",
        )
    )
    return load_component_selector, load_receiver_selector


# ── Assemble ─────────────────────────────────────────────────────────────
@app.cell
def _(file_picker, load_component_selector, load_file, load_receiver_selector, mo, stack_traces):
    if not file_picker.value:
        mo.stop(True, mo.md(""))

    _loaded = []
    _load_warnings = []
    for _f in file_picker.value:
        try:
            _loaded.append((len(_loaded), load_file(_f.path)))
        except (FileNotFoundError, OSError, KeyError) as _e:
            _load_warnings.append(f"{_f.path}: {type(_e).__name__} — skipped")

    def _sort_key(item):
        _idx, _fdata = item
        _sources = _fdata.get("sources", {})
        if "src1" in _sources:
            return (0, _sources["src1"]["position"][0])
        return (1, _idx)  # no position: keep original selection order, after positioned ones

    _ordered = [fdata for _, fdata in sorted(_loaded, key=_sort_key)]
    load_result = stack_traces(_ordered, load_component_selector.value, load_receiver_selector.value)
    load_result["warnings"] = _load_warnings + load_result["warnings"]
    load_result["sample_file"] = _ordered[0] if _ordered else None

    _msg = f"**{len(_ordered)} file(s) loaded**"
    if load_result["component"]:
        _msg += f"  ·  component `{load_result['component']}`"
    if load_result["receiver"]:
        _msg += f"  ·  receiver `{load_result['receiver']}`"
    _out = [mo.md(_msg)]
    if load_result["warnings"]:
        _out.append(
            mo.callout(
                mo.md("**Skipped:**\n\n" + "\n".join(f"- {w}" for w in load_result["warnings"])),
                kind="warn",
            )
        )
    mo.output.replace(mo.vstack(_out, gap="0.3rem"))
    return (load_result,)


# ── Step 9: Time window ─────────────────────────────────────────────────
@app.cell
def _(load_result, mo):
    if load_result["matrix"] is None:
        mo.stop(True, mo.md(""))

    _max_ns = float(load_result["time_ns"][-1])
    load_time_range = mo.ui.range_slider(
        start=0.0, stop=_max_ns, step=round(_max_ns / 600, 4),
        value=[0.0, _max_ns], label="Time window (ns)", full_width=True, debounce=True,
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 9 — Time window"),
                mo.md("_Zooms the plot only — CSV export always contains the full unzoomed data._"),
                load_time_range,
            ],
            gap="0.4rem",
        )
    )
    return (load_time_range,)


# ── Step 10: Appearance ──────────────────────────────────────────────────
@app.cell
def _(mo):
    load_view_mode = mo.ui.dropdown(options=["Heatmap", "3D Surface"], value="Heatmap", label="View")
    load_colourscale = mo.ui.dropdown(
        options=["RdBu", "Viridis", "Greys", "Turbo"], value="RdBu", label="Colourscale"
    )
    load_bg = mo.ui.dropdown(
        options={"Light": "#ffffff", "Paper white": "#f8f9fa", "Dark": "#0e1117"},
        value="Light",
        label="Background",
    )
    load_font_size = mo.ui.slider(start=10, stop=22, step=1, value=13, label="Font size", debounce=True)
    load_normalize = mo.ui.checkbox(label="Normalize each trace (view only — CSV stays raw)", value=False)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 10 — Appearance"),
                mo.hstack([load_view_mode, load_colourscale, load_bg, load_font_size], gap="1.5rem", justify="start"),
                load_normalize,
            ],
            gap="0.5rem",
        )
    )
    return load_bg, load_colourscale, load_font_size, load_normalize, load_view_mode


# ── Step 10b: Processing controls ─────────────────────────────────────────
# Widget creation only. load_gain_kind's .value drives the parameter cell
# below, and reading it here would raise at runtime.
@app.cell
def _(GAIN_KINDS, mo):
    load_gain_kind = mo.ui.dropdown(
        options={_meta["label"]: _k for _k, _meta in GAIN_KINDS.items()},
        value="None",
        label="Gain function",
    )
    load_remove_bg = mo.ui.checkbox(label="Remove background (mean trace)", value=False)
    load_bg_window = mo.ui.slider(
        start=0, stop=51, step=1, value=0,
        label="Background window (traces, 0 = all)", debounce=True,
    )
    load_show_curve = mo.ui.checkbox(label="Show gain curve", value=True)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 10b — Processing"),
                mo.md(
                    "_Background removal runs first, then gain, then display "
                    "normalisation, matching standard GPR processing order. "
                    "CSV export stays raw throughout._"
                ),
                mo.hstack([load_remove_bg, load_bg_window], gap="1.5rem", justify="start"),
                mo.hstack([load_gain_kind, load_show_curve], gap="1.5rem", justify="start"),
            ],
            gap="0.4rem",
        )
    )
    return load_bg_window, load_gain_kind, load_remove_bg, load_show_curve


# ── Step 10c: Gain parameters ─────────────────────────────────────────────
# Depends on load_gain_kind alone. Anything derived from the accumulated
# state would rebuild these widgets on every arriving trace and reset the
# gain settings in the middle of a live run, which is why the gain start is
# a percentage of the window rather than an absolute time.
@app.cell
def _(GAIN_KINDS, load_gain_kind, mo):
    _FACTOR_SLIDERS = {
        "constant": dict(start=0.1, stop=20.0, step=0.1, value=2.0, label="Factor (x)"),
        "linear": dict(start=0.0, stop=10.0, step=0.1, value=1.0, label="Slope (per ns)"),
        "exponential": dict(start=0.0, stop=5.0, step=0.05, value=1.0, label="Rate a (per ns)"),
        "db": dict(start=0.0, stop=60.0, step=0.5, value=6.0, label="Gain (dB per ns)"),
        "sec": dict(start=0.0, stop=5.0, step=0.05, value=0.5, label="Rate a (per ns)"),
    }

    _kind = load_gain_kind.value

    if _kind == "none":
        load_gain_factor = None
        load_gain_power = None
        load_gain_start = None
        load_gain_clamp = None
        load_gain_max = None
        mo.output.replace(mo.md(""))
    else:
        _cfg = _FACTOR_SLIDERS.get(_kind)
        load_gain_factor = mo.ui.slider(**_cfg, debounce=True) if _cfg else None
        load_gain_power = (
            mo.ui.slider(start=0.0, stop=4.0, step=0.1, value=1.0, label="Exponent b", debounce=True)
            if GAIN_KINDS[_kind]["uses_power"]
            else None
        )
        load_gain_start = mo.ui.slider(
            start=0.0, stop=100.0, step=0.5, value=0.0,
            label="Gain start (% of window)", debounce=True,
        )
        load_gain_clamp = mo.ui.checkbox(label="Limit maximum gain", value=True)
        load_gain_max = mo.ui.slider(
            start=1, stop=1000, step=1, value=100, label="Maximum gain (x)", debounce=True
        )
        _params = [w for w in (load_gain_factor, load_gain_power, load_gain_start) if w is not None]
        mo.output.replace(
            mo.vstack(
                [
                    mo.hstack(_params, gap="1.5rem", justify="start"),
                    mo.hstack([load_gain_clamp, load_gain_max], gap="1.5rem", justify="start"),
                ],
                gap="0.4rem",
            )
        )
    return load_gain_clamp, load_gain_factor, load_gain_max, load_gain_power, load_gain_start


# ── Load-files renderer + export ────────────────────────────────────────
@app.cell
def _(
    load_bg,
    load_bg_window,
    load_colourscale,
    load_font_size,
    load_gain_clamp,
    load_gain_factor,
    load_gain_kind,
    load_gain_max,
    load_gain_power,
    load_gain_start,
    load_normalize,
    load_remove_bg,
    load_result,
    load_show_curve,
    load_time_range,
    load_view_mode,
    mo,
    render_bscan_section,
):
    if load_result["matrix"] is None:
        mo.stop(
            True,
            mo.callout(mo.md("No usable traces in the current selection."), kind="neutral"),
        )

    mo.output.replace(
        render_bscan_section(
            matrix=load_result["matrix"],
            positions_x=load_result["positions_x"],
            time_ns=load_result["time_ns"],
            component=load_result["component"],
            all_positions_physical=load_result["all_positions_physical"],
            index_label="File index",
            colourscale=load_colourscale.value,
            bg=load_bg.value,
            font_size=load_font_size.value,
            view_mode=load_view_mode.value,
            normalize=load_normalize.value,
            title_text=f"Assembled radargram — {load_result['component']} ({load_result['matrix'].shape[1]} traces)",
            csv_filename="gprmax_bscan_assembled.csv",
            time_range=tuple(load_time_range.value),
            sample_file=load_result["sample_file"],
            processing=dict(
                kind=load_gain_kind.value,
                factor=load_gain_factor.value if load_gain_factor is not None else 1.0,
                power=load_gain_power.value if load_gain_power is not None else 1.0,
                start_pct=load_gain_start.value if load_gain_start is not None else 0.0,
                max_gain=(
                    load_gain_max.value
                    if (load_gain_clamp is not None and load_gain_clamp.value)
                    else None
                ),
                remove_background=load_remove_bg.value,
                background_window=(None if load_bg_window.value == 0 else load_bg_window.value),
                show_curve=load_show_curve.value,
            ),
        )
    )
    return


# ── Known gaps ────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "**Known gaps — deliberate, not oversights**\n\n"
            "- No fallback yet to a merged output file if gprMax stops "
            "writing fully-closed per-trace files. Deferred: current "
            "behaviour (one closed file per trace) has been confirmed "
            "against real runs, so this defends against a scenario with "
            "no indication it's coming.\n"
            "- Part 2 orders by source x-position when available; files "
            "without that metadata fall back to selection order, which "
            "may not reflect physical position.\n"
            "- No invert-polarity or time-shift controls. Antonis's "
            "\"manipulate\" scope wasn't fully pinned down — zoom, "
            "normalize, gain and background removal are built, the rest "
            "is worth clarifying before guessing further.\n"
            "- No AGC. It equalises amplitudes over a local window and "
            "destroys relative reflectivity, which is precisely what an "
            "FDTD simulation gets right and a field instrument does not.\n"
            "- Bandpass and lowpass filters, and B-scan minus B-scan "
            "subtraction, are not built. Same processing family as gain "
            "and background removal, and they plug into the same "
            "`processing.py` module when they are."
        ),
        kind="neutral",
    )
    return


if __name__ == "__main__":
    app.run()
