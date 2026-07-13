import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import io
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from toolboxes.Marimo.h5_reader import (
        build_label,
        format_metadata_text,
        get_time_axis,
        get_trace,
        get_unit_label,
        list_components,
        list_receivers,
        load_files,
    )

    # Research-quality colour palette (Matplotlib tab10, colorblind-friendly)
    PALETTE = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
    ]

    COLOUR_NAMES = {
        "Blue": "#1f77b4",
        "Red": "#d62728",
        "Green": "#2ca02c",
        "Orange": "#ff7f0e",
        "Purple": "#9467bd",
        "Brown": "#8c564b",
        "Pink": "#e377c2",
        "Teal": "#17becf",
        "Olive": "#bcbd22",
        "Grey": "#7f7f7f",
        "Cyan": "#4FC3F7",
        "Black": "#111111",
    }

    return (
        COLOUR_NAMES,
        PALETTE,
        Path,
        build_label,
        csv,
        format_metadata_text,
        get_time_axis,
        get_trace,
        get_unit_label,
        go,
        io,
        list_components,
        list_receivers,
        load_files,
        mo,
        np,
    )


# ── SECTION 1: File loading ────────────────────────────────────────────────
@app.cell
def _(mo):
    file_browser = mo.ui.file_browser(
        filetypes=[".h5"],
        label="",
        multiple=True,
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md("# gprMax A-Scan Post-Processing Dashboard"),
                mo.md(
                    "Load one or more `.h5` output files, select field components, "
                    "and build publication-quality overlay plots with SVG, PDF, "
                    "and CSV export."
                ),
                mo.md("---"),
                mo.md("### Step 1 — Load output files"),
                mo.md(
                    "_Select one or more gprMax `.h5` output files. "
                    "All field components (Ex, Ey, Ez, Hx, Hy, Hz) "
                    "are detected automatically from each file._"
                ),
                file_browser,
            ],
            gap="0.4rem",
        )
    )
    return (file_browser,)


# ── SECTION 2: Data loading + metadata banner ──────────────────────────────
@app.cell
def _(file_browser, format_metadata_text, load_files, mo):
    if not file_browser.value:
        mo.stop(True, mo.md(""))

    _paths = [f.path for f in file_browser.value]
    _loaded = load_files(_paths)
    get_data, set_data = mo.state({"files": _loaded})

    _cards = [
        mo.callout(
            mo.md(f"**{_fname}**\n\n{format_metadata_text(_fdata)}"),
            kind="info",
        )
        for _fname, _fdata in _loaded.items()
    ]

    mo.output.replace(
        mo.vstack(
            [
                mo.md(f"### {len(_loaded)} file(s) loaded"),
                *_cards,
                mo.md("---"),
            ],
            gap="0.4rem",
        )
    )
    return get_data, set_data


# ── STATE: isolated — no UI dependencies ──────────────────────────────────
@app.cell
def _(mo):
    get_traces, set_traces = mo.state([])
    return get_traces, set_traces


# ── SECTION 3: Trace picker ────────────────────────────────────────────────
@app.cell
def _(
    COLOUR_NAMES,
    PALETTE,
    get_data,
    get_traces,
    list_components,
    list_receivers,
    mo,
):
    _state = get_data()
    _files = _state["files"]

    if not _files:
        mo.stop(True, mo.md("No files loaded."))

    _file_options = list(_files.keys())
    _first_file = _file_options[0]
    _first_rx = list_receivers(_files[_first_file])[0]
    _first_comps = list_components(_files[_first_file], _first_rx)

    _next_hex = PALETTE[len(get_traces()) % len(PALETTE)]
    _auto_name = next((name for name, h in COLOUR_NAMES.items() if h == _next_hex), "Blue")

    file_selector = mo.ui.dropdown(
        options=_file_options,
        value=_first_file,
        label="File",
    )
    receiver_selector = mo.ui.dropdown(
        options=list_receivers(_files[_first_file]),
        value=_first_rx,
        label="Receiver",
    )
    component_selector = mo.ui.dropdown(
        options=_first_comps,
        value="Ez" if "Ez" in _first_comps else _first_comps[0],
        label="Component",
    )
    colour_selector = mo.ui.dropdown(
        options=COLOUR_NAMES,
        value=_auto_name,
        label="Trace colour",
    )
    add_button = mo.ui.run_button(label="＋  Add trace")
    clear_button = mo.ui.run_button(label="✕  Clear all")

    _traces_now = get_traces()
    if _traces_now:
        _remove_opts = {t["label"]: i for i, t in enumerate(_traces_now)}
        remove_selector = mo.ui.dropdown(
            options=_remove_opts,
            value=list(_remove_opts.keys())[0],
            label="Remove a trace",
        )
        remove_button = mo.ui.run_button(label="－  Remove selected")
        _remove_section = mo.vstack(
            [
                mo.md("**Remove a trace**"),
                mo.hstack([remove_selector, remove_button], gap="1rem"),
            ],
            gap="0.3rem",
        )
    else:
        remove_selector = None
        remove_button = None
        _remove_section = mo.md("")

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 2 — Build your plot"),
                mo.md(
                    "_Select a file, receiver, and field component. "
                    "Colour auto-advances through the research palette — "
                    "override before clicking Add._"
                ),
                mo.md("**Select source**"),
                mo.hstack([file_selector, receiver_selector], gap="2rem"),
                mo.md("**Select component and colour**"),
                mo.hstack([component_selector, colour_selector], gap="2rem"),
                mo.hstack([add_button, clear_button], gap="1rem"),
                mo.md("---"),
                _remove_section,
            ],
            gap="0.5rem",
        )
    )
    return (
        add_button,
        clear_button,
        colour_selector,
        component_selector,
        file_selector,
        receiver_selector,
        remove_button,
        remove_selector,
    )


# ── BUTTON HANDLER ─────────────────────────────────────────────────────────
@app.cell
def _(
    add_button,
    build_label,
    clear_button,
    colour_selector,
    component_selector,
    file_selector,
    get_data,
    get_traces,
    list_components,
    mo,
    receiver_selector,
    remove_button,
    remove_selector,
    set_traces,
):
    if add_button.value:
        _fname = file_selector.value
        _rx = receiver_selector.value
        _comp = component_selector.value
        _available = list_components(get_data()["files"][_fname], _rx)
        if _comp not in _available:
            mo.output.replace(
                mo.callout(
                    mo.md(
                        f"**Component not found.** "
                        f"`{_comp}` is not in `{_fname}` / `{_rx}`. "
                        f"Available: {', '.join(_available)}"
                    ),
                    kind="danger",
                )
            )
        else:
            _new = {
                "filename": _fname,
                "receiver": _rx,
                "component": _comp,
                "colour": colour_selector.value,
                "label": build_label(_fname, _rx, _comp),
            }
            _cur = get_traces()
            _dup = any(
                t["filename"] == _new["filename"]
                and t["receiver"] == _new["receiver"]
                and t["component"] == _new["component"]
                for t in _cur
            )
            if not _dup:
                set_traces(_cur + [_new])

    if remove_button is not None and remove_button.value and remove_selector is not None:
        _idx = remove_selector.value
        set_traces([t for i, t in enumerate(get_traces()) if i != _idx])

    if clear_button.value:
        set_traces([])

    _traces = get_traces()
    if _traces:
        _header = "| # | Colour | Component | File | Receiver |"
        _sep = "|---|--------|-----------|------|----------|"
        _rows = "\n".join(
            f"| {i + 1} "
            f"| <span style='background:{t['colour']};display:inline-block;"
            f"width:32px;height:14px;border-radius:3px;'></span> "
            f"| `{t['component']}` "
            f"| `{t['filename']}` "
            f"| `{t['receiver']}` |"
            for i, t in enumerate(_traces)
        )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md(f"**{len(_traces)} active trace(s)**"),
                    mo.md(f"{_header}\n{_sep}\n{_rows}"),
                    mo.md("---"),
                ],
                gap="0.3rem",
            )
        )
    else:
        mo.output.replace(
            mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            "No traces added yet. "
                            "Use the picker above and click **＋ Add trace**."
                        ),
                        kind="neutral",
                    ),
                    mo.md("---"),
                ],
                gap="0.25rem",
            )
        )


# ── SECTION 4: Plot appearance controls ───────────────────────────────────
@app.cell
def _(mo):
    bg_colour = mo.ui.dropdown(
        options={
            "Light": "#ffffff",
            "Paper white": "#f8f9fa",
            "Dark": "#0e1117",
        },
        value="Light",
        label="Background",
    )
    line_width = mo.ui.slider(
        start=0.5,
        stop=4.0,
        step=0.5,
        value=1.5,
        label="Line width",
    )
    font_size = mo.ui.slider(
        start=10,
        stop=22,
        step=1,
        value=13,
        label="Font size",
    )
    show_grid = mo.ui.checkbox(label="Show grid", value=True)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 3 — Adjust appearance"),
                mo.hstack(
                    [bg_colour, line_width, font_size, show_grid],
                    gap="1.5rem",
                    justify="start",
                ),
            ],
            gap="0.5rem",
        )
    )
    return bg_colour, font_size, line_width, show_grid


# ── SECTION 5: Time zoom slider ────────────────────────────────────────────
@app.cell
def _(get_data, get_traces, mo):
    if not get_traces():
        mo.stop(True, mo.md(""))

    _files = get_data()["files"]
    _max_ns = max(
        round(f["meta"]["iterations"] * f["meta"]["dt"] * 1e9, 4) for f in _files.values()
    )
    _step = round(_max_ns / 600, 4)

    time_slider = mo.ui.range_slider(
        start=0.0,
        stop=_max_ns,
        step=_step,
        value=[0.0, _max_ns],
        label="Time window (ns)",
        full_width=True,
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 4 — Zoom time window"),
                mo.md(
                    "_Drag handles to focus on a specific time range. "
                    "CSV export always contains the full unzoomed data._"
                ),
                time_slider,
                mo.md("---"),
            ],
            gap="0.4rem",
        )
    )
    return (time_slider,)


# ── SECTION 6: Figure renderer + export ───────────────────────────────────
@app.cell
def _(
    bg_colour,
    csv,
    font_size,
    get_data,
    get_time_axis,
    get_trace,
    get_traces,
    get_unit_label,
    go,
    io,
    line_width,
    mo,
    show_grid,
    time_slider,
):
    _traces = get_traces()
    if not _traces:
        mo.stop(
            True,
            mo.callout(
                mo.md("Add at least one trace using the picker above " "to render the plot."),
                kind="warn",
            ),
        )

    _files = get_data()["files"]
    _bg = bg_colour.value
    _is_dark = _bg == "#0e1117"
    _fc = "#e8e8e8" if _is_dark else "#1a1a1a"
    _gc = "rgba(255,255,255,0.1)" if _is_dark else "rgba(0,0,0,0.07)"
    _lc = "rgba(255,255,255,0.25)" if _is_dark else "rgba(0,0,0,0.18)"

    # ── Build figure ────────────────────────────────────────────────────────
    _fig = go.Figure()
    _has_e = False
    _has_h = False
    _comps_seen = []
    _files_seen = []

    # Store full arrays for CSV export (unaffected by zoom)
    _csv_time: dict = {}
    _csv_data: dict = {}

    for _t in _traces:
        _fdata = _files[_t["filename"]]
        _arr = get_trace(_fdata, _t["component"], _t["receiver"])
        _time = get_time_axis(_fdata, unit="ns")
        _unit = get_unit_label(_t["component"])
        _on_y2 = _unit == "A/m"

        if _on_y2:
            _has_h = True
        else:
            _has_e = True

        # Track unique time axes by (n_steps, dt) tuple
        _fmeta = _fdata["meta"]
        _tax_key = (_fmeta["iterations"], round(_fmeta["dt"], 15))
        if _tax_key not in _csv_time:
            _csv_time[_tax_key] = _time
        _csv_data[_t["label"]] = (_tax_key, _arr)

        _fig.add_trace(
            go.Scatter(
                x=_time,
                y=_arr,
                mode="lines",
                name=_t["label"],
                line=dict(
                    color=_t["colour"],
                    width=line_width.value,
                    dash="dash" if _on_y2 else "solid",
                ),
                yaxis="y2" if _on_y2 else "y",
                hovertemplate=(
                    "%{x:.4f} ns<br>%{y:.5g} " + _unit + "<extra>" + _t["label"] + "</extra>"
                ),
            )
        )

        if _t["component"] not in _comps_seen:
            _comps_seen.append(_t["component"])
        if _t["filename"] not in _files_seen:
            _files_seen.append(_t["filename"])

    _title = ", ".join(_comps_seen) + "  ·  " + "  +  ".join(_files_seen)
    _dual = _has_e and _has_h

    _axis_base = dict(
        showgrid=show_grid.value,
        gridcolor=_gc,
        gridwidth=0.5,
        showline=True,
        linecolor=_lc,
        linewidth=1,
        tickfont=dict(size=font_size.value - 1, color=_fc),
        title_font=dict(size=font_size.value, color=_fc),
        zeroline=True,
        zerolinecolor=_lc,
        zerolinewidth=1,
    )

    if _dual:
        _y1_label = "E-field [V/m]"
    elif _has_e:
        _y1_label = "Field strength [V/m]"
    else:
        _y1_label = "Field strength [A/m]"

    _t_start, _t_end = time_slider.value

    _layout = dict(
        title=dict(
            text=_title,
            font=dict(size=font_size.value + 1, color=_fc),
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(
            title="Time (ns)",
            range=[_t_start, _t_end],
            **_axis_base,
        ),
        yaxis=dict(title=_y1_label, **_axis_base),
        paper_bgcolor=_bg,
        plot_bgcolor=_bg,
        legend=dict(
            font=dict(size=font_size.value - 1, color=_fc),
            bgcolor="rgba(0,0,0,0.0)",
            bordercolor=_lc,
            borderwidth=1,
        ),
        height=500,
        margin=dict(l=72, r=80 if _dual else 32, t=55, b=60),
        hovermode="x unified",
    )

    if _dual:
        _layout["yaxis2"] = dict(
            title="H-field [A/m]",
            overlaying="y",
            side="right",
            showgrid=False,
            showline=True,
            linecolor=_lc,
            linewidth=1,
            tickfont=dict(size=font_size.value - 1, color=_fc),
            title_font=dict(size=font_size.value, color=_fc),
            zeroline=False,
        )

    _fig.update_layout(**_layout)

    # ── CSV export ──────────────────────────────────────────────────────────
    # Always exports FULL data (not the zoomed window).
    # If all traces share the same time axis: one time column.
    # If traces come from files with different dt/iterations: paired columns.
    try:
        _buf = io.StringIO()
        _writer = csv.writer(_buf)

        _single_axis = len(_csv_time) == 1

        if _single_axis:
            _shared_time = list(_csv_time.values())[0]
            _header_row = ["time_ns"] + [t["label"] for t in _traces]
            _writer.writerow(_header_row)
            for _i in range(len(_shared_time)):
                _row = [f"{_shared_time[_i]:.8g}"] + [
                    f"{_csv_data[t['label']][1][_i]:.10g}" for t in _traces
                ]
                _writer.writerow(_row)
        else:
            # Paired time + value columns per trace
            _header_row = []
            for _t in _traces:
                _header_row.extend([f"time_ns ({_t['label']})", _t["label"]])
            _writer.writerow(_header_row)
            _max_len = max(len(_csv_data[t["label"]][1]) for t in _traces)
            for _i in range(_max_len):
                _row = []
                for _t in _traces:
                    _tax_key, _arr = _csv_data[_t["label"]]
                    _time = _csv_time[_tax_key]
                    if _i < len(_arr):
                        _row.extend([f"{_time[_i]:.8g}", f"{_arr[_i]:.10g}"])
                    else:
                        _row.extend(["", ""])
                _writer.writerow(_row)

        _csv_bytes = _buf.getvalue().encode("utf-8")
        _csv_btn = mo.download(
            data=_csv_bytes,
            filename="gprmax_ascan.csv",
            label="Download CSV (full data)",
            mimetype="text/csv",
        )
    except Exception as _e:
        _csv_btn = mo.md(f"_CSV export error: `{_e}`_")

    # ── SVG export ──────────────────────────────────────────────────────────
    try:
        import plotly.io as _pio

        _svg_bytes = _pio.to_image(_fig, format="svg", width=1200, height=600, scale=2)
        _svg_btn = mo.download(
            data=_svg_bytes,
            filename="gprmax_ascan.svg",
            label="Download SVG",
            mimetype="image/svg+xml",
        )
    except Exception as _e:
        _svg_btn = mo.md(
            f"_SVG: `{type(_e).__name__}` — "
            f"run `plotly_get_chrome` to install kaleido's Chrome, "
            f"or use the camera icon in the plot toolbar._"
        )

    # ── PDF export ──────────────────────────────────────────────────────────
    try:
        import plotly.io as _pio

        _pdf_bytes = _pio.to_image(_fig, format="pdf", width=1200, height=600, scale=2)
        _pdf_btn = mo.download(
            data=_pdf_bytes,
            filename="gprmax_ascan.pdf",
            label="Download PDF",
            mimetype="application/pdf",
        )
    except Exception as _e:
        _pdf_btn = mo.md(f"_PDF: `{type(_e).__name__}` — run `plotly_get_chrome` first._")

    # ── Interactive HTML — no kaleido needed ────────────────────────────────
    try:
        _html_bytes = _fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8")
        _html_btn = mo.download(
            data=_html_bytes,
            filename="gprmax_ascan.html",
            label="Download HTML (interactive)",
            mimetype="text/html",
        )
    except Exception:
        _html_btn = mo.md("")

    # ── Notes ────────────────────────────────────────────────────────────────
    _notes = []
    if _dual:
        _notes.append(
            "_Solid lines → E-field (left axis, V/m)  ·  "
            "Dashed lines → H-field (right axis, A/m)_"
        )
    _notes.append(
        "_Hover for exact values. "
        "Camera icon in toolbar → SVG (no kaleido needed via toolbar). "
        "Run `plotly_get_chrome` once to enable SVG/PDF download buttons._"
    )

    mo.vstack(
        [
            mo.md("### Plot"),
            mo.ui.plotly(
                _fig,
                config={
                    "toImageButtonOptions": {
                        "format": "svg",
                        "filename": "gprmax_ascan",
                        "height": 600,
                        "width": 1200,
                        "scale": 2,
                    },
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            ),
            mo.md("  \n".join(_notes)),
            mo.md("**Export**"),
            mo.hstack(
                [_csv_btn, _html_btn, _svg_btn, _pdf_btn],
                gap="1rem",
                justify="start",
            ),
            mo.md("---"),
            mo.callout(
                mo.md(
                    "**Upcoming features**\n\n"
                    "- Background subtraction "
                    "(target trace minus free-space trace)\n"
                    "- FFT frequency spectrum toggle\n"
                    "- Assemble multiple A-scan files → B-scan radargram\n"
                    "- 3D surface viewer from multi-position A-scan data"
                ),
                kind="neutral",
            ),
        ],
        gap="0.5rem",
    )
    return


if __name__ == "__main__":
    app.run()
