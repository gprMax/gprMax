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
    from toolboxes.Marimo.processing import (
        GAIN_KINDS,
        apply_gain,
        fft_spectrum,
        gain_label,
        spectrum_view_limit,
        subtract_traces,
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
        GAIN_KINDS,
        PALETTE,
        Path,
        apply_gain,
        build_label,
        csv,
        fft_spectrum,
        format_metadata_text,
        gain_label,
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
        spectrum_view_limit,
        subtract_traces,
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


# ── SECTION 2b: Background subtraction reference ──────────────────────────
# Depends on the loaded files and nothing else. Tying it to the trace list
# would rebuild the dropdown, and reset the choice, on every added trace.
@app.cell
def _(get_data, mo):
    _file_names = list(get_data()["files"].keys())

    if len(_file_names) < 2:
        subtract_ref = None
        mo.output.replace(
            mo.md(
                "_Load a second file, a free-space run of the same model with the "
                "target removed, to enable background subtraction._"
            )
            if _file_names
            else mo.md("")
        )
    else:
        subtract_ref = mo.ui.dropdown(
            options={"None": "", **{n: n for n in _file_names}},
            value="None",
            label="Subtract reference file",
        )
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("### Step 1b — Background subtraction"),
                    mo.md(
                        "_Pick a free-space run of the same model. For every plotted "
                        "trace, the matching receiver and component from that file is "
                        "subtracted, which cancels the source pulse and the direct "
                        "coupling and leaves the target response. Display only: CSV "
                        "export stays raw._"
                    ),
                    subtract_ref,
                    mo.md("---"),
                ],
                gap="0.4rem",
            )
        )
    return (subtract_ref,)


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
                mo.hstack([remove_selector, remove_button], gap="1rem", justify="start"),
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
                mo.hstack([file_selector, receiver_selector], gap="2rem", justify="start"),
                mo.md("**Select component and colour**"),
                mo.hstack([component_selector, colour_selector], gap="2rem", justify="start"),
                mo.hstack([add_button, clear_button], gap="1rem", justify="start"),
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
        debounce=True,
    )
    font_size = mo.ui.slider(
        start=10,
        stop=22,
        step=1,
        value=13,
        label="Font size",
        debounce=True,
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
        debounce=True,
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


# ── SECTION 5b: Gain kind ─────────────────────────────────────────────────
# Widget creation only. Its .value is read in the next cell, which builds the
# parameter sliders for whichever kind is selected — reading .value in the
# cell that creates the widget raises at runtime and marimo check misses it.
@app.cell
def _(GAIN_KINDS, get_traces, mo):
    if not get_traces():
        mo.stop(True, mo.md(""))

    gain_kind = mo.ui.dropdown(
        options={_meta["label"]: _k for _k, _meta in GAIN_KINDS.items()},
        value="None",
        label="Gain function",
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 5 — Gain"),
                mo.md(
                    "_Time-varying amplification to compensate spherical spreading "
                    "and attenuation. Power is `t^b`, exponential is `exp(a·t)`, "
                    "and SEC is their product, the gain most published GPR "
                    "processing flows use. Display only: CSV export stays raw._"
                ),
                gain_kind,
            ],
            gap="0.4rem",
        )
    )
    return (gain_kind,)


# ── SECTION 5c: Gain parameters ───────────────────────────────────────────
# Rebuilt per kind: `factor` means a multiplier, a slope, a rate, or dB per ns
# depending on the kind, so one fixed slider range would be wrong for most of
# them. All widgets are None when no gain is selected, same pattern the trace
# picker uses for its remove controls.
#
# Depends on gain_kind alone, deliberately. Deriving the gain-start range from
# the loaded time window would make this cell depend on get_traces through the
# zoom slider, so adding a trace would rebuild these widgets and silently reset
# every gain setting. Start is a percentage of the window instead; the renderer
# resolves it against the real time axis and the resolved ns lands in the title.
@app.cell
def _(GAIN_KINDS, gain_kind, mo):
    _FACTOR_SLIDERS = {
        "constant": dict(start=0.1, stop=20.0, step=0.1, value=2.0, label="Factor (x)"),
        "linear": dict(start=0.0, stop=10.0, step=0.1, value=1.0, label="Slope (per ns)"),
        "exponential": dict(start=0.0, stop=5.0, step=0.05, value=1.0, label="Rate a (per ns)"),
        "db": dict(start=0.0, stop=60.0, step=0.5, value=6.0, label="Gain (dB per ns)"),
        "sec": dict(start=0.0, stop=5.0, step=0.05, value=0.5, label="Rate a (per ns)"),
    }

    _kind = gain_kind.value

    if _kind == "none":
        gain_factor = None
        gain_power = None
        gain_start = None
        gain_clamp = None
        gain_max = None
        show_gain_curve = None
        mo.output.replace(mo.md(""))
    else:
        _cfg = _FACTOR_SLIDERS.get(_kind)
        gain_factor = mo.ui.slider(**_cfg, debounce=True) if _cfg else None
        gain_power = (
            mo.ui.slider(start=0.0, stop=4.0, step=0.1, value=1.0, label="Exponent b", debounce=True)
            if GAIN_KINDS[_kind]["uses_power"]
            else None
        )
        gain_start = mo.ui.slider(
            start=0.0,
            stop=100.0,
            step=0.5,
            value=0.0,
            label="Gain start (% of window)",
            debounce=True,
        )
        gain_clamp = mo.ui.checkbox(label="Limit maximum gain", value=True)
        gain_max = mo.ui.slider(
            start=1, stop=1000, step=1, value=100, label="Maximum gain (x)", debounce=True
        )
        show_gain_curve = mo.ui.checkbox(label="Show gain curve", value=True)

        _params = [w for w in (gain_factor, gain_power, gain_start) if w is not None]
        mo.output.replace(
            mo.vstack(
                [
                    mo.hstack(_params, gap="1.5rem", justify="start"),
                    mo.hstack([gain_clamp, gain_max, show_gain_curve], gap="1.5rem", justify="start"),
                    mo.md(
                        "_Time before the gain start keeps a gain of exactly 1, so the "
                        "direct wave can be left alone rather than saturating the plot._"
                    ),
                    mo.md("---"),
                ],
                gap="0.5rem",
            )
        )
    return gain_clamp, gain_factor, gain_max, gain_power, gain_start, show_gain_curve


# ── SECTION 5d: Domain ────────────────────────────────────────────────────
# No data dependency, so switching domain or reference never disturbs the
# gain settings and vice versa.
@app.cell
def _(get_traces, mo):
    if not get_traces():
        mo.stop(True, mo.md(""))

    domain = mo.ui.dropdown(
        options={"Time": "time", "Frequency (FFT)": "freq"},
        value="Time",
        label="Domain",
    )
    fft_reference = mo.ui.dropdown(
        options={"Shared (compare amplitudes)": "shared", "Per trace (each peaks at 0 dB)": "own"},
        value="Shared (compare amplitudes)",
        label="dB reference",
    )
    fft_full_range = mo.ui.checkbox(label="Show full spectrum to Nyquist", value=False)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 6 — Domain"),
                mo.md(
                    "_The spectrum comes from gprMax's own `fft_power`, so it matches "
                    "`plot_Ascan.py`. That function normalises every trace against its "
                    "own peak, which makes a loud trace and a quiet one look identical; "
                    "the shared reference undoes that so overlaid spectra are "
                    "comparable. Traces in different units (V/m against A/m) are "
                    "referenced within their own unit, since a decibel difference "
                    "across units means nothing._"
                ),
                mo.hstack([domain, fft_reference, fft_full_range], gap="1.5rem", justify="start"),
                mo.md("---"),
            ],
            gap="0.4rem",
        )
    )
    return domain, fft_full_range, fft_reference


# ── SECTION 6: Figure renderer + export ───────────────────────────────────
@app.cell
def _(
    apply_gain,
    bg_colour,
    csv,
    domain,
    fft_full_range,
    fft_reference,
    fft_spectrum,
    font_size,
    gain_clamp,
    gain_factor,
    gain_kind,
    gain_label,
    gain_max,
    gain_power,
    gain_start,
    get_data,
    get_time_axis,
    get_trace,
    get_traces,
    get_unit_label,
    go,
    io,
    line_width,
    mo,
    np,
    show_gain_curve,
    show_grid,
    spectrum_view_limit,
    subtract_ref,
    subtract_traces,
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
    _freq_mode = domain.value == "freq"

    # Background subtraction. A data operation rather than a display one, so
    # unlike gain it also applies to the spectrum: the point of subtracting is
    # that the difference is what you want to analyse. CSV still exports the
    # untouched file contents, with the operation recorded on the first line.
    _ref_name = subtract_ref.value if subtract_ref is not None else ""
    _subtracted = []
    _sub_warnings = []

    def _apply_subtraction(arr, trace, fdata):
        if not _ref_name or trace["filename"] == _ref_name:
            return arr
        _ref_fdata = _files[_ref_name]
        try:
            _ref_arr = get_trace(_ref_fdata, trace["component"], trace["receiver"])
            _out = subtract_traces(
                arr, _ref_arr, fdata["meta"]["dt"], _ref_fdata["meta"]["dt"]
            )
        except (KeyError, ValueError) as _err:
            _sub_warnings.append(f"{trace['label']}: {_err}")
            return arr
        _subtracted.append(trace["label"])
        return _out

    # ── Gain settings ───────────────────────────────────────────────────────
    # Every parameter widget is None when gain is off, and gain_factor is also
    # None for the power kind, which has no factor of its own.
    _kind = gain_kind.value
    _gain_on = _kind != "none" and not _freq_mode
    _factor = gain_factor.value if gain_factor is not None else 1.0
    _power = gain_power.value if gain_power is not None else 1.0
    _max_gain = gain_max.value if (gain_clamp is not None and gain_clamp.value) else None
    _window_ns = max(
        f["meta"]["iterations"] * f["meta"]["dt"] * 1e9 for f in _files.values()
    )
    _start = (gain_start.value / 100.0) * _window_ns if gain_start is not None else 0.0

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
    _files_seen = []
    _comps_seen = []
    for _t in _traces:
        if _t["component"] not in _comps_seen:
            _comps_seen.append(_t["component"])
        if _t["filename"] not in _files_seen:
            _files_seen.append(_t["filename"])
    _source_title = ", ".join(_comps_seen) + "  ·  " + "  +  ".join(_files_seen)

    _fig = go.Figure()
    _gain_panel = []
    _flat_traces = []
    _csv_x: dict = {}
    _csv_data: dict = {}
    _gain_curves: dict = {}

    if _freq_mode:
        # ── Frequency domain ────────────────────────────────────────────────
        # Always transforms the raw trace, never the gained one. Gain
        # multiplies in time, which convolves in frequency, so the centre
        # frequency read off a gained spectrum would be wrong — and checking
        # that the antenna behaved as specified is the reason for this view.
        _spectra = []
        for _t in _traces:
            _fdata = _files[_t["filename"]]
            _raw = get_trace(_fdata, _t["component"], _t["receiver"])
            _dt = _fdata["meta"]["dt"]
            _freqs, _pdb, _peak_db = fft_spectrum(
                _apply_subtraction(_raw, _t, _fdata), _dt
            )
            if _peak_db is None:
                _flat_traces.append(_t["label"])
                continue
            _spectra.append(
                dict(
                    trace=_t,
                    freqs=_freqs,
                    power=_pdb,
                    peak_db=_peak_db,
                    unit=get_unit_label(_t["component"]),
                )
            )

        if not _spectra:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        "**Nothing to transform.** Every selected trace is all "
                        "zeros, so it has no spectrum: "
                        + ", ".join(f"`{n}`" for n in _flat_traces)
                        + ". A component that isn't excited by the model (Ex in a "
                        "2D TMz run, for example) is stored as zeros."
                    ),
                    kind="warn",
                ),
            )

        # Shared reference per unit family. Referencing a V/m trace against an
        # A/m one would report a decibel difference between different
        # quantities, which is not a physical statement.
        _refs = {}
        if fft_reference.value == "shared":
            for _s in _spectra:
                _refs[_s["unit"]] = max(_refs.get(_s["unit"], -np.inf), _s["peak_db"])

        _limit_hz = 0.0
        for _s in _spectra:
            _offset = (
                _s["peak_db"] - _refs[_s["unit"]] if fft_reference.value == "shared" else 0.0
            )
            _shown = _s["power"] + _offset
            _t = _s["trace"]
            _key = (len(_s["freqs"]), round(float(_s["freqs"][1] - _s["freqs"][0]), 6))
            if _key not in _csv_x:
                _csv_x[_key] = _s["freqs"]
            _csv_data[_t["label"]] = (_key, _shown)
            _limit_hz = max(_limit_hz, spectrum_view_limit(_s["freqs"], _s["power"]))
            _fig.add_trace(
                go.Scatter(
                    x=_s["freqs"] / 1e9,
                    y=_shown,
                    mode="lines",
                    name=_t["label"],
                    line=dict(color=_t["colour"], width=line_width.value),
                    hovertemplate=(
                        "%{x:.4g} GHz<br>%{y:.4g} dB<extra>" + _t["label"] + "</extra>"
                    ),
                )
            )

        _x_range = (
            None
            if fft_full_range.value
            else [0.0, max(_limit_hz / 1e9, 1e-6)]
        )
        _ref_text = (
            "shared reference per unit"
            if fft_reference.value == "shared"
            else "each trace at its own 0 dB"
        )
        # Title names only what is actually drawn. _source_title covers every
        # selected trace, including any dropped for having no spectrum, and an
        # exported figure must not claim a component that isn't in it.
        _plot_comps = []
        _plot_files = []
        for _s in _spectra:
            if _s["trace"]["component"] not in _plot_comps:
                _plot_comps.append(_s["trace"]["component"])
            if _s["trace"]["filename"] not in _plot_files:
                _plot_files.append(_s["trace"]["filename"])
        _title = (
            ", ".join(_plot_comps)
            + "  ·  "
            + "  +  ".join(_plot_files)
            + f"  ·  power spectrum ({_ref_text})"
        )
        if _subtracted:
            _title += f"  ·  minus {_ref_name}"
        _layout = dict(
            title=dict(
                text=_title,
                font=dict(size=font_size.value + 1, color=_fc),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(title="Frequency (GHz)", range=_x_range, **_axis_base),
            yaxis=dict(title="Power (dB)", **_axis_base),
            paper_bgcolor=_bg,
            plot_bgcolor=_bg,
            legend=dict(
                font=dict(size=font_size.value - 1, color=_fc),
                bgcolor="rgba(0,0,0,0.0)",
                bordercolor=_lc,
                borderwidth=1,
            ),
            height=500,
            margin=dict(l=72, r=32, t=55, b=60),
            hovermode="x unified",
        )
        _fig.update_layout(**_layout)
        _csv_first_col = "freq_hz"
        _csv_filename = "gprmax_ascan_spectrum.csv"
        _csv_comment = (
            f"# domain: frequency ({_ref_text}). "
            f"Power in dB, full positive spectrum, view range not applied."
        )
        if _subtracted:
            _csv_comment += f" Displayed spectra had {_ref_name} subtracted."
        _gain_text = "no gain"
        _dual = False

    else:
        # ── Time domain ─────────────────────────────────────────────────────
        _has_e = False
        _has_h = False

        for _t in _traces:
            _fdata = _files[_t["filename"]]
            _raw = get_trace(_fdata, _t["component"], _t["receiver"])
            _time = get_time_axis(_fdata, unit="ns")
            _unit = get_unit_label(_t["component"])
            _on_y2 = _unit == "A/m"

            if _on_y2:
                _has_h = True
            else:
                _has_e = True

            _fmeta = _fdata["meta"]
            _tax_key = (_fmeta["iterations"], round(_fmeta["dt"], 15))
            if _tax_key not in _csv_x:
                _csv_x[_tax_key] = _time
            _csv_data[_t["label"]] = (_tax_key, _raw)

            _base = _apply_subtraction(_raw, _t, _fdata)

            if _gain_on:
                _arr, _curve = apply_gain(
                    _base,
                    _time,
                    _kind,
                    factor=_factor,
                    power=_power,
                    start_ns=_start,
                    max_gain=_max_gain,
                )
                if _tax_key not in _gain_curves:
                    _gain_curves[_tax_key] = (_time, _curve)
            else:
                _arr = _base

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

        # Only report the clamp when it actually bound. Naming a limit the gain
        # never reached puts a misleading number in an exported figure title.
        _clamp_bound = _max_gain is not None and any(
            float(np.max(_c)) >= _max_gain for _, _c in _gain_curves.values()
        )
        _gain_text = gain_label(_kind, _factor, _power, _start, _max_gain if _clamp_bound else None)

        _title = _source_title
        if _subtracted:
            _title += f"  ·  minus {_ref_name}"
        if _gain_on:
            _title += "  ·  " + _gain_text
        _dual = _has_e and _has_h

        if _dual:
            _y1_label = "E-field [V/m]"
        elif _has_e:
            _y1_label = "Field strength [V/m]"
        else:
            _y1_label = "Field strength [A/m]"
        if _gain_on:
            _y1_label += " (gained)"

        _t_start, _t_end = time_slider.value

        _layout = dict(
            title=dict(
                text=_title,
                font=dict(size=font_size.value + 1, color=_fc),
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(title="Time (ns)", range=[_t_start, _t_end], **_axis_base),
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
                title="H-field [A/m]" + (" (gained)" if _gain_on else ""),
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
        _csv_first_col = "time_ns"
        _csv_filename = "gprmax_ascan.csv"
        _csv_comment = f"# gain: {_gain_text.replace(',', ';')}. Values below are raw."
        if _subtracted:
            _csv_comment += f" Plot had {_ref_name} subtracted; values below have not."

        # ── Gain curve panel ────────────────────────────────────────────────
        # Its own figure rather than a third overlaying y-axis on the main
        # plot. The right axis is already taken whenever an H-field trace is
        # active, so sharing it would make the curve appear or vanish
        # depending on which traces happen to be added. A separate panel is
        # also how RADAN and ReflexW present gain curves.
        if _gain_on and show_gain_curve is not None and show_gain_curve.value and _gain_curves:
            _gfig = go.Figure()
            for _gt, _gcurve in _gain_curves.values():
                _gfig.add_trace(
                    go.Scatter(
                        x=_gt,
                        y=_gcurve,
                        mode="lines",
                        name="gain",
                        showlegend=False,
                        line=dict(color="#7f7f7f", width=line_width.value),
                        hovertemplate="%{x:.4f} ns<br>x%{y:.4g}<extra>gain</extra>",
                    )
                )
            _all_gain = np.concatenate([c for _, c in _gain_curves.values()])
            _gfig.update_layout(
                xaxis=dict(
                    range=[_t_start, _t_end],
                    showgrid=show_grid.value,
                    gridcolor=_gc,
                    showline=True,
                    linecolor=_lc,
                    tickfont=dict(size=font_size.value - 2, color=_fc),
                ),
                yaxis=dict(
                    title="Gain (x)",
                    type="log" if float(np.min(_all_gain)) > 0 else "linear",
                    showgrid=show_grid.value,
                    gridcolor=_gc,
                    showline=True,
                    linecolor=_lc,
                    tickfont=dict(size=font_size.value - 2, color=_fc),
                    title_font=dict(size=font_size.value - 1, color=_fc),
                ),
                paper_bgcolor=_bg,
                plot_bgcolor=_bg,
                height=170,
                margin=dict(l=72, r=80 if _dual else 32, t=10, b=34),
            )
            _gain_panel = [
                mo.md(f"**Applied gain curve** — {_gain_text}"),
                mo.ui.plotly(_gfig, config={"displaylogo": False, "displayModeBar": False}),
            ]

    # ── CSV export ──────────────────────────────────────────────────────────
    # Always the full unzoomed data for whichever domain is on screen: raw
    # amplitudes in time, full positive spectrum in frequency. The first line
    # is written in both cases so the format never changes silently, and
    # commas inside it become semicolons to keep it one unquoted field.
    # One shared x column when every trace has the same axis, paired columns
    # when they don't.
    try:
        _buf = io.StringIO()
        _buf.write(_csv_comment.replace(",", ";") + "\n")
        _writer = csv.writer(_buf)
        _single_axis = len(_csv_x) == 1

        if _single_axis:
            _shared_x = list(_csv_x.values())[0]
            _writer.writerow([_csv_first_col] + [t["label"] for t in _traces if t["label"] in _csv_data])
            _cols = [t for t in _traces if t["label"] in _csv_data]
            for _i in range(len(_shared_x)):
                _writer.writerow(
                    [f"{_shared_x[_i]:.8g}"]
                    + [f"{_csv_data[t['label']][1][_i]:.10g}" for t in _cols]
                )
        else:
            _cols = [t for t in _traces if t["label"] in _csv_data]
            _header_row = []
            for _t in _cols:
                _header_row.extend([f"{_csv_first_col} ({_t['label']})", _t["label"]])
            _writer.writerow(_header_row)
            _max_len = max(len(_csv_data[t["label"]][1]) for t in _cols)
            for _i in range(_max_len):
                _row = []
                for _t in _cols:
                    _k, _vals = _csv_data[_t["label"]]
                    _xs = _csv_x[_k]
                    if _i < len(_vals):
                        _row.extend([f"{_xs[_i]:.8g}", f"{_vals[_i]:.10g}"])
                    else:
                        _row.extend(["", ""])
                _writer.writerow(_row)

        _csv_btn = mo.download(
            data=_buf.getvalue().encode("utf-8"),
            filename=_csv_filename,
            label="Download CSV (full data)",
            mimetype="text/csv",
        )
    except Exception as _e:
        _csv_btn = mo.md(f"_CSV export error: `{_e}`_")

    # ── SVG/PDF export ─────────────────────────────────────────────────────
    # Lazy on purpose: mo.download accepts a zero-arg callable and only
    # calls it when the button is clicked. kaleido spins up headless Chrome
    # per export, which is slow — computing it eagerly here meant this cell
    # paid that cost twice on every re-render, which includes every
    # appearance-control change and every drag tick of an undebounced
    # slider. Same bug and same fix as bscan_dashboard.py.
    _stem = _csv_filename.rsplit(".", 1)[0]

    def _make_svg():
        import plotly.io as _pio

        return _pio.to_image(_fig, format="svg", width=1200, height=600, scale=2)

    def _make_pdf():
        import plotly.io as _pio

        return _pio.to_image(_fig, format="pdf", width=1200, height=600, scale=2)

    def _make_html():
        return _fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8")

    _svg_btn = mo.download(
        data=_make_svg, filename=f"{_stem}.svg", label="Download SVG", mimetype="image/svg+xml"
    )
    _pdf_btn = mo.download(
        data=_make_pdf, filename=f"{_stem}.pdf", label="Download PDF", mimetype="application/pdf"
    )
    _html_btn = mo.download(
        data=_make_html,
        filename=f"{_stem}.html",
        label="Download HTML (interactive)",
        mimetype="text/html",
    )

    # ── Notes ────────────────────────────────────────────────────────────────
    _notes = []
    _warn = []
    if _sub_warnings:
        _warn.append(
            mo.callout(
                mo.md(
                    "**Could not subtract the reference from every trace:**\n\n"
                    + "\n".join(f"- {w}" for w in _sub_warnings)
                ),
                kind="warn",
            )
        )
    if _subtracted:
        _notes.append(
            f"_`{_ref_name}` subtracted from {len(_subtracted)} trace(s). "
            "Any trace from that file itself is left alone._"
        )
    if _freq_mode:
        _notes.append(
            "_Spectra are computed from the raw traces. Gain is a time-domain "
            "correction and reshapes a spectrum, so it is not applied here._"
        )
        if _flat_traces:
            _warn.append(
                mo.callout(
                    mo.md(
                        "**Skipped, no spectrum:** "
                        + ", ".join(f"`{n}`" for n in _flat_traces)
                        + ". These traces are all zeros, which `fft_power` would "
                        "otherwise render as a flat 0 dB line indistinguishable "
                        "from real data."
                    ),
                    kind="warn",
                )
            )
    else:
        if _dual:
            _notes.append(
                "_Solid lines → E-field (left axis, V/m)  ·  "
                "Dashed lines → H-field (right axis, A/m)_"
            )
        if _gain_on:
            _notes.append(
                "_Plot and image exports show gained amplitudes. "
                "CSV stays raw, with the gain recorded on its first line._"
            )
    _notes.append(
        "_Hover for exact values. "
        "Camera icon in toolbar → SVG (no kaleido needed via toolbar). "
        "Run `plotly_get_chrome` once to enable SVG/PDF download buttons._"
    )

    mo.vstack(
        [
            mo.md("### Plot"),
            *_warn,
            mo.ui.plotly(
                _fig,
                config={
                    "toImageButtonOptions": {
                        "format": "svg",
                        "filename": _stem,
                        "height": 600,
                        "width": 1200,
                        "scale": 2,
                    },
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            ),
            *_gain_panel,
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
                    "**Elsewhere in the toolbox**\n\n"
                    "- `bscan_dashboard.py` assembles a radargram from multiple "
                    "A-scan files or watches one being written, with a 3D surface "
                    "view, gain, and mean-trace background removal for the case "
                    "where no free-space run exists.\n"
                    "- `recipes/ascan_workflow.py` goes from model parameters "
                    "through a solver run to this dashboard's input in one "
                    "notebook.\n"
                    "- `recipes/velocity_permittivity.py` recovers a soil "
                    "permittivity from the shape of a reflection hyperbola."
                ),
                kind="neutral",
            ),
        ],
        gap="0.5rem",
    )
    return


if __name__ == "__main__":
    app.run()
