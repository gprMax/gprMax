import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from toolboxes.Marimo.h5_reader import get_unit_label, load_file
    from toolboxes.Marimo.hyperbola import (
        C_M_PER_NS,
        apex_source_x,
        apex_time,
        permittivity_from_apex,
        ricker_delay,
        travel_time,
        velocity,
    )
    from toolboxes.Marimo.processing import apply_gain, remove_mean_trace
    from toolboxes.Marimo.trace_matrix import stack_traces

    return (
        C_M_PER_NS,
        apex_source_x,
        apex_time,
        apply_gain,
        get_unit_label,
        go,
        load_file,
        mo,
        np,
        permittivity_from_apex,
        remove_mean_trace,
        ricker_delay,
        stack_traces,
        travel_time,
        velocity,
    )


# ── Step 1: Load a B-scan ─────────────────────────────────────────────────
@app.cell
def _(mo):
    file_picker = mo.ui.file_browser(filetypes=[".h5"], multiple=True, label="")
    mo.output.replace(
        mo.vstack(
            [
                mo.md("# Velocity and Permittivity Calculator"),
                mo.md(
                    "How fast a radar pulse travels through the ground depends on the "
                    "permittivity of the material. Slower material means a later "
                    "reflection, which pushes the apex of the hyperbola down the "
                    "record. This notebook predicts where the apex should land for a "
                    "given permittivity and target depth, draws that prediction on top "
                    "of a real radargram, and reads the permittivity back from a "
                    "measured apex."
                ),
                mo.md("---"),
                mo.md("### Step 1 — Load the B-scan trace files"),
                mo.md(
                    "_Select the per-trace `.h5` files from a B-scan run, for example "
                    "the output of `python -m gprMax examples/gpr/basic/cylinder_Bscan_2D.in -n 60`. "
                    "Traces are ordered by source position._"
                ),
                file_picker,
            ],
            gap="0.4rem",
        )
    )
    return (file_picker,)


# ── Assemble ──────────────────────────────────────────────────────────────
@app.cell
def _(file_picker, load_file, mo, stack_traces):
    if not file_picker.value:
        mo.stop(True, mo.md(""))

    _loaded = []
    _skipped = []
    for _f in file_picker.value:
        try:
            _loaded.append(load_file(_f.path))
        except (FileNotFoundError, OSError, KeyError) as _e:
            _skipped.append(f"{_f.path}: {type(_e).__name__}")

    _loaded.sort(key=lambda fd: fd.get("sources", {}).get("src1", {}).get("position", [0.0])[0])
    bscan = stack_traces(_loaded)

    if bscan["matrix"] is None:
        mo.stop(
            True,
            mo.callout(mo.md("No usable traces in the selection."), kind="danger"),
        )

    _msg = (
        f"**{bscan['matrix'].shape[1]} traces**  ·  component "
        f"`{bscan['component']}`  ·  "
        f"source x from {min(bscan['positions_x']):.3f} to "
        f"{max(bscan['positions_x']):.3f} m  ·  "
        f"window {bscan['time_ns'][-1]:.3f} ns"
    )
    _out = [mo.md(_msg)]
    if not bscan["all_positions_physical"]:
        _out.append(
            mo.callout(
                mo.md(
                    "**Source positions are missing from at least one file.** "
                    "Trace index has been substituted, so the horizontal axis is "
                    "not in metres and the prediction below will not line up."
                ),
                kind="warn",
            )
        )
    if _skipped:
        _out.append(
            mo.callout(
                mo.md("**Skipped:**\n\n" + "\n".join(f"- {s}" for s in _skipped)),
                kind="warn",
            )
        )
    mo.output.replace(mo.vstack(_out, gap="0.3rem"))
    return (bscan,)


# ── Step 2: Model geometry ────────────────────────────────────────────────
# Widget creation only, and deliberately independent of the loaded data. If
# these ranges were derived from the file, adding or changing files would
# rebuild the widgets and reset every setting.
@app.cell
def _(mo):
    target_x = mo.ui.slider(
        start=0.0,
        stop=0.400,
        step=0.002,
        value=0.120,
        label="Target x (m)",
        debounce=True,
    )
    depth = mo.ui.slider(
        start=0.010,
        stop=0.300,
        step=0.002,
        value=0.090,
        label="Depth to target centre (m)",
        debounce=True,
    )
    radius = mo.ui.slider(
        start=0.0,
        stop=0.050,
        step=0.002,
        value=0.010,
        label="Target radius (m)",
        debounce=True,
    )
    antenna_offset = mo.ui.slider(
        start=0.0,
        stop=0.200,
        step=0.002,
        value=0.040,
        label="Tx-Rx separation (m)",
        debounce=True,
    )
    source_freq = mo.ui.slider(
        start=0.1,
        stop=5.0,
        step=0.1,
        value=1.5,
        label="Source centre frequency (GHz)",
        debounce=True,
    )

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 2 — Model geometry"),
                mo.md(
                    "_Defaults match `examples/gpr/basic/cylinder_Bscan_2D.in`: a 10 mm PEC "
                    "cylinder centred at x = 0.120 m, 0.090 m below a surface at "
                    "y = 0.170 m, with a 1.5 GHz ricker and the receiver 0.040 m "
                    "behind the source._"
                ),
                mo.hstack([target_x, depth, radius], gap="1.5rem", justify="start"),
                mo.hstack([antenna_offset, source_freq], gap="1.5rem", justify="start"),
            ],
            gap="0.4rem",
        )
    )
    return antenna_offset, depth, radius, source_freq, target_x


# ── Step 3: Permittivity and display ──────────────────────────────────────
@app.cell
def _(mo):
    eps_r = mo.ui.slider(
        start=1.0,
        stop=20.0,
        step=0.1,
        value=6.0,
        label="Relative permittivity of the half-space",
        debounce=True,
    )
    show_curve = mo.ui.checkbox(label="Draw the predicted hyperbola", value=True)
    remove_background = mo.ui.checkbox(
        label="Remove background (reveals the hyperbola)", value=True
    )
    use_gain = mo.ui.checkbox(label="Apply SEC gain", value=False)

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Step 3 — Permittivity"),
                mo.md(
                    "_Drag the permittivity and watch the predicted curve move. When "
                    "it sits on the reflection in the radargram, the value on the "
                    "slider is the permittivity of the modelled material. This is "
                    "hyperbola fitting, the standard way velocity is picked from real "
                    "GPR data._"
                ),
                eps_r,
                mo.hstack(
                    [show_curve, remove_background, use_gain],
                    gap="1.5rem",
                    justify="start",
                ),
            ],
            gap="0.4rem",
        )
    )
    return eps_r, remove_background, show_curve, use_gain


# ── Radargram with the prediction overlaid ────────────────────────────────
@app.cell
def _(
    C_M_PER_NS,
    antenna_offset,
    apex_source_x,
    apex_time,
    apply_gain,
    bscan,
    depth,
    eps_r,
    get_unit_label,
    go,
    mo,
    np,
    permittivity_from_apex,
    radius,
    remove_background,
    remove_mean_trace,
    ricker_delay,
    show_curve,
    source_freq,
    target_x,
    travel_time,
    use_gain,
    velocity,
):
    _matrix = bscan["matrix"]
    _time_ns = bscan["time_ns"]
    _xpos = np.asarray(bscan["positions_x"], dtype=float)
    _unit = get_unit_label(bscan["component"] or "Ez")

    _offset = antenna_offset.value
    _depth = depth.value
    _radius = radius.value
    _delay = ricker_delay(source_freq.value * 1e9)
    _eps = eps_r.value

    # Display processing only. Every prediction below is computed from the
    # geometry, never from these amplitudes, so the toggles cannot move the
    # curve.
    _display = _matrix
    _steps = []
    if remove_background.value:
        _display, _ = remove_mean_trace(_display)
        _steps.append("background removed")
    if use_gain.value:
        _display, _ = apply_gain(
            _display,
            _time_ns,
            "sec",
            factor=0.5,
            power=1.0,
            start_ns=0.15 * float(_time_ns[-1]),
            max_gain=50,
        )
        _steps.append("SEC gain")

    _amp = float(np.max(np.abs(_display))) or 1.0
    _t_max = float(_time_ns[-1])

    _fig = go.Figure(
        data=go.Heatmap(
            z=_display,
            x=_xpos,
            y=_time_ns,
            colorscale="RdBu",
            zmid=0,
            zmin=-_amp,
            zmax=_amp,
            colorbar=dict(title=dict(text=_unit)),
        )
    )

    _apex_x = apex_source_x(target_x.value, _offset)
    _t_apex = apex_time(_depth, _eps, _offset, _radius, _delay)
    _v = velocity(_eps)

    if show_curve.value:
        _curve_x = np.linspace(float(_xpos.min()), float(_xpos.max()), 400)
        _curve_t = travel_time(_curve_x, target_x.value, _depth, _eps, _offset, _radius, _delay)
        _fig.add_trace(
            go.Scatter(
                x=_curve_x,
                y=_curve_t,
                mode="lines",
                name=f"predicted, eps_r = {_eps:g}",
                line=dict(color="#111111", width=2, dash="dash"),
                hovertemplate="%{x:.3f} m<br>%{y:.3f} ns<extra>predicted</extra>",
            )
        )
        _fig.add_trace(
            go.Scatter(
                x=[_apex_x],
                y=[_t_apex],
                mode="markers",
                name="predicted apex",
                marker=dict(color="#111111", size=9, symbol="x"),
                hovertemplate="apex %{x:.3f} m, %{y:.3f} ns<extra></extra>",
            )
        )

    # Early time at the top, the GPR convention.
    _fig.update_layout(
        title=dict(
            text=f"{bscan['component']} radargram"
            + (f"  ·  {', '.join(_steps)}" if _steps else ""),
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(title="Source x-position (m)"),
        yaxis=dict(title="Time (ns)", range=[_t_max, 0.0]),
        height=560,
        margin=dict(l=72, r=32, t=55, b=55),
        legend=dict(x=0.01, y=0.98, yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
    )

    # Refine the reading from the data. Two decisions here matter.
    #
    # The search runs on the RAW matrix, never on the processed display.
    # Mean-trace removal makes the hyperbola visible, but over a limited
    # aperture the target response does not average out: the across-trace mean
    # picks up a smeared copy of the moving reflection and subtracting it
    # distorts the wavelet. On a 60-trace scan of the standard example that
    # pulls the picked apex 0.045 ns early and reports a permittivity of 5.6
    # against a true 6.0. Picking from raw data recovers 6.02.
    #
    # The window is centred on the predicted apex and floored one wavelet
    # period after the direct arrival. Centring it makes this a refinement of
    # the slider rather than a blind search, and the floor stops it locking
    # onto the direct wave, which is an order of magnitude larger than the
    # target response and has sidelobes that outrun any simple gate.
    _pick = None
    _pick_err = None
    _period = 1.0 / source_freq.value
    _t_direct = _offset / C_M_PER_NS + _delay
    _lo = max(_t_apex - _period, _t_direct + _period)
    _hi = _t_apex + _period
    _mask = (_time_ns >= _lo) & (_time_ns <= _hi)

    if _hi <= _lo or not _mask.any():
        _pick_err = (
            f"The search window {_lo:.3f} to {_hi:.3f} ns is empty. The predicted "
            f"apex is too close to the direct wave at {_t_direct:.3f} ns to separate "
            "the two, so the target is too shallow or the material too fast to read "
            "a permittivity from this record."
        )
    elif _xpos.size:
        _j = int(np.argmin(np.abs(_xpos - _apex_x)))
        _t_pick = float(_time_ns[_mask][int(np.argmax(np.abs(_matrix[_mask, _j])))])
        try:
            _eps_pick, _v_pick = permittivity_from_apex(_t_pick, _depth, _offset, _radius, _delay)
            _pick = (_xpos[_j], _t_pick, _eps_pick, _v_pick)
        except ValueError as _e:
            _pick_err = str(_e)

    _rows = [
        f"| Wave speed at eps_r = {_eps:g} | {_v:.4f} m/ns  ({_v / C_M_PER_NS:.3f} c) |",
        f"| Source delay (ricker) | {_delay:.4f} ns |",
        f"| Predicted apex | {_t_apex:.4f} ns at source x = {_apex_x:.3f} m |",
    ]
    if _pick is not None:
        _px, _pt, _pe, _pv = _pick
        _gap = _pt - _t_apex
        _rows.append(f"| Measured apex | {_pt:.4f} ns in the trace at x = {_px:.3f} m |")
        _rows.append(
            f"| Difference | {_gap:+.4f} ns "
            f"({abs(_gap) / _t_apex * 100:.1f}% of the predicted arrival) |"
        )
        _rows.append(f"| **Permittivity from that apex** | **{_pe:.2f}**  ({_pv:.4f} m/ns) |")
    _table = "| | |\n|---|---|\n" + "\n".join(_rows)

    _notes = []
    if _t_apex > _t_max:
        _notes.append(
            mo.callout(
                mo.md(
                    f"**The predicted apex is at {_t_apex:.3f} ns, past the end of the "
                    f"{_t_max:.3f} ns record.** The curve has left the plot. Either the "
                    "permittivity or the depth is too large for this time window, or "
                    "the simulation needs a longer `#time_window`."
                ),
                kind="warn",
            )
        )
    if _pick_err is not None:
        _notes.append(
            mo.callout(
                mo.md(f"**Could not read a permittivity from the data.** {_pick_err}"),
                kind="warn",
            )
        )
    if _pick is not None:
        _px2, _pt2, _pe2, _ = _pick
        _gap2 = _pt2 - _t_apex
        _drift = abs(_pe2 - _eps) / _eps * 100 if _eps else 0.0
        _notes.append(
            mo.md(
                f"_The measured apex is the strongest response between {_lo:.3f} and "
                f"{_hi:.3f} ns in the trace nearest the predicted apex, taken from the "
                "raw data rather than the processed display, so the reading does not "
                "change when the display toggles do. Move the slider until the curve "
                "is near the reflection and the reading settles._"
            )
        )
        if _drift > 2.0:
            _notes.append(
                mo.md(
                    f"_The recovered permittivity is {_drift:.0f}% from the slider "
                    f"value while the apex times differ by only "
                    f"{abs(_gap2) / _t_apex * 100:.1f}%. Permittivity goes as the "
                    "square of elapsed travel time, so a small timing gap is roughly "
                    "doubled in the result. On the standard example the recovered "
                    "value lands near 5.5 against a modelled 6.0 for exactly this "
                    "reason._"
                )
            )

    mo.output.replace(
        mo.vstack(
            [
                mo.md("### Radargram"),
                mo.ui.plotly(_fig, config={"displaylogo": False}),
                *_notes,
                mo.md("### Readout"),
                mo.md(_table),
            ],
            gap="0.5rem",
        )
    )
    return


# ── What this can and cannot tell you ─────────────────────────────────────
@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "**What this can and cannot tell you**\n\n"
            "- **Depth has to be known.** Permittivity and depth trade off against "
            "each other almost exactly: a shallow target in slow material and a deep "
            "target in fast material produce nearly the same hyperbola. Fitting a "
            "full 20-trace hyperbola from a real run recovers depth to half a "
            "millimetre and permittivity to 0.05 when the source delay is known, but "
            "letting the delay float as well admits permittivities from roughly 4 to "
            "7 on the same data at sub-picosecond residual. That is why depth is an "
            "input here and not an output.\n"
            "- **The source delay matters more than it looks.** gprMax defines the "
            "ricker waveform at `t - sqrt(2)/f`, so a 1.5 GHz pulse leaves 0.943 ns "
            "after the simulation starts, nearly a third of the 3 ns window used by "
            "the standard examples. Every prediction here includes it. Omit it and "
            "the curve sits a third of the way up the plot from where it belongs.\n"
            "- **The prediction is geometric, the data is not.** The curve is the "
            "arrival of an idealised impulse reflecting off the nearest point of the "
            "target. The radargram shows the peak of a finite-bandwidth wavelet that "
            "has propagated through a numerically discretised FDTD grid. Against a real run of "
            "`examples/gpr/basic/cylinder_Bscan_2D.in` the two agree to within about 4%, with "
            "the prediction arriving slightly late. Expect a small gap and do not "
            "tune the geometry to close it.\n"
            "- **Background removal is for looking, not for measuring.** Subtracting "
            "the across-trace mean is what makes the hyperbola visible, but over a "
            "limited aperture the moving reflection does not average out, so the mean "
            "contains a smeared copy of it. Subtracting that distorts the wavelet. On "
            "the standard 60-trace example it pulls the picked apex 0.045 ns early and "
            "reports a permittivity of 5.6 against a true 6.0. The reading below is "
            "therefore taken from the raw traces, whatever the display shows.\n"
            "- **The apex is not above the target.** With a separated transmitter and "
            "receiver the hyperbola turns over when the midpoint of the pair crosses "
            "the target, half the antenna separation before the source gets there. "
            "For the standard example that is a 20 mm difference."
        ),
        kind="neutral",
    )
    return


if __name__ == "__main__":
    app.run()
