"""
Interactive B-Scan viewer for gprMax output files.

Usage:
    python tools/plot_Bscan_interactive.py user_models/cylinder_Bscan_2D_merged.out
    python tools/plot_Bscan_interactive.py mydata.out --rx Ex
    marimo run tools/plot_Bscan_interactive.py

Requirements:
    pip install marimo plotly h5py numpy
"""

import sys
import argparse
import os
from pathlib import Path
import os
import warnings

# Suppress common Windows/websockets keepalive noise
warnings.filterwarnings("ignore", message="keepalive ping failed")
os.environ.setdefault("WEBSOCKETS_PING_INTERVAL", "30")
os.environ.setdefault("WEBSOCKETS_PING_TIMEOUT", "20")

def parse_args():
    """standard argparse setup for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Interactive B-Scan viewer for gprMax .out files",
    )
    parser.add_argument(
        "outfile", type=str,
        help="path to gprMax .out file",
    )
    parser.add_argument(
        "--rx", type=str, default="Ez",
        choices=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        help="field component to visualize (default: Ez)",
    )
    parser.add_argument(
        "--port", type=int, default=2718,
        help="port for the viewer (default: 2718)",
    )
    return parser.parse_args()


# pass config to marimo cells via environment variables
# this is the only reliable way since cells can't see module-level vars
if __name__ == "__main__":
    try:
        args = parse_args()
        _p = Path(args.outfile)
        if not _p.exists():
            print(f"❌ File not found: {args.outfile}")
            sys.exit(1)
        os.environ["GPRMAX_VIEWER_FILE"] = str(_p)
        os.environ["GPRMAX_VIEWER_RX"] = args.rx
        print(f"\n📡 gprMax B-Scan Viewer")
        print(f"   File : {_p}")
        print(f"   Field: {args.rx}\n")
    except SystemExit:
        pass


# ══════════════════════════════════════════════════════
# marimo app
# ══════════════════════════════════════════════════════
import marimo
app = marimo.App(width="full")


# config cell — reads from env vars (set by CLI) or uses defaults
# every other cell that needs the file path depends on this one
@app.cell
def _():
    import os
    import marimo as mo
    import h5py
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # env vars set by CLI, or sensible defaults for marimo run
    VIEWER_FILE = os.environ.get(
        "GPRMAX_VIEWER_FILE",
        "user_models/cylinder_Bscan_2D_merged.out"
    )
    VIEWER_RX = os.environ.get("GPRMAX_VIEWER_RX", "Ez")

    return go, h5py, mo, np, make_subplots, VIEWER_FILE, VIEWER_RX


# read the gprMax output file
@app.cell
def _(h5py, np, mo, VIEWER_FILE, VIEWER_RX):
    data = None
    dist_m = None
    time_ns = None
    n_traces = 0

    try:
        with h5py.File(VIEWER_FILE, "r") as f:
            # gprMax stores receiver data under /rxs/rx1/<component>
            data = f[f"/rxs/rx1/{VIEWER_RX}"][:]
            dt = float(f.attrs["dt"])

            # srcsteps tells us the spatial step between traces
            # can be a 3-element array [dx, dy, dz] or a scalar
            srcsteps = f.attrs.get("srcsteps", None)
            if srcsteps is not None and hasattr(srcsteps, "__len__"):
                dx = float(srcsteps[0])
            elif srcsteps is not None:
                dx = float(srcsteps)
            else:
                dx = 0.002

        # build the axis vectors
        time_ns = np.arange(data.shape[0]) * dt * 1e9
        dist_m = np.arange(data.shape[1]) * dx
        n_traces = data.shape[1]

        _status = f"""
<div style="background:linear-gradient(135deg,#e8f5e9,#f1f8e9);
            border-left:4px solid #43a047; border-radius:8px;
            padding:12px 20px; font-family:'Segoe UI',sans-serif;
            color:#2e7d32; font-size:14px; margin:8px 0;">
    ✅ &nbsp;<strong>{VIEWER_FILE}</strong> &nbsp;·&nbsp;
    Component: <strong>{VIEWER_RX}</strong> &nbsp;·&nbsp;
    <strong>{data.shape[0]}</strong> samples &nbsp;·&nbsp;
    <strong>{n_traces}</strong> traces &nbsp;·&nbsp;
    dt = <strong>{dt*1e12:.2f} ps</strong> &nbsp;·&nbsp;
    dx = <strong>{dx*1e3:.2f} mm</strong>
</div>"""

    except FileNotFoundError:
        _status = f"""
<div style="background:#fff5f5; border-left:4px solid #e53935;
            border-radius:8px; padding:12px 20px;
            font-family:'Segoe UI',sans-serif; color:#c62828; font-size:14px;">
    ❌ &nbsp;File not found: <code>{VIEWER_FILE}</code>
</div>"""

    except KeyError:
        _status = f"""
<div style="background:#fff5f5; border-left:4px solid #e53935;
            border-radius:8px; padding:12px 20px;
            font-family:'Segoe UI',sans-serif; color:#c62828; font-size:14px;">
    ❌ &nbsp;Field <code>{VIEWER_RX}</code> not found in <code>{VIEWER_FILE}</code>.
    Available: Ex, Ey, Ez, Hx, Hy, Hz
</div>"""

    except Exception as e:
        _status = f"""
<div style="background:#fff5f5; border-left:4px solid #e53935;
            border-radius:8px; padding:12px 20px;
            font-family:'Segoe UI',sans-serif; color:#c62828; font-size:14px;">
    ❌ &nbsp;Error: <code>{e}</code>
</div>"""

    mo.md(_status)
    return data, dist_m, time_ns, n_traces


# header — logo loaded as base64 since marimo can't serve local files
@app.cell
def _(mo, VIEWER_RX):
    import base64
    from pathlib import Path

    # try relative path first (works when run from repo root)
    # fallback to absolute path if needed
    # logo sits next to this script in the tools/ folder
    
    logo_path = Path(__file__).parent / "gprMax_logo_small.png"
    if logo_path.exists():
        _logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        _logo_html = f'<img src="data:image/png;base64,{_logo_b64}" style="height:80px; object-fit:contain; display:block; margin:0 auto;" alt="gprMax logo"/>' 
    else:
        _logo_html = """
        <span style="font-size:48px; font-weight:900; letter-spacing:-2px;
            background:linear-gradient(135deg,#1565c0,#42a5f5,#e53935,#43a047,#fb8c00,#ab47bc);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text; font-family:'Georgia',serif;">gpr</span><span
            style="font-size:48px; font-weight:900; letter-spacing:-2px;
            background:linear-gradient(135deg,#26c6da,#ab47bc,#1565c0,#e53935,#43a047);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text; font-family:'Georgia',serif;">Max</span>"""

    mo.md(f"""
<div style="
    background:#ffffff;
    border-bottom:3px solid #e0e0e0;
    padding:40px 40px 32px 40px;
    font-family:'Segoe UI','Helvetica Neue',sans-serif;
    text-align:center;
">
    <!-- logo -->
    {_logo_html}

    <!-- title -->
    <div style="font-size:24px; font-weight:700; color:#1a1a2e;
                margin-top:16px; letter-spacing:-0.5px;">
        B-Scan Visualization Tool
    </div>

    <!-- subtitle -->
    <div style="font-size:13px; color:#9e9e9e; margin-top:6px;">
        Interactive GPR Signal Analysis &nbsp;·&nbsp; Field: <strong style="color:#424242;">{VIEWER_RX}</strong>
    </div>

    <!-- tag pills -->
    <div style="margin-top:20px; display:flex; gap:10px;
                flex-wrap:wrap; justify-content:center;">
        <span style="background:#e3f2fd;color:#1565c0;font-size:11px;font-weight:600;
                     padding:3px 10px;border-radius:20px;">FDTD SIMULATION</span>
        <span style="background:#fce4ec;color:#c62828;font-size:11px;font-weight:600;
                     padding:3px 10px;border-radius:20px;">B-SCAN</span>
        <span style="background:#e8f5e9;color:#2e7d32;font-size:11px;font-weight:600;
                     padding:3px 10px;border-radius:20px;">A-SCAN</span>
        <span style="background:#f3e5f5;color:#6a1b9a;font-size:11px;font-weight:600;
                     padding:3px 10px;border-radius:20px;">BACKGROUND REMOVAL</span>
        <span style="background:#fff3e0;color:#e65100;font-size:11px;font-weight:600;
                     padding:3px 10px;border-radius:20px;">SEC GAIN</span>
        <span style="background:#e0f7fa;color:#00695c;font-size:11px;font-weight:600;
                     padding:3px 10px;border-radius:20px;">DEWOW</span>
    </div>
</div>
""")
    return
# controls
@app.cell
def _(mo, n_traces):
    mode_toggle = mo.ui.dropdown(
        options=["Processed (Background Removed)", "Raw Signal"],
        value="Processed (Background Removed)",
        label="Processing Mode",
    )
    db_slider = mo.ui.slider(
        start=-80, stop=-5, value=-35, step=1,
        label="Amplitude Floor (dB)", debounce=True,
    )
    gain_slider = mo.ui.slider(
        start=0.0, stop=3.0, value=0.0, step=0.1,
        label="Time Gain (SEC)", debounce=True,
    )
    colormap_picker = mo.ui.dropdown(
        options=["RdBu", "Greys_r", "Viridis", "Plasma", "Seismic", "Hot_r"],
        value="RdBu", label="Colormap",
    )
    dewow_toggle = mo.ui.switch(label="Dewow", value=False)
    peak_toggle = mo.ui.switch(label="Peak Overlay", value=False)
    trace_slider = mo.ui.slider(
        start=0, stop=max(n_traces - 1, 1),
        value=max(n_traces - 1, 1) // 2,
        step=1, label="📍 A-Scan Trace", debounce=True,
    )

    mo.vstack([
        mo.md("""
<div style="font-family:'Segoe UI',sans-serif; font-size:11px; font-weight:700;
            letter-spacing:2px; color:#9e9e9e; text-transform:uppercase;
            margin:24px 0 12px 0; padding-bottom:6px; border-bottom:1px solid #eee;">
    ⚙️ &nbsp; Controls
</div>"""),
        mo.hstack([mode_toggle, colormap_picker, dewow_toggle, peak_toggle],
                  justify="start", gap=2),
        mo.hstack([db_slider, gain_slider], justify="start", gap=2),
        trace_slider,
    ])

    return (
        db_slider, mode_toggle, gain_slider,
        colormap_picker, dewow_toggle, peak_toggle, trace_slider
    )


# signal processing pipeline
@app.cell
def _(data, mode_toggle, gain_slider, dewow_toggle, time_ns, np):
    processed = None
    proc_label = ""

    if data is not None:
        _sig = data.copy().astype(np.float64)

        # dewow — vectorized running mean subtraction
        if dewow_toggle.value:
            _w = max(3, _sig.shape[0] // 20)
            if _w % 2 == 0:
                _w += 1
            _k = np.ones(_w) / _w
            _sig -= np.apply_along_axis(
                lambda trace: np.convolve(trace, _k, mode="same"),
                axis=0, arr=_sig
            )
            proc_label += " → Dewow"

        # background removal — subtract mean trace to kill direct wave
        if mode_toggle.value == "Processed (Background Removed)":
            _sig -= np.mean(_sig, axis=1, keepdims=True)
            _sig -= np.mean(_sig)
            proc_label += " → BG Removed"

        # SEC gain — exponential depth compensation, not normalized
        if gain_slider.value > 0:
            _t = np.linspace(1e-6, 1, len(time_ns))
            _sig *= np.exp(gain_slider.value * _t)[:, np.newaxis]
            proc_label += f" → SEC({gain_slider.value:.1f})"

        processed = _sig
        proc_label = proc_label.lstrip(" → ") or "None"

    return processed, proc_label


# B-scan heatmap with peak overlay
@app.cell
def _(processed, dist_m, time_ns, db_slider, mode_toggle,
      colormap_picker, proc_label, trace_slider, peak_toggle,
      np, go, mo):

    if processed is not None:
        # dB scale, referenced to 99th percentile
        _db = 20.0 * np.log10(np.abs(processed) + 1e-12)
        _db -= np.percentile(_db, 99)
        _db = np.clip(_db, db_slider.value, 0)

        # adaptive downsampling for rendering speed
        _nt, _nx = _db.shape
        _ts = max(1, _nt // 800)
        _xs = max(1, _nx // 400)
        _tx = dist_m[trace_slider.value]

        _fig = go.Figure()

        _fig.add_trace(go.Heatmap(
            z=_db[::_ts, ::_xs],
            x=dist_m[::_xs],
            y=time_ns[::_ts],
            colorscale=colormap_picker.value,
            zmin=db_slider.value, zmax=0,
            colorbar=dict(
                title=dict(text="dB", font=dict(size=13)),
                thickness=16,
            ),
        ))

        # peak overlay — smoothed argmax across traces
        if peak_toggle.value:
            _peaks = np.argmax(np.abs(processed), axis=0)
            _peaks = np.convolve(_peaks, np.ones(5)/5, mode="same").astype(int)
            _fig.add_trace(go.Scatter(
                x=dist_m, y=time_ns[_peaks],
                mode="lines",
                line=dict(color="#FFD600", width=2.5),
                name="Peak reflection",
            ))

        # yellow line for selected A-scan trace
        _fig.add_vline(
            x=_tx,
            line_width=2, line_dash="dash", line_color="#FFD600",
            annotation_text=f"  {_tx:.4f} m",
            annotation_position="top",
            annotation_font=dict(color="#FFD600", size=12),
        )

        _fig.update_layout(
            title=dict(
                text=f"B-Scan  ·  {mode_toggle.value}  ·  [{proc_label}]",
                font=dict(size=16, color="#1a1a2e", family="Segoe UI"),
                x=0, xanchor="left",
            ),
            xaxis=dict(title="Distance (m)", showgrid=True,
                       gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(title="Time (ns)", autorange="reversed",
                       showgrid=True, gridcolor="#f0f0f0", zeroline=False),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Segoe UI", color="#424242"),
            height=600, margin=dict(t=60, b=50, l=60, r=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11),
            ),
        )

        _out = mo.vstack([
            mo.md("""
<div style="font-family:'Segoe UI',sans-serif; font-size:11px; font-weight:700;
            letter-spacing:2px; color:#9e9e9e; text-transform:uppercase;
            margin:28px 0 12px 0; padding-bottom:6px; border-bottom:1px solid #eee;">
    📡 &nbsp; B-Scan
</div>"""),
            _fig,
        ])
    else:
        _out = mo.md("⚠️ No data — check file path.")

    _out


# A-scan — raw vs processed side by side
@app.cell
def _(data, processed, dist_m, time_ns, trace_slider, VIEWER_RX,
      np, go, mo, make_subplots):

    if data is not None and processed is not None:
        _i = int(np.clip(trace_slider.value, 0, data.shape[1] - 1))
        _tx = dist_m[_i]

        _fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Raw  ·  {_tx:.4f} m  (trace {_i})",
                f"Processed  ·  {_tx:.4f} m  (trace {_i})",
            ],
            horizontal_spacing=0.08,
        )

        _fig.add_trace(go.Scatter(
            x=time_ns, y=data[:, _i],
            line=dict(color="#1565c0", width=1.5),
            fill="tozeroy", fillcolor="rgba(21,101,192,0.07)",
        ), row=1, col=1)

        _fig.add_trace(go.Scatter(
            x=time_ns, y=processed[:, _i],
            line=dict(color="#c62828", width=1.5),
            fill="tozeroy", fillcolor="rgba(198,40,40,0.07)",
        ), row=1, col=2)

        for _c in [1, 2]:
            _fig.update_xaxes(
                title_text="Time (ns)", showgrid=True,
                gridcolor="#f0f0f0", zeroline=True,
                zerolinecolor="#e0e0e0", row=1, col=_c,
            )
            _fig.update_yaxes(
                title_text=f"{VIEWER_RX} Amplitude", showgrid=True,
                gridcolor="#f0f0f0", zeroline=True,
                zerolinecolor="#e0e0e0", row=1, col=_c,
            )

        _fig.update_layout(
            height=320, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Segoe UI", color="#424242"),
            showlegend=False, margin=dict(t=50, b=40, l=60, r=20),
        )

        _out = mo.vstack([
            mo.md("""
<div style="font-family:'Segoe UI',sans-serif; font-size:11px; font-weight:700;
            letter-spacing:2px; color:#9e9e9e; text-transform:uppercase;
            margin:28px 0 12px 0; padding-bottom:6px; border-bottom:1px solid #eee;">
    📈 &nbsp; A-Scan
</div>"""),
            _fig,
            mo.md(f"""
<div style="font-family:'Segoe UI',sans-serif; font-size:13px; color:#616161;
            margin-top:8px; padding:8px 16px; background:#f5f7fa;
            border-radius:6px; display:inline-block;">
    📍 &nbsp;<strong>Trace {_i}</strong> &nbsp;·&nbsp;
    Distance <strong>{_tx:.4f} m</strong>
</div>"""),
        ])
    else:
        _out = mo.md("")

    _out


# signal statistics
@app.cell
def _(processed, np, mo):

    if processed is not None:
        _a = np.abs(processed)
        _p = float(_a.max())
        _m = float(_a.mean())
        _r = float(np.sqrt(np.mean(processed**2)))
        _dr = 20 * np.log10(_p / (_m + 1e-30))
        _snr = 20 * np.log10(_p / (_r + 1e-30))
        _db = 20 * np.log10(_a + 1e-12)
        _ref = float(np.percentile(_db, 99))

        def _card(label, value, color):
            return f"""
<div style="background:#fff; border:1px solid #e0e0e0;
            border-top:3px solid {color}; border-radius:8px;
            padding:12px 16px; flex:1 1 0; min-width:0;
            font-family:'Segoe UI',sans-serif;">
    <div style="font-size:10px; color:#9e9e9e; font-weight:600;
                text-transform:uppercase; letter-spacing:1px;
                white-space:nowrap;">{label}</div>
    <div style="font-size:18px; font-weight:700; color:#1a1a2e;
                margin-top:6px; white-space:nowrap;">{value}</div>
</div>"""

        _out = mo.vstack([
            mo.md("""
<div style="font-family:'Segoe UI',sans-serif; font-size:11px; font-weight:700;
            letter-spacing:2px; color:#9e9e9e; text-transform:uppercase;
            margin:28px 0 12px 0; padding-bottom:6px; border-bottom:1px solid #eee;">
    📊 &nbsp; Signal Statistics
</div>"""),
            mo.md(f"""
<div style="display:flex; gap:12px; flex-wrap:nowrap; margin-top:4px; width:100%;">
    {_card("Shape", f"{processed.shape[0]}×{processed.shape[1]}", "#42a5f5")}
    {_card("Peak", f"{_p:.1f}", "#1565c0")}
    {_card("RMS", f"{_r:.1f}", "#7b1fa2")}
    {_card("Dyn. Range", f"{_dr:.1f} dB", "#e53935")}
    {_card("SNR", f"{_snr:.1f} dB", "#43a047")}
    {_card("dB Ref", f"{_ref:.1f} dB", "#fb8c00")}
</div>"""),
        ])
    else:
        _out = mo.md("")

    _out


# help section
@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
<div style="font-family:'Segoe UI',sans-serif; font-size:11px; font-weight:700;
            letter-spacing:2px; color:#9e9e9e; text-transform:uppercase;
            margin:28px 0 12px 0; padding-bottom:6px; border-bottom:1px solid #eee;">
    ❓ &nbsp; Help
</div>"""),
        mo.accordion({
            "⚙️  Processing Mode": mo.md("""
**Processed** removes the direct wave by subtracting the mean trace.
Reveals scattered reflections from buried objects.

**Raw** shows the unprocessed receiver output.
"""),
            "📉  Amplitude Floor (dB)": mo.md("""
Controls visible dynamic range. Referenced to 99th percentile.
- **-10 dB** → only strongest features
- **-35 dB** → balanced view
- **-70 dB** → maximum detail + noise
"""),
            "📈  Time Gain (SEC)": mo.md("""
Compensates for signal decay with depth: `gain(t) = exp(k × t/t_max)`.
Not normalized — preserves relative amplitudes.
"""),
            "🔘  Dewow": mo.md("""
Removes low-frequency baseline drift via running mean subtraction.
Useful for field data. Usually skip for gprMax simulations.
"""),
            "🟡  Peak Overlay": mo.md("""
Yellow line tracing strongest reflection per trace.
Smoothed with 5-point moving average.
Follows hyperbola shapes from buried scatterers.
"""),
            "📍  A-Scan Trace": mo.md("""
Selects which trace to show as a waveform.
Yellow dashed line on B-scan shows the position.
"""),
            "📊  Signal Statistics": mo.md("""
| Metric | Meaning |
|---|---|
| **Shape** | Time samples × traces |
| **Peak** | Max absolute amplitude |
| **RMS** | Average signal energy |
| **Dyn. Range** | 20·log₁₀(peak/mean) |
| **SNR** | 20·log₁₀(peak/RMS) |
| **dB Ref** | 99th percentile — matches B-scan 0 dB |
"""),
        })
    ])


# mean trace
@app.cell
def _(data, time_ns, VIEWER_RX, np, go, mo):
    if data is not None:
        _fig = go.Figure(go.Scatter(
            x=time_ns, y=np.mean(data, axis=1),
            line=dict(color="#7b1fa2", width=1.5),
            fill="tozeroy", fillcolor="rgba(123,31,162,0.07)",
        ))
        _fig.update_layout(
            title=dict(text="Mean Trace · Background Signal",
                       font=dict(size=14, color="#1a1a2e"),
                       x=0, xanchor="left"),
            xaxis=dict(title="Time (ns)", showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(title=f"Mean {VIEWER_RX}", showgrid=True, gridcolor="#f0f0f0"),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Segoe UI"), height=240,
            margin=dict(t=40, b=40, l=60, r=20),
        )
        _out = mo.accordion({
            "📊  Mean Trace — what background removal subtracts": _fig
        })
    else:
        _out = mo.md("")
    _out


# footer
@app.cell
def _(mo, VIEWER_RX):
    mo.md(f"""
<div style="margin-top:48px; padding:20px 40px; border-top:1px solid #eee;
            font-family:'Segoe UI',sans-serif; font-size:12px; color:#bdbdbd;
            display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
    <span>
        <strong style="background:linear-gradient(90deg,#1565c0,#e53935,#43a047);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       background-clip:text;">gprMax</strong>
        &nbsp;·&nbsp; B-Scan Viewer &nbsp;·&nbsp; Field: {VIEWER_RX}
    </span>
    <span>Ground Penetrating Radar · FDTD Simulation Output</span>
</div>
""")
    return


if __name__ == "__main__":
    app.run()