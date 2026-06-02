import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go

    return go, mo


@app.cell
def _(mo):
    permittivity = mo.ui.slider(
        start=1.0, stop=20.0, step=0.5, value=7.0, label="Soil Relative Permittivity (εr)"
    )
    frequency = mo.ui.slider(
        start=0.5, stop=3.0, step=0.1, value=1.0, label="Antenna Centre Frequency (GHz)"
    )
    src_x = mo.ui.slider(
        start=0.010, stop=0.220, step=0.002, value=0.060, label="Source X-Position (m)"
    )
    target_depth = mo.ui.slider(
        start=0.030, stop=0.150, step=0.005, value=0.080, label="Target Depth (m)"
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md("### gprMax Parameter Controls"),
                permittivity,
                frequency,
                src_x,
                target_depth,
            ]
        )
    )
    return frequency, permittivity, src_x, target_depth


@app.cell
def _(frequency, mo, permittivity, src_x, target_depth):
    rx_x = round(src_x.value + 0.040, 3)
    surface_y = 0.170
    in_text = "\n".join(
        [
            "#domain: 0.240 0.210 0.002",
            "#dx_dy_dz: 0.002 0.002 0.002",
            "#time_window: 3e-9",
            "",
            f"#material: {permittivity.value} 0 1 0 half_space",
            f"#box: 0 0 0 0.240 {surface_y} 0.002 half_space",
            f"#cylinder: 0.120 {target_depth.value:.3f} 0.000 0.120 {target_depth.value:.3f} 0.002 0.010 pec",
            "",
            f"#waveform: ricker 1 {frequency.value:.1f}e9 my_pulse",
            f"#hertzian_dipole: z {src_x.value:.3f} {surface_y} 0 my_pulse",
            f"#rx: {rx_x:.3f} {surface_y} 0",
        ]
    )
    mo.output.replace(
        mo.vstack(
            [
                mo.md(
                    "### Live `.in` File Preview\n\nAdjust sliders above — output is valid gprMax syntax ready to write directly to `model.in`."
                ),
                mo.md(f"```text\n{in_text}\n```"),
            ]
        )
    )
    return in_text, rx_x, surface_y


@app.cell
def _(go, in_text, mo, rx_x, src_x, surface_y):
    shapes = []
    annotations = []
    domain_x, domain_y = 0.240, 0.210

    for raw_line in in_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        try:
            if line.startswith("#domain:"):
                parts = line[8:].split()
                domain_x = float(parts[0])
                domain_y = float(parts[1])
            elif line.startswith("#box:"):
                parts = line[5:].split()
                bx1, by1 = float(parts[0]), float(parts[1])
                bx2, by2 = float(parts[3]), float(parts[4])
                shapes.append(
                    dict(
                        type="rect",
                        x0=bx1,
                        y0=by1,
                        x1=bx2,
                        y1=by2,
                        fillcolor="rgba(139, 115, 85, 0.35)",
                        line=dict(color="rgba(160, 130, 90, 0.7)", width=1),
                        layer="below",
                    )
                )
            elif line.startswith("#cylinder:"):
                parts = line[10:].split()
                cx, cy = float(parts[0]), float(parts[1])
                radius = float(parts[6])
                mat = parts[7] if len(parts) > 7 else ""
                fill = "rgba(210, 60, 60, 0.55)" if mat == "pec" else "rgba(210, 200, 60, 0.45)"
                border = "rgba(230, 80, 80, 0.9)" if mat == "pec" else "rgba(230, 220, 80, 0.9)"
                shapes.append(
                    dict(
                        type="circle",
                        x0=cx - radius,
                        y0=cy - radius,
                        x1=cx + radius,
                        y1=cy + radius,
                        fillcolor=fill,
                        line=dict(color=border, width=1.5),
                    )
                )
                annotations.append(
                    dict(
                        x=cx,
                        y=cy,
                        text=mat if mat else "cyl",
                        showarrow=False,
                        font=dict(size=9, color="white"),
                    )
                )
            elif line.startswith("#sphere:"):
                parts = line[8:].split()
                cx, cy = float(parts[0]), float(parts[1])
                radius = float(parts[3])
                shapes.append(
                    dict(
                        type="circle",
                        x0=cx - radius,
                        y0=cy - radius,
                        x1=cx + radius,
                        y1=cy + radius,
                        fillcolor="rgba(60, 190, 60, 0.45)",
                        line=dict(color="rgba(80, 210, 80, 0.9)", width=1.5),
                    )
                )
        except (IndexError, ValueError):
            pass

    fig = go.Figure(
        data=[
            go.Scatter(
                x=[src_x.value],
                y=[surface_y],
                mode="markers+text",
                marker=dict(size=12, color="yellow", symbol="triangle-down"),
                text=["Tx"],
                textposition="top center",
                textfont=dict(color="yellow", size=10),
                name="Tx (source)",
            ),
            go.Scatter(
                x=[rx_x],
                y=[surface_y],
                mode="markers+text",
                marker=dict(size=12, color="cyan", symbol="triangle-down"),
                text=["Rx"],
                textposition="top center",
                textfont=dict(color="cyan", size=10),
                name="Rx (receiver)",
            ),
        ]
    )
    fig.update_layout(
        title="2D Geometry Preview (x-y plane, z = 0.002 m)",
        xaxis=dict(
            title="x (m)",
            range=[-0.01, domain_x + 0.01],
            constrain="domain",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="y (m)",
            range=[-0.01, domain_y + 0.01],
            scaleanchor="x",
            constrain="domain",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        ),
        template="plotly_dark",
        shapes=shapes,
        annotations=annotations,
        height=500,
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
    )
    return mo.ui.plotly(fig)


if __name__ == "__main__":
    app.run()
