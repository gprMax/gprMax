# gprMax Marimo Dashboard

Reactive dashboards for gprMax built with [marimo](https://marimo.io). Replaces the static scripts in `toolboxes/Plotting/` with an interactive interface where UI controls and visualizations update automatically when parameters change.

marimo uses a DAG-based execution model: every UI widget is a reactive input, and only the cells that depend on a changed value re-execute. There is no hidden state. For a simulation dashboard this is structurally better than Jupyter.

## Dashboards

### `parameter_controls.py`

Four sliders: soil relative permittivity, antenna centre frequency, source x-position, and target depth, that reactively update two outputs on every tick:

- A live `.in` file preview showing valid gprMax syntax ready to write to disk
- A 2D geometry preview rendered via Plotly `fig.add_shape()`, showing the domain, half-space, buried target, and Tx/Rx positions

The geometry parser supports `#box`, `#cylinder`, and `#sphere` primitives. Additional primitives can be added by mapping them to Plotly shapes without modifying the core rendering logic.

```bash
marimo run toolboxes/Marimo/parameter_controls.py
```

### `ascan_dashboard.py`

Reads a gprMax HDF5 output file and renders the Ez field at a receiver as an interactive Plotly waveform. A range slider lets you zoom into any time-step window with instant updates.

```bash
marimo run toolboxes/Marimo/ascan_dashboard.py
```

### `bscan_dashboard.py`

Polls an output directory for new `.h5` trace files and appends each trace to a live radargram heatmap as it appears, without rebuilding the matrix from scratch on each tick. Uses `mo.state` to hold the accumulated 2D array and `mo.ui.refresh` to drive polling.

```bash
marimo run toolboxes/Marimo/bscan_dashboard.py
```

Point the dashboard at the directory where gprMax is writing trace files. The radargram builds trace-by-trace in real time.

## Setup

Install dashboard dependencies into your existing gprMax environment:

```bash
pip install -r toolboxes/Marimo/requirements.txt
```

Or manually:

```bash
pip install "marimo>=0.9.0" h5py plotly numpy
```

## Generating test data

The `devel` branch `examples/` directory contains ready-to-use input files.

**A-scan:**
```bash
python -m gprMax examples/cylinder_Ascan_2D.in
marimo run toolboxes/Marimo/ascan_dashboard.py
```

**B-scan (60 traces):**
```bash
python -m gprMax examples/cylinder_Bscan_2D.in -n 60
python toolboxes/Utilities/outputfiles_merge.py examples/cylinder_Bscan_2D
marimo run toolboxes/Marimo/bscan_dashboard.py
```

For the B-scan dashboard, enter the full path to `examples/` in the output directory field when the dashboard opens.

## Architecture notes

**Why `fig.add_shape()` over subprocess calling `-geom`:**
Invoking gprMax's geometry flag on every slider tick introduces 200–500 ms latency per update. Plotly native shapes render in under 5 ms. For a reactive interface this is not a marginal difference.

**Why marimo over Jupyter:**
gprMax currently ships `toolboxes/Plotting/plot_Ascan.py` and `toolboxes/Plotting/plot_Bscan.py` as static terminal scripts, and `jupyter-notebooks/` contains static example notebooks. Changing a parameter in any of these requires editing and re-running manually. marimo's DAG model makes parameter exploration correct by construction.

**Why `mo.state` for B-scan accumulation:**
marimo re-executes all cells that depend on a changed reactive input. If the B-scan cell read all available trace files and rebuilt the full NumPy array on every `mo.ui.refresh` tick, it would stutter on large simulations. Holding the accumulated matrix in `mo.state` and appending only new traces keeps memory-per-tick constant at one column regardless of simulation size.

## Known limitations

- `bscan_dashboard.py` reads from a live directory of individual trace files. The static merged-file viewer (`outputfiles_merge.py` output) is available via `toolboxes/Plotting/plot_Bscan.py`.
- Geometry preview supports 2D TMz configurations only. 3D model preview is out of scope for this project.
- Tested on Linux and macOS (Apple Silicon, Python 3.11–3.13). Windows support is a stretch goal pending HDF5/Conda availability.