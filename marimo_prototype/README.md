# gprMax Marimo Prototype

> A reactive, web-based interface for configuring, running, and visualising gprMax simulations — built with [marimo](https://github.com/marimo-team/marimo).

---

## Changelog

## 31 March 2026
- **Live Log Implementation** — `reactbscan.py` can now show logs while processing takes place with the help of ` mo.output.replace()`

## 29 March 2026 
- **Cleaned and Re-structured code** - Cleaned code, removed a bug in `react_bscan_runner.py`

## 27 March 2026 
- **Live Log Implementation** — `react_run_simulation.py` cleaned code along with a better and easier to understand structure. 

## 25 March 2026 
- **Live Log Implementation** — `reactascan.py` can now show logs while processing takes place with the help of ` mo.output.replace()`
- **Tried implementing contolled run** — `reactascan.py`can now do controlled run if user wants to enter multiple parameters. Currently its a bit shabby as user will need to change the default state again and again using `Run Simulation` at the bottom of sidebar. I will further upgrade it to be more user friendly and robust. 

### 23 March 2026
- **B-scan migrated from CLI to Python API** — `reactbscan.py` no longer uses `subprocess` to call gprMax. It now interfaces directly with `gprMax.gprMax.api`, consistent with the A-scan approach.
- Added `react_bscan_model.py` — a dedicated `GPRMaxBscanModel` class that builds B-scan `.in` file content programmatically, with support for scan axis, scan target (source / receiver / both), and per-step position updates.
- Added `react_bscan_runner.py` — handles per-step model generation, simulation execution, output merging via `tools.outputfiles_merge`, and HDF5 data extraction.
- `bscan.in` template file is no longer required for B-scan runs.

---

## Overview

This prototype integrates **marimo** into [gprMax](https://github.com/gprMax/gprMax) to replace static `.in` file editing with a live, reactive GUI. All simulation parameters (domain, material, waveform, source, receiver) are exposed as interactive controls in the browser. Changing any value immediately propagates through the notebook — no manual re-running of cells required.

Two workflows are supported:

| Workflow | File | Simulation method |
|---|---|---|
| **A-scan** | `reactascan.py` | Python API (`gprMax.gprMax.main()`) |
| **B-scan** | `reactbscan.py` | Python API (`gprMax.gprMax.api`) |

---

## Running the Apps

**A-scan** — single trace:
```bash
marimo run reactascan.py
```
Adjust the parameters in the sidebar, then click **Run Simulation** to see the trace.

**B-scan** — radargram:
```bash
marimo run reactbscan.py
```
Set the B-scan start, end, and step positions in the sidebar. The app runs all traces automatically and displays the radargram.

**To edit the code while using the UI:**
```bash
marimo edit reactascan.py
marimo edit reactbscan.py
```

---

## File Structure

```
marimo_prototype/
├── reactascan.py           # Marimo app: single-trace (A-scan) workflow
├── reactbscan.py           # Marimo app: multi-trace (B-scan) workflow
├── react_model_builder.py  # GPRMaxModel class — builds .in file content (A-scan)
├── react_run_simulation.py # Runs gprMax via Python API, returns logs (A-scan)
├── react_bscan_model.py    # GPRMaxBscanModel class — builds .in file content (B-scan)
└── react_bscan_runner.py   # Runs B-scan via Python API, merges outputs, extracts data
```

---

## Module Descriptions

### `react_model_builder.py` — `GPRMaxModel`

A plain Python class that holds all model parameters and serialises them into gprMax `.in` file commands via `to_in_file()`.

**Default configuration:**

| Parameter | Default |
|---|---|
| Spatial resolution (dx/dy/dz) | 0.02 m |
| Domain | 0.4 × 0.4 × 0.2 m |
| Time window | 5 ns |
| PML cells | 0 |
| Material (ε, σ, μr, σm) | 4, 0.01, 1, 0 — `half_space` |
| Waveform | Gaussian, 1 V/m, 100 MHz — `pulse` |
| Source (Hertzian dipole, z-dir) | (0.1, 0.1, 0.05) m |
| Receiver | (0.15, 0.1, 0.05) m |

**Usage:**
```python
from react_model_builder import GPRMaxModel

model = GPRMaxModel()
model.dx = 0.01
model.material["eps"] = 6
print(model.to_in_file())
```

---

### `react_run_simulation.py` — `run_model()`

Writes the model to a temporary file (`temp_model.in`), invokes `gprMax.gprMax.main()` via the Python API, captures all stdout into a string buffer, and returns the output filename and captured logs.

```python
from react_run_simulation import run_model

output_file, logs = run_model(model)
```

> **Note:** gprMax is invoked by patching `sys.argv` and redirecting stdout with `contextlib.redirect_stdout`. This keeps the simulation in-process, avoiding subprocess overhead.

---

### `react_bscan_model.py` — `GPRMaxBscanModel`

A dedicated model class for B-scan simulations. Holds all simulation parameters and generates `.in` file content via `build_input()`. Supports configurable scan axis (`x`, `y`, or `z`) and scan target (`source`, `receiver`, or `both`).

**Default configuration:**

| Parameter | Default |
|---|---|
| Spatial resolution (dx/dy/dz) | 0.01 m |
| Domain | 0.1 × 0.1 × 0.1 m |
| Time window | 5 ns |
| PML cells | 2 |
| Material (ε, σ, μr, σm) | 4, 0.0, 1, 0 — `soil` |
| Waveform | Ricker, 1 V/m, 100 MHz — `pulse` |
| Source (z-dir) | (0.05, 0.05, 0.05) m |
| Receiver | (0.06, 0.05, 0.05) m |
| Scan range | 0.0 → 0.06 m, step 0.02 m |
| Scan axis / target | x / source |
| Field component | Ez |

**Usage:**
```python
from react_bscan_model import GPRMaxBscanModel

model = GPRMaxBscanModel()
model.start = 0.0
model.end = 0.1
model.step = 0.01
print(model.build_input())
```

---

### `react_bscan_runner.py` — `run_bscan()` / `extract_bscan_data()`

Drives the full B-scan pipeline entirely via the Python API — no subprocess calls.

**`run_bscan(model)`**
- Computes scan positions from `model.start`, `model.end`, `model.step`.
- For each position: deep-copies the model, updates the scan axis coordinate, writes a numbered `temp_bscanN.in` file, and calls `gprMax.gprMax.api`.
- Merges all per-trace `.out` files using `tools.outputfiles_merge.merge_files()`.
- Returns the path to the merged HDF5 file (`temp_bscan_merged.out`).

**`extract_bscan_data(merged_file, field)`**
- Opens the merged HDF5 file with `h5py`.
- Reads the chosen field component (e.g. `Ez`) across all receivers.
- Returns `(data, dt, nrx, field)` ready for plotting.

```python
from react_bscan_runner import run_bscan, extract_bscan_data

merged_file = run_bscan(model)
data, dt, nrx, field = extract_bscan_data(merged_file, "Ez")
```

---

### `reactascan.py` — A-scan Marimo App

A single-page marimo app for running one gprMax trace and visualising it.

**Sidebar controls (collapsible accordions):**

- **Domain** — dx, dy, dz, domain_x, domain_y, domain_z
- **Material** — εr, σ, μr, σm, material name (dropdown)
- **Waveform** — amplitude, frequency, waveform name
- **Source** — dipole direction (x/y/z dropdown), x, y, z
- **Receiver** — x, y, z
- **Run Simulation** button

**App flow:**

1. UI values are applied to a `GPRMaxModel` instance.
2. `run_model()` is called — gprMax runs in-process and logs are captured.
3. ANSI escape codes are stripped; logs are displayed in a styled dark terminal panel.
4. `tools.plot_Ascan.mpl_plot` renders the time-domain trace.

**Run:**
```bash
marimo run reactascan.py      # app mode (no code visible)
marimo edit reactascan.py     # notebook / edit mode
```

---

### `reactbscan.py` — B-scan Marimo App

A marimo app for running a multi-trace B-scan survey and rendering the radargram.

**Additional sidebar section — B-scan Parameters:**

- Start position (m)
- End position (m)
- Step size (m)

The number of traces is computed as:
```
n = floor((end - start) / step) + 1
```

**App flow:**

1. UI values are applied to a `GPRMaxBscanModel` instance.
2. `run_bscan()` iterates over scan positions — each step writes a temporary `.in` file and calls `gprMax.gprMax.api` directly.
3. Per-trace `.out` files are merged with `tools.outputfiles_merge.merge_files()`, producing `temp_bscan_merged.out`.
4. `extract_bscan_data()` reads the chosen field component across all receivers from the merged HDF5 file.
5. `tools.plot_Bscan.mpl_plot` renders the radargram.

**Run:**
```bash
marimo run reactbscan.py
marimo edit reactbscan.py
```

---