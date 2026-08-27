# Marimo toolbox for gprMax

Interactive dashboards for building simple gprMax models, monitoring runs, and
exploring A-scans and B-scans. The notebooks were contributed by Gaurav Sharma
(`alphaleporus`) through Google Summer of Code 2026 and subsequently integrated
and maintained in the gprMax repository.

The dashboards use [marimo](https://marimo.io), whose reactive cells update only
the calculations affected by a changed control. Reusable numerical operations
are kept in ordinary Python modules and do not depend on marimo.

## Installation

Install gprMax with the optional dashboard dependencies:

```bash
pip install "gprMax[marimo]"
```

From a source checkout, use:

```bash
pip install -e ".[marimo]"
```

Kaleido v1 uses Chrome to create SVG and PDF downloads. Run
`plotly_get_chrome` once if those download buttons are required. Interactive
plots and the Plotly toolbar do not require this step.

## Quick start

From a source checkout:

```bash
python -m gprMax examples/gpr/basic/cylinder_Ascan_2D.in
python -m gprMax toolboxes/Marimo/examples/cylinder_Ascan_2D_background.in
marimo run toolboxes/Marimo/ascan_dashboard.py
```

The second model is a **target-free background reference**. It retains the
same dielectric half-space, source, receiver, grid, and time window as the
target model, but omits the PEC cylinder. It is not a free-space model.
Subtracting it cancels responses common to both simulations, including the
direct coupling and planar-interface response.

Installed dashboards can also be launched as modules, for example:

```bash
python -m toolboxes.Marimo.ascan_dashboard
python -m toolboxes.Marimo.bscan_dashboard
```

Use `python -m gprMax.examples copy DESTINATION` to obtain writable,
version-matched gprMax examples after installing a wheel.

## Contents

| Module | Purpose |
|---|---|
| `h5_reader.py` | Reads root-grid receivers, component timing metadata, source metadata, and exact stored source excitation histories from current gprMax HDF5 files. Legacy root-grid files fall back to the root time step. |
| `trace_matrix.py` | Stacks single-trace files into an `(n_samples, n_traces)` radargram and rejects inconsistent sample axes. |
| `processing.py` | Gain, mean-trace background removal, target-minus-background subtraction, and FFT helpers. |
| `hyperbola.py` | Homogeneous-medium two-way travel-time and permittivity helpers. |

| Dashboard | Purpose |
|---|---|
| `parameter_controls.py` | Creates a simple 2-D model from interactive controls and previews its geometry. |
| `progress_tracker.py` | Runs gprMax as a subprocess and displays parsed progress and failures. |
| `ascan_dashboard.py` | Overlays receiver traces, performs background-reference subtraction and gain, and plots spectra. |
| `bscan_dashboard.py` | Assembles live or completed B-scans and provides heat-map and surface views. |
| `recipes/ascan_workflow.py` | Demonstrates the parameter-to-run-to-result workflow. |
| `recipes/velocity_permittivity.py` | Demonstrates manual hyperbola fitting in a homogeneous, nondispersive half-space. |

The dashboards currently read receivers on the main/root grid. Subgrid output
is stored under `/subgrids/<name>` in current gprMax files and is not yet
exposed by this introductory interface.

## Processing conventions

### Background-reference subtraction

Run geometrically identical target and target-free models, then subtract:

```python
from toolboxes.Marimo.h5_reader import get_trace, load_file
from toolboxes.Marimo.processing import subtract_traces

target = load_file("examples/gpr/basic/cylinder_Ascan_2D.h5")
background = load_file(
    "toolboxes/Marimo/examples/cylinder_Ascan_2D_background.h5"
)

target_only = subtract_traces(
    get_trace(target, "Ez"),
    get_trace(background, "Ez"),
    target["meta"]["dt"],
    background["meta"]["dt"],
)
```

Both runs must use the same grid, source, receiver, and time sampling. The
helper checks shape and time-step compatibility, but it cannot prove that the
geometries differ only by the target.

Mean-trace removal is a different operation. It is appropriate when the
direct wave and layered background response are stationary across a B-scan.
It can distort a target response over a limited aperture and should not be
described as a physical free-space subtraction.

### Time axes

Current gprMax HDF5 receiver datasets carry `SampleInterval` and
`TimeSampleOffset`. Electric fields are stored on the whole time step, while
magnetic fields and derived currents are offset by half a step. Request a
component-aware axis when processing a trace:

```python
from toolboxes.Marimo.h5_reader import get_time_axis

t_ns = get_time_axis(target, "ns", receiver="rx1", component="Ez")
```

### Gain and spectra

Gain changes the displayed data, not the raw HDF5 arrays. SEC gain combines a
power term and an exponential term. AGC is deliberately omitted because it
destroys relative amplitudes.

An all-zero trace has no meaningful normalised spectrum and is reported rather
than drawn. The FFT helper can use a shared absolute dB reference when traces
with different amplitudes are compared.

### Hyperbola prediction

The travel-time helper assumes a homogeneous, nondispersive material and a
known source delay. The finite-radius correction is exact for a monostatic
circular target and is a small-radius approximation for bistatic acquisition.
Depth and permittivity are strongly coupled; the recipe therefore requires a
known depth and is educational rather than an automatic inversion.

## Tests

Only the reusable numerical and HDF5 helpers have focused tests; the generated
interactive layouts are not duplicated with brittle UI tests:

```bash
pytest tests/toolboxes/test_marimo_h5_reader.py \
       tests/toolboxes/test_marimo_trace_matrix.py \
       tests/toolboxes/test_marimo_processing.py \
       tests/toolboxes/test_marimo_hyperbola.py
```

The HDF5 fixtures are synthetic and small. The measured hyperbola check is a
regression check against the bundled cylinder model, not an independent
analytical validation of FDTD.
