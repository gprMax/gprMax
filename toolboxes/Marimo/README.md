# marimo notebooks for gprMax

Six interactive notebooks for building gprMax models, watching them run, and reading the output.

gprMax writes HDF5 and ships two matplotlib scripts to plot it. Those work. But changing a parameter means editing a text file, re-running the solver from a terminal, and running a plotting script that decides for you what to show. These notebooks let you move a slider instead.

They are [marimo](https://marimo.io) notebooks rather than Jupyter ones. marimo tracks which cells depend on which, so moving a slider re-runs only what that slider affects, and there is no execution order to keep straight. `marimo run` serves a notebook as a web app with the code hidden, which is what you want if you came here to look at data.

---

## Quick start

```bash
pip install -r toolboxes/Marimo/requirements.txt
python -m gprMax examples/cylinder_Ascan_2D.in
marimo run toolboxes/Marimo/ascan_dashboard.py
```

Load the `.h5` file that just appeared in `examples/`, add the Ez trace, and start moving things.

---

## Setup

You need a working gprMax first. From the repository root:

```bash
pip install -e .
pip install -r toolboxes/Marimo/requirements.txt
```

Then, once, if you want the SVG and PDF download buttons to work:

```bash
plotly_get_chrome
```

Installing kaleido alone is not enough. kaleido v1 drives a headless Chrome, and this command fetches a Chrome for Testing binary into kaleido's own cache without touching the browser you actually use. Skip it and the download buttons fail. The camera icon in every plot toolbar exports SVG regardless and needs no setup.

To generate something to look at:

```bash
python -m gprMax examples/cylinder_Ascan_2D.in
python -m gprMax examples/cylinder_Ascan_2D_freespace.in
python -m gprMax examples/cylinder_Bscan_2D.in -n 60
```

A metal cylinder buried in a dielectric half-space, then the same model with the cylinder taken out, then the antenna stepped across the surface in 2 mm increments. The second is what background subtraction needs. The third writes one file per antenna position.

---

## What is here

Four of these files import nothing but numpy and h5py. You can use them from a script, and the notebooks are built on top of them.

| Module | |
|---|---|
| `h5_reader.py` | Reads gprMax v4 `.h5` output into a dict. Multi-file loading, every receiver and component, three time-axis units. |
| `trace_matrix.py` | Stacks single-trace files into an `(n_samples, n_traces)` radargram. Skips unusable files with a recorded reason rather than failing the assembly. |
| `processing.py` | Gain, background removal, free-space subtraction, FFT. Each function takes one trace or a whole matrix. |
| `hyperbola.py` | Two-way travel time to a buried target, forwards and backwards. |

| Notebook | |
|---|---|
| `parameter_controls.py` | Build a model with sliders and watch the input file and geometry update. |
| `progress_tracker.py` | Run a simulation and watch it progress. |
| `ascan_dashboard.py` | Single traces: overlay, subtract, gain, spectrum, export. |
| `bscan_dashboard.py` | Radargrams, live as they are written or from a finished run. |
| `recipes/ascan_workflow.py` | Parameters to input file to solver run to waveform, in one notebook. |
| `recipes/velocity_permittivity.py` | Recover a soil permittivity from the shape of a reflection hyperbola. |

---

## The notebooks

Use `marimo run` to open one, or `marimo edit` to see and change the code.

### parameter_controls.py

```bash
marimo run toolboxes/Marimo/parameter_controls.py
```

Four sliders: soil permittivity, antenna centre frequency, source position, target depth. The `.in` file underneath updates as you move them, and a 2D preview shows where the half-space, the target and the two antennas actually sit. Target depth is measured downward from the surface at y = 0.170.

This previews the file. It does not write it. Copy the text out, or use the A-scan workflow recipe, which writes and runs it for you.

### progress_tracker.py

```bash
marimo run toolboxes/Marimo/progress_tracker.py
```

Pick a `.in` file, press Run, watch the iteration count climb.

gprMax has no callback API for progress, so the notebook launches it as a subprocess and reads its tqdm output. If the solver exits non-zero, you get the return code and the last few lines of output, which is usually enough to see what happened without going back to a terminal.

### ascan_dashboard.py

```bash
marimo run toolboxes/Marimo/ascan_dashboard.py
```

Load one or more `.h5` files. Components are read from each file rather than assumed, so a model that only excites Ez, Hx and Hy offers you those.

Add traces one at a time from any file, receiver and component. They overlay. E-fields and H-fields get separate axes, since V/m and A/m do not belong on one scale.

**Background subtraction** appears once two files are loaded. Pick a free-space run of the same model and it is subtracted from every plotted trace. Everything the two runs share cancels: the source pulse, the antenna coupling, any flat-interface reflection. What survives is the target. On the standard example the cylinder response is roughly a tenth of the direct wave and hard to pick out; subtract, and it becomes the largest thing in the trace.

**Gain** compensates for the fact that a signal returning from deeper has spread further and lost more energy. Six kinds are available. SEC, `exp(a·t)·t^b`, is the one most published GPR processing uses. Whichever you pick, the curve that was applied is drawn underneath the plot, since a gained trace on its own tells you nothing about what was done to it. Gain can be made to start partway down the record, which leaves the direct wave alone instead of saturating the plot with it.

**The Domain toggle** switches to a power spectrum, computed with gprMax's own `fft_power` so it matches `plot_Ascan.py`. Two things about that function are worth knowing before you read the result. It normalises every trace against its own peak, so a loud trace and a quiet one look identical when overlaid, and the shared dB reference here exists to undo that. And an all-zero trace comes out as a flat 0 dB line that looks entirely plausible, so those traces are named instead of drawn.

CSV export contains the untouched file contents at full length, with the processing recorded on the first line. Nothing you do to the display changes what you export.

### bscan_dashboard.py

```bash
marimo run toolboxes/Marimo/bscan_dashboard.py
```

Two ways in. Part 1 watches a directory while a B-scan runs and adds each trace as its file closes, so you can point it somewhere before starting the simulation. Part 2 takes files you already have, orders them by source position and assembles them in one go. Either way you get a heatmap or a 3D surface, with time running downward as GPR convention expects.

**Background removal** subtracts the average trace from every column. On a survey the direct wave is identical in every trace while a target reflection moves across them, so the average is almost entirely the part you do not want. Removing it is what makes a hyperbola visible. Use the global mean for a gprMax B-scan, where every trace comes from the same model. The moving window is for real survey lines, where the background drifts along the profile.

Gain works as it does in the A-scan dashboard, applied across the whole matrix by the same function.

### recipes/ascan_workflow.py

```bash
marimo run toolboxes/Marimo/recipes/ascan_workflow.py
```

The whole loop in one place: set up a model with sliders, write the input file, run the solver with a live progress bar, read the output back, plot it.

Before the solver starts, the notebook works out from the geometry alone when the direct wave and the target reflection should arrive, and warns you if the reflection would land past the end of your time window. That saves running a model whose answer cannot be recorded. Afterwards, those predicted times are drawn on the measured trace.

### recipes/velocity_permittivity.py

```bash
marimo run toolboxes/Marimo/recipes/velocity_permittivity.py
```

Load a B-scan and the notebook draws the hyperbola a target at a given depth in a material of a given permittivity would produce. Move the permittivity slider until the curve sits on the reflection in your data. The slider is then telling you the permittivity of the material. This is hyperbola fitting, which is how velocity gets picked from real GPR surveys.

Depth is an input here rather than something the notebook works out. Depth and permittivity trade off against each other almost exactly: a shallow target in slow material and a deep one in fast material produce nearly the same hyperbola. Depth has to come from somewhere else.

---

## Using the modules directly

```python
from toolboxes.Marimo.h5_reader import load_file, get_trace, get_time_axis
from toolboxes.Marimo.processing import apply_gain, subtract_traces
from toolboxes.Marimo.trace_matrix import stack_traces
from toolboxes.Marimo.hyperbola import travel_time, ricker_delay

data = load_file("examples/cylinder_Ascan_2D.h5")
ez = get_trace(data, "Ez", "rx1")
t = get_time_axis(data, unit="ns")

gained, curve = apply_gain(ez, t, "sec", factor=0.5, power=1.0, start_ns=0.5)

free = load_file("examples/cylinder_Ascan_2D_freespace.h5")
target_only = subtract_traces(
    ez, get_trace(free, "Ez", "rx1"), data["meta"]["dt"], free["meta"]["dt"]
)
```

`apply_gain` and `subtract_traces` accept a single trace or an `(n_samples, n_traces)` matrix, so the same call covers an A-scan and a radargram.

Pass both time steps to `subtract_traces` whenever you know them. Two runs can produce the same number of samples from different `#time_window` settings, and then the arrays line up index by index while representing different instants. A sample-count check on its own does not catch that.

---

## Tests

```bash
pytest tests/test_h5_reader.py tests/test_trace_matrix.py \
       tests/test_processing.py tests/test_hyperbola.py -v
```

156 tests against synthetic HDF5 fixtures, so none of them need real simulation output. Line coverage across the four modules is 99%. Only the FFT tests need a built gprMax, and they skip rather than fail without one.

---

## If something looks wrong

*SVG or PDF download fails.* `plotly_get_chrome` has not been run. Use the camera icon in the plot toolbar meanwhile.

*A component is drawn as a flat line.* It is genuinely zero. A 2D TMz model only excites Ez, Hx and Hy; the rest are stored as zeros. In the frequency domain those traces are named rather than plotted, because `fft_power` turns them into a convincing flat 0 dB spectrum.

*Two spectra look identical despite very different amplitudes.* The dB reference is set to per-trace. Switch it to shared.

*A predicted hyperbola or arrival marker is close but not exact.* Expect a few percent. The prediction is the geometric arrival of an ideal impulse; your data is the peak of a finite-bandwidth wavelet that travelled through a dispersive grid. Resist adjusting the geometry to close the gap.

*A recovered permittivity is further out than the timing looks.* Permittivity goes as the square of elapsed travel time, so a 3% error in a picked arrival lands near 6% in the result.

*A radargram is a wash of red and blue with no hyperbola in it.* Turn on background removal. The direct wave is usually an order of magnitude larger than anything the target sends back.

*The live B-scan is not picking up new traces.* It reads files matching `<basename><N>.h5` and ignores anything with "merged" in the name. Check the base name in Step 2 against what gprMax is actually writing.

---

## Notes

Readouts and exports come from raw data, never from the processed display. Background removal and gain change what you see. They do not change what a measurement reports. Removing the background biases a picked arrival by a few percent, because over a limited aperture a moving reflection does not average out and the mean carries a smeared copy of it, so the velocity recipe processes for display and measures from the raw traces.

AGC is deliberately absent. It equalises amplitudes over a local window, which destroys relative reflectivity, and relative reflectivity is what an FDTD simulation gets right where a field instrument does not.

Not built yet: bandpass and lowpass filters, noise injection, B-scan minus B-scan subtraction, migration, and a model geometry viewer reading gprMax's VTKHDF output. The first three would fit the existing shape of `processing.py`.
