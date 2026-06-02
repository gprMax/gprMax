# gprMax Marimo Dashboard — Developer Notes

Running log of architecture decisions, framework gotchas, and things
to surface in the official docs. Updated as each component is built.

---

## Environment

### Apple Silicon build (macOS 15, M1/M2)
`setup.py` hardcodes `/opt/homebrew/bin/gcc-15` in the darwin block,
ignoring `CC` overrides. On macOS 15 this triggers a fatal SDK header
conflict (`_bounds.h` not found).

**Fix:** patch darwin block to use Apple Clang + LLVM libomp.

```python
os.environ["CC"] = "clang"
compile_args = ["-O3", "-w", "-Xpreprocessor", "-fopenmp", "-march=native", ...]
libraries = ["omp"]   # LLVM omp, not GNU gomp
```

Build command:
```bash
CFLAGS="-I/opt/homebrew/opt/libomp/include" \
LDFLAGS="-L/opt/homebrew/opt/libomp/lib" \
pip install -e . --no-build-isolation
```

**Doc flag:** needs a macOS install note in the official setup guide.

---

## marimo 0.23.x Behaviours

### Orphan cell problem
Cells that return a display element (e.g. `mo.vstack`, `mo.md`) but no
named variables are not executed by the reactive DAG. They have no
downstream dependents so marimo skips them silently.

**Fix:** use `mo.output.replace()` inside the cell that owns the data.
This is a side-effect call — sets display output independently of
what the cell returns.

```python
@app.cell
def _(mo):
    slider = mo.ui.slider(...)
    mo.output.replace(mo.vstack([mo.md("### Header"), slider]))
    return slider   # export for downstream cells
```

**Doc flag:** document this pattern in the contributor guide as the
standard way to display + export from the same cell.

### UI element embedding in mo.md f-strings
Embedding slider objects via f-string interpolation in `mo.md()` does
not render the widget in 0.23.x. Use `mo.vstack()` with the element
as a direct child instead.

```python
# Broken in 0.23.x
mo.md(f"### Header\n{slider}")

# Works
mo.vstack([mo.md("### Header"), slider])
```

### Confirmed working
- `mo.ui.plotly(fig)` returned from a cell renders correctly
- `mo.vstack([...])` called via `mo.output.replace()` renders correctly
- `mo.ui.slider`, `mo.ui.refresh`, `mo.ui.range_slider` — not yet tested

---

## Plotly

### scaleanchor and axis padding
Applying `scaleanchor="y"` on the x-axis causes Plotly to pad the
x-range symmetrically to match the y scale, pushing the domain off
to the right (x goes negative).

**Fix:** apply `scaleanchor="x"` on the y-axis only, and add
`constrain="domain"` to both axes.

```python
xaxis=dict(range=[-0.01, domain_x + 0.01], constrain="domain", ...),
yaxis=dict(range=[-0.01, domain_y + 0.01], scaleanchor="x",
           constrain="domain", ...),
```

---

## Component 1 — Parameter Controls

### Architecture decisions
- 4 cells: imports / sliders + display / in_text + preview / geometry
- Receiver position hardcoded as `src_x + 0.040 m` (40 mm offset).
  Will need a slider in a later iteration.
- `surface_y = 0.170` hardcoded. Acceptable for 2D TMz scope.
  Should become a slider when domain controls are added.
- Geometry parser is fault-tolerant by design — `try/except
  (IndexError, ValueError): pass` on every line. Parser reads the
  same `in_text` string shown in the preview, so geometry is always
  in sync.

### Parser coverage
| Primitive   | Plotly shape | Notes                        |
|-------------|--------------|------------------------------|
| `#box`      | rect         | Uses y-coords from .in file  |
| `#cylinder` | circle       | Axis-along-z assumed (2D)    |
| `#sphere`   | circle       | Projected to x-y plane       |

### Known gaps (for later)
- No `#rx_array` support
- No multi-material colour mapping
- No `#fractal_box` or `#add_surface_roughness`
- Cylinder rendered as ellipse if `constrain="domain"` is missing

---

## Component 2 — A-scan Viewer (upcoming)

### HDF5 schema (confirmed from `cylinder_Ascan_2D.h5`)

/rxs/rx1/Ez      — electric field time series at receiver

/rxs/rx1/Hx      — (present in some configs)

/rxs/rx1/Hy      — (present in some configs)

Field component availability varies by model configuration.
Need to confirm schema holds across all sim types before hardcoding
`Ez` as default.

### Open questions
- Does `mo.ui.range_slider` update correctly when the HDF5 file
  is reloaded? Need to test state reset behaviour.
- What is the correct time axis unit label — ns or time steps?
  Need to check gprMax output conventions.