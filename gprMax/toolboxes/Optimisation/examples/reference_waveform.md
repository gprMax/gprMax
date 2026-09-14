# Reference waveform for the editable example

`reference_waveform.npz` is synthetic gprMax data generated with the
`build_model` function in [waveform_matching.py](waveform_matching.py), using:

| Setting | Value |
|---|---|
| Block width | 0.012 m |
| Block relative permittivity | 4.0 |
| Cubic domain | 0.064 m on each side |
| Cubic cell size | 0.002 m |
| Requested time window | 3 ns |
| Source | 1 GHz Ricker pulse; z-directed Hertzian dipole |
| Output | Receiver `probe`, component `Ez` |
| Solver | CPU, single precision, one thread |
| gprMax version | 4.0.0, devel commit `f18ce497ebd41d960979594ddcc4721e9a728f46` |

The file has 780 samples at a time step of approximately 3.851666403 ps. Its keys
are `time_s` (seconds), `Ez` (V/m), and `unit` (the string `V/m`). These are this
example's reference-data conventions; the framework imposes none of these keys
on a user's data. Other sources and quantities can be loaded in `evaluate`.

The short simulation is for demonstrating the optimisation workflow. It is not
a measured GPR dataset or an antenna validation. A different mesh, model or
solver version can change the waveform.

To generate a new reference, save this as a separate script next to your copied
`waveform_matching.py` and run it in the same gprMax environment. Use a fresh
simulation directory and reference filename; then set `REFERENCE` in your
example to the new file.

```python
from pathlib import Path
import numpy as np
from gprMax.toolboxes.Optimisation import simulate
from waveform_matching import build_model

if __name__ == "__main__":
    folder = Path(__file__).parent
    output = simulate(
        model=build_model,
        parameters={"width": 0.012, "permittivity": 4.0},
        directory=folder / "reference_simulation",
    )
    trace = output.receiver("probe", "Ez")
    # Exclusive creation avoids overwriting an existing reference.
    with (folder / "my_reference.npz").open("xb") as destination:
        np.savez_compressed(destination, time_s=trace.time,
                            Ez=trace.values, unit=trace.unit)
```
