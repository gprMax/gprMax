# Optimisation toolbox

Start with the [user guide](README.rst) and copy [start_here.py](examples/start_here.py).
The example has four numbered sections: choose parameters, build a gprMax model,
score its output, and launch the search. Supporting helpers appear at the end of
the larger examples.

Documentation lives **beside the toolbox**. The same reStructuredText user guide
is included in the main Sphinx manual through `docs/source/inc_Optimisation.rst`;
there is no separate copy to maintain in `docs/`.

| You want to… | Read |
| --- | --- |
| Install or switch TPE, RF, GA, PSO or DE | [Packages and short examples](OPTIMISERS.rst) |
| Read outputs or write a custom scoring function | [Output readers and their fields](OUTPUT_READERS.rst) |
| Reuse the dipole, waveform or published patch example | [Example guide](EXAMPLES.rst) |
| Check the patch dimensions and literature comparison | [Patch notes](examples/rectangular_patch.md) |
| Use GPUs, MPI, caching or checkpoints | [Execution and processing](execution-and-processing.md) |
| Follow the internal call sequence | [Code walkthrough](CODE_WALKTHROUGH.md) |
| Extend the toolbox | [Developer guide](DEVELOPER_GUIDE.md) and [adapter contract](ADAPTERS.md) |

In an existing gprMax development environment, install the tried optional packages:

```sh
python -m pip install -r gprMax/toolboxes/Optimisation/requirements-optimisers.txt
python -m gprMax.toolboxes.Optimisation.examples.start_here
```

You can install only the package you intend to use; see the package guide. The
starter runs a small CPU example. The patch is substantially more expensive.
Use a new results directory for each run. `Integer(..., step=2)` declares physical
increments for all five adapters; `target_value` optionally stops at an acceptable
objective after a complete batch.
