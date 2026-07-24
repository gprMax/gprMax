"""Regression tests for gprMax/contexts.py's geometry_fixed multi-run path
(Context._run_model / MPIContext._run_model) and gprMax/config.py's restart
(-i/i=) model-number indexing.

Bugs found and fixed, all chained/discovered together:

1. `model` was a plain local variable, assigned only inside
   `if not model_config.reuse_geometry():`. Each call to `_run_model()` is a
   separate invocation with its own local scope, so nothing carried `model`
   over from run 1 to run 2. On run 2 (model_num=1, geometry_fixed=True),
   `reuse_geometry()` is True, so that `if` block - the only assignment to
   `model` - never runs, and the very next line (`model.build()`) raised
   `UnboundLocalError`. Fixed by persisting the model on the Context instance
   (`self.model`) instead of a bare local variable.
2. Once (1) was fixed, a second, previously-masked bug surfaced:
   `model_config` is recreated fresh on *every* call (even reuse_geometry
   ones), but `ompthreads` is only ever resolved inside `Model.__init__()`,
   which doesn't run when geometry is reused - so the fresh model_config's
   `ompthreads` stayed `None`, crashing later in `Model.solve()`
   (`TypeError: '>' not supported between instances of 'NoneType' and 'int'`).
   Fixed by re-resolving `ompthreads` via `set_omp_threads()` explicitly on
   the reuse_geometry path, mirroring what Model.__init__() already does.
3. `ModelConfig.reuse_geometry()` compared `model_num` against the literal
   `0` instead of the run's actual starting model number
   (`sim_config.model_start`). With a restart (-i/i=), the first model in
   model_range can itself be non-zero (e.g. i=2 -> model_start=1), so the
   very first _run_model() call already had model_num != 0 and was wrongly
   treated as "reuse a never-built self.model" -> AttributeError. Fixed by
   comparing against sim_config.model_start instead of 0.
4. Even with (3) fixed, restart + n>1 could still crash:
   `SimulationConfig.get_scene()`/`get_model_config()` (and their `set_*`
   counterparts) indexed `self.scenes`/`self.model_configs` directly by the
   *absolute* model number, but both lists are sized to hold exactly `n`
   entries (one per model in *this* run). The moment model_start != 0 (any
   restart), the absolute model numbers used (model_start..model_start+n-1)
   no longer match the lists' own index range (0..n-1) -> IndexError. Fixed
   by indexing both lists relative to model_start
   (SimulationConfig._list_index()) instead of by the absolute model number.
"""
import h5py
import numpy as np

import gprMax


def test_geometry_fixed_multiple_runs_completes_without_exception(tmp_path):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.015, 0.01, 0.01)))

    outputfile = tmp_path / "run"
    # Would previously raise UnboundLocalError on model 2, or (once that was
    # fixed in isolation) TypeError on model 2's ompthreads comparison.
    gprMax.run(scenes=[scene], n=3, geometry_fixed=True, outputfile=outputfile, hide_progress_bars=True)

    for i in (1, 2, 3):
        with h5py.File(f"{outputfile}{i}.h5", "r") as f:
            assert not np.any(np.isnan(f["rxs/rx1/Ez"][:]))


def _make_scene():
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-11))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0.01), waveform_id="w"))
    scene.add(gprMax.Rx(p1=(0.015, 0.01, 0.01)))
    return scene


def test_restart_with_single_scene_and_geometry_fixed(tmp_path):
    """API scenes=[one_scene] (the natural pattern for a reused geometry),
    restarted from a non-first model number (i=2). Before the fix, the
    absolute model_num (1, 0-indexed) indexed directly into a length-1
    scenes list and raised IndexError - regardless of geometry_fixed."""
    outputfile = tmp_path / "run"
    gprMax.run(
        scenes=[_make_scene()],
        n=2,
        i=2,
        geometry_fixed=True,
        outputfile=outputfile,
        hide_progress_bars=True,
    )
    for i in (2, 3):
        with h5py.File(f"{outputfile}{i}.h5", "r") as f:
            assert not np.any(np.isnan(f["rxs/rx1/Ez"][:]))


def test_restart_without_geometry_fixed_single_model(tmp_path):
    """Restart with n=1 (no geometry_fixed at all) - the simplest restart
    case, isolating the config.py indexing fix from the contexts.py
    reuse_geometry() fix (this path never goes anywhere near reuse_geometry
    being True, so it only exercises the get_scene()/get_model_config()
    relative-indexing fix). n=1 means no model-number suffix is appended to
    the output filename (config.py's appendmodelnumber), regardless of i.
    """
    outputfile = tmp_path / "run"
    gprMax.run(scenes=[_make_scene()], n=1, i=2, outputfile=outputfile, hide_progress_bars=True)
    with h5py.File(f"{outputfile}.h5", "r") as f:
        assert not np.any(np.isnan(f["rxs/rx1/Ez"][:]))


def test_restart_multiple_models_via_input_file(tmp_path):
    """Restart (i=2, n=3) via the plain input-file/CLI path (not the API
    scenes= parameter), without geometry_fixed - every model needs its own
    _get_scene()/get_model_config() call, so this is the case most likely
    to walk off the end of the n-sized scenes/model_configs lists."""
    infile = tmp_path / "restart.in"
    infile.write_text(
        "#title: restart multi-model test\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.02 0.02 0.02\n"
        "#pml_cells: 0\n"
        "#time_window: 2e-11\n"
        "#waveform: ricker 1 1.5e10 w\n"
        "#hertzian_dipole: z 0.01 0.01 0.01 w\n"
        "#rx: 0.015 0.01 0.01\n"
    )
    outputfile = tmp_path / "run"
    gprMax.run(inputfile=str(infile), n=3, i=2, outputfile=outputfile, hide_progress_bars=True)

    for i in (2, 3, 4):
        with h5py.File(f"{outputfile}{i}.h5", "r") as f:
            assert not np.any(np.isnan(f["rxs/rx1/Ez"][:]))
