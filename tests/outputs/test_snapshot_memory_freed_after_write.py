"""Regression test for a second real leak contributing to GitHub
gprMax/gprMax#389 ("Running out of memory" - multi-scene sweeps
accumulating memory until the process OOMs), on top of the FractalVolume
leak already fixed in
tests/cmds_geometry/test_fractal_box_memory_freed_after_build.py.

`Snapshot.snapfields` (gprMax/snapshots.py) holds six full-size numpy
arrays (one per field component) sized to the snapshot's own grid_view -
allocated in `initialise_snapfields()` (called once, from
`FDTDGrid.build()`), populated every time `store()` fires during
time-stepping, and read only by `write_file()`/`write_vtk()`/
`write_hdf5()` at the end of the run. Nothing reads `snapfields` again
after that write - confirmed via a full-codebase grep - so in a
multi-scene sweep (`gprMax.run(scenes=scenes, n=len(scenes))`), the
caller's own `scenes` list keeps every Snapshot object (and its
full-size arrays) alive for the whole run, exactly like the
FractalVolume case.

Fixed in `save_snapshots()` (gprMax/snapshots.py) by clearing
`snap.snapfields` right after `write_file()` returns.

One real complication (not present for FractalVolume): under
`geometry_fixed=True` with more than one model requested,
`FDTDGrid.build()` - and so `initialise_snapfields()` - only runs on the
first model; the *same* Snapshot object is written again on every
subsequent reused-geometry run, and its `store()` call needs
`snapfields` to still hold real arrays to write into. Freeing
unconditionally would silently break that case (`store()` would try to
write into `None`). So the free is gated on
`not (geometry_fixed and number_of_models > 1)`.

While building the geometry_fixed test, found a SECOND, independent,
pre-existing bug (unrelated to the memory fix - reproduces identically
with the memory fix reverted): `save_snapshots()` did
`snap.filename = snapshotdir / snap.filename`, permanently mutating
`snap.filename` to an absolute path. Under geometry_fixed reuse, the
same Snapshot object is written again on run 2+, by which point
`snap.filename` is already absolute from run 1 - and `Path.__truediv__`
discards the left operand entirely whenever the right side is already
absolute, so the join silently collapses back to run 1's own snapshot
directory instead of the new run's. Every run after the first ends up
silently overwriting run 1's file, even though its own `runN_snaps/`
directory (named via the pre-existing, unconditional
`ModelConfig.appendmodelnumber` mechanism - already applied automatically
to every multi-model run, geometry_fixed or not, with no user action
required) sits there created but empty. Fixed by re-deriving from just
`Path(snap.filename).name` (the basename) each time, which is idempotent
regardless of how many times this function has already run on the same
Snapshot object.

This test suite covers all three properties: the free actually happening
in the ordinary (non-geometry_fixed) case, no memory accumulation across
independent scenes, and - under geometry_fixed reuse - correct,
non-crashing, per-run-distinct snapshot files (neither silently
overwritten nor broken by the freed-buffer guard).
"""
import glob

import h5py

import gprMax
import gprMax.model as model_mod


def _capture_snapshots(monkeypatch):
    captured = []
    orig_build = model_mod.Model.build

    def patched(self):
        orig_build(self)
        captured.append(list(self.G.snapshots))

    monkeypatch.setattr(model_mod.Model, "build", patched)
    return captured


def _scene_with_snapshot(filename, snap_time=5e-13, fileext=None):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1.5e10, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0.01), waveform_id="w"))
    scene.add(
        gprMax.Snapshot(
            p1=(0, 0, 0), p2=(0.02, 0.02, 0.02), dl=(dl, dl, dl),
            filename=filename, time=snap_time, fileext=fileext,
        )
    )
    return scene


def test_snapfields_freed_after_normal_run(monkeypatch, tmp_path):
    captured = _capture_snapshots(monkeypatch)
    scene = _scene_with_snapshot("snap")
    gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "run", hide_progress_bars=True)

    assert len(captured) == 1
    snap = captured[0][0]
    assert snap.snapfields == {}


def test_repeated_scenes_do_not_accumulate_snapshot_memory(monkeypatch, tmp_path):
    captured = _capture_snapshots(monkeypatch)
    scenes = [_scene_with_snapshot(f"snap{i}") for i in range(3)]
    gprMax.run(scenes=scenes, n=3, outputfile=tmp_path / "run", hide_progress_bars=True)

    assert len(captured) == 3
    for models in captured:
        assert models[0].snapfields == {}


def test_geometry_fixed_reuse_still_writes_correct_distinct_snapshots(tmp_path):
    """The risky edge case: with geometry_fixed=True and n>1, the SAME
    Snapshot object is written on every reused-geometry run - freeing its
    snapfields after the first write would crash (or silently corrupt)
    every subsequent run's store()/write. Confirm each run still produces
    a valid, non-empty snapshot file."""
    scene = _scene_with_snapshot("snap_fixed", fileext=".h5")
    gprMax.run(
        scenes=[scene], n=3, geometry_fixed=True,
        outputfile=tmp_path / "run", hide_progress_bars=True,
    )

    files = sorted(glob.glob(str(tmp_path / "*_snaps" / "*")))
    assert len(files) == 3
    for f in files:
        with h5py.File(f, "r") as h5:
            assert h5["Ex"].shape[0] > 0
