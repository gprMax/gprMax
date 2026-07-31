"""Regression test for GitHub gprMax/gprMax#389 ("Running out of memory" -
multi-scene sweeps accumulating memory, one scene at a time, until the
process OOMs).

Root cause traced: FractalVolume.fractalvolume (and, when surface
roughness/grass/water are used, FractalVolume.mask) is a real numpy array
sized to the fractal box's full cell count - not an estimate, an actual
large allocation. It persists as a plain instance attribute on the
FractalVolume object (referenced by the FractalBox command object, which
lives inside whatever Scene the user added it to) for as long as that
Scene object is reachable. In a multi-scene sweep
(`gprMax.run(scenes=scenes, n=len(scenes))`), the CALLER's own `scenes`
list keeps every Scene - and therefore every FractalBox's fully-realised
array - alive for the entire run, even though gprMax itself has long
since finished with that scene's model and moved on. Memory grows by one
fractal box's worth per scene.

This is not fixable by gprMax clearing its own grid references (which it
already does, see contexts.py's `del self.model.G` + `gc.collect()`
between runs) - that data lives on the caller's own Scene object, not the
grid. Confirmed (via full-codebase grep) that `.fractalvolume`/`.mask`
are never read again anywhere after `FractalBox.build()` finishes
consuming them into `grid.solid`/`rigidE`/`rigidH`/`ID` - not by
`#geometry_objects_write` (which reads the grid's own already-built
arrays, not the FractalVolume's), not by `#add_grass`/
`#add_surface_roughness`/`#add_surface_water` (which only reference the
volume object for attaching FractalSurface objects, never the raw
array), and not by a subsequent `geometry_fixed` reuse run (which skips
calling `build()` again entirely). Fixed by explicitly setting both to
`None` right after they're consumed, at the end of `FractalBox.build()`.
"""
import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod


def _capture(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched(self):
        orig_build(self)
        captured["volume"] = self.G.fractalvolumes[0]
        captured["solid"] = np.array(self.G.solid)

    monkeypatch.setattr(model_mod.Model, "build", patched)
    return captured


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-2, 1e-2, 1e-2)))
    scene.add(gprMax.Domain(p1=(0.5, 0.5, 0.5)))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.SoilPeplinski(
            sand_fraction=0.5, clay_fraction=0.5, bulk_density=2.0, sand_density=2.66,
            water_fraction_lower=0.001, water_fraction_upper=0.25, id="soil1",
        )
    )
    return scene


def test_fractalvolume_freed_after_build_no_surfaces(monkeypatch, tmp_path):
    """The common case: a plain fractal box, no surface roughness/grass/water."""
    captured = _capture(monkeypatch)
    scene = _base_scene()
    scene.add(
        gprMax.FractalBox(
            p1=(0.1, 0.1, 0.1), p2=(0.4, 0.4, 0.4), frac_dim=1.5, weighting=(1, 1, 1),
            n_materials=5, mixing_model_id="soil1", id="fb1", seed=1, averaging="y",
        )
    )
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / "run", hide_progress_bars=True,
    )

    assert captured["volume"].fractalvolume is None
    # Geometry itself must still be correctly built despite freeing the
    # intermediate array - the grid's real material data is unaffected.
    assert len(np.unique(captured["solid"])) > 1


def test_fractalvolume_and_mask_freed_after_build_with_surface_roughness(monkeypatch, tmp_path):
    """The surface-roughness/grass/water path additionally allocates .mask -
    must also be freed."""
    captured = _capture(monkeypatch)
    scene = _base_scene()
    scene.add(
        gprMax.FractalBox(
            p1=(0.1, 0.1, 0.1), p2=(0.4, 0.4, 0.4), frac_dim=1.5, weighting=(1, 1, 1),
            n_materials=5, mixing_model_id="soil1", id="fb1", seed=1, averaging="y",
        )
    )
    scene.add(
        gprMax.AddSurfaceRoughness(
            p1=(0.1, 0.1, 0.4), p2=(0.4, 0.4, 0.4), frac_dim=1.5, weighting=(1, 1),
            limits=(0.39, 0.41), fractal_box_id="fb1", seed=1,
        )
    )
    gprMax.run(
        scenes=[scene], n=1, geometry_only=True,
        outputfile=tmp_path / "run", hide_progress_bars=True,
    )

    assert captured["volume"].fractalvolume is None
    assert captured["volume"].mask is None


def test_repeated_scenes_do_not_accumulate_fractalvolume_memory(monkeypatch, tmp_path):
    """The actual reported scenario: multiple independent scenes (as in a
    multi-scene sweep), each with its own fractal box - after each model's
    build() completes, that scene's array must already be released, not
    accumulating across the sweep."""
    captured_all = []
    orig_build = model_mod.Model.build

    def patched(self):
        orig_build(self)
        captured_all.append(self.G.fractalvolumes[0])

    monkeypatch.setattr(model_mod.Model, "build", patched)

    scenes = []
    for i in range(3):
        scene = _base_scene()
        scene.add(
            gprMax.FractalBox(
                p1=(0.1, 0.1, 0.1), p2=(0.4, 0.4, 0.4), frac_dim=1.5, weighting=(1, 1, 1),
                n_materials=5, mixing_model_id="soil1", id=f"fb{i}", seed=i + 1, averaging="y",
            )
        )
        scenes.append(scene)

    gprMax.run(
        scenes=scenes, n=3, geometry_only=True,
        outputfile=tmp_path / "run", hide_progress_bars=True,
    )

    assert len(captured_all) == 3
    for volume in captured_all:
        assert volume.fractalvolume is None
