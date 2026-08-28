# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Regression tests: the CPU solver's reduced 2D field-update Cython
kernels (gprMax/cython/fields_updates_normal.pyx) for TE mode, mirroring
the pre-existing TM reduction.

Design: `update_electric()`/`update_magnetic()` now take an explicit
`mode2d` int (-1 = 3D, 0/1/2 = TM invariant axis x/y/z, 3/4/5 = TE
invariant axis x/y/z) computed once in `CPUUpdates.__init__()` from
`config.get_model_config().mode`, rather than inferring the reduction from
nx/ny/nz==1 (TM's old approach) or nx/ny/nz==2 (which would have been
ambiguous with a genuine small 3D domain).

For TE specifically, before this change every TE-mode simulation fell
through to the full 3D branch (computing all 6 field components at every
cell, 3 of which are structurally guaranteed zero, forced dead by
tex()/tey()/tez()) - this was physically correct (see
tests/grid/test_te_mode_boundaries.py) but wasteful. The new TE branches
compute only the genuinely live components, only at the interior reference
layer - verified here to be bit-exact against the pre-existing 3D fallback
path they replace, not just "close" or "physically plausible".

Each comparison runs in its own subprocess (via a small driver script)
rather than back-to-back gprMax.run() calls in the test process - the
latter was found to hit pre-existing, unrelated cross-run global-state
leakage in gprMax's config module (a domain with `inf` on one axis run
immediately after a different domain with `inf` on a different axis, in
the same process, occasionally produced spurious mismatches that vanished
when either run was executed in isolation) - not something this change
introduced or is responsible for fixing, but something a same-process
before/after comparison needed to route around to give a reliable answer.

`SubgridUpdates`/`SubgridUpdater`/`MPIUpdates` all inherit
`CPUUpdates.__init__`/`update_electric_a`/`update_magnetic` unchanged, so
this fix reaches them automatically; not separately tested here since 2D
mode is already mutually exclusive with both subgrids and MPI, so mode2d
always resolves to -1 (3D, unaffected) for those paths.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

INF = float("inf")

TE_AXIS_CONFIG = {
    "x": {"domain": (INF, 0.01, 0.01), "src_p1": (0.001, 0.005, 0.005), "pol": "y"},
    "y": {"domain": (0.01, INF, 0.01), "src_p1": (0.005, 0.001, 0.005), "pol": "x"},
    "z": {"domain": (0.01, 0.01, INF), "src_p1": (0.005, 0.005, 0.001), "pol": "x"},
}

_DRIVER = textwrap.dedent(
    """
    import sys
    import numpy as np
    import gprMax
    import gprMax.model as model_mod
    from gprMax.updates.cpu_updates import CPUUpdates

    inf = float("inf")
    domain = {domain!r}
    src_p1 = {src_p1!r}
    pol = {pol!r}
    force_3d = {force_3d!r}
    outfile = sys.argv[1]

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation=pol, p1=src_p1, waveform_id="w"))

    if force_3d:
        orig_init = CPUUpdates.__init__
        def patched_init(self, G):
            orig_init(self, G)
            self.mode2d = -1
        CPUUpdates.__init__ = patched_init

    captured = {{}}
    orig_build = model_mod.Model.build
    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G
    model_mod.Model.build = patched_build

    gprMax.run(scenes=[scene], n=1, outputfile=outfile, hide_progress_bars=True)

    grid = captured["grid"]
    np.savez(
        outfile + "_fields.npz",
        Ex=np.asarray(grid.Ex), Ey=np.asarray(grid.Ey), Ez=np.asarray(grid.Ez),
        Hx=np.asarray(grid.Hx), Hy=np.asarray(grid.Hy), Hz=np.asarray(grid.Hz),
    )
    """
)


def _run_te_subprocess(tmp_path, axis, force_3d):
    cfg = TE_AXIS_CONFIG[axis]
    script = _DRIVER.format(
        domain=cfg["domain"], src_p1=cfg["src_p1"], pol=cfg["pol"], force_3d=force_3d
    )
    script_path = tmp_path / f"driver_{axis}_{force_3d}.py"
    script_path.write_text(script)
    outfile = tmp_path / f"te_{axis}_{'3d' if force_3d else 'reduced'}"

    result = subprocess.run(
        [sys.executable, str(script_path), str(outfile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"driver subprocess failed:\n{result.stdout}\n{result.stderr}"

    return np.load(str(outfile) + "_fields.npz")


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_te_reduced_kernel_bit_exact_vs_3d_fallback(tmp_path, axis):
    """The core correctness guarantee: the new reduced-kernel path must
    produce byte-for-byte identical fields to the pre-existing (slower)
    3D fallback path it replaces, across a real time-stepped solve with a
    real source - not just "physically plausible", but exactly the same
    numbers.
    """
    fields_reduced = _run_te_subprocess(tmp_path, axis, force_3d=False)
    fields_3d = _run_te_subprocess(tmp_path, axis, force_3d=True)

    for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        assert np.array_equal(
            fields_reduced[comp], fields_3d[comp]
        ), f"{comp} differs between reduced and 3D-fallback paths for TE{axis}"

    # Sanity: confirm this is a genuinely non-trivial comparison (some
    # component actually carries real, non-zero field) rather than two
    # empty arrays trivially matching.
    assert any(np.max(np.abs(fields_reduced[c])) > 0 for c in ["Ex", "Ey", "Ez"])


def test_tm_mode2d_dispatch_still_correct(tmp_path):
    """TM's detection moved from implicit (nx/ny/nz==1) to explicit
    (mode2d, driven by config.get_model_config().mode) - confirm it still
    correctly identifies the mode and produces the right live/dead
    component split.
    """
    import gprMax
    import gprMax.model as model_mod
    from gprMax.updates.cpu_updates import CPUUpdates

    captured_mode2d = {}
    orig_init = CPUUpdates.__init__

    def patched_init(self, G):
        orig_init(self, G)
        captured_mode2d["value"] = self.mode2d

    CPUUpdates.__init__ = patched_init

    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    model_mod.Model.build = patched_build

    try:
        scene = gprMax.Scene()
        scene.add(gprMax.DomainMode(mode="TM"))
        scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
        scene.add(gprMax.Domain(p1=(0.01, 0.01, INF)))
        scene.add(gprMax.PMLThickness(thickness=0))
        scene.add(gprMax.TimeWindow(time=1e-10))
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
        scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.005, 0.005, 0), waveform_id="w"))
        gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "tm_dispatch", hide_progress_bars=True)
    finally:
        CPUUpdates.__init__ = orig_init
        model_mod.Model.build = orig_build

    assert captured_mode2d["value"] == 2  # TMz
    grid = captured["grid"]
    assert np.max(np.abs(grid.Ez)) > 0
    assert np.max(np.abs(grid.Ex)) == 0
    assert np.max(np.abs(grid.Ey)) == 0
    assert not np.any(np.isnan(grid.Ez))


def test_3d_model_unaffected(tmp_path):
    """A plain 3D model must resolve mode2d to -1 and take the unchanged
    3D branch."""
    import gprMax
    from gprMax.updates.cpu_updates import CPUUpdates

    captured_mode2d = {}
    orig_init = CPUUpdates.__init__

    def patched_init(self, G):
        orig_init(self, G)
        captured_mode2d["value"] = self.mode2d

    CPUUpdates.__init__ = patched_init

    try:
        scene = gprMax.Scene()
        scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
        scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
        scene.add(gprMax.PMLThickness(thickness=0))
        scene.add(gprMax.TimeWindow(time=1e-10))
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
        scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.005, 0.005, 0.005), waveform_id="w"))
        gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "plain_3d", hide_progress_bars=True)
    finally:
        CPUUpdates.__init__ = orig_init

    assert captured_mode2d["value"] == -1
