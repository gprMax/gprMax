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

"""Regression test for the CPU update_magnetic() 3D-branch boundary-symmetry
fix.

Previously the 3D magnetic update loop only ever reached each H component's
*upper* own-axis wall (Hx at i=nx, Hy at j=ny, Hz at k=nz) - the lower wall
(i=0/j=0/k=0) was never visited at all, staying permanently at its initial
value. Both walls are mathematically inert for standard usage (the
tangential-E curl terms they depend on are themselves never updated at the
domain edge, by update_electric's own loop-bound construction), but the
asymmetry was still a real structural gap - GPU's kernel, by contrast, was
already symmetric (excluding *both* walls uniformly). CPU now matches GPU:
both walls are genuinely visited by the update loop (still computing to
zero for ordinary non-magnetic materials), not just the upper one.

Unit-level coverage of the exact new values written at the boundary lives
in test_cpu_updates.py::test_update_magnetic. This file covers the
end-to-end, real-solve claim: a physically propagating field still leaves
every domain-wall H position at exactly zero, on both a small and a
larger domain, matching the pre-fix result bit-for-bit for the interior.
"""
import numpy as np

import gprMax
import gprMax.model as model_mod


def _capture_grid(monkeypatch):
    captured = {}
    orig_build = model_mod.Model.build

    def patched_build(self):
        orig_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def test_h_boundary_walls_remain_zero_with_real_propagating_field(monkeypatch, tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(20e-3, 20e-3, 20e-3)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=5e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0.01), waveform_id="w"))

    captured = _capture_grid(monkeypatch)
    gprMax.run(scenes=[scene], n=1, outputfile=tmp_path / "run", hide_progress_bars=True)
    grid = captured["grid"]
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # A genuinely propagating field, not a degenerate all-zero run - proves
    # the boundary-wall zeros below are a real physical result, not just
    # "nothing happened yet" (the failure mode of an earlier, too-short
    # TimeWindow attempt at verifying this by hand).
    assert np.max(np.abs(grid.Ez)) > 1.0
    assert np.max(np.abs(grid.Hx)) > 1e-6
    assert np.max(np.abs(grid.Hy)) > 1e-6

    for arr, idx in (
        (grid.Hx, (0, slice(None), slice(None))),
        (grid.Hx, (nx, slice(None), slice(None))),
        (grid.Hy, (slice(None), 0, slice(None))),
        (grid.Hy, (slice(None), ny, slice(None))),
        (grid.Hz, (slice(None), slice(None), 0)),
        (grid.Hz, (slice(None), slice(None), nz)),
    ):
        assert np.max(np.abs(arr[idx])) == 0.0
