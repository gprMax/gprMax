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

"""Regression test for the "fine-detail target" subgrid pattern: a subgrid
encloses a stationary target, while the source and receiver live on the
MAIN grid and step between runs via #src_steps/#rx_steps for a B-scan.

This is a different scenario from tests/subgrids/test_subgrid_em_correctness.py,
where the source/rx are placed *inside* the subgrid (subgrid.add(...)) - that
scenario is not covered by src_steps/rx_steps at all (see the "still open"
gaps in the subgrid src/rx bug investigation). Here, src/rx are main-grid
objects (scene.add(...)), so Model.build()'s self.G.update_sources_and_recievers()
call - which only ever touches the main grid - should step them normally,
completely independent of whether a subgrid exists elsewhere in the model.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_main_grid_src_rx_steps_with_stationary_subgrid_target(tmp_path: Path):
    dl_main = 3e-3
    ratio = 3

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="mainrx_subgrid_steps"))
    scene.add(gprMax.Discretisation(p1=(dl_main, dl_main, dl_main)))
    scene.add(gprMax.Domain(p1=(0.18, 0.12, 0.12)))
    scene.add(gprMax.TimeWindow(time=1.5e-9))

    # Stationary fine-detail target subgrid, away from the src/rx sweep path.
    subgrid = gprMax.SubGridHSG(
        p1=(0.12, 0.045, 0.045), p2=(0.15, 0.075, 0.075), ratio=ratio, id="sg"
    )
    scene.add(subgrid)
    target = gprMax.Box(p1=(0.13, 0.055, 0.055), p2=(0.14, 0.065, 0.065), material_id="pec")
    subgrid.add(target)

    # Source/rx on the MAIN grid, stepped between runs.
    wf = gprMax.Waveform(wave_type="gaussianprime", amp=1, freq=1e9, id="mypulse")
    hd = gprMax.HertzianDipole(polarisation="z", p1=(0.03, 0.06, 0.06), waveform_id="mypulse")
    rx = gprMax.Rx(p1=(0.05, 0.06, 0.06))
    scene.add(wf)
    scene.add(hd)
    scene.add(rx)

    step = (0.006, 0.0, 0.0)  # 2 main-grid cells in x per run
    scene.add(gprMax.SrcSteps(p1=step))
    scene.add(gprMax.RxSteps(p1=step))

    n = 3
    outfile = tmp_path / "base"
    gprMax.run(
        scenes=[scene] * n,
        n=n,
        outputfile=outfile,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
    )

    src_positions = []
    rx_positions = []
    for i in range(1, n + 1):
        with h5py.File(tmp_path / f"base{i}.h5", "r") as f:
            src_positions.append(f["srcs/src1"].attrs["Position"])
            rx_positions.append(f["rxs/rx1"].attrs["Position"])

    for i in range(1, n):
        np.testing.assert_allclose(
            np.array(src_positions[i]) - np.array(src_positions[i - 1]), step
        )
        np.testing.assert_allclose(
            np.array(rx_positions[i]) - np.array(rx_positions[i - 1]), step
        )
