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

"""Compare the ``Helix`` rasterisation with the official MATLAB reference model.

The reference replaces the wire with 15-degree PEC cylinder chords of radius
one cell. A chord sags up to 0.19 cells from the smooth centreline, so the
masks cannot match exactly at that wire thickness; the test therefore checks
extent, cell count, and bidirectional containment. The reference starts its
winding on +x and ``build_helix`` on +y, so the chain is rotated 90 degrees
about the axis; the axis lies on a grid plane, which makes that rotation exact.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import gprMax
import gprMax.model as model_mod

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_SCRIPT = (
    ROOT
    / "testing"
    / "other_codes"
    / "matlab_mom"
    / "antenna_helix_fs"
    / "helix_antenna_gprmax.py"
)
if not REFERENCE_SCRIPT.is_file():  # pragma: no cover
    pytest.skip("MATLAB reference model is not available", allow_module_level=True)

ACTIVATION = importlib.util.spec_from_file_location("helix_antenna_reference", REFERENCE_SCRIPT)
reference = importlib.util.module_from_spec(ACTIVATION)
ACTIVATION.loader.exec_module(reference)

DOMAIN = (0.160, 0.160, 0.160)


def _capture_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(reference.DL,) * 3))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    return scene


def _run(monkeypatch, tmp_path, label, scene):
    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )
    return captured["grid"]


def _pec_mask(grid):
    pec = next(material for material in grid.materials if material.ID == "pec")
    return grid.solid == pec.numID


def _rotate_about_axis(mask, axis):
    """Rotate a mask 90 degrees about a grid-plane axis in the xy plane."""

    rotated = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        source = 2 * axis - 1 - i
        if 0 <= source < mask.shape[1]:
            rotated[i] = mask[:, source, :]
    return rotated


def _bbox(mask):
    cells = np.argwhere(mask)
    return cells.min(axis=0), cells.max(axis=0)


def test_helix_primitive_matches_official_cylinder_chain(monkeypatch, tmp_path):
    chain_scene = _base_scene()
    segments = reference.helix_segments()
    assert len(segments) > 0
    for start, stop in segments:
        chain_scene.add(
            gprMax.Cylinder(
                p1=start,
                p2=stop,
                r=reference.HELIX_WIRE_RADIUS,
                material_id="pec",
                averaging="n",
            )
        )

    helix_scene = _base_scene()
    helix_scene.add(
        gprMax.Helix(
            p1=(reference.CENTRE_X, reference.CENTRE_Y, reference.GROUND_Z + reference.FEED_STUB_HEIGHT),
            p2=(
                reference.CENTRE_X,
                reference.CENTRE_Y,
                reference.GROUND_Z + reference.FEED_STUB_HEIGHT + reference.DL,
            ),
            coil_radius=reference.HELIX_RADIUS,
            pitch=reference.HELIX_SPACING,
            turns=float(reference.HELIX_TURNS),
            wire_radius=reference.HELIX_WIRE_RADIUS,
            handedness="r",
            averaging="n",
            material_id="pec",
        )
    )

    chain_mask = _pec_mask(_run(monkeypatch, tmp_path, "chain", chain_scene))
    helix_mask = _pec_mask(_run(monkeypatch, tmp_path, "helix", helix_scene))

    axis = int(round(reference.CENTRE_X / reference.DL))
    rotated_chain = _rotate_about_axis(chain_mask, axis)
    chain_cells = np.count_nonzero(rotated_chain)
    helix_cells = np.count_nonzero(helix_mask)
    intersection = np.count_nonzero(rotated_chain & helix_mask)

    chain_lower, chain_upper = _bbox(rotated_chain)
    helix_lower, helix_upper = _bbox(helix_mask)
    assert (tuple(chain_lower), tuple(chain_upper)) == (tuple(helix_lower), tuple(helix_upper))
    # The measured parity for the official one-cell wire is about 0.83.
    detail = f"chain cells {chain_cells}, helix cells {helix_cells}, intersection {intersection}"
    assert abs(chain_cells - helix_cells) / chain_cells <= 0.05, detail
    assert intersection / chain_cells >= 0.80, detail
    assert intersection / helix_cells >= 0.80, detail