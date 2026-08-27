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

"""Regression tests for Discretisation/Domain validation gaps
(Codex-reported):

1. Discretisation.build() used `any(self.discretisation) <= 0`, which
   compares a bool to 0 - `any()` is only False when EVERY element is
   falsy (exactly 0), so a discretisation with one negative and other
   positive values (e.g. (0.001, -0.001, 0.001)), or even an all-negative
   tuple, passed straight through with no error.

2. Domain.build() only rejected a resolved cell count of exactly 0
   (`model.nx == 0 or ...`), not negative counts from a negative raw
   domain dimension (discretise_static_point() does no sign checking).

3. Domain.build()'s implicit 2D-mode detection picked the FIRST axis
   with exactly 1 cell via an elif chain, silently ignoring that a
   SECOND axis might also have only 1 cell (ambiguous - 2D kernels
   assume exactly one invariant axis).
"""
import pytest

import gprMax


def _run(scene, tmp_path, label):
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / label,
        hide_progress_bars=True,
    )


def _scene_with(discretisation, domain, time=1e-12):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=discretisation))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.TimeWindow(time=time))
    return scene


# --- Discretisation: negative values ---


@pytest.mark.parametrize(
    "discretisation",
    [
        (0.001, -0.001, 0.001),
        (-0.001, -0.001, -0.001),
        (-0.001, 0.001, 0.001),
    ],
)
def test_negative_discretisation_rejected(discretisation, tmp_path):
    scene = _scene_with(discretisation, (0.05, 0.05, 0.05))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "neg_dl")


def test_positive_discretisation_still_works(tmp_path):
    scene = _scene_with((1e-3, 1e-3, 1e-3), (0.05, 0.05, 0.05))
    _run(scene, tmp_path, "pos_dl")


# --- Domain: negative resolved dimensions ---


def test_negative_domain_dimension_rejected(tmp_path):
    scene = _scene_with((1e-3, 1e-3, 1e-3), (-0.05, 0.05, 0.05))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "neg_domain")


# --- Domain: ambiguous multi-singleton-axis 2D detection ---


def test_two_singleton_axes_rejected_as_ambiguous(tmp_path):
    # 1 cell in both x and y - the old elif chain would silently pick x
    # (TMx) and ignore that y is also 1 cell.
    scene = _scene_with((1e-3, 1e-3, 1e-3), (1e-3, 1e-3, 0.1))
    with pytest.raises(ValueError):
        _run(scene, tmp_path, "two_singleton_axes")


def test_single_singleton_axis_still_resolves_2d(tmp_path):
    scene = _scene_with((1e-3, 1e-3, 1e-3), (0.05, 0.05, 1e-3))
    _run(scene, tmp_path, "one_singleton_axis")
