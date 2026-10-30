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

"""A fractal's seed is a whole number: given as a number or as text (an input file), 42, 42.0
and "42" are accepted, and 4.2 is refused with a clear message (it was cut down to 4 from the
Python API, and stopped with a bare ValueError from an input file)."""

import numpy as np
import pytest

import gprMax
from gprMax.user_objects.cmds_geometry.cmds_geometry import fractal_seed


@pytest.mark.parametrize("value", [42, 42.0, "42", " 42 ", "42.0", np.int64(42)])
def test_whole_seeds_are_taken(value):
    assert fractal_seed(value, "#fractal_box") == 42


@pytest.mark.parametrize("value", [4.2, "4.2", "abc", "nan", float("inf"), True])
def test_other_seeds_are_refused_with_a_reason(value):
    with pytest.raises(ValueError, match="the seed must be a whole number"):
        fractal_seed(value, "#fractal_box")


def test_no_seed():
    assert fractal_seed(None, "#fractal_box") is None


@pytest.mark.parametrize("seed", ["42", "42 n"])  # with and without averaging
def test_an_input_file_seed_is_a_whole_number_either_way(seed, tmp_path):
    path = tmp_path / "fractal.in"
    path.write_text(
        "#domain: 0.1 0.1 0.1\n#dx_dy_dz: 0.002 0.002 0.002\n#time_window: 1e-10\n"
        "#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.25 my_soil\n"
        f"#fractal_box: 0 0 0 0.1 0.1 0.05 1.5 1 1 1 10 my_soil my_box {seed}\n"
    )
    gprMax.run(inputfile=str(path), geometry_only=True)
    bad = tmp_path / "bad.in"
    bad.write_text(path.read_text().replace(f"my_box {seed}", seed.replace("42", "my_box 4.2")))
    with pytest.raises(ValueError, match="the seed must be a whole number, not '4.2'"):
        gprMax.run(inputfile=str(bad), geometry_only=True)
