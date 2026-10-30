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
Python API, and stopped with a bare ValueError from an input file). The check is exact: a seed
is never rounded through a float on the way."""

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import gprMax
from gprMax.user_objects.cmds_geometry.cmds_geometry import fractal_seed


@pytest.mark.parametrize(
    "value",
    [
        42,
        42.0,
        "42",
        " 42 ",
        "42.0",
        "4.2e1",
        np.int64(42),
        np.float64(42),
        np.float32(42),
        Decimal("42.0"),
        Fraction(84, 2),
        np.array(42.0),
    ],
)
def test_whole_seeds_are_taken(value):
    assert fractal_seed(value, "#fractal_box") == 42


# 2**53 + 1: the first whole number a float64 cannot hold (it rounds to 2**53).
@pytest.mark.parametrize(
    "value",
    [
        9007199254740993,
        "9007199254740993",
        "9007199254740993.0",
        Decimal("9007199254740993.0"),
        np.longdouble(9007199254740993),
    ],
)
def test_a_large_seed_keeps_every_digit(value):
    if isinstance(value, np.longdouble) and np.finfo(np.longdouble).nmant < 53:
        pytest.skip("long double is a float64 here")
    assert fractal_seed(value, "#fractal_box") == 9007199254740993


def test_a_float_seed_is_taken_at_its_exact_value():
    assert fractal_seed(1e300, "#fractal_box") == int(1e300)


@pytest.mark.parametrize(
    "value",
    [
        4.2,
        "4.2",
        "abc",
        "nan",
        "inf",
        "sNaN",
        float("inf"),
        float("nan"),
        True,
        "4.0000000000000001",  # a float would round it to 4
        "1e-400",  # a float would round it to 0
        Decimal("9007199254740992.5"),
    ],
)
def test_other_seeds_are_refused_with_a_reason(value):
    with pytest.raises(ValueError, match="the seed must be a whole number"):
        fractal_seed(value, "#fractal_box")


def test_a_fractal_box_is_built_with_the_exact_seed(monkeypatch, tmp_path):
    from gprMax.fractals.fractal_volume import FractalVolume

    seeds = []
    build = FractalVolume.__init__

    def record(self, *args, **kwargs):
        build(self, *args, **kwargs)
        seeds.append(self.seed)

    monkeypatch.setattr(FractalVolume, "__init__", record)
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.06, 0.06, 0.06)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=1e-10))
    scene.add(
        gprMax.SoilPeplinski(
            sand_fraction=0.5,
            clay_fraction=0.5,
            bulk_density=2.0,
            sand_density=2.66,
            water_fraction_lower=0.001,
            water_fraction_upper=0.25,
            id="soil",
        )
    )
    scene.add(
        gprMax.FractalBox(
            p1=(0.02, 0.02, 0.02),
            p2=(0.04, 0.04, 0.04),
            frac_dim=1.5,
            weighting=(1, 1, 1),
            n_materials=10,
            mixing_model_id="soil",
            id="box",
            seed=Decimal("9007199254740993.0"),
        )
    )
    gprMax.run(scenes=[scene], n=1, geometry_only=True, outputfile=str(tmp_path / "exact_seed"))
    assert seeds == [9007199254740993]


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
