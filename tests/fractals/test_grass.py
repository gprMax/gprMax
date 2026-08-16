"""Unit tests for ``gprMax/fractals/grass.py``.

``Grass`` holds the random geometry of a clump of grass blades. Its
``(numblades, 6)`` parameter table is drawn once at construction:

- columns 0, 1 — the curvature scale of each blade in x and y, drawn from
  ``10 + 20 * U(0, 1)``. A blade of height ``h`` is displaced by
  ``(h / scale) ** 2`` cells, so a larger scale is a straighter blade.
- columns 2, 3 — the curvature *direction* in x and y, drawn from ±1.
- columns 4, 5 — the running x/y position of the blade's root. These start
  at zero and are mutated by ``calculate_root_geometry``, which walks the
  root one random step per call: roots wander, blades do not.

Everything here is deterministic given a seed.
"""

import numpy as np
import pytest

from gprMax.fractals.grass import Grass
from gprMax.utilities.utilities import round_value

SEED = 42


class TestGeometryParameters:
    def test_table_shape_and_dtype(self, fractal_config):
        g = Grass(5, SEED)
        assert g.geometryparams.shape == (5, 6)
        assert g.geometryparams.dtype == fractal_config.sim_config.dtypes["float_or_double"]

    def test_numblades_and_seed_are_stored(self):
        g = Grass(7, SEED)
        assert g.numblades == 7
        assert g.seed == SEED

    def test_curvature_scales_lie_in_the_documented_range(self):
        g = Grass(50, SEED)
        scales = g.geometryparams[:, 0:2]
        assert np.all(scales >= 10.0)
        assert np.all(scales < 30.0)

    def test_direction_columns_are_plus_or_minus_one(self):
        g = Grass(50, SEED)
        directions = g.geometryparams[:, 2:4]
        assert set(np.unique(directions)) <= {-1.0, 1.0}

    def test_root_accumulators_start_at_zero(self):
        g = Grass(5, SEED)
        assert np.all(g.geometryparams[:, 4:6] == 0)

    def test_same_seed_gives_the_same_table(self):
        assert np.array_equal(Grass(10, SEED).geometryparams, Grass(10, SEED).geometryparams)

    def test_different_seed_gives_a_different_table(self):
        assert not np.array_equal(Grass(10, SEED).geometryparams, Grass(10, 43).geometryparams)

    def test_zero_blades_gives_an_empty_table(self):
        g = Grass(0, SEED)
        assert g.geometryparams.shape == (0, 6)

    def test_six_generators_are_created(self):
        # R1-R4 fill the table at construction; R5/R6 drive the root walk.
        g = Grass(3, SEED)
        for name in ("R1", "R2", "R3", "R4", "R5", "R6"):
            assert hasattr(g, name)


class TestBladeGeometry:
    def test_a_blade_starts_directly_above_its_root(self):
        g = Grass(3, SEED)
        assert g.calculate_blade_geometry(0, 0.0) == (0, 0)

    def test_matches_the_quadratic_formula(self):
        g = Grass(3, SEED)
        blade, height = 1, 22.0
        params = g.geometryparams
        expected = (
            round_value(params[blade, 2] * (height / params[blade, 0]) ** 2),
            round_value(params[blade, 3] * (height / params[blade, 1]) ** 2),
        )
        assert g.calculate_blade_geometry(blade, height) == expected

    @pytest.mark.parametrize("height", [0.0, 5.0, 15.0, 30.0, 60.0])
    def test_matches_the_formula_at_every_height(self, height):
        g = Grass(4, SEED)
        params = g.geometryparams
        for blade in range(4):
            expected = (
                round_value(params[blade, 2] * (height / params[blade, 0]) ** 2),
                round_value(params[blade, 3] * (height / params[blade, 1]) ** 2),
            )
            assert g.calculate_blade_geometry(blade, height) == expected

    def test_displacement_grows_with_height(self):
        # The blade curves away from vertical: the magnitude of the offset
        # is non-decreasing in height.
        g = Grass(3, SEED)
        magnitudes = [abs(g.calculate_blade_geometry(0, h)[0]) for h in (0.0, 20.0, 40.0, 80.0)]
        assert magnitudes == sorted(magnitudes)

    def test_offset_sign_follows_the_direction_column(self):
        g = Grass(30, SEED)
        height = 60.0
        for blade in range(30):
            x, y = g.calculate_blade_geometry(blade, height)
            # At this height the quadratic term always rounds to a nonzero
            # value, so the sign is the direction column's.
            assert np.sign(x) == g.geometryparams[blade, 2]
            assert np.sign(y) == g.geometryparams[blade, 3]

    def test_returns_integer_cell_offsets(self):
        g = Grass(3, SEED)
        x, y = g.calculate_blade_geometry(0, 33.0)
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_is_a_pure_function_of_blade_and_height(self):
        # Unlike the root walk, asking twice gives the same answer and
        # leaves the parameter table untouched.
        g = Grass(3, SEED)
        before = g.geometryparams.copy()
        first = g.calculate_blade_geometry(2, 25.0)
        second = g.calculate_blade_geometry(2, 25.0)
        assert first == second
        assert np.array_equal(g.geometryparams, before)


class TestRootGeometry:
    def test_root_walk_advances_the_accumulator(self):
        g = Grass(3, SEED)
        assert g.geometryparams[0, 4] == 0
        g.calculate_root_geometry(0, 0)
        assert g.geometryparams[0, 4] != 0

    def test_each_step_moves_the_accumulator_by_at_most_one_cell(self):
        # The step is drawn from -1 + 2 * U(0, 1), i.e. U(-1, 1).
        g = Grass(3, SEED)
        previous_x, previous_y = 0.0, 0.0
        for depth in range(10):
            g.calculate_root_geometry(0, depth)
            x, y = g.geometryparams[0, 4], g.geometryparams[0, 5]
            assert abs(x - previous_x) <= 1.0
            assert abs(y - previous_y) <= 1.0
            previous_x, previous_y = x, y

    def test_returns_the_rounded_accumulator(self):
        g = Grass(3, SEED)
        x, y = g.calculate_root_geometry(1, 0)
        assert x == round(g.geometryparams[1, 4])
        assert y == round(g.geometryparams[1, 5])

    def test_successive_calls_are_a_random_walk_not_a_pure_function(self):
        # Roots wander: the same (root, depth) asked twice gives two
        # different positions because the accumulator has moved on.
        g = Grass(3, SEED)
        positions = [g.geometryparams[0, 4]]
        for _ in range(5):
            g.calculate_root_geometry(0, 0)
            positions.append(float(g.geometryparams[0, 4]))
        assert len(set(positions)) == len(positions)

    def test_roots_are_independent_per_blade(self):
        g = Grass(3, SEED)
        g.calculate_root_geometry(0, 0)
        assert g.geometryparams[1, 4] == 0
        assert g.geometryparams[2, 4] == 0

    def test_walk_is_reproducible_across_instances(self):
        a, b = Grass(3, SEED), Grass(3, SEED)
        walk_a = [a.calculate_root_geometry(0, d) for d in range(6)]
        walk_b = [b.calculate_root_geometry(0, d) for d in range(6)]
        assert walk_a == walk_b


pytestmark = pytest.mark.unit
