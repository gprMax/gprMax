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

"""Property, dispatch and coordinate tests for ``FDTDGrid``.

Covers the parts of ``gprMax/grid/fdtd_grid.py`` that need no field arrays:
the ``size``/``dl`` property views, PML thickness bookkeeping, source and
receiver dispatch, bounds checking, and the coordinate helpers every geometry
command in PRs 6-8 relied on through a stub.
"""

import numpy as np
import pytest

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.receivers import Rx
from gprMax.sources import (
    DiscretePlaneWave,
    HertzianDipole,
    MagneticDipole,
    TransmissionLine,
    VoltageSource,
)

from .conftest import DL, DL_ANISO


class TestSizeProperties:
    """``nx``/``ny``/``nz`` are views onto the ``size`` array."""

    def test_defaults_to_zero(self):
        g = FDTDGrid()
        assert (g.nx, g.ny, g.nz) == (0, 0, 0)

    @pytest.mark.parametrize("axis,name", [(0, "nx"), (1, "ny"), (2, "nz")])
    def test_getter_reads_size(self, make_grid, axis, name):
        g = make_grid(nx=4, ny=5, nz=6, arrays=False)
        assert getattr(g, name) == g.size[axis]

    @pytest.mark.parametrize("axis,name", [(0, "nx"), (1, "ny"), (2, "nz")])
    def test_setter_writes_size(self, make_grid, axis, name):
        g = make_grid(arrays=False)
        setattr(g, name, 17)
        assert g.size[axis] == 17

    @pytest.mark.parametrize("name", ["nx", "ny", "nz"])
    def test_round_trip(self, make_grid, name):
        g = make_grid(arrays=False)
        setattr(g, name, 23)
        assert getattr(g, name) == 23

    def test_axes_are_independent(self, make_grid):
        """Setting one axis must not disturb the other two."""
        g = make_grid(nx=4, ny=5, nz=6, arrays=False)
        g.ny = 99
        assert (g.nx, g.ny, g.nz) == (4, 99, 6)


class TestDiscretisationProperties:
    """``dx``/``dy``/``dz`` are views onto the ``dl`` array."""

    def test_defaults_to_unity(self):
        g = FDTDGrid()
        assert (g.dx, g.dy, g.dz) == (1.0, 1.0, 1.0)

    @pytest.mark.parametrize("axis,name", [(0, "dx"), (1, "dy"), (2, "dz")])
    def test_getter_reads_correct_axis(self, make_grid, axis, name):
        """The whole point of the anisotropic fixture: a getter reading the
        wrong element of ``dl`` cannot pass here.
        """
        g = make_grid(dl=DL_ANISO, arrays=False)
        assert getattr(g, name) == DL_ANISO[axis]

    @pytest.mark.parametrize("axis,name", [(0, "dx"), (1, "dy"), (2, "dz")])
    def test_setter_writes_correct_axis(self, make_grid, axis, name):
        g = make_grid(arrays=False)
        setattr(g, name, 0.007)
        assert g.dl[axis] == 0.007

    @pytest.mark.parametrize("name", ["dx", "dy", "dz"])
    def test_round_trip(self, make_grid, name):
        g = make_grid(dl=DL_ANISO, arrays=False)
        setattr(g, name, 0.009)
        assert getattr(g, name) == 0.009


class TestSetPmlThickness:
    """PML thickness bookkeeping.

    Only the forms that actually work are pinned: a scalar and a 6-element
    sequence. The 1-element branch and the 2-to-5-element lengths are both
    broken in ways recorded in the analogy doc's Observations.
    """

    def test_constructor_default_is_ten(self):
        g = FDTDGrid()
        assert set(g.pmls["thickness"].values()) == {10}

    def test_scalar_sets_all_six(self, make_grid):
        g = make_grid(arrays=False)
        g.set_pml_thickness(4)
        assert all(v == 4 for v in g.pmls["thickness"].values())

    def test_six_element_sequence_maps_in_documented_order(self, make_grid):
        g = make_grid(arrays=False)
        g.set_pml_thickness((1, 2, 3, 4, 5, 6))
        assert g.pmls["thickness"] == {
            "x0": 1,
            "y0": 2,
            "z0": 3,
            "xmax": 4,
            "ymax": 5,
            "zmax": 6,
        }

    def test_zero_thickness_is_allowed(self, make_grid):
        """All-zero is the documented way to turn the PML off; ``build()``
        checks for exactly this.
        """
        g = make_grid(arrays=False)
        g.set_pml_thickness(0)
        assert all(v == 0 for v in g.pmls["thickness"].values())

    def test_key_order_is_stable(self, make_grid):
        """The dict is an ``OrderedDict`` specifically so PML *update* order
        never varies between models — the docstring calls this out because a
        different order changes the floating-point result.
        """
        g = make_grid(arrays=False)
        before = list(g.pmls["thickness"].keys())
        g.set_pml_thickness((1, 2, 3, 4, 5, 6))
        assert list(g.pmls["thickness"].keys()) == before

    def test_values_are_coerced_to_int(self, make_grid):
        g = make_grid(arrays=False)
        g.set_pml_thickness((1, 2, 3, 4, 5, 6))
        assert all(isinstance(v, int) for v in g.pmls["thickness"].values())


class TestAddSource:
    """``add_source`` routes five source types to five lists."""

    @staticmethod
    def _transmission_line():
        # The only source whose constructor takes arguments.
        return TransmissionLine(iterations=10, dt=1e-12)

    def test_voltage_source(self, make_grid):
        g = make_grid(arrays=False)
        src = VoltageSource()
        g.add_source(src)
        assert g.voltagesources == [src]

    def test_hertzian_dipole(self, make_grid):
        g = make_grid(arrays=False)
        src = HertzianDipole()
        g.add_source(src)
        assert g.hertziandipoles == [src]

    def test_magnetic_dipole(self, make_grid):
        g = make_grid(arrays=False)
        src = MagneticDipole()
        g.add_source(src)
        assert g.magneticdipoles == [src]

    def test_transmission_line(self, make_grid):
        g = make_grid(arrays=False)
        src = self._transmission_line()
        g.add_source(src)
        assert g.transmissionlines == [src]

    def test_discrete_plane_wave(self, make_grid):
        g = make_grid(arrays=False)
        # The only source whose constructor needs the grid itself.
        src = DiscretePlaneWave(g)
        g.add_source(src)
        assert g.discreteplanewaves == [src]

    def test_unknown_type_raises_type_error(self, make_grid):
        g = make_grid(arrays=False)
        with pytest.raises(TypeError):
            g.add_source(object())

    def test_each_source_lands_in_exactly_one_list(self, make_grid):
        """A source appended to two lists would be updated twice per step."""
        g = make_grid(arrays=False)
        g.add_source(HertzianDipole())
        populated = [
            name
            for name in (
                "voltagesources",
                "hertziandipoles",
                "magneticdipoles",
                "transmissionlines",
                "discreteplanewaves",
            )
            if getattr(g, name)
        ]
        assert populated == ["hertziandipoles"]

    def test_sources_accumulate_in_order(self, make_grid):
        g = make_grid(arrays=False)
        first, second = HertzianDipole(), HertzianDipole()
        g.add_source(first)
        g.add_source(second)
        assert g.hertziandipoles == [first, second]


class TestAddReceiver:
    def test_appends(self, make_grid):
        g = make_grid(arrays=False)
        rx = Rx()
        g.add_receiver(rx)
        assert g.rxs == [rx]

    def test_accumulates_in_order(self, make_grid):
        g = make_grid(arrays=False)
        first, second = Rx(), Rx()
        g.add_receiver(first)
        g.add_receiver(second)
        assert g.rxs == [first, second]


class TestWithinBounds:
    """The contract PRs 7 and 8 imitated in their stubs, tested for real."""

    def test_interior_point_returns_true(self, make_grid):
        g = make_grid(nx=8, ny=8, nz=8, arrays=False)
        assert g.within_bounds(np.array([4, 4, 4])) is True

    def test_origin_is_inside(self, make_grid):
        g = make_grid(nx=8, ny=8, nz=8, arrays=False)
        assert g.within_bounds(np.array([0, 0, 0])) is True

    def test_upper_bound_is_inclusive(self, make_grid):
        """``p > n`` is the failure test, so ``p == n`` is legal."""
        g = make_grid(nx=8, ny=9, nz=10, arrays=False)
        assert g.within_bounds(np.array([8, 9, 10])) is True

    @pytest.mark.parametrize(
        "point,axis",
        [
            ([-1, 0, 0], "x"),
            ([9, 0, 0], "x"),
            ([0, -1, 0], "y"),
            ([0, 10, 0], "y"),
            ([0, 0, -1], "z"),
            ([0, 0, 11], "z"),
        ],
    )
    def test_out_of_bounds_raises_naming_the_axis(self, make_grid, point, axis):
        g = make_grid(nx=8, ny=9, nz=10, arrays=False)
        with pytest.raises(ValueError, match=axis):
            g.within_bounds(np.array(point))

    def test_x_is_checked_before_y(self, make_grid):
        """Both axes are out of range; the error names ``x``, establishing
        that the checks run in x, y, z order.
        """
        g = make_grid(nx=8, ny=8, nz=8, arrays=False)
        with pytest.raises(ValueError, match="x"):
            g.within_bounds(np.array([99, 99, 0]))


class TestDiscretisePoint:
    def test_on_lattice_point_is_exact(self, make_grid):
        g = make_grid(dl=DL, arrays=False)
        assert g.discretise_point((3 * DL, 4 * DL, 5 * DL)) == (3, 4, 5)

    def test_origin(self, make_grid):
        g = make_grid(dl=DL, arrays=False)
        assert g.discretise_point((0.0, 0.0, 0.0)) == (0, 0, 0)

    def test_uses_the_matching_axis_of_dl(self, make_grid):
        """With anisotropic spacing, reading the wrong axis of ``dl`` gives a
        different cell index, so this pins the axis mapping.
        """
        g = make_grid(dl=DL_ANISO, arrays=False)
        point = (2 * DL_ANISO[0], 3 * DL_ANISO[1], 4 * DL_ANISO[2])
        assert g.discretise_point(point) == (2, 3, 4)

    def test_returns_plain_ints(self, make_grid):
        g = make_grid(dl=DL, arrays=False)
        assert all(isinstance(v, int) for v in g.discretise_point((DL, DL, DL)))

    def test_rounds_halves_toward_zero(self, make_grid):
        """``round_value`` uses ``ROUND_HALF_DOWN``: ties go toward zero, so
        2.5 becomes 2, not 3 (half-up) and not 2 by luck of banker's rounding
        — 3.5 would be 4 under banker's but is 3 here.
        """
        g = make_grid(dl=1.0, arrays=False)
        assert g.discretise_point((2.5, 3.5, 4.5)) == (2, 3, 4)

    def test_rounds_to_nearest_when_not_a_tie(self, make_grid):
        g = make_grid(dl=1.0, arrays=False)
        assert g.discretise_point((2.4, 3.6, 4.5001)) == (2, 4, 5)


class TestRoundToGrid:
    def test_on_lattice_point_round_trips(self, make_grid):
        g = make_grid(dl=DL, arrays=False)
        point = (3 * DL, 4 * DL, 5 * DL)
        assert g.round_to_grid(point) == pytest.approx(point)

    def test_snaps_off_lattice_point_to_nearest_cell(self, make_grid):
        g = make_grid(dl=1.0, arrays=False)
        assert g.round_to_grid((2.4, 3.6, 0.0)) == pytest.approx((2.0, 4.0, 0.0))

    def test_uses_the_matching_axis_of_dl(self, make_grid):
        g = make_grid(dl=DL_ANISO, arrays=False)
        point = (2 * DL_ANISO[0], 3 * DL_ANISO[1], 4 * DL_ANISO[2])
        assert g.round_to_grid(point) == pytest.approx(point)

    def test_is_idempotent(self, make_grid):
        """Rounding an already-rounded point must be a no-op."""
        g = make_grid(dl=DL_ANISO, arrays=False)
        once = g.round_to_grid((0.0027, 0.0051, 0.0093))
        assert g.round_to_grid(once) == pytest.approx(once)


class TestWithinPml:
    """``within_pml`` is inclusive of the slab's inner face on the low side
    and exclusive on the high side, per the ``<`` / ``>`` comparisons.

    The method builds its answer from numpy comparisons, so it returns
    ``np.bool_`` rather than a builtin ``bool``; assertions here coerce with
    ``bool()`` instead of using ``is True`` / ``is False``.
    """

    @pytest.fixture
    def pml_grid(self, make_grid):
        # 20 cells per axis with a 2-cell PML leaves a genuine interior.
        return make_grid(nx=20, ny=20, nz=20, arrays=False, pml_thickness=2)

    def test_centre_is_not_in_pml(self, pml_grid):
        assert bool(pml_grid.within_pml(np.array([10, 10, 10]))) is False

    @pytest.mark.parametrize(
        "point",
        [
            [1, 10, 10],  # inside x0
            [19, 10, 10],  # inside xmax
            [10, 1, 10],  # inside y0
            [10, 19, 10],  # inside ymax
            [10, 10, 1],  # inside z0
            [10, 10, 19],  # inside zmax
        ],
    )
    def test_points_inside_each_slab(self, pml_grid, point):
        assert bool(pml_grid.within_pml(np.array(point))) is True

    def test_inner_face_of_low_slab_is_outside(self, pml_grid):
        """``p < thickness`` means ``p == thickness`` is already interior."""
        assert bool(pml_grid.within_pml(np.array([2, 10, 10]))) is False

    def test_inner_face_of_high_slab_is_outside(self, pml_grid):
        """``p > n - thickness`` means ``p == n - thickness`` is interior."""
        assert bool(pml_grid.within_pml(np.array([18, 10, 10]))) is False

    def test_zero_thickness_means_nothing_is_in_pml(self, make_grid):
        g = make_grid(nx=20, ny=20, nz=20, arrays=False, pml_thickness=0)
        assert bool(g.within_pml(np.array([0, 0, 0]))) is False
        assert bool(g.within_pml(np.array([10, 10, 10]))) is False


class TestGetWaveformById:
    def test_returns_the_matching_waveform(self, make_grid, make_waveform):
        g = make_grid(arrays=False)
        wanted = make_waveform("gaussian")
        wanted.ID = "wanted"
        other = make_waveform("ricker")
        other.ID = "other"
        g.waveforms = [other, wanted]
        assert g.get_waveform_by_id("wanted") is wanted

    def test_returns_the_first_match(self, make_grid, make_waveform):
        g = make_grid(arrays=False)
        first = make_waveform("gaussian")
        first.ID = "dup"
        second = make_waveform("ricker")
        second.ID = "dup"
        g.waveforms = [first, second]
        assert g.get_waveform_by_id("dup") is first


class TestGridIdentity:
    def test_default_name(self):
        assert FDTDGrid().name == "main_grid"

    def test_id_lookup_covers_all_six_components(self):
        assert FDTDGrid.IDlookup == {
            "Ex": 0,
            "Ey": 1,
            "Ez": 2,
            "Hx": 3,
            "Hy": 4,
            "Hz": 5,
        }

    def test_collections_start_empty(self):
        g = FDTDGrid()
        assert g.materials == []
        assert g.mixingmodels == []
        assert g.fractalvolumes == []
        assert g.waveforms == []
        assert g.rxs == []
        assert g.snapshots == []

    def test_collections_are_not_shared_between_instances(self):
        """A class-level mutable default would leak sources between models."""
        a, b = FDTDGrid(), FDTDGrid()
        a.add_receiver(Rx())
        assert b.rxs == []

    def test_average_volume_objects_defaults_true(self):
        assert FDTDGrid().averagevolumeobjects is True


pytestmark = pytest.mark.unit
