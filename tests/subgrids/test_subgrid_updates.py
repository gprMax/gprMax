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

"""Tests for ``subgrids/updates.py`` — the HSG choreography.

The main grid's leapfrog has two halves, so each main time step drives the
subgrid twice: ``hsg_1`` carries it from the main electric update to the main
magnetic update, and ``hsg_2`` mirrors that back.

Within each half the subgrid takes ``(ratio - 1) / 2`` *interpolated* steps
and one final *exact* step, then hands its result outward across the Outer
Surface. That ordering is the entire correctness argument for HSG, and it can
be asserted without any physics by recording the call sequence.
"""

from types import SimpleNamespace

import pytest

from gprMax.subgrids.precursor_nodes import PrecursorNodes, PrecursorNodesFiltered
from gprMax.subgrids.updates import SubgridUpdater, SubgridUpdates, create_updates


@pytest.fixture
def make_model(coupled_grids):
    """A minimal ``Model`` stand-in carrying one coupled subgrid."""

    def _make(filtered=True, count=1, **overrides):
        pairs = [coupled_grids(filtered=filtered, **overrides) for _ in range(count)]
        model = SimpleNamespace(
            G=pairs[0].main,
            subgrids=[p.sub for p in pairs],
        )
        for p in pairs:
            p.sub.filter = filtered
        return model, pairs

    return _make


class TestCreateUpdates:
    def test_returns_subgrid_updates(self, make_model):
        model, _ = make_model()
        assert isinstance(create_updates(model), SubgridUpdates)

    def test_one_updater_per_subgrid(self, make_model):
        model, _ = make_model(count=2)
        assert len(create_updates(model).updaters) == 2

    def test_filtered_subgrid_gets_filtered_precursors(self, make_model):
        model, _ = make_model(filtered=True)
        updater = create_updates(model).updaters[0]
        assert isinstance(updater.precursors, PrecursorNodesFiltered)

    def test_unfiltered_subgrid_gets_plain_precursors(self, make_model):
        model, _ = make_model(filtered=False)
        updater = create_updates(model).updaters[0]
        assert isinstance(updater.precursors, PrecursorNodes)
        assert not isinstance(updater.precursors, PrecursorNodesFiltered)

    def test_non_subgrid_raises(self, make_model):
        model, _ = make_model()
        model.subgrids = [object()]
        with pytest.raises(ValueError):
            create_updates(model)

    def test_no_subgrids_gives_no_updaters(self, make_model):
        model, _ = make_model()
        model.subgrids = []
        assert create_updates(model).updaters == []

    def test_updater_holds_the_main_grid(self, make_model):
        model, _ = make_model()
        assert create_updates(model).updaters[0].G is model.G

    def test_updater_holds_the_subgrid(self, make_model):
        model, _ = make_model()
        assert create_updates(model).updaters[0].grid is model.subgrids[0]


class TestSubgridUpdaterState:
    @pytest.fixture
    def updater(self, coupled_grids):
        c = coupled_grids()
        return SubgridUpdater(c.sub, c.precursors, c.main), c

    def test_iteration_starts_at_zero(self, updater):
        u, _ = updater
        assert u.iteration == 0

    def test_wires_up_its_three_collaborators(self, updater):
        u, c = updater
        assert u.grid is c.sub
        assert u.precursors is c.precursors
        assert u.G is c.main

    def test_electric_sources_advance_the_iteration(self, updater, monkeypatch):
        u, _ = updater
        monkeypatch.setattr(type(u).__mro__[1], "update_electric_sources", lambda self, it: None)
        u.update_electric_sources()
        assert u.iteration == 1

    def test_magnetic_sources_do_not_advance_the_iteration(self, updater, monkeypatch):
        u, _ = updater
        monkeypatch.setattr(type(u).__mro__[1], "update_magnetic_sources", lambda self, it: None)
        u.update_magnetic_sources()
        assert u.iteration == 0

    def test_store_outputs_does_not_advance_the_iteration(self, updater, monkeypatch):
        u, _ = updater
        monkeypatch.setattr(type(u).__mro__[1], "store_outputs", lambda self, it: None)
        u.store_outputs()
        assert u.iteration == 0


class TestHsgPhaseOne:
    """``hsg_1`` runs from the main electric update to the main magnetic one."""

    @pytest.fixture
    def phase(self, coupled_grids, spy_updater):
        def _make(ratio=3):
            c = coupled_grids(ratio=ratio)
            u = SubgridUpdater(c.sub, c.precursors, c.main)
            calls = spy_updater(u, c.precursors, c.sub)
            return u, calls, c

        return _make

    def test_starts_by_sampling_the_main_electric_field(self, phase):
        u, calls, _ = phase()
        u.hsg_1()
        assert calls[0] == "precursors.update_electric"

    def test_ends_by_pushing_out_across_the_outer_surface(self, phase):
        u, calls, _ = phase()
        u.hsg_1()
        assert calls[-1] == "sub.update_electric_os"

    def test_final_magnetic_sample_is_exact_not_interpolated(self, phase):
        u, calls, _ = phase()
        u.hsg_1()
        assert "precursors.calc_exact_magnetic_in_time" in calls
        # The exact sample is the last magnetic one taken.
        last_exact = len(calls) - 1 - calls[::-1].index("precursors.calc_exact_magnetic_in_time")
        last_interp = (
            len(calls) - 1 - calls[::-1].index("precursors.interpolate_magnetic_in_time")
            if "precursors.interpolate_magnetic_in_time" in calls
            else -1
        )
        assert last_exact > last_interp

    @pytest.mark.parametrize("ratio,expected", [(3, 1), (5, 2), (7, 3)])
    def test_interpolated_step_count_is_half_the_ratio(self, phase, ratio, expected):
        u, calls, _ = phase(ratio=ratio)
        u.hsg_1()
        assert calls.count("precursors.interpolate_magnetic_in_time") == expected

    @pytest.mark.parametrize("ratio,expected", [(3, 2), (5, 3), (7, 4)])
    def test_electric_substeps_equal_interpolated_plus_one(self, phase, ratio, expected):
        """Each loop iteration plus the trailing exact step."""
        u, calls, _ = phase(ratio=ratio)
        u.hsg_1()
        assert calls.count("update_electric_a") == expected

    def test_injects_across_the_inner_surface_each_substep(self, phase):
        u, calls, _ = phase(ratio=5)
        u.hsg_1()
        assert calls.count("sub.update_electric_is") == 3

    def test_pml_is_updated_alongside_the_fields(self, phase):
        u, calls, _ = phase()
        u.hsg_1()
        assert "update_electric_pml" in calls
        assert "update_magnetic_pml" in calls

    def test_outer_surface_is_pushed_exactly_once(self, phase):
        u, calls, _ = phase(ratio=5)
        u.hsg_1()
        assert calls.count("sub.update_electric_os") == 1

    def test_does_not_touch_the_magnetic_outer_surface(self, phase):
        u, calls, _ = phase()
        u.hsg_1()
        assert "sub.update_magnetic_os" not in calls


class TestHsgPhaseTwo:
    """``hsg_2`` is the mirror image of ``hsg_1``."""

    @pytest.fixture
    def phase(self, coupled_grids, spy_updater):
        def _make(ratio=3):
            c = coupled_grids(ratio=ratio)
            u = SubgridUpdater(c.sub, c.precursors, c.main)
            calls = spy_updater(u, c.precursors, c.sub)
            return u, calls, c

        return _make

    def test_starts_by_sampling_the_main_magnetic_field(self, phase):
        u, calls, _ = phase()
        u.hsg_2()
        assert calls[0] == "precursors.update_magnetic"

    def test_ends_by_pushing_out_across_the_outer_surface(self, phase):
        u, calls, _ = phase()
        u.hsg_2()
        assert calls[-1] == "sub.update_magnetic_os"

    def test_final_electric_sample_is_exact(self, phase):
        u, calls, _ = phase()
        u.hsg_2()
        assert "precursors.calc_exact_electric_in_time" in calls

    @pytest.mark.parametrize("ratio,expected", [(3, 1), (5, 2), (7, 3)])
    def test_interpolated_step_count_is_half_the_ratio(self, phase, ratio, expected):
        u, calls, _ = phase(ratio=ratio)
        u.hsg_2()
        assert calls.count("precursors.interpolate_electric_in_time") == expected

    @pytest.mark.parametrize("ratio,expected", [(3, 2), (5, 3), (7, 4)])
    def test_magnetic_substeps_equal_interpolated_plus_one(self, phase, ratio, expected):
        u, calls, _ = phase(ratio=ratio)
        u.hsg_2()
        assert calls.count("update_magnetic") == expected

    def test_outer_surface_is_pushed_exactly_once(self, phase):
        u, calls, _ = phase(ratio=5)
        u.hsg_2()
        assert calls.count("sub.update_magnetic_os") == 1

    def test_does_not_touch_the_electric_outer_surface(self, phase):
        u, calls, _ = phase()
        u.hsg_2()
        assert "sub.update_electric_os" not in calls


class TestSubgridUpdatesFanOut:
    """``SubgridUpdates`` simply fans each phase out to every subgrid."""

    class _Recorder:
        def __init__(self):
            self.calls = []

        def hsg_1(self):
            self.calls.append("hsg_1")

        def hsg_2(self):
            self.calls.append("hsg_2")

    def test_phase_one_reaches_every_updater(self, coupled_grids):
        c = coupled_grids()
        recorders = [self._Recorder(), self._Recorder()]
        updates = SubgridUpdates(c.main, recorders)
        updates.hsg_1()
        assert all(r.calls == ["hsg_1"] for r in recorders)

    def test_phase_two_reaches_every_updater(self, coupled_grids):
        c = coupled_grids()
        recorders = [self._Recorder(), self._Recorder()]
        updates = SubgridUpdates(c.main, recorders)
        updates.hsg_2()
        assert all(r.calls == ["hsg_2"] for r in recorders)

    def test_holds_the_main_grid(self, coupled_grids):
        c = coupled_grids()
        assert SubgridUpdates(c.main, []).grid is c.main

    def test_no_updaters_is_a_no_op(self, coupled_grids):
        c = coupled_grids()
        SubgridUpdates(c.main, []).hsg_1()


pytestmark = pytest.mark.unit
