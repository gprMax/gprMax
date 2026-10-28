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

"""Pre-flight validation of MPI domain decomposition against subgrids.

The validator is pure integer arithmetic, so none of this needs ``mpi4py``.
Values are hand-computed: with is_os_sep=1 and interpolation=1 the coupling
region extends 1 + 1 + 1 = 3 cells beyond the working region (4 with
``#dispersive_averaging``).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.subgrids import mpi_preflight
from gprMax.subgrids.mpi_preflight import (
    MPISubgridConfigurationError,
    coupling_footprint,
    find_straddled_boundaries,
    partition_boundaries,
    validate_subgrid_against_mpi,
)
from gprMax.subgrids.user_objects import SubGridHSG

SIZE = (100, 100, 100)


def check(i0, i1, tasks, *, disp=False, size=SIZE, name="sg"):
    validate_subgrid_against_mpi(name, i0, i1, 1, 1, disp, size, tasks)


class TestPartitionBoundaries:
    def test_even_split(self):
        assert partition_boundaries(100, 2) == [50]

    def test_remainder_goes_to_the_lowest_ranks(self):
        # 10 cells over 3 ranks -> sizes 4, 3, 3 -> starts at 4 and 7.
        assert partition_boundaries(10, 3) == [4, 7]

    def test_single_rank_has_no_boundary(self):
        assert partition_boundaries(100, 1) == []

    def test_matches_mpigrid_local_extents(self):
        """Same arithmetic as ``MPIGrid.calculate_local_extents``."""
        n, p = 97, 4
        size, overflow = n // p, n % p
        expected = [size * r + min(r, overflow) for r in range(1, p)]
        assert partition_boundaries(n, p) == expected


class TestFootprint:
    def test_padding_is_is_os_sep_plus_precursor_plus_interpolation(self):
        lo, hi = coupling_footprint((10, 10, 10), (20, 20, 20), 1, 1, False)
        assert lo == (7, 7, 7) and hi == (23, 23, 23)

    def test_dispersive_averaging_adds_one_cell(self):
        lo, hi = coupling_footprint((10, 10, 10), (20, 20, 20), 1, 1, True)
        assert lo == (6, 6, 6) and hi == (24, 24, 24)

    def test_boundary_at_upper_edge_counts_as_crossed(self):
        """Cell b belongs to the higher rank, so hi == b touches both ranks."""
        assert find_straddled_boundaries((40,) * 3, (50,) * 3, SIZE, (2, 1, 1)) == [(0, 50)]

    def test_boundary_at_lower_edge_is_not_crossed(self):
        assert find_straddled_boundaries((50, 0, 0), (60, 9, 9), SIZE, (2, 1, 1)) == []


class TestValidator:

    def test_single_rank_cannot_straddle_but_is_still_unsupported(self, monkeypatch):
        with pytest.raises(MPISubgridConfigurationError) as exc:
            check((45, 45, 45), (55, 55, 55), (1, 1, 1))

        msg = str(exc.value)
        assert "not currently supported with MPI domain decomposition" in msg
        assert "Omit --mpi" in msg

        monkeypatch.setattr(
            mpi_preflight, "SUBGRIDS_ON_MPI_SOLVER_SUPPORTED", True
        )
        check((45, 45, 45), (55, 55, 55), (1, 1, 1))

    def test_straddling_subgrid_fails_with_a_descriptive_message(self):
        with pytest.raises(MPISubgridConfigurationError) as exc:
            check((45, 20, 20), (55, 30, 30), (2, 1, 1), name="gpr_patch")

        msg = str(exc.value)
        assert "gpr_patch" in msg
        assert "x = 50" in msg
        assert "straddles an MPI subdomain boundary" in msg
        assert "(2, 1, 1)" in msg
        assert "not currently supported with MPI domain decomposition" in msg
        assert "Omit --mpi" in msg

    def test_error_is_a_value_error(self):
        """Callers that catch ``ValueError`` (the repo convention) still work."""
        with pytest.raises(ValueError):
            check((45, 20, 20), (55, 30, 30), (2, 1, 1))

    def test_halo_alone_can_cause_the_crossing(self):
        # Working region ends at 46; the 3-cell coupling halo reaches 49 (fine)...
        mpi_preflight.SUBGRIDS_ON_MPI_SOLVER_SUPPORTED = True
        try:
            check((30, 20, 20), (46, 30, 30), (2, 1, 1))
            # ...but one more cell reaches the boundary at 50.
            with pytest.raises(MPISubgridConfigurationError, match="straddles"):
                check((30, 20, 20), (47, 30, 30), (2, 1, 1))
        finally:
            mpi_preflight.SUBGRIDS_ON_MPI_SOLVER_SUPPORTED = False

    def test_dispersive_averaging_can_cause_the_crossing(self):
        mpi_preflight.SUBGRIDS_ON_MPI_SOLVER_SUPPORTED = True
        try:
            check((30, 20, 20), (46, 30, 30), (2, 1, 1), disp=False)
            with pytest.raises(MPISubgridConfigurationError, match="dispersive averaging"):
                check((30, 20, 20), (46, 30, 30), (2, 1, 1), disp=True)
        finally:
            mpi_preflight.SUBGRIDS_ON_MPI_SOLVER_SUPPORTED = False

    def test_straddling_on_a_non_x_axis_is_reported_on_that_axis(self):
        with pytest.raises(MPISubgridConfigurationError, match="z = 50"):
            check((10, 10, 45), (20, 20, 55), (1, 1, 2))

    def test_subgrid_inside_one_rank_is_still_rejected_while_solver_lacks_support(self):
        with pytest.raises(MPISubgridConfigurationError) as exc:
            check((10, 10, 10), (20, 20, 20), (2, 1, 1))

        msg = str(exc.value)
        assert "not currently supported with MPI domain decomposition" in msg
        assert "complete coupling region lies" in msg
        assert "Omit --mpi" in msg

    def test_subgrid_inside_one_rank_passes_once_solver_supports_it(self, monkeypatch):
        monkeypatch.setattr(mpi_preflight, "SUBGRIDS_ON_MPI_SOLVER_SUPPORTED", True)
        check((10, 10, 10), (20, 20, 20), (2, 1, 1))

    @pytest.mark.parametrize(
        "i0, i1, tasks",
        [
            ((45, 20, 20), (55, 30, 30), (2, 1, 1)),
            ((45, 20, 20), (55, 30, 30), (1, 2, 1)),
            ((45, 20, 20), (55, 30, 30), (1, 1, 2)),
        ],
    )
    def test_ratio_one_subgrid_crossing_reports_mpi_limitation(
        self, i0, i1, tasks
    ):
        with pytest.raises(MPISubgridConfigurationError) as exc:
            validate_subgrid_against_mpi(
                "ratio_one_patch", i0, i1, 1, 1, False, SIZE, tasks
            )

        msg = str(exc.value)
        assert "not currently supported with MPI domain decomposition" in msg
        assert "Omit --mpi" in msg

    def test_ratio_one_subgrid_inside_one_rank_reports_mpi_limitation(self):
        with pytest.raises(MPISubgridConfigurationError) as exc:
            validate_subgrid_against_mpi(
                "ratio_one_patch",
                (10, 10, 10),
                (20, 20, 20),
                1,
                1,
                False,
                SIZE,
                (2, 1, 1),
            )

        msg = str(exc.value)
        assert "not currently supported with MPI domain decomposition" in msg
        assert "complete coupling region lies" in msg
        assert "Omit --mpi" in msg


class TestFailsDuringConstruction:
    """The check must fire in ``SubGridHSG.build`` (input processing), i.e.
    before any grid arrays exist and long before the time-stepping loop."""

    @staticmethod
    def _model(tasks, subgrids=None):
        G = SimpleNamespace(
            is_distributed=True,
            global_size=np.array(SIZE),
            mpi_tasks=np.array(tasks),
            dl=np.array([0.001] * 3),
        )
        return SimpleNamespace(G=G, subgrids=subgrids if subgrids is not None else [])

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        import gprMax.config as config

        monkeypatch.setattr(
            config, "get_model_config", lambda: SimpleNamespace(dispersive_averaging=False)
        )
        monkeypatch.setattr(
            SubGridHSG,
            "_create_uip",
            lambda self, grid: SimpleNamespace(
                discretise_static_point=lambda p: np.rint(np.array(p) / 0.001).astype(np.int32)
            ),
        )

    def _user_object(self, p1, p2):
        return SubGridHSG(p1=p1, p2=p2, ratio=3, id="patch", is_os_sep=1, interpolation=1)

    def test_straddling_subgrid_raises_before_any_subgrid_is_created(self):
        model = self._model((2, 1, 1))
        sg = SimpleNamespace(ratio=3)
        with pytest.raises(MPISubgridConfigurationError, match="straddles"):
            # working region x: 45..55 cells straddles the boundary at x=50
            self._user_object((0.045, 0.02, 0.02), (0.055, 0.03, 0.03)).setup(sg, model)
        assert model.subgrids == []  # nothing was registered or allocated

    def test_single_rank_model_is_untouched_by_the_check(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "gprMax.subgrids.user_objects.validate_subgrid_against_mpi",
            lambda **kw: called.append(kw),
        )
        model = self._model((1, 1, 1))
        model.G.is_distributed = False  # an ordinary FDTDGrid
        try:
            self._user_object((0.045, 0.02, 0.02), (0.055, 0.03, 0.03)).setup(
                SimpleNamespace(ratio=3), model
            )
        except Exception:
            pass  # later setup steps need a full grid; only the hook matters here
        assert called == []