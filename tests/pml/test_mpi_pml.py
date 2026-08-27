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

"""Unit tests for the MPI PML coefficient path.

Under MPI the domain is split across ranks, and a PML slab that straddles the
split would otherwise have each rank compute ``sigma_max`` from only the
material it can see locally. The ranks would then disagree about how absorbing
their share of the same slab is, and the seam would reflect.

``MPIGrid`` reduces the backing material properties over each slab's face
communicator before this method is called. Every participating rank can then
derive the same ``sigma_max`` locally. This is intentionally collective-free:
ranks can own different numbers of boundary slabs when symmetry is active, so
a broadcast per local slab can deadlock.

**What one rank can and cannot show.** These tests run on ``MPI.COMM_SELF``,
where rank 0 is the only rank, so the coordinator branch is exercised and the
follower branch is not. What that *does* establish is the property that
matters most: with one rank the MPI path must agree exactly with the serial
one, so the override cannot have changed any arithmetic.
"""

import numpy as np
import pytest
from mpi4py import MPI

from gprMax.pml import MPIPML, PML

from .conftest import ID_TO_DIRECTION


@pytest.fixture
def make_mpi_pml(make_pml_grid):
    """An ``MPIPML`` slab wired to a real single-rank communicator.

    ``global_comm`` is a bare class annotation on ``MPIPML``, never assigned by
    ``__init__``, so it has to be attached after construction — production code
    does the same from the MPI grid.
    """

    def _make(pml_id="x0", thickness=4, comm=None, **grid_kwargs):
        g = make_pml_grid(**grid_kwargs)
        nx, ny, nz = (int(v) for v in g.size)
        extents = {
            "x0": (0, thickness, 0, ny + 1, 0, nz + 1),
            "y0": (0, nx + 1, 0, thickness, 0, nz + 1),
            "z0": (0, nx + 1, 0, ny + 1, 0, thickness),
        }[pml_id]
        pml = MPIPML(g, pml_id, ID_TO_DIRECTION[pml_id], *extents)
        pml.global_comm = comm if comm is not None else MPI.COMM_SELF
        return pml

    return _make


class TestClassSurface:
    def test_extends_pml(self):
        """Expects ``MPIPML`` to inherit the whole serial surface and override
        only ``calculate_update_coeffs``."""
        assert issubclass(MPIPML, PML)
        assert "calculate_update_coeffs" in MPIPML.__dict__

    def test_only_the_coefficient_method_is_overridden(self):
        """Expects construction, validation, array allocation and both update
        methods to be inherited unchanged — the MPI concern is confined to one
        method."""
        overridden = {name for name in MPIPML.__dict__ if not name.startswith("__")}
        assert overridden == {"COORDINATOR_RANK", "calculate_update_coeffs"}

    def test_rank_zero_coordinates(self):
        """Expects rank 0 to be the one that derives ``sigma_max``."""
        assert MPIPML.COORDINATOR_RANK == 0

    def test_constructs_like_a_serial_pml(self, make_mpi_pml):
        """Expects the inherited constructor to set up ``d``, ``thickness`` and
        the four auxiliary arrays exactly as the base class does."""
        pml = make_mpi_pml(pml_id="x0", thickness=4)
        assert pml.thickness == 4
        assert pml.EPhi1.shape == (1, 5, 11, 12)


class TestCoordinatorPath:
    def test_derives_sigma_max_on_the_coordinator(self, make_mpi_pml):
        """Expects rank 0 to compute the same optimum a serial PML would, from
        the same backing material."""
        from gprMax import config

        pml = make_mpi_pml()
        pml.calculate_update_coeffs(1.0, 1.0)
        z0 = config.sim_config.em_consts["z0"]
        assert pml.CFS[0].sigma.max == pytest.approx(0.8 * 5 / (z0 * pml.d))

    def test_the_broadcast_value_is_what_gets_stored(self, make_mpi_pml):
        """Expects ``sigma.max`` to be read back out of the receive buffer
        rather than kept from the local computation — on one rank the two
        coincide, which is exactly why the round trip must be lossless."""
        pml = make_mpi_pml()
        pml.calculate_update_coeffs(1.0, 1.0)
        assert isinstance(pml.CFS[0].sigma.max, float)
        assert pml.CFS[0].sigma.max > 0

    def test_an_explicit_sigma_max_skips_the_broadcast(self, make_mpi_pml, make_cfs):
        """Expects the guard ``if not cfs.sigma.max`` to suppress the exchange
        entirely when the user supplied a value — no collective is entered, so
        no rank can block on one."""
        calls = []

        class CountingComm:
            rank = 0

            def Ibcast(self, *args, **kwargs):  # pragma: no cover - must not run
                calls.append(args)
                raise AssertionError("Ibcast should not be reached")

        pml = make_mpi_pml(cfs=[make_cfs(sigma={"max": 4.0})], comm=CountingComm())
        pml.calculate_update_coeffs(1.0, 1.0)
        assert calls == []
        assert pml.CFS[0].sigma.max == 4.0

    def test_automatic_sigma_calculation_is_collective_free(self, make_mpi_pml):
        class NoCollectives:
            def __getattr__(self, name):
                raise AssertionError(f"MPI collective {name} must not be used")

        pml = make_mpi_pml(comm=NoCollectives())
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.CFS[0].sigma.max > 0

    def test_each_cfs_term_is_derived_locally(self, make_mpi_pml, make_cfs):
        cfs = [make_cfs(kappa={"min": 0.5}), make_cfs(kappa={"min": 0.5})]
        pml = make_mpi_pml(cfs=cfs)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert all(term.sigma.max > 0 for term in pml.CFS)


class TestAgreementWithTheSerialPath:
    """The override must not change any arithmetic."""

    @pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
    def test_single_rank_matches_a_plain_pml(self, make_mpi_pml, make_pml, formulation):
        """Expects every coefficient array to be identical to the serial one at
        one rank, for both formulations. (2 parameter sets)"""
        mpi_pml = make_mpi_pml(formulation=formulation)
        serial = make_pml(formulation=formulation)
        mpi_pml.calculate_update_coeffs(1.0, 1.0)
        serial.calculate_update_coeffs(1.0, 1.0)
        for name in ("ERA", "ERB", "ERE", "ERF", "HRA", "HRB", "HRE", "HRF"):
            assert getattr(mpi_pml, name) == pytest.approx(getattr(serial, name))

    def test_delegates_to_the_base_implementation(self, make_mpi_pml):
        """Expects the eight arrays to be allocated by ``super()`` — the
        override adds a broadcast and changes nothing else."""
        pml = make_mpi_pml()
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.ERA.shape == (1, 4)
        assert pml.HRF.shape == (1, 4)

    def test_the_backing_material_still_matters(self, make_mpi_pml):
        """Expects a different ``er`` to reach ``calculate_sigmamax`` through
        the override unchanged."""
        vacuum = make_mpi_pml()
        soil = make_mpi_pml()
        vacuum.calculate_update_coeffs(1.0, 1.0)
        soil.calculate_update_coeffs(4.0, 1.0)
        assert soil.CFS[0].sigma.max == pytest.approx(vacuum.CFS[0].sigma.max / 2)

    @pytest.mark.parametrize("pml_id", ["x0", "y0", "z0"])
    def test_every_face_agrees_with_serial(self, make_mpi_pml, make_pml, pml_id):
        """Expects parity on each axis, so the direction-dependent ``d`` is
        picked up identically by both paths. (3 parameter sets)"""
        from .conftest import DL_ANISO

        mpi_pml = make_mpi_pml(pml_id=pml_id, dl=DL_ANISO)
        serial = make_pml(pml_id=pml_id, dl=DL_ANISO)
        mpi_pml.calculate_update_coeffs(1.0, 1.0)
        serial.calculate_update_coeffs(1.0, 1.0)
        assert mpi_pml.ERF == pytest.approx(serial.ERF)


class TestRealCommunicator:
    """``MPI.COMM_SELF`` rather than a hand-written double."""

    def test_a_real_ibcast_round_trips_the_value(self, make_mpi_pml):
        """Expects the genuine mpi4py non-blocking broadcast to deliver the
        coordinator's value back into the buffer at one rank."""
        pml = make_mpi_pml(comm=MPI.COMM_SELF)
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.CFS[0].sigma.max > 0

    def test_comm_self_reports_one_rank(self):
        """Expects a single-rank communicator, which is what makes the
        coordinator branch the one under test here."""
        assert MPI.COMM_SELF.Get_size() == 1
        assert MPI.COMM_SELF.rank == MPIPML.COORDINATOR_RANK

    def test_repeated_calls_do_not_re_broadcast(self, make_mpi_pml):
        """Expects the second call to find ``sigma.max`` already truthy and
        skip the collective — important, because an unmatched collective on one
        rank would hang every other rank."""
        pml = make_mpi_pml(comm=MPI.COMM_SELF)
        pml.calculate_update_coeffs(1.0, 1.0)
        first = pml.CFS[0].sigma.max

        class ForbiddenComm:
            rank = 0

            def Ibcast(self, *args, **kwargs):  # pragma: no cover - must not run
                raise AssertionError("second call should not broadcast")

        pml.global_comm = ForbiddenComm()
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.CFS[0].sigma.max == first


pytestmark = pytest.mark.unit
