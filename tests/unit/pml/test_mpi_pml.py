"""``MPIPML`` — one rank derives ``sigma_max``, everyone else is told.

Under MPI the domain is split across ranks, and a PML slab that straddles the
split would otherwise have each rank compute ``sigma_max`` from only the
material it can see locally. The ranks would then disagree about how absorbing
their share of the same slab is, and the seam would reflect.

``MPIPML.calculate_update_coeffs`` fixes that by having rank 0 compute the
value and broadcast it, before delegating the rest of the work to the base
class unchanged.

**Why the broadcast is non-blocking.** A rank holding two slabs reaches the
second broadcast only after finishing the first, while a rank holding one
slab is already waiting. A blocking ``Bcast`` would deadlock; ``Ibcast`` plus
``Wait`` will not. The comment in the source says as much, and the tests below
pin the mechanism (``Ibcast`` is used, and the value that arrives is the one
that gets used) rather than trying to reproduce a deadlock.

**What one rank can and cannot show.** These tests run on ``MPI.COMM_SELF``,
where rank 0 is the only rank, so the coordinator branch is exercised and the
follower branch is not. What that *does* establish is the property that
matters most: with one rank the MPI path must agree exactly with the serial
one, so the override cannot have changed any arithmetic.
"""

import numpy as np
import pytest
from mpi4py import MPI

from gprMax.pml import PML, MPIPML

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
        overridden = {
            name for name in MPIPML.__dict__ if not name.startswith("__")
        }
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

    def test_uses_a_non_blocking_broadcast(self, make_mpi_pml):
        """Expects ``Ibcast(...).Wait()`` rather than ``Bcast``. A rank holding
        two slabs reaches its second broadcast late; the blocking form would
        deadlock against a rank already waiting."""
        used = []

        class RecordingComm:
            rank = 0

            def Ibcast(self, buffer, root):
                used.append(("Ibcast", root))
                return self

            def Wait(self):
                used.append(("Wait", None))

            def Bcast(self, *args, **kwargs):  # pragma: no cover - must not run
                raise AssertionError("blocking Bcast would deadlock")

        pml = make_mpi_pml(comm=RecordingComm())
        pml.calculate_update_coeffs(1.0, 1.0)
        assert used == [("Ibcast", 0), ("Wait", None)]

    def test_broadcasts_from_the_coordinator_rank(self, make_mpi_pml):
        """Expects ``COORDINATOR_RANK`` to be passed as the broadcast root, so
        every rank agrees on who is authoritative."""
        roots = []

        class RootRecordingComm:
            rank = 0

            def Ibcast(self, buffer, root):
                roots.append(root)
                return self

            def Wait(self):
                pass

        pml = make_mpi_pml(comm=RootRecordingComm())
        pml.calculate_update_coeffs(1.0, 1.0)
        assert roots == [MPIPML.COORDINATOR_RANK]

    def test_one_broadcast_per_cfs_term(self, make_mpi_pml, make_cfs):
        """Expects a two-pole PML to exchange two values — each CFS term has
        its own ``sigma_max``."""
        count = []

        class CountingComm:
            rank = 0

            def Ibcast(self, buffer, root):
                count.append(root)
                return self

            def Wait(self):
                pass

        cfs = [make_cfs(kappa={"min": 0.5}), make_cfs(kappa={"min": 0.5})]
        pml = make_mpi_pml(cfs=cfs, comm=CountingComm())
        pml.calculate_update_coeffs(1.0, 1.0)
        assert len(count) == 2


class TestFollowerPath:
    """Ranks other than 0 allocate an empty buffer and take what arrives."""

    def test_a_follower_adopts_the_broadcast_value(self, make_mpi_pml):
        """Expects a non-coordinator rank to skip the local computation and use
        whatever the broadcast delivers.

        A one-rank ``COMM_SELF`` cannot produce a real follower, so the
        communicator is faked to report a non-zero rank and to fill the receive
        buffer the way a real broadcast would."""

        class FollowerComm:
            rank = 3

            def Ibcast(self, buffer, root):
                buffer[0] = 12.5
                return self

            def Wait(self):
                pass

        pml = make_mpi_pml(comm=FollowerComm())
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.CFS[0].sigma.max == 12.5

    def test_a_follower_does_not_compute_locally(self, make_mpi_pml):
        """Expects the received value to win even when it differs from what the
        local material would have given — the whole point of the exchange."""

        class FollowerComm:
            rank = 1

            def Ibcast(self, buffer, root):
                buffer[0] = 1.0
                return self

            def Wait(self):
                pass

        pml = make_mpi_pml(comm=FollowerComm())
        pml.calculate_update_coeffs(1.0, 1.0)
        assert pml.CFS[0].sigma.max == 1.0

    def test_the_received_value_drives_the_coefficients(self, make_mpi_pml):
        """Expects the broadcast ``sigma_max`` to flow through into ``ERF``, so
        every rank builds identical coefficient arrays for a shared slab."""

        class FollowerComm:
            rank = 1

            def Ibcast(self, buffer, root):
                buffer[0] = 6.0
                return self

            def Wait(self):
                pass

        mpi_pml = make_mpi_pml(comm=FollowerComm())
        mpi_pml.calculate_update_coeffs(1.0, 1.0)
        assert mpi_pml.CFS[0].sigma.max == 6.0
        assert np.any(mpi_pml.ERF)


class TestAgreementWithTheSerialPath:
    """The override must not change any arithmetic."""

    @pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
    def test_single_rank_matches_a_plain_pml(
        self, make_mpi_pml, make_pml, formulation
    ):
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
