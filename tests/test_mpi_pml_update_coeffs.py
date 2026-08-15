"""Tests for MPI PML update-coefficient initialisation."""

from types import SimpleNamespace

import pytest

from gprMax.pml import MPIPML, PML


class _AutomaticCFS:
    def __init__(self):
        self.sigma = SimpleNamespace(max=None)
        self.calls = []

    def calculate_sigmamax(self, d, er, mr):
        self.calls.append((d, er, mr))
        self.sigma.max = 123.0


class _ExplicitCFS(_AutomaticCFS):
    def __init__(self):
        super().__init__()
        self.sigma.max = 456.0


class _NoCollectiveComm:
    """Fail if the coefficient setup attempts an MPI collective."""

    def __getattr__(self, name):
        pytest.fail(f"MPI PML coefficient setup attempted communicator operation {name!r}")


def _make_pml(cfs):
    pml = object.__new__(MPIPML)
    pml.CFS = cfs
    pml.d = 0.002
    pml.global_comm = _NoCollectiveComm()
    return pml


def test_mpi_pml_calculates_automatic_sigma_locally(monkeypatch):
    """Each slab uses its already face-reduced er/mr without a collective.

    A global broadcast per local slab deadlocks when symmetry boundaries
    leave different ranks with different numbers of exterior PML slabs.
    """
    automatic = _AutomaticCFS()
    explicit = _ExplicitCFS()
    pml = _make_pml([automatic, explicit])
    parent_calls = []

    monkeypatch.setattr(
        PML,
        "calculate_update_coeffs",
        lambda self, er, mr: parent_calls.append((er, mr)),
    )

    pml.calculate_update_coeffs(er=4.0, mr=2.0)

    assert automatic.calls == [(0.002, 4.0, 2.0)]
    assert explicit.calls == []
    assert parent_calls == [(4.0, 2.0)]
