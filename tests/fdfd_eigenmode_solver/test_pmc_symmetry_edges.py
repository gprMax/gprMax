"""Port mode solver on windows cut by PMC symmetry planes.

A rectangular PEC waveguide is symmetric about both mid-planes. For the
TE10 mode (E along the narrow axis), the mid-plane across the wide axis is a
magnetic wall and the mid-plane across the narrow axis an electric wall. A
window cut by the magnetic wall, with that edge declared PMC, must give the
same mode as the full cross-section.
"""

import numpy as np
import pytest

from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.modal_window import pec_electric_masks
from testing.eigenmode_support.validate_fdfd_eigenmodes import (
    _configured_solver_constants,
    _homogeneous_2d_arrays,
)

SPACING = 1e-3
FREQUENCY = 10e9


def _solve(nu, nv, pmc_edges=()):
    with _configured_solver_constants():
        solver = FDFD_2D_mode_solver(
            frequency=FREQUENCY,
            du=SPACING,
            dv=SPACING,
            mode_index=0,
            artificial_pec_masks=pec_electric_masks((nu, nv)),
            pmc_edges=pmc_edges,
            **_homogeneous_2d_arrays(nu, nv),
        )
        solver.solve()
    return solver


def _neff(solver):
    return complex(np.atleast_1d(solver.complex_neff)[0])


@pytest.fixture(scope="module")
def full_wide_u():
    # 24 x 12 mm guide, wide along u: TE10 cut-off 6.25 GHz, propagating at 10 GHz.
    return _neff(_solve(24, 12))


@pytest.mark.parametrize("edge", ["u_max", "u_min"])
def test_half_window_with_pmc_edge_matches_full_guide(full_wide_u, edge):
    half = _neff(_solve(12, 12, pmc_edges=(edge,)))
    assert half == pytest.approx(full_wide_u, rel=1e-10, abs=1e-12)


def test_quarter_window_with_pmc_and_pec_edges_matches_full_guide(full_wide_u):
    # The v_max edge keeps its artificial PEC: the narrow-axis mid-plane is an electric wall.
    quarter = _neff(_solve(12, 6, pmc_edges=("u_max",)))
    assert quarter == pytest.approx(full_wide_u, rel=1e-10, abs=1e-12)


def test_pmc_edge_along_v_matches_full_guide():
    full = _neff(_solve(12, 24))
    half = _neff(_solve(12, 12, pmc_edges=("v_max",)))
    assert half == pytest.approx(full, rel=1e-10, abs=1e-12)


def test_without_pmc_edge_the_half_window_is_a_narrower_guide(full_wide_u):
    # Closing the mid-plane with PEC halves the width: the 12 mm guide is cut off at 10 GHz.
    closed = _neff(_solve(12, 12))
    assert abs(closed.real - full_wide_u.real) > 0.1


def test_pmc_edge_keeps_pec_on_corner_shared_with_a_pec_edge():
    solver = _solve(12, 6, pmc_edges=("u_max",))
    # E_w at (u_max, v_max) is tangential to the PEC v_max edge and must stay constrained.
    assert solver.pec_w_mask[-1, -1]
    # E_w elsewhere on the PMC edge is free.
    assert not solver.pec_w_mask[-1, 1:-1].any()


def test_unknown_pmc_edge_is_rejected():
    with pytest.raises(ValueError, match="Unknown PMC window edge"):
        _solve(4, 4, pmc_edges=("w_max",))
