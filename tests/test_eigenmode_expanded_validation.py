"""Independent truth checks for combined crossings and degenerate subspaces."""

import pytest

from testing.validation.expanded_mode_tracking import (
    CASES, recovery_and_refinement_trials, controlled_trial, physical_sweeps,
)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("reference", (-1., 0., 1.))
def test_expanded_known_subspace_continuation(case, reference):
    controlled_trial(case, seed=129, reference=reference)


def test_rotated_crossings_recover_and_coarse_avoided_crossings_require_refinement():
    recovery_and_refinement_trials(seeds=3)


@pytest.mark.integration
def test_full_vector_fdfd_crossings_against_independent_dispersion():
    physical_sweeps()
