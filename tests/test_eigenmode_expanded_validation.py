"""Independent truth checks for combined crossings and degenerate subspaces."""

import pytest
import numpy as np

from testing.eigenmode_support.subspaces.expanded_mode_tracking import (
    CASES, recovery_and_refinement_trials, controlled_trial, physical_sweeps,
)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("reference", (-1., 0., 1.))
def test_expanded_known_subspace_continuation(case, reference):
    controlled_trial(case, seed=129, reference=reference)


def test_rotated_crossings_recover_and_coarse_avoided_crossings_require_refinement():
    recovery_and_refinement_trials(seeds=3)


@pytest.mark.parametrize('case', ['three_pairs', 'triple_pair_single', 'repeated_crossings', 'complex_nonnormal'])
def test_crossing_projectors_converge_with_anchor_spacing(case):
    errors = [controlled_trial(case, seed=129, reference=-1., anchors=count)['maximum_projector_error']
              for count in (21, 41, 81)]
    # Neighbor projection is first order for evolving physical subspaces.
    np.testing.assert_allclose(np.array(errors[1:]) / errors[:-1], .5, rtol=.02)


@pytest.mark.integration
def test_full_vector_fdfd_crossings_against_independent_dispersion():
    physical_sweeps()
