"""A repeated eigenspace must survive an unrelated branch crossing it."""

from collections import defaultdict

import numpy as np
import pytest

from gprMax.eigenmode_config import EigenmodeTrackingConfig
from gprMax.eigenmode_tracking import _assign_step, _persistent_groups, _span_members, track_solver_bank
from gprMax.hash_cmds_multiuse import process_multicmds
from gprMax.user_objects.cmds_multiuse import EigenmodePort
from gprMax.sources import EigenmodeAnchorMismatchError
from testing.validation.persistent_mode_groups import make_owner, make_solver, pair_crossing_bank, unitary, validate


@pytest.mark.parametrize("reference", (-1.0, 0.0, 1.0))
@pytest.mark.parametrize("two_pairs", (False, True))
@pytest.mark.parametrize("seed", (7, 41, 129))
def test_subgroups_survive_cluster_merger_and_split(reference, two_pairs, seed):
    frequencies = np.linspace(-1, 1, 5)
    solvers = pair_crossing_bank(frequencies, seed=seed, two_pairs=two_pairs)
    owner = make_owner(reference)
    centre = int(np.argmin(abs(frequencies - reference)))
    reference_projectors = []
    for coordinate in (0, 2) if two_pairs else (0,):
        members = np.flatnonzero(np.sum(abs(solvers[centre].Ea[coordinate:coordinate + 2]) ** 2, axis=0) > 0.99)
        projector = np.diag([
            int(i in range(coordinate, coordinate + 2)) for i in range(solvers[0].num_modes)
        ])
        reference_projectors.append((members, projector))
    track_solver_bank(owner, frequencies, solvers, solvers[0].num_modes)
    expected = tuple(sorted(tuple(int(i + 1) for i in members) for members, _ in reference_projectors))
    assert owner.degenerate == expected
    for members, projector in reference_projectors:
        for solver in solvers:
            frame = solver.Ea[:, members]
            np.testing.assert_allclose(frame @ frame.conj().T, projector, atol=2e-14)
    assert np.max(owner.tracking_diagnostics["residuals"]) < 1e-14
    assert np.min(owner.tracking_diagnostics["overlaps"]) > 1 - 1e-13


def test_nearby_but_resolved_crossing_uses_individual_fields():
    values = np.array([-1.0, -1.0 + 4e-6])
    assigned, scores = _assign_step(np.eye(2), np.eye(2)[:, ::-1], values, values, EigenmodeTrackingConfig(), values)
    np.testing.assert_array_equal(assigned, (1, 0))
    np.testing.assert_allclose(scores, 1)


def test_nearby_resolved_modes_do_not_get_a_span_overlap_exemption():
    values = np.array([-1.0, -1.0 + 4e-6])
    rotation = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
    assigned, _ = _assign_step(np.eye(2), rotation, values, values, EigenmodeTrackingConfig(), values)
    assert np.all(assigned == -1)


def test_prediction_separates_a_temporary_coincidence():
    values = np.array([-1.0, -1.0])
    predicted = np.array([-0.99, -1.01])
    assigned, _ = _assign_step(np.eye(2), np.eye(2)[:, ::-1], values, values, EigenmodeTrackingConfig(), predicted)
    np.testing.assert_array_equal(assigned, (1, 0))


def test_full_group_spread_prevents_chained_degeneracy():
    gap = 1e-8
    groups = _persistent_groups(np.array([[0, 0.75 * gap, 1.5 * gap]]), gap)
    assert groups == ((0, 1), (2,))


def test_nonorthogonal_competing_column_requires_membership_margin():
    reference = np.eye(3)[:, :2]
    candidates = np.eye(3)
    candidates[:, 2] = (np.sqrt(0.99), 0, 0.1)
    assert _span_members(reference, candidates, (0, 1, 2), 0.02) is None
    selected, score = _span_members(reference, candidates, (0, 1, 2), 0.005)
    assert selected == (0, 1)
    assert score == pytest.approx(1)


def test_single_public_channel_retains_its_persistent_hidden_partner_at_crossing():
    solvers = pair_crossing_bank([-1, 0, 1])
    owner = make_owner()
    mapping = track_solver_bank(owner, [-1, 0, 1], solvers, 1)
    assert mapping.shape == (3, 3)
    assert owner.degenerate == ((1, 2),)
    assert owner._auto_tracking_mode_count == 3


def test_persistent_groups_can_be_larger_than_two():
    rng = np.random.default_rng(24)
    solvers = [make_solver([-4] * 4, unitary(rng, 4)) for _ in range(3)]
    owner = make_owner()
    track_solver_bank(owner, [-1, 0, 1], solvers, 4)
    assert owner.degenerate == ((1, 2, 3, 4),)


@pytest.mark.parametrize("reference", (-1.0, 0.0, 1.0))
def test_fully_rotated_crossing_abstains_without_modifying_solved_fields(reference):
    solvers = pair_crossing_bank([-1, 0, 1], mixed_crossing=True)
    before = [solver.Ea.copy() for solver in solvers]
    with pytest.raises(EigenmodeAnchorMismatchError, match="unresolved"):
        track_solver_bank(make_owner(reference), [-1, 0, 1], solvers, 3)
    for solver, original in zip(solvers, before):
        np.testing.assert_array_equal(solver.Ea, original)


def test_reproducible_validation_report():
    report = validate()
    assert report["cases"]["fully_mixed_exact_crossing"]["status"] == "unresolved"


def test_subspace_margin_is_available_through_hash_input():
    commands = defaultdict(lambda: None)
    commands["#eigenmode_band"] = ["band 6e9 14e9 3"]
    commands["#eigenmode_excitation"] = ["1 1 auto n"]
    commands["#eigenmode_port"] = ["1 0 0 0.02 0.05 0.05 0.02 + 1,2 auto tracking=auto subspace_margin=0.03"]
    port = next(obj for obj in process_multicmds(commands) if isinstance(obj, EigenmodePort))
    assert port.kwargs["tracking_config"].subspace_margin == 0.03


@pytest.mark.parametrize("value", (0, -1, np.nan, np.inf, 1.1))
def test_subspace_margin_requires_a_finite_probability_gap(value):
    with pytest.raises(ValueError, match="subspace_margin"):
        EigenmodeTrackingConfig(subspace_margin=value)
