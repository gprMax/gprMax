"""Physical labels must survive arbitrary raw eigensolver gauges."""
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.eigenmode_tracking import (
    _canonical_subspace_transform,
    _moments,
    _power,
    align_groups,
    normalize_groups,
    normalize_polarizations,
    parse_port_options,
)
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.sources import EigenmodeSource
from testing.validation.validate_fdfd_eigenmodes import (
    CIRCULAR_SPACING,
    _circular_pec_masks,
    _circular_theory,
    _configured_solver_constants,
    _homogeneous_2d_arrays,
)


@pytest.fixture(scope="module")
def circular_solvers():
    solvers = []
    with _configured_solver_constants():
        for frequency in (6e9, 10e9, 14e9):
            masks = _circular_pec_masks(44)
            solver = FDFD_2D_mode_solver(
                frequency=frequency,
                du=CIRCULAR_SPACING,
                dv=CIRCULAR_SPACING,
                mode_index=1,
                pec_u_mask=masks[0],
                pec_v_mask=masks[1],
                pec_w_mask=masks[2],
                guess=-float(_circular_theory(frequency)) ** 2,
                **_homogeneous_2d_arrays(44, 44)
            )
            solver.retain_tracking_operator = True
            solver.solve()
            solvers.append(solver)
    return solvers


def bank(
    solvers,
    axis=2,
    direction="+",
    references=None,
    propagating=None,
    automatic=False,
    groups=((1, 2),),
):
    owner = EigenmodeSource(None)
    owner.normal_axis = axis
    owner.normal = "xyz"[axis]
    owner.direction = direction
    owner.transverse_axes = tuple(i for i in range(3) if i != axis)
    owner.port_index = 1
    owner.degenerate = groups
    mode_indices = tuple(sorted(item for group in groups for item in group))
    owner.mode_indices = mode_indices
    owner.mode_count = max(mode_indices)
    owner.mode_index = mode_indices[0]
    if automatic:
        owner.anchor_policy = "auto"
        owner.dft_start = solvers[1].frequency
        owner.dft_stop = solvers[-1].frequency
    owner.mode_polarizations = normalize_polarizations(references, owner.degenerate, axis)
    owner._tracking_impedance = config.SimulationConfig.em_consts["z0"]
    grid = SimpleNamespace(dl=np.full(3, CIRCULAR_SPACING))
    owner._tracking_grid = grid
    electric, magnetic = [], []
    for solver in solvers:
        fields = [owner._fields_from_solver_mode(solver, mode) for mode in mode_indices]
        electric.append([field[0] for field in fields])
        magnetic.append([field[1] for field in fields])
    align_groups(
        owner,
        grid,
        tuple(s.frequency for s in solvers),
        solvers,
        mode_indices,
        electric,
        magnetic,
        np.ones((len(solvers), len(mode_indices)), dtype=bool)
        if propagating is None
        else propagating,
    )
    return owner, grid, electric, magnetic


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("direction", ("+", "-"))
@pytest.mark.parametrize("diagonal", (False, True))
def test_circular_physical_basis_invariant(
    circular_solvers, axis, direction, diagonal, record_property
):
    transverse = [i for i in range(3) if i != axis]
    a, b = np.eye(3)[transverse]
    references = {1: a + b if diagonal else b, 2: -a + b if diagonal else a}
    owner, grid, expected_e, expected_h = bank(circular_solvers, axis, direction, references)
    changed = deepcopy(circular_solvers)
    rng = np.random.default_rng(171)
    for solver in changed:
        # Nonunitary invertible rotations include phase and ordering changes.
        rotation = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw"):
            setattr(solver, name, getattr(solver, name) @ rotation)
    actual, _, electric, magnetic = bank(changed, axis, direction, references)
    errors, cast_errors = [], []
    for k in range(len(changed)):
        for family, expected in ((electric, expected_e), (magnetic, expected_h)):
            for i in range(2):
                for j in range(3):
                    np.testing.assert_allclose(
                        family[k][i][j], expected[k][i][j], atol=1e-10, rtol=1e-10
                    )
        moment = _moments(actual, grid, electric[k])
        wanted = np.array(list(actual.mode_polarizations.values())).T[transverse]
        errors.append(
            float(np.max(np.linalg.norm(moment / np.linalg.norm(moment, axis=0) - wanted, axis=0)))
        )
        np.testing.assert_allclose(moment / np.linalg.norm(moment, axis=0), wanted, atol=1e-10)
        cast = [[f.astype(np.complex64) for f in e] for e in electric[k]]
        moment = _moments(actual, grid, cast)
        cast_errors.append(
            float(np.max(np.linalg.norm(moment / np.linalg.norm(moment, axis=0) - wanted, axis=0)))
        )
        np.testing.assert_allclose(moment / np.linalg.norm(moment, axis=0), wanted, atol=1e-5)
        np.testing.assert_allclose(
            np.diag(_power(owner, grid, electric[k], magnetic[k])), 1, atol=1e-12
        )
    record_property("polarization_direction_error", max(errors))
    record_property("single_cast_direction_error", max(cast_errors))
    record_property(
        "mixed_mode_residual",
        max(float(max(row["residual"])) for row in actual.degenerate_diagnostics[0]["anchors"]),
    )


def test_generic_subspace_tracking(circular_solvers):
    owner, _, electric, _ = bank(circular_solvers)
    assert owner.degenerate_diagnostics[0]["orientation"] == "automatic_transverse_axes"
    assert owner.mode_polarizations == {
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
    }
    assert min(owner.degenerate_diagnostics[0]["overlaps"]) > 0.9
    for index, record in enumerate(owner.degenerate_diagnostics[0]["anchors"]):
        np.testing.assert_allclose(record["power_gram"], np.eye(2), atol=1e-11)
        assert max(record["residual"]) < 1e-9
        moments = _moments(owner, owner._tracking_grid, electric[index])
        normalized = moments / np.linalg.norm(moments, axis=0)
        np.testing.assert_allclose(normalized, np.eye(2), atol=1e-10)


def test_intermediate_frequency_against_independent_solve(circular_solvers, record_property):
    solvers = []
    with _configured_solver_constants():
        for frequency in (9.5e9, 10.5e9):
            masks = _circular_pec_masks(44)
            solver = FDFD_2D_mode_solver(
                frequency=frequency,
                du=CIRCULAR_SPACING,
                dv=CIRCULAR_SPACING,
                mode_index=1,
                pec_u_mask=masks[0],
                pec_v_mask=masks[1],
                pec_w_mask=masks[2],
                guess=-float(_circular_theory(frequency)) ** 2,
                **_homogeneous_2d_arrays(44, 44)
            )
            solver.retain_tracking_operator = True
            solver.solve()
            solvers.append(solver)
    owner, grid, e, h = bank(solvers, references={1: "y", 2: "x"})
    _, _, reference_e, reference_h = bank([circular_solvers[1]], references={1: "y", 2: "x"})
    errors = []
    for mode in range(2):
        ie = [0.5 * (e[0][mode][axis] + e[1][mode][axis]) for axis in range(3)]
        ih = [0.5 * (h[0][mode][axis] + h[1][mode][axis]) for axis in range(3)]
        scale = np.sqrt(owner._modal_cross_power(ie, ih, grid).real)
        ie, ih = [f / scale for f in ie], [f / scale for f in ih]
        for actual, reference in ((ie, reference_e[0][mode]), (ih, reference_h[0][mode])):
            norm = np.sqrt(sum(np.linalg.norm(f) ** 2 for f in reference))
            errors.append(
                np.sqrt(sum(np.linalg.norm(a - r) ** 2 for a, r in zip(actual, reference))) / norm
            )
    record_property("intermediate_relative_field_error", float(max(errors)))
    # Piecewise linear profiles are approximate, including longitudinal H.
    assert max(errors) < 2e-3


def test_split_rejected(circular_solvers):
    solvers = deepcopy(circular_solvers)
    solvers[1].eigenvalues[1] += 1e-5
    with pytest.raises(ValueError, match="resolved eigenvalue splitting"):
        bank(solvers)


def test_equal_squared_eigenvalues_do_not_allow_opposite_phase_branches(circular_solvers):
    solvers = deepcopy(circular_solvers)
    solvers[0].operator_neff[1] *= -1
    with pytest.raises(ValueError, match="opposite propagation branches"):
        bank(solvers)


def test_hash_python_equivalence():
    tail, options = parse_port_options(
        ["auto", "n", "degenerate=1,2;3,4", "mode_polarizations=1:0,1,0;2:x"]
    )
    assert tail == ["auto", "n"]
    groups = normalize_groups(options["degenerate"], (1, 2, 3, 4))
    assert groups == ((1, 2), (3, 4))
    assert normalize_groups((1, 2), (1, 2)) == ((1, 2),)
    assert normalize_polarizations(options["mode_polarizations"], groups, 2) == {
        1: (0, 1, 0),
        2: (1, 0, 0),
    }


@pytest.mark.parametrize("groups", [(1,), ((1, 2), (2, 3)), (1, 4), 1, (1, True)])
def test_invalid_groups(groups):
    with pytest.raises(ValueError):
        normalize_groups(groups, (1, 2, 3))


@pytest.mark.parametrize(
    "directions",
    [
        {1: "y"},
        {2: "y"},
        "z",
        "bad",
        (0, 0, 0),
        (1j, 0, 0),
        (1, 0),
        {1: "x", 2: "x"},
        {1: "z", 2: "y"},
        {1: (0, 0, 0), 2: "y"},
        {1: (np.inf, 0, 0), 2: "y"},
        {1: (1j, 0, 0), 2: "y"},
        {3: "y"},
    ],
)
def test_invalid_polarizations(directions):
    with pytest.raises(ValueError):
        normalize_polarizations(directions, ((1, 2),), 2)


def test_default_polarization_preserves_automatic_assignment():
    assert normalize_polarizations(None, (), 2, unresolved=True) == {}
    assert normalize_polarizations(None, ((1, 2), (3, 4)), 2) == {}


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("first", ("axis", "diagonal", "name"))
def test_single_direction_completes_each_pair(axis, first):
    transverse = [i for i in range(3) if i != axis]
    direction = np.eye(3)[transverse[0]]
    if first == "diagonal":
        direction += np.eye(3)[transverse[1]]
    if first == "name":
        direction = "xyz"[transverse[0]]
    result = normalize_polarizations(direction, ((1, 2), (3, 4)), axis)
    assert result[1] == result[3]
    for one, two in ((1, 2), (3, 4)):
        np.testing.assert_allclose(result[two], np.cross(np.eye(3)[axis], result[one]))
        assert np.dot(result[one], result[two]) == pytest.approx(0, abs=1e-15)
        assert np.linalg.norm(result[two]) == pytest.approx(1)


@pytest.mark.parametrize("value, primary", [("y", (0, 1, 0)), ("1,1,0", np.array((1, 1, 0)) / np.sqrt(2))])
def test_single_direction_hash_and_auto_group_resolution(value, primary):
    _, options = parse_port_options([f"mode_polarizations={value}"])
    pending = normalize_polarizations(options["mode_polarizations"], (), 2, unresolved=True)
    np.testing.assert_allclose(pending, primary)
    resolved = normalize_polarizations(pending, ((1, 2), (3, 4)), 2)
    assert resolved[1] == resolved[3]
    assert resolved[2] == resolved[4]
    np.testing.assert_allclose(resolved[2], np.cross((0, 0, 1), primary))
    assert normalize_polarizations(resolved, ((1, 2), (3, 4)), 2) == resolved


@pytest.mark.parametrize("direction", ("+", "-"))
def test_single_direction_alignment_matches_explicit_pair(circular_solvers, direction):
    _, _, e, h = bank(deepcopy(circular_solvers), references="y", direction=direction)
    _, _, expected_e, expected_h = bank(
        deepcopy(circular_solvers), references={1: "y", 2: (-1, 0, 0)}, direction=direction
    )
    for actual, expected in ((e, expected_e), (h, expected_h)):
        for anchor, reference in zip(actual, expected):
            for mode, reference_mode in zip(anchor, reference):
                for component, reference_component in zip(mode, reference_mode):
                    np.testing.assert_allclose(component, reference_component, atol=1e-12)


@pytest.mark.parametrize("groups, invariant", [((), None), (((1, 2, 3),), None), (((1, 2),), 0)])
def test_shared_direction_requires_3d_pairs(groups, invariant):
    with pytest.raises(ValueError):
        normalize_polarizations("y", groups, 2, invariant)


def test_nonorthogonal_references_retain_power_gram(circular_solvers):
    owner, grid, e, h = bank(circular_solvers, references={1: "x", 2: (1, 1, 0)})
    for k in range(len(e)):
        power = _power(owner, grid, e[k], h[k])
        np.testing.assert_allclose(np.diag(power), 1, atol=1e-12)
        np.testing.assert_allclose(power[0, 1], 1 / np.sqrt(2), atol=1e-10)


def test_basis_condition_is_not_squared_by_power_validation(circular_solvers):
    changed = deepcopy(circular_solvers)
    rotation = np.array([[1, 1j], [1j, 1]]) @ np.diag([1e4, 1.0])
    for solver in changed:
        for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw"):
            setattr(solver, name, getattr(solver, name) @ rotation)
    _, _, e, h = bank(changed, references={1: "y", 2: "x"})
    _, _, re, rh = bank(circular_solvers, references={1: "y", 2: "x"})
    for family, reference in ((e, re), (h, rh)):
        for k in range(len(changed)):
            for mode in range(2):
                norm = np.sqrt(sum(np.linalg.norm(f) ** 2 for f in reference[k][mode]))
                error = np.sqrt(
                    sum(
                        np.linalg.norm(a - b) ** 2
                        for a, b in zip(family[k][mode], reference[k][mode])
                    )
                )
                assert error / norm < 1e-10


@pytest.mark.parametrize("kind", ("rank", "zero_moment", "residual", "missing"))
def test_invalid_solved_group(circular_solvers, kind):
    solvers = deepcopy(circular_solvers)
    solver = solvers[0]
    if kind == "rank":
        for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw"):
            getattr(solver, name)[..., 1] = getattr(solver, name)[..., 0]
        match = "rank loss"
    elif kind == "zero_moment":
        for name, component in (("Eu", "eu"), ("Ev", "ev")):
            fields = getattr(solver, name)
            for i in range(2):
                fields[..., i] -= np.mean(
                    EigenmodeSource._average_to_transverse_cells(fields[..., i], component)
                )
        match = "integrated transverse E"
    elif kind == "residual":
        solver.mode_tracking_operator *= 1.0001
        match = "mixed-mode eigen-residual"
    else:
        solver.eigenvalues = np.append(solver.eigenvalues, solver.eigenvalues[0])
        match = "partner outside"
    with pytest.raises(ValueError, match=match):
        bank(solvers, references={1: "x", 2: "y"})


@pytest.mark.parametrize("tracking", ("legacy", "auto"))
@pytest.mark.parametrize("hash_value, python_value", [
    ("y", "y"),
    ("1,1,0", (1, 1, 0)),
    ("y;x", ("y", "x")),
    ("0,1,0;x", ((0, 1, 0), "x")),
    ("1,1,0;-1,1,0", ((1, 1, 0), (-1, 1, 0))),
    ("1:y;2:1,0,0", {1: "y", 2: "x"}),
])
def test_python_hash_build_equivalence(monkeypatch, tracking, hash_value, python_value):
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.hash_cmds_multiuse import process_multicmds
    from gprMax.user_objects.cmds_multiuse import EigenmodePort

    monkeypatch.setattr(config, "sim_config", SimpleNamespace(general={"solver": "cpu"}, mpi=False))
    monkeypatch.setattr(config, "get_model_config", lambda: SimpleNamespace(mode="3D"))
    commands = defaultdict(lambda: None)
    commands["#eigenmode_band"] = ["band 6e9 14e9 3"]
    commands["#eigenmode_excitation"] = ["1 1 auto n"]
    commands["#eigenmode_port"] = [
        f"1 0 0 0.02 0.05 0.05 0.02 + 1,2 auto n degenerate=1,2 tracking={tracking} mode_polarizations={hash_value}"
    ]
    parsed = next(obj for obj in process_multicmds(commands) if isinstance(obj, EigenmodePort))
    explicit = EigenmodePort(
        port=1,
        p1=(0, 0, 0.02),
        p2=(0.05, 0.05, 0.02),
        direction="+",
        modes=(1, 2),
        anchors="auto",
        plot_fields=False,
        degenerate=(1, 2),
        mode_polarizations=python_value,
        tracking=tracking,
    )
    grids = [FDTDGrid(), FDTDGrid()]
    for obj, grid in zip((parsed, explicit), grids):
        grid.eigenmodeband = SimpleNamespace()
        if tracking == "auto" and isinstance(python_value, dict):
            with pytest.raises(ValueError, match="mappings require tracking='legacy'"):
                obj.build(grid)
        else:
            obj.build(grid)
    if tracking == "auto" and isinstance(python_value, dict):
        return
    assert grids[0].eigenmodeportdefs[1] == grids[1].eigenmodeportdefs[1]


def test_physical_selection_rejects_reduced_mode():
    with pytest.raises(ValueError, match="3D"):
        normalize_polarizations({1: "x", 2: "y"}, ((1, 2),), 2, 0)


def test_group_cutoff_masks(circular_solvers):
    active = np.array([[False, False], [True, True], [True, True]])
    owner, _, _, _ = bank(circular_solvers, propagating=active)
    np.testing.assert_array_equal(owner._degenerate_masks[0], active[:, 0])
    np.testing.assert_array_equal(owner._degenerate_masks[1], active[:, 1])
    active[0, 0] = True
    with pytest.raises(ValueError, match="inconsistent propagation"):
        bank(circular_solvers, propagating=active)
    active[:] = [[True, True], [False, False], [True, True]]
    with pytest.raises(ValueError, match="contiguous"):
        bank(circular_solvers, propagating=active)


@pytest.mark.parametrize("automatic", (True, False))
def test_group_guard_trimming_never_falls_back_by_member(monkeypatch, circular_solvers, automatic):
    import gprMax.eigenmode_tracking as tracking

    original = tracking._frame
    calls = []

    def frame(*args):
        result = original(*args)
        calls.append(1)
        zeros = np.zeros_like(result)
        return np.vstack((result, zeros)) if len(calls) == 1 else np.vstack((zeros, result))

    monkeypatch.setattr(tracking, "_frame", frame)
    if automatic:
        owner, _, _, _ = bank(circular_solvers, automatic=True)
        np.testing.assert_array_equal(owner._degenerate_masks[0], [False, True, True])
        np.testing.assert_array_equal(owner._degenerate_masks[1], [False, True, True])
        assert owner._degenerate_guard_trimmed[0]
    else:
        with pytest.raises(ValueError, match="subspace overlap"):
            bank(circular_solvers)


def test_canonical_subspace_basis_ignores_numerical_pivot_ties():
    frame = np.array(((1, 0), (0, 1), (1, 0), (0, 1)), dtype=complex)
    perturbed = frame.copy()
    perturbed[1, 1] += 2e-13
    reference = frame @ _canonical_subspace_transform(frame)
    actual = perturbed @ _canonical_subspace_transform(perturbed)
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)


def test_higher_order_zero_moment_and_multiple_groups():
    with _configured_solver_constants():
        masks = _circular_pec_masks(44)
        solver = FDFD_2D_mode_solver(
            frequency=14e9,
            du=0.001,
            dv=0.001,
            mode_index=7,
            pec_u_mask=masks[0],
            pec_v_mask=masks[1],
            pec_w_mask=masks[2],
            **_homogeneous_2d_arrays(44, 44)
        )
        solver.retain_tracking_operator = True
        solver.solve()
    owner, _, _, _ = bank([solver], groups=((1, 2), (7, 8)), references={1: "y", 2: "x"})
    assert len(owner.degenerate_diagnostics) == 2
    assert owner.degenerate_diagnostics[0]["physical"]
    assert not owner.degenerate_diagnostics[1]["physical"]
    assert owner.degenerate_diagnostics[1]["orientation"] == "deterministic_subspace"

    reference, _, reference_e, reference_h = bank([solver], groups=((7, 8),))
    changed = deepcopy(solver)
    rotation = np.asarray(((1 + 1j, 0.3), (-0.2j, 1 - 0.4j)))
    for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw"):
        fields = getattr(changed, name)
        fields[..., 6:8] = fields[..., 6:8] @ rotation
    actual, _, electric, magnetic = bank([changed], groups=((7, 8),))
    assert reference.degenerate_diagnostics[0]["orientation"] == "deterministic_subspace"
    assert actual.degenerate_diagnostics[0]["orientation"] == "deterministic_subspace"
    for family, expected in ((electric, reference_e), (magnetic, reference_h)):
        for mode in range(2):
            for component in range(3):
                np.testing.assert_allclose(
                    family[0][mode][component],
                    expected[0][mode][component],
                    atol=1e-10,
                    rtol=1e-10,
                )
    with pytest.raises(ValueError, match="integrated transverse E"):
        bank([solver], groups=((7, 8),), references={7: "y", 8: "x"})


def test_elliptical_cross_section_resolved_split():
    with _configured_solver_constants():
        masks = _circular_pec_masks(44)
        # A circular voxel mask on unequal physical spacings is an ellipse.
        solver = FDFD_2D_mode_solver(
            frequency=10e9,
            du=0.001,
            dv=0.0012,
            mode_index=1,
            pec_u_mask=masks[0],
            pec_v_mask=masks[1],
            pec_w_mask=masks[2],
            **_homogeneous_2d_arrays(44, 44)
        )
        solver.retain_tracking_operator = True
        solver.solve()
    with pytest.raises(ValueError, match="resolved eigenvalue splitting"):
        bank([solver], references={1: "y", 2: "x"})


def test_actual_below_cutoff_group_is_excluded(circular_solvers):
    with _configured_solver_constants():
        masks = _circular_pec_masks(44)
        solver = FDFD_2D_mode_solver(
            frequency=3e9,
            du=0.001,
            dv=0.001,
            mode_index=1,
            pec_u_mask=masks[0],
            pec_v_mask=masks[1],
            pec_w_mask=masks[2],
            **_homogeneous_2d_arrays(44, 44)
        )
        solver.retain_tracking_operator = True
        solver.solve()
    solvers = [solver, *circular_solvers[:2]]
    active = np.asarray([s.power_valid[:2] for s in solvers])
    assert not np.any(active[0])
    owner, _, _, _ = bank(solvers, propagating=active, references={1: "y", 2: "x"})
    np.testing.assert_array_equal(owner._degenerate_masks[0], [False, True, True])


def test_plots_exclude_unaligned_cutoff_anchors(circular_solvers, monkeypatch, tmp_path):
    import gprMax.sources as sources

    active = np.array([[False, False], [True, True], [True, True]])
    owner, _, _, _ = bank(circular_solvers, propagating=active, references={1: "y", 2: "x"})
    owner.port_anchor_frequencies = tuple(s.frequency for s in circular_solvers)
    owner.port_mode_solvers = circular_solvers
    owner.port_anchor_mode_reference_valid = active
    owner.mpi_coordinator = True
    owner._should_plot_eigenmode_fields = lambda: True
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(input_file_path=tmp_path / "test.in"))
    calls = []
    monkeypatch.setattr(
        sources, "plot_eigenmode_port_fields", lambda **kwargs: calls.append(kwargs)
    )
    owner._plot_eigenmode_fields()
    assert len(calls) == 2
    assert all(call["frequencies"] == (10e9, 14e9) for call in calls)
    assert all(call["solvers"][0].mode_polarizations == owner.mode_polarizations for call in calls)


@pytest.mark.parametrize("directions", [("y", "x"), ((1, 1, 0), (-1, 1, 0)), ("x", (1, 1, 0))])
def test_two_shared_directions_apply_to_every_pair(directions):
    pending = normalize_polarizations(directions, (), 2, unresolved=True)
    resolved = normalize_polarizations(pending, ((1, 2), (4, 5)), 2)
    expected = normalize_polarizations({1: directions[0], 2: directions[1]}, ((1, 2),), 2)
    assert resolved == {1: expected[1], 2: expected[2], 4: expected[1], 5: expected[2]}


@pytest.mark.parametrize("directions", [("x", "x"), ("x", "z"), ("x", (0, 0, 0)), ("x", (1j, 1, 0)), ("x", "y", "z")])
def test_invalid_two_direction_input_rejected_before_solving(directions):
    with pytest.raises(ValueError):
        normalize_polarizations(directions, (), 2, unresolved=True)


@pytest.mark.parametrize("direction", ("+", "-"))
@pytest.mark.parametrize("references", [("y", "x"), ((1, 1, 0), (-1, 1, 0)), ("x", (1, 1, 0))])
def test_two_directions_align_like_exact_modes(circular_solvers, direction, references):
    _, _, e, h = bank(deepcopy(circular_solvers), references=references, direction=direction)
    _, _, expected_e, expected_h = bank(
        deepcopy(circular_solvers), references=dict(zip((1, 2), references)), direction=direction
    )
    for actual, expected in ((e, expected_e), (h, expected_h)):
        for anchor, reference in zip(actual, expected):
            for mode, reference_mode in zip(anchor, reference):
                for component, reference_component in zip(mode, reference_mode):
                    np.testing.assert_allclose(component, reference_component, atol=1e-12)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("scale", (1e-100, 1.0, 1e100))
@pytest.mark.parametrize("sign", (-1, 1))
def test_shared_pair_rejects_nearly_dependent_directions(axis, scale, sign):
    transverse = [i for i in range(3) if i != axis]
    first = np.eye(3)[transverse[0]]
    second = scale * (sign * first + 1e-10 * np.eye(3)[transverse[1]])
    with pytest.raises(ValueError, match="linearly dependent or ill-conditioned"):
        normalize_polarizations((first, second), (), axis, unresolved=True)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("member", (0, 1))
def test_shared_pair_rejects_out_of_plane_component(axis, member):
    transverse = [i for i in range(3) if i != axis]
    directions = [np.eye(3)[i] for i in transverse]
    directions[member][axis] = 1e-6
    with pytest.raises(ValueError, match="transverse to the port normal"):
        normalize_polarizations(directions, (), axis, unresolved=True)


@pytest.mark.parametrize("mapping", (False, True))
@pytest.mark.parametrize("unresolved", (False, True))
@pytest.mark.parametrize("angle, accepted", [(10, False), (12, True), (45, True), (90, True), (168, True), (170, False)])
def test_user_directions_must_be_clearly_distinct(mapping, unresolved, angle, accepted):
    radians = np.deg2rad(angle)
    directions = ("x", (np.cos(radians), np.sin(radians), 0))
    value = dict(zip((1, 2), directions)) if mapping else directions
    if accepted:
        normalize_polarizations(value, ((1, 2),), 2, unresolved=unresolved)
    else:
        with pytest.raises(ValueError, match="clearly distinct.*at most 10"):
            normalize_polarizations(value, ((1, 2),), 2, unresolved=unresolved)
