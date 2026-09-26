"""Degenerate modal groups and optional automatic mode tracking.

The conventional assignment and confinement algorithms are adapted from
``fdfd_mode_tracking`` revision 8c617a2.
gprMax-specific field conventions, injection eligibility, and persistence are
implemented here under gprMax's GPL license.
"""

import logging
import numbers
from copy import copy

import numpy as np
from scipy.optimize import linear_sum_assignment

_CONDITION_LIMIT = 1e8
_DEGENERACY_TOLERANCE = 1e-8
_RESIDUAL_TOLERANCE = 1e-9
_SUBSPACE_PIVOT_TOLERANCE = 1e-12
logger = logging.getLogger(__name__)


def _solver_mode_fields(solver, mode):
    """Return solver-local physical E/H components for one candidate."""
    if hasattr(solver, "Eu"):
        electric = (solver.Eu[..., mode], solver.Ev[..., mode], solver.Ew[..., mode])
        magnetic = (solver.Hu[..., mode], solver.Hv[..., mode], solver.Hw[..., mode])
    else:
        electric = (solver.Et[:, mode], solver.Ea[:, mode], solver.Ew[:, mode])
        magnetic = (solver.Ht[:, mode], solver.Ha[:, mode], solver.Hw[:, mode])
    return electric, magnetic


def _tracking_vectors(solver, impedance):
    """Return common E/H-normalized candidate columns."""
    columns = []
    for mode in range(solver.num_modes):
        electric, magnetic = _solver_mode_fields(solver, mode)
        column = np.concatenate(
            [*(np.ravel(value) for value in electric), *(impedance * np.ravel(value) for value in magnetic)]
        ).astype(np.complex128, copy=False)
        norm = np.linalg.norm(column)
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            column = np.full(column.shape, np.nan + 0j)
        else:
            column = column / norm
        columns.append(column)
    return np.column_stack(columns)


def solver_mode_residual(solver, mode):
    """Relative residual of the discrete eigenproblem used for field reconstruction."""
    if hasattr(solver, "mode_tracking_operator"):
        operator = solver.mode_tracking_operator
        vector = solver.eigenvectors[solver.free_euv_mask, mode]
    elif hasattr(solver, "operator") and hasattr(solver, "free_scalar_mask"):
        free = solver.free_scalar_mask
        operator = solver.operator[free, :][:, free]
        field = solver.Ea if solver.polarization == "TM" else solver.Ha
        vector = field[free, mode]
    else:
        return np.inf
    target = np.asarray(solver.eigenvalues)[mode] * vector
    applied = operator @ vector
    denominator = np.linalg.norm(applied) + np.linalg.norm(target)
    eigenpair = float(np.linalg.norm(applied - target) / max(float(denominator), 1e-300))
    field_values = getattr(solver, "field_residuals", None)
    field = eigenpair if field_values is None else np.asarray(field_values)[mode]
    return max(eigenpair, float(field))


def _clusters(values, gap):
    """Connected components of the relative eigenvalue-gap graph."""
    remaining = set(range(len(values)))
    groups = []
    while remaining:
        group = {min(remaining)}
        while True:
            extra = {
                j
                for j in remaining - group
                if any(
                    abs(values[j] - values[i])
                    <= gap * max(1.0, abs(values[j]), abs(values[i]))
                    for i in group
                )
            }
            if not extra:
                break
            group.update(extra)
        remaining -= group
        groups.append(tuple(sorted(group)))
    return tuple(groups)


def _assign_step(old_vectors, new_vectors, old_values, new_values, config, prediction):
    """Assign single modes and equal-rank degenerate spans with abstention."""
    count = old_vectors.shape[1]
    result = np.full(count, -1, dtype=int)
    scores = np.zeros(count, dtype=float)
    occupied = set()

    old_groups = [group for group in _clusters(old_values, config.cluster_gap) if len(group) > 1]
    new_groups = [group for group in _clusters(new_values, config.cluster_gap) if len(group) > 1]
    for old_group in old_groups:
        # Eigensolvers need not return orthogonal vectors within a repeated
        # eigenvalue. Principal angles compare spans, so normalize the whole
        # basis before scoring; column normalization alone depends on the
        # solver's arbitrary choice of basis and can create false identity gaps.
        try:
            old_span = _orthonormal(old_vectors[:, old_group])
        except (ValueError, np.linalg.LinAlgError):
            continue
        best = None
        for new_group in new_groups:
            if len(new_group) != len(old_group) or occupied.intersection(new_group):
                continue
            try:
                new_span = _orthonormal(new_vectors[:, new_group])
            except (ValueError, np.linalg.LinAlgError):
                continue
            singular = np.linalg.svd(
                old_span.conj().T @ new_span, compute_uv=False
            )
            score = float(np.clip(np.min(singular) ** 2, 0.0, 1.0))
            drift = abs(np.mean(new_values[list(new_group)]) - np.mean(prediction[list(old_group)]))
            drift /= max(1.0, abs(np.mean(old_values[list(old_group)])))
            cost = 1.0 - score + 0.1 * min(float(drift) ** 2, 4.0)
            candidate = (cost, -score, new_group)
            if score >= config.tracking_overlap and (best is None or candidate < best):
                best = candidate
        if best is None:
            continue
        new_group = best[2]
        # Member labels inside an exactly degenerate space are a gauge choice;
        # align_groups performs the physical transport after this reservation.
        for old, new in zip(old_group, new_group):
            result[old] = new
            scores[old] = -best[1]
            occupied.add(new)

    old_remaining = np.flatnonzero(result < 0)
    new_remaining = np.asarray([i for i in range(len(new_values)) if i not in occupied], dtype=int)
    if not len(old_remaining) or not len(new_remaining):
        return result, scores
    overlap = np.clip(
        abs(old_vectors[:, old_remaining].conj().T @ new_vectors[:, new_remaining]) ** 2,
        0.0,
        1.0,
    )
    drift = abs(new_values[new_remaining][None, :] - prediction[old_remaining, None])
    drift /= np.maximum(1.0, abs(old_values[old_remaining]))[:, None]
    cost = 1.0 - overlap + 0.1 * np.minimum(drift**2, 4.0)
    cost[overlap < config.tracking_overlap] = 1e6
    augmented = np.column_stack((cost, np.full((len(old_remaining), len(old_remaining)), config.unmatched_cost)))
    rows, columns = linear_sum_assignment(augmented)
    best_total = float(augmented[rows, columns].sum())
    for row, column in zip(rows, columns):
        if column >= len(new_remaining) or cost[row, column] >= config.unmatched_cost:
            continue
        alternative = augmented.copy()
        alternative[row, column] = 1e6
        alt_rows, alt_columns = linear_sum_assignment(alternative)
        margin = float(alternative[alt_rows, alt_columns].sum() - best_total)
        if margin < config.assignment_margin:
            continue
        old = old_remaining[row]
        result[old] = new_remaining[column]
        scores[old] = overlap[row, column]
    return result, scores


def _reorder_solver_modes(solver, public_candidates):
    public_candidates = tuple(int(value) for value in public_candidates)
    order = public_candidates + tuple(i for i in range(solver.num_modes) if i not in public_candidates)
    vector_arrays = (
        "eigenvalues", "operator_neff", "beta", "complex_neff", "real_neff",
        "spatially_resolved", "raw_powers", "forward_power_metrics", "power_valid", "powers",
        "eigenpair_residuals", "field_residuals",
    )
    field_arrays = (
        "eigenvectors", "Eu", "Ev", "Ew", "Hu", "Hv", "Hw",
        "Et", "Ea", "Ht", "Ha",
    )
    for name in vector_arrays:
        value = getattr(solver, name, None)
        if value is not None and np.ndim(value) == 1 and len(value) == solver.num_modes:
            setattr(solver, name, np.asarray(value)[list(order)].copy())
    for name in field_arrays:
        value = getattr(solver, name, None)
        if value is not None and np.ndim(value) >= 2 and value.shape[-1] == solver.num_modes:
            setattr(solver, name, np.asarray(value)[..., list(order)].copy())
    if hasattr(solver, "_set_modal_fields"):
        solver._set_modal_fields()


def track_solver_bank(owner, frequencies, solvers, mode_count):
    """Track and reorder a solver bank so public labels retain their identity."""
    config = owner.tracking_config
    frequencies = np.asarray(frequencies, dtype=float)
    solvers = tuple(solvers)
    centre_frequency = owner.fallback_frequency
    if centre_frequency is None:
        centre_frequency = 0.5 * (frequencies[0] + frequencies[-1])
    centre = min(range(len(frequencies)), key=lambda i: (abs(frequencies[i] - centre_frequency), frequencies[i]))
    impedance = float(owner._tracking_impedance)
    centre_values = np.asarray(solvers[centre].eigenvalues)
    seed_indices = set(range(mode_count))
    if not owner.degenerate:
        for group in _clusters(centre_values, _DEGENERACY_TOLERANCE):
            if seed_indices.intersection(group):
                seed_indices.update(group)
    seed_indices = tuple(sorted(seed_indices))
    tracking_count = len(seed_indices)
    owner._auto_tracking_mode_count = tracking_count
    vectors = [_tracking_vectors(solver, impedance) for solver in solvers]
    residuals = [
        np.asarray([solver_mode_residual(solver, mode) for mode in range(solver.num_modes)])
        for solver in solvers
    ]
    eigenpair_residuals = [
        np.asarray(getattr(solver, "eigenpair_residuals", residual), dtype=float)
        for solver, residual in zip(solvers, residuals)
    ]
    field_residuals = [
        np.asarray(getattr(solver, "field_residuals", residual), dtype=float)
        for solver, residual in zip(solvers, residuals)
    ]
    valid = [
        np.isfinite(record) & (record <= config.residual_tolerance)
        & np.all(np.isfinite(vector), axis=0)
        for record, vector in zip(residuals, vectors)
    ]
    mapping = np.full((len(solvers), tracking_count), -1, dtype=int)
    mapping[centre] = seed_indices
    overlaps = np.full((len(solvers), tracking_count), np.nan, dtype=float)
    overlaps[centre] = 1.0

    for path in (range(centre + 1, len(solvers)), range(centre - 1, -1, -1)):
        previous = centre
        before_previous = None
        for current in path:
            old_indices = mapping[previous]
            old_vectors = vectors[previous][:, old_indices]
            old_values = np.asarray(solvers[previous].eigenvalues)[old_indices]
            prediction = old_values.copy()
            if before_previous is not None:
                earlier_indices = mapping[before_previous]
                earlier_values = np.asarray(solvers[before_previous].eigenvalues)[earlier_indices]
                delta = frequencies[previous] - frequencies[before_previous]
                if delta:
                    prediction += (frequencies[current] - frequencies[previous]) / delta * (
                        old_values - earlier_values
                    )
            new_values = np.asarray(solvers[current].eigenvalues)
            assigned, score = _assign_step(
                old_vectors, vectors[current], old_values, new_values, config, prediction
            )
            assigned[(assigned >= 0) & ~valid[current][np.maximum(assigned, 0)]] = -1
            if np.any(assigned < 0):
                missing = tuple(int(i + 1) for i in np.flatnonzero(assigned < 0))
                from gprMax.sources import EigenmodeAnchorMismatchError

                raise EigenmodeAnchorMismatchError(
                    f"Automatic mode tracking for eigenmode port {owner.port_index} is unresolved "
                    f"between {frequencies[previous]:g} and {frequencies[current]:g} Hz for "
                    f"public mode(s) {missing}.",
                    first_frequency=float(min(frequencies[previous], frequencies[current])),
                    second_frequency=float(max(frequencies[previous], frequencies[current])),
                    mode_index=missing,
                    context=f"Eigenmode port {owner.port_index} automatic tracking",
                )
            mapping[current] = assigned
            overlaps[current] = score
            before_previous, previous = previous, current

    original_mapping = mapping.copy()
    public_residuals = np.empty_like(overlaps)
    public_eigenpair_residuals = np.empty_like(overlaps)
    public_field_residuals = np.empty_like(overlaps)
    public_valid = np.empty_like(mapping, dtype=bool)
    for index, solver in enumerate(solvers):
        public_residuals[index] = residuals[index][mapping[index]]
        public_eigenpair_residuals[index] = eigenpair_residuals[index][mapping[index]]
        public_field_residuals[index] = field_residuals[index][mapping[index]]
        public_valid[index] = valid[index][mapping[index]]
        _reorder_solver_modes(solver, mapping[index])

    automatic_groups = []
    centre_values = np.asarray(solvers[centre].eigenvalues)[:tracking_count]
    for group in _clusters(centre_values, _DEGENERACY_TOLERANCE):
        if len(group) < 2:
            continue
        labels = tuple(item + 1 for item in group)
        if all(
            np.max(abs(np.asarray(solver.eigenvalues)[list(group), None] - np.asarray(solver.eigenvalues)[list(group)]))
            <= _DEGENERACY_TOLERANCE
            * max(1.0, float(np.max(abs(np.asarray(solver.eigenvalues)[list(group)]))))
            for solver in solvers
        ):
            automatic_groups.append(labels)
    if not owner.degenerate:
        owner.degenerate = tuple(automatic_groups)
    if owner.mode_polarizations:
        owner.mode_polarizations = normalize_polarizations(
            owner.mode_polarizations,
            owner.degenerate,
            owner.normal_axis,
            owner.invariant_axis,
        )
    owner.tracking_diagnostics = {
        "schema_version": 1,
        "source_revision": "8c617a2",
        "reference_anchor": int(centre),
        "candidate_indices": original_mapping,
        "overlaps": overlaps,
        "residuals": public_residuals,
        "eigenpair_residuals": public_eigenpair_residuals,
        "field_residuals": public_field_residuals,
        "numerical_valid": public_valid,
        "automatic_degenerate_groups": tuple(automatic_groups),
        "unresolved_intervals": (),
    }
    owner._auto_numerical_valid = public_valid
    return mapping


def _refine_axis(values, axis, cells, factor):
    if values.shape[axis] == cells:
        return np.repeat(values, factor, axis=axis)
    if values.shape[axis] == cells + 1:
        indices = np.rint(np.arange(cells * factor + 1) / factor).astype(int)
        return np.take(values, indices, axis=axis)
    raise ValueError("Unsupported Yee array shape for mesh verification.")


def _refine_array(values, cells, factor=2):
    result = np.asarray(values)
    for axis, count in enumerate(cells):
        result = _refine_axis(result, axis, count, factor)
    return result


def _refined_solver(solver, factor=2):
    """Re-solve a nearest-voxel refinement of the represented port geometry."""
    if getattr(solver, "surface_boundary_rows", ()):
        raise ValueError("surface_impedance_refinement_unavailable")
    if hasattr(solver, "Eu"):
        from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver

        cells = (solver.Nu, solver.Nv)
        names = (
            "eps_r_uu", "eps_r_vv", "eps_r_ww", "mu_r_uu", "mu_r_vv", "mu_r_ww",
            "pec_u_mask", "pec_v_mask", "pec_w_mask", "pmc_u_mask", "pmc_v_mask", "pmc_w_mask",
        )
        values = {name: _refine_array(getattr(solver, name), cells, factor) for name in names}
        refined = FDFD_2D_mode_solver(
            frequency=solver.frequency,
            du=solver.du / factor,
            dv=solver.dv / factor,
            mode_index=solver.num_modes - 1,
            guess=solver.guess,
            fdtd_dt=solver.fdtd_dt,
            propagation_spacing=solver.propagation_spacing,
            **values,
        )
        refined.retain_tracking_operator = True
    else:
        from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver

        names = (
            "eps_r_t", "eps_r_a", "eps_r_w", "mu_r_t", "mu_r_a", "mu_r_w",
            "pec_t_mask", "pec_a_mask", "pec_w_mask", "pmc_t_mask", "pmc_a_mask", "pmc_w_mask",
        )
        values = {name: _refine_array(getattr(solver, name), (solver.N,), factor) for name in names}
        refined = FDFD_1D_mode_solver(
            frequency=solver.frequency,
            dt=solver.dt / factor,
            mode_index=solver.num_modes - 1,
            polarization=solver.polarization,
            guess=solver.guess,
            fdtd_dt=solver.fdtd_dt,
            propagation_spacing=solver.propagation_spacing,
            **values,
        )
    refined.calculate_diagnostics = True
    refined.solve()
    return refined


def _cell_fields(solver, mode):
    electric, magnetic = _solver_mode_fields(solver, mode)
    if hasattr(solver, "Eu"):
        electric = (
            solver._field_to_cells(electric[0], "u"),
            solver._field_to_cells(electric[1], "v"),
            solver._field_to_cells(electric[2], "w_e"),
        )
        magnetic = (
            solver._field_to_cells(magnetic[0], "hu"),
            solver._field_to_cells(magnetic[1], "hv"),
            solver._field_to_cells(magnetic[2], "w_h"),
        )
    else:
        def cells(values):
            return 0.5 * (values[:-1] + values[1:]) if values.size == solver.N + 1 else values

        electric = tuple(cells(value) for value in electric)
        magnetic = tuple(cells(value) for value in magnetic)
    return electric, magnetic


def _comparison_vector(solver, mode, *, crop=None, reduce_factor=None):
    electric, magnetic = _cell_fields(solver, mode)
    arrays = []
    for family, scale in ((electric, 1.0), (magnetic, float(solver.eta0))):
        for values in family:
            values = np.asarray(values)
            if crop is not None:
                values = values[tuple(slice(lo, hi) for lo, hi in crop)]
            if reduce_factor is not None:
                factor = int(reduce_factor)
                if values.ndim == 1:
                    values = values.reshape(-1, factor).mean(axis=1)
                else:
                    values = values.reshape(
                        values.shape[0] // factor,
                        factor,
                        values.shape[1] // factor,
                        factor,
                    ).mean(axis=(1, 3))
            arrays.append(scale * values.ravel())
    vector = np.concatenate(arrays).astype(np.complex128, copy=False)
    norm = np.linalg.norm(vector)
    return vector / norm if np.isfinite(norm) and norm > np.finfo(float).tiny else vector * np.nan


def _edge_fraction(solver, mode):
    electric, magnetic = _cell_fields(solver, mode)
    energy = sum(abs(value) ** 2 for value in electric)
    energy += solver.eta0**2 * sum(abs(value) ** 2 for value in magnetic)
    edge = np.zeros(energy.shape, dtype=bool)
    for axis, count in enumerate(energy.shape):
        width = max(1, int(np.ceil(0.1 * count)))
        index = [slice(None)] * energy.ndim
        index[axis] = slice(0, width)
        edge[tuple(index)] = True
        index[axis] = slice(count - width, count)
        edge[tuple(index)] = True
    total = float(np.sum(energy))
    return float(np.sum(energy[edge]) / total) if np.isfinite(total) and total > 0 else 1.0


def _physical_enclosure(solver):
    """Conservatively recognize physical PEC/PMC walls on every outer edge."""
    if hasattr(solver, "Eu"):
        pec = (
            np.all(solver.pec_v_mask[0, :]) and np.all(solver.pec_w_mask[0, :]),
            np.all(solver.pec_v_mask[-1, :]) and np.all(solver.pec_w_mask[-1, :]),
            np.all(solver.pec_u_mask[:, 0]) and np.all(solver.pec_w_mask[:, 0]),
            np.all(solver.pec_u_mask[:, -1]) and np.all(solver.pec_w_mask[:, -1]),
        )
        pmc = (
            np.all(solver.pmc_v_mask[0, :]) and np.all(solver.pmc_w_mask[0, :]),
            np.all(solver.pmc_v_mask[-1, :]) and np.all(solver.pmc_w_mask[-1, :]),
            np.all(solver.pmc_u_mask[:, 0]) and np.all(solver.pmc_w_mask[:, 0]),
            np.all(solver.pmc_u_mask[:, -1]) and np.all(solver.pmc_w_mask[:, -1]),
        )
        return bool(all(a or b for a, b in zip(pec, pmc)))
    if solver.polarization == "TM":
        return bool(solver.pec_a_mask[0] and solver.pec_a_mask[-1])
    return bool(solver.pmc_a_mask[0] and solver.pmc_a_mask[-1])


def _verification_check(
    base,
    mode,
    variant,
    config,
    *,
    crop=None,
    reduce_factor=None,
    kind,
    base_group=None,
):
    base_group = tuple(base_group or (mode,))
    references = np.column_stack(
        [_comparison_vector(base, member) for member in base_group]
    )
    candidates = np.column_stack(
        [
            _comparison_vector(variant, candidate, crop=crop, reduce_factor=reduce_factor)
            for candidate in range(variant.num_modes)
        ]
    )
    if len(base_group) == 1:
        overlap = np.clip(abs(references[:, 0].conj() @ candidates) ** 2, 0.0, 1.0)
        selected = (int(np.nanargmax(overlap)),)
        score = float(overlap[selected[0]])
    else:
        reference_span = np.linalg.qr(references)[0]
        choices = [
            group
            for group in _clusters(np.asarray(variant.eigenvalues), config.cluster_gap)
            if len(group) == len(base_group)
        ]
        if not choices:
            raise ValueError("matching_degenerate_verification_span_unavailable")
        scored = []
        for group in choices:
            candidate_span = np.linalg.qr(candidates[:, group])[0]
            singular = np.linalg.svd(
                reference_span.conj().T @ candidate_span,
                compute_uv=False,
            )
            scored.append((float(np.min(singular) ** 2), group))
        score, selected = max(scored, key=lambda item: item[0])
    residual = max(solver_mode_residual(variant, candidate) for candidate in selected)
    base_beta = np.mean(np.asarray(base.beta)[list(base_group)])
    variant_beta = np.mean(np.asarray(variant.beta)[list(selected)])
    beta_drift = float(abs(variant_beta - base_beta) / max(abs(base_beta), base.k0))
    passed = bool(
        np.isfinite(residual)
        and residual <= config.residual_tolerance
        and score >= config.verification_overlap
        and beta_drift <= config.beta_drift_tolerance
    )
    edge = max(_edge_fraction(variant, candidate) for candidate in selected)
    if kind.startswith("padding"):
        passed &= edge <= config.edge_fraction_max
    return {
        "kind": kind,
        "candidate": selected[0] if len(selected) == 1 else tuple(selected),
        "overlap": score,
        "beta_drift": beta_drift,
        "residual": residual,
        "edge_fraction": edge,
        "passed": passed,
    }


def assess_anchor_quality(owner, grid, frequencies, solvers, mode_indices):
    """Classify confinement separately from numerical/injection eligibility."""
    records = []
    config = owner.tracking_config
    solve_count = int(getattr(owner, "_tracking_extra_solves", 0))
    for frequency_index, (frequency, solver) in enumerate(zip(frequencies, solvers)):
        variants = {}
        failures = []
        enclosed = _physical_enclosure(solver)
        if owner.verification == "full":
            if solve_count < config.max_solves:
                try:
                    variants["mesh"] = (_refined_solver(solver), None)
                    solve_count += 1
                except Exception as exc:
                    failures.append(f"mesh_verification_unavailable:{type(exc).__name__}")
            for label, fraction in (() if enclosed else (("padding_25", 0.25), ("padding_50", 0.5))):
                if solve_count >= config.max_solves:
                    failures.append("verification_solve_budget_exhausted")
                    continue
                variant, padding, reason = owner._solve_padding_verification(
                    grid, frequency, fraction
                )
                if variant is None:
                    failures.append(reason)
                else:
                    variants[label] = (variant, padding)
                    solve_count += 1
        for mode_position, mode_index in enumerate(mode_indices):
            mode = int(mode_index) - 1
            verification_group = next(
                (
                    tuple(label - 1 for label in group)
                    for group in owner.degenerate
                    if mode_index in group
                ),
                (mode,),
            )
            residual = solver_mode_residual(solver, mode)
            field_residual = float(np.asarray(solver.field_residuals)[mode])
            numerical_valid = bool(
                np.isfinite(residual) and residual <= config.residual_tolerance
            )
            edge = _edge_fraction(solver, mode)
            checks = []
            if owner.verification == "full":
                if "mesh" in variants:
                    try:
                        check = _verification_check(
                            solver,
                            mode,
                            variants["mesh"][0],
                            config,
                            reduce_factor=2,
                            kind="mesh",
                            base_group=verification_group,
                        )
                    except Exception as exc:
                        failures.append(
                            f"mesh_verification_comparison_failed:{type(exc).__name__}"
                        )
                    else:
                        checks.append(check)
                base_shape = _cell_fields(solver, mode)[0][0].shape
                for label in ("padding_25", "padding_50"):
                    if label not in variants:
                        continue
                    variant, padding = variants[label]
                    crop = tuple((pad, pad + size) for pad, size in zip(padding, base_shape))
                    try:
                        check = _verification_check(
                            solver,
                            mode,
                            variant,
                            config,
                            crop=crop,
                            kind=label,
                            base_group=verification_group,
                        )
                    except Exception as exc:
                        failures.append(
                            f"{label}_verification_comparison_failed:{type(exc).__name__}"
                        )
                    else:
                        checks.append(check)
                expected_checks = 1 if enclosed else 3
                complete = len(checks) == expected_checks and not failures
                edge_ok = enclosed or edge <= config.edge_fraction_max
                if complete and all(check["passed"] for check in checks) and edge_ok:
                    confinement = "bound"
                    artifact = "none"
                    reasons = ()
                elif complete:
                    failed_checks = tuple(
                        check["kind"] for check in checks if not check["passed"]
                    )
                    boundary_evidence = any(
                        kind.startswith("padding") for kind in failed_checks
                    ) or (not enclosed and edge > config.edge_fraction_max)
                    confinement = "unbound_suspect" if boundary_evidence else "unresolved"
                    artifact = (
                        "artificial_boundary_or_box_suspect"
                        if boundary_evidence
                        else "none"
                    )
                    reasons = failed_checks + (
                        ("artificial_edge_participation",)
                        if not enclosed and edge > config.edge_fraction_max
                        else ()
                    )
                else:
                    confinement = "unresolved"
                    artifact = "unresolved"
                    reasons = tuple(dict.fromkeys(failures))
            else:
                checks = []
                confinement = "bound" if enclosed or edge <= config.edge_fraction_max else "unbound_suspect"
                artifact = "none" if confinement == "bound" else "artificial_boundary_or_box_suspect"
                reasons = () if confinement == "bound" else ("artificial_edge_participation",)
            records.append(
                {
                    "frequency": float(frequency),
                    "mode": int(mode_index),
                    "numerical_valid": numerical_valid,
                    "residual": residual,
                    "field_residual": field_residual,
                    "edge_fraction": edge,
                    "physical_enclosure": enclosed,
                    "confinement": confinement,
                    "artifact": artifact,
                    "reasons": reasons,
                    "verification": tuple(checks),
                }
            )
    owner.anchor_quality_diagnostics = records
    owner.tracking_diagnostics.setdefault(
        "requested_frequencies", np.asarray(frequencies, dtype=float)
    )
    owner.tracking_diagnostics.setdefault("adaptive_frequencies", np.empty(0, dtype=float))
    owner.tracking_diagnostics["verification_solve_count"] = int(
        solve_count - getattr(owner, "_tracking_extra_solves", 0)
    )
    owner.tracking_diagnostics["quality"] = records

    if owner.mpi_coordinator:
        for mode_index in mode_indices:
            suspect = [
                record
                for record in records
                if record["mode"] == mode_index and record["confinement"] != "bound"
            ]
            if not suspect:
                continue
            low = min(record["frequency"] for record in suspect)
            high = max(record["frequency"] for record in suspect)
            states = ", ".join(sorted({record["confinement"] for record in suspect}))
            reasons = ", ".join(
                sorted({reason for record in suspect for reason in record["reasons"]})
            ) or "confinement evidence is incomplete"
            logger.warning(
                f"Eigenmode port {owner.port_index} mode {mode_index} has {states} confinement "
                f"diagnostics from {low:g} to {high:g} Hz ({reasons}). Numerically valid, "
                "forward-power anchors remain in use, but modal injection and S-parameters "
                "may be less accurate; inspect the tracked dispersion and field plots."
            )
    return records


def _index(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError("Degenerate mode labels must be positive integers.")
    return int(value)


def normalize_groups(value, modes):
    if value is None:
        return ()
    try:
        groups = tuple(value)
    except TypeError as exc:
        raise ValueError("degenerate must contain groups of mode indices.") from exc
    if groups and all(isinstance(item, numbers.Integral) for item in groups):
        groups = (groups,)
    result, seen = [], set()
    for group in groups:
        try:
            group = tuple(_index(item) for item in group)
        except TypeError as exc:
            raise ValueError("degenerate must contain groups of mode indices.") from exc
        if len(group) < 2 or len(set(group)) != len(group):
            raise ValueError("Each degenerate group requires at least two distinct modes.")
        if seen.intersection(group) or not set(group).issubset(modes):
            raise ValueError("Degenerate groups must be disjoint and fully included in modes.")
        seen.update(group)
        result.append(group)
    return tuple(result)


def normalize_polarizations(value, groups, normal_axis, invariant_axis=None, *, unresolved=False):
    """Normalize a shared pair direction or explicit per-mode directions.

    The inferred partner is n_positive cross first, independent of the port's
    propagation sign. Automatic tracking expands it after detecting groups.
    """
    if value is None:
        return {}
    mapping = hasattr(value, "items")
    if (not mapping or value) and invariant_axis is not None:
        raise ValueError("Physical mode_polarizations require a 3D port cross-section.")

    def normalize_direction(direction):
        if isinstance(direction, str):
            if direction.lower() not in ("x", "y", "z"):
                raise ValueError("Mode polarization axes must be x, y, or z.")
            direction = np.eye(3)["xyz".index(direction.lower())]
        raw = np.asarray(direction)
        if np.iscomplexobj(raw):
            raise ValueError("Mode polarization directions must be real vectors.")
        vector = np.asarray(raw, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or not np.any(vector):
            raise ValueError("Mode polarization directions must be finite nonzero three-vectors.")
        vector = vector / np.max(np.abs(vector))
        vector /= np.linalg.norm(vector)
        if abs(vector[normal_axis]) > 1e-12:
            raise ValueError("Mode polarization must be transverse to the port normal.")
        vector[normal_axis] = 0
        return tuple(float(value) for value in vector / np.linalg.norm(vector))

    if mapping:
        result = {_index(mode): normalize_direction(direction) for mode, direction in value.items()}
    else:
        primary = normalize_direction(value)
        if unresolved:
            return primary
        if not groups or any(len(group) != 2 for group in groups):
            raise ValueError("A shared mode polarization requires degenerate two-mode pairs.")
        partner = tuple(float(v) for v in np.cross(np.eye(3)[normal_axis], primary))
        result = {mode: direction for group in groups for mode, direction in zip(group, (primary, partner))}
    if not unresolved and not set(result).issubset({item for group in groups for item in group}):
        raise ValueError("mode_polarizations must belong to a declared degenerate group.")
    for group in groups:
        if set(group).intersection(result):
            if len(group) != 2 or not set(group).issubset(result):
                raise ValueError(
                    "Physical polarization mappings require both members of a two-mode pair."
                )
            matrix = np.asarray([result[item] for item in group]).T
            if np.linalg.cond(matrix) > _CONDITION_LIMIT:
                raise ValueError(
                    "Requested polarization directions are linearly dependent or ill-conditioned."
                )
    return result


def parse_port_options(tokens):
    """Remove explicit key=value options before parsing the legacy anchor tail."""
    positional, options = [], {}
    for token in tokens:
        if "=" not in token:
            positional.append(token)
            continue
        key, value = token.split("=", 1)
        scalar_options = {
            "tracking": str,
            "verification": str,
            "residual_tolerance": float,
            "beta_drift_tolerance": float,
            "verification_overlap": float,
            "edge_fraction_max": float,
            "tracking_overlap": float,
            "assignment_margin": float,
            "unmatched_cost": float,
            "cluster_gap": float,
            "extra_candidates": int,
            "max_depth": int,
            "min_relative_step": float,
            "max_solves": int,
        }
        if key not in ("degenerate", "mode_polarizations", *scalar_options) or key in options:
            raise ValueError(f"Unknown or repeated eigenmode port option {key!r}.")
        try:
            if key in scalar_options:
                options[key] = scalar_options[key](value)
            elif key == "degenerate":
                options[key] = tuple(
                    tuple(int(v) for v in group.split(",")) for group in value.split(";")
                )
            elif ":" not in value:
                options[key] = tuple(float(v) for v in value.split(",")) if "," in value else value
            else:
                directions = {}
                for entry in value.split(";"):
                    label, vector = entry.split(":", 1)
                    label = int(label)
                    if label in directions:
                        raise ValueError("Repeated polarization label")
                    directions[label] = (
                        tuple(float(v) for v in vector.split(",")) if "," in vector else vector
                    )
                options[key] = directions
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid eigenmode port {key}: {value!r}.") from exc
    return positional, options


def _mix(fields, transform):
    return [
        [sum(fields[i][axis] * transform[i, j] for i in range(len(fields))) for axis in range(3)]
        for j in range(transform.shape[1])
    ]


def _frame(electric, magnetic, impedance):
    return np.column_stack(
        [
            np.concatenate([*(v.ravel() for v in e), *(impedance * v.ravel() for v in h)])
            for e, h in zip(electric, magnetic)
        ]
    )


def _orthonormal(frame, *, transformation=False):
    u, s, vh = np.linalg.svd(frame, full_matrices=False)
    if not np.all(np.isfinite(s)) or s[-1] <= s[0] / _CONDITION_LIMIT:
        raise ValueError("Degenerate modal group has rank loss or ill-conditioned fields.")
    return (u, vh.conj().T / s) if transformation else u


def _canonical_subspace_transform(frame):
    """Choose a repeatable basis from an unlabelled field subspace."""
    span = _orthonormal(frame)
    rank = span.shape[1]
    basis = []
    available = set(range(span.shape[0]))
    for _ in range(rank):
        strengths = {}
        for coordinate in available:
            candidate = span @ span[coordinate].conj()
            if basis:
                existing = np.column_stack(basis)
                candidate -= existing @ (existing.conj().T @ candidate)
            strengths[coordinate] = float(np.linalg.norm(candidate))
        strongest = max(strengths.values())
        # Symmetric apertures can give several mathematically equal pivots.
        # Choose the lowest coordinate among numerical ties so BLAS rounding
        # cannot swap the two oriented modes between platforms or raw gauges.
        coordinate = min(
            coordinate
            for coordinate, strength in strengths.items()
            if strongest - strength <= _SUBSPACE_PIVOT_TOLERANCE
        )
        candidate = span @ span[coordinate].conj()
        if basis:
            existing = np.column_stack(basis)
            candidate -= existing @ (existing.conj().T @ candidate)
        norm = float(np.linalg.norm(candidate))
        if not np.isfinite(norm) or norm <= np.finfo(float).eps:
            raise ValueError("Degenerate modal group has no deterministic full-rank basis.")
        candidate /= norm
        phase = candidate[coordinate]
        if abs(phase) > np.finfo(float).eps:
            candidate *= np.exp(-1j * np.angle(phase))
        basis.append(candidate)
        available.remove(coordinate)
    canonical = np.column_stack(basis)
    transform, _, _, _ = np.linalg.lstsq(frame, canonical, rcond=None)
    return transform


def _power(owner, grid, electric, magnetic):
    # Coefficient convention: power(c) = c.conj().T @ matrix @ c.
    matrix = np.asarray(
        [[owner._modal_cross_power(ej, hi, grid) for ej in electric] for hi in magnetic]
    )
    return (matrix + matrix.conj().T) / 2


def _moments(owner, grid, electric):
    u, v = owner.transverse_axes
    area = grid.dl[u] * grid.dl[v]
    return np.asarray(
        [
            [
                np.sum(owner._average_to_transverse_cells(e[u], "eu")) * area,
                np.sum(owner._average_to_transverse_cells(e[v], "ev")) * area,
            ]
            for e in electric
        ]
    ).T


def balanced_power(owner, grid, electric, magnetic):
    """Positive field norm using the solver's transverse quadrature."""
    impedance = owner._tracking_impedance
    if owner.invariant_axis is None:
        u, v = owner.transverse_axes
        fields = [
            owner._average_to_transverse_cells(field, component)
            for field, component in (
                (electric[u], "eu"),
                (electric[v], "ev"),
                (impedance * magnetic[u], "hu"),
                (impedance * magnetic[v], "hv"),
            )
        ]
        measure = grid.dl[u] * grid.dl[v]
    else:
        t, a = owner.physical_transverse_axis, owner.invariant_axis
        axes = (t, a) if owner.domain_polarization == "TE" else (a, t)
        fields = [
            owner._sample_1d_component(electric[axes[0]]),
            impedance * owner._sample_1d_component(magnetic[axes[1]]),
        ]
        if owner.domain_polarization == "TM":
            fields = [0.5 * (field[:-1] + field[1:]) for field in fields]
        measure = grid.dl[t]
    return float(sum(np.sum(abs(field) ** 2) for field in fields) * measure / (4 * impedance))


def _residuals(solver, group, transform):
    indices = np.asarray(group) - 1
    operator = getattr(solver, "mode_tracking_operator", None)
    if operator is not None:
        columns = np.column_stack(
            [
                np.concatenate(
                    (solver.Eu[..., i].ravel(order="F"), solver.Ev[..., i].ravel(order="F"))
                )
                for i in indices
            ]
        )[solver.free_euv_mask]
    else:
        operator = solver.operator
        free = solver.free_scalar_mask
        operator = operator[free, :][:, free]
        field = solver.Ea if solver.polarization == "TM" else solver.Ha
        columns = field[free][:, indices]
    mixed = columns @ transform
    applied = operator @ mixed
    target = mixed * np.asarray(solver.eigenvalues)[indices]
    denominator = np.linalg.norm(applied, axis=0) + np.linalg.norm(target, axis=0)
    return np.linalg.norm(applied - target, axis=0) / np.maximum(denominator, 1e-300)


def align_groups(owner, grid, frequencies, solvers, mode_indices, anchor_e, anchor_h, propagating):
    """Align declared groups in place; never change solved propagation constants."""
    owner.degenerate_diagnostics = []
    owner._degenerate_masks = {}
    owner._degenerate_overlaps = {}
    owner._degenerate_guard_trimmed = {}
    for group in owner.degenerate:
        positions = [mode_indices.index(item) for item in group]
        context = f"Eigenmode port {owner.port_index} degenerate group {group}"
        active = propagating[:, positions[0]].copy()
        if not np.all(propagating[:, positions] == active[:, None]):
            raise ValueError(f"{context} has inconsistent propagation/cutoff among its partners.")
        used = np.flatnonzero(active)
        if not len(used) or np.any(np.diff(used) != 1):
            raise ValueError(f"{context} requires one contiguous propagating anchor range.")
        explicit_physical = group[0] in owner.mode_polarizations
        automatic_axes = False
        if not explicit_physical and len(group) == 2 and owner.invariant_axis is None:
            automatic_axes = True
            for k in used:
                electric = [anchor_e[k][position] for position in positions]
                moment = _moments(owner, grid, electric)
                scale = np.asarray(
                    [
                        sum(
                            np.sum(abs(field))
                            for field in (
                                item[owner.transverse_axes[0]],
                                item[owner.transverse_axes[1]],
                            )
                        )
                        for item in electric
                    ]
                ) * np.prod(grid.dl[list(owner.transverse_axes)])
                if (
                    not np.isfinite(np.linalg.cond(moment))
                    or np.linalg.cond(moment) > _CONDITION_LIMIT
                    or np.any(np.linalg.norm(moment, axis=0) <= 1e-12 * scale)
                ):
                    automatic_axes = False
                    break
            if automatic_axes:
                for label, axis in zip(group, owner.transverse_axes):
                    owner.mode_polarizations[label] = tuple(float(value) for value in np.eye(3)[axis])
        physical = explicit_physical or automatic_axes
        orientation = (
            "explicit"
            if explicit_physical
            else "automatic_transverse_axes"
            if automatic_axes
            else "deterministic_subspace"
        )
        frames, transforms, raw, diagnostics = {}, {}, {}, []
        for k, solver in enumerate(solvers):
            eigenvalues = np.asarray(solver.eigenvalues)[np.asarray(group) - 1]
            if not np.all(np.isfinite(eigenvalues)):
                raise ValueError(f"{context} has nonfinite eigenvalues.")
            spread = float(np.max(abs(eigenvalues[:, None] - eigenvalues)))
            if spread > _DEGENERACY_TOLERANCE * max(1.0, float(np.max(abs(eigenvalues)))):
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has resolved eigenvalue splitting "
                    f"({spread:g}). Use independent modes or correct unintended geometry asymmetry."
                )
            # The second-order eigenvalue is -n_eff**2 and cannot distinguish
            # opposite phase branches. Such E/H pairs cannot be mixed even
            # when both carry forward real power (e.g. a backward-wave mode).
            indices = np.asarray(group) - 1
            propagation = np.asarray(solver.operator_neff)[indices]
            if active[k] and np.any(np.real(propagation[:, None] * propagation.conj()) < 0):
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has opposite propagation branches. "
                    "Use independent modes; equal squared eigenvalues do not permit mixing their E/H fields."
                )
            diagnostic = dict(
                frequency=frequencies[k],
                active=bool(active[k]),
                eigenvalue_spread=spread,
                eigenvalues=eigenvalues.copy(),
            )
            outside = [i for i in range(len(solver.eigenvalues)) if i + 1 not in group]
            if outside and np.any(
                abs(np.asarray(solver.eigenvalues)[outside, None] - eigenvalues)
                <= _DEGENERACY_TOLERANCE * max(1.0, float(np.max(abs(eigenvalues))))
            ):
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has a degenerate partner "
                    "outside the declared group. Include every partner in modes and degenerate."
                )
            diagnostics.append(diagnostic)
            if not active[k]:
                continue
            e = [anchor_e[k][p] for p in positions]
            h = [anchor_h[k][p] for p in positions]
            raw[k] = e, h
            frame = _frame(e, h, float(owner._tracking_impedance))
            # Test power in a well-conditioned representation of the subspace.
            # A raw Gram matrix squares the solver basis's condition number,
            # spuriously rejecting valid invertible changes of basis.
            _, preconditioner = _orthonormal(frame, transformation=True)
            power = _power(owner, grid, _mix(e, preconditioner), _mix(h, preconditioner))
            values, vectors = np.linalg.eigh(power)
            if values[0] <= values[-1] / _CONDITION_LIMIT or not np.all(np.isfinite(values)):
                raise ValueError(f"{context} has no independent positive-power basis.")
            diagnostic["power_condition"] = float(values[-1] / values[0])
            if physical:
                moment = _moments(owner, grid, e)
                condition = float(np.linalg.cond(moment))
                # Reject a numerically zero moment even if its two noise columns
                # happen to have a moderate relative condition number.
                scale = np.asarray(
                    [
                        sum(
                            np.sum(abs(field))
                            for field in (
                                item[owner.transverse_axes[0]],
                                item[owner.transverse_axes[1]],
                            )
                        )
                        for item in e
                    ]
                ) * np.prod(grid.dl[list(owner.transverse_axes)])
                if (
                    not np.isfinite(condition)
                    or condition > _CONDITION_LIMIT
                    or np.any(np.linalg.norm(moment, axis=0) <= 1e-12 * scale)
                ):
                    raise ValueError(
                        f"{context} at {frequencies[k]:g} Hz has zero or ill-conditioned "
                        "integrated transverse E. Axis/vector references cannot orient this group; "
                        "omit mode_polarizations to use generic subspace tracking."
                    )
                directions = np.asarray([owner.mode_polarizations[item] for item in group]).T
                diagnostic["direction_condition"] = float(np.linalg.cond(directions))
                transform = np.linalg.solve(moment, directions[list(owner.transverse_axes)])
                mixed_e, mixed_h = _mix(e, transform), _mix(h, transform)
                norms = np.real(np.diag(_power(owner, grid, mixed_e, mixed_h)))
                if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
                    raise ValueError(f"{context} has invalid polarized modal power.")
                transform /= np.sqrt(norms)[None, :]
                diagnostic["moment_condition"] = condition
            else:
                transform = _canonical_subspace_transform(frame)
                canonical_e = _mix(e, transform)
                canonical_h = _mix(h, transform)
                canonical_power = _power(owner, grid, canonical_e, canonical_h)
                canonical_values, canonical_vectors = np.linalg.eigh(canonical_power)
                if (
                    canonical_values[0] <= canonical_values[-1] / _CONDITION_LIMIT
                    or not np.all(np.isfinite(canonical_values))
                ):
                    raise ValueError(f"{context} has no deterministic positive-power basis.")
                transform = (
                    transform
                    @ canonical_vectors
                    @ np.diag(1 / np.sqrt(canonical_values))
                    @ canonical_vectors.conj().T
                )
            transforms[k] = transform
            frames[k] = frame @ transform
            residual = _residuals(solver, group, transform)
            if not np.all(np.isfinite(residual)) or np.max(residual) > _RESIDUAL_TOLERANCE:
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has mixed-mode eigen-residual "
                    f"{np.max(residual):g}; use independent modes or improve the eigensolve."
                )
        overlaps = np.full(max(0, len(frequencies) - 1), np.nan)
        guard_trimmed = False
        while len(used) > 1:
            retry = False
            for left, right in zip(used[:-1], used[1:]):
                singular = np.linalg.svd(
                    _orthonormal(frames[left]).conj().T @ _orthonormal(frames[right]),
                    compute_uv=False,
                )
                overlap = float(np.min(singular))
                overlaps[left] = overlap
                if not np.isfinite(overlap) or overlap < owner.ANCHOR_OVERLAP_ERROR_THRESHOLD:
                    trim = None
                    if owner._automatic_anchor_policy() and len(used) > 2:
                        tolerance = 1e-12 * max(1.0, max(frequencies))
                        if (
                            left == used[0]
                            and owner.dft_start is not None
                            and frequencies[right] <= owner.dft_start + tolerance
                        ):
                            trim = left
                        elif (
                            right == used[-1]
                            and owner.dft_stop is not None
                            and frequencies[left] >= owner.dft_stop - tolerance
                        ):
                            trim = right
                    if trim is not None:
                        active[trim] = False
                        diagnostics[trim]["active"] = False
                        diagnostics[trim]["guard_trimmed"] = True
                        used = np.flatnonzero(active)
                        guard_trimmed = retry = True
                        if owner.mpi_coordinator:
                            logger.warning(
                                f"{context}: trimming the whole group at guard anchor "
                                f"{frequencies[trim]:g} Hz (subspace overlap {overlap:g})."
                            )
                        break
                    raise ValueError(
                        f"{context} cannot be tracked between {frequencies[left]:g} and "
                        f"{frequencies[right]:g} Hz: subspace overlap {overlap:g}. "
                        "Add anchors or inspect missing partners and nearby modes."
                    )
                owner._check_anchor_overlap(
                    overlap,
                    frequencies[left],
                    frequencies[right],
                    group,
                    context,
                    coordinator=owner.mpi_coordinator,
                )
            if not retry:
                break
        if not np.all(propagating[:, positions[0]]) and owner.mpi_coordinator:
            logger.warning(
                f"{context}: excluding non-propagating anchors as a whole group "
                "from excitation and physical modal references; inspect validity masks near cutoff."
            )
        centre = owner.fallback_frequency
        if centre is None:
            centre = (frequencies[0] + frequencies[-1]) / 2
        reference = int(min(used, key=lambda k: (abs(frequencies[k] - centre), frequencies[k])))
        if not physical:
            for path in (
                list(range(reference + 1, used[-1] + 1)),
                list(range(reference - 1, used[0] - 1, -1)),
            ):
                previous = reference
                for k in path:
                    u, _, vh = np.linalg.svd(frames[k].conj().T @ frames[previous])
                    rotation = u @ vh
                    transforms[k] = transforms[k] @ rotation
                    frames[k] = frames[k] @ rotation
                    previous = k
        for k in used:
            e, h = raw[k]
            transform = transforms[k]
            residual = _residuals(solvers[k], group, transform)
            if not np.all(np.isfinite(residual)) or np.max(residual) > _RESIDUAL_TOLERANCE:
                raise ValueError(
                    f"{context} at {frequencies[k]:g} Hz has mixed-mode eigen-residual "
                    f"{np.max(residual):g}; use independent modes or improve the eigensolve."
                )
            e, h = _mix(e, transform), _mix(h, transform)
            for j, p in enumerate(positions):
                anchor_e[k][p], anchor_h[k][p] = e[j], h[j]
            diagnostics[k].update(
                transform=transform, residual=residual, power_gram=_power(owner, grid, e, h)
            )
            if physical:
                diagnostics[k]["electric_moments"] = _moments(owner, grid, e)
        for p in positions:
            owner._degenerate_masks[p] = active
            owner._degenerate_overlaps[p] = overlaps
            owner._degenerate_guard_trimmed[p] = guard_trimmed
        owner.degenerate_diagnostics.append(
            dict(
                modes=group,
                physical=physical,
                orientation=orientation,
                reference_anchor=reference,
                overlaps=overlaps,
                anchors=diagnostics,
            )
        )


def write_diagnostics(group, owner):
    diagnostics = getattr(owner, "tracking_diagnostics", None)
    if diagnostics is not None:
        root = group.create_group("mode_tracking")
        root.attrs["SchemaVersion"] = int(diagnostics["schema_version"])
        root.attrs["SourceRevision"] = diagnostics["source_revision"]
        root.attrs["Tracking"] = owner.tracking
        root.attrs["Verification"] = owner.verification
        root.attrs["ReferenceAnchorIndex"] = int(diagnostics["reference_anchor"])
        root.attrs["VerificationSolveCount"] = int(
            diagnostics.get("verification_solve_count", 0)
        )
        root["candidate_indices"] = diagnostics["candidate_indices"]
        root["tracking_overlaps"] = diagnostics["overlaps"]
        root["combined_residuals"] = diagnostics["residuals"]
        root["eigenpair_residuals"] = diagnostics.get(
            "eigenpair_residuals", diagnostics["residuals"]
        )
        root["field_residuals"] = diagnostics.get(
            "field_residuals", diagnostics["residuals"]
        )
        root["numerical_valid"] = diagnostics["numerical_valid"].astype(np.uint8)
        root["requested_frequencies"] = diagnostics.get(
            "requested_frequencies", np.empty(0, dtype=float)
        )
        root["adaptive_frequencies"] = diagnostics.get(
            "adaptive_frequencies", np.empty(0, dtype=float)
        )
        unresolved = diagnostics.get("unresolved_intervals", ())
        root["unresolved_interval_lower_frequency"] = np.asarray(
            [np.nan if item[0] is None else item[0] for item in unresolved], dtype=float
        )
        root["unresolved_interval_upper_frequency"] = np.asarray(
            [np.nan if item[1] is None else item[1] for item in unresolved], dtype=float
        )
        root["unresolved_interval_modes"] = np.asarray(
            [
                "" if item[2] is None else ",".join(str(mode) for mode in np.atleast_1d(item[2]))
                for item in unresolved
            ],
            dtype="S",
        )
        groups = diagnostics.get("automatic_degenerate_groups", ())
        root.attrs["AutomaticDegenerateGroups"] = ";".join(
            ",".join(str(item) for item in cluster) for cluster in groups
        )
        quality = diagnostics.get("quality", ())
        if quality:
            qgroup = root.create_group("anchor_quality")
            qgroup["frequency"] = [record["frequency"] for record in quality]
            qgroup["mode"] = [record["mode"] for record in quality]
            qgroup["residual"] = [record["residual"] for record in quality]
            qgroup["field_residual"] = [record["field_residual"] for record in quality]
            qgroup["edge_fraction"] = [record["edge_fraction"] for record in quality]
            qgroup["numerical_valid"] = np.asarray(
                [record["numerical_valid"] for record in quality], dtype=np.uint8
            )
            qgroup["confinement"] = np.asarray(
                [record["confinement"] for record in quality], dtype="S"
            )
            qgroup["artifact"] = np.asarray(
                [record["artifact"] for record in quality], dtype="S"
            )
            qgroup["reasons"] = np.asarray(
                [";".join(record["reasons"]) for record in quality], dtype="S"
            )

    if getattr(owner, "degenerate_diagnostics", None):
        root = group.create_group("degenerate_groups")
        for number, record in enumerate(owner.degenerate_diagnostics):
            output = root.create_group(str(number + 1))
            output.attrs["ModeIndices"] = record["modes"]
            output.attrs["PhysicalPolarization"] = record["physical"]
            output.attrs["Orientation"] = record["orientation"]
            output.attrs["ReferenceAnchorIndex"] = record["reference_anchor"]
            output.attrs["TransverseAxes"] = owner.transverse_axes
            output.attrs["TransformationConvention"] = "aligned fields = raw fields @ transform"
            output["subspace_overlaps"] = record["overlaps"]
            if record["physical"]:
                directions = [
                    owner.mode_polarizations[item] for item in record["modes"]
                ]
                output["polarization_directions"] = directions
                if record["orientation"] == "explicit":
                    output["requested_directions"] = directions
            for k, anchor in enumerate(record["anchors"]):
                row = output.create_group(f"anchor{k}")
                for key, value in anchor.items():
                    row[key] = value


def aligned_plot_solvers(owner, solvers):
    """Plot copies in the authoritative bank basis without altering raw solves."""
    result = [copy(solver) for solver in solvers]
    for record in owner.degenerate_diagnostics:
        indices = np.asarray(record["modes"]) - 1
        for k, diagnostic in enumerate(record["anchors"]):
            if not diagnostic["active"]:
                continue
            for name in ("Eu", "Ev", "Ew", "Hu", "Hv", "Hw", "Ea", "Ha", "Et", "Ht"):
                field = getattr(solvers[k], name, None)
                if field is None:
                    continue
                destination = np.array(getattr(result[k], name), copy=True)
                destination[..., indices] = field[..., indices] @ diagnostic["transform"]
                setattr(result[k], name, destination)
            result[k].mode_polarizations = owner.mode_polarizations
    return tuple(result)
