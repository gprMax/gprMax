"""Controlled algebraic validation of degeneracy versus accidental crossings.

These diagonal eigenproblems exercise production E/H association and bank
grouping. They are not Maxwell/FDTD simulations or new scattering results.
Run: python -m testing.validation.persistent_mode_groups --output results.json
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import diags

from gprMax.eigenmode_config import EigenmodeTrackingConfig
from gprMax.eigenmode_tracking import track_solver_bank
from gprMax.sources import EigenmodeAnchorMismatchError


def make_solver(values, rotation=None):
    """Return independently rank-ordered exact eigenpairs with mock E/H fields."""
    values = np.asarray(values, dtype=float)
    count = len(values)
    if rotation is None:
        rotation = np.eye(count, dtype=complex)
    order = np.argsort(values, kind="stable")
    basis = np.asarray(rotation, dtype=complex)[:, order]
    zero = np.zeros_like(basis)
    neff = np.sqrt(-values[order] + 0j)
    return SimpleNamespace(
        num_modes=count,
        eigenvalues=values[order].copy(),
        eigenvectors=basis.copy(),
        operator=diags(values, format="csr"),
        free_scalar_mask=np.ones(count, dtype=bool),
        polarization="TM",
        Et=zero.copy(), Ea=basis.copy(), Ew=zero.copy(),
        Ht=basis.copy(), Ha=zero.copy(), Hw=zero.copy(),
        beta=neff.copy(), operator_neff=neff.copy(), complex_neff=neff.copy(),
        real_neff=neff.real.copy(),
        field_residuals=np.zeros(count),
    )


def make_owner(reference=0.0):
    return SimpleNamespace(
        tracking_config=EigenmodeTrackingConfig(extra_candidates=0),
        fallback_frequency=reference, _tracking_impedance=1.0, port_index=1,
        degenerate=(), mode_polarizations={}, normal_axis=2, invariant_axis=None,
    )


def unitary(rng, size):
    return np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))[0]


def pair_crossing_bank(frequencies, *, seed=41, two_pairs=False, mixed_crossing=False):
    rng = np.random.default_rng(seed)
    count = 4 if two_pairs else 3
    solvers = []
    for f in frequencies:
        values = [-4.0, -4.0] + [-4.0 + 0.04 * f] * (count - 2)
        rotation = np.eye(count, dtype=complex)
        rotation[:2, :2] = unitary(rng, 2)
        if two_pairs:
            rotation[2:, 2:] = unitary(rng, 2)
        else:
            rotation[2, 2] = np.exp(1j * rng.uniform(-np.pi, np.pi))
        if f == 0 and mixed_crossing:
            i, j = np.indices((count, count))
            rotation = np.exp(2j * np.pi * i * j / count) / np.sqrt(count)
        solvers.append(make_solver(values, rotation))
    return solvers


def validate(seed=41):
    records = {}
    frequencies = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    for two_pairs, name in ((False, "pair_and_single"), (True, "two_pairs")):
        solvers = pair_crossing_bank(frequencies, seed=seed, two_pairs=two_pairs)
        owner = make_owner()
        mapping = track_solver_bank(owner, frequencies, solvers, solvers[0].num_modes)
        expected = ((1, 2), (3, 4)) if two_pairs else ((1, 2),)
        assert owner.degenerate == expected
        error = max(
            np.linalg.norm(s.Ea[:, :2] @ s.Ea[:, :2].conj().T - np.diag([1, 1] + [0] * (s.num_modes - 2)))
            for s in solvers
        )
        assert error < 1e-12
        records[name] = {
            "persistent_groups": owner.degenerate,
            "maximum_pair_projector_error": float(error),
            "minimum_assignment_overlap": float(np.min(owner.tracking_diagnostics["overlaps"])),
            "maximum_eigen_residual": float(np.max(owner.tracking_diagnostics["residuals"])),
            "candidate_indices": mapping.tolist(),
        }
    solvers = [make_solver([-1 + 2e-6 * f, -1 - 2e-6 * f]) for f in frequencies]
    owner = make_owner()
    track_solver_bank(owner, frequencies, solvers, 2)
    assert owner.degenerate == ()
    assert all(np.allclose(abs(s.Ea), np.eye(2)) for s in solvers)
    records["resolved_crossing_inside_candidate_gap"] = {"persistent_groups": owner.degenerate}
    rng = np.random.default_rng(seed)
    solvers = [make_solver([-4] * 4, unitary(rng, 4)) for f in frequencies]
    owner = make_owner()
    track_solver_bank(owner, frequencies, solvers, 4)
    assert owner.degenerate == ((1, 2, 3, 4),)
    records["fourfold_degeneracy"] = {"persistent_groups": owner.degenerate}
    solvers = pair_crossing_bank(frequencies, seed=seed, mixed_crossing=True)
    owner = make_owner(reference=-1.0)
    try:
        track_solver_bank(owner, frequencies, solvers, 3)
    except EigenmodeAnchorMismatchError as error:
        records["fully_mixed_exact_crossing"] = {"status": "unresolved", "reason": str(error)}
    else:
        raise AssertionError("A fully mixed crossing must not acquire arbitrary individual labels.")
    return {
        "validation_type": "controlled algebraic eigenproblem; no FDTD scattering data",
        "seed": seed, "frequency_parameter": frequencies.tolist(), "cases": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    results = validate()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
