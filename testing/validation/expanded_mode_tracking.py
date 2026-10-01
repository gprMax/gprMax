"""Expanded tracking validation with known subspaces and full-vector FDFD ports.

Run with ``python -m testing.validation.expanded_mode_tracking --output DIR``.
Constructed matrix problems test association, not Maxwell/source accuracy.
The FDFD material sweep is a sequence of independent constitutive models at
20 GHz, not a causal dispersive material or a time-domain scattering run.
"""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from gprMax.eigenmode_tracking import track_solver_bank
from gprMax.sources import EigenmodeAnchorMismatchError
from testing.validation.persistent_mode_groups import make_owner, make_solver, pair_crossing_bank, unitary


CASES = {
    "three_pairs": ((2, 2, 2), "Three degenerate pairs cross together"),
    "triple_pair_single": ((3, 2, 1), "Threefold group, pair, and single mode"),
    "repeated_crossings": ((2, 1, 1), "A pair meets two curved branches"),
    "resolved_split": ((1, 1, 1), "A resolved split pair stays separate"),
    "complex_nonnormal": ((2, 2, 1), "Complex eigenvalues, nonorthogonal fields"),
    "avoided_crossing": ((1, 1), "Avoided crossing: fields evolve continuously"),
}


def projector(frame):
    q = np.linalg.qr(frame)[0]
    return q @ q.conj().T


def controlled_bank(case, axis, seed):
    sizes, _ = CASES[case]
    groups = np.split(np.arange(sum(sizes)), np.cumsum(sizes)[:-1])
    rng = np.random.default_rng(seed)
    count = sum(sizes)
    random = rng.normal(size=(count, count)) + 1j * rng.normal(size=(count, count))
    generator = random - random.conj().T
    generator *= 0.25 / np.linalg.norm(generator, 2)
    solvers, truths, spectra, orders = [], [], [], []
    for x in axis:
        values = -4 + np.repeat(np.linspace(-0.3, 0.3, len(sizes)) * x, sizes)
        frame = expm(x * generator)
        if case == "repeated_crossings":
            values = np.array([-4, -4, -4 + 0.4 * (x*x - 0.25), -4 + 0.15*x])
        elif case == "resolved_split":
            values = np.array([-4 - 2e-6, -4 + 2e-6, -4 + 0.2*x])
        elif case == "complex_nonnormal":
            values = values.astype(complex) + 0.04j
            shear = np.eye(count, dtype=complex)
            shear[0, -1] = 0.15
            frame = frame @ shear
        elif case == "avoided_crossing":
            coupling = 0.12
            values = -4 + np.array([-1, 1]) * np.sqrt((0.3*x)**2 + coupling**2)
            angle = 0.5 * np.arctan2(coupling, 0.3*x)
            frame = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        gauge = np.zeros((count, count), dtype=complex)
        for group in groups:
            gauge[np.ix_(group, group)] = unitary(rng, len(group))
        fields = frame @ gauge
        fields /= np.linalg.norm(fields, axis=0)
        order = np.argsort(values.real, kind="stable")
        solver = make_solver(values.real)
        solver.operator = csr_matrix(frame @ np.diag(values) @ np.linalg.inv(frame))
        solver.eigenvalues = values[order].copy()
        solver.Ea = fields[:, order].copy()
        solver.Ht = solver.Ea.copy()
        solver.eigenvectors = solver.Ea.copy()
        neff = np.sqrt(-values[order] + 0j)
        for name in ("beta", "operator_neff", "complex_neff"):
            setattr(solver, name, neff.copy())
        solver.real_neff = neff.real.copy()
        solvers.append(solver)
        truths.append([projector(frame[:, group]) for group in groups])
        spectra.append(values)
        orders.append(order)
    return solvers, truths, np.array(spectra), orders, groups


def controlled_trial(case, seed=41, reference=0.0, anchors=41):
    axis = np.linspace(-1, 1, anchors)
    solvers, truths, _, orders, groups = controlled_bank(case, axis, seed)
    center = int(np.argmin(abs(axis - reference)))
    labels = [np.flatnonzero(np.isin(orders[center], group)) for group in groups]
    owner = make_owner(reference)
    mapping = track_solver_bank(owner, axis, solvers, solvers[0].num_modes)
    expected = tuple(sorted(tuple(int(i + 1) for i in label) for label in labels if len(label) > 1))
    assert owner.degenerate == expected, (case, owner.degenerate, expected)
    error = max(
        np.linalg.norm(projector(solver.Ea[:, label]) - truths[k][g], 2)
        for k, solver in enumerate(solvers) for g, label in enumerate(labels)
    )
    assert error < 1e-11, (case, error)
    diagnostics = owner.tracking_diagnostics
    residual = float(np.max(diagnostics["residuals"]))
    assert residual < 1e-12
    return {
        "case": case, "seed": seed, "reference": reference,
        "persistent_groups": expected, "maximum_projector_error": float(error),
        "maximum_residual": residual,
        "minimum_overlap": float(np.min(diagnostics["overlaps"])),
        "axis": axis.tolist(),
        "tracked_real_neff": np.array([s.complex_neff.real for s in solvers]).tolist(),
        "candidate_indices": mapping.tolist(), "group_labels": [label.tolist() for label in labels],
    }


def abstention_trials(seeds=12):
    records = []
    for seed in range(seeds):
        for reference in (-1., 0., 1.):
            solvers = pair_crossing_bank([-1, 0, 1], seed=seed, mixed_crossing=True)
            before = [s.Ea.copy() for s in solvers]
            try:
                track_solver_bank(make_owner(reference), [-1, 0, 1], solvers, 3)
            except EigenmodeAnchorMismatchError:
                for solver, fields in zip(solvers, before):
                    np.testing.assert_array_equal(solver.Ea, fields)
                records.append({"seed": seed, "reference": reference, "status": "unresolved"})
            else:
                raise AssertionError("A fully mixed crossing acquired unsupported member labels")
    # This checks the tracking routine's rejection and subsequent dense-bank
    # success; it does not invoke the source builder's automatic refinement.
    try:
        controlled_trial("avoided_crossing", anchors=3)
    except EigenmodeAnchorMismatchError:
        coarse = "unresolved"
    else:
        raise AssertionError("Three anchors should under-resolve this avoided crossing")
    dense = controlled_trial("avoided_crossing", anchors=17)
    return {"fully_mixed_crossings": records, "coarse_avoided_crossing": coarse,
            "denser_avoided_crossing": {"anchors": 17, "status": "resolved",
                "maximum_projector_error": dense["maximum_projector_error"]}}


def fd_solver(frequency, shape, spacing, epsilon, count=6):
    from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver

    nu, nv = shape
    eu = np.full((nu, nv + 1), epsilon[0], dtype=complex)
    ev = np.full((nu + 1, nv), epsilon[1], dtype=complex)
    ew = np.full((nu + 1, nv + 1), epsilon[2], dtype=complex)
    eu[:, [0, -1]] = np.inf
    ev[[0, -1], :] = np.inf
    ew[:, [0, -1]] = np.inf
    ew[[0, -1], :] = np.inf
    solver = FDFD_2D_mode_solver(
        frequency, spacing, spacing, count - 1, eu, ev, ew,
        np.ones_like(ev), np.ones_like(eu), np.ones((nu, nv)),
    )
    solver.retain_tracking_operator = True
    solver.calculate_diagnostics = True
    solver.solve()
    return solver


def square_two_pairs():
    """Two doubly degenerate Maxwell eigenspaces crossing in a square port."""
    import gprMax.config as config

    axis = np.linspace(2.3, 2.7, 40)
    solvers = [fd_solver(20e9, (12, 12), .02 / 12, (2, 2, ez), count=12) for ez in axis]
    raw = np.array([s.complex_neff.copy() for s in solvers])
    owner = make_owner(axis[0])
    owner._tracking_impedance = config.sim_config.em_consts["z0"]
    mapping = track_solver_bank(owner, axis, solvers, 8)
    assert (5, 6) in owner.degenerate and (7, 8) in owner.degenerate
    assert (5, 6, 7, 8) not in owner.degenerate
    k0 = 2*np.pi*20e9 / config.sim_config.em_consts["c"]
    kt1 = 2*np.sin(np.pi / 24) / (.02 / 12)
    kt2 = 2*np.sin(np.pi / 12) / (.02 / 12)
    expected_te = np.sqrt(2 - (kt2/k0)**2)
    expected_tm = np.sqrt(2 - 2/axis*((kt1/k0)**2 + (kt2/k0)**2))
    tracked = np.array([s.complex_neff[:8] for s in solvers])
    error = max(np.max(abs(tracked[:, 4:6] - expected_te)),
                np.max(abs(tracked[:, 6:8] - expected_tm[:, None])))
    assert error < 1e-10
    longitudinal = np.array([[np.linalg.norm(s.Ew[..., j]) /
        np.sqrt(sum(np.linalg.norm(getattr(s, field)[..., j])**2
                    for field in ("Eu", "Ev", "Ew")))
        for j in range(8)] for s in solvers])
    assert np.max(longitudinal[:, 4:6]) < 1e-8
    assert np.min(longitudinal[:, 6:8]) > 0.01
    assert np.max(owner.tracking_diagnostics["residuals"]) < 1e-8
    # Raw ranks exchange between the two pairs, while their spans stay distinct.
    assert set(mapping[0, 4:6]) == {4, 5}
    assert set(mapping[-1, 4:6]) == {6, 7}
    assert set(mapping[-1, 6:8]) == {4, 5}
    return {
        "axis": axis.tolist(), "axis_label": "Longitudinal relative permittivity",
        "frequency_hz": 20e9, "width_m": .02, "cells": 12,
        "persistent_groups": owner.degenerate, "candidate_indices": mapping.tolist(),
        "crossing_epsilon_z": float(2*(1 + (kt1/kt2)**2)),
        "raw_real_neff": raw.real.tolist(), "tracked_real_neff": tracked.real.tolist(),
        "longitudinal_electric_fraction": longitudinal.tolist(),
        "overlaps": owner.tracking_diagnostics["overlaps"].tolist(),
        "maximum_analytic_neff_error": float(error),
        "maximum_residual": float(np.max(owner.tracking_diagnostics["residuals"])),
        "minimum_overlap": float(np.min(owner.tracking_diagnostics["overlaps"])),
    }


def physical_sweeps():
    import gprMax.config as config

    previous_config = getattr(config, "sim_config", None)
    config.sim_config = SimpleNamespace(em_consts=dict(
        e0=8.8541878128e-12, m0=1.25663706212e-6, c=299792458., z0=376.73031366686166,
    ))
    results = {}
    try:
        for loss in (0.0, 0.01):
            # Avoid placing an anchor on the exact triple coincidence. A separate
            # test below deliberately inserts that potentially ambiguous anchor.
            axis = np.linspace(3.6, 4.4, 40)
            solvers = [fd_solver(20e9, (12, 12), .02 / 12,
                                 np.array([2, 2, ez]) * (1 - 1j*loss)) for ez in axis]
            raw = np.array([s.complex_neff.copy() for s in solvers])
            bank = copy.deepcopy(solvers)
            owner = make_owner(axis[0])
            owner._tracking_impedance = config.sim_config.em_consts["z0"]
            mapping = track_solver_bank(owner, axis, bank, 3)
            assert owner.degenerate == ((1, 2),)
            assert np.max(owner.tracking_diagnostics["residuals"]) < 1e-8
            # Analytic Yee-grid TE10/01 and TM11 dispersion for the homogeneous
            # uniaxial square. This oracle is independent of association logic.
            k0 = 2*np.pi*20e9 / config.sim_config.em_consts["c"]
            kt = 2*np.sin(np.pi / 24) / (.02 / 12)
            expected_pair = np.sqrt(2*(1 - 1j*loss) - (kt/k0)**2)
            expected_single = np.sqrt(2*(1 - 1j*loss) - 4/axis*(kt/k0)**2)
            tracked = np.array([s.complex_neff[:3] for s in bank])
            error = max(np.max(abs(tracked[:, :2] - expected_pair)),
                        np.max(abs(tracked[:, 2] - expected_single)))
            assert error < 1e-10, error
            # The pair is transverse electric; the crossing branch has Ez.
            longitudinal = np.array([[np.linalg.norm(s.Ew[..., j]) /
                np.sqrt(sum(np.linalg.norm(getattr(s, field)[..., j])**2
                            for field in ("Eu", "Ev", "Ew")))
                for j in range(3)] for s in bank])
            assert np.max(longitudinal[:, :2]) < 1e-8
            assert np.min(longitudinal[:, 2]) > 0.01
            key = "square_lossless" if loss == 0 else "square_lossy"
            results[key] = {
                "axis": axis.tolist(), "axis_label": "Longitudinal relative permittivity (real part)",
                "frequency_hz": 20e9, "width_m": .02, "cells": 12,
                "loss_tangent": loss, "persistent_groups": owner.degenerate,
                "raw_real_neff": raw.real.tolist(), "tracked_real_neff": tracked.real.tolist(),
                "tracked_imag_neff": tracked.imag.tolist(), "candidate_indices": mapping.tolist(),
                "longitudinal_electric_fraction": longitudinal.tolist(),
                "maximum_analytic_neff_error": float(error),
                "maximum_residual": float(np.max(owner.tracking_diagnostics["residuals"])),
                "minimum_overlap": float(np.min(owner.tracking_diagnostics["overlaps"])),
            }
            exact_bank = copy.deepcopy(solvers)
            exact_bank.insert(20, fd_solver(20e9, (12, 12), .02 / 12,
                                           np.array([2, 2, 4]) * (1 - 1j*loss)))
            exact_owner = make_owner(axis[0])
            exact_owner._tracking_impedance = owner._tracking_impedance
            try:
                track_solver_bank(exact_owner, np.insert(axis, 20, 4.0), exact_bank, 3)
            except EigenmodeAnchorMismatchError as exc:
                results[key]["exact_crossing_anchor"] = {"status": "unresolved", "reason": str(exc)}
            else:
                assert exact_owner.degenerate == ((1, 2),)
                results[key]["exact_crossing_anchor"] = {
                    "status": "resolved",
                    "minimum_overlap": float(np.min(exact_owner.tracking_diagnostics["overlaps"])),
                    "note": "Acceptance uses the configured span threshold; it does not uniquely fix the basis at exact coincidence.",
                }
            print(f"Completed {key}", flush=True)
        axis = np.linspace(14, 22, 65)
        solvers = [fd_solver(f*1e9, (16, 8), .001, (2.25, 1, 1), count=8) for f in axis]
        raw = np.array([s.complex_neff.copy() for s in solvers])
        owner = make_owner(18)
        owner._tracking_impedance = config.sim_config.em_consts["z0"]
        mapping = track_solver_bank(owner, axis, solvers, 4)
        assert owner.degenerate == ()
        # Independent discrete dispersion for the two fundamental polarizations.
        k0 = 2*np.pi*axis*1e9 / config.sim_config.em_consts["c"]
        ex_neff = np.sqrt(2.25 - (2*np.sin(np.pi/16)/.001/k0)**2 + 0j)
        ey_neff = np.sqrt(1 - (2*np.sin(np.pi/32)/.001/k0)**2 + 0j)
        tracked = np.array([s.complex_neff[:4] for s in solvers])
        error = max(np.max(abs(tracked[:, 0] - ex_neff)), np.max(abs(tracked[:, 1] - ey_neff)))
        assert error < 1e-9
        assert np.max(owner.tracking_diagnostics["residuals"]) < 1e-8
        results["anisotropic_frequency"] = {
            "axis": axis.tolist(), "axis_label": "Frequency (GHz)",
            "epsilon": [2.25, 1, 1], "width_height_m": [.016, .008], "spacing_m": .001,
            "persistent_groups": owner.degenerate, "candidate_indices": mapping.tolist(),
            "raw_real_neff": raw.real.tolist(),
            "tracked_real_neff": np.array([s.complex_neff[:4].real for s in solvers]).tolist(),
            "maximum_residual": float(np.max(owner.tracking_diagnostics["residuals"])),
            "maximum_fundamental_analytic_neff_error": float(error),
            "minimum_overlap": float(np.min(owner.tracking_diagnostics["overlaps"])),
        }
        print("Completed anisotropic frequency sweep", flush=True)
        results["square_two_pairs"] = square_two_pairs()
        print("Completed square-guide degenerate pair / degenerate pair crossing", flush=True)
    finally:
        config.sim_config = previous_config
    return results


def plots(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (case, record) in zip(axes.flat, report["representative"].items()):
        x = np.array(record["axis"])
        y = np.array(record["tracked_real_neff"])
        for g, labels in enumerate(record["group_labels"]):
            ax.plot(x, y[:, labels[0]], color=colors[g], lw=2,
                    label="B" + "/B".join(str(i+1) for i in labels))
        ax.set(title=CASES[case][1], xlabel="Continuation parameter", ylabel="Re effective index")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        if case == "resolved_split":
            inset = ax.inset_axes([.12, .12, .40, .27])
            for g in (0, 1):
                label = record["group_labels"][g][0]
                inset.plot(x, (y[:, label] - 2)*1e6, color=colors[g], lw=1.5)
            inset.set_title("Split branches: $(n-2)10^6$", fontsize=8)
            inset.tick_params(labelsize=7)
            inset.grid(alpha=.2)
    fig.suptitle("Constructed eigenproblems: tracked branches and persistent subspaces\nRandom complex phases and degenerate-basis rotations at every anchor", fontsize=15)
    fig.savefig(output / "controlled_tracking.png", dpi=160)
    fig.savefig(output / "controlled_tracking.pdf")
    plt.close(fig)
    physical = report["physical"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for column, (key, record) in enumerate(list(physical.items())[:3]):
        x = np.array(record["axis"])
        y = np.array(record["tracked_real_neff"])
        top, bottom = axes[:, column]
        groups = ((0, 1), (2,)) if key.startswith("square") else tuple((i,) for i in range(4))
        for g, labels in enumerate(groups):
            top.plot(x, y[:, labels[0]], color=colors[g], lw=2,
                     label="B" + "/B".join(str(i+1) for i in labels))
        top.set(title={"square_lossless": "Square guide: pair + single crossing",
                       "square_lossy": "Same crossing with dielectric loss",
                       "anisotropic_frequency": "Anisotropic guide: four tracked branches"}[key],
                xlabel=record["axis_label"], ylabel="Re effective index")
        top.legend()
        indices = np.array(record["candidate_indices"])
        for j in range(indices.shape[1]):
            bottom.step(x, indices[:, j] + 1, where="mid", color=colors[j], lw=1.6, label=f"B{j+1}")
        bottom.set(xlabel=record["axis_label"], ylabel="Raw candidate rank of tracked label")
        bottom.set_yticks(range(1, int(indices.max()) + 2))
        bottom.legend(ncol=2)
        for ax in (top, bottom):
            ax.grid(alpha=.2)
    fig.suptitle("Full-vector Yee-grid FDFD validation\nSquare guides: material sweep at 20 GHz; rectangular guide: frequency sweep", fontsize=15)
    fig.savefig(output / "fdfd_tracking.png", dpi=160)
    fig.savefig(output / "fdfd_tracking.pdf")
    plt.close(fig)
    record = physical["square_lossy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = record["axis"]
    for g, labels in enumerate(((0, 1), (2,))):
        label = "B1/B2 (TE pair)" if g == 0 else "B3 (TM mode)"
        axes[0].plot(x, np.array(record["tracked_imag_neff"])[:, labels[0]], color=colors[g], label=label)
        axes[1].plot(x, np.array(record["longitudinal_electric_fraction"])[:, labels[0]], color=colors[g], label=label)
    axes[0].set_ylabel("Im effective index")
    axes[1].set_ylabel("Longitudinal electric-field fraction")
    for ax in axes:
        ax.set_xlabel("Longitudinal relative permittivity (real part)")
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle("Lossy square guide: attenuation and independent TE/TM identity check")
    fig.savefig(output / "lossy_identity.png", dpi=160)
    plt.close(fig)
    record = physical["square_two_pairs"]
    x = np.array(record["axis"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    groups = ((4, 5), (6, 7))
    labels = ("B5/B6: TE20 / TE02", "B7/B8: TM12 / TM21")
    for g, (members, label) in enumerate(zip(groups, labels)):
        axes[0, 0].plot(x, np.array(record["tracked_real_neff"])[:, members[0]],
                        color=colors[g], lw=2, label=label)
        indices = np.array(record["candidate_indices"])[:, members] + 1
        axes[0, 1].fill_between(x, np.min(indices, axis=1), np.max(indices, axis=1),
                               step="mid", color=colors[g], alpha=.18)
        axes[0, 1].step(x, np.min(indices, axis=1), where="mid", color=colors[g], label=label)
        axes[0, 1].step(x, np.max(indices, axis=1), where="mid", color=colors[g])
        axes[1, 0].plot(x, np.max(np.array(record["longitudinal_electric_fraction"])[:, members], axis=1),
                        color=colors[g], lw=2, label=label)
        axes[1, 1].plot(x, 1 - np.min(np.array(record["overlaps"])[:, members], axis=1),
                        color=colors[g], lw=2, label=label)
    axes[0, 0].set_ylabel("Re effective index")
    axes[0, 1].set(ylabel="Raw ranks occupied by each tracked pair", yticks=(5, 6, 7, 8))
    axes[1, 0].set_ylabel("Maximum longitudinal E fraction in pair")
    axes[1, 1].set_ylabel("Adjacent span mismatch (1 - overlap)")
    axes[1, 1].set_ylim(-1e-7, 2.5e-6)
    for ax in axes.flat:
        ax.axvline(record["crossing_epsilon_z"], color=".4", ls=":", lw=1)
        ax.set_xlabel("Longitudinal relative permittivity")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Full-vector FDFD: one degenerate pair crosses another\n20 mm square guide, 20 GHz, transverse permittivity = 2", fontsize=14)
    fig.savefig(output / "square_two_pairs.png", dpi=160)
    fig.savefig(output / "square_two_pairs.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": __doc__, "trials": [], "representative": {}}
    for case in CASES:
        for seed in range(args.seeds):
            for reference in (-1., 0., 1.):
                record = controlled_trial(case, seed, reference)
                report["trials"].append({k: v for k, v in record.items() if k not in (
                    "axis", "raw_real_neff", "tracked_real_neff", "candidate_indices", "group_labels")})
        report["representative"][case] = controlled_trial(case)
        print(f"Completed {case}: {args.seeds * 3} reference/seed combinations", flush=True)
    report["abstention"] = abstention_trials(args.seeds)
    print("Completed ambiguous-crossing rejection and anchor-density checks", flush=True)
    report["physical"] = physical_sweeps()
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plots(report, args.output)
    print(f"PASS: {len(report['trials'])} controlled trials and {len(report['physical'])} FDFD sweeps", flush=True)


if __name__ == "__main__":
    main()
