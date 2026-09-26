"""Physical pulse and late-time checks for an SIBC continued through PML.

Run ``python -m testing.validation.impedance_surface.validate_sibc_pml``.
The reference is a longer *identical* guide: its far termination cannot
causally return within the comparison record. No PMC adapter or replacement
field equation is used. CPUUpdates exercises the production PML/SIBC order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import gprMax
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.updates.cpu_updates import CPUUpdates

RESULTS = Path(__file__).resolve().parent / "results" / "sibc_pml"
DL = 1e-3
ETA = np.sqrt(mu_0 / epsilon_0)
FIELD_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def add_bulk_host(scene, kind):
    """Lossy epsilon-infinity host with optional real or complex bulk poles."""
    scene.add(gprMax.DispersiveAveraging(enabled=True))
    scene.add(gprMax.Material(er=3, se=0.02, mr=1, sm=0, id="host"))
    common = dict(poles=2, material_ids=["host"])
    if kind == "debye":
        scene.add(gprMax.AddDebyeDispersion(er_delta=[2., 1.], tau=[3e-11, 8e-11], **common))
    elif kind == "lorentz":
        scene.add(gprMax.AddLorentzDispersion(
            er_delta=[2., 1.], omega=[8e9, 12e9], delta=[2e9, 3e9], **common,
        ))
    elif kind == "drude":
        scene.add(gprMax.AddDrudeDispersion(omega=[5e9, 9e9], alpha=[4e9, 6e9], **common))
    elif kind != "lossy":
        raise ValueError(kind)


def surface(kind):
    if kind == "pec":
        return None
    if kind == "resistance":
        return gprMax.SurfaceImpedance(id="wall", resistance=5.0)
    if kind == "high_resistance":
        return gprMax.SurfaceImpedance(id="wall", resistance=1e6)
    if kind == "foster":
        return gprMax.SurfaceImpedance(
            id="wall",
            conductivity=1e4,
            fit_frequency_range=(1e9, 200e9),
            fit_order=4,
            fit_tolerance=0.03,
        )
    raise ValueError(kind)


def permutation(axis):
    """Map canonical (transverse x, transverse y, propagation z) to xyz."""
    return ((axis + 1) % 3, (axis + 2) % 3, axis)


def xyz(values, axis):
    result = np.zeros(3, dtype=float)
    result[list(permutation(axis))] = values
    return tuple(result)


def scene_for(
    length,
    *,
    kind="resistance",
    axis=2,
    formulation="HORIPML",
    pml_cells=16,
    order=1,
    duplicated_unshifted=False,
    host_kind=None,
    layered=False,
):
    scene = gprMax.Scene()
    thickness = [0] * 6
    thickness[axis] = thickness[axis + 3] = pml_cells
    for obj in (
        gprMax.Domain(p1=xyz(np.asarray((16, 12, length)) * DL, axis)),
        gprMax.Discretisation(p1=(DL, DL, DL)),
        gprMax.TimeWindow(iterations=1),
        gprMax.OMPThreads(1),
        gprMax.PMLThickness(thickness=tuple(thickness)),
        gprMax.PMLFormulation(formulation=formulation),
        surface(kind),
    ):
        if obj is not None:
            scene.add(obj)
    if kind == "pec":
        scene.add(gprMax.TimeStepStabilityFactor(f=0.99))
    if order == 2:
        # HORIPML-2 uses the existing published profile from
        # testing/models_pmls/pml_3D_pec_plate/pml_3D_pec_plate.py. Simply
        # duplicating an unshifted first-order profile is a poor late-time
        # absorber for this guide. MRIPML adds its two entrance stretches,
        # so each of the two equal terms has kappa=0.5.
        if duplicated_unshifted:
            profiles = (("constant", 0.0, 0.0, "constant", 1.0, 1.0, "quartic", None),) * 2
        elif formulation == "HORIPML":
            sigma1 = 0.275 / (150 * np.pi * DL)
            profiles = (
                ("constant", 0.0, 0.0, "constant", 1.0, 1.0, "sextic", sigma1),
                (
                    "sextic",
                    0.07,
                    0.07 + sigma1,
                    "cubic",
                    1.0,
                    8.0,
                    "quadratic",
                    2.75 / (150 * np.pi * DL),
                ),
            )
        else:
            profiles = (("constant", 0.0, 0.0, "constant", 0.5, 0.5, "quartic", None),) * 2
        for alpha, amin, amax, kprofile, kmin, kmax, sigma, smax in profiles:
            scene.add(
                gprMax.PMLCFS(
                    alphascalingprofile=alpha,
                    alphascalingdirection="forward",
                    alphamin=amin,
                    alphamax=amax,
                    kappascalingprofile=kprofile,
                    kappascalingdirection="forward",
                    kappamin=kmin,
                    kappamax=kmax,
                    sigmascalingprofile=sigma,
                    sigmascalingdirection="forward",
                    sigmamin=0.0,
                    sigmamax=smax,
                )
            )
    if host_kind is not None:
        add_bulk_host(scene, host_kind)
        scene.add(gprMax.Box(
            p1=(0, 0, 0),
            p2=xyz(np.asarray((8 if layered else 16, 12, length)) * DL, axis),
            material_id="host",
        ))
    for lower, upper in (
        ((3, 3, 0), (4, 9, length)),
        ((12, 3, 0), (13, 9, length)),
        ((4, 3, 0), (12, 4, length)),
        ((4, 8, 0), (12, 9, length)),
    ):
        scene.add(
            gprMax.Box(
                p1=xyz(np.asarray(lower) * DL, axis),
                p2=xyz(np.asarray(upper) * DL, axis),
                material_id="pec" if kind == "pec" else "wall",
                averaging="n",
            )
        )
    return scene


def build_grid(scene, path, precision="double"):
    captured = []
    original = FDTDGrid.build

    def capture(grid):
        original(grid)
        captured.append(grid)

    with patch.object(FDTDGrid, "build", capture):
        gprMax.run(
            scenes=[scene],
            outputfile=path,
            geometry_only=True,
            cpu_precision=precision,
            hide_progress_bars=True,
            log_level=40,
        )
    return captured[0]


def seed_pulse(grid, axis=2):
    """Initial smooth TE-like packet; H=0 launches both guide directions."""
    field = getattr(grid, FIELD_NAMES[permutation(axis)[1]])
    xx, yy, zz = np.meshgrid(
        np.arange(4, 13), np.arange(4, 8), np.arange(field.shape[axis]), indexing="ij"
    )
    envelope = np.exp(-(((zz - 40) / 6.0) ** 2)) * np.cos(0.7 * (zz - 40))
    amplitude = np.sin(np.pi * (xx - 4) / 8) * envelope
    coords = [None] * 3
    for physical, index in zip(permutation(axis), (xx, yy, zz)):
        coords[physical] = index
    field[tuple(coords)] = amplitude


def advance(updates, *, omit_boundary_pml=False):
    updates.update_magnetic()
    updates.update_magnetic_pml()
    updates.update_electric_a()
    updates.update_electric_pml()
    if omit_boundary_pml:
        # Deliberately faulty validation control: histories advance, but the
        # sparse boundary discards their stretched-derivative forcing.
        updates.grid.impedance_surfaces._pending_pml_rhs = None
    updates.update_electric_b()
    updates.update_impedance_surfaces()


def field_norm(grid):
    return sum(
        float(np.sum(np.square(getattr(grid, name), dtype=np.float64)))
        * (1 if index < 3 else ETA**2)
        for index, name in enumerate(FIELD_NAMES)
    )


def run_packet(
    cache,
    *,
    length=120,
    kind="resistance",
    axis=2,
    formulation="HORIPML",
    order=1,
    precision="double",
    steps=600,
    omit_boundary_pml=False,
    host_kind=None,
    layered=False,
):
    name = f"{kind}_{axis}_{formulation}_{order}_{precision}_{length}_{host_kind}_{layered}"
    grid = build_grid(
        scene_for(length, kind=kind, axis=axis, formulation=formulation, order=order,
                  host_kind=host_kind, layered=layered),
        cache / name,
        precision,
    )
    seed_pulse(grid, axis)
    updates = CPUUpdates(grid)
    if grid.maxpoles:
        updates.set_dispersive_updates()
    probe = tuple(int(x) for x in xyz((8, 6, 64), axis))
    field = getattr(grid, FIELD_NAMES[permutation(axis)[1]])
    trace = np.empty(steps)
    energy = np.empty(steps)
    initial_norm = field_norm(grid)
    max_state = 0.0
    for step in range(steps):
        trace[step] = field[probe]
        energy[step] = field_norm(grid) / initial_norm
        advance(updates, omit_boundary_pml=omit_boundary_pml)
        if grid.impedance_surfaces.state_y.size:
            max_state = max(max_state, float(np.max(np.abs(grid.impedance_surfaces.state_y))))
    pml_history = max(
        float(np.max(np.abs(getattr(pml, name))))
        for pml in grid.pmls["slabs"]
        for name in ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
    )
    return grid, trace, energy, max_state, pml_history


def compare_packet(cache, **case):
    short, trace, energy, state_peak, pml_history = run_packet(cache, **case)
    reference, reference_trace, _, _, _ = run_packet(cache, length=320, **case)
    if short.dt != reference.dt:
        raise AssertionError("Reference and test must use identical timesteps")
    peak = float(np.max(np.abs(reference_trace)))
    delta = trace - reference_trace
    # A causal front travels no faster than c. Initial Gaussian tails are
    # negligible at the PML entrance (64 cells from the packet centre).
    earliest_return = ((120 - 16 - 40) + (120 - 16 - 64)) * DL / c
    early = np.arange(trace.size) * short.dt < 0.9 * earliest_return
    reference_return = ((320 - 16 - 40) + (320 - 16 - 64)) * DL / c
    metrics = dict(
        case,
        dt_s=short.dt,
        timestep_factor=short.dt / (DL / (c * np.sqrt(3))),
        reference_return_ns=reference_return * 1e9,
        record_ns=trace.size * short.dt * 1e9,
        peak_reference=peak,
        early_relative_error=float(np.max(np.abs(delta[early]))) / peak,
        relative_reflection_peak=float(np.max(np.abs(delta))) / peak,
        relative_reflection_l2=float(np.linalg.norm(delta) / np.linalg.norm(reference_trace)),
        final_field_norm=float(energy[-1]),
        peak_field_norm=float(np.max(energy)),
        surface_state_peak=state_peak,
        pml_history_final_max=pml_history,
        pml_edge_count=int(getattr(short.impedance_surfaces, "pml_edge_count", 0)),
    )
    metrics["passed"] = bool(
        metrics["record_ns"] < metrics["reference_return_ns"]
        and metrics["early_relative_error"] < 1e-5
        and metrics["relative_reflection_peak"] < 0.01
        and metrics["final_field_norm"] < 0.1
        and np.all(np.isfinite(energy))
        and (case["kind"] != "foster" or state_peak > 0)
    )
    return metrics, np.column_stack(
        (np.arange(trace.size) * short.dt, trace, reference_trace, delta, energy)
    )


def stability_case(
    cache,
    *,
    kind,
    precision,
    steps=20000,
    formulation="HORIPML",
    order=1,
    duplicated_unshifted=False,
):
    """Continue a packet plus weak random high-frequency seeds, source free."""
    grid = build_grid(
        scene_for(
            72,
            kind=kind,
            formulation=formulation,
            order=order,
            duplicated_unshifted=duplicated_unshifted,
        ),
        cache / f"late_{kind}_{precision}_{formulation}_{order}_{duplicated_unshifted}",
        precision,
    )
    seed_pulse(grid)
    rng = np.random.default_rng(718)
    interior = (slice(5, 12), slice(5, 8), slice(18, 54))
    for index, name in enumerate(FIELD_NAMES):
        field = getattr(grid, name)
        field[interior] += (
            rng.standard_normal(field[interior].shape) * 1e-3 / (1 if index < 3 else ETA)
        )
    updates = CPUUpdates(grid)
    initial_norm = field_norm(grid)
    samples = []
    finite = True
    peak_state = 0.0
    for step in range(steps):
        advance(updates)
        if step % 50 == 0 or step == steps - 1:
            norm = field_norm(grid) / initial_norm
            state = (
                grid.impedance_surfaces.state_y
                if grid.impedance_surfaces is not None
                else np.empty(0)
            )
            state_max = float(np.max(np.abs(state))) if state.size else 0.0
            peak_state = max(peak_state, state_max)
            finite = finite and np.isfinite(norm) and np.all(np.isfinite(state))
            finite = finite and all(
                np.all(np.isfinite(getattr(pml, name)))
                for pml in grid.pmls["slabs"]
                for name in ("EPhi1", "EPhi2", "HPhi1", "HPhi2")
            )
            samples.append(((step + 1) * grid.dt, norm, state_max))
            if not finite or norm > 1e10:
                break
    data = np.asarray(samples)
    completed = int(round(data[-1, 0] / grid.dt))
    metrics = dict(
        kind=kind,
        precision=precision,
        formulation=formulation,
        order=order,
        steps=completed,
        requested_steps=steps,
        dt_s=grid.dt,
        duration_ns=float(data[-1, 0] * 1e9),
        peak_field_norm=float(np.max(data[:, 1])),
        final_field_norm=float(data[-1, 1]),
        surface_state_peak=peak_state,
        finite=bool(finite),
    )
    metrics["passed"] = bool(
        finite
        and completed == steps
        and metrics["peak_field_norm"] < 2
        and metrics["final_field_norm"] < 0.1
        and (kind != "foster" or peak_state > 0)
    )
    return metrics, data


def profile_audit(output, steps=20000):
    """Retain the rejected arbitrary profile and compare ordinary PEC walls."""
    results = []
    for kind in ("resistance", "pec"):
        metrics, data = stability_case(
            output / "_cache",
            kind=kind,
            precision="double",
            order=2,
            duplicated_unshifted=True,
            steps=steps,
        )
        np.savetxt(
            output / f"duplicated_unshifted_{kind}.csv",
            data,
            delimiter=",",
            header="time_s,normalised_field_norm,surface_state_max",
            comments="",
        )
        results.append(metrics)
        print(json.dumps(dict(profile="duplicated_unshifted", **metrics)), flush=True)
    (output / "profile_audit.json").write_text(
        json.dumps(
            {
                "profile": "Two identical HORIPML factors, alpha=0, kappa=1, default quartic sigma",
                "purpose": "Determine whether the observed late residual also occurs with ordinary PEC walls",
                "cases": results,
            },
            indent=2,
        )
        + "\n"
    )
    return results


def omitted_pml_control(cache):
    """Check that discarding only the boundary PML forcing is observable."""
    # A large finite impedance keeps Et on the wall appreciable, making this
    # control sensitive to omitted boundary forcing. At 5 ohm Et is small and
    # bulk PML alone can hide the missing boundary term in a probe trace.
    options = dict(kind="high_resistance")
    grid, faulty, _, _, _ = run_packet(cache, omit_boundary_pml=True, **options)
    _, correct, _, _, _ = run_packet(cache, **options)
    _, reference, _, _, _ = run_packet(cache, length=320, **options)
    peak = np.max(np.abs(reference))
    correct_error = float(np.max(np.abs(correct - reference)) / peak)
    faulty_error = float(np.max(np.abs(faulty - reference)) / peak)
    return dict(
        resistance_ohm=1e6,
        correct_relative_reflection_peak=correct_error,
        omitted_relative_reflection_peak=faulty_error,
        error_increase=faulty_error / correct_error,
        passed=bool(faulty_error > 1e-3 and faulty_error > 5 * correct_error),
    ), np.column_stack((np.arange(len(correct)) * grid.dt, correct, faulty, reference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stability-only", action="store_true")
    parser.add_argument("--profile-audit-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "_cache"
    cache.mkdir(exist_ok=True)
    if args.profile_audit_only:
        profile_audit(args.output)
        return
    cases = [
        dict(kind=kind, axis=axis, formulation=formulation, order=order, precision=precision)
        for kind in ("resistance", "foster")
        for axis, formulation, order, precision in (
            (2, "HORIPML", 1, "double"),
            (0, "HORIPML", 1, "double"),
            (1, "MRIPML", 1, "double"),
            (2, "MRIPML", 2, "double"),
            (2, "HORIPML", 2, "double"),
            (2, "HORIPML", 1, "single"),
        )
    ]
    if args.quick:
        cases = cases[:1]
    previous = args.output / "summary.json"
    summaries = json.loads(previous.read_text())["pulse_cases"] if args.stability_only else []
    for case in [] if args.stability_only else cases:
        metrics, data = compare_packet(cache, **case)
        name = "_".join(str(case[k]) for k in ("kind", "axis", "formulation", "order", "precision"))
        np.savetxt(
            args.output / f"{name}.csv",
            data,
            delimiter=",",
            header="time_s,short_E,long_reference_E,difference_E,normalised_field_norm",
            comments="",
        )
        summaries.append(metrics)
        print(json.dumps(metrics), flush=True)
    stability = []
    if not args.quick:
        late_cases = [
            dict(kind=kind, precision=precision)
            for kind in ("resistance", "foster")
            for precision in ("double", "single")
        ]
        late_cases += [
            dict(kind="foster", precision=precision, formulation=formulation, order=2)
            for formulation, precision in (("HORIPML", "double"), ("MRIPML", "single"))
        ]
        for case in late_cases:
            metrics, data = stability_case(cache, **case)
            name = f"late_{metrics['kind']}_{metrics['precision']}_{metrics['formulation']}_{metrics['order']}"
            np.savetxt(
                args.output / f"{name}.csv",
                data,
                delimiter=",",
                header="time_s,normalised_field_norm,surface_state_max",
                comments="",
            )
            stability.append(metrics)
            print(json.dumps(metrics), flush=True)
        control, data = omitted_pml_control(cache)
        np.savetxt(
            args.output / "omitted_pml_control.csv",
            data,
            delimiter=",",
            header="time_s,correct_E,omitted_boundary_pml_E,reference_E",
            comments="",
        )
    else:
        control = None
    model = surface("foster")
    payload = {
        "description": __doc__,
        "pulse_cases": summaries,
        "stability_cases": stability,
        "negative_control": control,
        "foster_model": dict(
            conductivity_s_per_m=model.conductivity,
            fit_band_hz=model.fit_frequency_range,
            A=np.asarray(model.A).tolist(),
            B=np.asarray(model.B).tolist(),
            C=np.asarray(model.C).tolist(),
            D=model.D,
        ),
        "passed": all(item["passed"] for item in summaries + stability)
        and (control is None or control["passed"]),
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot_results(args.output)
    if not payload["passed"]:
        raise SystemExit("SIBC/PML validation failed; inspect summary.json")


def plot_results(output):
    payload = json.loads((output / "summary.json").read_text())
    summaries = payload["pulse_cases"]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for case in summaries:
        name = "_".join(str(case[k]) for k in ("kind", "axis", "formulation", "order", "precision"))
        data = np.loadtxt(output / f"{name}.csv", delimiter=",", skiprows=1)
        axes[0].plot(data[:, 0] * 1e9, data[:, 3] / case["peak_reference"], label=name)
        axes[1].semilogy(data[:, 0] * 1e9, data[:, 4], label=name)
    axes[0].set(ylabel="Probe difference / reference peak", xlabel="Time (ns)")
    axes[1].set(ylabel="Normalised squared field norm", xlabel="Time (ns)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.savefig(output / "sibc_pml_pulse.png", dpi=160)
    plt.close(fig)
    if payload.get("stability_cases"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for case in payload["stability_cases"]:
            name = f"late_{case['kind']}_{case['precision']}_{case['formulation']}_{case['order']}"
            data = np.loadtxt(output / f"{name}.csv", delimiter=",", skiprows=1)
            axes[0].semilogy(data[:, 0] * 1e9, data[:, 1], label=name.removeprefix("late_"))
        axes[0].set(
            title="Validated profiles: source-free late time",
            xlabel="Time (ns)",
            ylabel="Normalised squared field norm",
        )
        axes[0].legend(fontsize=6)
        audit = output / "profile_audit.json"
        if audit.exists():
            for case in json.loads(audit.read_text())["cases"]:
                data = np.loadtxt(
                    output / f"duplicated_unshifted_{case['kind']}.csv", delimiter=",", skiprows=1
                )
                axes[1].semilogy(data[:, 0] * 1e9, data[:, 1], label=case["kind"])
            axes[1].legend()
        axes[1].set(
            title="Rejected duplicate zero-alpha profile",
            xlabel="Time (ns)",
            ylabel="Normalised squared field norm",
        )
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.savefig(output / "sibc_pml_stability.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
