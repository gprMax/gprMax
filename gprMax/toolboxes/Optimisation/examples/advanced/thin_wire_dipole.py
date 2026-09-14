"""Advanced dipole CLI benchmark: profiles, checkpoints and comparison plots.

For the editable two-function user example, see examples/thin_wire_dipole.py.
"""

import math
from dataclasses import asdict
from pathlib import Path

from gprMax.toolboxes.Optimisation import (
    Campaign,
    Integer,
    LocalExecutor,
    ObjectiveResult,
    ParameterSpace,
    PreparedModel,
    Problem,
    Scenario,
    read_port,
)
from gprMax.toolboxes.Optimisation._storage import write_json
from gprMax.toolboxes.Optimisation.adapters import OPTIMISERS, make_optimiser

C0 = 299792458.0


# FIXED EXPERIMENT SETTINGS. The optimiser only changes arm_cells.
def settings(
    target_hz=1e9,
    *,
    cells_per_wavelength=100,
    cycles=24,
    resistance=50.0,
    padding_cells=16,
    axial_refinement=1,
    radius_m=None,
    dz_m=None,
):
    """Derive fixed mesh/domain settings and arm-cell bounds for this example.

    target_hz sets the nominal wavelength; cycles/target_hz is the time window
    in seconds. dl is transverse spacing; dz is axial spacing. Bounds apply
    to one arm, while total length is (2 * arm_cells + 1) * dz including the
    one-cell feed gap. These keys are this model's choices, not toolbox fields.
    """
    target_hz, resistance = float(target_hz), float(resistance)
    if (
        not math.isfinite(target_hz)
        or target_hz <= 0
        or not math.isfinite(resistance)
        or resistance <= 0
    ):
        raise ValueError("Frequency and resistance must be positive and finite")
    if any(
        isinstance(x, bool) or not isinstance(x, int)
        for x in (cells_per_wavelength, cycles, padding_cells, axial_refinement)
    ):
        raise ValueError("Mesh resolution, cycles and padding must be integers")
    if cells_per_wavelength < 80 or cycles < 12 or padding_cells < 12 or axial_refinement < 1:
        raise ValueError("Use at least 80 cells/wavelength, 12 periods and 12 padding cells")
    dl = C0 / target_hz / cells_per_wavelength
    radius = 0.2 * dl if radius_m is None else float(radius_m)
    if not math.isfinite(radius) or not 0 < radius < 0.5 * dl:
        raise ValueError(
            "Wire radius must be positive and smaller than half the transverse cell size"
        )
    if dz_m is not None and axial_refinement != 1:
        raise ValueError("Set either dz_m or axial_refinement, not both")
    dz = dl / axial_refinement if dz_m is None else float(dz_m)
    if not math.isfinite(dz) or dz <= 0 or dz > dl:
        raise ValueError("dz_m must be positive, finite and no coarser than the transverse spacing")
    axial_cells = C0 / target_hz / dz
    return {
        "target_hz": target_hz,
        "dl": dl,
        "dz": dz,
        "axial_refinement": axial_refinement,
        "radius": radius,
        "resistance": resistance,
        "cycles": cycles,
        "lower": math.ceil((0.36 * axial_cells - 1) / 2),
        "upper": math.floor((0.56 * axial_cells - 1) / 2),
        "padding_cells": padding_cells,
        "pml_cells": 8,
        "axial_padding_cells": math.ceil(padding_cells * dl / dz - 1e-12),
        "axial_pml_cells": math.ceil(8 * dl / dz - 1e-12),
    }


# MODEL BUILDING. Same model contract regardless of optimiser choice.
def build_model(parameters, scenario, context):
    """Build the proposed arm-cell count in the scenario's fixed experiment.

    effective_parameters records physical lengths and mesh spacing;
    metadata records the feed and target used later by the objective.
    The worker executes the returned PreparedModel.
    """
    import gprMax

    config = scenario.settings
    n, dl, dz = parameters["arm_cells"], config["dl"], config["dz"]
    # Fixed domain for the entire search. The central one-cell feed gap is
    # never occupied by ThinWire (its electric edges are PEC).
    transverse = 2 * (config["padding_cells"] + 4)
    centre_z = config["upper"] + config["axial_padding_cells"]
    x = y = (transverse // 2) * dl
    z = centre_z * dz
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(transverse * dl, transverse * dl, (2 * centre_z + 1) * dz)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dz)))
    scene.add(gprMax.TimeWindow(time=config["cycles"] / config["target_hz"]))
    pml = config["pml_cells"]
    scene.add(
        gprMax.PMLThickness(
            thickness=(pml, pml, config["axial_pml_cells"], pml, pml, config["axial_pml_cells"])
        )
    )
    scene.add(
        gprMax.ThinWire(p1=(x, y, (centre_z - n) * dz), p2=(x, y, z), radius=config["radius"])
    )
    scene.add(
        gprMax.ThinWire(
            p1=(x, y, (centre_z + 1) * dz),
            p2=(x, y, (centre_z + 1 + n) * dz),
            radius=config["radius"],
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=config["target_hz"], id="pulse"))
    scene.add(
        gprMax.VoltageSource(
            p1=(x, y, z),
            polarisation="z",
            resistance=config["resistance"],
            waveform_id="pulse",
            id="feed",
        )
    )
    return PreparedModel(
        scene,
        effective_parameters={
            "arm_cells": n,
            "arm_length_m": n * dz,
            "total_length_m": (2 * n + 1) * dz,
            "gap_m": dz,
            "radius_m": config["radius"],
            "dx_m": dl,
            "dy_m": dl,
            "dz_m": dz,
        },
        metadata={
            "port": "feed",
            "target_hz": config["target_hz"],
            "reference_ohm": config["resistance"],
            "frequency_tolerance_hz": config.get("frequency_tolerance_hz"),
        },
    )


# OUTPUT PROCESSING. Helpers below support the two selectable objectives.
def score_spectrum(spectrum, target_hz):
    """Match at target; report the discrete S11 dip and nearby reactance zero separately."""
    import numpy as np

    value = spectrum.at(target_hz)
    if np.isnan(spectrum.tail_relative_db) or spectrum.tail_relative_db > -40:
        raise ValueError(
            "Terminal trace has not demonstrably decayed below -40 dB; increase --cycles"
        )
    band = (spectrum.frequency >= 0.6 * target_hz) & (spectrum.frequency <= 1.4 * target_hz)
    eligible = np.flatnonzero(band & spectrum.valid)
    if not len(eligible):
        raise ValueError("No valid S11 bins in the diagnostic frequency band")
    index = eligible[np.argmin(np.abs(spectrum.s11[eligible]))]
    crossings = []
    for i in range(len(spectrum.frequency) - 1):
        if not (
            band[i]
            and band[i + 1]
            and spectrum.impedance_valid[i]
            and spectrum.impedance_valid[i + 1]
        ):
            continue
        a, b = spectrum.impedance[i].imag, spectrum.impedance[i + 1].imag
        if a == 0:
            crossings.append(float(spectrum.frequency[i]))
        elif a * b < 0:
            crossings.append(
                float(
                    spectrum.frequency[i]
                    - a * (spectrum.frequency[i + 1] - spectrum.frequency[i]) / (b - a)
                )
            )
    power = abs(value) ** 2
    return {
        "reflected_power": power,
        "s11_target_db": float(10 * np.log10(max(power, np.finfo(float).tiny))),
        "s11_minimum_hz": float(spectrum.frequency[index]),
        "s11_minimum_at_valid_band_edge": bool(index in (eligible[0], eligible[-1])),
        "reactance_zero_hz": min(crossings, key=lambda f: abs(f - target_hz))
        if crossings
        else None,
        "frequency_bin_hz": spectrum.independent_frequency_resolution_hz
        or float(np.median(np.diff(spectrum.frequency))),
        "tail_relative_db": spectrum.tail_relative_db
        if np.isfinite(spectrum.tail_relative_db)
        else None,
    }


def evaluate_s11(parameters, runs):
    """User-owned objective; independent of the optimiser and campaign loop."""
    run = runs["free_space"]
    target_hz = run.record["model_metadata"]["target_hz"]
    metrics = score_spectrum(read_port(run.output_file, "feed"), target_hz)
    metrics.update(run.record["effective_parameters"])
    print(
        f"candidate={run.candidate_id} arm_cells={parameters['arm_cells']:2d} "
        f"length={metrics['total_length_m'] * 1e3:.3f} mm S11(target)={metrics['s11_target_db']:.2f} dB",
        flush=True,
    )
    return ObjectiveResult(metrics["reflected_power"], metrics)


def resonance_metrics(spectrum, target_hz, tolerance_hz):
    """Estimate an interior S11 dip and score its frequency error in Hz.

    Fit power at three neighbouring bins. Report both the fit and the sampled
    minimum, and require adequate independent resolution and a clear dip.
    This estimates frequency within the numerical model, not physical accuracy.
    """
    import numpy as np

    metrics = score_spectrum(spectrum, target_hz)
    resolution = spectrum.independent_frequency_resolution_hz
    if resolution is None or resolution > tolerance_hz * (1 + 1e-6):
        raise ValueError(
            "Independent spectral resolution must be no coarser than the requested tolerance; increase --cycles"
        )
    band = (spectrum.frequency >= 0.6 * target_hz) & (spectrum.frequency <= 1.4 * target_hz)
    if not spectrum.valid[band].all():
        raise ValueError("Resonance search band contains invalid S11 bins")
    index = int(np.argmin(np.abs(spectrum.frequency - metrics["s11_minimum_hz"])))
    if (
        metrics["s11_minimum_at_valid_band_edge"]
        or index == 0
        or index == len(spectrum.frequency) - 1
    ):
        raise ValueError("S11 minimum is not bracketed inside the search band")
    f = spectrum.frequency[index - 1 : index + 2].astype(float)
    power = np.abs(spectrum.s11[index - 1 : index + 2].astype(complex)) ** 2
    if not (power[1] < power[0] and power[1] < power[2]) or power[1] > 0.1:
        raise ValueError("Require a resolved interior S11 dip below -10 dB")
    # Scaling prevents ill-conditioned polynomial fits to GHz-valued axes.
    x = (f - f[1]) / resolution
    a, b, c = np.polyfit(x, power, 2)
    vertex = -b / (2 * a) if a > 0 else float("nan")
    if not np.isfinite(vertex) or not x[0] < vertex < x[2]:
        raise ValueError("Invalid local S11-minimum fit")
    resonance = float(f[1] + vertex * resolution)
    metrics.update(
        objective="s11_minimum_frequency_error",
        objective_unit="Hz",
        resonance_frequency_hz=resonance,
        frequency_error_hz=abs(resonance - target_hz),
        signed_frequency_error_hz=resonance - target_hz,
        frequency_tolerance_hz=tolerance_hz,
        sampled_minimum_db=float(10 * np.log10(power[1])),
        resonance_method="three_bin_quadratic_power",
        fit_bracket_hz=[float(f[0]), float(f[2])],
    )
    return metrics


def evaluate_resonance(parameters, runs):
    """User objective locating the S11 dip; independent of the optimiser loop."""
    run = runs["free_space"]
    metadata = run.record["model_metadata"]
    metrics = resonance_metrics(
        read_port(run.output_file, "feed"),
        metadata["target_hz"],
        metadata["frequency_tolerance_hz"],
    )
    metrics.update(run.record["effective_parameters"])
    print(
        f"candidate={run.candidate_id} length={metrics['total_length_m'] * 1e3:.3f} mm "
        f"S11 dip={metrics['resonance_frequency_hz']/1e9:.6f} GHz "
        f"error={metrics['frequency_error_hz']/1e6:.3f} MHz",
        flush=True,
    )
    return ObjectiveResult(metrics["frequency_error_hz"], metrics)


# CAMPAIGN SETUP. Connect parameters, model, objective and stopping settings.
def optimise(
    directory,
    *,
    target_hz=1e9,
    cells_per_wavelength=100,
    cycles=24,
    resistance=50.0,
    cpu_threads=1,
    n_trials=12,
    optimiser_seed=0,
    objective="match",
    frequency_tolerance_hz=10e6,
    axial_refinement=1,
    radius_m=None,
    dz_m=None,
    optimiser="tpe",
    population_size=6,
    fixed_budget=False,
    execution=None,
    batch_size=1,
    checkpoint=False,
    resume=False,
    cache=None,
    max_simulations=None,
):
    """Configure the advanced dipole campaign, choose its criterion and run it.

    objective="match" minimises reflected power at the target frequency;
    "resonance" minimises the estimated S11-dip frequency error in Hz.
    frequency_tolerance_hz sets target stopping for the resonance case.
    fixed_budget continues after reaching that target. Other options select
    an adapter, execution backend, cache or explicit checkpoint recovery.

    This benchmark wrapper also writes history/plots. The shorter editable
    example in ../thin_wire_dipole.py is the normal user starting point.
    """
    if objective not in ("match", "resonance"):
        raise ValueError("objective must be match or resonance")
    if not math.isfinite(frequency_tolerance_hz) or frequency_tolerance_hz <= 0:
        raise ValueError("frequency_tolerance_hz must be positive and finite")
    config = settings(
        target_hz,
        cells_per_wavelength=cells_per_wavelength,
        cycles=cycles,
        resistance=resistance,
        axial_refinement=axial_refinement,
        radius_m=radius_m,
        dz_m=dz_m,
    )
    config["frequency_tolerance_hz"] = frequency_tolerance_hz
    if objective == "resonance" and target_hz / cycles > frequency_tolerance_hz * (1 + 1e-6):
        raise ValueError(
            "Increase --cycles so independent frequency spacing is no coarser than the tolerance"
        )
    directory = Path(directory).resolve()
    problem = Problem(
        ParameterSpace({"arm_cells": Integer(config["lower"], config["upper"])}),
        "gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole:build_model",
        scenarios=(Scenario("free_space", config),),
        dependencies=(Path(__file__),),
        version="1",
    )
    if isinstance(optimiser, str) and optimiser not in ("tpe", "rf") and batch_size != 1:
        raise ValueError("Use population_size for population algorithms; batch_size is for RF/TPE")
    options = {"batch_size": batch_size} if optimiser in ("tpe", "rf") else {}
    adapter = (
        None
        if resume
        else (
            make_optimiser(
                optimiser, seed=optimiser_seed, population_size=population_size, **options
            )
            if isinstance(optimiser, str)
            else optimiser
        )
    )
    execution = execution or LocalExecutor(cpu_threads=cpu_threads, timeout=600)
    campaign = (
        Campaign.resume(problem, directory, execution, cache=cache)
        if resume
        else Campaign(problem, directory, execution, cache=cache)
    )
    result = campaign.optimise(
        optimiser=adapter,
        evaluator="gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole:"
        + ("evaluate_resonance" if objective == "resonance" else "evaluate_s11"),
        n_trials=n_trials,
        max_simulations=n_trials if max_simulations is None else max_simulations,
        target_value=frequency_tolerance_hz if objective == "resonance" else None,
        stop_on_target=not fixed_budget,
        checkpoint=checkpoint,
        resume=resume,
    )
    if result.stop_reason == "trial_failed" or result.best is None:
        raise RuntimeError(
            f"Optimisation stopped: {result.stop_reason}; see {directory / 'optimiser/result.json'}"
        )
    from gprMax.toolboxes.Optimisation import load_candidate

    history = [
        {
            "trial_token": trial.token,
            "candidate_id": trial.candidate_id,
            "output_file": str(
                load_candidate(directory / "candidates" / trial.candidate_id)[1][
                    "free_space"
                ].output_file.relative_to(directory)
            ),
            **trial.metrics,
        }
        for trial in result.trials
        if trial.status == "complete"
    ]
    write_json(directory / "search_history.json", history)
    summary = {
        "target_hz": target_hz,
        "reference_ohm": resistance,
        "settings": config,
        "objective": objective,
        "initial": history[0],
        "best": next(row for row in history if row["trial_token"] == result.best.token),
        "search": asdict(result),
        "length_increment_m": 2 * config["dz"],
        "note": "External optimiser proposals with objective feedback; first trial is the plotted initial candidate. Finite-budget demonstration, no global guarantee.",
    }
    write_json(directory / "optimisation.json", summary)
    plot_results(directory, summary, history)
    return summary


# REPORTING HELPER. Consumes saved results; does not influence the search.
def plot_results(directory, summary, history):
    """Plot the recorded objective history and initial/best spectra without running new simulations."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    trials = np.arange(1, len(history) + 1)
    resonance_mode = summary.get("objective") == "resonance"
    db_history = [
        r["frequency_error_hz"] / 1e6 if resonance_mode else r["s11_target_db"] for r in history
    ]
    axes[0].plot(trials, db_history, "o", label="trial")
    axes[0].step(trials, np.minimum.accumulate(db_history), where="post", label="best so far")
    axes[0].set(
        xlabel="Trial (proposal order)",
        ylabel="S11-dip frequency error (MHz)" if resonance_mode else "S11 at target (dB)",
    )
    if resonance_mode:
        tolerance = summary["settings"]["frequency_tolerance_hz"]
        axes[0].axhline(tolerance / 1e6, color="green", linestyle=":", label="target tolerance")
    axes[0].legend()
    for label in ("initial", "best"):
        row = summary[label]
        spectrum = read_port(directory / row["output_file"], "feed")
        valid = (
            spectrum.valid
            & (spectrum.frequency >= 0.6 * summary["target_hz"])
            & (spectrum.frequency <= 1.4 * summary["target_hz"])
        )
        # NaNs preserve any invalid gaps rather than joining through them.
        db = np.where(
            valid, 20 * np.log10(np.maximum(np.abs(spectrum.s11), np.finfo(float).tiny)), np.nan
        )
        axes[1].plot(
            spectrum.frequency / 1e9, db, label=f"{label}: {row['total_length_m'] * 1e3:.1f} mm"
        )
    axes[1].axvline(summary["target_hz"] / 1e9, color="black", linestyle="--", label="target")
    if resonance_mode:
        target = summary["target_hz"]
        axes[1].axvspan(
            (target - tolerance) / 1e9,
            (target + tolerance) / 1e9,
            color="green",
            alpha=0.12,
            label="tolerance",
        )
        axes[1].axvline(
            summary["best"]["resonance_frequency_hz"] / 1e9,
            color="tab:orange",
            linestyle=":",
            label="estimated dip",
        )
    axes[1].set(
        xlabel="Frequency (GHz)",
        ylabel="S11 (dB)",
        xlim=(0.6 * summary["target_hz"] / 1e9, 1.4 * summary["target_hz"] / 1e9),
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "dipole_optimisation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New campaign output directory")
    parser.add_argument("--target-hz", type=float, default=1e9)
    parser.add_argument("--cells-per-wavelength", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=24)
    parser.add_argument("--resistance", type=float, default=50)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--optimiser-seed", type=int, default=0)
    parser.add_argument("--objective", choices=("match", "resonance"), default="match")
    parser.add_argument("--frequency-tolerance-hz", type=float, default=10e6)
    parser.add_argument("--axial-refinement", type=int, default=1)
    parser.add_argument("--radius-m", type=float)
    parser.add_argument("--dz-m", type=float, help="Explicit axial cell size in metres")
    parser.add_argument("--optimiser", choices=OPTIMISERS, default="tpe")
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument(
        "--execution-profile", type=Path, help="JSON local or MPI execution configuration"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Synchronous RF/TPE proposal count"
    )
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-simulations", type=int)
    parser.add_argument(
        "--fixed-budget",
        action="store_true",
        help="Continue after target success to inspect feedback-driven proposals",
    )
    arguments = vars(parser.parse_args())
    profile = arguments.pop("execution_profile")
    if profile is None:
        outcome = optimise(**arguments)
    else:
        from gprMax.toolboxes.Optimisation import execution_from_profile

        with execution_from_profile(profile) as execution:
            outcome = optimise(**arguments, execution=execution) if execution is not None else None
    if outcome is None:
        raise SystemExit(0)
    print(f"Best evaluated total length: {outcome['best']['total_length_m'] * 1e3:.3f} mm")
    if outcome["objective"] == "resonance":
        print(
            f"Stop reason: {outcome['search']['stop_reason']}; target met: {outcome['search']['target_met']}"
        )
