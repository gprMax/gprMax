"""Fit a dielectric block using processed receiver traces and a reference waveform.

The demo creates synthetic reference data. A user's measured waveform can use
the same evaluation interface, with explicit timing, units and preprocessing.
"""

from pathlib import Path

from gprMax.toolboxes.Optimisation import (
    Campaign,
    Evaluation,
    LocalPool,
    ObjectiveResult,
    ParameterSpace,
    Problem,
    Real,
    SkoptRF,
    read_receiver,
)


def evaluate_waveform(parameters, runs, context):
    """Interpolate the simulated trace onto reference times and minimise NRMSE."""
    import numpy as np

    settings = context.settings
    run = runs[settings.get("scenario", "default")]
    trace = read_receiver(
        run.output_file, settings.get("receiver", "received"), settings.get("component", "Ez")
    )
    with np.load(settings["reference"], allow_pickle=False) as reference:
        time = np.asarray(reference["time"], dtype=float)
        target = np.asarray(reference["values"], dtype=float)
        unit = str(reference["unit"].item())
    if (
        time.ndim != 1
        or time.size < 2
        or target.shape != time.shape
        or not np.isfinite(time).all()
        or not np.isfinite(target).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Reference needs finite, increasing time samples and matching values")
    if unit != trace.unit:
        raise ValueError(
            "Reference and simulation units must match; convert explicitly before evaluation"
        )
    lower, upper = settings.get("time_gate_s", [float(time[0]), float(time[-1])])
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("Time gate must have finite increasing endpoints")
    mask = (time >= lower) & (time <= upper)
    time, target = time[mask], target[mask]
    if time.size < 2 or time[0] < trace.time[0] or time[-1] > trace.time[-1]:
        raise ValueError(
            "Time gate needs at least two samples inside the simulation; extrapolation is disabled"
        )
    predicted = np.interp(time, trace.time, trace.values)
    # Optional offset removal is a declared modelling choice, never implicit.
    if settings.get("remove_mean", False):
        predicted = predicted - predicted.mean()
        target = target - target.mean()
    residual = predicted - target
    reference_rms = float(np.sqrt(np.mean(target**2)))
    if not np.isfinite(reference_rms) or reference_rms <= 0:
        raise ValueError("Reference RMS must be positive for normalised error")
    rmse = float(np.sqrt(np.mean(residual**2)))
    context.save_npz(
        "waveforms.npz",
        time=time,
        reference=target,
        predicted=predicted,
        residual=residual,
        units={"time": "s", "reference": unit, "predicted": unit, "residual": unit},
    )
    # Only the first argument is minimised. The dictionary records explanatory
    # metrics that can be inspected without becoming extra objectives.
    return ObjectiveResult(
        rmse / reference_rms,
        {
            "rmse": rmse,
            "reference_rms": reference_rms,
            "signal_unit": unit,
            "samples": int(time.size),
            "objective_unit": "1",
        },
    )


def run_example(directory, execution=None, optimiser=None, *, checkpoint=False):
    """Generate a synthetic reference, then fit it in a separate campaign.

    The reference file is a declared evaluator dependency. Evaluation.settings
    holds this example's file/receiver/component choices; none are fixed
    requirements of the generic processing interface. The returned result
    describes the fit, not the initial reference simulation.
    """
    import numpy as np

    from gprMax.toolboxes.Optimisation.examples import dielectric_block

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    problem = Problem(
        ParameterSpace({"width": Real(0.008, 0.016, "m"), "permittivity": Real(2, 8)}),
        "gprMax.toolboxes.Optimisation.examples.dielectric_block:build_model",
        dependencies=(Path(dielectric_block.__file__),),
    )
    execution = execution or LocalPool(workers=2)
    reference_run = Campaign(problem, directory / "reference", execution).evaluate_one(
        {"width": 0.012, "permittivity": 4}
    )[0]
    if reference_run.status != "complete":
        raise RuntimeError(f"Reference simulation failed: {reference_run.record.get('failure')}")
    trace = read_receiver(reference_run.output_file, "received", "Ez")
    reference = directory / "synthetic-reference.npz"
    np.savez(reference, time=trace.time, values=trace.values, unit=trace.unit)
    evaluator = Evaluation(
        "gprMax.toolboxes.Optimisation.examples.advanced.waveform_matching:evaluate_waveform",
        settings={"reference": str(reference), "receiver": "received", "component": "Ez"},
        dependencies=(reference,),
    )
    return Campaign(problem, directory / "fit", execution).optimise(
        optimiser=optimiser or SkoptRF(seed=7, batch_size=2),
        evaluator=evaluator,
        n_trials=8,
        max_simulations=8,
        stop_on_target=False,
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    import argparse

    from gprMax.toolboxes.Optimisation import execution_from_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--execution-profile", type=Path)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    pool = (
        execution_from_profile(args.execution_profile)
        if args.execution_profile
        else LocalPool(workers=2)
    )
    with pool as execution:
        if execution is not None:
            result = run_example(args.directory, execution=execution, checkpoint=args.checkpoint)
            print(result.stop_reason, result.best)
