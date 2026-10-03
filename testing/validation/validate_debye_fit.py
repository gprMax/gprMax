"""Validate DebyeFit against known spectra and exported hash commands.

Run from a source checkout with:
    python -m testing.validation.validate_debye_fit --output-dir /tmp/debye-fit-validation

The dense validation frequencies are independent of the fitting grid. This
checks approximation within the selected band, not extrapolation or global
optimality of the stochastic optimizer. Solver round-trip/field equivalence
is covered by tests/toolboxes/test_debye_fit_validation.py.
"""

import argparse
import contextlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.constants import epsilon_0

from gprMax.hash_cmds_file import get_user_objects
from gprMax.toolboxes.DebyeFit import Debye_Fit
from gprMax.toolboxes.DebyeFit.Debye_Fit import Crim, HavriliakNegami, Jonscher, Rawdata
from gprMax.toolboxes.DebyeFit.optimization import DA_DLS, DE_DLS, PSO_DLS

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def debye(f, epsilon=3, strength=5, tau=1e-9):
    return epsilon + strength / (1 + 2j * np.pi * f * tau)


def exported_spectrum(commands, f):
    """Decode the *emitted* commands, independently of optimizer return arrays."""
    objects = get_user_objects(commands, checkessential=False)
    base = objects[0].kwargs
    q = np.full(f.shape, base["er"], dtype=complex)
    if len(objects) == 2:
        poles = objects[1].kwargs
        for strength, tau in zip(poles["er_delta"], poles["tau"]):
            assert strength > 0 and tau > 0
            q += strength / (1 + 2j * np.pi * f * tau)
    sigma = base["se"]
    return q, q + sigma / (2j * np.pi * f * epsilon_0)


def metrics(actual, target):
    components = (
        np.array(
            [
                abs(actual.real - target.real) / (1 + abs(target.real)),
                abs(actual.imag - target.imag) / (1 + abs(target.imag)),
            ]
        )
        * 100
    )
    return float(np.mean(components, axis=1).sum()), float(components.max())


def cases():
    common = dict(
        f_min=1e6,
        f_max=1e10,
        alpha=1,
        beta=1,
        e_inf=3,
        de=5,
        tau_0=1e-9,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="fit",
        f_n=50,
        number_of_debye_poles=1,
    )
    for optimizer in (PSO_DLS, DA_DLS, DE_DLS):
        yield optimizer.__name__ + "_one_pole", HavriliakNegami(
            **common,
            optimizer=optimizer,
            optimizer_options={"seed": 111, "maxiter": 250 if optimizer is PSO_DLS else 80},
        ), debye, 0.01, 0.05
    yield "Kelley_automatic", HavriliakNegami(
        **(
            common
            | dict(
                f_min=1e7,
                f_max=1e11,
                alpha=0.91,
                beta=0.45,
                e_inf=2.7,
                de=5.9,
                tau_0=9.4e-10,
                number_of_debye_poles=-1,
                optimizer_options={"seed": 111},
            )
        )
    ), lambda f: 2.7 + 5.9 / (1 + (2j * np.pi * f * 9.4e-10) ** 0.91) ** 0.45, 5, 10
    yield "lossless", HavriliakNegami(
        **(
            common
            | dict(
                de=0,
                optimizer_options={"seed": 111, "maxiter": 2, "swarmsize": 4},
            )
        )
    ), lambda f: np.full_like(f, 3, dtype=complex), 1e-10, 1e-10
    fractions = (0.6, 0.119, 0.281)
    materials = ((5, 0, 1), (4.9, 73.34, 8.0994e-12), (1, 0, 1))

    def mixed(f):
        return sum(v * np.sqrt(debye(f, *m)) for v, m in zip(fractions, materials)) ** 2

    yield "CRIM", Crim(
        1e6,
        3e9,
        0.5,
        fractions,
        materials,
        0,
        1,
        0,
        "mix",
        number_of_debye_poles=-1,
        optimizer_options={"seed": 111},
    ), mixed, 5, 10
    yield "Jonscher", Jonscher(
        1e6,
        1e10,
        3,
        1,
        1e9,
        0.7,
        0,
        1,
        0,
        "jonscher",
        number_of_debye_poles=-1,
        optimizer_options={"seed": 111},
    ), (lambda f: 3 + (2 * np.pi * f / 1e9) ** (-0.3) * (1 - 1j / np.tan(0.7 * np.pi / 2))), 5, 10
    path = Path(Debye_Fit.__file__).parent / "examples/Test.txt"
    yield "Rawdata", Rawdata(
        path,
        0,
        1,
        0,
        "raw",
        number_of_debye_poles=1,
        f_n=41,
        optimizer_options={"seed": 111, "maxiter": 250},
    ), (lambda f: debye(f, 10, 20, 1e-9)), 0.01, 0.05


def validate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, model, reference, mean_limit, max_limit in cases():
        with (output_dir / f"{name}.log").open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                reported_error, commands = model.run()
        sampled, _ = exported_spectrum(commands, model.freq)
        export_error, _ = metrics(sampled, model.rl + 1j * model.im)
        np.testing.assert_allclose(export_error, reported_error, rtol=1e-10, atol=1e-10)
        f = np.geomspace(model.freq[0], model.freq[-1], 1001)
        fitted, total = exported_spectrum(commands, f)
        target = reference(f)
        mean_error, max_error = metrics(fitted, target)
        finite_passive = bool(
            np.isfinite(total).all() and np.all(total.real >= 1) and np.all(total.imag <= 0)
        )
        passed = finite_passive and mean_error <= mean_limit and max_error <= max_limit
        record = dict(
            case=name,
            seed=111,
            trial_poles=model.number_of_debye_poles,
            maxiter=model.optimizer.maxiter,
            exported_poles=model.fitted_pole_count,
            fit_points=len(model.freq),
            validation_points=len(f),
            reported_mean_error_percent=float(reported_error),
            export_mean_error_percent=export_error,
            dense_mean_error_percent=mean_error,
            dense_max_component_error_percent=max_error,
            mean_limit_percent=mean_limit,
            max_limit_percent=max_limit,
            finite_passive=finite_passive,
            commands=commands,
            passed=bool(passed),
        )
        results.append(record)
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for col, (truth, fit, samples, label) in enumerate(
            [
                (target.real, fitted.real, model.rl, "Real relative permittivity"),
                (-target.imag, -fitted.imag, -model.im, "Positive loss"),
            ]
        ):
            axes[0, col].semilogx(f, truth, label="Independent reference")
            axes[0, col].semilogx(f, fit, "--", label="Exported material")
            axes[0, col].semilogx(model.freq, samples, ".", markersize=3, label="Fit samples")
            axes[0, col].set_ylabel(label)
            axes[0, col].legend(fontsize=8)
            axes[1, col].semilogx(f, 100 * (fit - truth) / (1 + abs(truth)))
            axes[1, col].set_ylabel("Normalised residual (%)")
            axes[1, col].set_xlabel("Frequency (Hz)")
        fig.suptitle(name)
        fig.savefig(output_dir / f"{name}.png", dpi=140)
        plt.close(fig)
        print(
            f"{name}: poles={model.fitted_pole_count}, mean={mean_error:.6g}%, "
            f"max={max_error:.6g}% — {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
    report = dict(
        numpy=np.__version__,
        scipy=scipy.__version__,
        metric="Mean-error sum and maximum component error, normalised by 1+abs(target)",
        results=results,
        passed=all(item["passed"] for item in results),
    )
    with (output_dir / "results.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/debye-fit-validation"))
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.output_dir) else 1)
