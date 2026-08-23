"""Measure planar-interface dependence on material construction order.

The same final voxel geometry is built in two ways.  The normal construction
starts from the implicit free-space background and adds boxes from the far
side toward the source.  The reverse construction first fills the full domain
with the first layer, then overwrites source-side free space and the remaining
layers.  All results are de-embedded to the same requested first interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gprmax-matplotlib")

import matplotlib
import numpy as np

from .validate_multilayer_fdtd import (
    ANALYTICAL_PLOT_FREQUENCIES,
    RESULTS,
    STACKS,
    _analyse,
    _analytical_stack,
    _metrics,
    _phase_for_plot,
    _run,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULT_CASES = ("dielectric_slab", "debye_lorentz")


def _order_difference(first, second, analytical):
    return {
        "complex_relative_l2_difference": float(
            np.linalg.norm(first - second) / np.linalg.norm(analytical)
        ),
        "maximum_complex_absolute_difference": float(np.max(np.abs(first - second))),
    }


def _load_normal_result(case_name, mode, normal_results_dir):
    """Load the already validated normal-order complex reflection samples."""

    table = np.genfromtxt(normal_results_dir / f"{case_name}.csv", delimiter=",", names=True)
    analytical = table["analytical_magnitude"] * np.exp(
        1j * np.deg2rad(table["analytical_phase_degrees"])
    )
    fdtd = table[f"{mode}_magnitude"] * np.exp(1j * np.deg2rad(table[f"{mode}_phase_degrees"]))
    return {
        "frequencies": table["frequency_hz"],
        "gamma_fdtd": fdtd,
        "gamma_analytical": analytical,
    }


def _write_case(case_name, stack, results, output_dir):
    frequencies = results["averaged_normal"]["frequencies"]
    analytical = results["averaged_normal"]["gamma_analytical"]
    dense_analytical = _analytical_stack(stack, ANALYTICAL_PLOT_FREQUENCIES)
    columns = ["frequency_hz", "analytical_real", "analytical_imag"]
    for key in results:
        columns.extend((f"{key}_real", f"{key}_imag"))
    with (output_dir / f"construction_order_{case_name}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index, frequency in enumerate(frequencies):
            row = [frequency, analytical[index].real, analytical[index].imag]
            for result in results.values():
                value = result["gamma_fdtd"][index]
                row.extend((value.real, value.imag))
            writer.writerow(row)

    frequency_ghz = frequencies / 1e9
    dense_frequency_ghz = ANALYTICAL_PLOT_FREQUENCIES / 1e9
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.4, 1.4, 1)},
    )
    axes[0].plot(
        dense_frequency_ghz,
        np.abs(dense_analytical),
        color="black",
        linewidth=2,
        label="Analytical recursion",
    )
    axes[1].plot(
        dense_frequency_ghz,
        np.unwrap(np.angle(dense_analytical)) * 180 / np.pi,
        color="black",
        linewidth=2,
        label="Analytical recursion",
    )
    styles = {
        "averaged_normal": ("black", "o", "Average: background + boxes"),
        "averaged_reverse": ("black", "x", "Average: fill + overwrite"),
        "staircased_normal": ("black", "s", "Staircase: background + boxes"),
        "staircased_reverse": ("black", "+", "Staircase: fill + overwrite"),
    }
    for key, result in results.items():
        colour, marker, label = styles[key]
        axes[0].plot(
            frequency_ghz,
            np.abs(result["gamma_fdtd"]),
            linestyle="none",
            marker=marker,
            markerfacecolor="none",
            markersize=3.5,
            color=colour,
            label=label,
        )
        axes[1].plot(
            frequency_ghz,
            _phase_for_plot(result["gamma_fdtd"], analytical),
            linestyle="none",
            marker=marker,
            markerfacecolor="none",
            markersize=3.5,
            color=colour,
            label=label,
        )
    axes[2].semilogy(
        frequency_ghz,
        np.maximum(
            np.abs(
                results["averaged_normal"]["gamma_fdtd"] - results["averaged_reverse"]["gamma_fdtd"]
            ),
            1e-16,
        ),
        color="black",
        linestyle="-",
        label="Average construction-order difference",
    )
    axes[2].semilogy(
        frequency_ghz,
        np.maximum(
            np.abs(
                results["staircased_normal"]["gamma_fdtd"]
                - results["staircased_reverse"]["gamma_fdtd"]
            ),
            1e-16,
        ),
        color="black",
        linestyle="--",
        label="Staircase construction-order difference",
    )
    axes[0].set(ylabel=r"$|\Gamma|$", title=stack.label)
    axes[1].set_ylabel(r"Unwrapped phase of $\Gamma$ [deg]")
    axes[2].set(
        xlabel="Frequency [GHz]",
        ylabel=r"$|\Gamma_{\mathrm{normal}}-\Gamma_{\mathrm{reverse}}|$",
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    figure.savefig(output_dir / f"construction_order_{case_name}.png", dpi=180)
    plt.close(figure)


def run_validation(
    output_dir,
    cache_dir,
    normal_results_dir,
    *,
    cases,
    gpu,
    precision,
    threads,
    reuse,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    incident, dt, _ = _run(
        "free_space",
        "reference",
        cache_dir,
        gpu=gpu,
        precision=precision,
        threads=threads,
        reuse=reuse,
    )
    summary = {"cases": {}}
    for case_name in cases:
        stack = STACKS[case_name]
        results = {}
        case_summary = {"runs": {}}
        for mode in ("averaged", "staircased"):
            normal_key = f"{mode}_normal"
            normal_result = _load_normal_result(case_name, mode, normal_results_dir)
            results[normal_key] = normal_result
            case_summary["runs"][normal_key] = {
                **_metrics(normal_result),
                "runtime_seconds": None,
                "source": str(normal_results_dir / f"{case_name}.csv"),
            }

            total, case_dt, runtime = _run(
                case_name,
                mode,
                cache_dir,
                gpu=gpu,
                precision=precision,
                threads=threads,
                reuse=reuse,
                construction_order="reverse",
            )
            if not np.isclose(case_dt, dt, rtol=1e-12, atol=0):
                raise RuntimeError(f"Time-step mismatch for {case_name}/{mode}/reverse")
            reverse_key = f"{mode}_reverse"
            reverse_result = _analyse(incident, total, dt, stack)
            results[reverse_key] = reverse_result
            case_summary["runs"][reverse_key] = {
                **_metrics(reverse_result),
                "runtime_seconds": runtime,
            }
        analytical = results["averaged_normal"]["gamma_analytical"]
        case_summary["construction_order_difference"] = {
            mode: _order_difference(
                results[f"{mode}_normal"]["gamma_fdtd"],
                results[f"{mode}_reverse"]["gamma_fdtd"],
                analytical,
            )
            for mode in ("averaged", "staircased")
        }
        _write_case(case_name, stack, results, output_dir)
        summary["cases"][case_name] = case_summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=tuple(STACKS), default=DEFAULT_CASES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS.parent / "multilayer_construction_order",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--normal-results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.cache_dir is None:
        args.cache_dir = args.output_dir / "cache"
    summary = run_validation(
        args.output_dir,
        args.cache_dir,
        args.normal_results_dir,
        cases=args.cases,
        gpu=args.gpu,
        precision=args.precision,
        threads=args.threads,
        reuse=args.reuse,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
