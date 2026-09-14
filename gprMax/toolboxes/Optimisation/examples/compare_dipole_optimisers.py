"""Run separate optimiser campaigns using the same dipole model and criterion.

run launches one advanced dipole command per optimiser in its own result
directory. jobs controls concurrent campaigns; population_size controls
members inside each search. summarise reads their committed records to
compare proposal histories, scores and best spectra.
This is a mechanics benchmark, not the copy-and-edit model template.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gprMax.toolboxes.Optimisation._storage import write_json
from gprMax.toolboxes.Optimisation.adapters import OPTIMISERS


def summarise(directory):
    """Rebuild comparison tables and plots from committed campaign results."""
    import matplotlib.pyplot as plt
    import numpy as np

    from gprMax.toolboxes.Optimisation import read_port

    directory = Path(directory)
    config = json.loads((directory / "comparison-settings.json").read_text())
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for number, name in enumerate(config["optimisers"]):
        style = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)))[number % 5]
        path = directory / name / "optimisation.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        trials = data["search"]["trials"]
        history = json.loads((path.parent / "search_history.json").read_text())
        best = data["best"]
        ledger = [
            json.loads(p.read_text())
            for p in sorted((path.parent / "optimiser").glob("trial-*.json"))
        ]
        # Count where the target was first met, even though these campaigns
        # continue to their fixed budgets to expose subsequent optimiser feedback.
        first_hit = next(
            (
                i + 1
                for i, t in enumerate(trials)
                if t["status"] == "complete" and t["value"] <= config["frequency_tolerance_hz"]
            ),
            None,
        )
        rows.append(
            {
                "optimiser": name,
                "simulations": data["search"]["run_attempts"],
                "proposals": len(trials),
                "unique_lengths": len({t["parameters"]["arm_cells"] for t in trials}),
                "first_target_evaluation": first_hit,
                "target_met": data["search"]["target_met"],
                "stop_reason": data["search"]["stop_reason"],
                "best_length_mm": best["total_length_m"] * 1e3,
                "best_frequency_ghz": best["resonance_frequency_hz"] / 1e9,
                "best_error_mhz": best["frequency_error_hz"] / 1e6,
                "initial_error_mhz": history[0]["frequency_error_hz"] / 1e6,
                "feedback_delivered": all(t["feedback"] == "delivered" for t in ledger),
                "wall_seconds": sum(t["elapsed_seconds"] for t in ledger),
            }
        )
        x = np.arange(1, len(history) + 1)
        errors = np.array([h["frequency_error_hz"] / 1e6 for h in history])
        (line,) = axes[0].step(
            x, np.minimum.accumulate(errors), where="post", label=name.upper(), linestyle=style
        )
        axes[1].plot(
            x,
            [h["total_length_m"] * 1e3 for h in history],
            "-o",
            markersize=3,
            color=line.get_color(),
            label=name.upper(),
        )
        port = read_port(path.parent / best["output_file"], "feed")
        valid = port.valid & (port.frequency >= 0.85e9) & (port.frequency <= 1.15e9)
        axes[2].plot(
            port.frequency / 1e9,
            np.where(valid, 20 * np.log10(np.maximum(abs(port.s11), np.finfo(float).tiny)), np.nan),
            label=name.upper(),
            linestyle=style,
        )
    axes[0].axhline(
        config["frequency_tolerance_hz"] / 1e6, color="black", ls=":", label="Tolerance"
    )
    axes[0].set(
        xlabel="Actual simulations", ylabel="Best frequency error (MHz)", title="Objective feedback"
    )
    axes[1].set(
        xlabel="Actual simulations", ylabel="Total dipole length (mm)", title="Proposed geometries"
    )
    axes[2].axvline(1, color="black", ls=":")
    axes[2].set(
        xlabel="Frequency (GHz)",
        ylabel="S11 (dB)",
        xlim=(0.85, 1.15),
        title="Best evaluated designs",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"1 GHz dipole | Δz = {config['dz_m']*1e3:g} mm | {config['cycles']} ns | same model and objective"
    )
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=170)
    fig.savefig(directory / "comparison.pdf")
    plt.close(fig)
    write_json(directory / "comparison.json", {"settings": config, "results": rows})
    lines = [
        "# Optimiser mechanics: 1 mm dipole comparison",
        "",
        "All campaigns use the same model builder, parameter bounds, mesh, source, time window and resonance-frequency objective. These are single-seed integration demonstrations, not algorithm rankings.",
        "",
        "| Optimiser | Simulations | Unique lengths | Initial error (MHz) | Best length (mm) | Dip (GHz) | Error (MHz) | First target evaluation | Target met |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['optimiser'].upper()} | {r['simulations']} | {r['unique_lengths']} | {r['initial_error_mhz']:.3f} | {r['best_length_mm']:.3f} | {r['best_frequency_ghz']:.6f} | {r['best_error_mhz']:.3f} | {r['first_target_evaluation'] or '—'} | {r['target_met']} |"
        )
    lines.extend(
        [
            "",
            "![Convergence, proposals and S11](comparison.png)",
            "",
            "Curves overlap when algorithms have the same best objective history or select the same geometry. Different line styles keep those overlaps visible.",
            "",
            f"The objective is |estimated S11-dip frequency − 1 GHz| in Hz, using the same three-bin local quadratic fit. Target tolerance is ±{config['frequency_tolerance_hz']/1e6:g} MHz. Runs deliberately continue after success to expose subsequent proposals. Population algorithms update after all {config['population_size']} members have been evaluated; RF/TPE update after each candidate.",
            "",
            f"Axial spacing is {config['dz_m']*1e3:g} mm, total symmetric length increments are {2*config['dz_m']*1e3:g} mm and the feed gap is {config['dz_m']*1e3:g} mm. Transverse spacing and physical wire radius are fixed across algorithms. Each proposal receives a fresh solve, including repeated geometries. The nominal {config['cycles']} ns window provides approximately {1000/config['cycles']:g} MHz independent bins; interpolation does not increase that independent resolution.",
            "",
            "Inspect each algorithm folder for `campaign.json`, `optimiser/session.json`, population records `batch-*.json`, trial records `trial-*.json`, candidate manifests and saved HDF5 port spectra. Native coordinates and integer mapping are recorded for population proposals.",
            "",
        ]
    )
    (directory / "comparison.md").write_text("\n".join(lines))
    return rows


def run(
    directory,
    *,
    optimisers=OPTIMISERS,
    dz_m=0.001,
    cycles=100,
    n_trials=18,
    population_size=6,
    cpu_threads=2,
    jobs=1,
    optimiser_seed=7,
    frequency_tolerance_hz=1e7,
):
    """Launch independent campaigns with shared physical settings and fixed budgets.

    Every optimiser gets a fresh directory/process, the same simulation model
    and the same scalar criterion. jobs limits concurrent campaigns; each
    subprocess requests cpu_threads for its serial solver. A single seed
    illustrates integration mechanics rather than ranking algorithms.
    """
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x < 1
        for x in (jobs, cpu_threads, n_trials, population_size)
    ):
        raise ValueError(
            "jobs, cpu_threads, n_trials and population_size must be positive integers"
        )
    if (
        not optimisers
        or len(set(optimisers)) != len(optimisers)
        or any(name not in OPTIMISERS for name in optimisers)
    ):
        raise ValueError("Choose distinct supported optimiser names")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = {
        "optimisers": list(optimisers),
        "dz_m": dz_m,
        "cycles": cycles,
        "n_trials": n_trials,
        "population_size": population_size,
        "cpu_threads": cpu_threads,
        "jobs": jobs,
        "optimiser_seed": optimiser_seed,
        "frequency_tolerance_hz": frequency_tolerance_hz,
        "fixed_budget": True,
    }
    write_json(directory / "comparison-settings.json", config)

    def execute(name):
        """Run one named optimiser CLI, preserve its combined log and record the process outcome."""
        command = [
            sys.executable,
            "-m",
            "gprMax.toolboxes.Optimisation.examples.advanced.thin_wire_dipole",
            str(directory / name),
            "--objective",
            "resonance",
            "--target-hz",
            "1e9",
            "--cycles",
            str(cycles),
            "--dz-m",
            str(dz_m),
            "--frequency-tolerance-hz",
            str(frequency_tolerance_hz),
            "--n-trials",
            str(n_trials),
            "--population-size",
            str(population_size),
            "--cpu-threads",
            str(cpu_threads),
            "--optimiser-seed",
            str(optimiser_seed),
            "--optimiser",
            name,
            "--fixed-budget",
        ]
        started = time.perf_counter()
        with (directory / f"{name}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = {
            "optimiser": name,
            "returncode": completed.returncode,
            "wall_seconds": time.perf_counter() - started,
            "command": command,
        }
        write_json(directory / f"{name}-execution.json", result)
        return result

    executions = []
    # Processes isolate library state; each campaign still executes serially.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(execute, name) for name in optimisers]
        for future in as_completed(futures):
            item = future.result()
            executions.append(item)
            print(
                f"{item['optimiser']}: exit={item['returncode']}, {item['wall_seconds']:.1f} seconds",
                flush=True,
            )
    summarise(directory)
    if any(item["returncode"] for item in executions):
        raise RuntimeError(f"Some campaigns failed; inspect logs in {directory}")
    return executions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--optimisers", nargs="+", choices=OPTIMISERS, default=list(OPTIMISERS))
    parser.add_argument("--dz-m", type=float, default=0.001)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=18)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Independent concurrent campaigns; total requested solver threads = jobs × cpu-threads",
    )
    parser.add_argument("--optimiser-seed", type=int, default=7)
    parser.add_argument("--frequency-tolerance-hz", type=float, default=1e7)
    run(**vars(parser.parse_args()))
