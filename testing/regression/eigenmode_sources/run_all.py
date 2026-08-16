from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_input_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.in")
        if path.is_file() and "legacy" not in path.relative_to(root).parts
    )


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "gprMax").is_dir():
            return candidate
    raise SystemExit(f"Could not find the gprMax repository root above {start}")


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every eigenmode-source regression input and regenerate "
            "the snapshot and case-specific diagnostic plots."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing regression models. Defaults to this suite directory.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run gprMax and the plotting helper. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Only regenerate plots from existing snapshot files.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only run the gprMax input files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--gprmax-arg",
        action="append",
        default=[],
        help="Extra argument passed to every 'python -m gprMax' invocation. Repeat for multiple arguments.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    suite_root = Path(__file__).resolve().parent
    repo_root = find_repository_root(suite_root)
    plot_script = suite_root / "plot_snapshots.py"
    sparameter_plot_script = suite_root / "plot_sparameters.py"
    validation_script = suite_root / "validate_sparameters.py"

    if not root.is_dir():
        raise SystemExit(f"Test root does not exist: {root}")
    if not plot_script.is_file():
        raise SystemExit(f"Plotting helper not found: {plot_script}")

    input_files = find_input_files(root)
    if not input_files:
        raise SystemExit(f"No .in files found under {root}")

    case_dirs = []
    seen_case_dirs = set()
    for input_file in input_files:
        case_dir = input_file.parent.resolve()
        if case_dir not in seen_case_dirs:
            seen_case_dirs.add(case_dir)
            case_dirs.append(case_dir)

    env_python = os.environ.get("PYTHON")
    python = env_python if env_python else args.python

    if not args.skip_runs:
        for index, input_file in enumerate(input_files, start=1):
            print(
                f"\n[{index}/{len(input_files)}] Running {input_file.relative_to(repo_root)}",
                flush=True,
            )
            command = [
                python,
                "-m",
                "gprMax",
                str(input_file),
                "--hide-progress-bars",
                *args.gprmax_arg,
            ]
            run_command(command, cwd=repo_root, dry_run=args.dry_run)

    if not args.skip_plots:
        print(f"\nValidating modal S-parameter expectations below {root}", flush=True)
        run_command(
            [python, str(validation_script), str(root)],
            cwd=repo_root,
            dry_run=args.dry_run,
        )
        print(f"\nPlotting snapshots for {len(case_dirs)} case directories", flush=True)
        command = [python, str(plot_script), *(str(case_dir) for case_dir in case_dirs)]
        run_command(command, cwd=repo_root, dry_run=args.dry_run)
        print(f"\nPlotting modal S-parameters below {root}", flush=True)
        run_command(
            [python, str(sparameter_plot_script), str(root)],
            cwd=repo_root,
            dry_run=args.dry_run,
        )
        for plotter_script in sorted(root.rglob("plot_*.py")):
            if plotter_script.resolve() in {
                plot_script.resolve(),
                sparameter_plot_script.resolve(),
            }:
                continue
            print(f"\nRunning plot helper {plotter_script}", flush=True)
            run_command(
                [python, str(plotter_script), str(plotter_script.parent)],
                cwd=repo_root,
                dry_run=args.dry_run,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
