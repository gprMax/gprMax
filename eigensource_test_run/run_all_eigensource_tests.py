from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_input_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.in") if path.is_file())


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run every eigensource test input and regenerate combined |E| "
            "snapshot plots with plot_direction_snapshots.py."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing eigensource test cases. Defaults to this script's directory.",
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
    repo_root = root.parent.resolve()
    plot_script = root / "plot_direction_snapshots.py"

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
            print(f"\n[{index}/{len(input_files)}] Running {input_file.relative_to(repo_root)}", flush=True)
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
        print(f"\nPlotting snapshots for {len(case_dirs)} case directories", flush=True)
        command = [python, str(plot_script), *(str(case_dir) for case_dir in case_dirs)]
        run_command(command, cwd=repo_root, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
