#!/usr/bin/env python3
import argparse
import subprocess
import sys
import shutil


def check_command(cmd, name):
    if not shutil.which(cmd):
        print(
            f"Error: Required dependency '{name}' ({cmd}) not found in PATH.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Build script for gprMax supporting CPU, MPI, and CUDA backends."
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "mpi", "cuda"],
        default="cpu",
        help="Target backend",
    )
    parser.add_argument("--install-prefix", help="Installation prefix directory")
    args = parser.parse_args()

    print(f"Configuring build for {args.backend} backend...")

    if args.backend == "mpi":
        check_command("mpicc", "MPI Compiler")
    elif args.backend == "cuda":
        check_command("nvcc", "NVIDIA CUDA Compiler")

    # Call the original setup.py to build Cython extensions
    build_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]

    print(f"Running: {' '.join(build_cmd)}")
    try:
        subprocess.run(build_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

    if args.install_prefix:
        install_cmd = [
            sys.executable,
            "setup.py",
            "install",
            "--prefix",
            args.install_prefix,
        ]
        print(f"Running: {' '.join(install_cmd)}")
        try:
            subprocess.run(install_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"Installation failed with error code {e.returncode}", file=sys.stderr
            )
            sys.exit(e.returncode)

    print("Build step completed successfully.")


if __name__ == "__main__":
    main()
