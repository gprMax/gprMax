"""Portable compiler configuration shared by source and wheel builds."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class CompilerConfiguration:
    """Arguments required to compile and link the Cython extensions."""

    compile_args: tuple[str, ...]
    linker_args: tuple[str, ...]
    libraries: tuple[str, ...] = ()
    include_dirs: tuple[str, ...] = ()
    library_dirs: tuple[str, ...] = ()


def _enabled(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def build_jobs(*, environ: Mapping[str, str] = os.environ) -> int:
    """Return a conservative number of parallel extension-build jobs."""

    configured = environ.get("GPRMAX_BUILD_JOBS")
    if configured is None:
        return min(2, os.cpu_count() or 1)
    try:
        jobs = int(configured)
    except ValueError as exc:
        raise ValueError("GPRMAX_BUILD_JOBS must be a positive integer") from exc
    if jobs < 1:
        raise ValueError("GPRMAX_BUILD_JOBS must be a positive integer")
    return jobs


def _native_flag(platform_name: str, machine: str) -> Optional[str]:
    """Return an explicit opt-in host-CPU optimisation flag."""

    if platform_name == "linux":
        return "-march=native"
    if platform_name == "darwin":
        return "-mcpu=native" if machine.lower() in {"arm64", "aarch64"} else "-march=native"
    return None


def _valid_libomp_prefix(prefix: Path) -> bool:
    return (prefix / "include" / "omp.h").is_file() and (prefix / "lib" / "libomp.dylib").is_file()


def find_libomp_prefix(
    *,
    environ: Mapping[str, str] = os.environ,
    python_prefix: Path = Path(sys.prefix),
) -> Optional[Path]:
    """Locate a macOS ``libomp`` installation without assuming one package manager."""

    candidates: list[Path] = []
    configured = environ.get("GPRMAX_LIBOMP_PREFIX")
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.append(python_prefix)

    # Prefer an explicit setting or the active Python environment. In
    # particular, do not require Homebrew to be runnable when either of these
    # already provides a complete libomp installation.
    for candidate in candidates:
        candidate = candidate.resolve()
        if _valid_libomp_prefix(candidate):
            return candidate

    candidates = []
    brew = shutil.which("brew")
    if brew is not None:
        result = subprocess.run(
            [brew, "--prefix", "libomp"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            candidates.append(Path(result.stdout.strip()))

    candidates.extend((Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")))

    for candidate in candidates:
        candidate = candidate.resolve()
        if _valid_libomp_prefix(candidate):
            return candidate
    return None


def compiler_configuration(
    *,
    platform_name: str = sys.platform,
    machine: str = platform.machine(),
    environ: Mapping[str, str] = os.environ,
    libomp_prefix: Optional[Path] = None,
) -> CompilerConfiguration:
    """Return portable OpenMP build flags for the requested platform.

    Host-specific instruction selection is deliberately disabled by default so
    binary wheels run on machines other than the build host. Developers making
    private source builds may opt in with ``GPRMAX_BUILD_NATIVE=1``.
    """

    native = _enabled(environ.get("GPRMAX_BUILD_NATIVE"))

    if platform_name == "win32":
        if native:
            raise ValueError("GPRMAX_BUILD_NATIVE is not supported by the MSVC build")
        return CompilerConfiguration(compile_args=("/O2", "/openmp", "/w"), linker_args=())

    if platform_name == "linux":
        compile_args = ["-O3", "-w", "-fopenmp"]
        if native:
            compile_args.append("-march=native")
        return CompilerConfiguration(
            compile_args=tuple(compile_args),
            linker_args=("-fopenmp",),
        )

    if platform_name == "darwin":
        prefix = libomp_prefix or find_libomp_prefix(environ=environ)
        if prefix is None or not _valid_libomp_prefix(Path(prefix)):
            raise RuntimeError(
                "Cannot find libomp on macOS. Install it with 'brew install libomp' or set "
                "GPRMAX_LIBOMP_PREFIX to a prefix containing include/omp.h and lib/libomp.dylib."
            )

        prefix = Path(prefix).resolve()
        compile_args = ["-O3", "-w", "-Xpreprocessor", "-fopenmp"]
        if native:
            native_flag = _native_flag(platform_name, machine)
            if native_flag is not None:
                compile_args.append(native_flag)

        lib_dir = prefix / "lib"
        return CompilerConfiguration(
            compile_args=tuple(compile_args),
            linker_args=(f"-Wl,-rpath,{lib_dir}",),
            libraries=("omp",),
            include_dirs=(str(prefix / "include"),),
            library_dirs=(str(lib_dir),),
        )

    raise RuntimeError(f"Unsupported platform for compiling gprMax extensions: {platform_name}")
