"""Shared reporting helpers for wall-impedance rectangular-waveguide validations."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def validation_cache_stem(prefix: str, configuration: dict) -> str:
    """Return a deterministic cache stem for one complete validation setup."""

    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"validation cache configuration cannot encode {type(value)!r}")

    payload = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=json_default,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{prefix}_{digest}"


def plot_wall_waveguide_validation(
    path: Path,
    result: dict,
    *,
    title: str,
    maximum_source_reflection_db: float,
) -> None:
    """Write the common four-panel wall-waveguide validation figure."""

    frequency_ghz = result["frequency_hz"] * 1e-9
    impedance_frequency_ghz = result["impedance_frequency_hz"] * 1e-9
    target_impedance = result["target_impedance_ohm"]
    fitted_impedance = result["fitted_impedance_ohm"]

    figure, axes = plt.subplots(4, 1, figsize=(9.0, 12.0), sharex=True)
    axes[0].plot(
        impedance_frequency_ghz,
        target_impedance.real,
        color="black",
        linewidth=2.0,
        label=r"Actual Re($Z_s$)",
    )
    axes[0].plot(
        impedance_frequency_ghz,
        fitted_impedance.real,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label=r"Fit Re($Z_s$)",
    )
    axes[0].plot(
        impedance_frequency_ghz,
        target_impedance.imag,
        color="0.35",
        linewidth=2.0,
        label=r"Actual Im($Z_s$)",
    )
    axes[0].plot(
        impedance_frequency_ghz,
        fitted_impedance.imag,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label=r"Fit Im($Z_s$)",
    )
    axes[0].set_ylabel(r"$Z_s$ ($\Omega$)")
    axes[0].set_title(r"Surface impedance: actual and fit")
    axes[0].legend(ncol=2)

    axes[1].plot(
        frequency_ghz,
        result["source_reflection_db"],
        "o-",
        markersize=3.5,
        label="FDTD source-plane S11",
    )
    axes[1].axhline(
        maximum_source_reflection_db,
        color="tab:red",
        linestyle="--",
        label="Acceptance limit",
    )
    axes[1].set_ylabel("S11 (dB)")
    axes[1].set_title("Driven-port reflection")
    axes[1].legend()

    axes[2].plot(
        frequency_ghz,
        result["fdfd_theory_alpha_per_m"],
        color="black",
        linewidth=2.0,
        label="Perturbation theory",
    )
    axes[2].plot(
        frequency_ghz,
        result["fdfd_alpha_per_m"],
        "o--",
        markersize=3.5,
        label=r"FDFD $-k_0\operatorname{Im}(n_\mathrm{eff})$",
    )
    axes[2].set_ylabel(r"$\alpha$ (Np/m)")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("FDFD modal attenuation")
    axes[2].legend()

    axes[3].plot(
        frequency_ghz,
        result["fdtd_theory_alpha_per_m"],
        color="black",
        linewidth=2.0,
        label="Perturbation theory",
    )
    axes[3].plot(
        frequency_ghz,
        result["fdtd_alpha_per_m"],
        "o--",
        markersize=3.5,
        label="FDTD two-plane S21",
    )
    axes[3].set_xlabel("Frequency (GHz)")
    axes[3].set_ylabel(r"$\alpha$ (Np/m)")
    axes[3].set_ylim(bottom=0)
    axes[3].set_title("FDTD S21 attenuation")
    axes[3].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def complex_relative_l2_error(calculated, reference) -> float:
    """Return the relative L2 error between two complex response arrays."""

    calculated = np.asarray(calculated, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    return float(np.linalg.norm(calculated - reference) / np.linalg.norm(reference))
