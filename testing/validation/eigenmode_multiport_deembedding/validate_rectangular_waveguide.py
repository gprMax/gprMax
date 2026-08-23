"""Validate full incident-matrix modal de-embedding in a lossless TE10 guide."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gprMax
from gprMax.studies import _deembed_modal_responses


SPEED_OF_LIGHT = 299_792_458.0
GUIDE_WIDTH = 0.010
REFERENCE_PLANE_SPACING = 0.020


def build_study():
    """Return a two-port matched-guide scene and its independent-run study."""

    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.06, 0.010, 0.012)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1.2e-9))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, dl, 0.012), material_id="pec"))
    scene.add(
        gprMax.Box(
            p1=(0, 0.009, 0),
            p2=(0.06, 0.010, 0.012),
            material_id="pec",
        )
    )
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.010, dl), material_id="pec"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0.011),
            p2=(0.06, 0.010, 0.012),
            material_id="pec",
        )
    )
    scene.add(gprMax.EigenmodeBand(id="band", fmin=20e9, fmax=24e9, points=17))
    for port, x, direction in ((1, 0.020, "+"), (2, 0.040, "-")):
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(x, dl, dl),
                p2=(x, 0.009, 0.011),
                direction=direction,
                modes=(1,),
                anchors=(16.9e9, 22e9, 27e9),
                plot_fields=False,
            )
        )
        scene.add(
            gprMax.VirtualWaveguide(
                port=port,
                length_cells=14,
                pml_cells=6,
                source_clearance_cells=3,
            )
        )
    excitation = gprMax.EigenmodeExcitation(
        port=1,
        mode=1,
        waveform="auto",
        plot_waveform=False,
    )
    scene.add(excitation)
    study = gprMax.EigenmodeStudy(
        [
            gprMax.StudyCase(
                "drive_port1",
                [gprMax.ObjectState(excitation, port=1, mode=1)],
            ),
            gprMax.StudyCase(
                "drive_port2",
                [gprMax.ObjectState(excitation, port=2, mode=1)],
            ),
        ]
    )
    return scene, study


def analytical_te10(frequency):
    cutoff = SPEED_OF_LIGHT / (2 * GUIDE_WIDTH)
    if np.any(frequency <= cutoff):
        raise ValueError("This validation requires propagating TE10 frequencies")
    beta = np.sqrt(np.square(2 * np.pi * frequency / SPEED_OF_LIGHT) - np.square(np.pi / GUIDE_WIDTH))
    return np.exp(-1j * beta * REFERENCE_PLANE_SPACING)


def magnitude_db(values):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20 * np.log10(np.abs(values))


def validate_exact_network(root: Path) -> dict[str, float]:
    """Exercise the production solve with an exact frequency-dependent network."""

    frequency = np.linspace(1e9, 10e9, 101)
    coordinate = (frequency - frequency[0]) / np.ptp(frequency)
    transmission = 0.75 * np.exp(-1j * (0.2 + 1.1 * coordinate))
    true_s = np.empty((frequency.size, 2, 2), dtype=np.complex128)
    true_s[:, 0, 0] = 0.10 + 0.03j * coordinate
    true_s[:, 1, 1] = -0.12 + 0.02j * coordinate
    true_s[:, 0, 1] = transmission
    true_s[:, 1, 0] = transmission
    incident = np.empty_like(true_s)
    incident[:, 0, 0] = 1.0 + 0.02 * coordinate
    incident[:, 1, 1] = 0.97 - 0.01 * coordinate
    incident[:, 0, 1] = (0.02 + 0.01j) * np.exp(0.4j * coordinate)
    incident[:, 1, 0] = (-0.015 + 0.008j) * np.exp(-0.3j * coordinate)
    outgoing = np.einsum("fij,fjk->fik", true_s, incident)
    recovered, valid, condition, basis_valid = _deembed_modal_responses(
        incident,
        outgoing,
    )
    diagonal = np.diagonal(incident, axis1=1, axis2=2)
    approximate = outgoing / diagonal[:, np.newaxis, :]
    full_error = np.linalg.norm(recovered - true_s, axis=(1, 2))
    diagonal_error = np.linalg.norm(approximate - true_s, axis=(1, 2))
    full_residual = np.linalg.norm(
        np.einsum("fij,fjk->fik", recovered, incident) - outgoing,
        axis=(1, 2),
    ) / np.linalg.norm(outgoing, axis=(1, 2))
    diagonal_residual = np.linalg.norm(
        np.einsum("fij,fjk->fik", approximate, incident) - outgoing,
        axis=(1, 2),
    ) / np.linalg.norm(outgoing, axis=(1, 2))
    if not (np.all(valid) and np.all(basis_valid)):
        raise ValueError("The prescribed analytical incident basis was rejected")

    with (root / "analytical_matrix_deembedding.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frequency_hz",
                "full_S_error_norm",
                "diagonal_S_error_norm",
                "full_B_minus_SA_relative_residual",
                "diagonal_B_minus_SA_relative_residual",
                "incident_condition_number",
            )
        )
        writer.writerows(
            zip(
                frequency,
                full_error,
                diagonal_error,
                full_residual,
                diagonal_residual,
                condition,
            )
        )

    figure, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True)
    axes[0].semilogy(frequency * 1e-9, full_error, "o-", label="Full A-matrix solve")
    axes[0].semilogy(
        frequency * 1e-9,
        diagonal_error,
        "s--",
        label="Diagonal approximation",
    )
    axes[0].set_ylabel(r"$\|S_{\mathrm{recovered}}-S_{\mathrm{exact}}\|_F$")
    axes[0].set_title("Prescribed analytical two-port with non-diagonal incident waves")
    axes[0].legend()
    axes[1].semilogy(
        frequency * 1e-9,
        np.maximum(full_residual, np.finfo(float).eps),
        "o-",
        label="Full A-matrix solve",
    )
    axes[1].semilogy(
        frequency * 1e-9,
        diagonal_residual,
        "s--",
        label="Diagonal approximation",
    )
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"$\|B-SA\|_F/\|B\|_F$")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(root / "analytical_matrix_deembedding.png", dpi=200)
    plt.close(figure)
    return {
        "maximum_prescribed_full_S_error_norm": float(np.max(full_error)),
        "minimum_prescribed_diagonal_S_error_norm": float(np.min(diagonal_error)),
        "maximum_prescribed_full_B_minus_SA_relative_residual": float(np.max(full_residual)),
        "minimum_prescribed_diagonal_B_minus_SA_relative_residual": float(np.min(diagonal_residual)),
    }


def validate(root: Path, *, reuse: bool = False) -> dict[str, float]:
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "_cache"
    cache.mkdir(exist_ok=True)
    aggregate = cache / "rectangular_guide_study.h5"
    if reuse and aggregate.exists():
        result = gprMax.EigenmodeStudyResult.from_hdf5(aggregate)
    else:
        scene, study = build_study()
        gprMax.run(
            scenes=[scene],
            study=study,
            outputfile=cache / "rectangular_guide",
            cpu_precision="double",
            hide_progress_bars=True,
            log_level=30,
        )
        result = study.result

    frequency = result.frequency
    analytical = analytical_te10(frequency)
    full = result.s[:, 1, 0]
    diagonal = np.diagonal(result.incident_matrix, axis1=1, axis2=2)
    diagonal_approximation = result.outgoing_matrix / diagonal[:, np.newaxis, :]
    approximate = diagonal_approximation[:, 1, 0]
    full_magnitude_error = np.abs(np.abs(full) - 1)
    approximate_magnitude_error = np.abs(np.abs(approximate) - 1)
    full_phase_error = np.abs(np.rad2deg(np.angle(full / analytical)))
    approximate_phase_error = np.abs(np.rad2deg(np.angle(approximate / analytical)))
    exact_residual = np.linalg.norm(
        np.einsum("fij,fjk->fik", result.s, result.incident_matrix) - result.outgoing_matrix,
        axis=(1, 2),
    ) / np.linalg.norm(result.outgoing_matrix, axis=(1, 2))
    approximate_residual = np.linalg.norm(
        np.einsum("fij,fjk->fik", diagonal_approximation, result.incident_matrix) - result.outgoing_matrix,
        axis=(1, 2),
    ) / np.linalg.norm(result.outgoing_matrix, axis=(1, 2))

    metrics = {
        "maximum_full_magnitude_error": float(np.max(full_magnitude_error)),
        "maximum_diagonal_magnitude_error": float(np.max(approximate_magnitude_error)),
        "maximum_full_phase_error_deg": float(np.max(full_phase_error)),
        "maximum_diagonal_phase_error_deg": float(np.max(approximate_phase_error)),
        "maximum_full_B_minus_SA_relative_residual": float(np.max(exact_residual)),
        "maximum_diagonal_B_minus_SA_relative_residual": float(np.max(approximate_residual)),
        "maximum_incident_matrix_condition_number": float(np.max(result.deembedding_condition_number)),
        "maximum_passive_to_active_incident_ratio": float(
            np.max(
                np.abs(
                    result.incident_matrix
                    - np.eye(2)[np.newaxis, ...]
                    * np.diagonal(result.incident_matrix, axis1=1, axis2=2)[:, np.newaxis, :]
                )
            )
            / np.max(np.abs(diagonal))
        ),
    }
    metrics.update(validate_exact_network(root))

    with (root / "rectangular_waveguide_deembedding.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frequency_hz",
                "analytical_S21_real",
                "analytical_S21_imag",
                "full_S21_real",
                "full_S21_imag",
                "diagonal_S21_real",
                "diagonal_S21_imag",
                "full_phase_error_deg",
                "diagonal_phase_error_deg",
                "full_B_minus_SA_relative_residual",
                "diagonal_B_minus_SA_relative_residual",
                "incident_condition_number",
            )
        )
        for values in zip(
            frequency,
            analytical.real,
            analytical.imag,
            full.real,
            full.imag,
            approximate.real,
            approximate.imag,
            full_phase_error,
            approximate_phase_error,
            exact_residual,
            approximate_residual,
            result.deembedding_condition_number,
        ):
            writer.writerow(values)
    (root / "summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    frequency_ghz = frequency * 1e-9
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 10.2), sharex=True)
    axes[0].plot(frequency_ghz, magnitude_db(analytical), "k-", label="Analytical TE10")
    axes[0].plot(frequency_ghz, magnitude_db(full), "o-", label="Full A-matrix solve")
    axes[0].plot(
        frequency_ghz,
        magnitude_db(approximate),
        "s--",
        label="Diagonal approximation",
    )
    axes[0].set_ylabel(r"$|S_{21}|$ (dB)")
    axes[0].legend()
    axes[0].set_title("Two-port TE10 de-embedding: gprMax versus analytical propagation")

    analytical_phase = np.rad2deg(np.unwrap(np.angle(analytical)))
    axes[1].plot(frequency_ghz, analytical_phase, "k-", label="Analytical TE10")
    axes[1].plot(
        frequency_ghz,
        np.rad2deg(np.unwrap(np.angle(full))),
        "o-",
        label="Full A-matrix solve",
    )
    axes[1].plot(
        frequency_ghz,
        np.rad2deg(np.unwrap(np.angle(approximate))),
        "s--",
        label="Diagonal approximation",
    )
    axes[1].set_ylabel(r"$\angle S_{21}$ (degrees)")
    axes[1].legend()

    axes[2].semilogy(
        frequency_ghz,
        np.maximum(exact_residual, np.finfo(float).eps),
        "o-",
        label=r"Full solve $\|B-SA\|/\|B\|$",
    )
    axes[2].semilogy(
        frequency_ghz,
        np.maximum(approximate_residual, np.finfo(float).tiny),
        "s--",
        label=r"Diagonal approximation $\|B-SA\|/\|B\|$",
    )
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].set_ylabel("Network-equation residual")
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(root / "rectangular_waveguide_deembedding.png", dpi=200)
    plt.close(figure)

    if metrics["maximum_full_magnitude_error"] >= 2.5e-3:
        raise ValueError("Full de-embedded TE10 magnitude exceeded the validation tolerance")
    if metrics["maximum_full_phase_error_deg"] >= 1.5:
        raise ValueError("Full de-embedded TE10 phase exceeded the validation tolerance")
    if metrics["maximum_full_B_minus_SA_relative_residual"] >= 1e-12:
        raise ValueError("Full de-embedding did not satisfy B = S A to numerical precision")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    metrics = validate(args.output, reuse=args.reuse)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
