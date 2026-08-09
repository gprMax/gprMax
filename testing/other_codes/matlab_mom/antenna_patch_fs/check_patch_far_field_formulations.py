"""Compare KSIR and Love-current far fields for the MATLAB patch benchmark."""

import argparse
import csv
import json
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from patch_antenna_gprmax import FREQUENCY, RESULTS_DIR, build_scene

import gprMax

OUTPUT_STEM = "patch_antenna_far_field_formulations"
PLOT_FLOOR_DB = -40.0
EC_TRANSFORM_ID = "patch_equivalent_current"


def _add_equivalent_current_outputs(scene, angle):
    """Attach independent Love-current cuts to the existing KSIR scene."""

    theta = np.abs(angle)
    phi_xz = np.where(angle < 0, 180.0, 0.0)
    phi_yz = np.where(angle < 0, -90.0, 90.0)
    scene.add(
        gprMax.NTFFFrequencyTransform(
            surface_id="patch_surface",
            id=EC_TRANSFORM_ID,
            frequencies=(FREQUENCY,),
            save_surface_dft=False,
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=theta,
            phi=phi_xz,
            transform_id=EC_TRANSFORM_ID,
            id="xz_plane",
            outputs=("Etheta", "Ephi"),
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=theta,
            phi=phi_yz,
            transform_id=EC_TRANSFORM_ID,
            id="yz_plane",
            outputs=("Etheta", "Ephi"),
        )
    )


def _complex_field(h5, transform_id, plane, component, size):
    path = f"ntff/patch_surface/frequency/{transform_id}/far_field/" f"{plane}/fields/{component}"
    values = np.asarray(h5[path])
    if values.shape != (1, size):
        raise ValueError(f"{path} has shape {values.shape}; expected {(1, size)}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite values")
    return values[0]


def _read_gprmax_patterns(path, angle):
    """Read both persisted transforms and verify their identities and grids."""

    theta = np.abs(angle)
    phi = {
        "xz_plane": np.where(angle < 0, 180.0, 0.0),
        "yz_plane": np.where(angle < 0, -90.0, 90.0),
    }
    transforms = {
        "ksir": ("patch_spectrum", "ksir"),
        "equivalent_current": (EC_TRANSFORM_ID, "equivalent_current"),
    }
    result = {}
    with h5py.File(path, "r") as h5:
        for name, (transform_id, formulation) in transforms.items():
            group = h5[f"ntff/patch_surface/frequency/{transform_id}"]
            if group.attrs["formulation"] != formulation:
                raise ValueError(
                    f"Transform {transform_id!r} reports formulation "
                    f"{group.attrs['formulation']!r}, expected {formulation!r}"
                )
            frequency = np.asarray(group["frequencies"])
            if frequency.shape != (1,) or not np.allclose(frequency, FREQUENCY):
                raise ValueError(f"Transform {transform_id!r} has the wrong frequency")
            for plane in phi:
                far = group[f"far_field/{plane}"]
                if not np.allclose(far["theta"], theta):
                    raise ValueError(f"{name} {plane} theta grid is inconsistent")
                if not np.allclose(far["phi"], phi[plane]):
                    raise ValueError(f"{name} {plane} phi grid is inconsistent")
            result[name] = {
                "xz_co": _complex_field(h5, transform_id, "xz_plane", "Etheta", angle.size),
                "xz_cross": _complex_field(h5, transform_id, "xz_plane", "Ephi", angle.size),
                "yz_co": _complex_field(h5, transform_id, "yz_plane", "Ephi", angle.size),
                "yz_cross": _complex_field(h5, transform_id, "yz_plane", "Etheta", angle.size),
            }
    return result


def _read_matlab_pattern(path, angle):
    data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
    if not np.array_equal(np.asarray(data["angle_deg"]), angle):
        raise ValueError("MATLAB and gprMax signed-angle grids do not match")
    peak = max(
        np.max(data["xz_directivity_dbi"]),
        np.max(data["yz_directivity_dbi"]),
    )
    return {
        "xz": np.asarray(data["xz_directivity_dbi"] - peak),
        "yz": np.asarray(data["yz_directivity_dbi"] - peak),
    }


def _normalised_db(pattern):
    peak = max(np.max(np.abs(pattern["xz_co"])), np.max(np.abs(pattern["yz_co"])))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Far-field pattern has no finite non-zero co-polar peak")
    floor = np.finfo(float).tiny
    return {
        "xz": 20 * np.log10(np.maximum(np.abs(pattern["xz_co"]) / peak, floor)),
        "yz": 20 * np.log10(np.maximum(np.abs(pattern["yz_co"]) / peak, floor)),
    }


def _difference_metrics(reference, candidate, selection):
    valid = selection & (reference > PLOT_FLOOR_DB) & (candidate > PLOT_FLOOR_DB)
    if not np.any(valid):
        raise ValueError("No pattern samples remain above the comparison floor")
    difference = candidate[valid] - reference[valid]
    return {
        "sample_count": int(np.count_nonzero(valid)),
        "rms_db": float(np.sqrt(np.mean(difference**2))),
        "mean_db": float(np.mean(difference)),
        "maximum_absolute_db": float(np.max(np.abs(difference))),
    }


def _complex_metrics(reference, candidate, angle):
    upper = np.abs(angle) <= 90
    reference_vector = np.concatenate((reference["xz_co"][upper], reference["yz_co"][upper]))
    candidate_vector = np.concatenate((candidate["xz_co"][upper], candidate["yz_co"][upper]))
    reference_norm = np.linalg.norm(reference_vector)
    relative_error = np.linalg.norm(candidate_vector - reference_vector) / reference_norm
    best_scale = np.vdot(reference_vector, candidate_vector) / np.vdot(
        reference_vector, reference_vector
    )
    residual = candidate_vector - best_scale * reference_vector
    return {
        "raw_relative_l2_error": float(relative_error),
        "best_fit_complex_scale_real": float(best_scale.real),
        "best_fit_complex_scale_imag": float(best_scale.imag),
        "best_fit_magnitude_ratio": float(abs(best_scale)),
        "best_fit_phase_difference_deg": float(np.angle(best_scale, deg=True)),
        "shape_relative_l2_error_after_complex_scaling": float(
            np.linalg.norm(residual) / np.linalg.norm(candidate_vector)
        ),
    }


def _component_metrics(reference, candidate, co_reference, selection):
    """Measure one complex component, retaining a useful null-field scale."""

    reference_values = reference[selection]
    candidate_values = candidate[selection]
    co_values = co_reference[selection]
    difference_norm = np.linalg.norm(candidate_values - reference_values)
    reference_norm = np.linalg.norm(reference_values)
    co_norm = np.linalg.norm(co_values)
    return {
        "ksir_norm": float(reference_norm),
        "ksir_norm_relative_to_co_polar": float(reference_norm / co_norm),
        "difference_relative_to_same_component": (
            None if reference_norm == 0 else float(difference_norm / reference_norm)
        ),
        "difference_relative_to_co_polar": float(difference_norm / co_norm),
    }


def _write_results(angle, matlab, patterns):
    normalised = {name: _normalised_db(pattern) for name, pattern in patterns.items()}
    upper = np.abs(angle) <= 90
    metrics = {
        "frequency_hz": FREQUENCY,
        "comparison_floor_db": PLOT_FLOOR_DB,
        "normalisation": "one co-polar peak across the two principal planes per formulation",
        "upper_hemisphere": {},
        "ksir_vs_equivalent_current_complex": _complex_metrics(
            patterns["ksir"], patterns["equivalent_current"], angle
        ),
        "upper_hemisphere_complex_components": {},
    }
    for plane in ("xz", "yz"):
        metrics["upper_hemisphere"][plane] = {
            "ksir_vs_matlab": _difference_metrics(matlab[plane], normalised["ksir"][plane], upper),
            "equivalent_current_vs_matlab": _difference_metrics(
                matlab[plane], normalised["equivalent_current"][plane], upper
            ),
            "equivalent_current_vs_ksir": _difference_metrics(
                normalised["ksir"][plane], normalised["equivalent_current"][plane], upper
            ),
        }
        co_name = f"{plane}_co"
        cross_name = f"{plane}_cross"
        metrics["upper_hemisphere_complex_components"][plane] = {
            "co_polar": _component_metrics(
                patterns["ksir"][co_name],
                patterns["equivalent_current"][co_name],
                patterns["ksir"][co_name],
                upper,
            ),
            "cross_polar": _component_metrics(
                patterns["ksir"][cross_name],
                patterns["equivalent_current"][cross_name],
                patterns["ksir"][co_name],
                upper,
            ),
        }

    csv_path = RESULTS_DIR / f"{OUTPUT_STEM}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "angle_deg",
                "matlab_xz_normalized_db",
                "ksir_xz_normalized_db",
                "equivalent_current_xz_normalized_db",
                "matlab_yz_normalized_db",
                "ksir_yz_normalized_db",
                "equivalent_current_yz_normalized_db",
            )
        )
        writer.writerows(
            zip(
                angle,
                matlab["xz"],
                normalised["ksir"]["xz"],
                normalised["equivalent_current"]["xz"],
                matlab["yz"],
                normalised["ksir"]["yz"],
                normalised["equivalent_current"]["yz"],
            )
        )

    json_path = RESULTS_DIR / f"{OUTPUT_STEM}_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, (plane, title) in enumerate((("xz", "E-plane (x-z)"), ("yz", "H-plane (y-z)"))):
        axes[0, column].plot(angle, matlab[plane], color="#d95f02", label="MATLAB MoM")
        axes[0, column].plot(
            angle,
            normalised["ksir"][plane],
            color="#1b6ca8",
            linestyle="--",
            label="gprMax KSIR",
        )
        axes[0, column].plot(
            angle,
            normalised["equivalent_current"][plane],
            color="#2b8c4b",
            linestyle=":",
            linewidth=2,
            label="gprMax equivalent currents",
        )
        axes[0, column].set_title(title)
        axes[0, column].set_ylim(PLOT_FLOOR_DB, 1)
        axes[0, column].grid(alpha=0.35)

        valid = upper & (matlab[plane] > PLOT_FLOOR_DB)
        axes[1, column].plot(
            angle[valid],
            normalised["ksir"][plane][valid] - matlab[plane][valid],
            color="#1b6ca8",
            linestyle="--",
            label="KSIR - MATLAB",
        )
        axes[1, column].plot(
            angle[valid],
            normalised["equivalent_current"][plane][valid] - matlab[plane][valid],
            color="#2b8c4b",
            linestyle=":",
            linewidth=2,
            label="Equivalent currents - MATLAB",
        )
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].set_xlim(-90, 90)
        axes[1, column].set_xlabel("Signed angle from +z (degrees)")
        axes[1, column].grid(alpha=0.35)

    axes[0, 0].set_ylabel("Normalised co-polar pattern (dB)")
    axes[1, 0].set_ylabel("Difference from MATLAB (dB)")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    fig.suptitle("Rectangular patch at 2.37 GHz: independent far-field formulations")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figure_path = RESULTS_DIR / f"{OUTPUT_STEM}.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return csv_path, json_path, figure_path, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="read the existing comparison HDF5 file without rerunning gprMax",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_base = RESULTS_DIR / OUTPUT_STEM
    angle = np.arange(-180.0, 180.0 + 2.0, 2.0)
    if not args.postprocess_only:
        scene, angle = build_scene(feed_mode="distributed")
        # Geometry was already inspected in the original benchmark. Avoid
        # rewriting its 8 MB fine-edge VTKHDF file during this formulation check.
        scene.output_objects = [
            item for item in scene.output_objects if not isinstance(item, gprMax.GeometryView)
        ]
        _add_equivalent_current_outputs(scene, angle)
        options = {}
        if args.gpu is not None:
            options.update(gpu=[args.gpu], gpu_precision="single")
        gprMax.run(
            scenes=[scene],
            outputfile=output_base,
            hide_progress_bars=False,
            log_level=logging.INFO,
            **options,
        )

    h5_path = output_base.with_suffix(".h5")
    patterns = _read_gprmax_patterns(h5_path, angle)
    matlab = _read_matlab_pattern(RESULTS_DIR / "patch_antenna_matlab_pattern.csv", angle)
    csv_path, json_path, figure_path, metrics = _write_results(angle, matlab, patterns)
    print(json.dumps(metrics, indent=2))
    print(f"Read persisted fields from {h5_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
