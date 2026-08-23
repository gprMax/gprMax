"""Compare finite-radius GSSI-like energy patterns with layered NTFF.

This reproduces the lossless epsilon_r=5 half-space configuration and the
0.10--0.58 m observation-radius sequence of Warren and Giannopoulos.  The
finite-radius curves use their time-domain field-energy measure.  A broadband
asymptotic reference is formed independently by integrating the planar-layered
equivalent-current NTFF spectrum over frequency using Parseval's relation.

Reference
---------
C. Warren and A. Giannopoulos, "Characterisation of a ground penetrating
radar antenna in lossless homogeneous and lossy heterogeneous environments,"
Signal Processing, 132, 221--226, 2017,
doi:10.1016/j.sigpro.2016.04.010.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants

import gprMax
from toolboxes.AntennaPatterns.initial_save import process_pattern
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "gssi_energy_convergence"

EPSR = 5.0
MUR = 1.0
ETA0 = physical_constants["characteristic impedance of vacuum"][0]
RADII = np.arange(0.10, 0.581, 0.02)
ANGLES = np.concatenate((np.arange(3.0, 90.0, 3.0), np.arange(93.0, 270.0, 3.0), np.arange(273.0, 360.0, 3.0)))
HALF_THETA = np.concatenate((np.arange(3.0, 90.0, 3.0), np.arange(93.0, 181.0, 3.0)))
FREQUENCIES = np.arange(0.1e9, 4.01e9, 0.1e9)
TIME_WINDOW = 8.0e-9


def _centred_extent(length: float, dl: float) -> float:
    cells = int(np.ceil(length / dl))
    if cells % 2:
        cells += 1
    return cells * dl


def build_scene(pattern: str, dl: float):
    """Build one finite-radius principal-plane and layered-NTFF model."""

    pattern = pattern.upper()
    free_space = 0.040
    diameter = 2 * RADII[-1]
    if pattern == "E":
        domain = np.array((_centred_extent(0.170 + 2 * free_space, dl), diameter + 2 * free_space, diameter + 2 * free_space))
        origin = np.array((domain[0] / 2, free_space + RADII[-1], free_space + RADII[-1]))
    elif pattern == "H":
        domain = np.array((diameter + 2 * free_space, _centred_extent(0.108 + 2 * free_space, dl), diameter + 2 * free_space))
        origin = np.array((free_space + RADII[-1], domain[1] / 2, free_space + RADII[-1]))
    else:
        raise ValueError("pattern must be E or H")

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f"GSSI-like {pattern}-plane broadband energy convergence"))
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.PMLThickness(thickness=14))
    scene.add(gprMax.Domain(p1=tuple(domain)))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    for obj in antenna_like_GSSI_1500(*origin, resolution=dl):
        scene.add(obj)

    scene.add(gprMax.Material(er=EPSR, se=0, mr=MUR, sm=0, id="er5"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(domain[0], domain[1], origin[2]), material_id="er5"))

    prefix = f"gssi_energy_{pattern}_"
    components = ["Ex", "Ey", "Ez"] if pattern == "E" else ["Hx", "Hy", "Hz"]
    theta = np.deg2rad(ANGLES)
    for radius_index, radius in enumerate(RADII):
        if pattern == "E":
            offsets = np.column_stack((np.zeros(theta.size), radius * np.sin(theta), radius * np.cos(theta)))
        else:
            offsets = np.column_stack((-radius * np.sin(theta), np.zeros(theta.size), radius * np.cos(theta)))
        for angle_index, offset in enumerate(offsets):
            scene.add(
                gprMax.Rx(
                    p1=tuple(origin + offset),
                    id=f"{prefix}r{radius_index:03d}_a{angle_index:03d}",
                    outputs=components,
                )
            )

    # The surface encloses the complete antenna and crosses the planar
    # interface only where the specified layered background is homogeneous.
    scene.add(
        gprMax.NTFFSurface(
            p1=(origin[0] - 0.090, origin[1] - 0.060, origin[2] - 0.012),
            p2=(origin[0] + 0.090, origin[1] + 0.060, origin[2] + 0.052),
            id="surface",
            origin=tuple(origin),
        )
    )
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="half_space",
            axis="z",
            materials=("free_space", "er5"),
            interfaces=(origin[2],),
        )
    )
    scene.add(
        gprMax.NTFFLayeredFrequencyTransform(
            surface_id="surface",
            id="spectrum",
            background_id="half_space",
            frequencies=FREQUENCIES,
            window="rectangular",
            save_surface_dft=False,
        )
    )
    if pattern == "E":
        cuts = (("first_half", 90.0), ("second_half", 270.0))
    else:
        cuts = (("first_half", 180.0), ("second_half", 0.0))
    for output_id, phi in cuts:
        scene.add(
            gprMax.NTFFFarField(
                theta=HALF_THETA,
                phi=np.full(HALF_THETA.shape, phi),
                transform_id="spectrum",
                id=output_id,
                outputs=("Etheta", "Ephi"),
            )
        )

    metadata = {
        "pattern": pattern,
        "origin": origin.tolist(),
        "radii": RADII.tolist(),
        "theta_degrees": ANGLES.tolist(),
        "receiver_prefix": prefix,
        "relative_permittivity": EPSR,
        "relative_permeability": MUR,
        "impedance_scaling": False,
        "centre_frequency": 1.5e9,
        "antenna_dimension": 0.060,
        "spatial_resolution": dl,
    }
    return scene, metadata


def _full_cut(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Assemble increasing 3--357 degree data from two spherical half-cuts."""

    return np.concatenate((first, second[..., -2::-1]), axis=-1)


def _ntff_energy(output_path: Path, pattern: str) -> np.ndarray:
    with h5py.File(output_path, "r") as output:
        root = output["ntff/surface/frequency/spectrum/far_field"]
        fields_first = root["first_half/fields"]
        fields_second = root["second_half/fields"]
        component = "Etheta" if pattern == "E" else "Ephi"
        spectrum = _full_cut(np.asarray(fields_first[component]), np.asarray(fields_second[component]))
        frequencies = np.asarray(output["ntff/surface/frequency/spectrum/frequencies"])

    if pattern == "H":
        ground = (ANGLES > 90) & (ANGLES < 270)
        impedance = np.full(ANGLES.size, ETA0)
        impedance[ground] = ETA0 * np.sqrt(MUR / EPSR)
        spectrum = spectrum / impedance[None, :]
    return np.trapezoid(np.abs(spectrum) ** 2, frequencies, axis=0)


def _finite_band_capture(output_path: Path, config_path: Path) -> dict[str, float]:
    """Measure how much sampled finite-radius field energy lies in the NTFF band."""

    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    pattern = config["pattern"].upper()
    origin = np.asarray(config["origin"], dtype=np.float64)
    prefix = config["receiver_prefix"]
    components = ("Ex", "Ey", "Ez") if pattern == "E" else ("Hx", "Hy", "Hz")

    with h5py.File(output_path, "r") as output:
        receivers = []
        for group_name, group in output["rxs"].items():
            name = str(group.attrs.get("Name", ""))
            if name.startswith(prefix):
                receivers.append((name, group_name))
        receivers.sort()
        iterations = int(output.attrs["Iterations"])
        dt = float(output.attrs["dt"])
        coords = np.empty((len(receivers), 3), dtype=np.float64)
        fields = np.empty((iterations, len(receivers), 3), dtype=np.float64)
        for receiver_index, (_, group_name) in enumerate(receivers):
            group = output[f"rxs/{group_name}"]
            coords[receiver_index] = np.asarray(group.attrs["Position"]) - origin
            for component_index, component in enumerate(components):
                fields[:, receiver_index, component_index] = group[component]

    radial_distance = np.linalg.norm(coords, axis=1)
    cylindrical_radius = np.hypot(coords[:, 0], coords[:, 1])
    theta_basis = np.column_stack(
        (
            coords[:, 0] * coords[:, 2] / (cylindrical_radius * radial_distance),
            coords[:, 1] * coords[:, 2] / (cylindrical_radius * radial_distance),
            -cylindrical_radius / radial_distance,
        )
    )
    theta_field = np.einsum("irc,rc->ir", fields, theta_basis, optimize=True)
    spectrum = np.fft.rfft(theta_field, axis=0)
    frequencies = np.fft.rfftfreq(iterations, dt)
    weights = np.full(frequencies.size, 2.0)
    weights[0] = 1.0
    if iterations % 2 == 0:
        weights[-1] = 1.0
    spectral_energy = weights[:, None] * np.abs(spectrum) ** 2
    total = np.sum(spectral_energy, axis=0)
    in_band = (frequencies >= FREQUENCIES[0]) & (frequencies <= FREQUENCIES[-1])
    captured = np.sum(spectral_energy[in_band], axis=0) / total
    significant = total >= 1e-6 * np.max(total)
    return {
        "frequency_min_hz": float(FREQUENCIES[0]),
        "frequency_max_hz": float(FREQUENCIES[-1]),
        "aggregate_fraction": float(np.sum(spectral_energy[in_band]) / np.sum(total)),
        "minimum_significant_trace_fraction": float(np.min(captured[significant])),
        "fifth_percentile_significant_trace_fraction": float(np.percentile(captured[significant], 5)),
        "median_significant_trace_fraction": float(np.median(captured[significant])),
        "significant_trace_count": int(np.count_nonzero(significant)),
    }


def _normalise(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=-1, keepdims=True)
    if np.any(maximum <= 0):
        raise ValueError("cannot normalise a zero energy pattern")
    return values / maximum


def analyse(output_directory: Path):
    results = {}
    summary = {
        "reference": "Warren and Giannopoulos, Signal Processing 132 (2017), 221-226",
        "relative_permittivity": EPSR,
        "radii_m": RADII.tolist(),
    }
    for pattern in ("E", "H"):
        finite_path = output_directory / f"gssi_energy_{pattern}.npz"
        with np.load(finite_path) as data:
            finite = np.asarray(data["patterns"], dtype=float)
        far = _ntff_energy(output_directory / f"gssi_energy_{pattern}.h5", pattern)
        finite_normalised = _normalise(finite)
        far_normalised = _normalise(far[None, :])[0]
        differences = finite_normalised - far_normalised[None, :]
        regular = ~(((ANGLES > 84) & (ANGLES < 96)) | ((ANGLES > 264) & (ANGLES < 276)))
        rms = np.sqrt(np.mean(differences[:, regular] ** 2, axis=1))
        maximum = np.max(np.abs(differences[:, regular]), axis=1)
        results[pattern] = {
            "finite": finite_normalised,
            "far": far_normalised,
            "rms": rms,
            "maximum": maximum,
        }
        summary[pattern] = {
            "rms_normalised_energy_error": rms.tolist(),
            "maximum_normalised_energy_error": maximum.tolist(),
            "rms_at_0p10_m": float(rms[0]),
            "rms_at_0p58_m": float(rms[-1]),
            "maximum_at_0p58_m": float(maximum[-1]),
            "finite_record_energy_in_ntff_band": _finite_band_capture(
                output_directory / f"gssi_energy_{pattern}.h5",
                output_directory / f"gssi_energy_{pattern}_pattern_config.json",
            ),
        }
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return results, summary


def _db(values: np.ndarray, floor: float = -40.0) -> np.ndarray:
    return np.maximum(10 * np.log10(np.maximum(values, np.finfo(float).tiny)), floor)


def plot(results, output_directory: Path):
    figure = plt.figure(figsize=(12.0, 10.0))
    grid = figure.add_gridspec(2, 2, height_ratios=(3.0, 1.25), hspace=0.30)
    selected = (0, 5, 10, 15, 20, 24)
    theta = np.deg2rad(ANGLES)
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(selected)))
    legend_handles = None
    legend_labels = None
    for column, pattern in enumerate(("E", "H")):
        axis = figure.add_subplot(grid[0, column], projection="polar")
        for colour, radius_index in zip(colours, selected):
            axis.plot(theta, _db(results[pattern]["finite"][radius_index]), color=colour, lw=1.15, label=f"{RADII[radius_index]:.2f} m")
        axis.plot(theta, _db(results[pattern]["far"]), "k--", lw=2.0, label="broadband layered NTFF")
        critical = np.arcsin(1 / np.sqrt(EPSR))
        for angle in (np.pi - critical, np.pi + critical):
            axis.plot((angle, angle), (-40, 0), color="0.72", ls=":", lw=0.9)
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_rlim(-40, 0)
        axis.set_rticks((-40, -30, -20, -10, 0))
        axis.set_rlabel_position(45)
        axis.grid(True, alpha=0.3)
        axis.set_title(f"{pattern}-plane normalised field energy", pad=6)
        if column == 0:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

        error_axis = figure.add_subplot(grid[1, column])
        error_axis.plot(RADII, results[pattern]["rms"], "ko-", ms=3.5, label="RMS")
        error_axis.plot(RADII, results[pattern]["maximum"], "ks--", ms=3.0, markerfacecolor="white", label="maximum")
        error_axis.set_xlabel("observation radius (m)")
        error_axis.set_ylabel("normalised energy difference")
        error_axis.grid(True, alpha=0.3)
        error_axis.legend(fontsize=8)
    figure.suptitle(r"GSSI-like 1.5 GHz antenna over a lossless $\epsilon_r=5$ half-space", y=0.995)
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        fontsize=8,
        ncol=7,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output_directory / "gssi_energy_convergence.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=float, choices=(0.001, 0.002), default=0.002)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    for pattern in ("E", "H"):
        output_stem = output_directory / f"gssi_energy_{pattern}"
        config_path = output_directory / f"gssi_energy_{pattern}_pattern_config.json"
        if not args.no_run:
            scene, metadata = build_scene(pattern, args.resolution)
            config_path.write_text(json.dumps(metadata, indent=2) + "\n")
            gprMax.run(scenes=[scene], n=1, outputfile=output_stem, gpu=[args.gpu], hide_progress_bars=True)
        process_pattern(output_stem.with_suffix(".h5"), config_path, output_stem.with_suffix(".npz"))

    results, summary = analyse(output_directory)
    plot(results, output_directory)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
