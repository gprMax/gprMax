# Copyright (C) 2016, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1016/j.sigpro.2016.04.010

"""Plot finite-radius field-intensity patterns for a GPR antenna."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c


def load_pattern_data(filename):
    """Load the processed pattern data and metadata."""

    with np.load(filename) as data:
        required = {
            "patterns",
            "radii",
            "theta_degrees",
            "pattern",
            "relative_permittivity",
            "relative_permeability",
            "centre_frequency",
            "antenna_dimension",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Pattern data is missing: {', '.join(sorted(missing))}")
        return {key: np.asarray(data[key]) for key in data.files}


def plot_pattern(data, destination, minimum_db=-72, ring_step_db=12, show=False):
    """Create the polar GPR antenna field-intensity plot."""

    patterns = np.asarray(data["patterns"], dtype=np.float64)
    radii = np.asarray(data["radii"], dtype=np.float64)
    theta_degrees = np.asarray(data["theta_degrees"], dtype=np.float64)
    pattern = str(data["pattern"].item()).upper()
    epsr_value = float(data["relative_permittivity"].item())
    epsr = None if np.isnan(epsr_value) else epsr_value
    mur = float(data["relative_permeability"].item())
    frequency = float(data["centre_frequency"].item())
    antenna_dimension = float(data["antenna_dimension"].item())

    if patterns.shape != (radii.size, theta_degrees.size):
        raise ValueError(
            f"Pattern array shape {patterns.shape} does not match "
            f"{radii.size} radii and {theta_degrees.size} angles."
        )
    peak = np.max(patterns)
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Pattern data has no positive finite values.")

    theta = np.deg2rad(np.append(theta_degrees, theta_degrees[0]))
    figure = plt.figure(figsize=(8, 8), facecolor="white")
    axes = figure.add_subplot(111, polar=True)
    colourmap = plt.get_cmap("rainbow")
    axes.set_prop_cycle("color", [colourmap(value) for value in np.linspace(0, 1, radii.size)])

    if epsr is not None:
        critical_angle = np.rad2deg(np.arcsin(1 / np.sqrt(epsr * mur)))
        axes.plot([0, np.deg2rad(180 - critical_angle)], [minimum_db, 0], color="0.7", lw=2)
        axes.plot([0, np.deg2rad(180 + critical_angle)], [minimum_db, 0], color="0.7", lw=2)
    axes.plot([np.deg2rad(270), np.deg2rad(90)], [0, 0], color="0.7", lw=2)
    axes.annotate("Air", xy=(np.deg2rad(270), 0), xytext=(8, 8), textcoords="offset points")
    axes.annotate("Ground", xy=(np.deg2rad(270), 0), xytext=(8, -15), textcoords="offset points")

    floor = peak * 10 ** (minimum_db / 10)
    for radius_index, radius in enumerate(radii):
        values = np.append(patterns[radius_index], patterns[radius_index, 0])
        power_db = 10 * np.log10(np.maximum(values, floor) / peak)
        axes.plot(theta, power_db, label=f"{radius:.2f} m", marker=".", ms=6, lw=1.5)

    axes.set_theta_zero_location("N")
    axes.set_theta_direction(-1)
    axes.set_thetagrids(np.arange(0, 360, 30))
    axes.set_ylim(minimum_db, 0)
    axes.set_rlabel_position(45)
    ticks = np.arange(minimum_db, ring_step_db, ring_step_db)
    labels = [f"{tick:g}" for tick in ticks]
    labels[-1] = "0 dB"
    axes.set_yticks(ticks)
    axes.set_yticklabels(labels)
    axes.grid(True)
    axes.set_title(f"GPR antenna {pattern}-plane field-intensity pattern", pad=28)

    handles, labels = axes.get_legend_handles_labels()
    selected = [0] if len(handles) == 1 else [0, -1]
    legend = axes.legend(
        [handles[index] for index in selected],
        [labels[index] for index in selected],
        ncol=len(selected),
        loc=(0.27, -0.12),
        frameon=False,
    )
    for line in legend.get_lines():
        line.set_linewidth(2)

    if epsr is not None:
        velocity = c / np.sqrt(epsr * mur)
        wavelength = velocity / frequency
        print(f"Critical angle for relative permittivity {epsr:g}: {critical_angle:.1f} degrees")
        print(f"Wavelength in the homogeneous half-space: {wavelength:.3f} m")
        print(
            "Reactive/radiating near-field boundary: "
            f"{0.62 * np.sqrt(antenna_dimension**3 / wavelength):.3f} m"
        )
        print(
            "Radiating near-field/far-field boundary: "
            f"{2 * antenna_dimension**2 / wavelength:.3f} m"
        )

    figure.savefig(destination, bbox_inches="tight", pad_inches=0.1)
    print(f"Written plot: {destination}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Plot processed finite-radius GPR antenna field-intensity patterns.",
        usage="python -m toolboxes.AntennaPatterns.plot_fields patternfile",
    )
    parser.add_argument("patternfile", type=Path, help="processed NPZ pattern data")
    parser.add_argument("--output", type=Path, help="plot filename; defaults to PDF")
    parser.add_argument("--minimum-db", type=float, default=-72, help="radial plot minimum")
    parser.add_argument("--ring-step-db", type=float, default=12, help="radial ring spacing")
    parser.add_argument("--show", action="store_true", help="open the interactive plot window")
    args = parser.parse_args()

    patternfile = args.patternfile.resolve()
    destination = args.output.resolve() if args.output else patternfile.with_suffix(".pdf")
    plot_pattern(
        load_pattern_data(patternfile),
        destination,
        minimum_db=args.minimum_db,
        ring_step_db=args.ring_step_db,
        show=args.show,
    )


if __name__ == "__main__":
    main()
