# Copyright (C) 2016, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1016/j.sigpro.2016.04.010

"""Process receiver samples for the GPR antenna-pattern toolbox."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import physical_constants

Z0 = physical_constants["characteristic impedance of vacuum"][0]


def _text_attribute(value):
    """Return an HDF5 text attribute as a Python string."""

    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _default_config_path(outputfile):
    return outputfile.with_name(f"{outputfile.stem}_pattern_config.json")


def process_pattern(outputfile, configfile, destination):
    """Calculate a time-integrated field-intensity pattern."""

    with configfile.open(encoding="utf-8") as stream:
        config = json.load(stream)

    pattern = config["pattern"].upper()
    if pattern not in {"E", "H"}:
        raise ValueError("Pattern must be 'E' or 'H'.")

    radii = np.asarray(config["radii"], dtype=np.float64)
    theta_degrees = np.asarray(config["theta_degrees"], dtype=np.float64)
    origin = np.asarray(config["origin"], dtype=np.float64)
    receiver_prefix = config["receiver_prefix"]
    expected_receivers = radii.size * theta_degrees.size
    components = ("Ex", "Ey", "Ez") if pattern == "E" else ("Hx", "Hy", "Hz")

    with h5py.File(outputfile, "r") as output:
        if "rxs" not in output:
            raise ValueError(f"{outputfile} contains no receiver outputs.")

        receivers = []
        for group_name, group in output["rxs"].items():
            name = _text_attribute(group.attrs.get("Name", ""))
            if name.startswith(receiver_prefix):
                receivers.append((name, group_name))
        receivers.sort()

        if len(receivers) != expected_receivers:
            raise ValueError(
                f"Expected {expected_receivers} receivers beginning with "
                f"'{receiver_prefix}', but found {len(receivers)}."
            )

        iterations = int(output.attrs["Iterations"])
        dt = float(output.attrs["dt"])
        coords = np.empty((expected_receivers, 3), dtype=np.float64)
        fields = np.empty((iterations, expected_receivers, 3), dtype=np.float64)
        for receiver_index, (_, group_name) in enumerate(receivers):
            group = output[f"rxs/{group_name}"]
            coords[receiver_index] = np.asarray(group.attrs["Position"]) - origin
            for component_index, component in enumerate(components):
                if component not in group:
                    raise ValueError(
                        f"Receiver '{group_name}' does not contain required output '{component}'."
                    )
                fields[:, receiver_index, component_index] = group[component]

    radial_distance = np.linalg.norm(coords, axis=1)
    cylindrical_radius = np.hypot(coords[:, 0], coords[:, 1])
    if np.any(radial_distance == 0) or np.any(cylindrical_radius == 0):
        raise ValueError("Pattern receivers cannot lie at the origin or on the polar axis.")

    theta_basis = np.column_stack(
        (
            coords[:, 0] * coords[:, 2] / (cylindrical_radius * radial_distance),
            coords[:, 1] * coords[:, 2] / (cylindrical_radius * radial_distance),
            -cylindrical_radius / radial_distance,
        )
    )
    theta_field = np.einsum("irc,rc->ir", fields, theta_basis, optimize=True)

    epsr = config.get("relative_permittivity")
    mur = config.get("relative_permeability", 1.0)
    impedance = np.full(expected_receivers, Z0)
    if config.get("impedance_scaling", False) and epsr is not None:
        impedance[coords[:, 2] < 0] = Z0 * np.sqrt(mur / epsr)

    field_integral = np.sum(theta_field**2, axis=0, dtype=np.float64) * dt
    if pattern == "E":
        intensity = field_integral / impedance
    else:
        intensity = field_integral * impedance
    patterns = intensity.reshape(radii.size, theta_degrees.size)

    np.savez(
        destination,
        patterns=patterns,
        radii=radii,
        theta_degrees=theta_degrees,
        pattern=pattern,
        relative_permittivity=np.nan if epsr is None else epsr,
        relative_permeability=mur,
        centre_frequency=config["centre_frequency"],
        antenna_dimension=config["antenna_dimension"],
        impedance_scaling=config.get("impedance_scaling", False),
        metric=f"time-integrated {pattern}-theta field intensity",
    )
    print(f"Processed {expected_receivers} pattern receivers.")
    print(f"Written pattern data: {destination}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process finite-radius field-intensity patterns sampled around a GPR antenna."
        ),
        usage="python -m toolboxes.AntennaPatterns.initial_save outputfile",
    )
    parser.add_argument("outputfile", type=Path, help="gprMax HDF5 output file")
    parser.add_argument(
        "--config",
        type=Path,
        help="pattern metadata JSON; defaults to <output stem>_pattern_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="processed NPZ file; defaults to the HDF5 path with an .npz suffix",
    )
    args = parser.parse_args()

    outputfile = args.outputfile.resolve()
    configfile = args.config.resolve() if args.config else _default_config_path(outputfile)
    destination = args.output.resolve() if args.output else outputfile.with_suffix(".npz")
    if not configfile.is_file():
        raise FileNotFoundError(f"Pattern metadata file not found: {configfile}")
    process_pattern(outputfile, configfile, destination)


if __name__ == "__main__":
    main()
