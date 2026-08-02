"""Sample finite-radius GPR antenna field-intensity patterns."""

import argparse
import json
from pathlib import Path

import numpy as np

import gprMax
from toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT = SCRIPT_PATH.with_suffix("")


def centred_grid_extent(length, dl):
    """Round an extent up to an even number of cells."""

    cells = int(np.ceil(length / dl))
    if cells % 2:
        cells += 1
    return cells * dl


def build_scene(pattern="E", dl=0.001, impedance_scaling=False):
    """Build the GSSI-like antenna and its finite-radius receiver circles."""

    radii = np.linspace(0.1, 0.3, 20)
    theta_degrees = np.linspace(3, 357, 60)
    theta = np.deg2rad(theta_degrees)
    free_space = np.array([0.040, 0.040, 0.040])

    if pattern == "E":
        centred_extent = centred_grid_extent(2 * free_space[0] + 0.170, dl)
        domain_size = np.array(
            [
                centred_extent,
                2 * free_space[1] + 2 * radii[-1],
                2 * free_space[2] + 2 * radii[-1],
            ]
        )
        antenna_position = np.array(
            [domain_size[0] / 2, free_space[1] + radii[-1], free_space[2] + radii[-1]]
        )
    else:
        centred_extent = centred_grid_extent(2 * free_space[1] + 0.108, dl)
        domain_size = np.array(
            [
                2 * free_space[0] + 2 * radii[-1],
                centred_extent,
                2 * free_space[2] + 2 * radii[-1],
            ]
        )
        antenna_position = np.array(
            [free_space[0] + radii[-1], domain_size[1] / 2, free_space[2] + radii[-1]]
        )

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f"GPR antenna {pattern}-plane field-intensity pattern"))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.PMLThickness(thickness=14))
    scene.add(gprMax.Domain(p1=tuple(domain_size)))
    scene.add(gprMax.TimeWindow(time=4.5e-9))

    for obj in antenna_like_GSSI_1500(*antenna_position, resolution=dl):
        scene.add(obj)

    # Homogeneous, lossless half-space with its interface at the antenna base.
    scene.add(gprMax.Material(er=5, se=0, mr=1, sm=0, id="er5"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(domain_size[0], domain_size[1], free_space[2] + radii[-1]),
            material_id="er5",
        )
    )

    receiver_prefix = f"gpr_pattern_{pattern}_"
    receiver_outputs = ["Ex", "Ey", "Ez"] if pattern == "E" else ["Hx", "Hy", "Hz"]
    for radius_index, radius in enumerate(radii):
        if pattern == "E":
            offsets = np.column_stack(
                (
                    np.zeros(theta.size),
                    radius * np.sin(theta),
                    radius * np.cos(theta),
                )
            )
        else:
            offsets = np.column_stack(
                (
                    -radius * np.sin(theta),
                    np.zeros(theta.size),
                    radius * np.cos(theta),
                )
            )

        for angle_index, offset in enumerate(offsets):
            scene.add(
                gprMax.Rx(
                    p1=tuple(antenna_position + offset),
                    id=f"{receiver_prefix}r{radius_index:03d}_a{angle_index:03d}",
                    outputs=receiver_outputs,
                )
            )

    metadata = {
        "pattern": pattern,
        "origin": antenna_position.tolist(),
        "radii": radii.tolist(),
        "theta_degrees": theta_degrees.tolist(),
        "receiver_prefix": receiver_prefix,
        "relative_permittivity": 5,
        "relative_permeability": 1,
        "impedance_scaling": impedance_scaling,
        "centre_frequency": 1.5e9,
        "antenna_dimension": 0.060,
    }
    return scene, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", choices=("E", "H"), default="E", help="principal plane")
    parser.add_argument(
        "--resolution",
        type=float,
        choices=(0.001, 0.002),
        default=0.001,
        help="spatial resolution in metres",
    )
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument(
        "--impedance-scaling",
        action="store_true",
        help="scale samples using the impedance of the homogeneous half-space",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="output path without the .h5 suffix",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.suffix:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    scene, metadata = build_scene(
        pattern=args.pattern,
        dl=args.resolution,
        impedance_scaling=args.impedance_scaling,
    )
    configfile = output.with_name(f"{output.name}_pattern_config.json")
    configfile.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Written pattern metadata: {configfile}")

    options = {}
    if args.gpu is not None:
        options["gpu"] = [args.gpu]
    gprMax.run(scenes=[scene], outputfile=output, **options)


if __name__ == "__main__":
    main()
