"""Example 9 - opt-in tracking through a crossing in an anisotropic guide.

Automatic mode tracking is under development. Run this example in geometry
mode first and verify that each label keeps the same polarization while the
two phase-index curves cross.
"""

import argparse
from pathlib import Path

import gprMax

EXAMPLE_DIR = Path(__file__).resolve().parent
ANCHORS = (
    12.8e9,
    12.9e9,
    13.0e9,
    13.1e9,
    13.2e9,
    13.3e9,
    13.4e9,
    13.5e9,
    13.6e9,
    13.8e9,
    14.0e9,
    14.15e9,
    14.3e9,
    14.6e9,
    14.9e9,
    15.2e9,
    15.5e9,
    15.8e9,
    16.1e9,
)


def build_scene(mode=1):
    """Build one anisotropic PEC guide whose polarized branches cross."""
    if mode not in (1, 2):
        raise ValueError("Choose tracked mode 1 or 2.")

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Example 9 - automatic tracking through a mode crossing"))
    scene.add(gprMax.Domain(p1=(0.020, 0.012, 0.096)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=20e-9))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.PMLThickness(thickness=(0, 0, 16, 0, 0, 16)))
    scene.add(gprMax.Material(er=2.25, se=0, mr=1, sm=0, id="epsilon_x"))
    scene.add(gprMax.Material(er=1, se=0, mr=1, sm=0, id="epsilon_y"))
    scene.add(gprMax.Material(er=1, se=0, mr=1, sm=0, id="epsilon_z"))

    # A single 16 x 8 mm bore is filled with the diagonal tensor
    # eps_r = diag(2.25, 1, 1). The Ex- and Ey-polarized fundamental branches
    # have different cutoff terms and slopes, so their propagation constants
    # cross near 14.4 GHz on this voxel grid. Raw eigensolver order changes at
    # the crossing while automatic tracking preserves each polarization.
    scene.add(
        gprMax.Box(
            p1=(0.002, 0.002, 0),
            p2=(0.018, 0.010, 0.096),
            material_ids=("epsilon_x", "epsilon_y", "epsilon_z"),
            averaging="n",
        )
    )
    # Build the PEC walls after the anisotropic fill, whose Yee components are
    # assigned without averaging. This preserves the tangential-E constraints
    # shared by the FDTD geometry and modal solver at all four walls.
    for x0, y0, x1, y1 in (
        (0, 0, 0.002, 0.012),
        (0.018, 0, 0.020, 0.012),
        (0.002, 0, 0.018, 0.002),
        (0.002, 0.010, 0.018, 0.012),
    ):
        scene.add(gprMax.Box(p1=(x0, y0, 0), p2=(x1, y1, 0.096), material_id="pec"))

    scene.add(
        gprMax.EigenmodeBand(
            id="crossing",
            fmin=13.1e9,
            fmax=15.8e9,
            points=55,
            transition=0.3e9,
        )
    )
    # Denser low-frequency anchors resolve the rapid impedance change of the
    # second branch near cutoff. Tracking identity alone does not control the
    # interpolation error of a broadband source or modal measurement.
    for port, z, direction in ((1, 0.028, "+"), (2, 0.080, "-")):
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(0.001, 0.001, z),
                p2=(0.019, 0.011, z),
                direction=direction,
                modes=(1, 2),
                anchors=ANCHORS,
                tracking="auto",
                verification="fast",
                plot_fields=True,
            )
        )
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=mode, waveform="auto"))
    scene.add(gprMax.Rx(p1=(0.010, 0.006, 0.058), id="guide_centre"))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="solve and plot the crossing dispersion and tracked modal fields",
    )
    parser.add_argument("--output", type=Path, help="output path without .h5")
    args = parser.parse_args()
    output = args.output or EXAMPLE_DIR / f"auto_mode_crossing_mode{args.mode}"
    gprMax.run(
        scenes=[build_scene(mode=args.mode)],
        outputfile=output,
        geometry_only=args.geometry_only,
        cpu_precision="double",
    )


if __name__ == "__main__":
    main()
