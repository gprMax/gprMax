"""Run matched PEC and copper surface-impedance rectangular waveguides.

The guide propagates in +x and supports only TE10 over the requested band.
Its four walls are ordinary :class:`gprMax.Box` objects.  The PEC case assigns
the built-in ``pec`` material; the copper case assigns a ``SurfaceImpedance``
ID in exactly the same ``material_id`` position.

The walls stop before the x-directed PML.  The short time window ends before
the earliest wall-end reflection can return to the source or receivers, so
the receiver comparison demonstrates the local wall law rather than a guide
termination.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gprMax

EXAMPLE_DIR = Path(__file__).resolve().parent

DL = 0.1e-3
DOMAIN = (0.040, 0.0028, 0.0020)
PML_CELLS = 3
TIME_WINDOW = 100e-12

GUIDE_LOWER = (0.0006, 0.0006)
GUIDE_UPPER = (0.0022, 0.0014)
WALL_OUTER_LOWER = (0.0004, 0.0004)
WALL_OUTER_UPPER = (0.0024, 0.0016)
WALL_X = (4 * DL, 396 * DL)

PORT_X = 0.020
RECEIVER_X = 0.025
FMIN = 130e9
FMAX = 150e9
FIT_FREQUENCY_RANGE = (80e9, 200e9)
ANCHORS = (90e9, 110e9, 120e9, 130e9, 140e9, 150e9, 170e9, 190e9)

COPPER_MODEL_ID = "copper_wall"
FIT_TOLERANCE = 2e-3


def wall_boxes() -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Return the four thick boxes forming the rectangular guide wall."""

    x0, x1 = WALL_X
    y0, z0 = WALL_OUTER_LOWER
    y1, z1 = WALL_OUTER_UPPER
    yi0, zi0 = GUIDE_LOWER
    yi1, zi1 = GUIDE_UPPER
    return (
        ((x0, y0, z0), (x1, yi0, z1)),
        ((x0, yi1, z0), (x1, y1, z1)),
        ((x0, yi0, z0), (x1, yi1, zi0)),
        ((x0, yi0, zi1), (x1, yi1, z1)),
    )


def build_scene(case: str, threads: int = 1) -> gprMax.Scene:
    """Build one PEC or copper TE10 guide scene."""

    case = str(case).lower()
    if case not in {"pec", "copper"}:
        raise ValueError("case must be 'pec' or 'copper'")

    scene = gprMax.Scene()
    scene.add(
        gprMax.Title(
            name=f"Rectangular TE10 guide: {case} walls",
        )
    )
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(gprMax.OMPThreads(n=threads))

    wall_material = "pec"
    if case == "copper":
        scene.add(
            gprMax.SurfaceImpedance(
                id=COPPER_MODEL_ID,
                preset="copper",
                fit_frequency_range=FIT_FREQUENCY_RANGE,
                fit_order="auto",
                fit_tolerance=FIT_TOLERANCE,
                plot_fit=True,
            )
        )
        wall_material = COPPER_MODEL_ID

    for lower, upper in wall_boxes():
        scene.add(
            gprMax.Box(
                p1=lower,
                p2=upper,
                material_id=wall_material,
                averaging="n",
            )
        )

    scene.add(
        gprMax.EigenmodeBand(
            id="te10_band",
            fmin=FMIN,
            fmax=FMAX,
            points=5,
            transition=40e9,
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(PORT_X, GUIDE_LOWER[0], GUIDE_LOWER[1]),
            p2=(PORT_X, GUIDE_UPPER[0], GUIDE_UPPER[1]),
            direction="+",
            modes=(1,),
            anchors=ANCHORS,
            plot_fields=True,
        )
    )
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=1,
            waveform="auto",
            plot_waveform=False,
        )
    )

    aperture_centre = (
        0.5 * (GUIDE_LOWER[0] + GUIDE_UPPER[0]),
        0.5 * (GUIDE_LOWER[1] + GUIDE_UPPER[1]),
    )
    # TE10 has dominant Ez for propagation along x. At the y-normal side
    # wall Ez is tangential: PEC forces it to zero, while copper permits the
    # small finite value required by E_t = Z_s (n x H).
    scene.add(
        gprMax.Rx(
            p1=(RECEIVER_X, aperture_centre[0], aperture_centre[1]),
            id="aperture_centre",
            outputs=["Ez"],
        )
    )
    scene.add(
        gprMax.Rx(
            p1=(RECEIVER_X, GUIDE_LOWER[0], aperture_centre[1]),
            id="lower_sidewall_tangential",
            outputs=["Ez"],
        )
    )
    return scene


def _preserve_eigenmode_plot(output_stem: Path) -> None:
    """Rename the standard port plot to a stable, ignored example output."""

    generated = output_stem.with_name(f"{output_stem.name}_Port1_Mode1.png")
    destination = output_stem.with_name(f"{output_stem.name}_eigenmode_fields.png")
    if generated.is_file():
        generated.replace(destination)
        print(f"Wrote {destination}")


def run_case(
    case: str,
    *,
    output_dir: Path,
    threads: int,
    geometry_only: bool,
) -> None:
    """Run one case in double precision and retain its modal-field plot."""

    output_stem = output_dir / f"{case}_rectangular_waveguide"
    gprMax.run(
        scenes=[build_scene(case, threads)],
        outputfile=output_stem,
        geometry_only=geometry_only,
        cpu_precision="double",
        hide_progress_bars=False,
        log_level=logging.INFO,
    )
    _preserve_eigenmode_plot(output_stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("both", "pec", "copper"),
        default="both",
        help="run both matched cases or only one of them",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR,
        help="directory for HDF5 and modal-field outputs",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="build geometry and FDFD mode plots without advancing FDTD",
    )
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = ("pec", "copper") if args.case == "both" else (args.case,)
    for case in cases:
        run_case(
            case,
            output_dir=args.output_dir,
            threads=args.threads,
            geometry_only=args.geometry_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
