"""Small source/port/plane-wave workflows; not converged antenna benchmarks.

Run from the repository root, for example:
    python examples/features/studies/run_study.py port --output /tmp/port_api
"""

import argparse
from pathlib import Path

import gprMax


def build_model(kind):
    """Return the reusable scene and study (None for the passive-Rx demo)."""
    if kind not in {"passive", "source", "port", "plane_wave"}:
        raise ValueError(f"Unknown tutorial: {kind}")
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.048,) * 3))
    scene.add(gprMax.Discretisation(p1=(0.002,) * 3))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))

    if kind == "plane_wave":
        wave = gprMax.DiscretePlaneWaveAngles(
            p1=(0.018,) * 3,
            p2=(0.030,) * 3,
            theta=90,
            phi=0,
            psi=90,
            waveform_id="pulse",
        )
        scene.add(wave)
        scene.add(gprMax.Sphere(p1=(0.024,) * 3, r=0.004, material_id="pec"))
        scene.add(
            gprMax.NTFFSurface(
                p1=(0.012,) * 3,
                p2=(0.036,) * 3,
                id="surface",
            )
        )
        scene.add(gprMax.KSIRFrequencyTransform("surface", "band", (5e9,)))
        for name, phi in (("back_x", 180), ("back_y", 270)):
            scene.add(
                gprMax.KSIRFarField(
                    theta=90,
                    phi=phi,
                    transform_id="band",
                    id=name,
                    outputs=("rcs",),
                )
            )
        study = gprMax.PlaneWaveStudy(
            [
                gprMax.StudyCase(name, [gprMax.ObjectState(wave, theta=90, phi=phi, psi=90)])
                for name, phi in (("from_x", 0), ("from_y", 90))
            ]
        )
        return scene, study

    points = ((0.020, 0.024, 0.024), (0.028, 0.024, 0.024))
    if kind == "source":
        scene.add(gprMax.RationalNetwork(id="load50", conductance=1 / 50))
        drives = []
        for name, point in zip(("feed", "receive"), points):
            scene.add(gprMax.NetworkTerminal(point, "z", "load50", name))
            scene.add(gprMax.NetworkPort(name))
            drive = gprMax.NetworkExcitation(name, "pulse")
            scene.add(drive)
            drives.append(drive)
        study = gprMax.SourceStudy(
            [
                gprMax.StudyCase("feed_only", [gprMax.ObjectState(drives[0], scale=1)]),
                gprMax.StudyCase(
                    "opposed",
                    [
                        gprMax.ObjectState(drives[0], scale=1),
                        gprMax.ObjectState(drives[1], scale=-1),
                    ],
                ),
            ]
        )
        return scene, study

    scene.add(gprMax.Waveform(wave_type="ricker", amp=0, freq=5e9, id="off"))
    feed = gprMax.VoltageSource(points[0], "z", 50, "pulse", id="feed")
    receive = gprMax.VoltageSource(
        points[1],
        "z",
        50,
        "off" if kind == "passive" else "pulse",
        id="receive",
    )
    scene.add(feed)
    scene.add(receive)
    if kind == "passive":
        return scene, None
    study = gprMax.PortStudy(
        [
            gprMax.StudyCase("drive_feed", [gprMax.ObjectState(feed)]),
            gprMax.StudyCase("drive_receive", [gprMax.ObjectState(receive)]),
        ]
    )
    return scene, study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("passive", "source", "port", "plane_wave"))
    parser.add_argument("--output", type=Path, required=True, help="Output filename prefix")
    parser.add_argument("--gpu", type=int, help="CUDA device index; default is CPU")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene, study = build_model(args.kind)
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=args.output,
        gpu=None if args.gpu is None else [args.gpu],
    )


if __name__ == "__main__":
    main()
