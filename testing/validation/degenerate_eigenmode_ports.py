"""Circular TE11 channels: change mode=1 to mode=2 for the other polarization.

Run: python -m testing.validation.degenerate_eigenmode_ports --mode 1
"""
import argparse
from pathlib import Path

import numpy as np

import gprMax


def circular_scene(
    *,
    mode=1,
    normal_axis=2,
    direction="+",
    virtual=True,
    broadband=False,
    combination=None,
    steps=400,
    monitor=False,
    tracking="legacy",
    verification="full",
):
    dl = 1e-3
    transverse = tuple(axis for axis in range(3) if axis != normal_axis)
    basis = (*transverse, normal_axis)

    def point(local):
        result = np.zeros(3)
        result[list(basis)] = np.asarray(local) * dl
        return tuple(result)

    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=point((16, 16, 72))),
        gprMax.Discretisation(p1=(dl, dl, dl)),
        gprMax.TimeWindow(iterations=steps),
        gprMax.OMPThreads(1),
    ):
        scene.add(obj)
    pml = [0] * 6
    pml[normal_axis] = pml[normal_axis + 3] = 8
    scene.add(gprMax.PMLThickness(thickness=tuple(pml)))
    scene.add(gprMax.Box(p1=point((0, 0, 0)), p2=point((16, 16, 72)), material_id="pec"))
    scene.add(
        gprMax.Cylinder(
            p1=point((8, 8, 0)),
            p2=point((8, 8, 72)),
            r=6e-3,
            material_id="free_space",
            averaging="n",
        )
    )
    plane = (24 if virtual else 12) if direction == "+" else (48 if virtual else 60)
    scene.add(
        gprMax.EigenmodeBand(
            id="band",
            fmin=20e9 if broadband else 22e9,
            fmax=24e9 if broadband else 22e9,
            points=5 if broadband else 1,
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=point((1, 1, plane)),
            p2=point((15, 15, plane)),
            direction=direction,
            modes=(1, 2),
            anchors="auto" if broadband else (22e9,),
            plot_fields=False,
            tracking=tracking,
            verification=verification,
            degenerate=(1, 2) if tracking == "legacy" else None,
            mode_polarizations={1: "xyz"[transverse[1]], 2: "xyz"[transverse[0]]},
        )
    )
    if monitor:
        scene.add(
            gprMax.EigenmodePort(
                port=2,
                p1=point((1, 1, 40)),
                p2=point((15, 15, 40)),
                direction="-",
                modes=(1, 2),
                anchors="auto" if broadband else (22e9,),
                plot_fields=False,
                tracking=tracking,
                verification=verification,
                degenerate=(1, 2) if tracking == "legacy" else None,
                mode_polarizations={1: "xyz"[transverse[1]], 2: "xyz"[transverse[0]]},
            )
        )
    if virtual:
        scene.add(
            gprMax.VirtualWaveguide(port=1, length_cells=24, pml_cells=8, source_clearance_cells=4)
        )
    if not broadband:
        scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=22e9, id="wave"))
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1, mode=mode, waveform="auto" if broadband else "wave", plot_waveform=False
        )
    )
    if combination is not None:
        scene.add(
            gprMax.EigenmodeExcitation(
                port=1,
                mode=3 - mode,
                amplitude=0.7,
                phase_deg=combination,
                waveform="auto" if broadband else "wave",
                plot_waveform=False,
            )
        )
    for u, v in ((8, 8), (6, 9)):
        scene.add(gprMax.Rx(p1=point((u, v, 36)), id=f"probe{u}"))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path, default=Path("circular_te11"))
    args = parser.parse_args()
    gprMax.run(
        scenes=[circular_scene(mode=args.mode)], outputfile=args.output, cpu_precision="double"
    )


if __name__ == "__main__":
    main()
