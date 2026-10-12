"""Persistent doublets cross in a square bore as axial permittivity varies.

Geometry-only port snapshots accompany the undriven FDFD parameter sweep.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import gprMax
from examples.features.eigenmode_tracking.plot_results import plot_case
from testing.validation.eigenmode_tracking.branch_crossing.square_anisotropic_bore import run


def build_scene(epsilon_z, name, spacing=.00025):
    length = 4*spacing
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f'Persistent square-guide doublets: epsilon_z={epsilon_z}'))
    scene.add(gprMax.Domain(p1=(.024, .024, length)))
    scene.add(gprMax.Discretisation(p1=(spacing,)*3))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(n=1))
    for material, epsilon in [('transverse', 2.), ('axial', epsilon_z)]:
        scene.add(gprMax.Material(er=epsilon, se=0, mr=1, sm=0, id=material))
    scene.add(gprMax.Box(p1=(.002, .002, 0), p2=(.022, .022, length),
              material_ids=('transverse', 'transverse', 'axial'), averaging='n'))
    for x0, y0, x1, y1 in ((0, 0, .002, .024), (.022, 0, .024, .024),
                           (.002, 0, .022, .002), (.002, .022, .022, .024)):
        scene.add(gprMax.Box(p1=(x0, y0, 0), p2=(x1, y1, length), material_id='pec'))
    scene.add(gprMax.EigenmodeBand(id='inspection', fmin=19.9e9, fmax=20.1e9, points=3))
    scene.add(gprMax.EigenmodePort(port=1, p1=(spacing, spacing, length/2),
              p2=(.024-spacing, .024-spacing, length/2), direction='+', modes=8,
              anchors=(20e9,), tracking='auto', verification='fast'))
    scene.add(gprMax.EigenmodeFieldOutput(filename=name, ports=(1,)))
    return scene


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spacing-mm', type=float, default=.25)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a fresh output directory to preserve previous results.')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, epsilon in [('before_crossing', 2.3), ('after_crossing', 2.7)]:
        gprMax.run(scenes=[build_scene(epsilon, name, args.spacing_mm*1e-3)],
                   outputfile=args.output/name, geometry_only=True,
                   cpu_precision='double', hide_progress_bars=True, log_level=30)
        plot_case(args.output/f'{name}.modes.h5')
    run(args.output/'parameter_sweep')
