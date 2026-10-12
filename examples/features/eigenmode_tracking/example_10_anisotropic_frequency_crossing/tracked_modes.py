"""Two polarized branches cross in frequency: geometry-only anisotropic bore."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import gprMax
from examples.features.eigenmode_tracking.plot_results import plot_case
from testing.regression.eigenmode_sources.auto_mode_crossing import ANCHORS


def build_scene(spacing=.00025):
    scene = gprMax.Scene()
    length = 4*spacing
    scene.add(gprMax.Title(name='Two polarized branches crossing in frequency'))
    scene.add(gprMax.Domain(p1=(.020, .012, length)))
    scene.add(gprMax.Discretisation(p1=(spacing,)*3))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(n=1))
    for name, epsilon in [('epsilon_x', 2.25), ('epsilon_y', 1.), ('epsilon_z', 1.)]:
        scene.add(gprMax.Material(er=epsilon, se=0, mr=1, sm=0, id=name))
    scene.add(gprMax.Box(p1=(.002, .002, 0), p2=(.018, .010, length),
                         material_ids=('epsilon_x', 'epsilon_y', 'epsilon_z'), averaging='n'))
    for x0, y0, x1, y1 in ((0, 0, .002, .012), (.018, 0, .020, .012),
                           (.002, 0, .018, .002), (.002, .010, .018, .012)):
        scene.add(gprMax.Box(p1=(x0, y0, 0), p2=(x1, y1, length), material_id='pec'))
    scene.add(gprMax.EigenmodeBand(id='crossing', fmin=13.1e9, fmax=15.8e9, points=55, transition=.3e9))
    scene.add(gprMax.EigenmodePort(port=1, p1=(spacing, spacing, length/2),
              p2=(.020-spacing, .012-spacing, length/2), direction='+', modes=(1, 2),
              anchors=ANCHORS, tracking='auto', verification='fast'))
    scene.add(gprMax.EigenmodeFieldOutput(filename='frequency_crossing', ports=(1,)))
    return scene


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spacing-mm', type=float, default=.25)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bank = args.output/'frequency_crossing.modes.h5'
    if bank.exists():
        parser.error('Choose a fresh output directory to preserve previous results.')
    gprMax.run(scenes=[build_scene(args.spacing_mm*1e-3)],
               outputfile=args.output/'frequency_crossing', geometry_only=True,
               cpu_precision='double', hide_progress_bars=True, log_level=30)
    plot_case(bank, geometry=(16, 8))
