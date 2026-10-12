"""Geometry-only tracking galleries for PEC, copper and open dielectric guides.

Run from the repository root. Each case writes a native Yee-field HDF5 bank,
dispersion curves, and electric/magnetic galleries at the middle anchor.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import gprMax

EXAMPLE_DIR = Path(__file__).resolve().parent
CASES = tuple(f'{material}_{shape}' for material in ('pec', 'copper', 'dielectric')
              for shape in ('rectangular', 'square', 'cylindrical'))


def anchors(case):
    centre = 40e9
    return tuple(np.linspace(centre-1e9, centre+1e9, 5))


def build_scene(case, modes=16, spacing=0.00025, verification='fast', profiles=False):
    """Build one undriven inspection port in a z-invariant guide."""
    if case not in CASES:
        raise ValueError(f'Choose one of {CASES}.')
    if modes < 1:
        raise ValueError('modes must be positive')
    material, shape = case.split('_')
    dielectric = material == 'dielectric'
    window = .06 if dielectric else .024
    width, height = .02, (.01 if shape == 'rectangular' else .02)
    length = 4*spacing
    frequencies = anchors(case)
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f'Tracked modes: {case.replace("_", " ")}'))
    scene.add(gprMax.Domain(p1=(window, window, length)))
    scene.add(gprMax.Discretisation(p1=(spacing, spacing, spacing)))
    scene.add(gprMax.TimeWindow(iterations=1))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.PMLThickness(thickness=0))
    if dielectric:
        scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id='core'))
        fill = 'core'
    else:
        wall = 'pec'
        if material == 'copper':
            wall = 'copper_wall'
            scene.add(gprMax.SurfaceImpedance(id=wall, preset='copper',
                fit_frequency_range=(1e9,120e9), fit_order=16, plot_fit=False))
        lower, upper = ((spacing,)*3, (window-spacing, window-spacing, length-spacing)) if material == 'copper' else ((0,0,0), (window,window,length))
        scene.add(gprMax.Box(p1=lower, p2=upper, material_id=wall))
        fill = 'free_space'
    centre = window / 2
    if shape == 'cylindrical':
        scene.add(gprMax.Cylinder(p1=(centre,centre,0), p2=(centre,centre,length),
                                 r=width/2, material_id=fill, averaging='n' if material == 'copper' else 'y'))
    else:
        scene.add(gprMax.Box(p1=(centre-width/2,centre-height/2,0),
                            p2=(centre+width/2,centre+height/2,length),
                            material_id=fill, averaging='n' if material == 'copper' else 'y'))
    scene.add(gprMax.EigenmodeBand(id='inspection', fmin=frequencies[0], fmax=frequencies[-1], points=5))
    scene.add(gprMax.EigenmodePort(
        port=1, p1=(spacing,spacing,length/2), p2=(window-spacing,window-spacing,length/2),
        direction='+', modes=modes, anchors=frequencies, tracking='auto',
        verification=verification, tracking_config=gprMax.EigenmodeTrackingConfig(extra_candidates=8),
        plot_fields=profiles))
    scene.add(gprMax.EigenmodeFieldOutput(filename=case, ports=(1,)))
    return scene


def main(default_case='all', default_output=None):
    from examples.features.eigenmode_tracking.plot_results import plot_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', *CASES), default=default_case)
    parser.add_argument('--modes', type=int, default=16)
    parser.add_argument('--spacing-mm', type=float, default=.25)
    parser.add_argument('--verification', choices=('fast', 'full'), default='fast')
    parser.add_argument('--profiles', action='store_true', help='also plot each mode at every anchor')
    parser.add_argument('--output', type=Path, default=default_output or EXAMPLE_DIR/'results')
    args = parser.parse_args()
    cases = CASES if args.case == 'all' else (args.case,)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists() or any((args.output/f'{case}.modes.h5').exists() for case in cases):
        parser.error('Choose a fresh output directory to preserve earlier results.')
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                    modes=args.modes, spacing_mm=args.spacing_mm, verification=args.verification,
                    anchors_hz={case:anchors(case) for case in cases}, excitation=False,
                    geometry_only=True, time_stepping=False, cases=[],
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), EXAMPLE_DIR/'plot_results.py')})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        for case in cases:
            print(f'INSPECT {case}', flush=True)
            scene = build_scene(case, args.modes, args.spacing_mm*1e-3, args.verification, args.profiles)
            gprMax.run(scenes=[scene], outputfile=args.output/case, geometry_only=True,
                       cpu_precision='double', hide_progress_bars=True, log_level=30)
            record = plot_case(args.output/f'{case}.modes.h5')
            manifest['cases'].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
            print(f'COMPLETE {case}: {record["mode_count"]} modes', flush=True)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
