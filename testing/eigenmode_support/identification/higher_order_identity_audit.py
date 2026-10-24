"""Higher-order and repeated-irrep crossing audit; production policy unchanged."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros

from testing.eigenmode_support.identification import combined_identity_audit as audit
from testing.eigenmode_support.identification import tracking_classification_edges as edges
from testing.eigenmode_support.subspaces.general_subspace_tracking import loaded_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


@lru_cache(maxsize=2)
def extended_patterns(nu, nv, du, dv):
    banks = {}
    for family, shape in [('TE', (nu, nv)), ('TM', (nu+1, nv+1))]:
        offset = .5 if family == 'TE' else 0.
        x, y = np.meshgrid((np.arange(shape[0])+offset-nu/2)*du,
                           (np.arange(shape[1])+offset-nv/2)*dv, indexing='ij')
        r = np.hypot(x, y); theta = np.arctan2(y, x)
        columns = []; metadata = []; start = 0
        for m in range(17):
            for n, zero in enumerate((jnp_zeros if family == 'TE' else jn_zeros)(m, 8), 1):
                base = jv(m, zero*r/.01)*(r < .01)
                patterns = [base*np.cos(m*theta)] + ([base*np.sin(m*theta)] if m else [])
                q = np.linalg.qr(np.stack([a.ravel() for a in patterns], axis=1))[0]
                columns.append(q); metadata.append((m, n, float(zero), slice(start, start+q.shape[1])))
                start += q.shape[1]
        banks[family] = (np.column_stack(columns), metadata)
    return banks


def parameter_analysis(bank, case, public, parameter):
    """Supply a material-sweep axis to the tracker, preserving physical frequency."""
    track, align = audit.track_solver_bank, audit.align_groups
    def tracked(owner, axis, solvers, count):
        owner.fallback_frequency = float(parameter[0])
        return track(owner, parameter, solvers, count)
    def aligned(owner, grid, axis, *args):
        return align(owner, grid, parameter, *args)
    audit.track_solver_bank, audit.align_groups = tracked, aligned
    try:
        return audit.analyse(bank, case, public)
    finally:
        audit.track_solver_bank, audit.align_groups = track, align


def run(output, mode):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), Path(audit.__file__), Path(edges.__file__),
               root/'gprMax/eigenmode_symmetry.py', root/'gprMax/eigenmode_tracking.py',
               root/'testing/eigenmode_support/subspaces/general_subspace_tracking.py']
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), mode=mode,
        cases=[], source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    def save():
        (output/'manifest.json').write_text(json.dumps(edges.serial(manifest), indent=2)+'\n')
    save()
    edges.circular_pattern_bank = extended_patterns
    if mode == 'higher':
        jobs = [(case, cells) for case in ('square_PEC', 'circular_PEC', 'square_copper',
                 'circular_copper', 'square_dielectric', 'circular_dielectric') for cells in (40, 64)]
        manifest.update(frequencies_GHz=[79, 80, 81], requested_modes=60, candidates=80,
                        seeds=[131, 982], circular_template_max_m=16, circular_template_max_n=8)
    else:
        jobs = [(name, 24) for name in ('hybrid_E_crossing', 'higher_TE_TM_E_crossing')]
    with _configured_solver_constants():
        for case, cells in jobs:
            name = f'{case}_{cells}'; entry = dict(name=name, status='running'); manifest['cases'].append(entry); save()
            print('START', name, flush=True)
            try:
                if mode == 'higher':
                    bank = [(edges.copper_solver(f*1e9, cells, 80, fit_max_frequency=120e9)
                             if case == 'square_copper' else audit.solve(case, f*1e9, cells, 80))
                            for f in (79, 80, 81)]
                    public = 60; parameter = None
                    analyse = lambda b: audit.analyse(b, case, public)
                elif case == 'hybrid_E_crossing':
                    parameter = np.sort(np.r_[np.linspace(-.12, .12, 49), -.001, .001])
                    bank = [loaded_solver(t, cells=cells) for t in parameter]
                    public = 10
                    analyse = lambda b: parameter_analysis(b, 'square_dielectric', public, parameter)
                else:
                    # Coincidence of TE14/41 and TM23/32, both C4v E pairs.
                    # The exact discrete anisotropic dispersion sets the crossing.
                    d = .02/cells
                    k = lambda n: 2*np.sin(n*np.pi/(2*cells))/d
                    center = 2*(k(2)**2+k(3)**2)/(k(1)**2+k(4)**2)
                    parameter = center + np.sort(np.r_[np.linspace(-.03, .03, 25), -.0001, .0001])
                    bank = [audit.fd_solver(40e9, (cells, cells), d, (2, 2, t), count=48) for t in parameter]
                    public = 40
                    analyse = lambda b: parameter_analysis(b, 'square_PEC', public, parameter)
                result, frames = analyse(deepcopy(bank))
                out = dict(name=name, parameter=parameter, baseline=result, rotations=[])
                # Preserve baseline evidence even if a randomized run fails.
                (output/f'{name}.json').write_text(json.dumps(edges.serial(out), indent=2)+'\n')
                for seed in (131, 982):
                    other = deepcopy(bank); spans = audit.rotate(other, seed)
                    changed, vectors = analyse(other)
                    out['rotations'].append(dict(seed=seed, rotated_spans=spans,
                        comparison=audit.compare(frames, vectors), result=changed))
                    (output/f'{name}.json').write_text(json.dumps(edges.serial(out), indent=2)+'\n')
                entry.update(status='complete', modes=len(result['modes']),
                    max_field_residual=max([result['max_field_residual']]+[r['result']['max_field_residual'] for r in out['rotations']]),
                    min_gauge_overlap=min(r['minimum_matched_overlap'] for t in out['rotations'] for r in t['comparison']),
                    min_same_channel_overlap=min(r['minimum_same_channel_overlap'] for t in out['rotations'] for r in t['comparison']),
                    min_continuity=result['minimum_channel_continuity'], max_parity_error=result['max_parity_error'],
                    fallback_groups=[r['modes'] for r in result['transport']],
                    separation_rejections=result['separation_attempts'])
                print('COMPLETE', name, edges.serial(entry), flush=True)
            except Exception as error:
                entry.update(status='failed', error=str(error), traceback=traceback.format_exc())
                print('FAILED', name, traceback.format_exc(), flush=True)
            save()
    manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('higher', 'crossings'), default='higher'); args = p.parse_args()
    run(args.output, args.mode)
