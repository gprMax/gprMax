"""A circular-continuum crossing whose branches can couple on a C4v grid."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.optimize import brentq

from testing.eigenmode_support.dielectric.analytical_step_index import modes, frame
from testing.eigenmode_support.identification.combined_identity_audit import solve, analyse
from testing.eigenmode_support.identification.tracking_classification_edges import serial
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


def references(f):
    return {m['name']: m for m in modes(f, max_m=4) if m['name'] in ('TE0,3', 'HE4,2')}


def difference(f):
    r = references(f)
    return r['TE0,3']['neff']-r['HE4,2']['neff']


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), Path(__file__).with_name('analytical_step_index.py'),
        root/'gprMax/eigenmode_tracking.py', root/'gprMax/eigenmode_symmetry.py',
        root/'testing/eigenmode_support/identification/combined_identity_audit.py']
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        cells=64, width_m=.04, radius_m=.01, epsilon_core=4., epsilon_cladding=1.,
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    def save():
        (output/'manifest.json').write_text(json.dumps(serial(manifest), indent=2)+'\n')
    save()
    try:
        fc = brentq(difference, 35e9, 40e9, xtol=.01)
        frequencies = fc+np.linspace(-1.5e9, 1.5e9, 13)
        manifest.update(analytical_crossing_Hz=fc, frequencies_Hz=frequencies); save()
        with _configured_solver_constants():
            bank = []
            for f in frequencies:
                print('SOLVE', f, flush=True)
                bank.append(solve('circular_dielectric', f, 64, 56))
            report, fields = analyse(bank, 'circular_dielectric', 44)
            rows = []
            for k, f in enumerate(frequencies):
                refs = references(f)
                a = np.column_stack([frame(refs['TE0,3'], 64),
                    frame(refs['HE4,2'], 64, partner=0), frame(refs['HE4,2'], 64, partner=1)])
                a /= np.linalg.norm(a, axis=0)
                b = fields[k]/np.linalg.norm(fields[k], axis=0)
                overlaps = abs(b.conj().T @ a)**2
                selected = np.argsort(overlaps.sum(axis=1))[-3:]
                qa = np.linalg.qr(a)[0]; qb = np.linalg.qr(b[:, selected])[0]
                rows.append(dict(frequency=f, analytical_neff={n:r['neff'] for n,r in refs.items()},
                    channels=[dict(channel=int(j+1), neff=report['mode_details'][k][j]['neff'],
                        parity=report['mode_details'][k][j]['parity'],
                        overlaps=dict(TE03=float(overlaps[j,0]), HE42_cos=float(overlaps[j,1]),
                                      HE42_sin=float(overlaps[j,2]))) for j in sorted(selected)],
                    minimum_three_channel_span_overlap=float(np.linalg.svd(qa.conj().T @ qb, compute_uv=False).min()**2)))
            result = dict(pipeline=report, crossing=rows)
            (output/'results.json').write_text(json.dumps(serial(result), indent=2)+'\n')
            manifest.update(status='complete', minimum_continuity=report['minimum_channel_continuity'],
                maximum_field_residual=report['max_field_residual'],
                fallback_groups=[r['modes'] for r in report['transport']],
                minimum_three_channel_span_overlap=min(r['minimum_three_channel_span_overlap'] for r in rows))
            print('COMPLETE', {k:v for k,v in manifest.items() if k not in ('source_sha256', 'frequencies_Hz')}, flush=True)
    except Exception as error:
        manifest.update(status='failed', error=str(error), traceback=traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
