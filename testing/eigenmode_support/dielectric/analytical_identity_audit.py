"""Compare automatic FDFD channels to exact full-vector step-index fields."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment

from testing.eigenmode_support.dielectric.analytical_step_index import modes, frame
from testing.eigenmode_support.identification.combined_identity_audit import solve, analyse
from testing.eigenmode_support.identification.tracking_classification_edges import serial
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


def comparison(references, numerical, report, cells):
    columns = []; names = []; roots = []; partners = []
    for mode in references:
        for partner in range(2 if mode['m'] else 1):
            columns.append(frame(mode, cells, partner=partner))
            names.append(mode['name']); roots.append(mode); partners.append(partner)
    analytical = np.column_stack(columns)
    analytical /= np.linalg.norm(analytical, axis=0)
    numerical = numerical/np.linalg.norm(numerical, axis=0)
    score = abs(numerical.conj().T @ analytical)**2
    rows, cols = linear_sum_assignment(-score)
    records = []
    for j, i in zip(rows, cols):
        n = report['mode_details'][-1][j]['neff'].real
        top = np.argsort(score[j])[-4:][::-1]
        records.append(dict(channel=int(j+1), analytical=names[i], partner=partners[i],
            squared_field_overlap=float(score[j, i]), neff_fdfd=float(n), neff_analytical=roots[i]['neff'],
            relative_neff_error=float(abs(n-roots[i]['neff'])/roots[i]['neff']),
            alternatives=[dict(name=names[t], partner=partners[t], overlap=float(score[j, t])) for t in top]))
    combined = []
    for families in (('EH1,1', 'HE3,1'), ('EH2,1', 'HE4,1'), ('EH3,1', 'HE5,1')):
        ni = [j for j, i in zip(rows, cols) if names[i] in families]
        ai = [i for i, name in enumerate(names) if name in families]
        if len(ni) == len(ai) and ni:
            qa = np.linalg.qr(analytical[:, ai])[0]; qn = np.linalg.qr(numerical[:, ni])[0]
            combined.append(dict(families=families, channels=[int(j+1) for j in ni],
                minimum_principal_overlap=float(np.linalg.svd(qa.conj().T @ qn, compute_uv=False).min()**2)))
    return dict(channels=records, combined_spans=combined,
        minimum_field_overlap=min(r['squared_field_overlap'] for r in records),
        maximum_relative_neff_error=max(r['relative_neff_error'] for r in records))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), Path(__file__).with_name('analytical_step_index.py'),
        root/'testing/eigenmode_support/identification/combined_identity_audit.py',
        root/'gprMax/eigenmode_symmetry.py', root/'gprMax/eigenmode_tracking.py',
        root/'gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py']
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        radius_m=.01, core_epsilon=4., cladding_epsilon=1., numerical_window_m=.04,
        analytical_cladding='infinite', cases=[],
        source='https://vtechworks.lib.vt.edu/bitstream/handle/10919/30522/jb-chap5.pdf equations 5.1--5.6',
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    def save():
        (output/'manifest.json').write_text(json.dumps(serial(manifest), indent=2)+'\n')
    save()
    with _configured_solver_constants():
        for frequency, public in ((20e9, 20), (40e9, 40)):
            references = modes(frequency)
            (output/f'reference_{frequency/1e9:g}GHz.json').write_text(json.dumps(serial(references), indent=2)+'\n')
            for cells in (40, 64, 96):
                name = f'circular_dielectric_{frequency/1e9:g}GHz_{cells}'
                entry = dict(name=name, status='running'); manifest['cases'].append(entry); save()
                print('START', name, flush=True)
                try:
                    bank = [solve('circular_dielectric', f, cells, public+12) for f in (frequency-.01e9, frequency)]
                    report, vectors = analyse(bank, 'circular_dielectric', public)
                    result = comparison(references, vectors[-1], report, cells)
                    result['pipeline'] = report
                    (output/f'{name}.json').write_text(json.dumps(serial(result), indent=2)+'\n')
                    entry.update(status='complete', modes=len(report['modes']),
                        minimum_field_overlap=result['minimum_field_overlap'],
                        maximum_relative_neff_error=result['maximum_relative_neff_error'],
                        max_field_residual=report['max_field_residual'],
                        fallback_groups=[r['modes'] for r in report['transport']],
                        selected=[r for r in result['channels'] if r['analytical'] in ('EH1,1', 'HE3,1')],
                        combined_spans=result['combined_spans'])
                    print('COMPLETE', name, 'min overlap', result['minimum_field_overlap'],
                          'max neff error', result['maximum_relative_neff_error'], flush=True)
                except Exception as error:
                    entry.update(status='failed', error=str(error), traceback=traceback.format_exc())
                    print('FAILED', name, traceback.format_exc(), flush=True)
                save()
    manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
