"""Check dielectric identities against a larger PEC truncation window."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from testing.eigenmode_support.identification.combined_identity_audit import analyse, compare
from testing.eigenmode_support.identification.tracking_classification_edges import serial
from testing.eigenmode_support.dielectric.dielectric_mode_separation import solve
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


def crop(frame, cells, target):
    offset = (cells-target)//2
    pieces = []; start = 0
    for a, b in ((0, 1), (1, 0), (1, 1), (1, 0), (0, 1), (0, 0)):
        size = (cells+a)*(cells+b)
        array = frame[start:start+size].reshape(cells+a, cells+b, -1)
        pieces.append(array[offset:offset+target+a, offset:offset+target+b].reshape(-1, frame.shape[1]))
        start += size
    assert start == frame.shape[0]
    return np.vstack(pieces)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), root/'testing/eigenmode_support/identification/combined_identity_audit.py',
        root/'testing/eigenmode_support/dielectric/dielectric_mode_separation.py',
        root/'gprMax/eigenmode_symmetry.py', root/'gprMax/eigenmode_tracking.py']
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        spacing_m=.001, windows_m=[.04, .06], cells=[40, 60], frequencies_GHz=[39, 40, 41],
        public_modes=20, candidates=24, cases=[],
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    def save():
        (output/'manifest.json').write_text(json.dumps(serial(manifest), indent=2)+'\n')
    save()
    with _configured_solver_constants():
        for shape in ('square', 'circular'):
            entry = dict(name=shape, status='running'); manifest['cases'].append(entry); save()
            print('START', shape, flush=True)
            try:
                records = []; frames = []
                for cells, width in ((40, .04), (60, .06)):
                    bank = [solve(shape, f*1e9, cells, width) for f in (39, 40, 41)]
                    report, field = analyse(bank, shape+'_dielectric', 20)
                    records.append(report); frames.append([crop(a, cells, 40) for a in field])
                comparison = compare(*frames)
                differences = []
                for k, row in enumerate(comparison):
                    old = np.array([v['neff'] for v in records[0]['mode_details'][k]])
                    new = np.array([v['neff'] for v in records[1]['mode_details'][k]])[row['permutation']]
                    differences.append(abs(old-new)/abs(old))
                result = dict(shape=shape, windows=records, comparison=comparison,
                              relative_neff_differences=differences)
                (output/f'{shape}.json').write_text(json.dumps(serial(result), indent=2)+'\n')
                entry.update(status='complete', minimum_cropped_overlap=min(r['minimum_matched_overlap'] for r in comparison),
                             maximum_relative_neff_difference=float(np.max(differences)))
                print('COMPLETE', serial(entry), flush=True)
            except Exception as error:
                entry.update(status='failed', error=str(error), traceback=traceback.format_exc())
                print('FAILED', traceback.format_exc(), flush=True)
            save()
    manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
