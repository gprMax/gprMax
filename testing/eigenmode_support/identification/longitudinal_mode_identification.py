"""Independent modal-reference checks; no templates enter runtime tracking."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from testing.eigenmode_support.subspaces import general_subspace_tracking as crossing
from testing.eigenmode_support.identification.tracking_classification_edges import analyse, voxel_circular_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants

ROOT = Path(__file__).resolve().parents[3]


def verify(report):
    assert report['max_field_residual'] < 1e-7
    assert report['max_mixed_residual'] < 1e-7
    assert report['minimum_power_eigenvalue'] > .9
    assert report['min_channel_overlap'] > .99
    assert all(g['applied'] in ('c4v','c2v') for g in report['symmetry_detection']['groups'])
    modes = report['anchors'][len(report['anchors'])//2]
    best = [row['templates'][0] for row in modes]
    # Circular templates include the sine/cosine pair of the same angular
    # and radial indices; rectangular templates are individual m,n modes.
    identified = [dict(branch=row['mode'],reference=top['name'],score=top['score'],
                       parity=row['parity']) for row,top in zip(modes,best)]
    parity_error = max(row['parity_error'] for anchor in report['anchors'] for row in anchor)
    assert parity_error < 1e-8
    if report['geometry'] == 'circular':
        assert min(b['score'] for b in best) > .98
    else:
        # Axis orientation must also recover the formerly mixed mn/nm partners.
        for row in modes:
            assert row['templates'][0]['score'] > 1-1e-9
            assert sum(t['score'] for t in row['templates'][:2]) > 1-1e-9
            assert len({t['name'].split('_')[0] for t in row['templates'] if t['score']>1e-8}) == 1
    return dict(mode_count=len(modes),identified=identified,
                minimum_reference_score=min(b['score'] for b in best),
                individually_resolved=sum(b['score']>.98 for b in best),
                maximum_parity_error=parity_error,
                maximum_field_residual=report['max_field_residual'],
                minimum_channel_overlap=report['min_channel_overlap'])


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        description='Native FDFD banks, automatic symmetry and longitudinal separation. Analytical mode names are validation only.',
        cases=[],source_sha256={})
    for path in (Path(__file__),ROOT/'gprMax/eigenmode_symmetry.py',ROOT/'gprMax/eigenmode_tracking.py',
                 ROOT/'gprMax/eigenmode_config.py',ROOT/'testing/eigenmode_support/identification/tracking_classification_edges.py',
                 ROOT/'testing/eigenmode_support/subspaces/general_subspace_tracking.py',ROOT/'testing/eigenmode_support/identification/modal_field_templates.py'):
        manifest['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    cases=[('rectangular_48',48,'rectangle'),('rectangular_64',64,'rectangle'),
           ('square_48',48,'square'),('square_64',64,'square'),
           ('circular_64',64,'circular'),('circular_96',96,'circular')]
    with _configured_solver_constants():
        for name,cells,geometry in cases:
            entry=dict(name=name,cells=[cells,int(cells*.75) if geometry=='rectangle' else cells],
                       geometry=geometry,frequency_GHz=[40,42,44],public_modes=32,candidates=40,
                       epsilon=2,dimensions_mm=[20,15] if geometry=='rectangle' else [20,20] if geometry=='square' else [24,24],
                       status='running')
            if geometry=='circular':entry['radius_mm']=10
            manifest['cases'].append(entry);save()
            try:
                bank=[]
                for f in (40,42,44):
                    print('SOLVE',name,f,'GHz',flush=True)
                    bank.append(voxel_circular_solver(f*1e9,cells,count=40) if geometry=='circular' else
                        crossing.fd_solver(f*1e9,tuple(entry['cells']),.02/cells,(2,2,2),count=40))
                report=analyse(name,bank,32,output,geometry)
                entry['verification']=verify(report)
                entry['status']='passed'
            except Exception as error:
                entry.update(status='failed',error=str(error));traceback.print_exc()
            save()
        entry=dict(name='hybrid_crossing_64',cells=[64,64],status='running')
        manifest['cases'].append(entry);save()
        try:
            axis=np.sort(np.r_[np.linspace(-.12,.12,25),-.001,.001])
            bank=[]
            for t in axis:
                print('SOLVE hybrid_crossing_64',t,flush=True)
                bank.append(crossing.loaded_solver(t,cells=64))
            crossing.stress_rotation(bank[len(bank)//2],[6,7,8,9])
            report=crossing.run_case(entry['name'],axis,bank,10,[7,8,9,10],output,
                'Dielectric perturbation t',expected_sizes=(2,2),symmetry='auto')
            assert all(g['applied']=='c4v' for g in report['symmetry_detection']['groups'])
            assert min(np.asarray(g['longitudinal_norm_ratios'])[[0,-1]].min() for g in report['groups'])>.01
            entry.update(status='passed',minimum_span_overlap=min(min(g['overlap']) for g in report['groups']),
                         maximum_field_residual=report['maximum_reconstructed_field_residual'],
                         maximum_mixed_residual=report['maximum_mixed_residual'])
        except Exception as error:
            entry.update(status='failed',error=str(error));traceback.print_exc()
        save()
    manifest.update(status='passed' if all(c['status']=='passed' for c in manifest['cases']) else 'failed',
                    finished_utc=datetime.now(timezone.utc).isoformat())
    save()
    print('FINISHED',manifest['status'],[(c['name'],c['status']) for c in manifest['cases']],flush=True)
    return 0 if manifest['status']=='passed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().output))
