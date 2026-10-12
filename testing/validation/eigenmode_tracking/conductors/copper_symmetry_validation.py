"""Reproducible SIBC symmetry and higher-order partner validation."""
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import json, hashlib, traceback
import numpy as np
from scipy.optimize import linear_sum_assignment
from testing.validation.eigenmode_tracking.identification.tracking_classification_edges import copper_solver, analyse
from testing.validation.eigenmode_tracking.dielectric.dielectric_mode_separation import rotate_bank
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants

root=Path(__file__).resolve().parents[4]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),conductivity=5.96e7,width_mm=20,frequencies_GHz=[40,42,44],public_modes=24,candidates=32,cases=[],source_sha256={str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in [Path('gprMax/eigenmode_symmetry.py'),Path('gprMax/eigenmode_tracking.py'),Path('testing/validation/eigenmode_tracking/identification/tracking_classification_edges.py'),Path('testing/validation/eigenmode_tracking/conductors/copper_symmetry_validation.py')]})
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    def frame(data,k,n):
     return np.column_stack([np.concatenate([data[f'{kind}_{k}_{j}_{c}'].ravel()*(376.730313668 if kind=='H' else 1) for kind in ['E','H'] for c in range(3)]) for j in range(n)])
    with _configured_solver_constants():
     for cells in [48,64,(48,36)]:
      entry=dict(cells=cells,status='running');manifest['cases'].append(entry);save()
      try:
       bank=[]
       for f in [40e9,42e9,44e9]:
        print('SOLVE',cells,f,flush=True);bank.append(copper_solver(f,cells,count=32))
       other=deepcopy(bank)
       name='copper_'+(str(cells) if isinstance(cells,int) else '48x36')
       geometry='square' if isinstance(cells,int) else 'rectangle'
       r=analyse(name,bank,24,out,geometry)
       rotate_bank(other);q=analyse(name+'_rotated',other,24,out,geometry)
       a=np.load(out/(name+'_fields.npz'));b=np.load(out/(name+'_rotated_fields.npz'))
       n=len(r['anchors'][0]);same=[];best=[]
       for k in range(3):
        x=frame(a,k,n);y=frame(b,k,n);s=abs(x.conj().T@y)**2/(np.sum(abs(x)**2,axis=0)[:,None]*np.sum(abs(y)**2,axis=0)[None,:]);rr,cc=linear_sum_assignment(-s)
        same.append(float(np.diag(s).min()));best.append(float(s[rr,cc].min()))
       reference=r['anchors'][1]
       entry.update(status='checked',detected=r['detected'],symmetry_detection=r['symmetry_detection'],groups=r['groups'],maximum_field_residual=max(r['max_field_residual'],q['max_field_residual']),maximum_mixed_residual=max(r['max_mixed_residual'],q['max_mixed_residual']),minimum_power_eigenvalue=min(r['minimum_power_eigenvalue'],q['minimum_power_eigenvalue']),minimum_channel_overlap=min(r['min_channel_overlap'],q['min_channel_overlap']),minimum_same_label_gauge_overlap=min(same),minimum_best_assignment_gauge_overlap=min(best),maximum_parity_error=max(row['parity_error'] for anchor in r['anchors'] for row in anchor),reference_modes=reference)
       assert entry['maximum_field_residual']<1e-7 and entry['maximum_mixed_residual']<1e-7 and entry['minimum_power_eigenvalue']>0
       assert r['detected']['detected'] == ('c4v' if isinstance(cells,int) else 'c2v')
       assert all(g['applied'] in ('c4v','c2v') for g in r['symmetry_detection']['groups'])
       assert min(same)>1-1e-8
       if isinstance(cells,int):
        assert min(row['templates'][0]['score'] for anchor in r['anchors'] for row in anchor)>.999
       else:
        # Conductive rectangular walls split the PEC TE/TM degeneracy into
        # hybrid eigenmodes. Verify spatial indices without rotating those
        # distinct eigenvalues into artificially pure TE/TM fields.
        assert not r['groups']
        for anchor in r['anchors']:
         for row in anchor:
          index=row['templates'][0]['name'].split('_')[1]
          assert sum(t['score'] for t in row['templates'] if t['name'].split('_')[1]==index)>.999
       assert all([row['templates'][0]['name'] for row in anchor]==[row['templates'][0]['name'] for row in reference] for anchor in r['anchors'])
       entry['numerical_validity']='passed'
       entry['individual_partner_identification']='passed'
       print('CHECKED',cells,{k:v for k,v in entry.items() if k.startswith(('minimum','maximum'))},flush=True)
      except Exception as exc:
       entry.update(status='failed',error=str(exc));traceback.print_exc()
      save()
    manifest['status']='passed' if all(c.get('individual_partner_identification')=='passed' for c in manifest['cases']) else 'failed';manifest['finished_utc']=datetime.now(timezone.utc).isoformat();save();print('COPPER_FINAL_COMPLETE',flush=True)

    if manifest['status']!='passed': raise SystemExit(1)


if __name__ == '__main__':
    main()
