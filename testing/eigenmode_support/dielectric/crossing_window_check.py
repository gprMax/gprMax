"""Separate dielectric-interface discretization from outer-window effects."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from testing.eigenmode_support.identification.combined_identity_audit import analyse, compare
from testing.eigenmode_support.identification.combined_identity_window_check import crop
from testing.eigenmode_support.identification.tracking_classification_edges import serial
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


def run(output, crossing_manifest):
    output.mkdir(parents=True, exist_ok=False)
    fc = json.loads(crossing_manifest.read_text())['analytical_crossing_Hz']
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        frequency_center_Hz=fc, spacing_m=.04/64, windows_m=[.04,.06], cells=[64,96],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save(): (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save(); reports=[]; frames=[]
    with _configured_solver_constants():
        for cells,width in ((64,.04),(96,.06)):
            bank=[];d=width/cells;edge=(np.arange(cells+1)-cells/2)*d;mid=(np.arange(cells)+.5-cells/2)*d
            def material(x,y):return np.where(x[:,None]**2+y[None,:]**2<.01**2,4.,1.).astype(complex)
            for f in (fc-.001e9,fc):
                eu,ev,ew=material(mid,edge),material(edge,mid),material(edge,edge)
                eu[:,[0,-1]]=np.inf;ev[[0,-1],:]=np.inf;ew[:,[0,-1]]=np.inf;ew[[0,-1],:]=np.inf
                s=FDFD_2D_mode_solver(f,d,d,55,eu,ev,ew,np.ones_like(ev),np.ones_like(eu),np.ones((cells,cells)))
                s.retain_tracking_operator=True;s.calculate_diagnostics=True;s.solve();bank.append(s)
            r,v=analyse(bank,'circular_dielectric',44);reports.append(r)
            frames.append([crop(a,cells,64) for a in v]); print('COMPLETE WINDOW',width,flush=True)
    match=compare(*frames);changes=[]
    for k,row in enumerate(match):
        a=np.array([r['neff'] for r in reports[0]['mode_details'][k]])
        b=np.array([r['neff'] for r in reports[1]['mode_details'][k]])[row['permutation']]
        changes.append(abs(a-b)/abs(a))
    (output/'results.json').write_text(json.dumps(serial(dict(windows=reports,comparison=match,relative_neff_change=changes)),indent=2)+'\n')
    manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
        minimum_field_overlap=min(r['minimum_matched_overlap'] for r in match),maximum_relative_neff_change=float(np.max(changes)))
    save();print(manifest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--crossing-manifest',type=Path,required=True)
    a=p.parse_args();run(a.output,a.crossing_manifest)
