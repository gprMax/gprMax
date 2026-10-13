"""A repeated, fully hybrid Maxwell eigenspace for axis-orientation validation.

Four PEC-separated triangular chambers carry rotated dielectric cores. This
controlled geometry makes exact equivalent copies without approximating a
split eigenpair. It supplements the ordinary single-core guide tests.
"""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from testing.eigenmode_support.dielectric.dielectric_mode_separation import analyse, rotate_bank
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
from copy import deepcopy
from scipy.optimize import linear_sum_assignment


def solve(f,cells=64):
    width=.04;d=width/cells
    edge=(np.arange(cells+1)-cells/2)*d;mid=(np.arange(cells)+.5-cells/2)*d
    def epsilon(x,y):
        x=x[:,None];y=y[None,:]
        core=((abs(abs(x)-.01)<.004)&(abs(y)<.003))|((abs(abs(y)-.01)<.004)&(abs(x)<.003))
        return np.where(core,4.,1.)
    i,j=np.meshgrid(np.arange(cells),np.arange(cells),indexing='ij')
    metal=(i==j)|(i+j==cells-1)
    pu=np.zeros((cells,cells+1),bool);pv=np.zeros((cells+1,cells),bool);pw=np.zeros((cells+1,cells+1),bool)
    pu[:,:-1]|=metal;pu[:,1:]|=metal;pv[:-1,:]|=metal;pv[1:,:]|=metal
    for a,b in ((0,0),(1,0),(0,1),(1,1)):pw[a:a+cells,b:b+cells]|=metal
    pu[:,[0,-1]]=True;pv[[0,-1],:]=True;pw[:,[0,-1]]=True;pw[[0,-1],:]=True
    s=FDFD_2D_mode_solver(f,d,d,23,epsilon(mid,edge),epsilon(edge,mid),epsilon(edge,edge),
        np.ones(pv.shape),np.ones(pu.shape),np.ones((cells,cells)),pec_u_mask=pu,pec_v_mask=pv,pec_w_mask=pw)
    s.retain_tracking_operator=True;s.calculate_diagnostics=True;s.solve();return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
        cells=64,window_mm=40,core_epsilon=4,cladding_epsilon=1,
        description=__doc__,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),root/'gprMax/eigenmode_symmetry.py',root/'gprMax/eigenmode_tracking.py')})
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    try:
        with _configured_solver_constants():
            bank=[]
            for f in np.linspace(18,22,9):
                print('SOLVE',f,flush=True);bank.append(solve(f*1e9))
            altered=deepcopy(bank)
            r,frames=analyse(bank,out,'hybrid_four_chamber')
            applications=sum(a['applied'] for g in r['orientation'] for row in g['anchors'].values() for a in row['axis_sensitive'])
            assert applications>0
            assert r['maximum_field_residual']<1e-7 and r['minimum_power_eigenvalue']>0
            assert r['minimum_channel_overlap']>.95
            assert r['maximum_parity_error']<1e-8
            rotate_bank(altered);_,other=analyse(altered,out,'hybrid_four_chamber_rotated',save_fields=False)
            overlap=1.
            for a,b in zip(frames,other):
                scores=abs(a.conj().T@b)**2/(np.sum(abs(a)**2,axis=0)[:,None]*np.sum(abs(b)**2,axis=0))
                rows,cols=linear_sum_assignment(-scores);overlap=min(overlap,float(scores[rows,cols].min()))
            assert overlap>1-1e-8
            manifest.update(status='passed',axis_applications=applications,minimum_gauge_overlap=overlap,
                maximum_field_residual=r['maximum_field_residual'],minimum_power_eigenvalue=r['minimum_power_eigenvalue'],
                minimum_channel_overlap=r['minimum_channel_overlap'],groups=r['groups'])
    except Exception as error:
        manifest.update(status='failed',error=str(error));raise
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
