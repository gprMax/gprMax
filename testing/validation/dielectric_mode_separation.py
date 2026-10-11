"""Automatic hybrid-mode separation in dielectric cores with air cladding."""
from copy import copy, deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import traceback
import numpy as np
from scipy.optimize import linear_sum_assignment
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.eigenmode_config import EigenmodeTrackingConfig
from gprMax.eigenmode_tracking import track_solver_bank, align_groups, _frame, _power
from gprMax.eigenmode_symmetry import mirror_u, mirror_v
from gprMax.sources import EigenmodeSource
import gprMax.config as config
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
from testing.validation.tracking_classification_edges import serial


def solve(shape,frequency,cells,width):
    d=width/cells
    edge=(np.arange(cells+1)-cells/2)*d
    mid=(np.arange(cells)+.5-cells/2)*d
    def epsilon(x,y):
        x=x[:,None];y=y[None,:]
        inside=(x*x+y*y<.01**2) if shape=='circular' else ((abs(x)<.01)&(abs(y)<(.007 if shape=='rectangular' else .01)))
        return np.where(inside,4.,1.).astype(complex)
    eu,ev,ew=epsilon(mid,edge),epsilon(edge,mid),epsilon(edge,edge)
    eu[:,[0,-1]]=np.inf;ev[[0,-1],:]=np.inf
    ew[:,[0,-1]]=np.inf;ew[[0,-1],:]=np.inf
    s=FDFD_2D_mode_solver(frequency,d,d,23,eu,ev,ew,np.ones_like(ev),np.ones_like(eu),np.ones((cells,cells)))
    s.retain_tracking_operator=True;s.calculate_diagnostics=True;s.solve()
    return s


def rotate_bank(bank):
    rng=np.random.default_rng(98314)
    for s in bank:
        used=set()
        for i,value in enumerate(s.eigenvalues):
            if i in used:continue
            indices=np.flatnonzero(abs(s.eigenvalues-value)<1e-11*max(1,abs(value)))
            used.update(indices.tolist());n=len(indices)
            u=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
            for name in ('Eu','Ev','Ew','Hu','Hv','Hw'):
                a=getattr(s,name);a[...,indices]=a[...,indices]@u
        s._normalize_modes_to_power()
        s.eigenvectors=np.column_stack([np.r_[s.Eu[...,i].ravel(order='F'),s.Ev[...,i].ravel(order='F')] for i in range(s.num_modes)])


def analyse(bank,output,name,save_fields=True):
    owner=EigenmodeSource(None)
    owner.normal_axis=2;owner.normal='z';owner.direction='+';owner.transverse_axes=(0,1)
    owner.tracking_config=EigenmodeTrackingConfig(extra_candidates=0)
    owner.port_index=1;owner.fallback_frequency=20e9;owner.degenerate=();owner.mode_polarizations={}
    owner.tracking='auto';owner.verification='none';owner.mode_symmetry='auto'
    owner._tracking_impedance=config.sim_config.em_consts['z0']
    axis=np.array([s.frequency for s in bank])
    mapping=track_solver_bank(owner,axis,bank,12)
    modes=tuple(range(1,owner._auto_tracking_mode_count+1))
    owner.mode_indices=modes;owner.mode_index=1;owner.mode_count=len(modes)
    grid=SimpleNamespace(dl=np.full(3,bank[0].du));owner._tracking_grid=grid
    fields=[[owner._fields_from_solver_mode(s,m) for m in modes] for s in bank]
    e=[[v[0] for v in row] for row in fields];h=[[v[1] for v in row] for row in fields]
    propagating=np.array([[s.complex_neff[j].real>0 for j in range(len(modes))] for s in bank])
    align_groups(owner,grid,axis,bank,modes,e,h,propagating)
    frames=[_frame(ee,hh,owner._tracking_impedance) for ee,hh in zip(e,h)]
    records=[];powers=[];residuals=[]
    for k,s in enumerate(bank):
        final=copy(s)
        for rows,names in ((e,('Eu','Ev','Ew')),(h,('Hu','Hv','Hw'))):
            for c,label in enumerate(names):
                a=getattr(final,label).copy()
                for j in range(len(modes)):a[...,j]=rows[k][j][c]
                setattr(final,label,a)
        residuals.extend(final.tracking_field_residuals(np.arange(len(modes))).tolist())
        powers.append(float(np.linalg.eigvalsh(_power(owner,grid,e[k],h[k])).min()))
        row=[]
        for j,m in enumerate(modes):
            signs=[];errors=[]
            for action in (mirror_u,mirror_v):
                te,th=action(s,e[k][j],h[k][j]);f=_frame([te],[th],owner._tracking_impedance)[:,0]
                original=frames[k][:,j];sign=1 if np.vdot(original,f).real>=0 else -1
                signs.append(sign);errors.append(float(np.linalg.norm(f-sign*original)/np.linalg.norm(original)))
            row.append(dict(mode=m,parity=signs,parity_error=max(errors),
                electric_longitudinal_fraction=float(np.linalg.norm(e[k][j][2])**2/sum(np.linalg.norm(a)**2 for a in e[k][j])),
                magnetic_longitudinal_fraction=float(np.linalg.norm(h[k][j][2])**2/sum(np.linalg.norm(a)**2 for a in h[k][j]))))
        records.append(row)
    overlaps=[abs(np.sum(a.conj()*b,axis=0))**2/(np.sum(abs(a)**2,axis=0)*np.sum(abs(b)**2,axis=0)) for a,b in zip(frames,frames[1:])]
    report=serial(dict(name=name,frequency_GHz=axis/1e9,modes=modes,neff=[s.complex_neff[:len(modes)] for s in bank],
        candidate_indices=mapping,groups=owner.degenerate,symmetry=owner.symmetry_detection,
        blocks=[dict(modes=r['modes'],blocks=r['blocks'],irreps=r['irreps']) for r in owner.symmetry_diagnostics],
        anchors=records,channel_overlaps=overlaps,minimum_channel_overlap=float(np.min(overlaps)),
        maximum_field_residual=max(residuals),minimum_power_eigenvalue=min(powers),
        maximum_parity_error=max(r['parity_error'] for row in records for r in row),
        maximum_mixed_residual=max((float(np.max(a['residual'])) for g in owner.degenerate_diagnostics for a in g['anchors'] if a['active']),default=0),
        orientation=[dict(modes=g['modes'],anchors=g['anchors']) for g in owner.symmetry_diagnostics]))
    if save_fields:
        (output/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(output/f'{name}_fields.npz',**{f'{kind}_{k}_{j}_{c}':a for kind,rows in [('E',e),('H',h)] for k,row in enumerate(rows) for j,field in enumerate(row) for c,a in enumerate(field)})
    return report,frames


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    manifest=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),cases=[],
        setup=dict(core_epsilon=4,cladding_epsilon=1,rectangular_core_mm=[20,14],square_core_mm=[20,20],circular_radius_mm=10,
                   boundary='outer PEC truncation; compare 40-mm and 60-mm windows at identical spacing',public_modes=12,candidates=24),source_sha256={})
    for path in (Path(__file__),root/'gprMax/eigenmode_tracking.py',root/'gprMax/eigenmode_symmetry.py',root/'gprMax/eigenmode_config.py'):
        manifest['source_sha256'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    def save(): (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    with _configured_solver_constants():
        for shape in ('rectangular','square','circular'):
            for cells,width in ((64,.04),(96,.04),(96,.06)):
                name=f'{shape}_{cells}_{round(width*1000)}mm'
                entry=dict(name=name,shape=shape,cells=cells,window_mm=width*1000,status='running');manifest['cases'].append(entry);save()
                try:
                    bank=[]
                    for f in np.linspace(18,22,9):
                        print('SOLVE',name,f,flush=True);bank.append(solve(shape,f*1e9,cells,width))
                    changed=deepcopy(bank) if cells==64 else None
                    report,frames=analyse(bank,output,name)
                    entry.update({key:report[key] for key in ('minimum_channel_overlap','maximum_field_residual','minimum_power_eigenvalue','maximum_parity_error','maximum_mixed_residual','groups','blocks')})
                    entry['minimum_neff']=min(z[0] for row in report['neff'] for z in row)
                    entry['detected_symmetry']=report['symmetry']['detected']
                    assert report['maximum_field_residual']<1e-7 and report['maximum_mixed_residual']<1e-7
                    assert report['minimum_power_eigenvalue']>0 and report['minimum_channel_overlap']>.95
                    assert report['maximum_parity_error']<1e-8
                    assert all(g['applied'] in ('c4v','c2v') for g in report['symmetry']['groups'])
                    if changed is not None:
                        rotate_bank(changed)
                        _,other=analyse(changed,output,name+'_rotated',save_fields=False)
                        minimum=1.
                        for a,b in zip(frames,other):
                            scores=abs(a.conj().T@b)**2/(np.sum(abs(a)**2,axis=0)[:,None]*np.sum(abs(b)**2,axis=0))
                            rows,cols=linear_sum_assignment(-scores);minimum=min(minimum,float(scores[rows,cols].min()))
                        entry['minimum_gauge_invariance_overlap']=minimum
                        assert minimum>1-1e-8
                    entry['status']='passed'
                except Exception as error:
                    entry.update(status='failed',error=str(error));traceback.print_exc()
                save();print('CASE',name,entry['status'],flush=True)
    manifest.update(status='passed' if all(c['status']=='passed' for c in manifest['cases']) else 'failed',finished_utc=datetime.now(timezone.utc).isoformat())
    save();print('FINISHED',manifest['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
