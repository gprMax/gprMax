"""Check automatic bases against circular/rectangular modal templates and copper walls.

This is a modal-bank validation, not a time-domain scattering simulation.
Template scores are diagnostics; they do not relabel or alter computed fields.
"""
import argparse
from dataclasses import asdict
from copy import copy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros
from gprMax.eigenmode_config import EigenmodeTrackingConfig
from gprMax.eigenmode_tracking import track_solver_bank, align_groups, _frame, _power
from gprMax.eigenmode_symmetry import mirror_u, mirror_v, detect_symmetry
from testing.validation.modal_field_templates import longitudinal_coefficients
from gprMax.sources import EigenmodeSource
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.surface_impedance_presets import fit_conductivity_surface_impedance
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import evaluate_surface_ade
from testing.validation.expanded_mode_tracking import fd_solver
from testing.validation.waveguide_symmetry_validation import circular_solver
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
import gprMax.config as config

ROOT=Path(__file__).resolve().parents[2]

def serial(value):
    if isinstance(value,np.ndarray): return serial(value.tolist())
    if isinstance(value,np.generic): return serial(value.item())
    if isinstance(value,complex): return [value.real,value.imag]
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [serial(v) for v in value]
    return value

def copper_solver(frequency,cells,count=14):
    # Use the independently tested clipped Ampere contour rows on all four walls.
    path=ROOT/'tests/fdfd_eigenmode_solver/test_surface_impedance_fdfd_operator.py'
    spec=importlib.util.spec_from_file_location('impedance_boundary_fixture',path)
    fixture=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    nu, nv = (cells, cells) if np.isscalar(cells) else cells
    d=.02/nu
    dt=.99*d/(config.sim_config.em_consts['c']*np.sqrt(3))
    fit=fit_conductivity_surface_impedance(5.96e7,1e9,50e9,16)
    model=SurfaceImpedanceModel('copper',A=fit.A,B=fit.B,C=fit.C,D=fit.D)
    discrete=model.discretise(dt)
    response=evaluate_surface_ade(frequency_hz=frequency,dt=dt,F=discrete.F,
        G=discrete.G,L=discrete.L,Z0=discrete.Z0)
    boundary=fixture._rectangular_impedance_boundary(nu,nv,d,d,response)
    u=np.ones((nu,nv+1)); v=np.ones((nu+1,nv)); w=np.ones((nu+1,nv+1))
    s=FDFD_2D_mode_solver(frequency,d,d,count-1,u,v,w,np.ones_like(v),np.ones_like(u),
        np.ones((nu,nv)),surface_boundary=boundary)
    s.retain_tracking_operator=True; s.calculate_diagnostics=True; s.solve()
    s.edge_validation_metadata=dict(sigma_S_per_m=5.96e7,ade_dt=dt,
        admittance=response.admittance,fit_max_relative_error=fit.max_relative_error)
    return s

def voxel_circular_solver(frequency,cells,count=20):
    """Conductor-cell staircase with tangential edge constraints on every face.

    This is the PEC cell-to-Yee-mask construction used by the analytical
    circular-waveguide validation, with a 10-mm radius and 24-mm window.
    """
    d=.024/cells; x=(np.arange(cells)+.5-cells/2)*d
    metal=x[:,None]**2+x[None,:]**2>=.01**2
    pu=np.zeros((cells,cells+1),bool);pv=np.zeros((cells+1,cells),bool)
    pw=np.zeros((cells+1,cells+1),bool)
    pu[:,:-1]|=metal;pu[:,1:]|=metal;pv[:-1,:]|=metal;pv[1:,:]|=metal
    for i,j in [(0,0),(1,0),(0,1),(1,1)]:pw[i:i+cells,j:j+cells]|=metal
    s=FDFD_2D_mode_solver(frequency,d,d,count-1,2*np.ones(pu.shape),2*np.ones(pv.shape),
        2*np.ones(pw.shape),np.ones(pv.shape),np.ones(pu.shape),np.ones((cells,cells)),
        pec_u_mask=pu,pec_v_mask=pv,pec_w_mask=pw)
    s.retain_tracking_operator=True;s.calculate_diagnostics=True;s.solve()
    return s

@lru_cache(maxsize=4)
def circular_pattern_bank(nu,nv,du,dv):
    """Cache sampled Bessel patterns; the same geometry serves all anchors."""
    banks={}
    for family,shape in [('TE',(nu,nv)),('TM',(nu+1,nv+1))]:
        offset=.5 if family=='TE' else 0.
        u=(np.arange(shape[0])+offset-nu/2)*du
        v=(np.arange(shape[1])+offset-nv/2)*dv
        x,y=np.meshgrid(u,v,indexing='ij'); r=np.hypot(x,y); theta=np.arctan2(y,x)
        columns=[]; metadata=[]; start=0
        for m in range(9):
            roots=(jnp_zeros if family=='TE' else jn_zeros)(m,4)
            for n,zero in enumerate(roots,1):
                base=jv(m,zero*r/.01)*(r<.01)
                patterns=[base*np.cos(m*theta)]
                if m:patterns.append(base*np.sin(m*theta))
                q=np.linalg.qr(np.stack([a.ravel() for a in patterns],axis=1))[0]
                columns.append(q); metadata.append((m,n,float(zero),slice(start,start+q.shape[1])))
                start+=q.shape[1]
        banks[family]=(np.column_stack(columns),metadata)
    return banks


def circular_templates(s,e,h):
    """Longitudinal projection scores, independent of runtime classification."""
    z0=config.sim_config.em_consts['z0']; fields={'TE':z0*h[2], 'TM':e[2]}
    energy=sum(np.linalg.norm(a)**2 for a in fields.values()); scored=[]
    patterns=circular_pattern_bank(s.Nu,s.Nv,s.du,s.dv)
    for family,field in fields.items():
        matrix,metadata=patterns[family]; coefficients=matrix.T@field.ravel()
        for m,n,zero,part in metadata:
                weights=coefficients[part]
                score=float(np.linalg.norm(weights)**2/max(energy,1e-300))
                scored.append(dict(name=f'{family}_{m}{n}',score=score,
                    partner_fractions=(abs(weights)**2/max(np.linalg.norm(weights)**2,1e-300)).tolist(),kc_radius=float(zero)))
    return sorted(scored,key=lambda r:r['score'],reverse=True)[:3]

def rectangular_templates(e,h):
    coeff,names=longitudinal_coefficients([e],[h],2,config.sim_config.em_consts['z0'])
    weights=abs(coeff[:,0])**2; weights/=max(weights.sum(),1e-300)
    return [dict(name=f'{names[j][0]}_{names[j][1]}{names[j][2]}',score=float(weights[j]))
            for j in np.argsort(weights)[-3:][::-1]]

def analyse(name,bank,public,output,geometry,diagnostic_groups=None,symmetry='auto'):
    print('ANALYSE',name,flush=True)
    axis=np.array([s.frequency for s in bank]); owner=EigenmodeSource(None)
    owner.normal_axis=2; owner.normal='z'; owner.direction='+'; owner.transverse_axes=(0,1)
    owner.tracking_config=EigenmodeTrackingConfig(extra_candidates=0)
    owner.port_index=1; owner.fallback_frequency=float(axis[len(axis)//2])
    owner.degenerate=(); owner.mode_polarizations={}; owner.tracking='auto'; owner.verification='none'
    owner._tracking_impedance=config.sim_config.em_consts['z0']
    mapping=track_solver_bank(owner,axis,bank,public)
    original_groups=owner.degenerate
    if diagnostic_groups is not None:
        owner.degenerate=tuple(g for g in owner.degenerate if set(g).issubset(diagnostic_groups))
    modes=tuple(range(1,owner._auto_tracking_mode_count+1))
    owner.mode_indices=modes; owner.mode_index=1; owner.mode_count=len(modes)
    grid=SimpleNamespace(dl=np.full(3,bank[0].du)); owner._tracking_grid=grid
    fields=[[owner._fields_from_solver_mode(s,m) for m in modes] for s in bank]
    e=[[v[0] for v in row] for row in fields]; h=[[v[1] for v in row] for row in fields]
    propagating=np.array([[s.complex_neff[j].real>0 for j in range(len(modes))] for s in bank])
    from testing.validation.general_subspace_tracking import align_validation_groups
    align_validation_groups(owner,grid,axis,bank,modes,e,h,propagating,symmetry=symmetry)
    report=dict(name=name,geometry=geometry,frequency_GHz=axis/1e9,
        detected=detect_symmetry(bank),symmetry_detection=owner.symmetry_detection,
        groups=owner.degenerate,blocks=[dict(modes=r['modes'],blocks=r['blocks'],irreps=r['irreps']) for r in owner.symmetry_diagnostics],
        neff=[s.complex_neff[:len(modes)] for s in bank],candidate_indices=mapping,anchors=[])
    report['diagnostic_group_selection']=diagnostic_groups
    report['requested_symmetry']=symmetry
    report['tracking_config']=asdict(owner.tracking_config)
    report['mixing_residual_limit']=owner.tracking_config.residual_tolerance
    report['symmetry_orientation']=[dict(modes=g['modes'],anchors=[
        dict(orientation=a.get('orientation'),residual=a.get('residual'),longitudinal=a.get('longitudinal'),axis_sensitive=a.get('axis_sensitive')) for a in g['anchors'].values()])
        for g in owner.symmetry_diagnostics]
    report['max_mixed_residual']=max((float(np.max(a['residual']))
        for g in owner.degenerate_diagnostics for a in g['anchors'] if 'residual' in a),default=0.)
    report['original_groups']=original_groups
    frames=[]; residuals=[]
    for k,s in enumerate(bank):
        frame=_frame(e[k],h[k],owner._tracking_impedance); frames.append(frame)
        final=copy(s)
        for kind,names in [(e,('Eu','Ev','Ew')),(h,('Hu','Hv','Hw'))]:
            for c,label in enumerate(names):
                a=getattr(final,label).copy()
                for j in range(len(modes)):a[...,j]=kind[k][j][c]
                setattr(final,label,a)
        residuals.extend(final.tracking_field_residuals(np.arange(len(modes))).tolist())
        row=[]
        for j,m in enumerate(modes):
            parity=[]; errors=[]
            for action in [mirror_u,mirror_v]:
                ee,hh=action(s,tuple(a[...,None] for a in e[k][j]),tuple(a[...,None] for a in h[k][j]))
                transformed=np.r_[*[a.ravel() for a in ee],*[owner._tracking_impedance*a.ravel() for a in hh]]
                v=frame[:,j]; sign=1 if np.vdot(v,transformed).real>=0 else -1
                parity.append(sign); errors.append(float(np.linalg.norm(transformed-sign*v)/np.linalg.norm(v)))
            scores=circular_templates(s,e[k][j],h[k][j]) if geometry=='circular' else rectangular_templates(e[k][j],h[k][j])
            row.append(dict(mode=m,templates=scores,parity=parity,parity_error=max(errors)))
        report['anchors'].append(row)
    overlap=[]
    for a,b in zip(frames,frames[1:]):
        overlap.append(abs(np.sum(a.conj()*b,axis=0))**2/(np.sum(abs(a)**2,axis=0)*np.sum(abs(b)**2,axis=0)))
    report.update(max_field_residual=max(residuals),channel_overlaps=overlap,
        min_channel_overlap=float(np.min(overlap)),
        minimum_power_eigenvalue=min(float(np.linalg.eigvalsh(_power(owner,grid,ee,hh)).min()) for ee,hh in zip(e,h)))
    if hasattr(bank[0],'edge_validation_metadata'):
        report['copper']=[s.edge_validation_metadata for s in bank]
    report=serial(report)
    (output/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/f'{name}_fields.npz',**{f'{kind}_{k}_{j}_{c}':a
        for kind,rows in [('E',e),('H',h)] for k,row in enumerate(rows) for j,field in enumerate(row) for c,a in enumerate(field)})
    print('COMPLETE',name,'symmetry',report['detected']['detected'],'groups',report['groups'],
          'residual',report['max_field_residual'],'overlap',report['min_channel_overlap'],flush=True)
    return report

def validate(output,followup=False,copper_only=False,higher_circular=False):
    output.mkdir(parents=True,exist_ok=True)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='running',
        description='FDFD banks only; no FDTD or manuscript changes.',cases=[],source_sha256={})
    for path in [Path(__file__),ROOT/'gprMax/eigenmode_tracking.py',ROOT/'gprMax/eigenmode_symmetry.py',ROOT/'testing/validation/modal_field_templates.py',ROOT/'gprMax/eigenmode_config.py',ROOT/'tests/fdfd_eigenmode_solver/test_surface_impedance_fdfd_operator.py']:
        manifest['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    def save(): (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save(); reports=[]; frequencies=[23e9,24e9,25e9]
    cases=[('square_isotropic',(2,2,2),24,10),('square_axial_anisotropy',(2,2,3),24,10),
           ('square_transverse_anisotropy',(2,3,2),24,10),
           ('circular_32',None,32,16),('circular_48',None,48,16),
           ('copper_square_24',None,24,10),('copper_square_32',None,32,10)]
    if followup:
        cases=[('voxel_circular_32',None,32,16),('voxel_circular_48',None,48,16),
               ('voxel_circular_64',None,64,16),
               ('copper_12_21_only_24',None,24,10),('copper_12_21_only_32',None,32,10)]
    if copper_only:
        cases=[('copper_relaxed_24',None,24,10),('copper_relaxed_32',None,32,10)]
    if higher_circular:
        cases=[('voxel_circular_higher_64',None,64,32),('voxel_circular_higher_96',None,96,32)]
        frequencies=[40e9,42e9,44e9]
    manifest['residual_tolerance']=EigenmodeTrackingConfig().residual_tolerance
    with _configured_solver_constants():
        for name,eps,cells,public in cases:
            entry=dict(name=name,cells=cells,epsilon=eps,status='running');manifest['cases'].append(entry);save()
            try:
                bank=[]
                for f in frequencies:
                    print('SOLVE',name,f/1e9,'GHz',flush=True)
                    s=(voxel_circular_solver(f,cells,count=40 if higher_circular else 20) if name.startswith('voxel') else
                       copper_solver(f,cells) if name.startswith('copper') else
                       circular_solver(f,cells=cells,count=20) if name.startswith('circular') else
                       fd_solver(f,(cells,cells),.02/cells,eps,count=16))
                    bank.append(s)
                if name.startswith('copper'):
                    raw=[]
                    for s in bank:
                        e=[tuple(getattr(s,a)[...,j] for a in ['Eu','Ev','Ew']) for j in range(10)]
                        h=[tuple(getattr(s,a)[...,j] for a in ['Hu','Hv','Hw']) for j in range(10)]
                        raw.append(dict(neff=s.complex_neff[:10],eigenvalues=s.eigenvalues[:10],
                            field_residual=s.tracking_field_residuals(np.arange(10)),
                            templates=[rectangular_templates(ee,hh) for ee,hh in zip(e,h)]))
                    (output/f'{name}_raw.json').write_text(json.dumps(serial(raw),indent=2)+'\n')
                result=analyse(name,bank,public,output,'circular' if 'circular' in name else 'square',
                    diagnostic_groups=(7,8,9,10) if '_only_' in name else None)
                reports.append(result);entry['status']='complete'
            except Exception as exc:
                entry.update(status='failed',error=str(exc));traceback.print_exc()
            save()
    manifest['status']='complete';manifest['finished_utc']=datetime.now(timezone.utc).isoformat();save()
    (output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('VALIDATION COMPLETE',[(c['name'],c['status']) for c in manifest['cases']],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--followup',action='store_true')
    p.add_argument('--copper-only',action='store_true')
    p.add_argument('--higher-circular',action='store_true');args=p.parse_args()
    validate(args.output,args.followup,args.copper_only,args.higher_circular)
