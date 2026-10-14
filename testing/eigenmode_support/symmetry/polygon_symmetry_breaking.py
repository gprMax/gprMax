"""C6v/C8v PEC waveguides sampled on a square Cartesian Yee grid."""
import argparse
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import map_coordinates

from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.eigenmode_symmetry import c4_rotation, mirror_u, validate_c4v_materials
from gprMax.eigenmode_tracking import track_solver_bank, align_groups, _frame, _power
from gprMax.eigenmode_config import EigenmodeTrackingConfig
from gprMax.sources import EigenmodeSource
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
import gprMax.config as config


def polygon_inside(x, y, sides, apothem=.009):
    """Regular polygon with a face normal along +u and exact coordinate mirrors."""
    x,y=np.broadcast_arrays(x,y)
    normals=np.arange(sides)*2*np.pi/sides
    signed=np.stack([np.cos(a)*x+np.sin(a)*y for a in normals])
    # A tiny symmetric margin avoids different decisions on equivalent faces
    # caused by roundoff in trigonometric constants.
    return np.max(signed,axis=0) < apothem-1e-14


def polygon_solver(sides, cells, frequency=20e9, count=12):
    width=.024; d=width/cells
    edge=(np.arange(cells+1)-cells/2)*d
    mid=(np.arange(cells)+.5-cells/2)*d
    def material(x,y):
        return np.where(polygon_inside(x[:,None],y[None,:],sides),2.,np.inf).astype(complex)
    eu,ev,ew=material(mid,edge),material(edge,mid),material(edge,edge)
    solver=FDFD_2D_mode_solver(frequency,d,d,count-1,eu,ev,ew,np.ones_like(ev),np.ones_like(eu),np.ones((cells,cells)))
    solver.retain_tracking_operator=True
    solver.calculate_diagnostics=True
    solver.solve()
    return solver


def mirror_v(solver,e,h):
    return ((e[0][:,::-1],-e[1][:,::-1],e[2][:,::-1]),
            (-h[0][:,::-1],h[1][:,::-1],-h[2][:,::-1]))


def rotation_action(angle):
    """Interpolated geometric rotation, used only to diagnose lost symmetry.

    For 45 or 60 degrees this is not a native Yee-grid permutation. Its
    closure error includes interpolation error and is reported as such.
    """
    c,s=np.cos(angle),np.sin(angle)
    def action(solver,e,h):
        def rotate(fields,offsets):
            output=[]
            for component,field in enumerate(fields):
                u=(np.arange(field.shape[0])+offsets[component][0]-solver.Nu/2)*solver.du
                v=(np.arange(field.shape[1])+offsets[component][1]-solver.Nv/2)*solver.dv
                uu,vv=np.meshgrid(u,v,indexing='ij')
                source_u=c*uu+s*vv; source_v=-s*uu+c*vv
                sampled=[]
                for a in range(3):
                    coordinates=np.array([source_u/solver.du+solver.Nu/2-offsets[a][0],source_v/solver.dv+solver.Nv/2-offsets[a][1]])
                    sampled.append(np.stack([map_coordinates(fields[a][...,j],coordinates,order=1,mode='constant',cval=0.) for j in range(field.shape[-1])],axis=-1))
                output.append(c*sampled[0]-s*sampled[1] if component==0 else s*sampled[0]+c*sampled[1] if component==1 else sampled[2])
            return tuple(output)
        return rotate(e,((.5,0),(0,.5),(0,0))),rotate(h,((0,.5),(.5,0),(.5,.5)))
    return action


def span_action(solver,indices,action):
    e=tuple(getattr(solver,n)[...,indices] for n in ('Eu','Ev','Ew'))
    h=tuple(getattr(solver,n)[...,indices] for n in ('Hu','Hv','Hw'))
    def frame(ee,hh):
        return np.vstack([a.reshape(-1,len(indices)) for a in (*ee,*(377.*a for a in hh))])
    original=frame(e,h); te,th=action(solver,e,h); transformed=frame(te,th)
    representation=np.linalg.lstsq(original,transformed,rcond=None)[0]
    closure=float(np.linalg.norm(transformed-original@representation)/np.linalg.norm(transformed))
    return closure,representation


def inspect(solver,sides):
    try:
        validate_c4v_materials(solver)
        c4v=True
    except ValueError:
        c4v=False
    rows=[]
    for indices in ([0,1],[3,4]):
        gap=float(abs(solver.complex_neff[indices[0]]-solver.complex_neff[indices[1]]))
        rotations={}
        for name,action in (('mirror_u',mirror_u),('mirror_v',mirror_v),('native_90',c4_rotation),
                            ('interpolated_60' if sides==6 else 'interpolated_45',rotation_action(2*np.pi/sides))):
            closure,representation=span_action(solver,indices,action)
            rotations[name]=dict(closure=closure,trace_real=float(np.trace(representation).real),
                                eigenvalues_real=np.linalg.eigvals(representation).real.tolist(),
                                eigenvalues_imag=np.linalg.eigvals(representation).imag.tolist())
        rows.append(dict(ranks=(np.array(indices)+1).tolist(),neff_real=solver.complex_neff[indices].real.tolist(),gap_neff=gap,actions=rotations))
    return dict(sides=sides,cells=solver.Nu,spacing_mm=solver.du*1e3,c4v_material_check=c4v,
                maximum_Maxwell_residual=float(np.max(solver.tracking_field_residuals(np.arange(8)))),
                pairs=rows,neff=solver.complex_neff.real.tolist())


def explore(output):
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for sides in (6,8):
        for cells in (24,32,48,64):
            solver=polygon_solver(sides,cells)
            record=inspect(solver,sides)
            records.append(record)
            (output/'exploration.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps(record),flush=True)


def track_frequency_bank(sides, output, cells=64):
    axis=np.linspace(18e9,22e9,17)
    solvers=[polygon_solver(sides,cells,f) for f in axis]
    owner=EigenmodeSource(None)
    owner.tracking_config=EigenmodeTrackingConfig(extra_candidates=0)
    owner.normal_axis=2; owner.normal='z'; owner.direction='+'; owner.transverse_axes=(0,1)
    owner.port_index=1; owner.fallback_frequency=20e9
    owner.degenerate=(); owner.mode_polarizations={}
    owner._tracking_impedance=config.sim_config.em_consts['z0']
    owner.tracking='auto'; owner.verification='none'
    mapping=track_solver_bank(owner,axis,solvers,5)
    modes=tuple(range(1,owner._auto_tracking_mode_count+1))
    owner.mode_indices=modes; owner.mode_index=1; owner.mode_count=len(modes)
    grid=SimpleNamespace(dl=np.full(3,solvers[0].du)); owner._tracking_grid=grid
    electric=[]; magnetic=[]
    for solver in solvers:
        fields=[owner._fields_from_solver_mode(solver,m) for m in modes]
        electric.append([f[0] for f in fields]); magnetic.append([f[1] for f in fields])
    align_groups(owner,grid,axis,solvers,modes,electric,magnetic,np.ones((len(axis),len(modes)),bool))
    frames=[_frame(e,h,owner._tracking_impedance) for e,h in zip(electric,magnetic)]
    groups=list(owner.degenerate)
    # Split partners are independently tracked. Paired span overlaps are also
    # reported as a diagnostic; they do not authorize field mixing.
    pairs=[]
    for labels in ((1,2),(4,5)):
        ix=np.array(labels)-1
        overlap=[]
        for old,new in zip(frames[:-1],frames[1:]):
            q=np.linalg.qr(old[:,ix])[0]; p=np.linalg.qr(new[:,ix])[0]
            overlap.append(float(np.min(np.linalg.svd(q.conj().T@p,compute_uv=False))**2))
        pairs.append(dict(labels=labels,overlap=overlap,mixed=labels in groups))
    residual=[]; min_power=[]
    for k,solver in enumerate(solvers):
        check=copy(solver)
        for family,names in ((electric,('Eu','Ev','Ew')),(magnetic,('Hu','Hv','Hw'))):
            for a,name in enumerate(names):
                array=getattr(solver,name).copy()
                for j in range(len(modes)):
                    array[...,j]=family[k][j][a]
                setattr(check,name,array)
        residual.append(float(np.max(check.tracking_field_residuals(np.arange(len(modes))))))
        min_power.append(float(np.min(np.linalg.eigvalsh(_power(owner,grid,electric[k],magnetic[k])))))
    assert max(residual)<1e-8 and min(min_power)>0
    assert min(min(p['overlap']) for p in pairs)>.99
    assert ((1,2) in groups) == (sides==8)
    assert (4,5) not in groups
    if sides==6:
        assert not groups
    report=dict(sides=sides,cells=cells,frequency_GHz=(axis/1e9).tolist(),
                tracked_neff=np.array([s.complex_neff.real for s in solvers]).tolist(),
                groups=groups,pairs=pairs,candidate_indices=mapping.tolist(),
                minimum_assignment_overlap=float(np.min(owner.tracking_diagnostics['overlaps'])),
                maximum_reconstructed_Maxwell_residual=max(residual),minimum_power_eigenvalue=min(min_power))
    (output/f'c{sides}v_tracking.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/f'c{sides}v_fields.npz',**{f'{kind}_{k}_{j}_{a}':field for kind,bank in [('E',electric),('H',magnetic)]
        for k,row in enumerate(bank) for j in (0,1,3,4) for a,field in enumerate(row[j])})
    print('tracked',sides,report['groups'],report['minimum_assignment_overlap'],max(residual),flush=True)
    return report


def validate(output):
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for sides in (6,8):
        for cells in (24,32,48,64):
            record=inspect(polygon_solver(sides,cells),sides)
            assert record['c4v_material_check']==(sides==8)
            assert record['maximum_Maxwell_residual']<1e-8
            assert max(p['actions']['mirror_u']['closure'] for p in record['pairs'])<1e-8
            assert max(p['actions']['mirror_v']['closure'] for p in record['pairs'])<1e-8
            assert record['pairs'][0]['gap_neff']>(1e-5 if sides==6 else -1)
            if sides==8:
                assert record['pairs'][0]['gap_neff']<1e-10
                assert record['pairs'][1]['gap_neff']>1e-5
                assert max(p['actions']['native_90']['closure'] for p in record['pairs'])<1e-8
            records.append(record)
            print('mesh',sides,cells,[r['gap_neff'] for r in record['pairs']],flush=True)
    (output/'mesh_study.json').write_text(json.dumps(records,indent=2)+'\n')
    tracked=[track_frequency_bank(sides,output) for sides in (6,8)]
    (output/'tracking_summary.json').write_text(json.dumps(tracked,indent=2)+'\n')
    root=Path(__file__).resolve().parents[3]
    sources=('testing/eigenmode_support/symmetry/polygon_symmetry_breaking.py','gprMax/eigenmode_tracking.py','gprMax/eigenmode_symmetry.py','gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),branch='codex/general-subspace-tracking',
        geometry='Regular hexagonal/octagonal homogeneous PEC bore, face normal along +u',
        apothem_m=.009,window_m=.024,relative_permittivity=2.,relative_permeability=1.,
        cells=[24,32,48,64],mesh_study_frequency_Hz=20e9,tracking_frequencies_Hz=np.linspace(18e9,22e9,17).tolist(),
        tracking_cells=64,candidates_per_solve=12,public_modes=5,tracking_config=vars(EigenmodeTrackingConfig(extra_candidates=0)),
        material_interface='Native Yee point sampling with symmetric boundary tolerance 1e-14 m',
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        rotation_diagnostic='60/45-degree rotations use linear interpolation; closure includes interpolation error. Mirrors/90-degree rotations use exact grid permutations.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'.gitignore').write_text('*_fields.npz\n')
    plot_results(output,records,tracked)
    write_readme(output,records,tracked)


def plot_results(output,records,tracked):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0068ad','#dd6b20','#30975c','#963eaa']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for col,sides in enumerate((6,8)):
        rows=[r for r in records if r['sides']==sides]
        for j,tag in enumerate(('Leading pair (B1/B2)','Higher pair (B4/B5)')):
            axes[0,col].plot([r['spacing_mm'] for r in rows],[r['pairs'][j]['gap_neff'] for r in rows],'-o',color=colors[j],label=tag)
        axes[0,col].invert_xaxis()
        axes[0,col].set(xlabel='Grid spacing (mm): refinement →',ylabel=r'$|\Delta n_{\mathrm{eff}}|$',title=f'C{sides}v geometry → '+('C2v Yee operator' if sides==6 else 'C4v Yee operator'))
        axes[0,col].legend(fontsize=9)
        r=tracked[col]; x=r['frequency_GHz']; neff=np.array(r['tracked_neff'])
        for j,mode in enumerate((0,1,3,4)):
            axes[1,col].plot(x,neff[:,mode],color=colors[j],ls='--' if j%2 else '-',lw=2,label=f'B{mode+1}')
        axes[1,col].set(xlabel='Frequency (GHz)',ylabel='Tracked effective index',title='Split branches remain separate' if sides==6 else 'One retained doublet; one split pair')
        axes[1,col].legend(fontsize=9,ncol=2)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle('Symmetry breaking of hexagonal and octagonal PEC ports on a Yee grid')
    fig.savefig(output/'symmetry_breaking.png',dpi=175); fig.savefig(output/'symmetry_breaking.pdf'); plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for row,sides in enumerate((6,8)):
        data=np.load(output/f'c{sides}v_fields.npz'); k=8
        for col,mode in enumerate((0,1,3,4)):
            e=[data[f'E_{k}_{mode}_{a}'] for a in range(3)]
            h=[data[f'H_{k}_{mode}_{a}'] for a in range(3)]
            hz=abs(h[2]); scale=max(np.max(abs(a)) for a in h)
            ax=axes[row,col]
            im=ax.imshow(hz.T/scale,origin='lower',extent=[-12,12]*2,cmap='magma',vmin=0,vmax=1)
            angles=(np.arange(sides+1)+.5)*2*np.pi/sides
            radius=9/np.cos(np.pi/sides)
            ax.plot(radius*np.cos(angles),radius*np.sin(angles),color='cyan',lw=.8)
            eu=.5*(e[0][:,:-1]+e[0][:,1:]); ev=.5*(e[1][:-1,:]+e[1][1:,:])
            flat=np.r_[eu.ravel(),ev.ravel()]; phase=np.exp(-1j*np.angle(flat[np.argmax(abs(flat))]))
            norm=max(np.max(abs(eu)),np.max(abs(ev)))
            centers=(np.arange(64)+.5-32)*24/64
            ax.quiver(centers[::4],centers[::4],(eu*phase).real[::4,::4].T/norm,(ev*phase).real[::4,::4].T/norm,color='white',scale=14,width=.004)
            ax.set_title(f'C{sides}v geometry: B{mode+1}',fontsize=10)
            ax.set(xlabel='x (mm)',ylabel='y (mm)')
        data.close()
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label=r'$|H_z|$ / peak H component')
    fig.suptitle('Tracked fields at 20 GHz, 0.375 mm grid\nWhite arrows: transverse E; cyan: ideal polygon boundary')
    fig.savefig(output/'tracked_fields.png',dpi=180); fig.savefig(output/'tracked_fields.pdf'); plt.close(fig)


def write_readme(output,records,tracked):
    fine={r['sides']:r for r in records if r['cells']==64}
    lines=['# C6v/C8v symmetry breaking on a Cartesian Yee grid','',
        'Regular hexagonal and octagonal homogeneous PEC waveguides, apothem 9 mm, relative permittivity 2. Full-vector FDFD solves use native Yee point sampling. The manuscript is unchanged.','',
        '| Geometry | Surviving exact transverse symmetry | Leading-pair gap at 20 GHz | Higher-pair gap |',
        '|---|---|---:|---:|']
    for sides,group in ((6,'C2v'),(8,'C4v')):
        r=fine[sides]
        lines.append(f"| C{sides}v | {group} | {r['pairs'][0]['gap_neff']:.6g} | {r['pairs'][1]['gap_neff']:.6g} |")
    lines += ['', 'Gaps are absolute effective-index differences on the 64-by-64 grid (0.375 mm spacing). Native reflections preserve both tested spans to numerical precision. Native 90-degree rotation preserves the octagonal spans. The hexagonal material masks fail the C4v check.', '',
        'The octagonal leading pair remains a two-dimensional C4v representation. The higher pair has 90-degree representation -I and opposite mirror parities: it resolves into C4v singlets. Thus loss of 45-degree symmetry splits some doublets while other doublets remain protected. The hexagonal pairs are tracked as separate modes.', '',
        'The mesh study uses 1.0, 0.75, 0.5, and 0.375 mm cells. Staircase errors are nonmonotonic in the octagonal case. These samples demonstrate symmetry loss and partial symmetry protection; they do not establish a convergence rate.', '',
        'Rotations by 60 or 45 degrees require interpolation between native Yee locations. Their reported span-closure defects include interpolation error and are diagnostic, not exact group actions. The corresponding traces approach the expected E1/E2 character patterns. Frequency tracking uses 17 anchors from 18 to 22 GHz with nominal mixing tolerances.', '']
    for r in tracked:
        lines.append(f"- C{r['sides']}v: detected mixing groups {r['groups']}; minimum assignment overlap {r['minimum_assignment_overlap']:.9f}; maximum reconstructed Maxwell residual {r['maximum_reconstructed_Maxwell_residual']:.3e}.")
    lines += ['', '## Reproduce', '', '```powershell', "$env:OMP_NUM_THREADS='1'", "$env:OPENBLAS_NUM_THREADS='1'", 'python -m testing.eigenmode_support.symmetry.polygon_symmetry_breaking --output results/eigenmode_tracking/polygon_symmetry', '```', '',
        '![Symmetry splitting and tracking](symmetry_breaking.png)', '', '![Tracked fields](tracked_fields.png)', '']
    (output/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--explore',action='store_true')
    args=parser.parse_args()
    with _configured_solver_constants():
        explore(args.output) if args.explore else validate(args.output)
