"""Full-vector Maxwell validation of subspace recovery at mode coincidences."""
import argparse
from copy import copy
import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
from testing.validation.expanded_mode_tracking import fd_solver
from gprMax.eigenmode_config import EigenmodeTrackingConfig
from gprMax.eigenmode_tracking import track_solver_bank, align_groups, _power, _frame
from gprMax.sources import EigenmodeSource
import gprMax.config as config


def dielectric_solver(ez, *, frequency=20e9, cells=24, count=20, width=.04, core=.02, et=4.):
    """Square dielectric core in air, with a square PEC truncation window."""
    d = width / cells
    half = core / 2
    def material(x, y, value):
        return np.where((abs(x[:, None]) < half) & (abs(y[None, :]) < half), value, 1.).astype(complex)
    edge = (np.arange(cells+1)-cells/2)*d
    mid = (np.arange(cells)+.5-cells/2)*d
    eu, ev, ew = material(mid, edge, et), material(edge, mid, et), material(edge, edge, ez)
    eu[:, [0,-1]] = np.inf
    ev[[0,-1], :] = np.inf
    ew[:, [0,-1]] = np.inf
    ew[[0,-1], :] = np.inf
    solver = FDFD_2D_mode_solver(frequency,d,d,count-1,eu,ev,ew,
        np.ones_like(ev),np.ones_like(eu),np.ones((cells,cells)))
    solver.retain_tracking_operator = True
    solver.calculate_diagnostics = True
    solver.solve()
    return solver


def loaded_solver(t, *, cells=16):
    """C4v dielectric-loaded square PEC guide with an exact hybrid-pair crossing.

    At t=0 the guide is homogeneous (epsilon=2). A C4v inhomogeneous
    anisotropic perturbation splits the TE12/21/TM12/21 quartet into two
    hybrid doublets. Both signs are positive, passive material profiles.
    """
    width=.02; d=width/cells
    edge=np.linspace(-1,1,cells+1); mid=(edge[:-1]+edge[1:])/2
    def pattern(x,y):
        x=x[:,None]; y=y[None,:]
        return (np.cos(np.pi*x)+np.cos(np.pi*y)) + .35*np.cos(np.pi*x)*np.cos(np.pi*y)
    eu=(2+t*pattern(mid,edge)).astype(complex)
    ev=(2+t*pattern(edge,mid)).astype(complex)
    ew=(2+t*(1.1+.7*pattern(edge,edge))).astype(complex)
    eu[:,[0,-1]]=np.inf; ev[[0,-1],:]=np.inf
    ew[:,[0,-1]]=np.inf; ew[[0,-1],:]=np.inf
    s=FDFD_2D_mode_solver(20e9,d,d,11,eu,ev,ew,np.ones_like(ev),np.ones_like(eu),np.ones((cells,cells)))
    s.retain_tracking_operator=True; s.calculate_diagnostics=True; s.solve()
    return s


def stress_rotation(solver, indices):
    """Apply a valid complex change of basis only within an exact eigenspace."""
    ix=np.asarray(indices)
    rng=np.random.default_rng(412)
    rotation=np.linalg.qr(rng.normal(size=(len(ix),len(ix)))+1j*rng.normal(size=(len(ix),len(ix))))[0]
    assert np.ptp(solver.eigenvalues[ix].real)<1e-10
    for name in ('Eu','Ev','Ew','Hu','Hv','Hw'):
        a=getattr(solver,name); a[...,ix]=a[...,ix] @ rotation
    solver._normalize_modes_to_power()
    solver.eigenvectors=np.column_stack([np.r_[solver.Eu[...,i].ravel(order='F'),solver.Ev[...,i].ravel(order='F')] for i in range(solver.num_modes)])


def span_overlap(a,b):
    qa=np.linalg.qr(a)[0]; qb=np.linalg.qr(b)[0]
    return float(np.min(np.linalg.svd(qa.conj().T @ qb,compute_uv=False))**2)


def run_case(name, axis, solvers, public, targets, output, xlabel, expected_sizes=(2, 2), symmetry=None):
    raw=np.array([s.complex_neff.real.copy() for s in solvers])
    owner=EigenmodeSource(None)
    owner.tracking_config=EigenmodeTrackingConfig(extra_candidates=0)
    owner.normal_axis=2; owner.normal='z'; owner.direction='+'; owner.transverse_axes=(0,1)
    owner.port_index=1; owner.fallback_frequency=float(axis[0]); owner.degenerate=(); owner.mode_polarizations={}
    owner._tracking_impedance=config.sim_config.em_consts['z0']
    owner.tracking='auto'; owner.verification='none'
    owner.mode_symmetry = symmetry
    mapping=track_solver_bank(owner,axis,solvers,public)
    modes=tuple(range(1,owner._auto_tracking_mode_count+1))
    owner.mode_indices=modes; owner.mode_index=1; owner.mode_count=len(modes)
    grid=SimpleNamespace(dl=np.full(3,solvers[0].du)); owner._tracking_grid=grid
    electric=[]; magnetic=[]
    for solver in solvers:
        f=[owner._fields_from_solver_mode(solver,m) for m in modes]
        electric.append([v[0] for v in f]); magnetic.append([v[1] for v in f])
    align_groups(owner,grid,axis,solvers,modes,electric,magnetic,np.ones((len(axis),len(modes)),bool))
    frames=[_frame(e,h,owner._tracking_impedance) for e,h in zip(electric,magnetic)]
    decomposed = {tuple(r['modes']): r['blocks'] for r in owner.symmetry_diagnostics}
    basis_groups = tuple(b for g in owner.degenerate for b in decomposed.get(tuple(g), (g,)))
    groups=[tuple(m-1 for m in g) for g in basis_groups if set(g).intersection(targets)]
    assert sorted(map(len, groups)) == sorted(expected_sizes)
    assert set().union(*[set(g) for g in groups]) == set(np.asarray(targets)-1)
    group_records=[]
    for indices in groups:
        field_ratios=[]
        power_error=[]
        for k in range(len(axis)):
            ratios=[]
            for j in indices:
                e,h=electric[k][j],magnetic[k][j]
                ratios.append([float(np.linalg.norm(e[2])/np.sqrt(sum(np.linalg.norm(v)**2 for v in e))),
                               float(np.linalg.norm(h[2])/np.sqrt(sum(np.linalg.norm(v)**2 for v in h)))])
            field_ratios.append(ratios)
            p=_power(owner,grid,[electric[k][j] for j in indices],[magnetic[k][j] for j in indices])
            power_error.append(float(np.max(abs(p-np.eye(len(indices))))))
        group_records.append({'labels':[j+1 for j in indices],
            'overlap':[span_overlap(frames[k][:,indices],frames[k+1][:,indices]) for k in range(len(axis)-1)],
            'longitudinal_norm_ratios':field_ratios,'maximum_power_identity_error':max(power_error)})
    target_indices=np.array(targets)-1
    gram_errors=[]; power_minima=[]; field_residuals=[]
    for k in range(len(axis)):
        p=_power(owner,grid,[electric[k][i] for i in target_indices],[magnetic[k][i] for i in target_indices])
        gram_errors.append(float(np.max(abs(p-np.eye(len(targets))))))
        power_minima.append(float(np.min(np.linalg.eigvalsh(p))))
        # Check the final source fields, including the subsequent within-group
        # power whitening and parallel transport, against the Maxwell operator.
        final = copy(solvers[k])
        for names, bank in ((('Eu','Ev','Ew'), electric), (('Hu','Hv','Hw'), magnetic)):
            for component, field_name in enumerate(names):
                field = np.asarray(getattr(final, field_name)).copy()
                for j in range(len(modes)):
                    field[..., j] = bank[k][j][component]
                setattr(final, field_name, field)
        field_residuals.extend(final.tracking_field_residuals(np.arange(len(modes))).tolist())
    residual=max(float(np.max(record['residual'])) for group in owner.degenerate_diagnostics for record in group['anchors'] if record['active'])
    full_field_residual=max(field_residuals)
    assert np.isfinite(full_field_residual) and full_field_residual < 1e-8
    assert residual < 1e-9 and min(power_minima) > 0.9
    assert all(min(g['overlap']) > .99 for g in group_records)
    assert all(g['maximum_power_identity_error'] < 1e-10 for g in group_records)
    if name == 'dielectric_hybrid_pair_crossing':
        # Both E_z and H_z must be appreciable in each of the continued modes.
        assert min(np.asarray(g['longitudinal_norm_ratios'])[[0,-1]].min() for g in group_records) > .01
    report={'name':name,'axis':np.asarray(axis).tolist(),'axis_label':xlabel,
        'physical_frequency_GHz':[s.frequency/1e9 for s in solvers],
        'groups':group_records,'all_groups':owner.degenerate,'recovery_count':len(owner.tracking_diagnostics['subspace_recovery']),
        'symmetry': symmetry, 'source_subspaces': basis_groups,
        'symmetry_detection': owner.symmetry_detection,
        'symmetry_irreps': [dict(modes=r['modes'], blocks=r['blocks'], irreps=r['irreps']) for r in owner.symmetry_diagnostics],
        'symmetry_orientation': [dict(modes=r['modes'], blocks=r['blocks'], anchors={str(k): [
            dict(method=d['method'], parity=d['parity'].tolist(), parity_residual=d['parity_residual'],
                 moment_relative_min=d['moment_relative_min'], channel_overlaps=d['channel_overlaps'].tolist())
            for d in row.get('orientation', [])] for k,row in r['anchors'].items()}) for r in owner.symmetry_diagnostics],
        'maximum_mixed_residual':residual,'maximum_reconstructed_field_residual':full_field_residual,
        'maximum_full_power_identity_error':max(gram_errors),
        'minimum_full_power_eigenvalue':min(power_minima),
        'minimum_assignment_overlap':float(np.min(owner.tracking_diagnostics['overlaps'])),
        'raw_neff':raw.tolist(),'tracked_neff':np.array([s.complex_neff.real for s in solvers]).tolist(),
        'candidate_indices':mapping.tolist()}
    (output/(name+'.json')).write_text(json.dumps(report,indent=2))
    np.savez_compressed(output/(name+'_fields.npz'),axis=axis,
        **{f'{kind}_{k}_{j}_{c}':field for kind,bank in [('E',electric),('H',magnetic)]
           for k,row in enumerate(bank) for j in target_indices for c,field in enumerate(row[j])})
    plot_case(report,solvers,groups,electric,magnetic,targets,output)
    print(name,json.dumps({k:v for k,v in report.items() if k in ('recovery_count','maximum_mixed_residual','maximum_full_power_identity_error','minimum_assignment_overlap','all_groups')}),flush=True)
    return report


def plot_case(report,solvers,groups,electric,magnetic,targets,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0068ad','#df6b19','#37964b','#9847a6']
    title = {'dielectric_hybrid_pair_crossing': 'Hybrid-pair crossing: dielectric-loaded square PEC guide',
             'c4v_metallic_quartet': 'Supplied C4v symmetry: square PEC quartet decomposed into doublets',
             'c4v_dielectric_hybrid': 'Supplied C4v symmetry: hybrid dielectric doublets',
             'square_dielectric_core': 'Square dielectric core in air (finite port window)',
             'metallic_pair_crossing': 'Pair crossing: anisotropic square PEC guide',
             'persistent_metallic_quartet': 'Persistent fourfold span: square PEC guide'}.get(report['name'], report['name'].replace('_',' ').capitalize())
    axis=np.array(report['axis']); values=np.array(report['tracked_neff'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for g,record in zip(groups,report['groups']):
        i=groups.index(g); label='Span '+str(i+1)+' ('+','.join('B'+str(j+1) for j in g)+')'
        axes[0].plot(axis,values[:,g].mean(axis=1),color=colors[i],lw=2,label=label)
        axes[1].plot((axis[1:]+axis[:-1])/2,record['overlap'],color=colors[i],lw=2,label=label)
    axes[0].set(ylabel='Effective index',xlabel=report['axis_label'])
    axes[1].set(ylabel='Squared minimum subspace overlap',xlabel=report['axis_label'])
    lowest = min(min(g['overlap']) for g in report['groups'])
    axes[0].legend(fontsize=8); axes[1].set_ylim(min(.99999,lowest-.15*(1-lowest)),1.000005)
    axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle(title)
    fig.savefig(output/(report['name']+'_tracking.png'),dpi=170); fig.savefig(output/(report['name']+'_tracking.pdf')); plt.close(fig)
    # |Ez| and |Hz| are phase-independent and show hybrid character directly.
    selected=[0,len(axis)//2,len(axis)-1]
    scales=[]
    for bank in (electric,magnetic):
        scales.append(max(float(np.max(np.sqrt(sum(abs(bank[k][j][2])**2 for j in g)))) for g in groups for k in selected))
    fig,axes=plt.subplots(len(groups),6,figsize=(13,3.2*len(groups)),squeeze=False,layout='constrained')
    for row,g in enumerate(groups):
        for q,k in enumerate(selected):
            # Sum longitudinal intensities over a unit-power subgroup. This is
            # invariant to unitary choices of basis within that subgroup.
            for c,(kind,bank) in enumerate([('Ez',electric),('Hz',magnetic)]):
                intensity=sum(abs(bank[k][j][2])**2 for j in g)
                field=np.sqrt(intensity); field/=max(scales[c],1e-300)
                ax=axes[row,2*q+c]
                extent=np.array([-1,1,-1,1])*solvers[k].Nu*solvers[k].du*500
                im=ax.imshow(field.T,origin='lower',extent=extent,cmap='magma',vmin=0,vmax=1)
                ax.set_title(f'Span {row+1}: {kind}\n{axis[k]:.4g}',fontsize=10)
                ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Normalized subgroup longitudinal amplitude')
    fig.suptitle(title+' — tracked subspace fields')
    fig.savefig(output/(report['name']+'_fields.png'),dpi=170); fig.savefig(output/(report['name']+'_fields.pdf')); plt.close(fig)
    # Longitudinal and transverse fields can be in quadrature. Show |Ez|
    # alongside the real transverse pattern at a common phase per mode.
    fig,axes=plt.subplots(3,len(targets),figsize=(3*len(targets),8),squeeze=False,layout='constrained')
    for row,k in enumerate(selected):
        for col,label in enumerate(targets):
            e=electric[k][label-1]; h=magnetic[k][label-1]
            # Fix one common phase for all components of this displayed mode.
            flat=np.r_[e[0].ravel(),e[1].ravel()]; phase=np.exp(-1j*np.angle(flat[np.argmax(abs(flat))]))
            eu=.5*(e[0][:,:-1]+e[0][:,1:]); ev=.5*(e[1][:-1,:]+e[1][1:,:])
            scale=max(np.max(abs(eu)),np.max(abs(ev)),np.max(abs(e[2])),1e-300)
            size=solvers[k].Nu*solvers[k].du*1000
            centers=np.linspace(-size/2+size/(2*solvers[k].Nu),size/2-size/(2*solvers[k].Nu),solvers[k].Nu)
            ax=axes[row,col]
            im=ax.imshow(abs(e[2]).T/scale,origin='lower',extent=[-size/2,size/2]*2,cmap='magma',vmin=0,vmax=1)
            ax.quiver(centers[::2],centers[::2],(eu*phase).real[::2,::2].T/scale,(ev*phase).real[::2,::2].T/scale,color='white',scale=8)
            re=np.linalg.norm(e[2])/np.sqrt(sum(np.linalg.norm(v)**2 for v in e))
            rh=np.linalg.norm(h[2])/np.sqrt(sum(np.linalg.norm(v)**2 for v in h))
            conventional = report.get('conventional_names', {}).get(str(label), f'B{label}')
            ax.set_title(f'{conventional}, {axis[k]:.4g}\nEz/E={re:.3f}; Hz/H={rh:.3f}',fontsize=9)
            ax.set(xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='|Ez| / peak |E component|')
    fig.suptitle(title+' — individual continued fields\nArrows: transverse E; color: longitudinal E amplitude')
    fig.savefig(output/(report['name']+'_individual_fields.png'),dpi=160); plt.close(fig)


def validate(output):
    output.mkdir(parents=True,exist_ok=True)
    cases=[]
    cells=16; d=.02/cells
    k1=2*np.sin(np.pi/(2*cells))/d; k2=2*np.sin(np.pi/cells)/d
    crossing=2*(k1*k1+k2*k2)/(k2*k2)
    axis=crossing+np.linspace(-.12,.12,31)
    solvers=[fd_solver(20e9,(cells,cells),d,(2,2,p),count=12) for p in axis]
    stress_rotation(solvers[15],[4,5,6,7])
    cases.append(run_case('metallic_pair_crossing',axis,solvers,8,[5,6,7,8],output,'Longitudinal relative permittivity'))
    axis=np.sort(np.r_[np.linspace(-.12,.12,49),-.001,.001])
    solvers=[loaded_solver(t,cells=cells) for t in axis]
    stress_rotation(solvers[int(np.argmin(abs(axis)))],[6,7,8,9])
    cases.append(run_case('dielectric_hybrid_pair_crossing',axis,solvers,10,[7,8,9,10],output,'Dielectric perturbation t'))
    axis=np.linspace(18e9,22e9,21)
    solvers=[dielectric_solver(4,frequency=f,cells=24,count=16) for f in axis]
    cases.append(run_case('square_dielectric_core',axis/1e9,solvers,8,[1,2,7,8],output,'Frequency (GHz)'))
    axis=np.linspace(18e9,22e9,11)
    solvers=[fd_solver(f,(cells,cells),d,(2,2,2),count=12) for f in axis]
    cases.append(run_case('persistent_metallic_quartet',axis/1e9,solvers,10,[7,8,9,10],output,'Frequency (GHz)',expected_sizes=(4,)))
    (output/'summary.json').write_text(json.dumps(cases,indent=2))
    write_manifest(output)
    return cases


def write_manifest(output):
    """Record source hashes and runtime versions alongside the numerical data."""
    import scipy
    root = Path(__file__).resolve().parents[2]
    sources = ('gprMax/eigenmode_tracking.py',
               'gprMax/eigenmode_symmetry.py',
               'gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py',
               'testing/validation/general_subspace_tracking.py',
               'testing/validation/expanded_mode_tracking.py')
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'upstream_base': '0aeeec56393232cbe84c3c3b1866b06dd0275f28',
        'branch': 'codex/general-subspace-tracking',
        'validation': 'Full-vector FDFD tracking and source-basis construction',
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'source_sha256': {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        'tracking_config': vars(EigenmodeTrackingConfig(extra_candidates=0)),
        'exact_cluster_rotation_seed': 412,
        'optional_symmetry_cases': {
            'c4v_metallic_quartet': {'base_case':'persistent_metallic_quartet', 'mode_symmetry':'c4v'},
            'c4v_dielectric_hybrid': {'base_case':'dielectric_hybrid_pair_crossing', 'mode_symmetry':'c4v'},
        },
        'cases': {
            'metallic_pair_crossing': {'width_m': .02, 'cells': 16, 'frequency_Hz': 20e9,
                'epsilon_x_y': 2., 'epsilon_z': '31 values spanning discrete crossing +/- 0.12'},
            'dielectric_hybrid_pair_crossing': {'width_m': .02, 'cells': 16, 'frequency_Hz': 20e9,
                't': '49 uniform values from -0.12 to 0.12 plus +/-0.001',
                'pattern': 'p=cos(pi*x)+cos(pi*y)+0.35*cos(pi*x)*cos(pi*y), x,y in [-1,1]',
                'epsilon_x_y': '2+t*p', 'epsilon_z': '2+t*(1.1+0.7*p)'},
            'square_dielectric_core': {'window_m': .04, 'core_m': .02, 'cells': 24,
                'epsilon_core': 4., 'epsilon_cladding': 1., 'outer_boundary': 'PEC truncation',
                'frequency_Hz': '21 uniform values from 18e9 to 22e9'},
            'persistent_metallic_quartet': {'width_m': .02, 'cells': 16, 'epsilon': 2.,
                'frequency_Hz': '11 uniform values from 18e9 to 22e9'},
        },
    }
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')


def validate_symmetry(output):
    from gprMax.eigenmode_symmetry import irreducible_blocks
    output.mkdir(parents=True, exist_ok=True)
    controlled = []
    rng = np.random.default_rng(27)
    for copies in (1,2,3,4,6,10):
        n = 2*copies
        gauge = np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
        actions = [gauge.conj().T @ np.kron(np.eye(copies), d) @ gauge
                   for d in (np.array([[0,-1],[1,0]]),np.diag([1,-1]))]
        blocks, error = irreducible_blocks(actions)
        assert [b.shape[1] for b in blocks] == [2]*copies and error < 1e-10
        controlled.append({'dimension':n, 'irreducible_dimensions':[b.shape[1] for b in blocks],
                           'invariance_error':error, 'validation_type':'controlled C4v representation'})
    (output/'arbitrary_multiplicity.json').write_text(json.dumps(controlled,indent=2))
    axis = np.linspace(18.,22.,11)
    bank = [fd_solver(f*1e9,(16,16),.02/16,(2,2,2),count=12) for f in axis]
    cases = [run_case('c4v_metallic_quartet',axis,bank,10,[7,8,9,10],output,'Frequency (GHz)',symmetry='c4v')]
    axis = np.sort(np.r_[np.linspace(-.12,.12,49),-.001,.001])
    bank = [loaded_solver(t) for t in axis]
    stress_rotation(bank[int(np.argmin(abs(axis)))],[6,7,8,9])
    cases.append(run_case('c4v_dielectric_hybrid',axis,bank,10,[7,8,9,10],output,'Dielectric perturbation t',symmetry='c4v'))
    (output/'symmetry_summary.json').write_text(json.dumps(cases,indent=2))
    write_manifest(output)
    return cases


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--symmetry', action='store_true', help='Validate optional C4v decomposition and arbitrary multiplicity.')
    args=parser.parse_args()
    with _configured_solver_constants():
        validate_symmetry(args.output) if args.symmetry else validate(args.output)
