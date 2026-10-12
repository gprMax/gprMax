"""Reproducible square/circular metallic and dielectric subspace validation."""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np

from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from testing.validation.eigenmode_tracking.subspaces.general_subspace_tracking import (
    fd_solver, dielectric_solver, loaded_solver, stress_rotation, run_case, write_manifest,
)
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants


def circular_solver(frequency=20e9, *, metallic=True, ez=2., cells=32, count=18):
    """Centered circular PEC bore, or eps=4 dielectric rod in an air window.

    Native Yee locations sample the circular interface without smoothing.
    The dielectric rod uses an explicitly reported finite square PEC window.
    """
    width = .024 if metallic else .04
    radius = .01
    d = width/cells
    edge = (np.arange(cells+1)-cells/2)*d
    mid = (np.arange(cells)+.5-cells/2)*d
    def material(x, y, value):
        inside = x[:,None]**2+y[None,:]**2 < radius**2
        return np.where(inside, value, np.inf if metallic else 1.).astype(complex)
    et = 2. if metallic else 4.
    eu, ev, ew = material(mid,edge,et), material(edge,mid,et), material(edge,edge,ez if metallic else 4.)
    eu[:,[0,-1]] = np.inf
    ev[[0,-1],:] = np.inf
    ew[:,[0,-1]] = np.inf
    ew[[0,-1],:] = np.inf
    solver = FDFD_2D_mode_solver(frequency,d,d,count-1,eu,ev,ew,np.ones_like(ev),np.ones_like(eu),np.ones((cells,cells)))
    solver.retain_tracking_operator = True
    solver.calculate_diagnostics = True
    solver.solve()
    return solver


def explore():
    for metal in (True,False):
        s = circular_solver(metallic=metal)
        from gprMax.eigenmode_symmetry import validate_c4v_materials
        validate_c4v_materials(s)
        print('metallic',metal,'neff',s.complex_neff.real.tolist(), flush=True)
        print('Ez/E', [float(np.linalg.norm(s.Ew[...,j])/np.sqrt(sum(np.linalg.norm(getattr(s,n)[...,j])**2 for n in ('Eu','Ev','Ew')))) for j in range(s.num_modes)], flush=True)


def validate(output):
    output.mkdir(parents=True,exist_ok=True)
    cases = []
    axis = np.linspace(18.,22.,11)
    bank = [fd_solver(f*1e9,(16,16),.02/16,(2,2,2),count=12) for f in axis]
    stress_rotation(bank[5],[6,7,8,9])
    cases.append(run_case('square_pec_symmetry',axis,bank,10,[7,8,9,10],output,'Frequency (GHz)',symmetry='c4v'))
    bank = [dielectric_solver(4,frequency=f*1e9,cells=32,count=16) for f in axis]
    cases.append(run_case('square_dielectric_symmetry',axis,bank,8,[1,2,7,8],output,'Frequency (GHz)',symmetry='c4v'))
    bank = [circular_solver(f*1e9,metallic=True) for f in axis]
    cases.append(run_case('circular_pec_symmetry',axis,bank,9,[1,2,6,7],output,'Frequency (GHz)',symmetry='c4v'))
    bank = [circular_solver(f*1e9,metallic=False) for f in axis]
    cases.append(run_case('circular_dielectric_symmetry',axis,bank,8,[1,2,7,8],output,'Frequency (GHz)',symmetry='c4v'))
    axis = np.sort(np.r_[np.linspace(-.12,.12,49),-.001,.001])
    bank = [loaded_solver(t) for t in axis]
    stress_rotation(bank[int(np.argmin(abs(axis)))],[6,7,8,9])
    cases.append(run_case('crossing_hybrid_doublets',axis,bank,10,[7,8,9,10],output,'Dielectric perturbation t',symmetry='c4v'))
    cells=16; d=.02/cells
    k1=2*np.sin(np.pi/(2*cells))/d; k2=2*np.sin(np.pi/cells)/d
    center=2*(k1*k1+k2*k2)/(k2*k2)
    axis=center+np.linspace(-.12,.12,31)
    bank=[fd_solver(20e9,(cells,cells),d,(2,2,p),count=12) for p in axis]
    stress_rotation(bank[15],[4,5,6,7])
    cases.append(run_case('crossing_metallic_doublets',axis,bank,8,[5,6,7,8],output,'Longitudinal relative permittivity',symmetry='auto',expected_sizes=(1,1,2)))
    (output/'summary.json').write_text(json.dumps(cases,indent=2)+'\n')
    write_manifest(output)
    manifest = json.loads((output/'manifest.json').read_text())
    manifest['source_sha256']['testing/validation/eigenmode_tracking/symmetry/waveguide_symmetry_validation.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['cases'] = {
        'square_pec_symmetry':dict(width_m=.02,cells=16,epsilon=2.,frequency_GHz=axis_description(),symmetry_selection='c4v'),
        'square_dielectric_symmetry':dict(window_m=.04,core_width_m=.02,cells=32,epsilon_core=4.,epsilon_cladding=1.,outer_boundary='PEC truncation',frequency_GHz=axis_description()),
        'circular_pec_symmetry':dict(window_m=.024,radius_m=.01,cells=32,epsilon=2.,boundary='point-sampled Yee staircase PEC masks',frequency_GHz=axis_description()),
        'circular_dielectric_symmetry':dict(window_m=.04,radius_m=.01,cells=32,epsilon_core=4.,epsilon_cladding=1.,outer_boundary='PEC truncation',frequency_GHz=axis_description()),
        'crossing_hybrid_doublets':dict(width_m=.02,cells=16,frequency_GHz=20.,t='49 values from -0.12 to 0.12, plus +/-0.001',epsilon='See loaded_solver; positive C4v inhomogeneous anisotropic profile'),
        'crossing_metallic_doublets':dict(width_m=.02,cells=16,frequency_GHz=20.,epsilon_transverse=2.,epsilon_longitudinal='31 values centered on discrete TE20/02-TM12/21 crossing, +/-0.12'),
    }
    manifest.pop('optional_symmetry_cases',None)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    controlled_multiplicity(output)
    summarize(output)
    return cases


def axis_description():
    return '11 equally spaced frequencies from 18 to 22 GHz'


def controlled_multiplicity(output):
    from gprMax.eigenmode_symmetry import irreducible_blocks, c4v_irrep_names
    rng=np.random.default_rng(27)
    records=[]
    for copies in (1,2,3,4,6,10):
        n=2*copies
        gauge=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
        actions=[gauge.conj().T @ np.kron(np.eye(copies),d) @ gauge for d in (np.array([[0,-1],[1,0]]),np.diag([1,-1]))]
        blocks,error=irreducible_blocks(actions)
        assert [b.shape[1] for b in blocks] == [2]*copies and error < 1e-10
        transform=np.column_stack(blocks)
        names=c4v_irrep_names([transform.conj().T @ d @ transform for d in actions],[np.arange(2*j,2*j+2) for j in range(copies)])
        records.append(dict(dimension=n,irreducible_dimensions=[2]*copies,irreps=names,invariance_error=error))
    (output/'arbitrary_multiplicity.json').write_text(json.dumps(records,indent=2)+'\n')


def summarize(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    cases=json.loads((output/'summary.json').read_text())
    by_name={r['name']:r for r in cases}
    colors=['#0068ad','#de6d18']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,name in enumerate(('crossing_metallic_doublets','crossing_hybrid_doublets')):
        r=by_name[name]; x=np.array(r['axis']); neff=np.array(r['tracked_neff'])
        for j,g in enumerate(r['groups']):
            names=[r['conventional_names'].get(str(m),'B'+str(m)) for m in g['labels']]
            label=' / '.join(names) if row==0 else 'Hybrid span '+str(j+1)
            axes[row,0].plot(x,neff[:,np.array(g['labels'])-1].mean(axis=1),color=colors[j],lw=2,label=label)
            axes[row,1].plot((x[:-1]+x[1:])/2,g['overlap'],color=colors[j],lw=2,label=label)
        axes[row,0].set_ylabel('Effective index')
        axes[row,1].set_ylabel('Squared minimum subspace overlap')
        axes[row,1].ticklabel_format(axis='y',style='plain',useOffset=False)
        axes[row,0].legend(fontsize=9)
        for ax in axes[row]:
            ax.set_xlabel(r['axis_label']); ax.grid(alpha=.2)
        axes[row,0].set_title('Metallic-guide degenerate pairs' if row==0 else 'Crossing hybrid-mode pairs')
        axes[row,1].set_title('Continuity through the coincidence')
    fig.suptitle('Tracked subspaces through crossings\nFull candidate span is randomly mixed at the crossing anchor',fontsize=13)
    fig.savefig(output/'crossing_subspaces.png',dpi=170)
    fig.savefig(output/'crossing_subspaces.pdf'); plt.close(fig)

    definitions=[('square_pec_symmetry','Square PEC',20),('square_dielectric_symmetry','Square dielectric core',40),
                 ('circular_pec_symmetry','Circular PEC staircase',24),('circular_dielectric_symmetry','Circular dielectric core',40)]
    fig,axes=plt.subplots(4,4,figsize=(10,10),layout='constrained')
    for row,(name,title,width) in enumerate(definitions):
        r=by_name[name]; k=len(r['axis'])//2
        with np.load(output/(name+'_fields.npz')) as data:
            fields=[]
            for group in r['groups']:
                for kind in ('E','H'):
                    fields.append(np.sqrt(sum(abs(data[f'{kind}_{k}_{m-1}_2'])**2 for m in group['labels'])))
        scale=[max(fields[0].max(),fields[2].max(),1e-300),max(fields[1].max(),fields[3].max(),1e-300)]
        for col,field in enumerate(fields):
            ax=axes[row,col]
            im=ax.imshow(field.T/scale[col%2],origin='lower',extent=[-width/2,width/2]*2,cmap='magma',vmin=0,vmax=1)
            if 'circular' in name:
                ax.add_patch(Circle((0,0),10,fill=False,edgecolor='cyan',lw=.7))
            elif 'dielectric' in name:
                ax.add_patch(Rectangle((-10,-10),20,20,fill=False,edgecolor='cyan',lw=.7))
            ax.set_title(('TE pair' if col<2 else 'TM pair') if row==0 else f'Span {col//2+1}',fontsize=10)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel(title+'\ny (mm)' if col==0 else 'y (mm)')
            ax.text(.04,.92,('E' if col%2==0 else 'H')+'$_z$',transform=ax.transAxes,color='white',fontsize=11)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Normalized subspace longitudinal amplitude')
    fig.suptitle('Tracked subspace fields at 20 GHz\nSquare root of summed longitudinal intensities for each pair',fontsize=13)
    fig.savefig(output/'waveguide_subspace_fields.png',dpi=180)
    fig.savefig(output/'waveguide_subspace_fields.pdf'); plt.close(fig)

    lines=['# Waveguide symmetry validation','',
        'Branch: `codex/general-subspace-tracking`. Manuscript unchanged. These are full-vector FDFD eigenmode and final source-basis checks. Separate short FDTD integration tests check source/receiving-port wiring and HDF5 output on all three port normals.','',
        '## Numerical results','',
        '| Case | Selected source spans | Minimum adjacent span overlap | Maximum reconstructed Maxwell residual |',
        '|---|---|---:|---:|']
    for r in cases:
        lines.append(f"| {r['name']} | {' + '.join(str(len(g['labels'])) for g in r['groups'])} | {min(min(g['overlap']) for g in r['groups']):.9f} | {r['maximum_reconstructed_field_residual']:.3e} |")
    lines += ['', 'C4v irreducible blocks have dimension one or two. Algebraic representation tests cover 2, 4, 6, 8, 12, and 20 dimensions (`arbitrary_multiplicity.json`). Repeated E copies are selected by a numerical reference basis and continued across anchors.', '',
        'The two crossing cases vary a material parameter at 20 GHz. At the exact coincidence the input fields are deliberately rotated by a seeded complex unitary matrix. The hybrid case uses a C4v inhomogeneous dielectric filling in a square PEC guide. The separate square and circular dielectric-core cases use finite square PEC truncation windows; their results validate tracking for those discrete port problems.', '',
        '## Reproduce', '', '```powershell', "$env:OMP_NUM_THREADS='1'", "$env:OPENBLAS_NUM_THREADS='1'", 'python -m testing.validation.eigenmode_tracking.symmetry.waveguide_symmetry_validation --output results/eigenmode_tracking/waveguide_symmetry', '```', '',
        'Geometry, material values, solver versions, and source hashes are recorded in `manifest.json`. Per-case JSON files contain the anchor-by-anchor tracking and power metrics. NPZ files retain the final plotted field arrays locally; they are excluded from Git.', '',
        '## Plots', '', '![Crossings](crossing_subspaces.png)', '', '![Subspace fields](waveguide_subspace_fields.png)', '']
    (output/'README.md').write_text('\n'.join(lines))
    (output/'.gitignore').write_text('*_fields.npz\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--explore',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    with _configured_solver_constants():
        if args.explore:
            explore()
        elif args.summarize:
            summarize(args.output)
        else:
            validate(args.output)
