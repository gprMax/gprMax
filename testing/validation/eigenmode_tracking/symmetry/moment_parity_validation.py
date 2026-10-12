"""Physical field-bank validation of moment-first, mirror-parity orientation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from testing.validation.eigenmode_tracking.subspaces import general_subspace_tracking as validation
from testing.validation.eigenmode_tracking.symmetry.waveguide_symmetry_validation import circular_solver
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants


def plot_partners(report,solvers,groups,electric,magnetic,targets,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[0,len(solvers)//2,len(solvers)-1]
    middle=selected[1]
    lookup={}
    for parent in report['symmetry_orientation']:
        for k,records in parent['anchors'].items():
            for group,record in zip(parent['blocks'],records):
                for label,parity in zip(group,record['parity']):
                    lookup[int(k),label]=(parity,record['method'])
    fig,axes=plt.subplots(3,len(targets),figsize=(3.15*len(targets),8.6),squeeze=False,layout='constrained')
    for col,label in enumerate(targets):
        j=label-1
        # Select a longitudinal component and a single display phase at the
        # central anchor. Use that phase unchanged at all plotted anchors.
        choices=[('Ew',electric[middle][j][2]),('Hw',377.*magnetic[middle][j][2])]
        name,central=max(choices,key=lambda item:np.linalg.norm(item[1]))
        bank=electric if name=='Ew' else magnetic
        central=bank[middle][j][2]
        phase=np.exp(-1j*np.angle(central.ravel()[np.argmax(abs(central))]))
        fields=[(bank[k][j][2]*phase).real for k in selected]
        scale=max(max(np.max(abs(f)) for f in fields),1e-300)
        for row,(k,field) in enumerate(zip(selected,fields)):
            ax=axes[row,col]
            size=solvers[k].Nu*solvers[k].du*1000
            im=ax.imshow(field.T/scale,origin='lower',extent=[-size/2,size/2]*2,cmap='RdBu_r',vmin=-1,vmax=1)
            parity,method=lookup[k,label]
            signs='/'.join('+' if p==1 else '-' for p in parity)
            ax.set_title(f'B{label}, {name}: {report["axis"][k]:.4g}\nvector parity {signs}; {method.replace("_"," ")}',fontsize=9)
            ax.set(xlabel='u (mm)',ylabel='v (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='Signed longitudinal field, normalized per channel')
    fig.suptitle(report['name'].replace('_',' ').capitalize()+'\nFixed display phase across anchors; parity signs refer to the complete E/H vector field')
    fig.savefig(output/(report['name']+'_parity_fields.png'),dpi=170)
    fig.savefig(output/(report['name']+'_parity_fields.pdf'))
    plt.close(fig)


def validate(output):
    output.mkdir(parents=True,exist_ok=True)
    original_plot=validation.plot_case
    def plot_all(*args):
        original_plot(*args)
        plot_partners(*args)
    validation.plot_case=plot_all
    reports=[]
    axis=np.linspace(19.,21.,5)
    for name,public,targets,sizes in [('fundamental_moment_alignment',2,[1,2],(2,)),
                                     ('higher_order_parity_partners',10,[7,8,9,10],(2,2))]:
        bank=[validation.fd_solver(f*1e9,(16,16),.02/16,(2,2,2),count=16) for f in axis]
        if public==10:
            validation.stress_rotation(bank[2],[6,7,8,9])
        reports.append(validation.run_case(name,axis,bank,public,targets,output,'Frequency (GHz)',
                                           expected_sizes=sizes,symmetry='auto'))
    bank=[circular_solver(f*1e9,metallic=True,cells=32,count=18) for f in axis]
    reports.append(validation.run_case('circular_higher_order_partners',axis,bank,9,[6,7],output,
                                       'Frequency (GHz)',expected_sizes=(2,),symmetry='auto'))
    axis=np.sort(np.r_[np.linspace(-.12,.12,25),-.001,.001])
    bank=[validation.loaded_solver(t,cells=16) for t in axis]
    validation.stress_rotation(bank[int(np.argmin(abs(axis)))],[6,7,8,9])
    reports.append(validation.run_case('crossing_hybrid_parity_partners',axis,bank,10,[7,8,9,10],output,
                                       'Dielectric perturbation t',symmetry='auto'))
    for report in reports:
        assert all(g['applied']=='c4v' for g in report['symmetry_detection']['groups'])
        records=[d for p in report['symmetry_orientation'] for row in p['anchors'].values() for d in row]
        report['maximum_parity_residual']=max(d['parity_residual'] for d in records)
        report['minimum_oriented_channel_overlap']=min(min(d['channel_overlaps']) for d in records)
        assert report['maximum_parity_residual']<1e-8
        assert report['minimum_oriented_channel_overlap']>.99
        target_labels={m for g in report['groups'] for m in g['labels']}
        methods={d['method'] for p in report['symmetry_orientation'] for row in p['anchors'].values()
                 for b,d in zip(p['blocks'],row) if target_labels.intersection(b)}
        expected=({'electric_moment'} if report['name'] in ('fundamental_moment_alignment','circular_higher_order_partners')
                  else {'electric_moment','mirror_parity'} if report['name']=='crossing_hybrid_parity_partners'
                  else {'mirror_parity'})
        assert methods==expected, (report['name'],methods)
        report['target_orientation_methods']=sorted(methods)
        (output/(report['name']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    validation.write_manifest(output)
    manifest=json.loads((output/'manifest.json').read_text())
    root=Path(__file__).resolve().parents[4]
    for path in ('testing/validation/eigenmode_tracking/symmetry/moment_parity_validation.py','testing/validation/eigenmode_tracking/symmetry/waveguide_symmetry_validation.py',
                 'gprMax/sources.py','gprMax/eigenmode_config.py','gprMax/user_objects/cmds_multiuse.py'):
        manifest['source_sha256'][path]=hashlib.sha256((root/path).read_bytes()).hexdigest()
    manifest.pop('optional_symmetry_cases',None)
    manifest['cases']={r['name']:dict(axis=r['axis'],axis_label=r['axis_label'],
        targets=[m for g in r['groups'] for m in g['labels']],symmetry_selection='auto') for r in reports}
    manifest['geometry_sources']='See the hashed moment_parity_validation.py and its imported builders.'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text('# Electric-moment and mirror-parity validation\n\n'
        'Four full-vector FDFD studies: fundamental moment alignment, a higher-order square metallic quartet, '
        'a higher-order circular metallic pair, and crossing hybrid doublets in an inhomogeneous anisotropic guide. '
        'The square higher-order modes use mirror parity. Crossing modes use electric moments away from '
        'the coincidence and mirror parity where their moments vanish; '
        'the circular higher-order pair has usable moments and uses moment alignment. '
        'The square quartet and crossing eigenspace receive seeded complex rotations.\n\n'
        'Each case verifies E/H Maxwell residuals, power, subspace overlap, individual channel overlap, and vector mirror parity. '
        'Field plots use a fixed display phase across anchors. The +/- signs describe the full E/H vector transformation; '
        'individual component reflection signs depend on whether the component is polar or axial. '
        'These are mode-basis tests, not FDTD scattering simulations or automatic TE/TM classification.\n')
    print('MOMENT_PARITY_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with _configured_solver_constants():
        validate(args.output)
