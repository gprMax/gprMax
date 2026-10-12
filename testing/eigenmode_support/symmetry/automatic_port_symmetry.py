"""Validate automatic symmetry detection on full-vector physical port banks.

Run with python -m testing.eigenmode_support.symmetry.automatic_port_symmetry --output PATH.
Outputs are new FDFD source-bank tests, not time-domain scattering runs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from testing.eigenmode_support.subspaces.general_subspace_tracking import (
    fd_solver, dielectric_solver, loaded_solver, stress_rotation, run_case, write_manifest,
)
from testing.eigenmode_support.symmetry.waveguide_symmetry_validation import circular_solver
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


def validate(output):
    output.mkdir(parents=True,exist_ok=True)
    axis=np.linspace(19.,21.,5)
    reports=[]
    cases=[
        ('automatic_square_metallic',lambda f:fd_solver(f*1e9,(16,16),.02/16,(2,2,2),count=16),10,[7,8,9,10]),
        ('automatic_square_dielectric',lambda f:dielectric_solver(4,frequency=f*1e9,cells=32,count=16),8,[1,2,7,8]),
        ('automatic_circular_metallic',lambda f:circular_solver(f*1e9,metallic=True),9,[1,2,6,7]),
        ('automatic_circular_dielectric',lambda f:circular_solver(f*1e9,metallic=False),8,[1,2,7,8]),
    ]
    for name,solve,public,targets in cases:
        bank=[solve(f) for f in axis]
        reports.append(run_case(name,axis,bank,public,targets,output,'Frequency (GHz)',symmetry='auto'))
    axis=np.sort(np.r_[np.linspace(-.12,.12,25),-.001,.001])
    bank=[loaded_solver(t,cells=16) for t in axis]
    stress_rotation(bank[int(np.argmin(abs(axis)))],[6,7,8,9])
    reports.append(run_case('automatic_crossing_hybrid_pairs',axis,bank,10,[7,8,9,10],output,
                            'Dielectric perturbation t',symmetry='auto'))
    for report in reports:
        assert report['symmetry_detection']['detected']=='c4v'
        assert all(g['applied']=='c4v' for g in report['symmetry_detection']['groups'])
    (output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    write_manifest(output)
    manifest=json.loads((output/'manifest.json').read_text())
    root=Path(__file__).resolve().parents[3]
    for name in ('testing/eigenmode_support/symmetry/automatic_port_symmetry.py','testing/eigenmode_support/symmetry/waveguide_symmetry_validation.py',
                 'gprMax/sources.py','gprMax/eigenmode_config.py','gprMax/user_objects/cmds_multiuse.py'):
        manifest['source_sha256'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    manifest.pop('optional_symmetry_cases',None)
    manifest['cases']={r['name']:dict(axis=r['axis'],axis_label=r['axis_label'],symmetry_selection='auto') for r in reports}
    manifest['geometry_sources']='Builders in automatic_port_symmetry.py and imported validation modules; their hashes are recorded.'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text('# Automatic port symmetry validation\n\n'
        'Five full-vector FDFD banks use automatic detection without a supplied symmetry label. '
        'Square and circular metallic and dielectric ports retain C4v. The final bank crosses two hybrid doublets, '
        'with a seeded complex basis rotation at the exact coincidence. Each case checks subgroup continuity, '
        'power, mixed eigen-residuals, and reconstructed Maxwell residuals.\n\n'
        'The pytest suite separately covers rectangular C2v ports, hexagonal and octagonal Yee staircases, '
        'asymmetric materials and masks, unsupported SIBC rows, transactional fallback, and source/receiver HDF5 output.\n\n'
        'See summary.json for numerical results and manifest.json for source hashes. These are FDFD source-bank validations.\n')
    print('AUTOMATIC_SYMMETRY_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with _configured_solver_constants():
        validate(args.output)
