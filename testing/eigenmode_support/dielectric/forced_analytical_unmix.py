"""Counterfactual: can purifying a grid avoided crossing remain an eigenmode?"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from testing.eigenmode_support.dielectric.analytical_step_index import modes, frame
from testing.eigenmode_support.identification.combined_identity_audit import solve
from testing.eigenmode_support.identification.tracking_classification_edges import serial
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
import gprMax.config as config


def run(output, crossing_manifest):
    output.mkdir(parents=True, exist_ok=False)
    f = json.loads(crossing_manifest.read_text())['analytical_crossing_Hz']
    with _configured_solver_constants():
        s = solve('circular_dielectric', f, 64, 56)
        z0 = config.sim_config.em_consts['z0']
        names = ('Eu','Ev','Ew','Hu','Hv','Hw')
        raw = np.column_stack([np.concatenate([getattr(s,name)[...,j].ravel()*(1 if i<3 else z0)
                    for i,name in enumerate(names)]) for j in range(s.num_modes)])
        references = {r['name']:r for r in modes(f,max_m=4)}
        analytical = np.column_stack([frame(references['TE0,3'],64),frame(references['HE4,2'],64,partner=1)])
        q = np.linalg.qr(analytical)[0]
        strength = np.linalg.norm(q.conj().T @ (raw/np.linalg.norm(raw,axis=0)),axis=0)**2
        selected = np.argsort(strength)[-2:]
        records=[]
        for k,label in enumerate(('TE03','HE42_A2')):
            coefficient = np.linalg.lstsq(raw[:,selected],analytical[:,k],rcond=None)[0]
            candidate = raw[:,selected]@coefficient
            overlap = abs(np.vdot(candidate,analytical[:,k]))**2/(np.linalg.norm(candidate)**2*np.linalg.norm(analytical[:,k])**2)
            trial = deepcopy(s); j=int(selected[0])
            for name in names:
                getattr(trial,name)[...,j]=getattr(s,name)[...,selected]@coefficient
            def residual(n):
                trial.operator_neff[j]=n
                return float(trial.tracking_field_residuals([j])[0])
            bounds = (float(s.operator_neff[selected].real.min()),float(s.operator_neff[selected].real.max()))
            best = minimize_scalar(residual,bounds=bounds,method='bounded',options={'xatol':1e-14})
            records.append(dict(target=label,projected_reference_overlap=float(overlap),
                optimized_neff=float(best.x),minimum_full_field_residual=float(best.fun),
                exceeds_production_residual_limit=bool(best.fun>1e-7),coefficients=coefficient))
        result=dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),frequency_Hz=f,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),raw_indices=(selected+1),
            raw_neff=s.operator_neff[selected],raw_field_residuals=s.tracking_field_residuals(selected),trials=records)
        (output/'results.json').write_text(json.dumps(serial(result),indent=2)+'\n');print(serial(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--crossing-manifest',type=Path,required=True)
    a=p.parse_args();run(a.output,a.crossing_manifest)
