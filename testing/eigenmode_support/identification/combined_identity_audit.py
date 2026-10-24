"""Audit the current automatic pipeline without changing runtime policy.

Run with python -m testing.eigenmode_support.identification.combined_identity_audit.
Random rotations are confined to numerically exact complex eigenspaces.
Dielectric cases are finite-window eigenproblems, not open-boundary solutions.
"""
import argparse
from copy import copy, deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

import gprMax.config as config
from gprMax.eigenmode_config import EigenmodeTrackingConfig
from gprMax.eigenmode_tracking import track_solver_bank, align_groups, _frame, _power
from gprMax.eigenmode_symmetry import mirror_u, mirror_v
from gprMax.sources import EigenmodeSource
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
from testing.eigenmode_support.identification.tracking_classification_edges import (
    copper_solver, circular_templates, rectangular_templates, serial)
from testing.eigenmode_support.subspaces.expanded_mode_tracking import fd_solver
from testing.eigenmode_support.subspaces.general_subspace_tracking import dielectric_solver
from testing.eigenmode_support.symmetry.waveguide_symmetry_validation import circular_solver as rod_solver
from testing.eigenmode_support.profiles.cylindrical_modal_profiles import circular_solver


def solve(case, frequency, cells, count=32):
    if case == 'square_PEC':
        return fd_solver(frequency, (cells, cells), .02/cells, (1, 1, 1), count=count)
    if case == 'square_copper':
        return copper_solver(frequency, cells, count)
    if case.startswith('circular_') and 'dielectric' not in case:
        return circular_solver(frequency, cells, case.split('_')[1], count)
    if case == 'square_dielectric':
        return dielectric_solver(4., frequency=frequency, cells=cells, count=count)
    return rod_solver(frequency, metallic=False, cells=cells, count=count)


def rotate(bank, seed):
    rng = np.random.default_rng(seed)
    spans = []
    for solver in bank:
        used = set()
        groups = []
        for i, value in enumerate(solver.eigenvalues):
            if i in used:
                continue
            ix = np.flatnonzero(abs(solver.eigenvalues-value) < 1e-11*max(1., abs(value)))
            used.update(ix.tolist())
            u = np.linalg.qr(rng.normal(size=(len(ix), len(ix))) +
                             1j*rng.normal(size=(len(ix), len(ix))))[0]
            for name in ('Eu', 'Ev', 'Ew', 'Hu', 'Hv', 'Hw'):
                a = getattr(solver, name)
                a[..., ix] = a[..., ix] @ u
            if len(ix) > 1:
                groups.append((ix+1).tolist())
        solver._normalize_modes_to_power()
        solver.eigenvectors = np.column_stack([
            np.r_[solver.Eu[..., i].ravel(order='F'), solver.Ev[..., i].ravel(order='F')]
            for i in range(solver.num_modes)])
        spans.append(groups)
    return spans


def analyse(bank, case, public):
    owner = EigenmodeSource(None)
    owner.normal_axis = 2; owner.normal = 'z'; owner.direction = '+'; owner.transverse_axes = (0, 1)
    owner.tracking_config = EigenmodeTrackingConfig(extra_candidates=0)
    owner.port_index = 1; owner.fallback_frequency = bank[len(bank)//2].frequency
    owner.degenerate = (); owner.mode_polarizations = {}; owner.tracking = 'auto'; owner.verification = 'none'
    owner._tracking_impedance = config.sim_config.em_consts['z0']
    axis = np.array([s.frequency for s in bank])
    mapping = track_solver_bank(owner, axis, bank, public)
    modes = tuple(range(1, owner._auto_tracking_mode_count+1))
    owner.mode_indices = modes; owner.mode_index = 1; owner.mode_count = len(modes)
    grid = SimpleNamespace(dl=np.full(3, bank[0].du)); owner._tracking_grid = grid
    fields = [[owner._fields_from_solver_mode(s, m) for m in modes] for s in bank]
    e = [[v[0] for v in row] for row in fields]; h = [[v[1] for v in row] for row in fields]
    propagating = np.array([[s.complex_neff[j].real > 0 for j in range(len(modes))] for s in bank])
    align_groups(owner, grid, axis, bank, modes, e, h, propagating)
    frames = [_frame(ee, hh, owner._tracking_impedance) for ee, hh in zip(e, h)]
    rows = []; residuals = []; power = []
    for k, solver in enumerate(bank):
        final = copy(solver)
        for values, names in ((e, ('Eu', 'Ev', 'Ew')), (h, ('Hu', 'Hv', 'Hw'))):
            for c, label in enumerate(names):
                a = getattr(final, label).copy()
                for j in range(len(modes)):
                    a[..., j] = values[k][j][c]
                setattr(final, label, a)
        residuals.extend(final.tracking_field_residuals(np.arange(len(modes))).tolist())
        power.append(float(np.linalg.eigvalsh(_power(owner, grid, e[k], h[k])).min()))
        row = []
        for j, m in enumerate(modes):
            parity = []; errors = []
            for action in (mirror_u, mirror_v):
                te, th = action(solver, e[k][j], h[k][j])
                f = _frame([te], [th], owner._tracking_impedance)[:, 0]
                v = frames[k][:, j]; sign = 1 if np.vdot(v, f).real >= 0 else -1
                parity.append(sign); errors.append(float(np.linalg.norm(f-sign*v)/np.linalg.norm(v)))
            le = np.linalg.norm(e[k][j][2])**2
            lh = owner._tracking_impedance**2*np.linalg.norm(h[k][j][2])**2
            templates = [] if 'dielectric' in case else (
                circular_templates(solver, e[k][j], h[k][j]) if case.startswith('circular')
                else rectangular_templates(e[k][j], h[k][j]))
            row.append(dict(mode=m, parity=parity, parity_error=max(errors), rho=le/(le+lh),
                            templates=templates, neff=solver.complex_neff[j]))
        rows.append(row)
    continuity = [abs(np.sum(a.conj()*b, axis=0))**2 /
                  (np.sum(abs(a)**2, axis=0)*np.sum(abs(b)**2, axis=0))
                  for a, b in zip(frames, frames[1:])]
    records = lambda attr: [dict(modes=r['modes'], blocks=r['blocks'], irreps=r.get('irreps'),
        anchors=r['anchors']) for r in getattr(owner, attr, [])]
    report = dict(modes=modes, candidate_indices=mapping, groups=owner.degenerate,
        detection=owner.symmetry_detection, separation=records('separation_diagnostics'),
        symmetry=records('symmetry_diagnostics'), transport=records('transport_diagnostics'),
        separation_attempts=owner.separation_attempts, transport_attempts=owner.transport_attempts,
        mode_details=rows, max_field_residual=max(residuals), minimum_power_eigenvalue=min(power),
        minimum_channel_continuity=float(np.min(continuity)), continuity=continuity,
        max_parity_error=max(r['parity_error'] for row in rows for r in row),
        all_propagating=bool(propagating.all()),
        material_metadata=getattr(bank[0], 'edge_validation_metadata', None))
    return report, frames


def compare(a, b):
    records = []
    for x, y in zip(a, b):
        x = x/np.linalg.norm(x, axis=0); y = y/np.linalg.norm(y, axis=0)
        score = abs(x.conj().T @ y)**2
        i, j = linear_sum_assignment(-score)
        records.append(dict(minimum_matched_overlap=float(score[i, j].min()),
                            minimum_same_channel_overlap=float(np.diag(score).min()),
                            permutation=j.tolist(), matched_overlaps=score[i, j]))
    return records


def main(output, meshes, cases, seeds):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), root/'gprMax/eigenmode_symmetry.py', root/'gprMax/eigenmode_tracking.py',
             root/'testing/eigenmode_support/profiles/cylindrical_modal_profiles.py',
             root/'testing/eigenmode_support/identification/tracking_classification_edges.py',
             root/'testing/eigenmode_support/subspaces/general_subspace_tracking.py',
             root/'testing/eigenmode_support/symmetry/waveguide_symmetry_validation.py']
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), numpy=np.__version__, frequencies_GHz=[39, 40, 41],
        requested_modes=20, candidates=32, meshes=meshes, seeds=seeds, cases=[],
        geometry=dict(square_width_m=.02, circular_radius_m=.01, circular_window_m=.024,
                      dielectric_window_m=.04, dielectric_core_epsilon=4., cladding_epsilon=1.,
                      dielectric_outer_boundary='PEC', conductor_filling_epsilon=1.),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    def save():
        (output/'manifest.json').write_text(json.dumps(serial(manifest), indent=2)+'\n')
    save()
    with _configured_solver_constants():
        for case in cases:
            for cells in meshes:
                name = f'{case}_{cells}'
                entry = dict(name=name, status='running'); manifest['cases'].append(entry); save()
                print('START', name, flush=True)
                try:
                    bank = [solve(case, f*1e9, cells) for f in (39, 40, 41)]
                    baseline, frames = analyse(deepcopy(bank), case, 20)
                    result = dict(name=name, baseline=baseline, rotations=[])
                    for seed in seeds:
                        rotated = deepcopy(bank); spans = rotate(rotated, seed)
                        report, other = analyse(rotated, case, 20)
                        result['rotations'].append(dict(seed=seed, rotated_spans=spans,
                            comparison=compare(frames, other), result=report))
                    (output/f'{name}.json').write_text(json.dumps(serial(result), indent=2)+'\n')
                    entry.update(status='complete', modes=len(baseline['modes']),
                        max_field_residual=max([baseline['max_field_residual']]+[r['result']['max_field_residual'] for r in result['rotations']]),
                        min_gauge_overlap=min(r['minimum_matched_overlap'] for t in result['rotations'] for r in t['comparison']),
                        min_continuity=baseline['minimum_channel_continuity'],
                        max_parity_error=baseline['max_parity_error'],
                        fallback_groups=[r['modes'] for r in baseline['transport']],
                        separation_rejections=baseline['separation_attempts'])
                    print('COMPLETE', name, serial(entry), flush=True)
                except Exception as error:
                    entry.update(status='failed', error=str(error), traceback=traceback.format_exc())
                    print('FAILED', name, traceback.format_exc(), flush=True)
                save()
    manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--meshes', nargs='+', type=int, default=[24, 40])
    parser.add_argument('--cases', nargs='+', default=['square_PEC', 'circular_PEC', 'square_copper',
        'circular_copper', 'square_dielectric', 'circular_dielectric'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[131, 982])
    args = parser.parse_args()
    main(args.output, args.meshes, args.cases, args.seeds)
