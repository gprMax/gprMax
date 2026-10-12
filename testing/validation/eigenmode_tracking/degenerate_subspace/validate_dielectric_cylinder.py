"""Compare tracked open-cylinder modes with full-vector Bessel boundary theory."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from scipy.optimize import linear_sum_assignment

from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from testing.eigenmode_support.identification.tracking_classification_edges import analyse
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
from .dielectric_cylinder import modes, sampled_fields


def solver(frequency, cells, count=24, window=.06):
    d = window/cells
    edge = (np.arange(cells+1)-cells/2)*d
    mid = (np.arange(cells)+.5-cells/2)*d
    def material(x, y):
        return np.where(x[:, None]**2+y[None, :]**2 <= .01**2, 4., 1.).astype(complex)
    eu, ev, ew = material(mid, edge), material(edge, mid), material(edge, edge)
    eu[:, [0, -1]] = ev[[0, -1], :] = np.inf
    ew[:, [0, -1]] = ew[[0, -1], :] = np.inf
    result = FDFD_2D_mode_solver(frequency, d, d, count-1, eu, ev, ew,
                                np.ones_like(ev), np.ones_like(eu), np.ones((cells, cells)))
    result.retain_tracking_operator = result.calculate_diagnostics = True
    result.solve()
    return result


def plot_profiles(numeric, reference, labels, output, cells):
    for page in range((len(labels)+7)//8):
        fig, axes = plt.subplots(4, 4, figsize=(12, 11), constrained_layout=True)
        image = None
        for slot, j in enumerate(range(8*page, min(8*page+8, len(labels)))):
            # Longitudinal E for hybrid/TM modes, H for TE.
            component = 5 if labels[j].startswith('TE') else 2
            a, b = numeric[j][component], reference[j][component]
            display_phase = np.exp(-1j*np.angle(b.flat[np.argmax(abs(b))]))
            a, b = a*display_phase, b*display_phase
            phase = np.exp(-1j*np.angle(np.vdot(b, a)))
            for side, field in enumerate((a*phase, b)):
                peak = max(np.max(abs(field)), 1e-300)
                ax = axes[slot//2, 2*(slot % 2)+side]
                image = ax.imshow((field/peak).real.T, origin='lower', cmap='RdBu_r',
                                  vmin=-1, vmax=1, extent=(-30, 30, -30, 30))
                ax.add_patch(plt.Circle((0, 0), 10, fill=False, color='.3', lw=.7))
                ax.set(title=f'{j+1}: {labels[j]} · {"Tracked" if side == 0 else "Analytical"}',
                       xlabel='x (mm)', ylabel='y (mm)')
        for ax in list(axes.flat)[2*min(8, len(labels)-8*page):]:
            ax.set_visible(False)
        fig.colorbar(image, ax=axes, shrink=.7, label='Phase-aligned normalized longitudinal field')
        fig.suptitle(f'Open dielectric cylinder · 40 GHz · {cells} × {cells}\n'
                     'Analytical orientation fitted within each exact degenerate pair')
        fig.savefig(output/f'modal_profiles_{page+1}.png', dpi=140)
        plt.close(fig)


def check_window(record):
    """Expand air cladding while holding grid spacing and core samples fixed."""
    cells = record['cases'][-1]['cells']
    expanded_cells = 8*int(np.ceil(cells*4/3/8))
    window = .06*expanded_cells/cells
    with _configured_solver_constants():
        enlarged = solver(40e9, expanded_cells, window=window)
    reference = np.sort(np.asarray(record['cases'][-1]['beta_fdfd_per_m'])[1])
    comparison = np.sort(enlarged.complex_neff[:len(reference)].real*2*np.pi*40e9/c)
    change = float(np.max(abs(reference-comparison)/reference))
    assert change < 1e-5, change
    return dict(frequency_Hz=40e9, reference_window_m=.06, reference_cells=cells,
                expanded_window_m=window, expanded_cells=expanded_cells,
                spacing_m=.06/cells, maximum_relative_beta_change=change)


def run(output, meshes=(192, 256), scratch=None):
    output.mkdir(parents=True, exist_ok=True)
    scratch = scratch or output/'raw'
    scratch.mkdir(parents=True, exist_ok=True)
    frequencies = np.array([39., 40., 41.])*1e9
    z0 = np.sqrt(mu_0/epsilon_0)
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  frequency_Hz=frequencies.tolist(), radius_m=.01, window_m=.06,
                  epsilon_core=4., epsilon_cladding=1., theory='Full-vector J/K interface matching; infinite air cladding',
                  numerical_boundary='Finite square PEC truncation; compare window convergence separately',
                  cases=[], source_sha256={})
    root = Path(__file__).resolve().parents[4]
    for path in [Path(__file__), Path(__file__).with_name('dielectric_cylinder.py'),
                 root/'gprMax/eigenmode_tracking.py', root/'gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py']:
        record['source_sha256'][path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    with _configured_solver_constants():
        for cells in meshes:
            print('SOLVE open cylinder', cells, flush=True)
            bank = [solver(f, cells) for f in frequencies]
            theoretical = [modes(f) for f in frequencies]
            count = len(theoretical[1])
            report = analyse(f'open_cylinder_{cells}', bank, count, scratch, 'circular')
            with np.load(scratch/f'open_cylinder_{cells}_fields.npz') as data:
                errors, scores, assignments, beta_theory, beta_numeric = [], [], [], [], []
                for k, frequency in enumerate(frequencies):
                    references = [sampled_fields(mode, orientation, frequency, cells)
                                  for mode, orientation in theoretical[k]]
                    numeric = [[data[f'{"E" if q < 3 else "H"}_{k}_{j}_{q % 3}']*(1 if q < 3 else z0)
                                for q in range(6)] for j in range(count)]
                    a = np.column_stack([np.concatenate([f.ravel() for f in row]) for row in numeric])
                    b = np.column_stack([np.concatenate([f.ravel() for f in row]) for row in references])
                    overlap = abs((a/np.linalg.norm(a, axis=0)).conj().T @ (b/np.linalg.norm(b, axis=0)))**2
                    _, assignment = linear_sum_assignment(-overlap)
                    score, fitted = [], []
                    for j, column in enumerate(assignment):
                        group = [i for i, (mode, _) in enumerate(theoretical[k]) if mode == theoretical[k][column][0]]
                        span = b[:, group]
                        q = np.linalg.qr(span)[0]
                        score.append(float(np.linalg.norm(q.conj().T @ a[:, j])**2/np.linalg.norm(a[:, j])**2))
                        coefficients = np.linalg.lstsq(span, a[:, j], rcond=None)[0]
                        fitted.append([sum(coef*references[g][component] for coef, g in zip(coefficients, group)) for component in range(6)])
                    theory_beta = np.array([theoretical[k][j][0].beta for j in assignment])
                    numeric_beta = bank[k].complex_neff[:count].real*2*np.pi*frequency/c
                    errors.append((abs(numeric_beta-theory_beta)/theory_beta).tolist())
                    scores.append(score); assignments.append(assignment.tolist())
                    beta_theory.append(theory_beta.tolist()); beta_numeric.append(numeric_beta.tolist())
                    if k == 1:
                        labels = [f'{theoretical[k][j][0].family} m={theoretical[k][j][0].m}, r={theoretical[k][j][0].radial}' for j in assignment]
                        if cells == meshes[-1]:
                            plot_profiles(numeric, fitted, labels, output, cells)
                case = dict(cells=cells, spacing_m=.06/cells, labels=labels,
                            analytical_assignment=assignments, beta_analytical_per_m=beta_theory,
                            beta_fdfd_per_m=beta_numeric, relative_beta_error=errors,
                            analytical_subspace_overlap=scores, persistent_groups=report['groups'],
                            maximum_Maxwell_residual=report['max_field_residual'],
                            minimum_forward_power_eigenvalue=report['minimum_power_eigenvalue'],
                            minimum_subspace_overlap=float(np.min(scores)),
                            maximum_relative_beta_error=float(np.max(errors)))
                assert np.all(np.isfinite(beta_numeric)) and np.all(np.asarray(beta_numeric) > 2*np.pi*frequencies[:, None]/c)
                assert case['minimum_subspace_overlap'] > .99, case
                assert case['maximum_relative_beta_error'] < .01, case
                assert case['maximum_Maxwell_residual'] < 1e-8, case
                assert case['minimum_forward_power_eigenvalue'] > 0, case
                record['cases'].append(case)
                (output/'results.json').write_text(json.dumps(record, indent=2)+'\n')
                print('COMPLETE', cells, case['minimum_subspace_overlap'], case['maximum_relative_beta_error'], flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for case in record['cases']:
        x = np.arange(1, len(case['labels'])+1)
        axes[0].plot(x, np.array(case['beta_fdfd_per_m'])[1], 'o', ms=4, label=f"FDFD {case['cells']}²")
        axes[1].semilogy(x, np.maximum(np.array(case['relative_beta_error']).max(axis=0), 1e-16), '.-', label=f"{case['cells']}²")
    axes[0].plot(x, np.array(record['cases'][-1]['beta_analytical_per_m'])[1], 'k_', ms=10, label='Analytical J/K')
    axes[0].set(xlabel='Tracked channel', ylabel='Propagation constant β (rad/m)', title='40 GHz')
    axes[1].set(xlabel='Tracked channel', ylabel='Worst relative β error', title='39 / 40 / 41 GHz')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(output/'propagation_constants.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for case in record['cases']:
        ax.semilogy(x, np.maximum(1-np.array(case['analytical_subspace_overlap']).min(axis=0), 1e-16), '.-', label=f"{case['cells']}²")
    ax.set(xlabel='Tracked channel', ylabel='Worst 1 − analytical subspace overlap', title='Full-vector profile comparison')
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(output/'profile_matching.png', dpi=160); plt.close(fig)
    record['status'] = 'complete'
    record['window_convergence'] = check_window(record)
    record['acceptance'] = dict(minimum_full_vector_subspace_overlap=.99,
                                maximum_relative_beta_error=.01, maximum_Maxwell_residual=1e-8,
                                maximum_relative_window_beta_change=1e-5)
    (output/'results.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def replot_saved_profiles(output, scratch):
    """Replot original fields without relabelling them as a new solver run."""
    record = json.loads((output/'results.json').read_text())
    case = record['cases'][-1]
    cells, frequency = case['cells'], 40e9
    theoretical = modes(frequency)
    references = [sampled_fields(mode, orientation, frequency, cells) for mode, orientation in theoretical]
    numeric, fitted = [], []
    z0 = np.sqrt(mu_0/epsilon_0)
    with np.load(scratch/f'open_cylinder_{cells}_fields.npz') as data:
        for j, column in enumerate(case['analytical_assignment'][1]):
            row = [data[f'{"E" if q < 3 else "H"}_1_{j}_{q % 3}']*(1 if q < 3 else z0) for q in range(6)]
            group = [i for i, (mode, _) in enumerate(theoretical) if mode == theoretical[column][0]]
            span = np.column_stack([np.concatenate([v.ravel() for v in references[g]]) for g in group])
            target = np.concatenate([v.ravel() for v in row])
            coefficients = np.linalg.lstsq(span, target, rcond=None)[0]
            fitted.append([sum(coef*references[g][q] for coef, g in zip(coefficients, group)) for q in range(6)])
            numeric.append(row)
    plot_profiles(numeric, fitted, case['labels'], output, cells)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--cells', nargs='+', type=int, default=[192, 256])
    args = parser.parse_args()
    run(args.output, tuple(args.cells), args.scratch)
