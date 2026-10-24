"""Reproducible investigation of longitudinal/gradient-first mode selection.

This is an offline experiment, not a replacement for certified runtime tracking.
Circular integrals use exact PEC Bessel fields on the unit disk; the square
experiment uses native Yee fields and the runtime forward-difference convention.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from scipy.linalg import eigh
from scipy.special import jn_zeros, jnp_zeros, jv, jvp


def gram(a):
    a = a.reshape(-1, a.shape[-1])
    return a.conj().T @ a


def hermitian(a):
    return (a + a.conj().T) / 2


def circular_forms(family, indices, radial_points=96, angular_points=128):
    """Return M, Ku-Kv, Ku+Kv and transverse moments (common factors omitted).

    indices contains (azimuthal order, radial root number, 'cos'/'sin').
    Integrate only inside the disk: zero-padding Neumann Hz would introduce a
    false boundary jump. TM uses grad(Ez); TE uses z cross grad(Hz).
    """
    nodes, weights = np.polynomial.legendre.leggauss(radial_points)
    r = (nodes[:, None] + 1) / 2
    theta = np.arange(angular_points)[None, :] * (2 * np.pi / angular_points)
    weight = np.broadcast_to(weights[:, None] / 2 * r * (2*np.pi/angular_points),
                             (radial_points, angular_points))
    fields, dus, dvs = [], [], []
    roots = []
    for m, n, parity in indices:
        root = (jnp_zeros if family == 'TE' else jn_zeros)(m, n)[-1]
        angular = np.cos(m*theta) if parity == 'cos' else np.sin(m*theta)
        derivative = -m*np.sin(m*theta) if parity == 'cos' else m*np.cos(m*theta)
        radial = jv(m, root*r)
        dr = root*jvp(m, root*r)*angular
        dt_over_r = radial/r*derivative
        fields.append(radial*angular)
        dus.append(np.cos(theta)*dr - np.sin(theta)*dt_over_r)
        dvs.append(np.sin(theta)*dr + np.cos(theta)*dt_over_r)
        roots.append(float(root))
    f, du, dv = (np.stack(a, axis=-1) for a in (fields, dus, dvs))
    norm = np.sqrt(np.sum(weight[..., None]*abs(f)**2, axis=(0, 1)))
    f, du, dv = (a/norm for a in (f, du, dv))
    weighted = np.sqrt(weight)[..., None]
    m, ku, kv = (gram(weighted*a) for a in (f, du, dv))
    moments = np.stack([np.sum(weight[..., None]*a, axis=(0, 1)) for a in (du, dv)])
    if family == 'TE':
        moments = np.stack([-moments[1], moments[0]])
    return m, ku-kv, ku+kv, moments, roots


def gradient_basis(le, metric, difference, total, *, use_sum=False, tolerance=1e-8):
    """Prototype rho -> signed difference -> optional sum on unresolved ties.

    No claims about physical identity follow from these small eigenproblems.
    Caller must restrict inputs to an admissible eigenspace and validate fields.
    """
    rho, families = eigh(hermitian(le), hermitian(metric))
    cuts = np.flatnonzero(np.diff(rho) > tolerance) + 1
    output = []
    for columns in np.split(np.arange(len(rho)), cuts):
        b = families[:, columns]
        scale = max(np.linalg.norm(b.conj().T @ total @ b, 2), np.finfo(float).tiny)
        eta, v = eigh(hermitian(b.conj().T @ difference @ b))
        b = b @ v
        ties = np.flatnonzero(np.diff(eta) > tolerance*scale) + 1
        for part in np.split(np.arange(len(eta)), ties):
            c = b[:, part]
            if use_sum and len(part) > 1:
                _, v = eigh(hermitian(c.conj().T @ total @ c))
                c = c @ v
            output.append(c)
    return np.column_stack(output)


def circular_cases():
    records = []
    for family in ('TE', 'TM'):
        for m in range(5):
            indices = [(m, 1, 'cos')] + ([(m, 1, 'sin')] if m else [])
            metric, difference, total, moments, roots = circular_forms(family, indices)
            records.append(dict(family=family, angular_order=m, radial_order=1,
                difference_eigenvalues=eigh(difference, metric, eigvals_only=True).tolist(),
                sum_eigenvalues=eigh(total, metric, eigvals_only=True).tolist(),
                kc_squared=roots[0]**2, moment_singular_values=np.linalg.svd(moments, compute_uv=False).tolist(),
                c4v_content=('A2' if family == 'TE' else 'A1') if m == 0 else
                    'E' if m % 2 else ('B1+B2' if m % 4 == 2 else 'A1+A2')))
    radial = []
    for family in ('TE', 'TM'):
        metric, difference, total, _, roots = circular_forms(family, [(0, 1, 'cos'), (0, 2, 'cos')])
        radial.append(dict(family=family, difference_norm=float(np.linalg.norm(difference)),
            sum_eigenvalues=eigh(total, metric, eigvals_only=True).tolist(),
            expected_kc_squared=np.square(roots).tolist(),
            mixing_allowed=False, reason='Distinct cutoffs: diagnostic comparison only.'))
    return dict(angular_pairs=records, radial_orders=radial)


def native_square_cases():
    from gprMax.eigenmode_tracking import _frame, _mix, _residuals
    from gprMax.eigenmode_symmetry import c4_rotation
    from testing.eigenmode_support.subspaces.general_subspace_tracking import fd_solver, stress_rotation
    from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants
    from testing.eigenmode_support.identification.tracking_classification_edges import rectangular_templates
    from tests.test_eigenmode_auto_symmetry import aligned_bank
    import gprMax.config as config

    records = []
    with _configured_solver_constants():
        for cells in (12, 20):
            solver = fd_solver(20e9, (cells, cells), .02/cells, (2, 2, 2), count=12)
            for labels in ((1, 2), (5, 6), (7, 8, 9, 10)):
                stress_rotation(solver, np.array(labels)-1)
                owner, _, baseline_e, baseline_h = aligned_bank([solver], labels)
                raw = [owner._fields_from_solver_mode(solver, label) for label in labels]
                e, h = [v[0] for v in raw], [v[1] for v in raw]
                z0 = config.sim_config.em_consts['z0']
                ez = np.stack([v[2] for v in e], axis=-1)
                hz = z0*np.stack([v[2] for v in h], axis=-1)
                metric = gram(ez)+gram(hz)
                ku, kv = [sum(gram(np.diff(a, axis=axis)/solver.du) for a in (ez, hz)) for axis in (0, 1)]
                c = gradient_basis(gram(ez), metric, ku-kv, ku+kv, use_sum=True)
                ee, hh = _mix(e, c), _mix(h, c)
                scores = [rectangular_templates(a, b)[0] for a, b in zip(ee, hh)]
                baseline = [rectangular_templates(a, b)[0] for a, b in zip(baseline_e[0], baseline_h[0])]
                frame = _frame(ee, hh, z0)
                local_e = tuple(np.stack([v[a] for v in ee], axis=-1) for a in range(3))
                local_h = tuple(np.stack([v[a] for v in hh], axis=-1) for a in range(3))
                re, rh = c4_rotation(solver, local_e, local_h)
                rotated = _frame([[a[..., j] for a in re] for j in range(len(labels))],
                                 [[a[..., j] for a in rh] for j in range(len(labels))], z0)
                representation = np.linalg.lstsq(frame, rotated, rcond=None)[0]
                records.append(dict(cells=cells, frequency_hz=20e9, width_m=.02, epsilon=2,
                    labels=labels, proposed_templates=scores, current_templates=baseline,
                    difference_scores=np.diag(c.conj().T@(ku-kv)@c).real.tolist(),
                    sum_scores=np.diag(c.conj().T@(ku+kv)@c).real.tolist(),
                    rotation_squared_trace=float(np.trace(representation@representation).real),
                    rotation_closure=float(np.linalg.norm(rotated-frame@representation)/np.linalg.norm(rotated)),
                    mixed_residual=np.asarray(_residuals(solver, labels, c)).tolist(),
                    current_irreps=owner.symmetry_diagnostics[0]['irreps']))
    return records


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), root/'gprMax/eigenmode_symmetry.py', root/'gprMax/eigenmode_tracking.py',
             root/'tests/test_eigenmode_auto_symmetry.py',
             root/'testing/eigenmode_support/subspaces/general_subspace_tracking.py',
             root/'testing/eigenmode_support/subspaces/expanded_mode_tracking.py',
             root/'testing/eigenmode_support/identification/tracking_classification_edges.py',
             root/'testing/eigenmode_support/identification/modal_field_templates.py',
             root/'gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py']
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        python=platform.python_version(), numpy=np.__version__,
        sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='PEC modal identification and exact-field quadrature; no scattering comparison or runtime change.')
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    try:
        print('Circular quadrature', flush=True)
        results = dict(circular=circular_cases())
        print('Native square banks at 12 and 20 cells', flush=True)
        results['square'] = native_square_cases()
        (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        manifest['status'] = 'completed'
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print('COMPLETED', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
