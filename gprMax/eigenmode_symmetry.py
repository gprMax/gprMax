"""Irreducible decomposition from sampled-port or supplied field symmetries.

The numerical kernel has no prescribed degeneracy or block dimension. It
recursively splits a unitary representation with Hermitian members of its
commutant; an irreducible block has a one-dimensional complex commutant.
"""

from copy import copy, deepcopy

import numpy as np


def c4_rotation(solver, electric, magnetic):
    if solver.Nu != solver.Nv or not np.isclose(solver.du, solver.dv, rtol=1e-12, atol=0):
        raise ValueError("C4v symmetry requires a square transverse grid with equal spacing.")
    def rotate(v):
        return (-np.rot90(v[1], axes=(0, 1)), np.rot90(v[0], axes=(0, 1)),
                np.rot90(v[2], axes=(0, 1)))
    return rotate(electric), rotate(magnetic)


def mirror_u(solver, electric, magnetic):
    # E is a polar vector, H an axial vector. The mirror fixes propagation w.
    e, h = electric, magnetic
    return ((-e[0][::-1], e[1][::-1], e[2][::-1]),
            (h[0][::-1], -h[1][::-1], -h[2][::-1]))


def mirror_v(solver, electric, magnetic):
    e, h = electric, magnetic
    return ((e[0][:, ::-1], -e[1][:, ::-1], e[2][:, ::-1]),
            (-h[0][:, ::-1], h[1][:, ::-1], -h[2][:, ::-1]))


def field_actions(value):
    if value == "c4v":
        return {"rotation_90": c4_rotation, "mirror_u": mirror_u}
    if value == "c2v":
        return {"mirror_u": mirror_u, "mirror_v": mirror_v}
    if hasattr(value, 'items') and value and all(callable(action) for action in value.values()):
        return value
    raise ValueError('Expected a detected C4v/C2v symmetry or internal field actions.')


def validate_c4v_materials(solver):
    """Certify the transverse grid, diagonal material tensors, and wall masks."""
    validate_material_symmetry(solver, "c4v")


def validate_surface_symmetry(solver, symmetry):
    """Certify retained Yee samples and signed, clipped Ampere rows.

    Compare the installed frequency-dependent rows, including their complex
    permittivity and circulation per dual area. E is polar and H axial, so
    their reflection signs must both enter the row transformation.
    """
    shapes = ((solver.shape_eu, solver.shape_ev, solver.shape_ew),
              (solver.shape_hu, solver.shape_hv, solver.shape_hw))
    tags, keys = [], []
    for family in shapes:
        arrays, addresses = [], []
        for axis, shape in enumerate(family):
            start = len(addresses)
            addresses.extend((axis, index) for index in np.ndindex(shape))
            arrays.append(np.arange(start + 1, len(addresses) + 1).reshape(shape))
        tags.append(tuple(arrays))
        keys.append(addresses)
    retained = (solver.surface_electric_retained, solver.surface_magnetic_retained)
    rows = {(r.electric_axis, r.electric_index): r for r in solver.surface_boundary_rows}
    for action in field_actions(symmetry).values():
        transformed = action(solver, *tags)
        maps = []
        for family, values in enumerate(transformed):
            mapping = {}
            for axis, array in enumerate(values):
                for index in np.ndindex(array.shape):
                    tag = int(array[index]); source = keys[family][abs(tag)-1]
                    mapping[source] = ((axis, index), 1 if tag > 0 else -1)
                    if retained[family][source[0]][source[1]] != retained[family][axis][index]:
                        raise ValueError('SIBC retained samples do not preserve the requested symmetry.')
            maps.append(mapping)
        for key, row in rows.items():
            target, electric_sign = maps[0][key]
            other = rows.get(target)
            if other is None:
                raise ValueError('SIBC Ampere-row topology does not preserve the requested symmetry.')
            if not np.isclose(row.relative_permittivity, other.relative_permittivity, rtol=1e-10, atol=1e-12):
                raise ValueError('SIBC admittance does not preserve the requested symmetry.')
            expected = {}
            for term in row.magnetic_terms:
                address, magnetic_sign = maps[1][(term.axis, term.index)]
                expected[address] = expected.get(address, 0.) + electric_sign * magnetic_sign * term.line_weight / row.retained_dual_area
            actual = {(t.axis, t.index): t.line_weight / other.retained_dual_area for t in other.magnetic_terms}
            if expected.keys() != actual.keys() or any(
                not np.isclose(value, actual[address], rtol=1e-10, atol=1e-12)
                for address, value in expected.items()
            ):
                raise ValueError('SIBC signed circulation rows do not preserve the requested symmetry.')


def validate_material_symmetry(solver, symmetry):
    """Check native Yee samples, including complex materials and wall masks.

    C2v uses the two grid-axis mirrors. C4v additionally requires a quarter
    turn. These are symmetries about the center of the finite port window.
    """
    if symmetry not in ("c4v", "c2v"):
        raise ValueError("Material symmetry must be C4v or C2v.")
    if symmetry == "c4v" and (solver.Nu != solver.Nv or not np.isclose(solver.du, solver.dv, rtol=1e-12, atol=0)):
        raise ValueError('C4v symmetry requires a square transverse grid with equal spacing.')
    for names in (('eps_r_uu', 'eps_r_vv', 'eps_r_ww'), ('mu_r_uu', 'mu_r_vv', 'mu_r_ww'),
                  ('pec_u_mask', 'pec_v_mask', 'pec_w_mask'), ('pmc_u_mask', 'pmc_v_mask', 'pmc_w_mask')):
        values = [np.asarray(getattr(solver, name)) for name in names]
        if any(a.ndim != 2 or not np.all(np.isfinite(a)) for a in values):
            raise ValueError('Symmetry detection requires finite two-dimensional material and mask arrays.')
        same = lambda a, b: a.shape == b.shape and np.allclose(a, b, rtol=1e-10, atol=1e-12)
        preserves = all(same(a, a[::-1]) and same(a, a[:, ::-1]) for a in values)
        if symmetry == "c4v":
            preserves &= all(same(a, b) for a, b in zip(values, (np.rot90(values[1]), np.rot90(values[0]), np.rot90(values[2]))))
        if not preserves:
            raise ValueError(f'The sampled materials or boundary masks do not preserve {symmetry.upper()} symmetry.')
    if getattr(solver, 'surface_boundary', None) is not None:
        validate_surface_symmetry(solver, symmetry)


def detect_symmetry(solvers):
    """Find the strongest supported native symmetry common to every anchor."""
    anchors = []
    for solver in solvers:
        supported, reasons = [], {}
        for name in ("c4v", "c2v"):
            try:
                validate_material_symmetry(solver, name)
            except (ValueError, AttributeError) as error:
                reasons[name] = str(error)
            else:
                supported.append(name)
        anchors.append(dict(supported=supported, reasons=reasons))
    common = [name for name in ("c4v", "c2v") if anchors and all(name in a['supported'] for a in anchors)]
    return dict(detected=common[0] if common else "none", candidates=common, anchors=anchors, groups=[])


def c2v_irrep_names(representations, columns, tolerance=1e-8):
    """C2v with C2 along w; B1 is u-like, B2 is v-like."""
    u, v = representations
    eye = np.eye(len(u))
    if max(np.linalg.norm(u @ u-eye), np.linalg.norm(v @ v-eye),
           np.linalg.norm(u @ v-v @ u)) > 10*tolerance:
        raise ValueError('Modal actions do not satisfy the C2v group relations.')
    names = []
    for indices in columns:
        if len(indices) != 1:
            raise ValueError('C2v irreducible blocks must be one-dimensional.')
        j = indices[0]
        signs = tuple(1 if a[j,j].real > 0 else -1 for a in (u,v))
        names.append({(1,1):'A1', (-1,-1):'A2', (-1,1):'B1', (1,-1):'B2'}[signs])
    return tuple(names)


def c4v_irrep_names(representations, columns, tolerance=1e-8):
    """Identify irreps and verify the C4v relations on the actual modal span."""
    rotation, mirror = representations
    eye = np.eye(len(rotation))
    if max(np.linalg.norm(np.linalg.matrix_power(rotation, 4)-eye),
           np.linalg.norm(mirror @ mirror-eye),
           np.linalg.norm(mirror @ rotation @ mirror-rotation.conj().T)) > 10*tolerance:
        raise ValueError('Modal actions do not satisfy the C4v group relations.')
    names = []
    for indices in columns:
        r, s = (a[np.ix_(indices, indices)] for a in representations)
        if len(indices) == 1:
            names.append({(1,1):'A1', (1,-1):'A2', (-1,1):'B1', (-1,-1):'B2'}[
                (1 if r[0,0].real > 0 else -1, 1 if s[0,0].real > 0 else -1)])
        elif len(indices) == 2 and np.linalg.norm(r @ r+np.eye(2)) < 10*tolerance:
            names.append('E')
        else:
            raise ValueError('A C4v irreducible block must be one-dimensional or an E doublet.')
    return tuple(names)


def commutant(generators, tolerance=1e-8):
    """Orthonormal vec(X) basis for X D_g = D_g X, in column-major order."""
    n = generators[0].shape[0]
    eye = np.eye(n)
    system = np.vstack([np.kron(eye, d) - np.kron(d.T, eye) for d in generators])
    _, singular, vh = np.linalg.svd(system, full_matrices=False)
    cutoff = tolerance * max(1., singular[0])
    return vh[singular <= cutoff].conj().T


def irreducible_blocks(generators, *, probe=None, tolerance=1e-8):
    """Return orthonormal block bases and certify invariance/irreducibility.

    ``probe`` can encode neighboring subspaces to orient equivalent copies.
    Otherwise the decomposition chooses a deterministic numerical gauge.
    Equivalent copies admit multiple valid decompositions.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Symmetry tolerance must be finite and between zero and one.")
    generators = tuple(np.asarray(d, dtype=complex) for d in generators)
    if not generators or generators[0].ndim != 2:
        raise ValueError("Supply at least one square symmetry representation matrix.")
    n = generators[0].shape[0]
    if n == 0 or any(d.shape != (n, n) or not np.all(np.isfinite(d)) for d in generators):
        raise ValueError("Symmetry matrices must be finite, square, and equally sized.")
    if any(np.linalg.norm(d.conj().T @ d - np.eye(n), 2) > tolerance for d in generators):
        raise ValueError("Symmetry actions must be unitary in the supplied basis.")
    if probe is None:
        probe = np.diag(np.arange(1., n + 1))
    probe = np.asarray(probe, dtype=complex)
    if probe.shape != (n, n) or not np.all(np.isfinite(probe)):
        raise ValueError("Symmetry decomposition probe has the wrong shape or nonfinite values.")

    def split(basis):
        reduced = tuple(basis.conj().T @ d @ basis for d in generators)
        kernel = commutant(reduced, tolerance)
        if kernel.shape[1] == 1:
            return [basis]
        if not kernel.shape[1]:
            raise ValueError("No stable commutant for supplied symmetry actions.")
        size = basis.shape[1]
        seed = basis.conj().T @ probe @ basis
        projected = (kernel @ (kernel.conj().T @ seed.ravel(order="F"))).reshape((size, size), order="F")
        choices = [(projected + projected.conj().T) / 2]
        for column in kernel.T:
            a = column.reshape((size, size), order="F")
            choices.extend(((a + a.conj().T) / 2, (a - a.conj().T) / (2j)))
        for hermitian in choices:
            values, vectors = np.linalg.eigh(hermitian)
            cuts = np.flatnonzero(np.diff(values) > tolerance * max(1., np.max(abs(values)))) + 1
            if len(cuts):
                return [block for part in np.split(np.arange(size), cuts)
                        for block in split(basis @ vectors[:, part])]
        raise ValueError("Reducible symmetry span has no numerically resolved split.")

    blocks = split(np.eye(n, dtype=complex))
    transform = np.column_stack(blocks)
    leakage = max(np.linalg.norm(d @ b - b @ (b.conj().T @ d @ b), 2)
                  for d in generators for b in blocks)
    if leakage > 10 * tolerance or np.linalg.norm(transform.conj().T @ transform - np.eye(n), 2) > tolerance:
        raise ValueError("Irreducible decomposition failed its invariance or rank check.")
    return tuple(blocks), float(leakage)


def orient_irreducible_block(block, frame, moments, moment_scale, mirrors, *, rotation=None,
                             previous=None, previous_parity=None, overlap_min=0.8, tolerance=1e-8):
    """Orient a power-orthonormal singlet/doublet without changing its span.

    Doublets first use the polar factor of their transverse electric moments.
    Zero or ill-conditioned moments use mirror-u eigenvectors ordered (-,+),
    consistent with u- then v-directed vector moments. Mirror parity refers
    to the complete vector field, including the axial transformation of H.
    Across anchors only individual phases are transported.
    """
    block = np.asarray(block, dtype=complex)
    size = block.shape[1]
    if size not in (1, 2):
        raise ValueError('Native mirror orientation requires a singlet or doublet.')
    moment = moments @ block
    singular = np.linalg.svd(moment, compute_uv=False)
    relative = float(singular[-1] / max(moment_scale, np.finfo(float).tiny))
    method = 'singlet'
    oriented = block.copy()
    if size == 2:
        method = 'mirror_parity'
        if relative > 1e-12 and singular[-1] > singular[0] / 1e8:
            left, _, right = np.linalg.svd(moment)
            transform = right.conj().T @ left.conj().T
            aligned = moment @ transform
            off_axis = aligned - np.diag(np.diag(aligned))
            candidate = block @ transform
            parity_error = max(np.linalg.norm(mirror @ candidate - candidate * signs) / np.sqrt(size)
                               for mirror, signs in zip(mirrors, ([-1, 1], [1, -1])))
            if np.linalg.norm(off_axis) <= tolerance * np.linalg.norm(aligned) and parity_error <= tolerance:
                oriented = candidate
                method = 'electric_moment'
        if method == 'mirror_parity':
            mirror = block.conj().T @ mirrors[0] @ block
            values, vectors = np.linalg.eigh((mirror + mirror.conj().T) / 2)
            if np.max(abs(values - [-1., 1.])) > tolerance:
                raise ValueError('A native doublet must contain both mirror-u parities.')
            oriented = block @ vectors

    # A deterministic reference phase for zero-moment modes; all E/H
    # components receive this same phase. Native array ordering breaks ties.
    if previous is None and method != 'electric_moment':
        fields = frame @ oriented
        for j in range(size):
            amplitude = abs(fields[:, j])
            peak = float(np.max(amplitude))
            pivot = np.flatnonzero(amplitude >= peak * (1 - 1e-12))[0]
            oriented[:, j] *= np.exp(-1j * np.angle(fields[pivot, j]))
        if size == 2 and rotation is not None:
            # Use a quarter turn to give the two partners a common phase.
            connection = np.vdot(oriented[:, 1], rotation @ oriented[:, 0])
            if abs(connection) < 1 - tolerance:
                raise ValueError('The oriented partners do not form a quarter-turn doublet.')
            oriented[:, 1] *= connection / abs(connection)

    parity, errors = [], []
    for mirror in mirrors:
        signs = np.where(np.diag(oriented.conj().T @ mirror @ oriented).real >= 0, 1, -1)
        parity.append(signs)
        errors.append(np.linalg.norm(mirror @ oriented - oriented * signs) / np.sqrt(size))
    parity = np.asarray(parity).T
    error = float(max(errors))
    if error > tolerance:
        raise ValueError('The oriented fields fail the vector mirror-parity check.')
    overlaps = np.ones(size)
    if previous is not None:
        if not np.array_equal(parity, previous_parity):
            raise ValueError('Mirror parity changes along an oriented channel.')
        fields = frame @ oriented
        for j in range(size):
            connection = np.vdot(fields[:, j], previous[:, j])
            overlaps[j] = abs(connection)**2 / (np.linalg.norm(fields[:, j])**2 * np.linalg.norm(previous[:, j])**2)
            if overlaps[j] < overlap_min:
                raise ValueError('Oriented symmetry channels require closer anchors or a stable subgroup.')
            oriented[:, j] *= connection / abs(connection)
    return oriented, dict(method=method, parity=parity, parity_residual=error,
                          moment_relative_min=relative, channel_overlaps=overlaps)


def longitudinal_blocks(generators, electric, magnetic, field_scale, *, probe=None, tolerance=1e-8):
    """Resolve repeated symmetry copies by longitudinal electric content.

    Inputs are Ez and Zref*Hz samples in a power-orthonormal parent basis.
    No waveguide geometry, TE/TM label, or spatial mode index is prescribed.
    Each resolved longitudinal family is whitened separately: whitening the
    complete result would mix families whose cross-power is nonzero.
    """
    from scipy.linalg import eigh

    n = generators[0].shape[0]
    electric = np.asarray(electric).reshape(-1, n)
    magnetic = np.asarray(magnetic).reshape(-1, n)
    le = electric.conj().T @ electric
    metric = le + magnetic.conj().T @ magnetic
    metric = (metric + metric.conj().T) / 2
    eigenvalues = np.linalg.eigvalsh(metric)
    record = dict(applied=False, reason='', scores=np.empty(0))

    def fallback(reason):
        record['reason'] = reason
        blocks, leakage = irreducible_blocks(generators, probe=probe, tolerance=tolerance)
        return blocks, leakage, record

    if eigenvalues[-1] <= 1e-12 * field_scale:
        return fallback('insignificant_longitudinal_fields')
    if eigenvalues[0] <= eigenvalues[-1] / 1e8:
        return fallback('singular_or_ill_conditioned_longitudinal_metric')
    scores, coefficients = eigh((le + le.conj().T) / 2, metric)
    record['scores'] = scores
    cuts = np.flatnonzero(np.diff(scores) > tolerance) + 1
    if not len(cuts):
        return fallback('unresolved_longitudinal_content')
    blocks, leakage = [], 0.
    for columns in np.split(np.arange(n), cuts):
        # QR restores unit power inside this family while keeping its span.
        basis = np.linalg.qr(coefficients[:, columns])[0]
        reduced = tuple(basis.conj().T @ d @ basis for d in generators)
        error = max(np.linalg.norm(d @ basis - basis @ a, 2)
                    for d, a in zip(generators, reduced))
        if error > tolerance:
            return fallback('longitudinal_split_not_symmetry_invariant')
        subblocks, suberror = irreducible_blocks(
            reduced, probe=None if probe is None else basis.conj().T @ probe @ basis,
            tolerance=tolerance)
        blocks.extend(basis @ b for b in subblocks)
        leakage = max(leakage, float(error), suberror)
    transform = np.column_stack(blocks)
    power = transform.conj().T @ transform
    values = np.linalg.eigvalsh(power)
    if values[0] <= values[-1] / 1e8:
        return fallback('ill_conditioned_separated_power')
    record.update(applied=True, reason='resolved_longitudinal_families')
    return tuple(blocks), leakage, record


def axis_sensitive_blocks(blocks, electric, magnetic, mirrors, spacing, field_scale, *, tolerance=1e-8,
                          family_tolerance=None, candidate_groups=None):
    """Orient equal-parity singlets in a common longitudinal family.

    The generalized directional-gradient eigenproblem uses native Ez and
    Zref*Hz samples. It may combine distinct rotation irreps; those original
    irreps are recorded separately from the resulting source basis.
    """
    from scipy.linalg import eigh

    family_tolerance = tolerance if family_tolerance is None else max(tolerance, family_tolerance)
    n = blocks[0].shape[0]
    fields = (np.asarray(electric), np.asarray(magnetic))
    gram = lambda a: a.reshape(-1, n).conj().T @ a.reshape(-1, n)
    le = gram(fields[0])
    metric = le + gram(fields[1])
    ku, kv = [sum(gram(np.diff(a, axis=axis) / spacing[axis]) for a in fields)
              for axis in (0, 1)]
    candidates = []
    for j, b in enumerate(blocks):
        if b.shape[1] != 1:
            continue
        energy = float((b.conj().T @ metric @ b)[0, 0].real)
        if energy <= 1e-12 * field_scale:
            continue
        signs = tuple(1 if (b.conj().T @ m @ b)[0, 0].real >= 0 else -1 for m in mirrors)
        rho = float((b.conj().T @ le @ b)[0, 0].real / energy)
        candidates.append((j, signs, rho))
    result = list(blocks)
    records, used = [], set()
    for j, signs, rho in candidates:
        if j in used:
            continue
        indices = [i for i, parity, score in candidates if i not in used and parity == signs
                   and abs(score-rho) <= family_tolerance
                   and (candidate_groups is None or candidate_groups[i] == candidate_groups[j])]
        used.update(indices)
        if len(indices) < 2:
            continue
        basis = np.column_stack([blocks[i] for i in indices])
        m = basis.conj().T @ metric @ basis
        m = (m+m.conj().T)/2
        record = dict(block_indices=indices, applied=False, reason='', scores=np.empty(0),
                      family_tolerance=family_tolerance)
        records.append(record)
        values = np.linalg.eigvalsh(m)
        if values[0] <= values[-1]/1e8:
            record['reason'] = 'ill_conditioned_longitudinal_metric'
            continue
        if np.linalg.norm(basis.conj().T @ le @ basis-rho*m, 2) > family_tolerance*values[-1]:
            record['reason'] = 'distinct_longitudinal_families'
            continue
        difference = basis.conj().T @ (ku-kv) @ basis
        scores, vectors = eigh((difference+difference.conj().T)/2, m)
        gradient = basis.conj().T @ (ku+kv) @ basis
        scale = max(float(np.linalg.norm(gradient, 2)/values[0]), np.finfo(float).tiny)
        record['scores'] = scores
        if np.any(np.diff(scores) <= tolerance*scale):
            record['reason'] = 'unresolved_directional_scores'
            continue
        oriented = basis @ vectors
        oriented /= np.linalg.norm(oriented, axis=0)
        # Preserve both complete-vector mirror parities.
        if max(np.linalg.norm(mirror @ oriented-sign*oriented, 2)
               for mirror, sign in zip(mirrors, signs)) > tolerance:
            record['reason'] = 'mirror_parity_not_preserved'
            continue
        for column, i in enumerate(indices):
            result[i] = oriented[:, column:column+1]
        record.update(applied=True, reason='resolved_axis_variation')
    return tuple(result), records


def _spectral_clusters(values, tolerance):
    """Complete-link clusters; a chain of small gaps cannot merge endpoints."""
    order = np.argsort(values)
    groups = []
    for index in order:
        if not groups or values[index] - values[groups[-1][0]] > tolerance:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return [np.asarray(group) for group in groups]


def longitudinal_gradient_blocks(electric, magnetic, spacing, field_scale, *,
                                 tolerance=1e-8, family_tolerance=None):
    """Separate rho, signed Ku-Kv, then Ku+Kv ties without spatial symmetry.

    Input columns have unit power. Each output block is whitened separately;
    cross-block power is retained. Unresolved eigenspaces remain blocks, not
    arbitrary eigenvectors. Candidate columns join equal |eta| and total
    gradient scores within one longitudinal family for later symmetry checks.
    """
    fields = tuple(np.asarray(a) for a in (electric, magnetic))
    n = fields[0].shape[-1]
    gram = lambda a: a.reshape(-1, n).conj().T @ a.reshape(-1, n)
    le = gram(fields[0])
    metric = le + gram(fields[1])
    ku, kv = [sum(gram(np.diff(a, axis=axis)/spacing[axis]) for a in fields) for axis in (0, 1)]
    return _separate_longitudinal_forms(le, metric, ku-kv, ku+kv, field_scale,
                                       tolerance=tolerance, family_tolerance=family_tolerance)


def _separate_longitudinal_forms(le, metric, difference, total, field_scale, *,
                                 tolerance=1e-8, family_tolerance=None):
    """Generalized spectral hierarchy, shared by native and exact-field checks."""
    from scipy.linalg import eigh

    family_tolerance = tolerance if family_tolerance is None else family_tolerance
    hermitian = lambda a: (a + a.conj().T)/2
    le, metric, difference, total = map(hermitian, (le, metric, difference, total))
    n = len(metric)
    longitudinal = dict(applied=False, reason='', scores=np.empty(0))
    record = dict(longitudinal=longitudinal, gradients=[], candidates=[np.arange(n)])
    values = np.linalg.eigvalsh(metric)
    if values[-1] <= 1e-12*field_scale:
        longitudinal['reason'] = 'insignificant_longitudinal_fields'
        return (np.eye(n),), record
    if values[0] <= values[-1]/1e8:
        longitudinal['reason'] = 'singular_or_ill_conditioned_longitudinal_metric'
        return (np.eye(n),), record
    rho, vectors = eigh(hermitian(le), metric)
    families = _spectral_clusters(rho, family_tolerance)
    longitudinal.update(applied=len(families) > 1, scores=rho,
                        reason='resolved_longitudinal_families' if len(families) > 1
                        else 'unresolved_longitudinal_content')
    blocks, descriptors = [], []
    for family, indices in enumerate(families):
        basis = np.linalg.qr(vectors[:, indices])[0] if len(families) > 1 else np.eye(n)
        m = hermitian(basis.conj().T @ metric @ basis)
        d = hermitian(basis.conj().T @ difference @ basis)
        s = hermitian(basis.conj().T @ total @ basis)
        scale = max(float(np.max(abs(eigh(s, m, eigvals_only=True)))), np.finfo(float).tiny)
        eta, directions = eigh(d, m)
        directional = _spectral_clusters(eta, tolerance*scale)
        for columns in directional:
            b = basis @ directions[:, columns] if len(directional) > 1 else basis
            b = np.linalg.qr(b)[0]
            restricted_m = hermitian(b.conj().T @ metric @ b)
            sigma, radial = eigh(hermitian(b.conj().T @ total @ b), restricted_m)
            radial_groups = _spectral_clusters(sigma, tolerance*scale)
            for part in radial_groups:
                c = b @ radial[:, part] if len(radial_groups) > 1 else b
                c = np.linalg.qr(c)[0]
                index = len(blocks)
                blocks.append(c)
                descriptors.append((family, float(np.mean(abs(eta[columns]))),
                                    float(np.mean(sigma[part])), scale))
                record['gradients'].append(dict(block_index=index, family=family,
                    difference_scores=eta[columns], sum_scores=sigma[part],
                    directional_split=len(directional) > 1, sum_split=len(radial_groups) > 1))
    # Group whole resolved blocks, using complete-link agreement in both scores.
    candidates = []
    for j, (family, eta, sigma, scale) in enumerate(descriptors):
        for group in candidates:
            if all(family == descriptors[i][0]
                   and abs(eta-descriptors[i][1]) <= tolerance*max(scale, descriptors[i][3])
                   and abs(sigma-descriptors[i][2]) <= tolerance*max(scale, descriptors[i][3]) for i in group):
                group.append(j)
                break
        else:
            candidates.append([j])
    cuts = np.cumsum([0]+[b.shape[1] for b in blocks])
    record['candidates'] = [np.concatenate([np.arange(cuts[i], cuts[i+1]) for i in group])
                            for group in candidates]
    return tuple(blocks), record


def _family_tolerance(solver, labels, tolerance):
    if getattr(solver, 'surface_boundary', None) is None:
        return tolerance
    eigenvalues = np.asarray(solver.eigenvalues)[np.asarray(labels)-1]
    loss = np.max(abs(eigenvalues.imag)/np.maximum(abs(eigenvalues), 1e-300))
    return max(tolerance, min(1e-3, float(loss)))


def _validate_measurement_basis(owner, grid, electric, magnetic):
    """Require the complete power-wave bank to pass monitor precision checks."""
    import gprMax.config as config
    from gprMax.eigenmode_ports import _coefficient_precision_limits, _solve_conditioned_gram

    dtype = getattr(config.sim_config, 'dtypes', {}).get('complex', np.complex128)
    epsilon, limit = _coefficient_precision_limits(dtype)
    gram = np.asarray([[owner._modal_cross_power(e, h, grid) for e in electric]
                       for h in magnetic])
    # The monitor normalizes propagating profiles to unit real power. Below-
    # cutoff references have separate validity handling and are not power waves.
    powers = np.real(np.diag(gram))
    active = np.flatnonzero(np.isfinite(powers) & (powers > 1e-12))
    if active.size == 0:
        raise ValueError('Symmetry-adapted fields have no valid measurement power.')
    gram = gram[np.ix_(active, active)] / np.sqrt(powers[active, None] * powers[None, active])
    gram = gram.astype(dtype)
    for matrix in (gram, gram.conj().T, (gram + gram.conj().T) / 2):
        _, valid, _ = _solve_conditioned_gram(
            matrix, np.zeros(active.size), component_epsilon=epsilon,
            condition_limit=limit, power_coordinates=np.ones(active.size, dtype=bool))
        if not np.all(valid):
            raise ValueError('Symmetry-adapted measurement basis exceeds the coefficient precision budget.')


def align_symmetry_subspaces(owner, grid, frequencies, solvers, modes, electric, magnetic):
    """Separate physical spans first, then independently certify spatial symmetry.

    Each trial uses private field-bank rows and diagnostics. A rejected trial
    cannot partially publish its transforms, including at earlier anchors.
    """
    # This pass deliberately precedes even material symmetry detection. Each
    # parent is transactional and uses the same full-field acceptance checks
    # as the symmetry pass below.
    owner.separation_diagnostics = []
    owner.separation_attempts = []
    owner.transport_diagnostics = []
    owner.transport_attempts = []
    if owner.invariant_axis is None:
        for parent in owner.degenerate_diagnostics:
            trial = copy(owner)
            trial.degenerate_diagnostics = deepcopy([parent])
            trial.separation_diagnostics = []
            trial_e, trial_h = [list(row) for row in electric], [list(row) for row in magnetic]
            try:
                _align_symmetry_subspaces(trial, grid, frequencies, solvers, modes,
                                         trial_e, trial_h, symmetry=None)
            except (ValueError, np.linalg.LinAlgError) as error:
                owner.separation_attempts.append(dict(modes=tuple(parent['modes']), reason=str(error)))
                continue
            for k in range(len(electric)):
                electric[k][:], magnetic[k][:] = trial_e[k], trial_h[k]
            parent.update(trial.degenerate_diagnostics[0])
            owner.separation_diagnostics.extend(trial.separation_diagnostics)
    detection = detect_symmetry(solvers)
    owner.symmetry_detection = detection
    owner.symmetry_diagnostics = []
    if owner.invariant_axis is not None:
        detection.update(detected="none", candidates=[], reason="A two-dimensional transverse port mesh is required.")
    for parent in owner.degenerate_diagnostics:
        fallback = ('longitudinal_gradient' if parent['orientation'] == 'longitudinal_gradient'
                    else 'electric_moment' if parent['orientation'] == 'automatic_transverse_axes' else 'power_whitening')
        record = dict(modes=tuple(parent['modes']), applied=fallback, attempts=[])
        detection['groups'].append(record)
        for name in detection['candidates']:
            trial = copy(owner)
            trial.degenerate_diagnostics = deepcopy([parent])
            trial.symmetry_diagnostics = []
            trial_e = [list(row) for row in electric]
            trial_h = [list(row) for row in magnetic]
            try:
                _align_symmetry_subspaces(trial, grid, frequencies, solvers, modes, trial_e, trial_h, symmetry=name)
            except (ValueError, np.linalg.LinAlgError) as error:
                record['attempts'].append(dict(symmetry=name, reason=str(error)))
                continue
            for k in range(len(electric)):
                electric[k][:], magnetic[k][:] = trial_e[k], trial_h[k]
            parent.update(trial.degenerate_diagnostics[0])
            owner.symmetry_diagnostics.extend(trial.symmetry_diagnostics)
            record['applied'] = name
            break

    # Symmetry-oriented channels are already individually fixed. For the
    # remaining parents, transport only unresolved blocks; singleton phases
    # may continue, but resolved families must never be remixed together.
    certified = {tuple(r['modes']) for r in owner.symmetry_diagnostics}
    separated = {tuple(r['modes']): r for r in owner.separation_diagnostics}
    for parent in owner.degenerate_diagnostics:
        labels = tuple(parent['modes'])
        if labels in certified:
            continue
        blocks = separated[labels]['blocks'] if labels in separated else (labels,)
        if all(len(block) == 1 for block in blocks):
            continue
        trial = copy(owner)
        trial.degenerate_diagnostics = deepcopy([parent])
        trial.transport_diagnostics = []
        trial._transport_blocks = blocks
        trial_e, trial_h = [list(row) for row in electric], [list(row) for row in magnetic]
        try:
            _align_symmetry_subspaces(trial, grid, frequencies, solvers, modes,
                                     trial_e, trial_h, symmetry='procrustes')
        except (ValueError, np.linalg.LinAlgError) as error:
            # Both earlier stages already validated their provisional gauges.
            # A rejected final trial cannot replace that usable field bank.
            owner.transport_attempts.append(dict(modes=labels, reason=str(error)))
            continue
        for k in range(len(electric)):
            electric[k][:], magnetic[k][:] = trial_e[k], trial_h[k]
        parent.update(trial.degenerate_diagnostics[0])
        owner.transport_diagnostics.extend(trial.transport_diagnostics)


def _align_symmetry_subspaces(owner, grid, frequencies, solvers, modes, electric, magnetic, *, symmetry):
    """Validate one complete parent bank for separation or symmetry refinement.

    With symmetry=None, select longitudinal/gradient blocks independently of
    spatial actions. Both stages share power, continuity, Maxwell, and monitor
    precision checks and publish only after every active anchor passes.
    """
    from scipy.optimize import linear_sum_assignment
    from gprMax.eigenmode_tracking import _frame, _mix, _power, _residuals, _orthonormal, _moments

    actions = {} if symmetry in (None, 'procrustes') else field_actions(symmetry)
    if owner.invariant_axis is not None:
        raise ValueError("Symmetry decomposition requires a two-dimensional transverse port mesh.")
    if symmetry in ('c4v', 'c2v'):
        for solver in solvers:
            validate_material_symmetry(solver, symmetry)
    owner.symmetry_diagnostics = []
    tolerance = 1e-8
    impedance = float(owner._tracking_impedance)
    axes = (*owner.transverse_axes, owner.normal_axis)
    for parent in owner.degenerate_diagnostics:
        labels = tuple(parent['modes'])
        positions = [modes.index(label) for label in labels]
        used = [k for k, row in enumerate(parent['anchors']) if row['active']]
        reference = parent['reference_anchor']
        results = {}
        paths = ([reference], list(range(reference + 1, used[-1] + 1)),
                 list(range(reference - 1, used[0] - 1, -1)))
        for path in paths:
            previous = None if path == [reference] else reference
            for k in path:
                e = [electric[k][j] for j in positions]
                h = [magnetic[k][j] for j in positions]
                power = _power(owner, grid, e, h)
                values, vectors = np.linalg.eigh(power)
                if not np.all(np.isfinite(values)) or values[0] <= values[-1] / 1e8:
                    raise ValueError("Symmetry decomposition requires positive, well-conditioned group power.")
                whitening = (vectors / np.sqrt(values)) @ vectors.conj().T
                e, h = _mix(e, whitening), _mix(h, whitening)
                frame = _frame(e, h, impedance)
                # Callbacks see local (u,v,w) components on their native Yee
                # grids, with a final mode-column dimension.
                local_e = tuple(np.stack([v[a] for v in e], axis=-1) for a in axes)
                local_h = tuple(np.stack([v[a] for v in h], axis=-1) for a in axes)
                generators = []
                closure = 0.
                for name, action in actions.items():
                    te, th = action(solvers[k], tuple(a.copy() for a in local_e),
                                    tuple(a.copy() for a in local_h))
                    if len(te) != 3 or len(th) != 3 or any(
                        np.shape(a) != np.shape(b) or not np.all(np.isfinite(a))
                        for a, b in zip((*te, *th), (*local_e, *local_h))
                    ):
                        raise ValueError(f"Symmetry action {name!r} returned invalid field arrays.")
                    ge, gh = [], []
                    for j in range(len(labels)):
                        ee, hh = [None]*3, [None]*3
                        for a, axis in enumerate(axes):
                            ee[axis], hh[axis] = te[a][...,j], th[a][...,j]
                        ge.append(ee); gh.append(hh)
                    transformed = _frame(ge, gh, impedance)
                    representation = np.linalg.lstsq(frame, transformed, rcond=None)[0]
                    error = np.linalg.norm(transformed-frame @ representation) / np.linalg.norm(transformed)
                    closure = max(closure, float(error))
                    if closure > tolerance:
                        raise ValueError(f"Symmetry action {name!r} does not preserve modal group {labels}.")
                    generators.append(representation)
                probe = None
                if previous is not None:
                    old = results[previous]
                    coordinates = np.linalg.lstsq(frame, old['frame'], rcond=None)[0]
                    probe = np.zeros((len(labels), len(labels)), dtype=complex)
                    for j, columns in enumerate(old['columns']):
                        q = np.linalg.qr(coordinates[:,columns])[0]
                        probe += (j+1) * (q @ q.conj().T)
                longitudinal, separation = None, None
                candidate_groups = None
                selected = next((r for r in getattr(owner, 'separation_diagnostics', ())
                                 if tuple(r['modes']) == labels), None)
                if symmetry == 'procrustes':
                    blocks = tuple(np.linalg.solve(whitening, np.eye(len(labels))[:,
                        [labels.index(label) for label in block]]) for block in owner._transport_blocks)
                    leakage = 0.
                elif symmetry is None:
                    blocks, separation = longitudinal_gradient_blocks(
                        local_e[2], impedance*local_h[2], (solvers[k].du, solvers[k].dv),
                        np.linalg.norm(frame, 'fro')**2, tolerance=tolerance,
                        family_tolerance=_family_tolerance(solvers[k], labels, tolerance))
                    longitudinal = separation['longitudinal']
                    leakage = 0.
                elif symmetry in ('c4v', 'c2v') and selected is not None:
                    # The incoming fields are the published separation basis.
                    # Undo only this pass's whitening to locate each candidate.
                    blocks, candidate_groups, leakage = [], [], 0.
                    longitudinal = selected['anchors'][k]['longitudinal']
                    for group_index, columns in enumerate(selected['anchors'][k]['separation']['candidates']):
                        b = np.linalg.qr(np.linalg.solve(whitening, np.eye(len(labels))[:, columns]))[0]
                        reduced = tuple(b.conj().T @ d @ b for d in generators)
                        error = max(np.linalg.norm(d @ b-b @ a, 2) for d, a in zip(generators, reduced))
                        if error > tolerance:
                            raise ValueError('Separated candidate is not invariant under the detected symmetry.')
                        subblocks, suberror = irreducible_blocks(reduced,
                            probe=None if probe is None else b.conj().T @ probe @ b, tolerance=tolerance)
                        blocks.extend(b @ c for c in subblocks)
                        candidate_groups.extend([group_index]*len(subblocks))
                        leakage = max(leakage, float(error), suberror)
                    blocks = tuple(blocks)
                elif symmetry in ('c4v', 'c2v'):
                    blocks, leakage, longitudinal = longitudinal_blocks(
                        generators, local_e[2], impedance * local_h[2],
                        np.linalg.norm(frame, 'fro')**2, probe=probe, tolerance=tolerance)
                else:
                    blocks, leakage = irreducible_blocks(generators, probe=probe, tolerance=tolerance)
                axis_records, source_irreps, irreducible_basis = [], (), None
                if symmetry in ('c4v', 'c2v'):
                    rotation = generators[0] if symmetry == 'c4v' else None
                    mirrors = (generators[1], rotation @ generators[1] @ rotation.conj().T) if rotation is not None else generators
                    ib = np.column_stack(blocks)
                    cuts = np.cumsum([0]+[b.shape[1] for b in blocks])
                    ic = [np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])]
                    naming = c4v_irrep_names if symmetry == 'c4v' else c2v_irrep_names
                    names = naming([np.linalg.solve(ib,d @ ib) for d in generators], ic)
                    irreducible_basis = dict(transform=ib, columns=ic, irreps=names)
                    family_tolerance = _family_tolerance(solvers[k], labels, tolerance)
                    blocks, axis_records = axis_sensitive_blocks(
                        blocks, local_e[2], impedance*local_h[2], mirrors,
                        (solvers[k].du,solvers[k].dv), np.linalg.norm(frame,'fro')**2,
                        tolerance=tolerance, family_tolerance=family_tolerance, candidate_groups=candidate_groups)
                    source_irreps = list(names)
                    for record in axis_records:
                        if record['applied']:
                            components = '+'.join(sorted(set(names[i] for i in record['block_indices'])))
                            for i in record['block_indices']:
                                source_irreps[i] = components
                if previous is not None:
                    old = results[previous]
                    scores = np.full((len(old['columns']), len(blocks)), -1e6)
                    for i, columns in enumerate(old['columns']):
                        for j, b in enumerate(blocks):
                            if symmetry == 'procrustes' and i != j:
                                continue  # Final transport cannot reassign established blocks.
                            if len(columns) == b.shape[1]:
                                a = _orthonormal(old['frame'][:,columns])
                                c = _orthonormal(frame @ b)
                                scores[i,j] = np.min(np.linalg.svd(a.conj().T @ c, compute_uv=False))**2
                    rows, order = linear_sum_assignment(-scores)
                    if (len(old['columns']) != len(blocks) or len(rows) != len(blocks)
                            or np.min(scores[rows,order]) < owner.tracking_config.tracking_overlap):
                        raise ValueError("Irreducible symmetry subspaces require closer anchors or corrected symmetry information.")
                    if separation is not None:
                        cuts = np.cumsum([0]+[b.shape[1] for b in blocks])
                        permutation = np.concatenate([np.arange(cuts[j], cuts[j+1]) for j in order[np.argsort(rows)]])
                        inverse_columns = np.argsort(permutation)
                        separation['candidates'] = [inverse_columns[c] for c in separation['candidates']]
                        inverse_blocks = np.argsort(order[np.argsort(rows)])
                        for record in separation['gradients']:
                            record['source_block_index'] = int(inverse_blocks[record['block_index']])
                    blocks = tuple(blocks[j] for j in order[np.argsort(rows)])
                    if source_irreps:
                        source_irreps = [source_irreps[j] for j in order[np.argsort(rows)]]
                        inverse = np.argsort(order[np.argsort(rows)])
                        for record in axis_records:
                            record['source_block_indices'] = [int(inverse[j]) for j in record['block_indices']]
                    if symmetry not in ('c4v', 'c2v'):
                        continued = []
                        for b, columns in zip(blocks, old['columns']):
                            left, _, right = np.linalg.svd((frame @ b).conj().T @ old['frame'][:,columns])
                            continued.append(b @ (left @ right))
                        blocks = tuple(continued)
                if symmetry is None and previous is None:
                    # Choose a deterministic reference inside each block. This
                    # is a phase for singlets; unresolved blocks still travel
                    # as complete spans, without claiming individual identity.
                    from gprMax.eigenmode_tracking import _canonical_subspace_transform
                    blocks = tuple(b @ _canonical_subspace_transform(frame @ b) for b in blocks)
                    # Canonicalization changes scale; restore unit block power.
                    normalized = []
                    for b in blocks:
                        v, u = np.linalg.eigh(b.conj().T @ b)
                        normalized.append(b @ ((u/np.sqrt(v)) @ u.conj().T))
                    blocks = tuple(normalized)
                orientation = []
                if symmetry in ('c4v', 'c2v'):
                    rotation = generators[0] if symmetry == 'c4v' else None
                    mirrors = (generators[1], rotation @ generators[1] @ rotation.conj().T) if rotation is not None else generators
                    moments = _moments(owner, grid, e)
                    oriented_blocks = []
                    for j, b in enumerate(blocks):
                        block_e = _mix(e, b)
                        scales = [sum(np.sum(abs(field[a])) for a in owner.transverse_axes)
                                  * np.prod(grid.dl[list(owner.transverse_axes)]) for field in block_e]
                        old = results[previous] if previous is not None else None
                        oriented, diagnostic = orient_irreducible_block(
                            b, frame, moments, np.linalg.norm(scales), mirrors, rotation=rotation,
                            previous=old['frame'][:, old['columns'][j]] if old is not None else None,
                            previous_parity=old['orientation'][j]['parity'] if old is not None else None,
                            overlap_min=owner.tracking_config.tracking_overlap, tolerance=tolerance)
                        oriented_blocks.append(oriented)
                        orientation.append(diagnostic)
                    blocks = tuple(oriented_blocks)
                transform = np.column_stack(blocks)
                columns, start = [], 0
                for b in blocks:
                    columns.append(np.arange(start, start+b.shape[1]))
                    start += b.shape[1]
                final_e, final_h = _mix(e, transform), _mix(h, transform)
                original = [owner._fields_from_solver_mode(solvers[k], label) for label in labels]
                original_frame = _frame([v[0] for v in original], [v[1] for v in original], impedance)
                final_frame = _frame(final_e, final_h, impedance)
                total = np.linalg.lstsq(original_frame, final_frame, rcond=None)[0]
                if irreducible_basis is not None:
                    irreducible_basis['transform'] = np.linalg.lstsq(
                        original_frame, frame @ irreducible_basis['transform'], rcond=None)[0]
                residual = _residuals(solvers[k], labels, total)
                if not np.all(np.isfinite(residual)) or np.max(residual) > owner.tracking_config.residual_tolerance:
                    raise ValueError("Symmetry-adapted fields fail the mixed eigen-residual check.")
                field_residual = np.nan
                if hasattr(solvers[k], 'mode_tracking_field_operators'):
                    check = copy(solvers[k])
                    indices = np.asarray(labels)-1
                    for name in ('Eu','Ev','Ew','Hu','Hv','Hw'):
                        values = getattr(check, name).copy()
                        values[...,indices] = values[...,indices] @ total
                        setattr(check, name, values)
                    field_residual = float(np.max(check.tracking_field_residuals(indices)))
                    if not np.isfinite(field_residual) or field_residual > owner.tracking_config.residual_tolerance:
                        raise ValueError("Symmetry-adapted fields fail the reconstructed Maxwell check.")
                power = _power(owner, grid, final_e, final_h)
                power_values = np.linalg.eigvalsh(power)
                if (not np.all(np.isfinite(power_values)) or power_values[0] <= power_values[-1] / 1e8 or
                    any(np.linalg.norm(power[np.ix_(c, c)]-np.eye(len(c)), 2) > tolerance
                        for c in columns)):
                    raise ValueError("Symmetry-adapted fields fail positive group power or block whitening.")
                bank_e, bank_h = list(electric[k]), list(magnetic[k])
                for j, position in enumerate(positions):
                    bank_e[position], bank_h[position] = final_e[j], final_h[j]
                _validate_measurement_basis(owner, grid, bank_e, bank_h)
                results[k] = dict(frame=final_frame, columns=columns, electric=final_e, magnetic=final_h,
                                  transform=total, residual=residual, power=power, closure=closure,
                                  leakage=leakage, field_residual=field_residual,
                                  orientation=orientation, longitudinal=longitudinal, separation=separation,
                                  axis_sensitive=axis_records, irreducible_basis=irreducible_basis,
                                  source_irreps=tuple(source_irreps),
                                  representations=[np.linalg.solve(transform, d @ transform) for d in generators])
                previous = k
        # Publish only after the complete parent bank passes symmetry checks.
        blocks = tuple(tuple(labels[j] for j in c) for c in results[reference]['columns'])
        irreps = ()
        if symmetry in ('c4v', 'c2v'):
            irreps = results[reference]['source_irreps']
            for row in results.values():
                if row['source_irreps'] != irreps:
                    raise ValueError('The symmetry representation changes along a tracked subspace.')
        for k, row in results.items():
            for j, position in enumerate(positions):
                electric[k][position], magnetic[k][position] = row['electric'][j], row['magnetic'][j]
            parent['anchors'][k].update(transform=row['transform'], residual=row['residual'], power_gram=row['power'])
            if parent['physical']:
                parent['anchors'][k]['electric_moments'] = _moments(owner, grid, row['electric'])
        parent['orientation'] = ('longitudinal_gradient' if symmetry is None else
                                 'unresolved_procrustes' if symmetry == 'procrustes' else 'symmetry_irreducible')
        saved = {k: {name: row[name] for name in ('transform', 'residual', 'power', 'closure', 'leakage', 'field_residual', 'representations', 'orientation', 'longitudinal', 'separation', 'axis_sensitive', 'irreducible_basis')}
                 for k, row in results.items()}
        diagnostics = (owner.separation_diagnostics if symmetry is None else
                       owner.transport_diagnostics if symmetry == 'procrustes' else owner.symmetry_diagnostics)
        symmetry_name = symmetry if symmetry is None or isinstance(symmetry, str) else 'custom'
        diagnostics.append(dict(modes=labels, blocks=blocks, irreps=irreps, symmetry=symmetry_name, generators=tuple(actions),
                                              reference_anchor=reference, anchors=saved))
