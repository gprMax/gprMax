"""Preflight validation for discrete plane-wave TFSF boundaries."""

import math
import warnings

import numpy as np


def _same_number(first, second):
    """Compare two constitutive values, treating numerical noise as equal."""

    if math.isinf(first) or math.isinf(second):
        return first == second
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-15)


def _same_signature(first, second):
    if isinstance(first, tuple) and isinstance(second, tuple):
        return len(first) == len(second) and all(
            _same_signature(a, b) for a, b in zip(first, second)
        )
    if isinstance(first, float) and isinstance(second, float):
        return _same_number(first, second)
    return first == second


def _dispersion_signature(material):
    """Return canonical electric dispersion terms and whether the model is known.

    The first value is None when the material has no effective dispersion.
    The second value is False when a dispersive model cannot be interpreted,
    in which case the caller must not reject the material.
    """

    poles = int(getattr(material, "poles", 0) or 0)
    if poles == 0:
        return None, True

    inclusive_w = getattr(material, "inclusive_w", None)
    if inclusive_w is not None and len(inclusive_w) > 0:
        terms = []
        for w, q in zip(inclusive_w, material.inclusive_q):
            w = complex(w)
            q = complex(q)
            if w == 0 and q == 0:
                continue
            terms.append((w.real, w.imag, q.real, q.imag))
        terms.sort()
        if not terms:
            return None, True
        return ("inclusive", float(material.inclusive_conductivity), tuple(terms)), True

    model = str(getattr(material, "type", ""))
    recognised = [name for name in ("debye", "lorentz", "drude") if name in model]
    if len(recognised) != 1:
        # A mixed or unrecognised pole model cannot be compared without
        # interpreting how its arrays map onto the individual terms.
        return None, False
    model = recognised[0]
    if model == "debye":
        terms = [
            (float(delta), float(tau))
            for delta, tau in zip(material.deltaer, material.tau)
            if float(delta) != 0.0
        ]
        terms.sort()
        return (("debye", tuple(terms)) if terms else None), True
    if model == "lorentz":
        terms = [
            (float(delta), float(frequency), float(damping))
            for delta, frequency, damping in zip(material.deltaer, material.tau, material.alpha)
            if float(delta) != 0.0
        ]
        terms.sort()
        return (("lorentz", tuple(terms)) if terms else None), True
    terms = [
        (float(plasma), float(collision))
        for plasma, collision in zip(material.tau, material.alpha)
        if float(plasma) != 0.0
    ]
    terms.sort()
    return (("drude", tuple(terms)) if terms else None), True


def _component_signature(material, component):
    """Properties read by this Yee component's update, or None if unreliable.

    An E component reads only the electric constitutive parameters and
    electric dispersion; an H component reads only the magnetic parameters.
    The other set never enters that component's update or TFSF correction.
    """

    if getattr(material, "directional_materials", None) is not None:
        return None
    if component < 3:
        dispersion = None
        if int(getattr(material, "poles", 0) or 0):
            dispersion, reliable = _dispersion_signature(material)
            if not reliable:
                return None
        return ("E", float(material.er), float(material.se), dispersion)
    return ("H", float(material.mr), float(material.sm))


def _component_name(component):
    return ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")[component]


def _point(axis, coord, u, v):
    if axis == 0:
        return coord, u, v
    if axis == 1:
        return u, coord, v
    return u, v, coord


def validate_plane_wave_tfsf_background(
    grid, corners, owned_lower, owned_upper, background, projections, skip_axis, *, comm=None
):
    """Reject a homogeneous plane wave whose TFSF boundary leaves its background.

    A vector/angle discrete plane wave assumes that the incident wave
    propagates in a homogeneous background. Objects fully inside the TFSF box
    are allowed, but a correction node whose component-relevant properties
    differ makes the incident-field injection inconsistent with the grid
    update and the model fails silently.

    Component nodes and transverse extents follow the TFSF kernels in
    ``gprMax/cython/plane_wave.pyx``. Nodes whose material ID equals the
    declared background ID are accepted immediately. For other IDs, only the
    properties read by the corrected component are compared: ``er``, ``se``
    and electric dispersion for E, ``mr`` and ``sm`` for H. This accepts
    electromagnetically equivalent materials with different names and ignores
    properties the component never reads, such as permeability at an electric
    node. Where equivalence cannot be established reliably, a warning is
    issued instead of rejecting the model.

    Args:
        grid: FDTD grid with ``ID`` and ``materials``.
        corners: local TFSF box corners ``[x0, y0, z0, x1, y1, z1]``.
        owned_lower: first local index owned by this rank.
        owned_upper: one past the last local index owned by this rank.
        background: declared background material of the plane wave.
        projections: six incident field projections; zero disables every
            correction that uses that incident component.
        skip_axis: invariant axis of a 2D model, or -1 in 3D.
        comm: MPI communicator, or None for a serial run.

    Raises:
        ValueError: if an owned TFSF correction node uses incompatible
            electromagnetic properties.

    Warns:
        UserWarning: if a boundary material's properties cannot be compared
            reliably.
    """

    x0, y0, z0, x1, y1, z1 = (int(value) for value in corners)
    lower = [int(value) for value in owned_lower]
    upper = [int(value) for value in owned_upper]
    background_id = int(background.numID)
    background_signatures = {
        "E": _component_signature(background, 0),
        "H": _component_signature(background, 3),
    }
    signatures = {}

    def signature_of(material, component):
        kind = "E" if component < 3 else "H"
        key = (int(material.numID), kind)
        if key not in signatures:
            signatures[key] = _component_signature(material, component)
        return signatures[key]

    # Active correction nodes, following applyTFSFMagnetic/applyTFSFElectric
    # in gprMax/cython/plane_wave.pyx. Each entry is (face axis, normal
    # index, corrected component, incident component). E and H corrections
    # lie on opposite sides of the staggered interface, so the union of these
    # nodes covers both sides. The incident component multiplies the
    # correction and therefore controls whether the node is active.
    face_specs = []
    if skip_axis != 0:
        face_specs += [
            (0, x0 - 1, 4, 2),  # Hy uses Ez
            (0, x0 - 1, 5, 1),  # Hz uses Ey
            (0, x0, 2, 4),  # Ez uses Hy
            (0, x0, 1, 5),  # Ey uses Hz
            (0, x1, 4, 2),
            (0, x1, 5, 1),
            (0, x1, 2, 4),
            (0, x1, 1, 5),
        ]
    if skip_axis != 1:
        face_specs += [
            (1, y0 - 1, 3, 2),  # Hx uses Ez
            (1, y0 - 1, 5, 0),  # Hz uses Ex
            (1, y0, 2, 3),  # Ez uses Hx
            (1, y0, 0, 5),  # Ex uses Hz
            (1, y1, 3, 2),
            (1, y1, 5, 0),
            (1, y1, 2, 3),
            (1, y1, 0, 5),
        ]
    if skip_axis != 2:
        face_specs += [
            (2, z0 - 1, 4, 0),  # Hy uses Ex
            (2, z0 - 1, 3, 1),  # Hx uses Ey
            (2, z0, 1, 3),  # Ey uses Hx
            (2, z0, 0, 4),  # Ex uses Hy
            (2, z1, 4, 0),
            (2, z1, 3, 1),
            (2, z1, 1, 3),
            (2, z1, 0, 4),
        ]

    def transverse_ranges(axis, component):
        """Transverse extent of the kernel that corrects this component.

        A corrected component and the incident component that drives it are
        an orthogonal tangential pair and share the same transverse Yee
        positions. On z faces those pairs are (Ex, Hy) and (Ey, Hx).
        """

        if axis == 0:
            if component in (4, 2):  # Hy, Ez
                return y0, y1 + 1, z0, z1
            return y0, y1, z0, z1 + 1  # Hz, Ey
        if axis == 1:
            if component in (3, 2):  # Hx, Ez
                return x0, x1 + 1, z0, z1
            return x0, x1, z0, z1 + 1  # Hz, Ex
        if component in (4, 0):  # Hy, Ex
            return x0, x1, y0, y1 + 1
        return x0, x1 + 1, y0, y1  # Hx, Ey

    def ids_on(axis, component, node, u_start, u_stop, v_start, v_stop):
        if axis == 0:
            return grid.ID[component, node, u_start:u_stop, v_start:v_stop]
        if axis == 1:
            return grid.ID[component, u_start:u_stop, node, v_start:v_stop]
        return grid.ID[component, u_start:u_stop, v_start:v_stop, node]

    error = None
    warning_reason = None
    for axis, node, component, source in face_specs:
        if projections[source] == 0:
            continue
        if not (lower[axis] <= node < upper[axis]):
            continue
        u_start, u_stop, v_start, v_stop = transverse_ranges(axis, component)
        if axis == 0:
            u_start = max(u_start, lower[1])
            u_stop = min(u_stop, upper[1])
            v_start = max(v_start, lower[2])
            v_stop = min(v_stop, upper[2])
        elif axis == 1:
            u_start = max(u_start, lower[0])
            u_stop = min(u_stop, upper[0])
            v_start = max(v_start, lower[2])
            v_stop = min(v_stop, upper[2])
        else:
            u_start = max(u_start, lower[0])
            u_stop = min(u_stop, upper[0])
            v_start = max(v_start, lower[1])
            v_stop = min(v_stop, upper[1])
        ids = ids_on(axis, component, node, u_start, u_stop, v_start, v_stop)
        if ids.size == 0:
            continue
        differing = np.unique(ids[ids != background_id])
        for material_id in differing:
            material = grid.materials[int(material_id)]
            u_offset, v_offset = np.argwhere(ids == material_id)[0]
            i, j, k = _point(
                axis,
                node,
                u_start + int(u_offset),
                v_start + int(v_offset),
            )
            offset = getattr(grid, "lower_extent", None)
            if offset is not None:
                # Report global coordinates under MPI domain decomposition.
                i += int(offset[0])
                j += int(offset[1])
                k += int(offset[2])
            location = f"{_component_name(component)} at ({i}, {j}, {k})"
            kind = "E" if component < 3 else "H"
            node_signature = signature_of(material, component)
            if node_signature is None or background_signatures[kind] is None:
                warning_reason = warning_reason or (
                    f"{location} uses material '{material.ID}', whose properties "
                    "cannot be compared reliably"
                )
                continue
            if not _same_signature(node_signature, background_signatures[kind]):
                error = f"{location} uses material '{material.ID}'"
                break
        if error is not None:
            break

    if comm is not None:
        reasons = comm.allgather((error, warning_reason))
        error = next((item[0] for item in reasons if item[0] is not None), None)
        warning_reason = next((item[1] for item in reasons if item[1] is not None), None)

    if error is not None:
        raise ValueError(
            "A vector/angle discrete plane wave assumes a homogeneous "
            f"background ({background.ID}), but {error}. Use "
            "#plane_wave_axial for layered or half-space models."
        )
    if warning_reason is not None:
        warnings.warn(
            "A vector/angle discrete plane wave assumes a homogeneous "
            f"background ({background.ID}), but {warning_reason}. The model "
            "will run, but check that the TFSF boundary is electromagnetically "
            "equivalent to the declared background.",
            UserWarning,
            stacklevel=2,
        )
