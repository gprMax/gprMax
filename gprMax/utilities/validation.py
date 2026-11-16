"""Shared validation for public keyword-based model definitions."""

from difflib import get_close_matches


def validate_keywords(kwargs, allowed, *, context):
    """Reject unknown keys, even when their values are None, False, or zero.

    Never correct a spelling automatically: a suggested option may change
    the physics of the model and must be chosen explicitly by the caller.
    """
    unknown = sorted(set(kwargs) - set(allowed))
    if not unknown:
        return
    details = []
    for name in unknown:
        matches = get_close_matches(name, sorted(allowed), n=1, cutoff=0.7)
        hint = f" (did you mean {matches[0]!r}?)" if matches else ""
        details.append(f"{name!r}{hint}")
    raise TypeError(
        f"{context} got unexpected keyword argument(s): {', '.join(details)}. "
        f"Accepted keywords: {', '.join(sorted(allowed))}."
    )


def _material_signature(material):
    """Return a comparable description of a material's electromagnetic properties."""

    if getattr(material, "directional_materials", None) is not None:
        return ("directional",)

    from gprMax.material_database import material_to_database_entry

    try:
        entry = material_to_database_entry(material)
    except (TypeError, ValueError):
        return ("unsupported", type(material).__name__)

    # These fields do not affect the fields seen by the plane wave.
    for key in ("name", "averagable", "mass_density_kg_per_m3"):
        entry.pop(key, None)
    # Pole order is not physically meaningful for an equivalent medium.
    if "poles" in entry:
        entry["poles"] = sorted(entry["poles"], key=lambda pole: repr(sorted(pole.items())))
    return entry


# Rigid component layers and neighbour offsets from get_rigid_* in
# gprMax/cython/yee_cell_setget_rigid.pyx.
_RIGID_OFFSETS = {
    0: (("E", 0, 0, 0, 0), ("E", 1, 0, -1, 0), ("E", 3, 0, 0, -1), ("E", 2, 0, -1, -1)),
    1: (("E", 4, 0, 0, 0), ("E", 7, -1, 0, 0), ("E", 5, 0, 0, -1), ("E", 6, -1, 0, -1)),
    2: (("E", 8, 0, 0, 0), ("E", 9, -1, 0, 0), ("E", 11, 0, -1, 0), ("E", 10, -1, -1, 0)),
    3: (("H", 0, 0, 0, 0), ("H", 1, -1, 0, 0)),
    4: (("H", 2, 0, 0, 0), ("H", 3, 0, -1, 0)),
    5: (("H", 4, 0, 0, 0), ("H", 5, 0, 0, -1)),
}


def _rigid_at(rigidE, rigidH, component, i, j, k):
    """Whether the given Yee component is rigid at this node."""

    for selector, layer, di, dj, dk in _RIGID_OFFSETS[component]:
        array = rigidE if selector == "E" else rigidH
        x, y, z = i + di, j + dj, k + dk
        if (
            0 <= x < array.shape[1]
            and 0 <= y < array.shape[2]
            and 0 <= z < array.shape[3]
            and array[layer, x, y, z]
        ):
            return True
    return False


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
    are allowed, but a boundary or the adjacent staggered node that uses a
    different material makes the incident-field injection inconsistent with
    the grid update and the model fails silently.

    Component nodes and rigid flags follow the TFSF kernels in
    ``gprMax/cython/plane_wave.pyx`` and ``yee_cell_setget_rigid.pyx``.
    Materials are compared by electromagnetic properties, not by identifier:
    a differently named material with the same properties is accepted.

    Args:
        grid: FDTD grid with ``ID``, ``materials``, ``rigidE`` and ``rigidH``.
        corners: local TFSF box corners ``[x0, y0, z0, x1, y1, z1]``.
        owned_lower: first local index owned by this rank.
        owned_upper: one past the last local index owned by this rank.
        background: declared background material of the plane wave.
        projections: six field projections; zero marks a structurally dead
            component in a 2D model.
        skip_axis: invariant axis of a 2D model, or -1 in 3D.
        comm: MPI communicator, or None for a serial run.

    Raises:
        ValueError: if an owned TFSF boundary or adjacent node is incompatible.
    """

    x0, y0, z0, x1, y1, z1 = (int(value) for value in corners)
    lower = [int(value) for value in owned_lower]
    upper = [int(value) for value in owned_upper]
    signature = _material_signature(background)
    signatures = {}

    def signature_of(material):
        key = int(material.numID)
        if key not in signatures:
            signatures[key] = _material_signature(material)
        return signatures[key]

    def check(component, i, j, k):
        if not (
            0 <= i < grid.ID.shape[1] and 0 <= j < grid.ID.shape[2] and 0 <= k < grid.ID.shape[3]
        ):
            return None
        material = grid.materials[int(grid.ID[component, i, j, k])]
        location = f"{_component_name(component)} at ({i}, {j}, {k})"
        if signature_of(material) != signature:
            return f"{location} uses material '{material.ID}'"
        if _rigid_at(grid.rigidE, grid.rigidH, component, i, j, k):
            return f"{location} is a rigid (PEC/PMC) component"
        return None

    # Active correction nodes and their adjacent staggered nodes, following
    # applyTFSFMagnetic/applyTFSFElectric in gprMax/cython/plane_wave.pyx.
    # Each entry is (face axis, field node, adjacent node, component).
    face_specs = []
    if skip_axis != 0:
        face_specs += [
            (0, x0 - 1, x0, 4),  # Hy
            (0, x0 - 1, x0, 5),  # Hz
            (0, x0, x0 - 1, 2),  # Ez
            (0, x0, x0 - 1, 1),  # Ey
            (0, x1, x1 + 1, 4),
            (0, x1, x1 + 1, 5),
            (0, x1, x1 + 1, 2),
            (0, x1, x1 + 1, 1),
        ]
    if skip_axis != 1:
        face_specs += [
            (1, y0 - 1, y0, 3),  # Hx
            (1, y0 - 1, y0, 5),  # Hz
            (1, y0, y0 - 1, 2),  # Ez
            (1, y0, y0 - 1, 0),  # Ex
            (1, y1, y1 + 1, 3),
            (1, y1, y1 + 1, 5),
            (1, y1, y1 + 1, 2),
            (1, y1, y1 + 1, 0),
        ]
    if skip_axis != 2:
        face_specs += [
            (2, z0 - 1, z0, 4),  # Hy
            (2, z0 - 1, z0, 3),  # Hx
            (2, z0, z0 - 1, 1),  # Ey
            (2, z0, z0 - 1, 0),  # Ex
            (2, z1, z1 + 1, 4),
            (2, z1, z1 + 1, 3),
            (2, z1, z1 + 1, 1),
            (2, z1, z1 + 1, 0),
        ]

    def transverse_ranges(axis, component):
        if axis == 0:
            if component in (4, 2):  # Hy, Ez
                return y0, y1 + 1, z0, z1
            return y0, y1, z0, z1 + 1  # Hz, Ey
        if axis == 1:
            if component in (3, 2):  # Hx, Ez
                return x0, x1 + 1, z0, z1
            return x0, x1, z0, z1 + 1  # Hz, Ex
        if component in (4, 1):  # Hy, Ey
            return x0, x1, y0, y1 + 1
        return x0, x1 + 1, y0, y1  # Hx, Ex

    reason = None
    for axis, node, adjacent, component in face_specs:
        if projections[component] == 0:
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
        for u in range(u_start, u_stop):
            for v in range(v_start, v_stop):
                i, j, k = _point(axis, node, u, v)
                ni, nj, nk = _point(axis, adjacent, u, v)
                reason = reason or check(component, i, j, k)
                reason = reason or check(component, ni, nj, nk)
                if reason is not None:
                    break
            if reason is not None:
                break
        if reason is not None:
            break

    if comm is not None:
        reasons = comm.allgather(reason)
        reason = next((item for item in reasons if item is not None), None)
    if reason is not None:
        raise ValueError(
            "A vector/angle discrete plane wave assumes a homogeneous "
            f"background ({background.ID}), but {reason}. Use "
            "#plane_wave_axial for layered or half-space models."
        )
