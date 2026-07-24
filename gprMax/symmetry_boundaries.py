# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""Per-iteration field updates for PMC (magnetic-wall/ghost-node) symmetry
boundaries. PEC symmetry boundaries need no per-iteration code at all - their
tangential E is already correctly forced to pec's zero coefficients at build
time (FDTDGrid._force_pec_tangential_e). Only PMC needs a runtime update,
since its on-wall tangential E is otherwise never touched by the ordinary
bulk update loop.

Covers both the face-interior region of each PMC face (trimmed by one cell
on every side, so it never touches a domain edge) and the 12 domain edges
(where two faces meet). Two material regimes, each with its own dispatch:

- Standard (non-dispersive) materials: `update_symmetry_boundaries_electric_normal()`,
  a single per-iteration call using only `updatecoeffsE`.
- Dispersive materials (Debye - real poles, or Lorentz/Drude - complex
  poles): `update_symmetry_boundaries_electric_dispersive()`/
  `update_symmetry_boundaries_electric_dispersive_b()`, a two-phase call
  mirroring the bulk dispersive kernel's own A/B split (see
  gprMax/cython/fields_updates_dispersive_template.jinja and
  gprMax/updates/cpu_updates.py's update_electric_a()/update_electric_b()) -
  phase A computes the ADE polarisation-current term (phi) from the T-array
  and updates T from the pre-iteration E, phase B corrects T using the final
  (post-PML, post-source) E. Which Cython module's functions actually run
  (gprMax.cython.symmetry_boundaries_dispersive for Debye,
  gprMax.cython.symmetry_boundaries_dispersive_complex for Lorentz/Drude) is
  chosen per-iteration from config.get_model_config().materials["drudelorentz"]
  - the same flag config.py itself uses to pick updatecoeffsdispersive's
  dtype (real vs complex), so the Cython functions called always match the
  actual dtype of grid.updatecoeffsdispersive/Tx/Ty/Tz.

Ghost-node derivation: tangential H is
odd under the PMC mirror, so the "ghost" H node just outside the domain
equals minus the real interior H node it mirrors. Substituting into the
standard curl term used by the bulk kernel collapses the missing
outside-neighbour difference into double one real H value - no ghost array
needed, just a modified update formula. For a face at the "0" end of its
axis (x0/y0/z0), the doubled term uses the wall's own H index with the same
sign the bulk kernel's own formula already has; for a "max" end
(xmax/ymax/zmax), it uses the interior-adjacent H index (one cell in, not
the wall index) with the opposite sign. This part is identical between the
non-dispersive and dispersive kernels.

At an edge (where two faces meet), the self term (Ca*E, or Ca*E - Ce*phi for
the dispersive variant) applies once if EITHER bordering face is PMC; each
face then separately, additively, contributes its own doubled ghost term
only if THAT SPECIFIC face is PMC - no owner/increment distinction needed.
Both flags are resolved once here, at build time (they never change once
#symmetry_boundary commands are processed), and edges where NEITHER
bordering face is PMC are dropped entirely - not called every iteration with
both flags False, which would still pay for a wasted loop-and-branch pass.
This keeps the feature's cost at zero for any model that doesn't use
#symmetry_boundary at all, and proportional only to the faces/edges actually
declared PMC otherwise. For the dispersive edge kernels, the phi/T
bookkeeping itself runs unconditionally (not gated by the a_pmc/b_pmc
flags) - see gprMax/cython/symmetry_boundaries_dispersive.pyx's module
docstring for why no explicit PEC-transparency branch is needed there.

The actual per-position math is implemented in Cython
(gprMax/cython/symmetry_boundaries.pyx for non-dispersive,
gprMax/cython/symmetry_boundaries_dispersive.pyx for dispersive), one
function per domain face and one per domain edge, mirroring the per-face
convention already used for PML (pml_updates_electric_HORIPML.pyx) - this
module is the build-time resolution (which faces/edges are active) and the
thin per-iteration dispatch layer, matching how cpu_updates.py calls into
fields_updates_normal.pyx.
"""

from gprMax import config
from gprMax.cython.symmetry_boundaries import (
    update_symmetry_boundary_electric_Ex_Y0_Z0,
    update_symmetry_boundary_electric_Ex_Y0_ZMax,
    update_symmetry_boundary_electric_Ex_YMax_Z0,
    update_symmetry_boundary_electric_Ex_YMax_ZMax,
    update_symmetry_boundary_electric_Ey_X0_Z0,
    update_symmetry_boundary_electric_Ey_X0_ZMax,
    update_symmetry_boundary_electric_Ey_XMax_Z0,
    update_symmetry_boundary_electric_Ey_XMax_ZMax,
    update_symmetry_boundary_electric_Ez_X0_Y0,
    update_symmetry_boundary_electric_Ez_X0_YMax,
    update_symmetry_boundary_electric_Ez_XMax_Y0,
    update_symmetry_boundary_electric_Ez_XMax_YMax,
    update_symmetry_boundary_electric_x0,
    update_symmetry_boundary_electric_xmax,
    update_symmetry_boundary_electric_y0,
    update_symmetry_boundary_electric_ymax,
    update_symmetry_boundary_electric_z0,
    update_symmetry_boundary_electric_zmax,
)
from gprMax.cython.symmetry_boundaries_dispersive import (
    update_symmetry_boundary_electric_dispersive_b_Ex_Y0_Z0,
    update_symmetry_boundary_electric_dispersive_b_Ex_Y0_ZMax,
    update_symmetry_boundary_electric_dispersive_b_Ex_YMax_Z0,
    update_symmetry_boundary_electric_dispersive_b_Ex_YMax_ZMax,
    update_symmetry_boundary_electric_dispersive_b_Ey_X0_Z0,
    update_symmetry_boundary_electric_dispersive_b_Ey_X0_ZMax,
    update_symmetry_boundary_electric_dispersive_b_Ey_XMax_Z0,
    update_symmetry_boundary_electric_dispersive_b_Ey_XMax_ZMax,
    update_symmetry_boundary_electric_dispersive_b_Ez_X0_Y0,
    update_symmetry_boundary_electric_dispersive_b_Ez_X0_YMax,
    update_symmetry_boundary_electric_dispersive_b_Ez_XMax_Y0,
    update_symmetry_boundary_electric_dispersive_b_Ez_XMax_YMax,
    update_symmetry_boundary_electric_dispersive_b_x0,
    update_symmetry_boundary_electric_dispersive_b_xmax,
    update_symmetry_boundary_electric_dispersive_b_y0,
    update_symmetry_boundary_electric_dispersive_b_ymax,
    update_symmetry_boundary_electric_dispersive_b_z0,
    update_symmetry_boundary_electric_dispersive_b_zmax,
    update_symmetry_boundary_electric_dispersive_Ex_Y0_Z0,
    update_symmetry_boundary_electric_dispersive_Ex_Y0_ZMax,
    update_symmetry_boundary_electric_dispersive_Ex_YMax_Z0,
    update_symmetry_boundary_electric_dispersive_Ex_YMax_ZMax,
    update_symmetry_boundary_electric_dispersive_Ey_X0_Z0,
    update_symmetry_boundary_electric_dispersive_Ey_X0_ZMax,
    update_symmetry_boundary_electric_dispersive_Ey_XMax_Z0,
    update_symmetry_boundary_electric_dispersive_Ey_XMax_ZMax,
    update_symmetry_boundary_electric_dispersive_Ez_X0_Y0,
    update_symmetry_boundary_electric_dispersive_Ez_X0_YMax,
    update_symmetry_boundary_electric_dispersive_Ez_XMax_Y0,
    update_symmetry_boundary_electric_dispersive_Ez_XMax_YMax,
    update_symmetry_boundary_electric_dispersive_x0,
    update_symmetry_boundary_electric_dispersive_xmax,
    update_symmetry_boundary_electric_dispersive_y0,
    update_symmetry_boundary_electric_dispersive_ymax,
    update_symmetry_boundary_electric_dispersive_z0,
    update_symmetry_boundary_electric_dispersive_zmax,
)
from gprMax.cython.symmetry_boundaries_dispersive_complex import (
    update_symmetry_boundary_electric_dispersive_b_Ex_Y0_Z0 as _c_update_symmetry_boundary_electric_dispersive_b_Ex_Y0_Z0,
    update_symmetry_boundary_electric_dispersive_b_Ex_Y0_ZMax as _c_update_symmetry_boundary_electric_dispersive_b_Ex_Y0_ZMax,
    update_symmetry_boundary_electric_dispersive_b_Ex_YMax_Z0 as _c_update_symmetry_boundary_electric_dispersive_b_Ex_YMax_Z0,
    update_symmetry_boundary_electric_dispersive_b_Ex_YMax_ZMax as _c_update_symmetry_boundary_electric_dispersive_b_Ex_YMax_ZMax,
    update_symmetry_boundary_electric_dispersive_b_Ey_X0_Z0 as _c_update_symmetry_boundary_electric_dispersive_b_Ey_X0_Z0,
    update_symmetry_boundary_electric_dispersive_b_Ey_X0_ZMax as _c_update_symmetry_boundary_electric_dispersive_b_Ey_X0_ZMax,
    update_symmetry_boundary_electric_dispersive_b_Ey_XMax_Z0 as _c_update_symmetry_boundary_electric_dispersive_b_Ey_XMax_Z0,
    update_symmetry_boundary_electric_dispersive_b_Ey_XMax_ZMax as _c_update_symmetry_boundary_electric_dispersive_b_Ey_XMax_ZMax,
    update_symmetry_boundary_electric_dispersive_b_Ez_X0_Y0 as _c_update_symmetry_boundary_electric_dispersive_b_Ez_X0_Y0,
    update_symmetry_boundary_electric_dispersive_b_Ez_X0_YMax as _c_update_symmetry_boundary_electric_dispersive_b_Ez_X0_YMax,
    update_symmetry_boundary_electric_dispersive_b_Ez_XMax_Y0 as _c_update_symmetry_boundary_electric_dispersive_b_Ez_XMax_Y0,
    update_symmetry_boundary_electric_dispersive_b_Ez_XMax_YMax as _c_update_symmetry_boundary_electric_dispersive_b_Ez_XMax_YMax,
    update_symmetry_boundary_electric_dispersive_b_x0 as _c_update_symmetry_boundary_electric_dispersive_b_x0,
    update_symmetry_boundary_electric_dispersive_b_xmax as _c_update_symmetry_boundary_electric_dispersive_b_xmax,
    update_symmetry_boundary_electric_dispersive_b_y0 as _c_update_symmetry_boundary_electric_dispersive_b_y0,
    update_symmetry_boundary_electric_dispersive_b_ymax as _c_update_symmetry_boundary_electric_dispersive_b_ymax,
    update_symmetry_boundary_electric_dispersive_b_z0 as _c_update_symmetry_boundary_electric_dispersive_b_z0,
    update_symmetry_boundary_electric_dispersive_b_zmax as _c_update_symmetry_boundary_electric_dispersive_b_zmax,
    update_symmetry_boundary_electric_dispersive_Ex_Y0_Z0 as _c_update_symmetry_boundary_electric_dispersive_Ex_Y0_Z0,
    update_symmetry_boundary_electric_dispersive_Ex_Y0_ZMax as _c_update_symmetry_boundary_electric_dispersive_Ex_Y0_ZMax,
    update_symmetry_boundary_electric_dispersive_Ex_YMax_Z0 as _c_update_symmetry_boundary_electric_dispersive_Ex_YMax_Z0,
    update_symmetry_boundary_electric_dispersive_Ex_YMax_ZMax as _c_update_symmetry_boundary_electric_dispersive_Ex_YMax_ZMax,
    update_symmetry_boundary_electric_dispersive_Ey_X0_Z0 as _c_update_symmetry_boundary_electric_dispersive_Ey_X0_Z0,
    update_symmetry_boundary_electric_dispersive_Ey_X0_ZMax as _c_update_symmetry_boundary_electric_dispersive_Ey_X0_ZMax,
    update_symmetry_boundary_electric_dispersive_Ey_XMax_Z0 as _c_update_symmetry_boundary_electric_dispersive_Ey_XMax_Z0,
    update_symmetry_boundary_electric_dispersive_Ey_XMax_ZMax as _c_update_symmetry_boundary_electric_dispersive_Ey_XMax_ZMax,
    update_symmetry_boundary_electric_dispersive_Ez_X0_Y0 as _c_update_symmetry_boundary_electric_dispersive_Ez_X0_Y0,
    update_symmetry_boundary_electric_dispersive_Ez_X0_YMax as _c_update_symmetry_boundary_electric_dispersive_Ez_X0_YMax,
    update_symmetry_boundary_electric_dispersive_Ez_XMax_Y0 as _c_update_symmetry_boundary_electric_dispersive_Ez_XMax_Y0,
    update_symmetry_boundary_electric_dispersive_Ez_XMax_YMax as _c_update_symmetry_boundary_electric_dispersive_Ez_XMax_YMax,
    update_symmetry_boundary_electric_dispersive_x0 as _c_update_symmetry_boundary_electric_dispersive_x0,
    update_symmetry_boundary_electric_dispersive_xmax as _c_update_symmetry_boundary_electric_dispersive_xmax,
    update_symmetry_boundary_electric_dispersive_y0 as _c_update_symmetry_boundary_electric_dispersive_y0,
    update_symmetry_boundary_electric_dispersive_ymax as _c_update_symmetry_boundary_electric_dispersive_ymax,
    update_symmetry_boundary_electric_dispersive_z0 as _c_update_symmetry_boundary_electric_dispersive_z0,
    update_symmetry_boundary_electric_dispersive_zmax as _c_update_symmetry_boundary_electric_dispersive_zmax,
)

_FACE_UPDATE_FUNCS = {
    "x0": update_symmetry_boundary_electric_x0,
    "xmax": update_symmetry_boundary_electric_xmax,
    "y0": update_symmetry_boundary_electric_y0,
    "ymax": update_symmetry_boundary_electric_ymax,
    "z0": update_symmetry_boundary_electric_z0,
    "zmax": update_symmetry_boundary_electric_zmax,
}

_FACE_UPDATE_FUNCS_DISPERSIVE = {
    "x0": update_symmetry_boundary_electric_dispersive_x0,
    "xmax": update_symmetry_boundary_electric_dispersive_xmax,
    "y0": update_symmetry_boundary_electric_dispersive_y0,
    "ymax": update_symmetry_boundary_electric_dispersive_ymax,
    "z0": update_symmetry_boundary_electric_dispersive_z0,
    "zmax": update_symmetry_boundary_electric_dispersive_zmax,
}

_FACE_UPDATE_FUNCS_DISPERSIVE_B = {
    "x0": update_symmetry_boundary_electric_dispersive_b_x0,
    "xmax": update_symmetry_boundary_electric_dispersive_b_xmax,
    "y0": update_symmetry_boundary_electric_dispersive_b_y0,
    "ymax": update_symmetry_boundary_electric_dispersive_b_ymax,
    "z0": update_symmetry_boundary_electric_dispersive_b_z0,
    "zmax": update_symmetry_boundary_electric_dispersive_b_zmax,
}

# Complex-pole (Lorentz/Drude) counterparts of the two dicts above -
# structurally identical, functions from symmetry_boundaries_dispersive_complex
# instead of symmetry_boundaries_dispersive.
_FACE_UPDATE_FUNCS_DISPERSIVE_COMPLEX = {
    "x0": _c_update_symmetry_boundary_electric_dispersive_x0,
    "xmax": _c_update_symmetry_boundary_electric_dispersive_xmax,
    "y0": _c_update_symmetry_boundary_electric_dispersive_y0,
    "ymax": _c_update_symmetry_boundary_electric_dispersive_ymax,
    "z0": _c_update_symmetry_boundary_electric_dispersive_z0,
    "zmax": _c_update_symmetry_boundary_electric_dispersive_zmax,
}

_FACE_UPDATE_FUNCS_DISPERSIVE_B_COMPLEX = {
    "x0": _c_update_symmetry_boundary_electric_dispersive_b_x0,
    "xmax": _c_update_symmetry_boundary_electric_dispersive_b_xmax,
    "y0": _c_update_symmetry_boundary_electric_dispersive_b_y0,
    "ymax": _c_update_symmetry_boundary_electric_dispersive_b_ymax,
    "z0": _c_update_symmetry_boundary_electric_dispersive_b_z0,
    "zmax": _c_update_symmetry_boundary_electric_dispersive_b_zmax,
}

_ALL_FACES = ("x0", "y0", "z0", "xmax", "ymax", "zmax")

# The 12 domain edges: (face_a, face_b, cython function, E component
# attribute, H1/H2 attributes in the function's own positional order).
_EDGE_TABLE = (
    ("x0", "y0", update_symmetry_boundary_electric_Ez_X0_Y0, "Ez", "Hx", "Hy"),
    ("x0", "ymax", update_symmetry_boundary_electric_Ez_X0_YMax, "Ez", "Hx", "Hy"),
    ("xmax", "y0", update_symmetry_boundary_electric_Ez_XMax_Y0, "Ez", "Hx", "Hy"),
    ("xmax", "ymax", update_symmetry_boundary_electric_Ez_XMax_YMax, "Ez", "Hx", "Hy"),
    ("x0", "z0", update_symmetry_boundary_electric_Ey_X0_Z0, "Ey", "Hx", "Hz"),
    ("x0", "zmax", update_symmetry_boundary_electric_Ey_X0_ZMax, "Ey", "Hx", "Hz"),
    ("xmax", "z0", update_symmetry_boundary_electric_Ey_XMax_Z0, "Ey", "Hx", "Hz"),
    ("xmax", "zmax", update_symmetry_boundary_electric_Ey_XMax_ZMax, "Ey", "Hx", "Hz"),
    ("y0", "z0", update_symmetry_boundary_electric_Ex_Y0_Z0, "Ex", "Hy", "Hz"),
    ("y0", "zmax", update_symmetry_boundary_electric_Ex_Y0_ZMax, "Ex", "Hy", "Hz"),
    ("ymax", "z0", update_symmetry_boundary_electric_Ex_YMax_Z0, "Ex", "Hy", "Hz"),
    ("ymax", "zmax", update_symmetry_boundary_electric_Ex_YMax_ZMax, "Ex", "Hy", "Hz"),
)

# Same 12 edges, dispersive Phase-A functions (same table shape as
# _EDGE_TABLE - only the function differs).
_EDGE_TABLE_DISPERSIVE = (
    ("x0", "y0", update_symmetry_boundary_electric_dispersive_Ez_X0_Y0, "Ez", "Hx", "Hy"),
    ("x0", "ymax", update_symmetry_boundary_electric_dispersive_Ez_X0_YMax, "Ez", "Hx", "Hy"),
    ("xmax", "y0", update_symmetry_boundary_electric_dispersive_Ez_XMax_Y0, "Ez", "Hx", "Hy"),
    ("xmax", "ymax", update_symmetry_boundary_electric_dispersive_Ez_XMax_YMax, "Ez", "Hx", "Hy"),
    ("x0", "z0", update_symmetry_boundary_electric_dispersive_Ey_X0_Z0, "Ey", "Hx", "Hz"),
    ("x0", "zmax", update_symmetry_boundary_electric_dispersive_Ey_X0_ZMax, "Ey", "Hx", "Hz"),
    ("xmax", "z0", update_symmetry_boundary_electric_dispersive_Ey_XMax_Z0, "Ey", "Hx", "Hz"),
    ("xmax", "zmax", update_symmetry_boundary_electric_dispersive_Ey_XMax_ZMax, "Ey", "Hx", "Hz"),
    ("y0", "z0", update_symmetry_boundary_electric_dispersive_Ex_Y0_Z0, "Ex", "Hy", "Hz"),
    ("y0", "zmax", update_symmetry_boundary_electric_dispersive_Ex_Y0_ZMax, "Ex", "Hy", "Hz"),
    ("ymax", "z0", update_symmetry_boundary_electric_dispersive_Ex_YMax_Z0, "Ex", "Hy", "Hz"),
    ("ymax", "zmax", update_symmetry_boundary_electric_dispersive_Ex_YMax_ZMax, "Ex", "Hy", "Hz"),
)

# Same 12 edges, dispersive Phase-B functions - no H arrays needed (matches
# the bulk Phase-B kernel's own signature), so only (face_a, face_b, func,
# e_attr) is needed.
_EDGE_TABLE_DISPERSIVE_B = (
    ("x0", "y0", update_symmetry_boundary_electric_dispersive_b_Ez_X0_Y0, "Ez"),
    ("x0", "ymax", update_symmetry_boundary_electric_dispersive_b_Ez_X0_YMax, "Ez"),
    ("xmax", "y0", update_symmetry_boundary_electric_dispersive_b_Ez_XMax_Y0, "Ez"),
    ("xmax", "ymax", update_symmetry_boundary_electric_dispersive_b_Ez_XMax_YMax, "Ez"),
    ("x0", "z0", update_symmetry_boundary_electric_dispersive_b_Ey_X0_Z0, "Ey"),
    ("x0", "zmax", update_symmetry_boundary_electric_dispersive_b_Ey_X0_ZMax, "Ey"),
    ("xmax", "z0", update_symmetry_boundary_electric_dispersive_b_Ey_XMax_Z0, "Ey"),
    ("xmax", "zmax", update_symmetry_boundary_electric_dispersive_b_Ey_XMax_ZMax, "Ey"),
    ("y0", "z0", update_symmetry_boundary_electric_dispersive_b_Ex_Y0_Z0, "Ex"),
    ("y0", "zmax", update_symmetry_boundary_electric_dispersive_b_Ex_Y0_ZMax, "Ex"),
    ("ymax", "z0", update_symmetry_boundary_electric_dispersive_b_Ex_YMax_Z0, "Ex"),
    ("ymax", "zmax", update_symmetry_boundary_electric_dispersive_b_Ex_YMax_ZMax, "Ex"),
)

# Complex-pole (Lorentz/Drude) counterparts of the two edge tables above -
# same shape, functions from symmetry_boundaries_dispersive_complex.
_EDGE_TABLE_DISPERSIVE_COMPLEX = (
    ("x0", "y0", _c_update_symmetry_boundary_electric_dispersive_Ez_X0_Y0, "Ez", "Hx", "Hy"),
    ("x0", "ymax", _c_update_symmetry_boundary_electric_dispersive_Ez_X0_YMax, "Ez", "Hx", "Hy"),
    ("xmax", "y0", _c_update_symmetry_boundary_electric_dispersive_Ez_XMax_Y0, "Ez", "Hx", "Hy"),
    ("xmax", "ymax", _c_update_symmetry_boundary_electric_dispersive_Ez_XMax_YMax, "Ez", "Hx", "Hy"),
    ("x0", "z0", _c_update_symmetry_boundary_electric_dispersive_Ey_X0_Z0, "Ey", "Hx", "Hz"),
    ("x0", "zmax", _c_update_symmetry_boundary_electric_dispersive_Ey_X0_ZMax, "Ey", "Hx", "Hz"),
    ("xmax", "z0", _c_update_symmetry_boundary_electric_dispersive_Ey_XMax_Z0, "Ey", "Hx", "Hz"),
    ("xmax", "zmax", _c_update_symmetry_boundary_electric_dispersive_Ey_XMax_ZMax, "Ey", "Hx", "Hz"),
    ("y0", "z0", _c_update_symmetry_boundary_electric_dispersive_Ex_Y0_Z0, "Ex", "Hy", "Hz"),
    ("y0", "zmax", _c_update_symmetry_boundary_electric_dispersive_Ex_Y0_ZMax, "Ex", "Hy", "Hz"),
    ("ymax", "z0", _c_update_symmetry_boundary_electric_dispersive_Ex_YMax_Z0, "Ex", "Hy", "Hz"),
    ("ymax", "zmax", _c_update_symmetry_boundary_electric_dispersive_Ex_YMax_ZMax, "Ex", "Hy", "Hz"),
)

_EDGE_TABLE_DISPERSIVE_B_COMPLEX = (
    ("x0", "y0", _c_update_symmetry_boundary_electric_dispersive_b_Ez_X0_Y0, "Ez"),
    ("x0", "ymax", _c_update_symmetry_boundary_electric_dispersive_b_Ez_X0_YMax, "Ez"),
    ("xmax", "y0", _c_update_symmetry_boundary_electric_dispersive_b_Ez_XMax_Y0, "Ez"),
    ("xmax", "ymax", _c_update_symmetry_boundary_electric_dispersive_b_Ez_XMax_YMax, "Ez"),
    ("x0", "z0", _c_update_symmetry_boundary_electric_dispersive_b_Ey_X0_Z0, "Ey"),
    ("x0", "zmax", _c_update_symmetry_boundary_electric_dispersive_b_Ey_X0_ZMax, "Ey"),
    ("xmax", "z0", _c_update_symmetry_boundary_electric_dispersive_b_Ey_XMax_Z0, "Ey"),
    ("xmax", "zmax", _c_update_symmetry_boundary_electric_dispersive_b_Ey_XMax_ZMax, "Ey"),
    ("y0", "z0", _c_update_symmetry_boundary_electric_dispersive_b_Ex_Y0_Z0, "Ex"),
    ("y0", "zmax", _c_update_symmetry_boundary_electric_dispersive_b_Ex_Y0_ZMax, "Ex"),
    ("ymax", "z0", _c_update_symmetry_boundary_electric_dispersive_b_Ex_YMax_Z0, "Ex"),
    ("ymax", "zmax", _c_update_symmetry_boundary_electric_dispersive_b_Ex_YMax_ZMax, "Ex"),
)


def _t_attr(e_attr: str) -> str:
    """Maps an E-field grid attribute name to its matching T-array
    attribute name, e.g. "Ez" -> "Tz"."""
    return "T" + e_attr[1:]


def build_symmetry_boundary_edges(grid) -> list:
    """Resolves, once at grid-build time, which of the 12 domain edges need
    a per-iteration PMC update, and with which (fixed) flags, for standard
    (non-dispersive) materials.

    An edge is included only if at least one of its two bordering faces is
    a declared PMC symmetry boundary - an edge where neither face is PMC is
    dropped entirely, not included with both flags False, so the
    per-iteration dispatcher never even calls into it.

    Args:
        grid: FDTDGrid class describing a grid in a model.

    Returns:
        edges: list of (cython_func, a_pmc, b_pmc, e_attr, h1_attr, h2_attr)
            tuples for FDTDGrid.symmetry_boundary_edges.
    """
    face_is_pmc = {face: grid.symmetry_boundaries.get(face) == "pmc" for face in _ALL_FACES}

    edges = []
    for face_a, face_b, func, e_attr, h1_attr, h2_attr in _EDGE_TABLE:
        a_pmc = face_is_pmc[face_a]
        b_pmc = face_is_pmc[face_b]
        if a_pmc or b_pmc:
            edges.append((func, a_pmc, b_pmc, e_attr, h1_attr, h2_attr))

    return edges


def build_symmetry_boundary_edges_dispersive(grid) -> list:
    """Dispersive Phase-A counterpart of build_symmetry_boundary_edges().

    Picks the real-pole (Debye) or complex-pole (Lorentz/Drude) edge table
    once here, at build time, from materials["drudelorentz"] - already set
    by Model._check_for_dispersive_materials() before grid.build() (and
    hence this method) runs for every grid. This matches
    grid.updatecoeffsdispersive/Tx/Ty/Tz's own dtype, chosen from the same
    flag in config.py's _set_precision()/set_dispersive_material_types(), so
    the functions returned here always match the arrays they'll be called
    with.

    Returns:
        edges: list of (cython_func, a_pmc, b_pmc, t_attr, e_attr, h1_attr,
            h2_attr) tuples for FDTDGrid.symmetry_boundary_edges_dispersive.
    """
    table = (
        _EDGE_TABLE_DISPERSIVE_COMPLEX
        if config.get_model_config().materials["drudelorentz"]
        else _EDGE_TABLE_DISPERSIVE
    )
    face_is_pmc = {face: grid.symmetry_boundaries.get(face) == "pmc" for face in _ALL_FACES}

    edges = []
    for face_a, face_b, func, e_attr, h1_attr, h2_attr in table:
        a_pmc = face_is_pmc[face_a]
        b_pmc = face_is_pmc[face_b]
        if a_pmc or b_pmc:
            edges.append((func, a_pmc, b_pmc, _t_attr(e_attr), e_attr, h1_attr, h2_attr))

    return edges


def build_symmetry_boundary_edges_dispersive_b(grid) -> list:
    """Dispersive Phase-B counterpart of build_symmetry_boundary_edges().

    See build_symmetry_boundary_edges_dispersive() for the real/complex
    table selection.

    Returns:
        edges: list of (cython_func, t_attr, e_attr) tuples for
            FDTDGrid.symmetry_boundary_edges_dispersive_b.
    """
    table = (
        _EDGE_TABLE_DISPERSIVE_B_COMPLEX
        if config.get_model_config().materials["drudelorentz"]
        else _EDGE_TABLE_DISPERSIVE_B
    )
    face_is_pmc = {face: grid.symmetry_boundaries.get(face) == "pmc" for face in _ALL_FACES}

    edges = []
    for face_a, face_b, func, e_attr in table:
        a_pmc = face_is_pmc[face_a]
        b_pmc = face_is_pmc[face_b]
        if a_pmc or b_pmc:
            edges.append((func, _t_attr(e_attr), e_attr))

    return edges


def update_symmetry_boundaries_electric_normal(grid) -> None:
    """Applies the per-iteration PMC ghost-node E update to every declared
    PMC symmetry boundary's face-interior region and, for domain edges
    touching at least one PMC face, the edge itself - for standard
    (non-dispersive) materials.

    Args:
        grid: FDTDGrid class describing a grid in a model.
    """
    if not grid.symmetry_boundaries:
        return

    nthreads = config.get_model_config().ompthreads

    for face, kind in grid.symmetry_boundaries.items():
        if kind != "pmc":
            continue

        _FACE_UPDATE_FUNCS[face](
            grid.nx,
            grid.ny,
            grid.nz,
            nthreads,
            grid.updatecoeffsE,
            grid.ID,
            grid.Ex,
            grid.Ey,
            grid.Ez,
            grid.Hx,
            grid.Hy,
            grid.Hz,
        )

    for func, a_pmc, b_pmc, e_attr, h1_attr, h2_attr in grid.symmetry_boundary_edges:
        func(
            grid.nx,
            grid.ny,
            grid.nz,
            nthreads,
            a_pmc,
            b_pmc,
            grid.updatecoeffsE,
            grid.ID,
            getattr(grid, e_attr),
            getattr(grid, h1_attr),
            getattr(grid, h2_attr),
        )


def update_symmetry_boundaries_electric_dispersive(grid) -> None:
    """Dispersive Phase-A counterpart of update_symmetry_boundaries_electric_normal().

    Called at the same point in the solve loop as the non-dispersive
    version (right after update_electric_a(), before PML/sources). Picks
    the real-pole (Debye) or complex-pole (Lorentz/Drude) face dict from
    materials["drudelorentz"] - see build_symmetry_boundary_edges_dispersive()
    for why this matches grid.updatecoeffsdispersive/Tx/Ty/Tz's own dtype.
    A plain dict lookup, done once per iteration (not per cell), so the
    cost is negligible either way.

    Args:
        grid: FDTDGrid class describing a grid in a model.
    """
    if not grid.symmetry_boundaries:
        return

    nthreads = config.get_model_config().ompthreads
    maxpoles = config.get_model_config().materials["maxpoles"]
    face_funcs = (
        _FACE_UPDATE_FUNCS_DISPERSIVE_COMPLEX
        if config.get_model_config().materials["drudelorentz"]
        else _FACE_UPDATE_FUNCS_DISPERSIVE
    )

    for face, kind in grid.symmetry_boundaries.items():
        if kind != "pmc":
            continue

        face_funcs[face](
            grid.nx,
            grid.ny,
            grid.nz,
            nthreads,
            maxpoles,
            grid.updatecoeffsE,
            grid.updatecoeffsdispersive,
            grid.ID,
            grid.Tx,
            grid.Ty,
            grid.Tz,
            grid.Ex,
            grid.Ey,
            grid.Ez,
            grid.Hx,
            grid.Hy,
            grid.Hz,
        )

    for func, a_pmc, b_pmc, t_attr, e_attr, h1_attr, h2_attr in grid.symmetry_boundary_edges_dispersive:
        func(
            grid.nx,
            grid.ny,
            grid.nz,
            nthreads,
            a_pmc,
            b_pmc,
            maxpoles,
            grid.updatecoeffsE,
            grid.updatecoeffsdispersive,
            grid.ID,
            getattr(grid, t_attr),
            getattr(grid, e_attr),
            getattr(grid, h1_attr),
            getattr(grid, h2_attr),
        )


def update_symmetry_boundaries_electric_dispersive_b(grid) -> None:
    """Dispersive Phase-B counterpart of update_symmetry_boundaries_electric_normal().

    Called at the same point in the solve loop as the bulk dispersive
    kernel's own Phase B (right before update_electric_b(), i.e. after PML
    and sources have possibly further modified E). See
    update_symmetry_boundaries_electric_dispersive() for the real/complex
    face dict selection.

    Args:
        grid: FDTDGrid class describing a grid in a model.
    """
    if not grid.symmetry_boundaries:
        return

    nthreads = config.get_model_config().ompthreads
    maxpoles = config.get_model_config().materials["maxpoles"]
    face_funcs = (
        _FACE_UPDATE_FUNCS_DISPERSIVE_B_COMPLEX
        if config.get_model_config().materials["drudelorentz"]
        else _FACE_UPDATE_FUNCS_DISPERSIVE_B
    )

    for face, kind in grid.symmetry_boundaries.items():
        if kind != "pmc":
            continue

        face_funcs[face](
            grid.nx,
            grid.ny,
            grid.nz,
            nthreads,
            maxpoles,
            grid.updatecoeffsdispersive,
            grid.ID,
            grid.Tx,
            grid.Ty,
            grid.Tz,
            grid.Ex,
            grid.Ey,
            grid.Ez,
        )

    for func, t_attr, e_attr in grid.symmetry_boundary_edges_dispersive_b:
        func(
            grid.nx,
            grid.ny,
            grid.nz,
            nthreads,
            maxpoles,
            grid.updatecoeffsdispersive,
            grid.ID,
            getattr(grid, t_attr),
            getattr(grid, e_attr),
        )
