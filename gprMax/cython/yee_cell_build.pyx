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

import numpy as np

cimport cython
cimport numpy as np

from gprMax.cython.yee_cell_setget_rigid cimport (
    get_rigid_Ex,
    get_rigid_Ey,
    get_rigid_Ez,
    get_rigid_Hx,
    get_rigid_Hy,
    get_rigid_Hz,
)

from gprMax.materials import Material


@cython.cdivision(True)
cdef inline double harmonic_mean2(double a, double b) noexcept nogil:
    """Harmonic mean of two values.

    Uses the reciprocal-sum form (not the algebraically equivalent
    2*a*b/(a+b)) so that the physically-relevant boundary values 0 and
    inf - both reachable for magnetic loss (sigma*), e.g. free_space
    (0) next to a custom sm=inf/builtin pmc material (inf) - resolve to
    their correct limits (0 and 2*finite value respectively) via ordinary
    IEEE-754 semantics, instead of the 0*inf / inf-inf indeterminate forms
    the product form would hit.

    Needs cython.cdivision(True): Cython's default division checks for a
    zero denominator and raises ZeroDivisionError to match Python
    semantics, even for floats (unlike plain C) - which would defeat the
    reciprocal-sum trick above, since a=0 is the common case (any
    non-magnetically-lossy material). cdivision(True) switches to plain
    IEEE-754 float division, where 1.0/0.0 is +inf rather than an
    exception.
    """
    return 2.0 / (1.0 / a + 1.0 / b)


cpdef void create_electric_average(
    int i,
    int j,
    int k,
    int numID1,
    int numID2,
    int numID3,
    int numID4,
    int componentID,
    G
):
    """Creates a new material by averaging the dielectric properties of the
        surrounding cells.

    Args:
        i, j, k: ints for cell coordinates.
        numID: ints for numeric IDs for materials in surrounding cells.
        componentID: int for numeric ID for electric field component.
        G: FDTDGrid class describing a grid in a model.
    """

    # Make an ID composed of the names of the four materials that will
    # be averaged. Sort the names to ensure the same four component
    # materials always form the same ID.
    requiredID = Material.create_compound_id(G.materials[numID1], G.materials[numID2], G.materials[numID3], G.materials[numID4])

    # Check if this material already exists
    material = [x for x in G.materials if x.ID == requiredID]

    if material:
        G.ID[componentID, i, j, k] = material[0].numID
    else:
        # Create new material
        newNumID = len(G.materials)
        m = Material(newNumID, requiredID)
        m.type = 'dielectric-smoothed'
        # Create averaged constituents for material
        m.er = np.mean((G.materials[numID1].er, G.materials[numID2].er,
                        G.materials[numID3].er, G.materials[numID4].er), axis=0)
        m.se = np.mean((G.materials[numID1].se, G.materials[numID2].se,
                        G.materials[numID3].se, G.materials[numID4].se), axis=0)
        m.mr = np.mean((G.materials[numID1].mr, G.materials[numID2].mr,
                        G.materials[numID3].mr, G.materials[numID4].mr), axis=0)
        m.sm = np.mean((G.materials[numID1].sm, G.materials[numID2].sm,
                        G.materials[numID3].sm, G.materials[numID4].sm), axis=0)

        # Append the new material object to the materials list
        G.materials.append(m)

        G.ID[componentID, i, j, k] = newNumID


cpdef void create_magnetic_average(
    int i,
    int j,
    int k,
    int numID1,
    int numID2,
    int componentID,
    G,
    bint harmonic
):
    """Creates a new material by averaging the properties of the
        surrounding cells for a magnetic (H) field component.

    Args:
        i, j, k: ints for cell coordinates.
        numID: ints for numeric IDs for materials in surrounding cells.
        componentID: int for numeric ID for magnetic field component.
        G: FDTDGrid class describing a grid in a model.
        harmonic: bool, True to combine mu_r/sigma* with a harmonic mean
            (default, physically correct for the field-normal direction
            these two cells are stacked in - see #magnetic_averaging),
            False for the legacy arithmetic mean.
    """

    # Make an ID composed of the names of the two materials that will be
    # averaged. Material.create_compound_id() duplicates a 2-material call
    # into a 4-part name ("A+A+B+B") specifically so it collides with (and
    # reuses) a 4-cell electric average of the same 2 materials - correct
    # only as long as both use the same mixing rule, since arithmetic mean
    # of {A,A,B,B} equals arithmetic mean of {A,B}. With harmonic H
    # averaging that equivalence no longer holds, so the harmonic case
    # gets its own "Hmag_" namespace to avoid silently reusing an
    # arithmetic-mean electric material for mu_r/sigma*. The arithmetic
    # case is left exactly as before (byte-for-byte, including the
    # electric/magnetic reuse) for backwards compatibility.
    #
    # Must not use ':' here (or any other character that could appear only
    # once elsewhere) - hash_cmds_file.py's command-line parser does a bare
    # `line.split(":")` and keeps only cmd[1], so a second ':' anywhere in
    # a material ID (e.g. from a round-trip through #geometry_objects_write
    # / #geometry_objects_read, which writes material.ID verbatim into a
    # #material: line) silently truncates the name there. '_' is already
    # used elsewhere in material IDs (e.g. "free_space") and is safe.
    requiredID = Material.create_compound_id(G.materials[numID1], G.materials[numID2])
    if harmonic:
        requiredID = "Hmag_" + requiredID

    # Check if this material already exists
    material = [x for x in G.materials if x.ID == requiredID]

    if material:
        G.ID[componentID, i, j, k] = material[0].numID
    else:
        # Create new material
        newNumID = len(G.materials)
        m = Material(newNumID, requiredID)
        m.type = 'dielectric-smoothed'
        # er/se are not used by the H-field update coefficients (only
        # mr/sm are - see Material.calculate_update_coeffsH()), so they
        # are always arithmetic-averaged here regardless of "harmonic",
        # purely for display/reporting consistency.
        m.er = np.mean((G.materials[numID1].er, G.materials[numID2].er), axis=0)
        m.se = np.mean((G.materials[numID1].se, G.materials[numID2].se), axis=0)
        if harmonic:
            m.mr = harmonic_mean2(<double>G.materials[numID1].mr, <double>G.materials[numID2].mr)
            m.sm = harmonic_mean2(<double>G.materials[numID1].sm, <double>G.materials[numID2].sm)
        else:
            m.mr = np.mean((G.materials[numID1].mr, G.materials[numID2].mr), axis=0)
            m.sm = np.mean((G.materials[numID1].sm, G.materials[numID2].sm), axis=0)

        # Append the new material object to the materials list
        G.materials.append(m)

        G.ID[componentID, i, j, k] = newNumID


cpdef void build_electric_components(
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidE,
    np.uint32_t[:, :, :, ::1] ID,
    G
):
    """Builds the electric field components in the ID array.

    Each E component has one "own" axis (no material dependency, full range
    0..n-1) and two "dependency" axes (need the cell at index v and v-1, full
    range 0..n to cover the ID array's n+1 sized dependency dimensions). At a
    dependency-axis wall (v=0 or v=n) the missing out-of-domain neighbour is
    clamped onto the cell that does exist, which collapses the usual 4-cell
    average to the correct 2-cell (edge) or 1-cell (corner) result via the
    existing "if all equal" fast path - no special-casing needed here.

    Note (MPI): for an MPIGrid rank, a "dependency-axis wall" reached here may
    be an internal partition seam rather than a true global-domain wall, in
    which case this clamp is a placeholder like before this fix (inert for
    the main FDTD update, which never reads these components). Code adding a
    boundary-plane update (e.g. PMC symmetry planes) must gate on
    has_neighbour(dim, dir) and only trust this ID at genuine global walls.

    Args:
        solid, rigid, ID: memoryviews to access solid, rigid and ID arrays.
        G: FDTDGrid class describing a grid in a model.
    """

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t i_hi, i_lo, j_hi, j_lo, k_hi, k_lo
    cdef Py_ssize_t nx, ny, nz
    cdef int numID1, numID2, numID3, numID4, IDEx, IDEy, IDEz

    IDEx = G.IDlookup['Ex']
    IDEy = G.IDlookup['Ey']
    IDEz = G.IDlookup['Ez']

    nx = G.nx
    ny = G.ny
    nz = G.nz

    # Ex: own axis i; dependency axes j, k
    for i in range(nx):
        for j in range(ny + 1):
            j_hi = j if j < ny else ny - 1
            j_lo = j - 1 if j > 0 else 0
            for k in range(nz + 1):
                k_hi = k if k < nz else nz - 1
                k_lo = k - 1 if k > 0 else 0

                # If rigid is True do not average
                if get_rigid_Ex(i, j, k, rigidE):
                    pass
                else:
                    numID1 = solid[i, j_hi, k_hi]
                    numID2 = solid[i, j_lo, k_hi]
                    numID3 = solid[i, j_lo, k_lo]
                    numID4 = solid[i, j_hi, k_lo]

                    # If all values are the same no need to average
                    if numID1 == numID2 and numID1 == numID3 and numID1 == numID4:
                        ID[IDEx, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_electric_average(i, j, k, numID1, numID2,
                                                numID3, numID4, IDEx, G)

    # Ey: own axis j; dependency axes i, k
    for i in range(nx + 1):
        i_hi = i if i < nx else nx - 1
        i_lo = i - 1 if i > 0 else 0
        for j in range(ny):
            for k in range(nz + 1):
                k_hi = k if k < nz else nz - 1
                k_lo = k - 1 if k > 0 else 0

                # If rigid is True do not average
                if get_rigid_Ey(i, j, k, rigidE):
                    pass
                else:
                    numID1 = solid[i_hi, j, k_hi]
                    numID2 = solid[i_lo, j, k_hi]
                    numID3 = solid[i_lo, j, k_lo]
                    numID4 = solid[i_hi, j, k_lo]

                    # If all values are the same no need to average
                    if numID1 == numID2 and numID1 == numID3 and numID1 == numID4:
                        ID[IDEy, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_electric_average(i, j, k, numID1, numID2,
                                                numID3, numID4, IDEy, G)

    # Ez: own axis k; dependency axes i, j
    for i in range(nx + 1):
        i_hi = i if i < nx else nx - 1
        i_lo = i - 1 if i > 0 else 0
        for j in range(ny + 1):
            j_hi = j if j < ny else ny - 1
            j_lo = j - 1 if j > 0 else 0
            for k in range(nz):

                # If rigid is True do not average
                if get_rigid_Ez(i, j, k, rigidE):
                    pass
                else:
                    numID1 = solid[i_hi, j_hi, k]
                    numID2 = solid[i_lo, j_hi, k]
                    numID3 = solid[i_lo, j_lo, k]
                    numID4 = solid[i_hi, j_lo, k]

                    # If all values are the same no need to average
                    if numID1 == numID2 and numID1 == numID3 and numID1 == numID4:
                        ID[IDEz, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_electric_average(i, j, k, numID1, numID2,
                                                numID3, numID4, IDEz, G)


cpdef void build_magnetic_components(
    np.uint32_t[:, :, ::1] solid,
    np.int8_t[:, :, :, ::1] rigidH,
    np.uint32_t[:, :, :, ::1] ID,
    G,
    bint harmonic=True
):
    """Builds the magnetic field components in the ID array.

    Each H component has two "own" axes (no material dependency, full range
    0..n-1 each) and one "dependency" axis (needs the cell at index v and
    v-1, full range 0..n to cover the ID array's n+1 sized dependency
    dimension). At a dependency-axis wall (v=0 or v=n) the missing
    out-of-domain neighbour is clamped onto the cell that does exist, which
    collapses the 2-cell average to a direct single-cell assignment via the
    existing "if equal" fast path - no special-casing needed here.

    Note (MPI): see build_electric_components - the same caveat about
    internal partition seams vs. genuine global-domain walls applies here.

    PEC transparency: PEC has no well-defined magnetic properties (Part A of
    the PMC-boundary groundwork already stops build_voxel/build_box from
    forcing/rigid-marking H directly at a PEC location) - but solid[] still
    (correctly) shows PEC's numID there, so without this check the averaging
    below would blend PEC's meaningless mr=1/sm=0 placeholder into a real
    neighbour's properties. A PEC neighbour is instead treated as transparent:
    if exactly one of the two neighbours is PEC, the real neighbour's numID is
    used directly (no averaging, matching what a model without the PEC object
    would produce there); if both are PEC, there is no real medium left to
    reference (solid[] no longer remembers what was there before the PEC
    object was placed) so the position is left untouched, consistent with
    Part A's "don't force anything" philosophy.

    Args:
        solid, rigid, ID: memoryviews to access solid, rigid and ID arrays.
        G: FDTDGrid class describing a grid in a model.
        harmonic: bool, see create_magnetic_average(). Resolved once by the
            caller (from #magnetic_averaging) rather than per averaged
            cell; defaults to True (the harmonic-mean default) so direct
            callers that don't care about this setting (e.g. low-level
            tests building a bare grid without full simulation config)
            still get sensible behaviour without needing config set up.
    """

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t i_hi, i_lo, j_hi, j_lo, k_hi, k_lo
    cdef Py_ssize_t nx, ny, nz
    cdef int numID1, numID2, IDHx, IDHy, IDHz
    cdef bint pec1, pec2
    cdef np.uint8_t[::1] is_pec_lookup

    IDHx = G.IDlookup['Hx']
    IDHy = G.IDlookup['Hy']
    IDHz = G.IDlookup['Hz']

    nx = G.nx
    ny = G.ny
    nz = G.nz

    is_pec_lookup = np.array([m.is_pec for m in G.materials], dtype=np.uint8)

    # Hx: own axes j, k; dependency axis i
    for i in range(nx + 1):
        i_hi = i if i < nx else nx - 1
        i_lo = i - 1 if i > 0 else 0
        for j in range(ny):
            for k in range(nz):

                # If rigid is True do not average
                if get_rigid_Hx(i, j, k, rigidH):
                    pass
                else:
                    numID1 = solid[i_hi, j, k]
                    numID2 = solid[i_lo, j, k]
                    pec1 = is_pec_lookup[numID1]
                    pec2 = is_pec_lookup[numID2]

                    if pec1 and pec2:
                        pass
                    elif pec1:
                        ID[IDHx, i, j, k] = numID2
                    elif pec2:
                        ID[IDHx, i, j, k] = numID1
                    elif numID1 == numID2:
                        ID[IDHx, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_magnetic_average(i, j, k, numID1, numID2, IDHx, G, harmonic)

    # Hy: own axes i, k; dependency axis j
    for i in range(nx):
        for j in range(ny + 1):
            j_hi = j if j < ny else ny - 1
            j_lo = j - 1 if j > 0 else 0
            for k in range(nz):

                # If rigid is True do not average
                if get_rigid_Hy(i, j, k, rigidH):
                    pass
                else:
                    numID1 = solid[i, j_hi, k]
                    numID2 = solid[i, j_lo, k]
                    pec1 = is_pec_lookup[numID1]
                    pec2 = is_pec_lookup[numID2]

                    if pec1 and pec2:
                        pass
                    elif pec1:
                        ID[IDHy, i, j, k] = numID2
                    elif pec2:
                        ID[IDHy, i, j, k] = numID1
                    elif numID1 == numID2:
                        ID[IDHy, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_magnetic_average(i, j, k, numID1, numID2, IDHy, G, harmonic)

    # Hz: own axes i, j; dependency axis k
    for i in range(nx):
        for j in range(ny):
            for k in range(nz + 1):
                k_hi = k if k < nz else nz - 1
                k_lo = k - 1 if k > 0 else 0

                # If rigid is True do not average
                if get_rigid_Hz(i, j, k, rigidH):
                    pass
                else:
                    numID1 = solid[i, j, k_hi]
                    numID2 = solid[i, j, k_lo]
                    pec1 = is_pec_lookup[numID1]
                    pec2 = is_pec_lookup[numID2]

                    if pec1 and pec2:
                        pass
                    elif pec1:
                        ID[IDHz, i, j, k] = numID2
                    elif pec2:
                        ID[IDHz, i, j, k] = numID1
                    elif numID1 == numID2:
                        ID[IDHz, i, j, k] = numID1
                    else:
                        # Averaging is required
                        create_magnetic_average(i, j, k, numID1, numID2, IDHz, G, harmonic)
