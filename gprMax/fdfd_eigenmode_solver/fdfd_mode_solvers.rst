FDFD Eigenmode Solvers
======================

Overview
--------

gprMax uses two finite-difference frequency-domain eigenmode solvers:

.. list-table::
   :header-rows: 1

   * - Time-domain model
     - Modal cross-section
     - Solver
   * - 2D TM or TE
     - One physical transverse coordinate
     - ``FDFD_1D_mode_solver``
   * - 3D
     - Two physical transverse coordinates
     - ``FDFD_2D_mode_solver``

Both solvers operate directly on component-sampled Yee grids, accept complex
permittivity and permeability, enforce PEC and PMC constraints at the
corresponding component locations, select the passive positive effective-index
branch, and return power-normalised fields for ``EigenmodeSource``.

1D Scalar Solver for 2D Models
------------------------------

``fdfd_1d_mode_solver.py`` supplies the scalar, Yee-staggered mode solve used
by eigenmode sources in gprMax 2D TM and TE domains.

1D Coordinates and Yee Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The local basis is ``(t, a, w)``:

``t``
    The one physical transverse coordinate.

``a``
    The invariant 2D axis.

``w``
    The in-plane propagation direction and source normal.

For ``N`` cells along ``t``, the native staggered shapes are:

.. list-table::
   :header-rows: 1

   * - Material array
     - Field component
     - Shape
   * - ``eps_r_t``
     - ``E_t``
     - ``(N,)``
   * - ``eps_r_a``
     - ``E_a``
     - ``(N + 1,)``
   * - ``eps_r_w``
     - ``E_w``
     - ``(N + 1,)``
   * - ``mu_r_t``
     - ``H_t``
     - ``(N + 1,)``
   * - ``mu_r_a``
     - ``H_a``
     - ``(N,)``
   * - ``mu_r_w``
     - ``H_w``
     - ``(N,)``

The TM reduction solves the node-sampled scalar field ``E_a`` and reconstructs
``H_t`` and ``H_w``. The TE reduction solves the cell-sampled scalar field
``H_a`` and reconstructs ``E_t`` and ``E_w``. No derivative is taken through
gprMax's artificial one-cell TM or two-cell TE invariant-axis thickness.

1D Inputs
^^^^^^^^^

The constructor signature is:

.. code-block:: python

   FDFD_1D_mode_solver(
       frequency,
       dt,
       mode_index,
       polarization,
       eps_r_t,
       eps_r_a,
       eps_r_w,
       mu_r_t,
       mu_r_a,
       mu_r_w,
       pec_t_mask=None,
       pec_a_mask=None,
       pec_w_mask=None,
       pmc_t_mask=None,
       pmc_a_mask=None,
       pmc_w_mask=None,
       guess=None,
   )

``frequency``
    Modal solve frequency in Hz.

``dt``
    Yee-cell spacing along the physical transverse coordinate ``t``, in
    metres. Despite its name, this is a spatial step rather than a time step.

``mode_index``
    Zero-based requested mode. The solver computes ``mode_index + 1`` modes
    and exposes the selected one through ``modal_Et``, ``modal_Ea``,
    ``modal_Ew``, ``modal_Ht``, ``modal_Ha``, ``modal_Hw`` and
    ``modal_real_neff``.

``polarization``
    ``"TM"`` selects the ``E_a`` scalar problem. ``"TE"`` selects the ``H_a``
    scalar problem.

``eps_r_*`` and ``mu_r_*``
    Complex relative material arrays sampled at the component locations in
    `1D Coordinates and Yee Shapes`_.

``pec_*_mask`` and ``pmc_*_mask``
    Optional boolean masks at the matching electric or magnetic component
    locations. Non-finite values in the corresponding material arrays are
    also interpreted as constraints.

``guess``
    Optional ARPACK shift. If omitted, the solver derives a shift from the
    largest finite material magnitude.

1D Eigenproblem and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The solver constructs a node-to-cell difference operator ``D_nc`` and its
negative adjoint:

.. code-block:: text

   D_nc : node fields -> cell fields
   D_cn = -D_nc.H : cell fields -> node fields

Both operators are normalised by ``k0 * dt``. For TM, the scalar eigenproblem
is assembled on the ``E_a`` nodes:

.. code-block:: text

   A_TM = -mu_t [D_cn inv(mu_w) D_nc + eps_a]
   A_TM E_a = lambda E_a

For TE, it is assembled on the ``H_a`` cells:

.. code-block:: text

   A_TE = -eps_t [D_nc inv(eps_w) D_cn + mu_a]
   A_TE H_a = lambda H_a

In both cases:

.. code-block:: text

   n_eff = sqrt(-lambda)

The square-root branch is chosen for positive phase propagation and
non-negative attenuation.

PEC constraints remove electric scalar degrees of freedom from the TM
problem, while PMC constraints remove magnetic scalar degrees of freedom from
the TE problem. Longitudinal inverse-material operators are evaluated only on
unconstrained degrees of freedom: constrained entries receive a zero inverse.
After the reduced sparse eigenproblem is solved, the eigenvectors are expanded
back to their full node or cell arrays and every constrained field component
is explicitly zeroed.

1D Field Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^

For TM, the selected ``E_a`` eigenvector gives:

.. code-block:: text

   H_t = -n_eff E_a / (eta0 mu_t)
   H_w = -i inv(mu_w) D_nc E_a / eta0

For TE, the selected ``H_a`` eigenvector gives:

.. code-block:: text

   E_t = eta0 n_eff H_a / eps_t
   E_w = i eta0 inv(eps_w) D_cn H_a

The other three field components are identically zero for the selected 2D
polarization. The reconstructed electric fields are in V/m and magnetic
fields are in A/m.

1D Normalisation and Phase Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each mode is normalised to one watt per metre of invariant-axis length. For
TM, node-sampled ``E_a`` and ``H_t`` are first averaged onto transverse cells:

.. code-block:: text

   P_TM = 0.5 Re sum(-E_a H_t*) dt

For TE, ``E_t`` and ``H_a`` already share the cell locations:

.. code-block:: text

   P_TE = 0.5 Re sum(E_t H_a*) dt

If the initial power is negative, all magnetic fields are reversed before
normalisation. Each complex mode is then phase-rotated so that the real-valued
profiles used by the FDTD injection carry positive real-profile power.

1D gprMax Integration
^^^^^^^^^^^^^^^^^^^^^^

``EigenmodeSource`` samples component materials from the mode's live invariant
layer, supplies the corresponding PEC/PMC masks, and maps the returned line
profiles back into the thin 3D Yee arrays used by the CPU update kernels.
The TM source uses the single live invariant layer; the TE source uses the
shared interior layer of its two-cell invariant thickness. Inactive components
and TE outer boundary planes are explicitly zero.

Modes are selected using the same shift-invert convention as the full-vector
2D solver. ``plot_fields`` writes one row per computed mode with line plots of
all three active fields, including their node- or cell-sampled locations.

2D Full-Vector Solver for 3D Models
-----------------------------------

``fdfd_2d_mode_solver.py`` contains ``FDFD_2D_mode_solver``, the full-vector
solver used when a gprMax eigenmode source has two physical transverse
coordinates.

2D Coordinates and Yee Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The solver works in a local coordinate system rather than the global gprMax
``x``, ``y`` and ``z`` axes:

``u``
    First transverse source-plane axis.

``v``
    Second transverse source-plane axis.

``w``
    Propagation-normal axis.

For example, a source normal to global ``z`` uses local ``u=x``, ``v=y`` and
``w=z``. A source normal to global ``x`` uses local ``u=y``, ``v=z`` and
``w=x``.

The solver is built on a true staggered Yee grid. Material arrays supplied to
the constructor must already be sampled at the corresponding local field
component locations. The solver does not average cell-centred material data and
it does not collapse all fields onto a common rectangular array.

For a transverse source region containing ``Nu`` by ``Nv`` Yee cells, the
expected local component shapes are:

.. list-table::
   :header-rows: 1

   * - Array
     - Field component
     - Shape
   * - ``eps_r_uu``
     - ``E_u``
     - ``(Nu, Nv + 1)``
   * - ``eps_r_vv``
     - ``E_v``
     - ``(Nu + 1, Nv)``
   * - ``eps_r_ww``
     - ``E_w``
     - ``(Nu + 1, Nv + 1)``
   * - ``mu_r_uu``
     - ``H_u``
     - ``(Nu + 1, Nv)``
   * - ``mu_r_vv``
     - ``H_v``
     - ``(Nu, Nv + 1)``
   * - ``mu_r_ww``
     - ``H_w``
     - ``(Nu, Nv)``

The returned modal fields use the same native Yee shapes:

.. list-table::
   :header-rows: 1

   * - Modal field
     - Shape
   * - ``modal_Eu``
     - ``(Nu, Nv + 1)``
   * - ``modal_Ev``
     - ``(Nu + 1, Nv)``
   * - ``modal_Ew``
     - ``(Nu + 1, Nv + 1)``
   * - ``modal_Hu``
     - ``(Nu + 1, Nv)``
   * - ``modal_Hv``
     - ``(Nu, Nv + 1)``
   * - ``modal_Hw``
     - ``(Nu, Nv)``

Only transverse fields participate in gprMax eigenmode source injection. The
longitudinal fields ``E_w`` and ``H_w`` are still reconstructed because they
are part of the full-vector mode solution, but the TF/SF correction kernels use
only the tangential/transverse modal components.

2D Inputs
^^^^^^^^^

The constructor signature is:

.. code-block:: python

   FDFD_2D_mode_solver(
       frequency,
       du,
       dv,
       mode_index,
       eps_r_uu,
       eps_r_vv,
       eps_r_ww,
       mu_r_uu,
       mu_r_vv,
       mu_r_ww,
       pec_u_mask=None,
       pec_v_mask=None,
       pec_w_mask=None,
       pmc_u_mask=None,
       pmc_v_mask=None,
       pmc_w_mask=None,
       guess=None,
   )

``frequency``
    Source frequency in Hz.

``du``, ``dv``
    Local transverse cell sizes in metres. The solver normalises finite-
    difference operators by ``k0 * du`` and ``k0 * dv``.

``mode_index``
    Zero-based modal index requested by the caller. The solver computes
    ``mode_index + 1`` modes, then exposes the requested mode through
    ``modal_Eu``, ``modal_Ev``, ``modal_Ew``, ``modal_Hu``, ``modal_Hv``,
    ``modal_Hw`` and ``modal_real_neff``.

``eps_r_*`` and ``mu_r_*``
    Complex relative permittivity and permeability arrays sampled at the local
    Yee component locations listed in `2D Coordinates and Yee Shapes`_.

``pec_u_mask``, ``pec_v_mask``, ``pec_w_mask``
    Optional explicit boolean masks for constrained electric degrees of
    freedom. They must match the ``E_u``, ``E_v`` and ``E_w`` shapes.

``pmc_u_mask``, ``pmc_v_mask``, ``pmc_w_mask``
    Optional explicit boolean masks for constrained magnetic degrees of
    freedom. Non-finite entries in the matching permeability arrays are also
    interpreted as PMC.

``guess``
    Optional ARPACK shift. If omitted, the solver chooses a conservative shift
    from the largest finite material magnitude.

2D Array Ordering
^^^^^^^^^^^^^^^^^

All component arrays are flattened with Fortran order:

.. code-block:: python

   flat = array.ravel(order="F")

and modal vectors are reshaped back with:

.. code-block:: python

   array = vector.reshape((*shape, num_modes), order="F")

There is no axis-order switch. gprMax must pass local ``u``/``v`` slices in the
same native transverse ordering used by the extracted source plane.

2D PEC Handling
^^^^^^^^^^^^^^^

PEC is represented as constrained electric degrees of freedom, not as a large
finite permittivity approximation.

The solver detects electric PEC in two ways:

1. Explicit electric masks passed through ``pec_u_mask``, ``pec_v_mask`` and
   ``pec_w_mask``.
2. Non-finite values, normally ``np.inf + 0j``, in the electric material arrays.

For example:

.. code-block:: python

   eps_r_uu[pec_u_mask] = np.inf + 0j
   eps_r_vv[pec_v_mask] = np.inf + 0j
   eps_r_ww[pec_w_mask] = np.inf + 0j

Each electric component is treated independently:

* ``pec_u_mask`` constrains ``E_u`` to zero.
* ``pec_v_mask`` constrains ``E_v`` to zero.
* ``pec_w_mask`` constrains ``E_w`` to zero.

After masks are built, PEC material entries are replaced by finite placeholders
before matrix assembly:

.. code-block:: python

   eps_r_uu[self.pec_u_mask] = 1.0 + 0j

The physics is carried by removed/constrained degrees of freedom, not by the
placeholder value. Large finite values such as ``1e8`` or ``1e10`` are ordinary
finite material values and are intentionally not treated as PEC.

2D Eigenproblem
^^^^^^^^^^^^^^^

The solver constructs rectangular sparse derivative matrices between true Yee
component grids. The core local operators are:

.. code-block:: text

   DEU_EW_TO_EU : E_w -> E_u
   DEV_EW_TO_EV : E_w -> E_v
   DEU_EV_TO_HW : E_v -> H_w
   DEV_EU_TO_HW : E_u -> H_w

and their adjoint magnetic-grid counterparts:

.. code-block:: text

   DHU_HV_TO_EW = -DEU_EW_TO_EU.H
   DHV_HU_TO_EW = -DEV_EW_TO_EV.H
   DHU_HW_TO_HU = -DEU_EV_TO_HW.H
   DHV_HW_TO_HV = -DEV_EU_TO_HW.H

The transverse electric field vector is:

.. code-block:: python

   Euv = [E_u, E_v]^T

The solver forms the standard full-vector FDFD ``P`` and ``Q`` matrices and
solves:

.. code-block:: text

   Omega * Euv = eigenvalue * Euv
   Omega = P * Q

where the effective index is recovered from:

.. code-block:: python

   neff = sqrt(-eigenvalue)

The square-root branch is chosen so that the phase constant is positive and
attenuation is non-negative.

Because the operators connect the correct staggered Yee component grids, there
is no separate PEC-neighbour spurious-mode rejection heuristic in this solver.
The old candidate scoring/filtering path has been removed.

2D Degree-of-Freedom Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PEC constraints are applied by removing constrained transverse electric degrees
of freedom from the eigenproblem:

.. code-block:: python

   Omega = Omega[self.free_euv_mask, :][:, self.free_euv_mask]

After ARPACK returns the reduced eigenvectors, the solver expands them back to
the full transverse field-vector size and explicitly zeros constrained fields.

The inverse ``eps_r_ww`` and ``mu_r_ww`` operators are built only on free
longitudinal degrees of freedom. Constrained entries receive zero inverse
values so that no division by ``np.inf`` or placeholder data affects the
reconstructed fields.

2D Field Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^

After solving the eigenproblem, the solver reconstructs:

* ``E_u`` and ``E_v`` directly from the transverse eigenvector.
* ``H_u`` and ``H_v`` from ``Q * Euv / sqrt(eigenvalue)``.
* ``E_w`` from transverse magnetic curl terms.
* ``H_w`` from transverse electric curl terms.

Magnetic fields are converted to physical A/m using ``eta0``:

.. code-block:: python

   H = 1j * H_normalized / eta0

The solver then zeroes all constrained fields to ensure returned modal fields
satisfy the enforced constraints exactly.

2D Normalisation and Phase Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Modes are normalised to carry one watt of time-average power. Power is computed
from cell-centred transverse Poynting flux by averaging the staggered
transverse fields onto local cells:

.. code-block:: text

   P = 0.5 * Re integral((E_u * H_v* - E_v * H_u*) dA)

If a mode initially carries negative power, the magnetic field is flipped
before normalisation. After normalisation, each complex mode is phase-rotated
so that its real-valued field profile carries positive real-profile power.
This makes plotted and injected real fields easier to interpret.

2D gprMax Integration
^^^^^^^^^^^^^^^^^^^^^

``sources.py`` extracts complex material tensors from ``G.ID`` after the Yee
grid has been built. This is the correct integration point because ``G.ID`` is
already sampled at Yee component locations.

For a source plane, ``sources.py`` maps global components into local
``u``/``v``/``w`` components:

.. code-block:: python

   local_to_global = (
       self.transverse_axes[0],
       self.transverse_axes[1],
       self.normal_axis,
   )

It then extracts six native Yee slices:

* electric local ``u`` component: ``(Nu, Nv + 1)``
* electric local ``v`` component: ``(Nu + 1, Nv)``
* electric local ``w`` component: ``(Nu + 1, Nv + 1)``
* magnetic local ``u`` component: ``(Nu + 1, Nv)``
* magnetic local ``v`` component: ``(Nu, Nv + 1)``
* magnetic local ``w`` component: ``(Nu, Nv)``

For electric materials:

* finite conductivity is folded into complex permittivity,
* ``se == inf`` is converted to ``np.inf + 0j``, which the solver treats as
  PEC.

For magnetic materials:

* finite magnetic conductivity is folded into complex permeability,
* ``sm == inf`` is converted to ``np.inf + 0j``, which the solver treats as
  PMC.

After solving, ``sources.py`` maps local modal fields back to global component
slots. The Cython injection kernels consume the transverse components with
their native staggered shapes; longitudinal modal fields are stored but are not
used for TF/SF source corrections.

Limitations
-----------

* Material tensors are diagonal in the local ``u``/``v``/``w`` basis.
* The finite-difference operators use first-order sparse Yee-grid differences.
* A mode is solved at one frequency and used to approximate the spatial field
  profile of the broadband FDTD source. Frequency-dependent changes in modal
  shape, effective index, dispersion, and loss are therefore not represented
  across the full source bandwidth.

Recommended 2D-Solver Usage
---------------------------

For gprMax integration, use this path:

1. Extract local ``eps_r_uu``, ``eps_r_vv`` and ``eps_r_ww`` from Yee electric
   component material IDs with native staggered shapes.
2. Mark electric PEC entries with ``np.inf + 0j`` or explicit local PEC masks.
3. Extract local ``mu_r_uu``, ``mu_r_vv`` and ``mu_r_ww`` from Yee magnetic
   component material IDs with native staggered shapes.
4. Mark magnetic PMC entries with ``np.inf + 0j`` or explicit local PMC masks.
5. Construct ``FDFD_2D_mode_solver`` using local ``du`` and ``dv``.
6. Call ``solver.solve()``.
7. Use ``solver.modal_Eu``, ``solver.modal_Ev``, ``solver.modal_Hu`` and
   ``solver.modal_Hv`` for transverse eigenmode source injection after mapping
   local components back to global gprMax components.
