.. _eigenmode-theory:

*********************************************
Eigenmode ports: theory and numerical methods
*********************************************

Start with :doc:`eigenmode_port` for Python and hash setup, matched feed
terminations, modal channel selection, and result interpretation. This page
defines the operators and conventions used by the solvers, sources, and
monitors. It distinguishes physical-frequency quantities from the discrete
symbols needed to reproduce the FDTD update.

The `published HTML guide <https://docs.gprmax.com/en/latest/eigenmode_port_theory.html>`_
renders the equations and cross-references in full.

.. contents:: On this page
   :local:
   :depth: 2

Mathematical formulation
========================

The remaining sections describe the component-sampled FDFD eigenproblems,
power normalization and modal reconstruction, followed by spectrum synthesis,
I/Q injection, Yee staggering, direct DFT reception, and multimode
decomposition. The top-level :ref:`eigenmode-auto-tracking-theory` section
then develops branch assignment, SVD subspace comparison, and basis transport.

Overview
--------

gprMax uses two finite-difference frequency-domain eigenmode solvers:

.. list-table::
   :class: api-parameters
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
corresponding component locations, select the passive effective-index branch,
and return either real-power-normalised or E/H-balanced modal fields. Only the
propagating, real-power-normalised subset can be used by ``EigenmodeSource``.

.. _eigenmode-frequency-symbols:

Frequency and wavenumber convention
-----------------------------------

Both FDFD solvers distinguish the physical solve frequency from the symbols
of the FDTD differences. At each modal solve frequency,

.. math::

   \omega = 2\pi f, \qquad k_0 = \frac{\omega}{c}.

These physical quantities are exposed as ``solver.omega`` and ``solver.k0``.

Given the owning grid's time step :math:`\Delta t`, the eigenproblem and
field reconstruction use the leapfrog temporal symbol

.. math::

   \Omega = \frac{2}{\Delta t}\sin\left(\frac{\omega\Delta t}{2}\right),
   \qquad k_{0,\mathrm{operator}} = \frac{\Omega}{c}.

These are exposed as ``solver.operator_omega`` and ``solver.operator_k0``.
The transverse operators retain their component-sampled Yee differences. The eigenvalue
determines the longitudinal difference symbol :math:`K_w`, rather than the
phase propagation constant :math:`\beta` directly:

.. math::

   \lambda&=-n_{\mathrm{operator}}^2,
   \qquad n_{\mathrm{operator}}=\frac{K_w}{k_{0,\mathrm{operator}}},\\
   \beta&=\frac{2}{\Delta w}\sin^{-1}\left(\frac{K_w\Delta w}{2}\right),
   \qquad n_{\mathrm{eff}}=\frac{\beta}{k_0}.

Here :math:`\Delta w` is the normal cell spacing. The inverse-sine branch is
chosen for passive forward propagation, including decay for evanescent
modes. ``solver.beta`` stores the phase propagation constant. Field
reconstruction uses ``operator_neff``; the public ``complex_neff`` and
``modal_real_neff`` describe its effective index. This distinction keeps the
E/H amplitude relationship consistent with the discrete curls while supplying
the correct phase to source and monitor staggering.

Lossless modes at or beyond the longitudinal spatial band edge
:math:`|K_w\Delta w/2|\geq1` are retained for inspection but have
``power_valid=False`` and cannot be used for source injection.

Eigenmode sources pass the owning grid's time step and normal spacing
automatically, including a subgrid's own values. Direct low-level callers
enable the same convention with ``fdtd_dt`` and ``propagation_spacing``.
Omitting ``fdtd_dt`` uses :math:`\Omega=\omega` and makes ``operator_k0`` equal
to ``k0``; omitting ``propagation_spacing`` uses :math:`\beta=K_w`. Omitting both preserves the
continuum time and longitudinal conventions of earlier low-level calls.
Waveforms, Fourier transforms, and half-time-step phase factors always use
the physical frequency :math:`\omega`.

The source material extraction also includes the midpoint factor in static
electric and magnetic conductivity, including static electric conductivity
in dispersive media. Bulk material poles and their associated Drude or
inclusive conductivity terms are still evaluated from their analytic
physical-frequency response, rather than the exact FDTD ADE transfer.
Dispersive materials therefore retain a time-discretization mismatch in
their pole response that should be checked by convergence.

1D Scalar Solver for 2D Models
------------------------------

``fdfd_1d_mode_solver.py`` supplies the scalar, Yee-staggered mode solve used
by eigenmode sources in gprMax 2D TM and TE domains.

.. _eigenmode-1d-coordinates:

1D Coordinates and Yee Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The local basis is ``(t, a, w)``:

``t``
    The one physical transverse coordinate.

``a``
    The invariant 2D axis.

``w``
    The in-plane propagation direction and source normal.

For ``N`` cells along ``t``, the native staggered shapes are:

.. list-table::
   :class: api-parameters
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
       *,
       fdtd_dt=None,
       propagation_spacing=None,
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
    ``TM`` selects the ``E_a`` scalar problem. ``TE`` selects the ``H_a``
    scalar problem.

``eps_r_*`` and ``mu_r_*``
    Complex relative material arrays sampled at the component locations in
    :ref:`eigenmode-1d-coordinates`.

``pec_*_mask`` and ``pmc_*_mask``
    Optional boolean masks at the matching electric or magnetic component
    locations. Non-finite values in the corresponding material arrays are
    also interpreted as constraints.

``guess``
    Optional ARPACK shift. If omitted, the solver derives a shift from the
    largest finite bulk material magnitude, before SIBC adds surface
    admittance to the electric coefficients. SIBC remains in the eigenproblem.

``fdtd_dt``
    Optional keyword-only FDTD time step in seconds. A positive finite value
    enables the leapfrog frequency symbol and requires ``frequency`` below
    temporal Nyquist. This is distinct from the transverse spatial ``dt``.

``propagation_spacing``
    Optional keyword-only positive finite cell spacing along ``w``, in
    metres. Enables conversion from the longitudinal difference symbol to
    the phase propagation constant. See :ref:`eigenmode-frequency-symbols`.

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

   operator_neff = sqrt(-lambda)

gprMax uses ``exp(+j*omega*t - j*beta*w)``. The square-root branch is
therefore chosen with ``Re(operator_neff) >= 0``. For a passive mode
``Im(operator_neff) <= 0``; a purely evanescent mode uses the negative-imaginary
branch so that it decays in positive ``w``. The public ``complex_neff`` is
then recovered as described in :ref:`eigenmode-frequency-symbols`.

PEC constraints remove electric scalar degrees of freedom from the TM
problem, while PMC constraints remove magnetic scalar degrees of freedom from
the TE problem. Longitudinal inverse-material operators are evaluated only on
unconstrained degrees of freedom: constrained entries receive a zero inverse.
After the reduced sparse eigenproblem is solved, the eigenvectors are expanded
back to their full node or cell arrays and every constrained field component
is explicitly zeroed.

1D Field Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^

For TM, the selected ``E_a`` eigenvector gives:

.. code-block:: text

   H_t = -operator_neff E_a / (eta0 mu_t)
   H_w = i inv(mu_w) D_nc E_a / eta0

For TE, the selected ``H_a`` eigenvector gives:

.. code-block:: text

   E_t = eta0 operator_neff H_a / eps_t
   E_w = -i eta0 inv(eps_w) D_cn H_a

The other three field components are identically zero for the selected 2D
polarization. The reconstructed electric fields are in V/m and magnetic
fields are in A/m.

1D Normalisation and Phase Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each propagating mode is normalised to one watt per metre of invariant-axis
length. A non-propagating mode instead uses the positive E/H-balanced scale
defined in :ref:`eigenmode-power`; it is retained
as a monitor reference but is not a one-watt power wave. For TM, node-sampled
``E_a`` and ``H_t`` are first averaged onto transverse cells:

.. code-block:: text

   P_TM = 0.5 Re sum(-E_a H_t*) dt

For TE, ``E_t`` and ``H_a`` already share the cell locations:

.. code-block:: text

   P_TE = 0.5 Re sum(E_t H_a*) dt

If a passive branch has negative real power along the solver's forward axis,
the propagation-constant branch is reversed and its dependent fields are
reconstructed before normalisation. Each complex mode is then phase-rotated
to a deterministic real-profile convention used for tracking and, for a
propagating source anchor, FDTD injection.

1D gprMax Integration
^^^^^^^^^^^^^^^^^^^^^

``EigenmodeSource`` samples component materials from the mode's live invariant
layer, supplies the corresponding PEC/PMC masks, and maps the returned line
profiles back into the thin 3D Yee arrays used by the FDTD source kernels.
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

.. _eigenmode-2d-coordinates:

2D Coordinates and Yee Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   :class: api-parameters
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
   :class: api-parameters
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
       surface_boundary=None,
       *,
       fdtd_dt=None,
       propagation_spacing=None,
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
    Yee component locations listed in :ref:`eigenmode-2d-coordinates`.

``pec_u_mask``, ``pec_v_mask``, ``pec_w_mask``
    Optional explicit boolean masks for constrained electric degrees of
    freedom. They must match the ``E_u``, ``E_v`` and ``E_w`` shapes.

``pmc_u_mask``, ``pmc_v_mask``, ``pmc_w_mask``
    Optional explicit boolean masks for constrained magnetic degrees of
    freedom. Non-finite entries in the matching permeability arrays are also
    interpreted as PMC.

``guess``
    Optional ARPACK shift. If omitted, the solver chooses a conservative shift
    from the largest finite bulk material magnitude, before SIBC adds surface
    admittance to the electric coefficients. SIBC remains in the eigenproblem.

``surface_boundary``
    Optional compiled impedance-volume boundary. See :doc:`impedance_surfaces`
    for retained-component masks and clipped curl rows.

``fdtd_dt``
    Optional keyword-only FDTD time step in seconds. A positive finite value
    enables the leapfrog frequency symbol and requires ``frequency`` below
    temporal Nyquist.

``propagation_spacing``
    Optional keyword-only positive finite cell spacing along ``w``, in
    metres. Enables conversion from the longitudinal difference symbol to
    the phase propagation constant. See :ref:`eigenmode-frequency-symbols`.

2D Array Ordering
^^^^^^^^^^^^^^^^^

All component arrays are flattened with Fortran order:

.. code-block:: python

   flat = array.ravel(order='F')

and modal vectors are reshaped back with:

.. code-block:: python

   array = vector.reshape((*shape, num_modes), order='F')

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

Port extraction uses the final native Yee electric material IDs, including
samples shared by neighbouring geometry objects. It does not expand
cell-centred PEC voxels into extra electric constraints. Consequently an air
bore carved with averaging disabled is solved with the same electric clamps
as its FDTD update. Averaging and object order can still change the represented
geometry and its cutoff.

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

where the operator index is recovered from:

.. code-block:: python

   operator_neff = sqrt(-eigenvalue)

Here the matrix ``Omega = P * Q`` is distinct from the scalar temporal symbol
:math:`\Omega`. The branch follows ``exp(+j*omega*t - j*beta*w)``:
``Re(operator_neff) >= 0`` and ``Im(operator_neff) <= 0`` for passive forward
propagation. When the real part is zero, the negative-imaginary branch gives
evanescent decay in positive ``w``. The public ``complex_neff`` is then
recovered as described in :ref:`eigenmode-frequency-symbols`.

Because the operators connect the correct staggered Yee component grids, there
is no separate PEC-neighbour spurious-mode rejection heuristic in this solver.
The old candidate scoring/filtering path has been removed.

2D Degree-of-Freedom Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PEC constraints are applied by removing constrained transverse electric degrees
of freedom from the eigenproblem:

.. code-block:: python

   Omega = Omega[self.free_euv_mask, :][:, self.free_euv_mask]

Removing electric samples does not, in general, justify setting their paired
transverse magnetic samples to zero. Write :math:`s=i n_{\mathrm{op}}` and use
the solver's normalized transverse magnetic vector :math:`h`, so that the
unconstrained equations are :math:`s e=P h` and :math:`s h=Q e`. Split magnetic
samples into :math:`a`, whose paired transverse electric update is live, and
:math:`c`, whose paired electric sample is PEC. The latter have no live Ampere
equation, but Faraday's law at the clamped electric sample still requires

.. math::

   P_{ca}h_a+P_{cc}h_c=0,\qquad
   s h_a=Q_a e,\qquad
   s h_c=-P_{cc}^{-1}P_{ca}Q_a e.

Here the first index of :math:`P` selects electric rows (free :math:`f` or
constrained :math:`c`); the second selects magnetic samples. The exact reduced
electric operator is therefore

.. math::

   \left(P_{fa}Q_a-P_{fc}P_{cc}^{-1}P_{ca}Q_a\right)e
   =s^2 e=-n_{\mathrm{op}}^2 e.

For conventional voxel PEC walls the forcing :math:`P_{ca}Q_a` vanishes on the
free electric columns, and the usual sparse product suffices. Otherwise the
solver applies this Schur complement through a sparse factorization of
:math:`P_{cc}`. Shift-invert iteration factors the sparse block system

.. math::

   \begin{pmatrix}
   P_{fa}Q_a-\sigma I & P_{fc}\\
   P_{ca}Q_a & P_{cc}
   \end{pmatrix},

whose upper inverse block is the shifted reduced inverse. No dense Schur
matrix is formed for the sparse eigensolve. The same static reconstruction
supplies the modal H fields and the operator used for tracking residuals.
A singular reconstruction is a numerical failure. Actual PMC constraints and
impedance-volume retained-sample masks continue to exclude their own magnetic
degrees of freedom.

When the reduced matrix has only one more degree of freedom than the requested
mode count, the solver uses a dense eigensolve because ARPACK requires
``k < N - 1``. Larger systems use shift-invert ARPACK and retry with a
roundoff-scale shift perturbation if the original shift produces a singular
factorisation. The reduced eigenvectors are then expanded back to the full
transverse field-vector size and constrained fields are explicitly zeroed.

The inverse ``eps_r_ww`` and ``mu_r_ww`` operators are built only on free
longitudinal degrees of freedom. Constrained entries receive zero inverse
values so that no division by ``np.inf`` or placeholder data affects the
reconstructed fields.

2D Field Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^

After solving the eigenproblem, the solver reconstructs:

* ``E_u`` and ``E_v`` directly from the transverse eigenvector.
* ``H_u`` and ``H_v`` from ``Q * Euv / sqrt(eigenvalue)``, using the branch
  ``sqrt(eigenvalue) = +j*operator_neff`` selected by the propagation convention.
* ``E_w`` from transverse magnetic curl terms.
* ``H_w`` from transverse electric curl terms.

Magnetic fields are converted to physical A/m using ``eta0``:

.. code-block:: python

   H = 1j * H_normalized / eta0

The solver then zeroes all constrained fields to ensure returned modal fields
satisfy the enforced constraints exactly.

2D Normalisation and Phase Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Propagating modes are normalised to carry one watt of time-average power.
Non-propagating modes instead use the finite E/H-balanced scale defined in
:ref:`eigenmode-power`; they can enter the tracked
monitor-reference bank but not the source/power bank. Real power is computed
from cell-centred transverse Poynting flux by averaging the staggered
transverse fields onto local cells:

.. code-block:: text

   P = 0.5 * Re integral((E_u * H_v* - E_v * H_u*) dA)

If a passive branch carries negative real power along the solver's forward
axis, the propagation-constant branch is reversed and all dependent fields
are reconstructed before normalisation. Each complex mode is then
phase-rotated to a deterministic real-profile convention. This makes plotted,
tracked, and (for propagating source anchors) injected fields easier to
compare.

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

PMC constraints, like PEC constraints, follow the final native Yee component
material IDs. Cell-centred PMC voxels are not expanded into additional magnetic
constraints, including when reducing the port cross-section to a 1D solve.

After solving, ``sources.py`` maps local modal fields back to global component
slots. The Cython injection kernels consume the transverse components with
their native staggered shapes; longitudinal modal fields are stored but are not
used for TF/SF source corrections.

Source synthesis and FDTD injection
-----------------------------------

This section connects the FDFD solvers described above to the complete
eigenmode band/port/excitation workflow. It fixes the phasor and propagation signs,
shows how a solved mode becomes real FDTD update terms, and describes the
single-frequency, in-phase/quadrature (I/Q), and broadband synthesis paths.

.. _eigenmode-conventions:

Phasor, Fourier, and Propagation Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

gprMax advances real fields in time. Whenever a complex frequency-domain
field is used, it follows the electrical-engineering convention

.. math::

   \mathbf{E}(u,v,w,t)
   = \operatorname{Re}\left\{
       \widetilde{\mathbf{E}}(u,v,\omega)
       \exp(+j\omega t-j\beta w)
     \right\}.

The forward Fourier transform uses the complementary negative-sign kernel,

.. math::

   X(\omega)=\int_{-\infty}^{\infty}x(t)\exp(-j\omega t)\,dt,

so NumPy's ``rfft`` returns the positive-frequency coefficients used by this
phasor convention, and ``irfft`` reconstructs the real signal with positive
frequency time dependence ``exp(+j*omega*t)``. The FFT sign is not a second
physical convention: it is the analysis kernel paired with the phasor
synthesis sign.

With ``exp(+j*omega*t)``, the continuum Maxwell curl equations are

.. math::

   \nabla\times\widetilde{\mathbf{E}}
   &= -j\omega\widetilde{\boldsymbol{\mu}}
      \widetilde{\mathbf{H}},\\
   \nabla\times\widetilde{\mathbf{H}}
   &= +j\omega\widetilde{\boldsymbol{\epsilon}}_c
      \widetilde{\mathbf{E}}.

Their continuum electric and magnetic conductivity terms are

.. math::

   \epsilon_{r,c}(\omega)
   &= \epsilon_r(\omega)
      -j\frac{\sigma}{\omega\epsilon_0},\\
   \mu_{r,c}(\omega)
   &= \mu_r(\omega)
      -j\frac{\sigma_m}{\omega\mu_0}.

For nondispersive material extraction on an FDTD grid, the solver instead
uses the exact midpoint conductivity terms

.. math::

   \epsilon_{r,c}^{\mathrm{Yee}}
   &= \epsilon_r-j\frac{\sigma\cos(\omega\Delta t/2)}{\Omega\epsilon_0},\\
   \mu_{r,c}^{\mathrm{Yee}}
   &= \mu_r-j\frac{\sigma_m\cos(\omega\Delta t/2)}{\Omega\mu_0},

with the temporal symbol from :ref:`eigenmode-frequency-symbols`.
Low-level callers supply their own complex relative material arrays; passing
``fdtd_dt`` does not reinterpret those arrays as conductivity parameters.

For a forward passive mode,

.. math::

   \beta=k_0n_{\mathrm{eff}}=\beta_r-j\alpha,
   \qquad \alpha\geq 0,

and therefore

.. math::

   \exp(-j\beta w)
   =\exp(-j\beta_r w)\exp(-\alpha w).

The selected propagation branch consequently satisfies

.. math::

   \operatorname{Re}(n_{\mathrm{eff}})&\geq 0,\\
   \operatorname{Im}(n_{\mathrm{eff}})&\leq 0
   \quad\text{for passive propagation}.

If the real part is numerically zero, a purely evanescent mode uses
``Im(n_eff) < 0`` so it decays in positive local ``w``. The imaginary part of
the square root is never replaced by its absolute value; its sign contains
the loss or gain information.

As a continuum example, at 5 GHz a homogeneous material with
``epsilon_r=9`` and ``sigma=2 S/m`` has

.. math::

   \epsilon_{r,c}\simeq 9-j7.190,
   \qquad
   n\simeq 3.203-j1.122.

This gives :math:`\alpha=-k_0\operatorname{Im}(n)\simeq
117.6\ \mathrm{m}^{-1}`. The field magnitude after 0.5 mm is approximately
``exp(-alpha * 0.5e-3) = 0.943``.

Build-Time Workflow
^^^^^^^^^^^^^^^^^^^

The source is prepared after geometry construction, when material IDs already
refer to their final Yee-component locations. The workflow is:

1. Validate the source plane, requested direction, mode index, waveform, and
   one or more solve frequencies.
2. Choose local coordinates ``(u, v, w)``. The ``u`` and ``v`` axes lie in the
   source plane and ``w`` is normal to it. In a 2D TM or TE model, one of the
   transverse axes is the invariant axis and only one is physical.
3. Extract complex relative permittivity and permeability at every native Yee
   component position on the source plane. PEC and PMC cells become explicit
   component constraints.
4. At each requested frequency, solve either the 1D scalar TM/TE problem for a
   2D FDTD model or the 2D full-vector problem for a 3D FDTD model, using the
   owning grid's time step and normal cell spacing.
5. Reconstruct all modal E and H components, zero constrained degrees of
   freedom, apply either real-power or balanced E/H normalization, and choose
   a consistent global phase.
6. Map local modal arrays back to global x/y/z component slots. If the local
   coordinate mapping is left-handed, reverse H so that the Poynting direction
   remains correct.
7. Select real-only, single-frequency I/Q, or multi-anchor broadband
   synthesis.
8. During time stepping, apply tangential incident E and H as TF/SF
   corrections on the appropriate side of the source plane.

The modal solution describes the cross-section at the source reference plane.
Propagation away from that plane is performed by the FDTD grid itself. The
explicit propagation constant is used in the broadband E/H staggering and in
frequency interpolation, not to overwrite fields throughout the guide.

.. _eigenmode-power:

Mode Selection, Fields, and Power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The user-facing ``mode_index`` is one-based: mode 1 selects the first eigenpair
after the solver orders the computed eigenvalues. The low-level Python FDFD
solver classes retain zero-based array indices internally. Shift-invert sparse
eigensolving targets modes near a material-derived guess. Near cut-off,
degeneracy, or an eigenvalue crossing, the same numerical index can identify a
different physical mode at another frequency; broadband sources and receivers
therefore check adjacent-anchor overlap.

For a 3D model, transverse modal power is evaluated on local cells as

.. math::

   P=\frac{1}{2}\operatorname{Re}
   \sum_{u,v}
   \left(E_uH_v^*-E_vH_u^*\right)\Delta u\Delta v.

The native Yee components are averaged only as needed to place each product on
the same transverse cell. For a 2D model, the equivalent line integral gives
power per metre along the invariant axis. If a passive branch initially has
negative real power along the solver's forward axis, gprMax reverses its
propagation-constant branch and reconstructs the dependent fields. A
propagating mode is then scaled to one watt in 3D or one watt per metre in 2D.
A non-propagating mode has no independent forward real power and instead
receives a finite balanced E/H field scale for tracking and diagnostics; that
scale is not a one-watt normalization.

This normalization defines the scale of the modal profile at the source
plane. Multiplying the source waveform amplitude by a factor multiplies both
incident E and H by that factor; for a monochromatic mode, time-average power
therefore scales with the square of the waveform amplitude only where the
mode has valid real-power normalization.

The one-watt propagating profiles form the source-synthesis and real-power
bank. Modal monitoring also retains a second, tracked reference bank. Before
a reference profile is used in a generalized-only bin, both its E and H
fields are multiplied by the same factor so that

.. math::

   P_\mathrm{bal}=\frac{1}{4\eta_0}\int
   \left(|\mathbf E_t|^2+\eta_0^2|\mathbf H_t|^2\right)\,dA=1

(or the corresponding 2D line integral). This positive balanced quantity is
a field-scale convention, not transported real power. Applying it to both
propagating and evanescent monitor references prevents their original solver
scales from introducing an artificial coefficient-scale jump. It does not
license interpolating through the cutoff singularity: propagating and
evanescent references remain in separate interpolation branches.

The solver returns the positive-local-``w`` mode. A source requested in the
negative global direction retains the electric profile and reverses the
magnetic orientation in the TF/SF updates, as required for the opposite
Poynting vector.

Global Phase and the Real-Only Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An eigenvector has arbitrary complex phase. Before deciding how to inject a
single-frequency profile, gprMax finds the phase rotation that makes all
*tangential* E and impedance-scaled H components as real as possible. If
:math:`F_q` denotes every such component sample, with H samples first scaled
by :math:`\eta_0`, the rotation is

.. math::

   \phi=-\frac{1}{2}\arg\left(\sum_q F_q^2\right),
   \qquad F_q' = F_q\exp(j\phi).

Only tangential components enter this test because longitudinal mode
components can be intrinsically in quadrature while not participating in the
TF/SF correction. After rotation, the normalized imaginary residual is

.. math::

   r=\sqrt{
     \frac{\sum_q\left[\operatorname{Im}(F_q')\right]^2}
          {\sum_q\left|F_q'\right|^2}
   }.

If ``r <= 1e-8``, the tangential spatial profiles are effectively real. gprMax
stores their real parts and multiplies them directly by the requested real
waveform. The H waveform is evaluated with its Yee half-time-step and
half-normal-cell phase delay. This path avoids unnecessary FFT preparation for
ordinary lossless fixed-profile modes.

Why Complex Modes Need I/Q Injection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single real scalar waveform cannot reproduce a general complex spatial
profile. Let

.. math::

   F=F_R+jF_I,
   \qquad Z(t)=\operatorname{irfft}\{C(\omega)\}.

The desired real field is

.. math::

   \operatorname{Re}\{F C(\omega)\exp(j\omega t)\}.

gprMax realizes this with two real bases:

.. math::

   F_R\operatorname{irfft}\{C\}
   +F_I\operatorname{irfft}\{jC\}.

Because ``irfft(j*C)`` is the negative quadrature of ``irfft(C)``, this sum is
exactly the required real part. No complex values are passed to the real FDTD
update arrays.

When the single-frequency residual exceeds the tolerance, gprMax uses this
I/Q construction with one modal anchor. The same solved profile and
``n_eff`` are used for every significant FFT bin of the waveform. This is a
fixed-profile approximation: use multiple solve frequencies when the modal
shape or propagation constant varies appreciably across the waveform
bandwidth.

Spectrum and Piecewise-Linear Modal Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the automatic waveform, gprMax first forms the requested smooth spectral
magnitude and finds the circular time support of its zero-phase analytic
envelope. It then applies the earliest discrete delay that places all envelope
samples above the significance threshold inside the causal record. This keeps
the requested spectrum while reserving as much of the simulation window as
possible for the packet to traverse the model and decay.

The real waveform is sampled at the FDTD interval for the requested number of
iterations. It is zero-padded to the next power of two at least twice as long
as the simulation record, and transformed with ``numpy.fft.rfft``. For FFT
bin :math:`m`, let its frequency and coefficient be :math:`f_m` and :math:`S_m`.
Non-finite samples and a zero or non-finite spectrum are errors; gprMax does
not replace the excitation with a zero-valued source.

For source synthesis, piecewise-linear weights :math:`w_{k,m}` interpolate
between surrounding propagating source anchors and satisfy

.. math::

   \sum_k w_{k,m}=1.

Below or above the anchor range, the nearest endpoint receives weight one.
This avoids a hard spectral truncation. Significant waveform energy outside
the anchor range is governed by the spectrum-coverage policy: the default is
an error, while an explicit ``warn`` policy permits endpoint extrapolation
with a warning. The source-interpolated fields and propagation constant are

.. math::

   \mathbf{E}_m &= \sum_k w_{k,m}\mathbf{E}_k,\\
   \mathbf{H}_m &= \sum_k w_{k,m}\mathbf{H}_k,\\
   n_{\mathrm{operator},m} &= \sum_k w_{k,m}n_{\mathrm{operator},k},\\
   K_m &= \frac{\Omega(f_m)}{c}n_{\mathrm{operator},m},\\
   \beta_m &= \frac{2}{\Delta w}\sin^{-1}\left(\frac{K_m\Delta w}{2}\right).

The inverse spatial difference is evaluated at each FFT bin after operator
index interpolation, including endpoint extrapolation. Single-frequency I/Q
sources, banks with only one retained anchor, and downstream solvers without
operator-index metadata retain the constant/legacy physical-index convention
:math:`\beta_m=2\pi f_m n_m/c`. In particular, a single-frequency source holds
its physical phase index constant across the pulse.

Significant source energy in the longitudinal grid stop band is an error:
refine the normal cell spacing or narrow the excitation bandwidth.
Sub-threshold tails in that stop band are discarded; the scalar waveform
reconstruction error includes their removal. DC and Nyquist bins are excluded
before this propagation calculation.

Linear interpolation of individually normalized modes does not in general
preserve unit power. gprMax constructs the cross-power matrix

.. math::

   P_{kl}=\frac{1}{2}\int
   \left(\mathbf{E}_k\times\mathbf{H}_l^*\right)
   \cdot\hat{\mathbf{w}}\,dA

(or the corresponding 2D line integral), and calculates

.. math::

   p_m=\operatorname{Re}
   \left\{\sum_{k,l}w_{k,m}P_{kl}w_{l,m}\right\},
   \qquad a_m=\frac{1}{\sqrt{p_m}}.

Thus the interpolated frequency-bin source mode is renormalized rather than
assuming that linear field weights retain one-watt power. Invalid or nearly
zero interpolated power is an error because a finite fallback would no longer
represent the requested one-watt incident wave.

Yee Time and Space Staggering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Electric and magnetic fields are offset by half a time step and their
tangential samples at the TF/SF surface are offset by half a normal cell. For
each bin, gprMax applies the magnetic staggering factor

.. math::

   M_m=\exp\left[j\left(
       \frac{\omega_m\Delta t}{2}
       +\frac{\beta_m\Delta w}{2}
   \right)\right].

Here :math:`\omega_m=2\pi f_m` is the physical angular frequency, and
:math:`\beta_m` is recovered from the longitudinal difference symbol. The
temporal difference symbol :math:`\Omega` is not a phase frequency.

The local coordinate is defined in the requested propagation direction. The
tangential H correction lies half a cell on the incident side of the electric
reference plane, giving the positive spatial phase in this relative factor.
For a lossy mode this local factor can have magnitude greater than one because
the incident-side sample precedes the reference plane; this does not represent
growth in the forward direction. Forward propagation over a positive distance
``d`` is always tested by ``exp(-j*beta*d)``, whose magnitude is below one for
a passive mode.

The spectral coefficients assigned to anchor ``k`` are therefore

.. math::

   C^E_{k,m} &= w_{k,m}S_m a_m,\\
   C^H_{k,m} &= w_{k,m}S_m a_m M_m.

Each anchor field is split into real and imaginary arrays, and each coefficient
set is inverse-transformed both normally and after multiplication by ``j``.
The FDTD source update then sums the two I/Q bases over all anchors.

DC and, for an even transform length, the Nyquist bin are self-conjugate. They
cannot carry a general complex modal coefficient while preserving a real time
record, so gprMax discards those two bins. Significant DC or Nyquist energy
produces a warning that the requested excitation has been changed. Use a
band-limited waveform; for a finite frequency band,
``EigenmodeExcitation(..., waveform='auto')`` synthesizes one automatically.

TF/SF Injection into the FDTD Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source plane is a total-field/scattered-field interface. In continuous
form, the incident mode corresponds to equivalent surface currents

.. math::

   \mathbf{J}_s &= \hat{\mathbf{n}}\times\mathbf{H}_{\mathrm{inc}},\\
   \mathbf{M}_s &= -\hat{\mathbf{n}}\times\mathbf{E}_{\mathrm{inc}}.

The discrete implementation applies the equivalent correction directly to
the curl updates adjacent to the plane:

* the magnetic-field update uses the incident tangential E profile;
* the electric-field update uses the incident tangential H profile;
* the side of the plane and correction sign depend on ``+`` or ``-`` source
  direction;
* only transverse/tangential components are injected. Longitudinal fields are
  reconstructed and retained for diagnostics but do not enter these TF/SF
  corrections.

Because modal fields are stored on their native Yee component grids, the
source does not resample every component onto a common rectangle. The update
kernels consume the component-specific array shapes documented earlier in
this page. This preserves PEC/PMC constraints, tangential staggering, and the
mode solver's discrete curl relationships.

For real-only sources, one real modal array per component is multiplied by the
waveform value. For I/Q and broadband sources, each update sums all anchor and
quadrature contributions prepared by the inverse FFT. In both cases, source
activation is clipped to the configured waveform start and stop times.

Single-frequency sources record ``single_frequency_iq_reasons`` and log the
individual selection reasons alongside the measured modal-profile residual:
``complex modal profile``, ``drive phase/delay``, and/or
``complex longitudinal staggering``. A small modal residual can therefore
coexist with I/Q injection. Negative real propagation is handled by the
signed real-only time shift and does not itself require I/Q. Eigenmode
solvers, source staggering, and monitor propagation use the active simulation's
``em_consts["c"]`` for the speed of light.

Reduced-mode surface impedance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 1D solver accepts an optional ``surface_boundary`` in its local
``(t, a, w)`` basis. gprMax projects this from the compiled SIBC geometry,
retaining only the live invariant layer. The TM electric scalar and TE
longitudinal electric rows use the clipped Ampere derivative and the exact
discrete surface-ADE admittance. Faraday retains its ordinary Yee derivative;
the two derivatives need not be negative adjoints at a clipped wall.
``resistance=float('inf')`` gives exactly zero surface admittance.

CPU ``VirtualWaveguide`` supports all 2D TE/TM orientations. Its modal window
spans the full invariant storage dimension. Its artificial PEC rim constrains
tangential E only at the ends of the physical transverse coordinate, including
where a cropped SIBC row needs a magnetic sample outside the window. All
ports constrain complete SIBC rows on that rim to PEC too, whether or not a
virtual guide is attached. This matches the auxiliary aperture and PML
updates. Opaque padding moves a physical wall inside the window and retains
its SIBC equation.
The guide and its retained host must remain uniform along propagation through
the aperture and PML. Both
ordinary and virtual sources apply the surface-row modal forcing and ADE
correction. See :ref:`sibc-pml` for setup and :ref:`impedance-pml-theory` for validation.

.. _eigenmode-virtual-coupling:

Virtual-guide aperture coupling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a virtual guide the TF/SF incident-mode correction is applied on an
internal plane of the auxiliary grid rather than the main grid. Let
:math:`w` be the port-normal coordinate and :math:`u,v` the transverse
coordinates. At the main/auxiliary split the normal magnetic sample is shared,

.. math::

   H_w^{\mathrm{aux}}\big|_{\Gamma}
   = H_w^{\mathrm{main}}\big|_{\Gamma},

and the aperture updates for :math:`E_u` and :math:`E_v` use the ordinary Yee
curl with the two normal-neighbour magnetic samples taken from different
grids. In schematic form,

.. math::

   E_u^{n+1}\big|_{\Gamma}
   &= C_A E_u^n\big|_{\Gamma}
      + C_v\,\Delta_v H_w
      - C_w\left(H_v^{\mathrm{aux}}-H_v^{\mathrm{main}}\right),\\
   E_v^{n+1}\big|_{\Gamma}
   &= C_A E_v^n\big|_{\Gamma}
      + C_w\left(H_u^{\mathrm{aux}}-H_u^{\mathrm{main}}\right)
      - C_u\,\Delta_u H_w.

The signs reverse consistently for the opposite port direction. The updated
tangential E samples are shared with the main-grid aperture, while the
duplicate main-grid continuation behind the aperture is disconnected. The
3D axis/direction variants use compiled Cython kernels. Reduced 2D coupling
uses vectorized operations on the live field layer, preserving its native
staggering without introducing invariant-axis side walls.

On the non-distributed CPU solver, bulk electric dispersion is supported in
both 3D and reduced TE/TM virtual guides. Conductivity and instantaneous
polarization storage are included in the electric update coefficients. For
each dispersive aperture sample, the curl result above is completed by

.. math::

   \Phi^n &= \sum_m \operatorname{Re}(a_m T_m^n),\\
   E^{n+1} &= E_{\mathrm{curl}}^{n+1} - C_\Phi\Phi^n,\\
   T_m^{n+1} &= f_m T_m^n + b_m(E^n-E^{n+1}).

These are the same real or complex bulk ADE coefficients used inside the
grid. The auxiliary boundary owns this history because the ordinary bulk
kernels omit tangential E on its outer plane. The final corrected field is
shared with the main grid. For SIBC contact rows, the auxiliary guide instead
copies the complete sparse constitutive equation, including every retained
bulk pole and independent Foster history, and uses the coupled sparse solve.
This also applies inside its longitudinal PML. Dispersive aperture coupling
on accelerator and MPI backends remains unsupported.

.. _eigenmode-measurement-theory:

Modal Receivers, Direct DFT, and S-parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exactly one global band and zero or more distinct port/mode excitations may
exist whenever modal ports are used. Excitation may be omitted only when every
port is a passive virtual guide; that form writes raw modal spectra but no S
matrix. Every ``EigenmodePort`` owns one monitor at its reference plane; each
selected excitation additionally applies a TF/SF source. Port indices are
one-based and unique, and each port carries an explicit tuple of monitored
mode indices. Every excitation selects one of the modes listed by its port.
All ports accumulate the common DFT bins from ``EigenmodeBand``. Exactly one
active channel permits S-parameter normalization; a simultaneous driven state
retains its decomposed waves without constructing S.
Automatic ports share one candidate-frequency list, but tracking, retained
source/reference masks, and fallback policies are resolved independently for
each port and mode. Ports with explicit anchors retain their individually
requested candidate frequencies.
For each requested frequency :math:`f_q`, a Cython kernel applies the
recursive DFT

.. math::

   X_q^{(n+1)}=X_q^{(n)}
     +\Delta t\,x_n\exp(-j2\pi f_q n\Delta t).

The phase factor is advanced once per bin and time step. To bound complex64
recurrence drift in long simulations, both oscillators are reconstructed from
their physical times every 1024 iterations using float64 argument reduction;
the HDF5 ``PhaseReanchorInterval`` attribute records this interval. Electric
fields use integer FDTD times, while magnetic fields use their half-time-step
phase.
Transverse Yee components are averaged to common cells only inside the
projection. A passive receiver samples the magnetic reference plane half a
normal cell upstream of the electric plane. At a TF/SF source, H is sampled
half a cell downstream so that both fields are on the total-field side. The
final decomposition corrects either offset using each mode's propagation
constant.

The DFT-bin monitor basis is selected independently from source synthesis. A
power-wave-valid bin uses the interpolated one-watt propagating bank. A
generalized-only bin uses ``anchor_mode_reference_valid`` instead. It first
selects the applicable contiguous evanescent reference run inside the solved
candidate range; interpolation never crosses cutoff or spans disconnected
evanescent runs. Outside that range, it uses the nearest tracked reference
endpoint. Every selected anchor E/H pair is divided by the square root of its
``anchor_balanced_power``, then E, H, and the operator index are
interpolated with identical branch-local weights. The propagation constant is
mapped at each DFT frequency using the same discrete symbols as source
synthesis; branches retaining only one anchor keep their physical phase
index constant. The interpolated
cell-centred E/H pair is balanced once more before its Gram matrices are
formed. Keeping all three quantities on the same tracked branch is essential
below cutoff, where modal admittance becomes reactive; interpolating only the
propagation constant while retaining a propagating endpoint E/H pair would
not correctly separate forward and backward amplitudes at one plane.

A bin that enters the numerical spatial stop band after interpolation loses
power-wave validity. It retains its selected propagating-bank weights,
fields, and propagation constant, with final cell-centred balanced
normalization for generalized coefficients. It does not select a different
physical-cutoff branch. Existing conditioning and separation checks still
determine whether those coefficients can be reported.

The HDF5 port group preserves ``anchor_complex_neff`` as the physical phase
index. When available, ``anchor_operator_neff`` stores the corresponding
dimensionless operator indices with the same anchor/mode axes. ``beta``
records the actual propagation constant used at every DFT bin, in radians
per metre, with frequency/mode axes. These datasets also describe runs that
reuse cached modal anchors.

For several requested modes, independent overlaps are insufficient when the
discrete profiles are not exactly orthogonal. At each frequency gprMax forms
electric and magnetic Gram matrices,

.. math::

   G^E_{mn}
     &= \frac{1}{2}\int
        (\mathbf E_n\times\mathbf H_m^*)\cdot\hat{\mathbf w}\,dA,\\
   G^H_{mn}
     &= \frac{1}{2}\int
        (\mathbf E_m^*\times\mathbf H_n)\cdot\hat{\mathbf w}\,dA,

and solves both systems for the total electric and magnetic modal
coefficients. If :math:`x=a+b` is the electric coefficient and the staggered
magnetic coefficient is
:math:`y=p_+a-p_-b`, where :math:`p_+` and :math:`p_-` are the forward
and backward half-cell phase factors, then

.. math::

   a=\frac{y+p_-x}{p_++p_-},\qquad b=x-a.

Here :math:`a` travels in the receiver's declared direction and :math:`b`
travels in the opposite direction. With port directions defined into the
device, the single-source scattering result is

.. math::

   S_{j m,\,1 n}(f)=\frac{b_{j,m}(f)}{a_{1,n}(f)}.

Consequently the explicitly numbered source port gives S11 and a downstream
multimode port gives one S21 result for every requested destination mode.
These are generalized modal-amplitude ratios. ``reference_basis_valid`` records
only pre-solve reference eligibility. ``coefficient_valid`` then marks each
coefficient that survives both the electric and magnetic conditioned solves,
finite half-cell phase reconstruction, and finite-value checks.
``coefficient_valid_S`` additionally requires a usable source coefficient and
a -60 dB incident-spectrum floor evaluated separately for power-wave and
generalized-only source bins. Bins remain present in the arrays, but unusable
S entries are NaN. ``power_wave_valid`` further requires destination-mode
power-wave support and a valid destination power matrix. ``power_wave_valid_S`` includes
those destination gates and additionally requires the launched source mode and
its power matrix to be physically valid.

For each electric or magnetic Gram matrix, a singular value is retained only
when

.. math::

   \sigma_i > \max\left(
      \frac{\epsilon}{10^{-3}},
      \frac{\sigma_{\max}}{\kappa_{\mathrm{lim}}}
   \right),
   \qquad
   \kappa_{\mathrm{lim}}=\min\left(10^{10},\frac{10^{-3}}{\epsilon}\right),

where :math:`\epsilon` is the precision of the stored field components. If all
singular values pass, gprMax solves the complete system directly. If not, a
truncated full-system SVD fallback is considered only when the active set
contains both power-wave and generalized-only coordinates. The discarded
right-singular subspace must have Frobenius projection no larger than
:math:`10^{-3}` onto the power-wave coordinates; otherwise the complete solve
is rejected. Among accepted fallbacks, only coordinates whose ambiguity in
the discarded subspace is no larger than :math:`10^{-3}` in *both* the
electric and magnetic solves become ``coefficient_valid``. This can preserve a
power-wave coordinate when only a generalized-only coordinate is singular,
but it rejects a nullspace that mixes physical modes.

``condition_number`` reports the larger electric/magnetic full-system
condition number for a direct solve. For a successful truncated fallback it
instead reports the larger condition number of the two retained singular
subspaces, not the singular original system; it is infinite when no
coefficient survives. The small Gram systems and SVD are evaluated in
complex128 even when the stored FDTD fields and Gram entries use complex64.

A below-cutoff mode can therefore produce finite incident/outgoing
coefficients and a continuous S21 amplitude. In a uniform guide its forward
amplitude varies as :math:`\exp(-\alpha L)` for
:math:`\beta=-j\alpha`; this is the attenuation of a field coefficient, not
the transport of real power by an isolated evanescent wave. At exact cutoff,
:math:`\beta=0` and the true forward/backward eigenmode basis coalesces. An
E/H-balanced reference may approach a finite limiting coefficient there, but
the directional decomposition is not unique and must be treated as
conditioning-sensitive.

The same Gram matrices define the Hermitian forward-wave power form

.. math::

   W=\frac{1}{2}\left(G^E+G^H\right),\qquad
   P(c)=\operatorname{Re}\{c^\mathrm{H}Wc\}.

The implementation symmetrizes :math:`W` against round-off and checks that it
is finite and positive semidefinite. Keeping the off-diagonal terms is
essential for degenerate, nearly degenerate, or merely non-orthogonal
finite-grid profiles. Individual coefficient magnitudes therefore are not
additive modal powers. A pure evanescent mode has zero independent real power,
so ``coefficient_valid_S`` may be true while
``power_basis_valid`` and ``power_wave_valid_S`` are
false. In particular, :math:`|S|^2` must not be used as an evanescent power
fraction.

For net accepted power, let :math:`x=a+b` be the total electric coefficient
and :math:`y=a-b` the co-located total magnetic coefficient after the
half-cell correction. The direct time-average flux is

.. math::

   P_{\mathrm{acc},p}
     =\operatorname{Re}\{y_p^\mathrm{H}G^E_p x_p\}.

This reduces to :math:`P(a_p)-P(b_p)` when :math:`G^E_p` is Hermitian.
For a lossy port, its anti-Hermitian part supplies a forward/backward
interference term which must be retained.

At each frequency this quadratic form is evaluated on the valid propagating
mode subspace. All off-diagonal terms within that subspace are retained, but
generalized-only rows and columns are excluded. An invalid propagating mode
invalidates the accepted-power result rather than being silently omitted.

For active port :math:`p`, let :math:`D_p` select its explicitly driven modes.
The externally driven incident power is

.. math::

   P_{\mathrm{inc}}
     = \sum_{p\,\mathrm{active}}
       \operatorname{Re}\!\left\{
       a_{p,D_p}^{\mathrm{H}}W_{p,D_pD_p}a_{p,D_p}\right\}.

Thus a one-channel run retains the previous single-mode definition, while a
simultaneous multimode run keeps the cross terms between non-orthogonal driven
modes on the same physical port. Passive modal receivers have zero generator
incident power, but their signed accepted power remains in the multiport
balance used for gain. This distinction makes realized gain use launched
source power while gain uses the net power accepted by the radiating
structure. The power adapter applies the power-normalization and power-matrix
masks, so generalized below-cutoff coefficients do not enter gain,
accepted-power, or energy-balance normalization.

Understanding Lossy-Mode Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a passive lossy mode, users should expect:

* positive ``Re(n_eff)`` for the selected forward phase branch;
* negative ``Im(n_eff)``;
* positive modal power in the requested source direction;
* a forward magnitude factor
  ``abs(exp(-j*k0*n_eff*d)) < 1`` for ``d > 0``;
* a nonzero complex-profile residual in general, causing I/Q injection;
* downstream attenuation to be produced by the FDTD material updates after
  the mode has been launched.

The port-mode field plots show tangential E and H vectors for every retained
anchor and report the complex effective index. E and H magnitudes are
normalised independently in their panels. A nonzero receiver field alone does
not validate a lossy mode: sign-sensitive validation must also check
``Im(n_eff)``, modal power direction, and forward attenuation.

Accuracy and convergence
^^^^^^^^^^^^^^^^^^^^^^^^

The practical accuracy checklist is in :doc:`eigenmode_port`. Material
dispersion is sampled at each anchor; interpolation between anchors remains
piecewise linear. A frequency-dependent bulk pole response is not generally
identical to the FDTD ADE transfer. Check convergence in time and space, as
well as anchor spacing, rather than treating modal overlap as a sufficient
accuracy certificate. Significant DC and Nyquist waveform bins are warned
about and discarded.

Complete matrices, active reflection, and antenna normalization
---------------------------------------------------------------

For independent cases, assemble the measured incident and outgoing columns
into :math:`A(f)` and :math:`B(f)`. The complete matrix satisfies

.. math::

   B = S A.

Solving this system accounts for measured incident waves at nominally passive
channels. A missing or ill-conditioned incident basis invalidates the affected
matrix bins. For reciprocal linear, time-invariant materials and consistent
modal normalization, phase conventions, and reference planes,
:math:`S_{ij}=S_{ji}`. Reciprocity alone does not require
:math:`S_{11}=S_{22}`; that equality in example 4 additionally follows from
the device's symmetry.

A simultaneous drive q has spectral multiplier

.. math::

   w_q(f) = A_q\exp(j\phi_q)\exp(-j2\pi f\tau_q).

The API takes :math:`\phi_q` in degrees and :math:`\tau_q` in seconds.
For a coherent incident vector :math:`a`, active reflection is

.. math::

   \Gamma_{\mathrm{active},q} = \frac{b_q}{a_q}
       = \frac{\sum_j S_{qj}a_j}{a_q}.

Its coefficient and power-wave masks also require an adequately excited
denominator. It depends on the incident vector, so it cannot replace the
independent cases used to determine :math:`S`.

For the array along y in example 5, spacing d and constant progressive phase
:math:`\Delta\phi` give the ideal forward xy-plane maximum

.. math::

   \sin\phi_{\mathrm{peak}}(f)
       = -\frac{\Delta\phi}{k(f)d},\qquad k(f)=\frac{2\pi f}{c}.

The phase increment is in radians in this equation. Positive observation
angle is measured from +x toward +y. For d = 18 mm and an increment of -108
degrees, the predicted angles at 8, 9, 10, 11, and 12 GHz are approximately
38.7, 33.7, 30.0, 27.0, and 24.6 degrees. These ideal array-factor angles do
not include the aperture element pattern or mutual coupling.

For radiation intensity U, total radiated power :math:`P_{\mathrm{rad}}`,
accepted power :math:`P_{\mathrm{acc}}`, and externally driven incident power
:math:`P_{\mathrm{inc}}`, the antenna quantities are

.. math::

   D = \frac{4\pi U}{P_{\mathrm{rad}}},\qquad
   G = \frac{4\pi U}{P_{\mathrm{acc}}},\qquad
   G_{\mathrm{realized}} = \frac{4\pi U}{P_{\mathrm{inc}}}.

The power adapter uses the physical power-wave subspace and its full power
matrix. Finite below-cutoff coefficients do not enter gain or energy-balance
normalization. The ``port_power/modal_ports`` output repeats the modal waves,
mode indices, power matrix, and physical masks at the NTFF frequencies so
that the normalization can be audited.


Limitations
-----------

Automatic mode tracking is experimental and requires further testing. Enable
it with ``tracking="auto"`` at your discretion; ``tracking="legacy"`` remains
the default. The additional matching and diagnostic evidence does not remove
the following numerical and physical limitations.

* Material tensors are diagonal in the local ``u``/``v``/``w`` basis.
* The finite-difference operators use first-order sparse Yee-grid differences.
* Bulk dispersive material poles use their analytic physical-frequency
  response. Matching their exact FDTD ADE transfer remains a separate step
  beyond the temporal and longitudinal difference compensation.
* A single-frequency source reuses one solved profile across the waveform
  spectrum. Broadband interpolation remains sensitive to anchor spacing even
  when tracking confidently identifies the same branch. It does not certify
  modal-impedance accuracy or a particular S11 floor.
* Automatic tracking can follow changes in eigenvalue order and transport
  detected degenerate subspaces. Extra candidates and adaptive frequency
  solves cannot guarantee a unique match in every guide. Unresolved identity
  gaps invoke the automatic-anchor fallback or an explicit-anchor error;
  interpolation never bridges them. Resolved eigenvalue splitting retains
  separate eigenmodes and propagation constants, rather than mixing the fields.
* Confinement and artificial-boundary diagnostics are evidence, not proofs
  that a mode is physical or spurious. Numerically valid, confidently tracked
  forward-power profiles remain eligible with warnings when confinement is
  suspect or unresolved. Failed residuals, nonfinite fields, singular
  reconstruction, and rank loss remain disqualifying. Evanescent references
  do not become injectable merely because their transverse fields are bound.
* Full verification concerns the represented voxel geometry. It requests
  twice the transverse resolution and two larger, PML-free windows for open
  cross-sections. Missing exterior geometry, unsupported faithful refinement
  of thin sheets or impedance boundaries, failed solves, and exhausted solve
  budgets leave diagnostics unresolved. Larger-window verification is
  currently unavailable with domain-decomposed MPI. ``verification="fast"``
  omits these extra solves and provides only residual and edge evidence.

See :ref:`eigenmode-auto-tracking-theory` for the assignment and diagnostic
checks, and Examples 8 and 9 in :doc:`eigenmode_port` for circular degeneracy
and a crossing in a single anisotropic guide.

.. _eigenmode-auto-tracking-theory:

Mode tracking theory
====================

A broadband source interpolates between fields solved at different frequencies.
Before it can do that, it must establish which fields belong together. Three
examples explain why:

* In an anisotropic guide, two propagation curves can cross. The same physical
  field can move from the first to the second position in the solver's list.
  Field matching follows that mode through the crossing.
* For a circular TE11 pair, x/y fields and two diagonal fields are equally
  valid ways to describe the same degenerate pair. Subspace alignment chooses
  consistent representatives before interpolating them.
* An artificial outer boundary can support or distort a mode. A successful
  eigenvalue solve does not establish that the profile represents the intended
  open guide. Mesh/window comparisons provide separate evidence and warnings;
  they do not automatically remove a usable profile.

The worked explanations and user controls are in
:ref:`eigenmode-mode-tracking`. This section gives the mathematics behind them.
Automatic tracking remains experimental and is selected at the user's
discretion per port with ``tracking="auto"``. ``anchors`` independently selects
the primary solve frequencies.

The automatic pipeline has five stages: solve primary anchors with extra
internal candidates; assign branch identities outwards from the reference
anchor; resolve ambiguity with midpoint solves; orient eligible degenerate
groups; and evaluate numerical, confinement, and artifact diagnostics. The
resulting bank supplies injection, receiver decomposition, cached studies,
and inspection plots. These stages do not collapse into a single valid/invalid
flag: a boundary-sensitive profile may remain usable even though its
confinement evidence is poor.

With ``tracking="auto"``, public labels are seeded at the solved anchor nearest
the band centre, choosing the lower frequency on a tie. Tracking proceeds
outwards in both directions using complex E/H overlap and a prediction of the
next eigenvalue. It uses the physical fields of the 1D and 2D solvers with
their Yee sampling and numerical-dispersion conventions. One-to-one
assignment includes an explicit unmatched state;
an ambiguous assignment is not forced merely to keep a mode number present.
Extra internal candidates, including missing degenerate partners, do not
create additional public source or monitor channels.
``extra_candidates`` requests four additional eigenpairs per solve by default,
bounded by the available free degrees of freedom. Hidden partners complete
the internal group while ``modes`` continues to define the public channels.

Field overlap and eigenvalue prediction
---------------------------------------

Field overlap measures how similar two patterns are after ignoring an
arbitrary overall phase. It allows an x-like mode to match the next x-like
mode even when their raw solution numbers differ. The propagation-constant
trend provides another clue; neither the list position nor that trend alone
decides the match.

For candidate :math:`i` at anchor :math:`k`, concatenate the native physical
electric and impedance-scaled magnetic fields into a normalized column:

.. math::

   q_{k,i} = \frac{(\mathbf E_{k,i},\eta_0\mathbf H_{k,i})}
                   {\|(\mathbf E_{k,i},\eta_0\mathbf H_{k,i})\|_2},
   \qquad
   O_{ij}=|q_{k,i}^{\mathrm H}q_{k+1,j}|^2.

The squared overlap is insensitive to arbitrary complex phase and is clipped
to :math:`[0,1]` against round-off. Invalid or nonfinite field columns cannot
provide a valid matched anchor. With two previously tracked anchors, the
native eigenvalue is predicted by linear extrapolation in frequency:

.. math::

   \widehat\lambda_i(f_{k+1}) = \lambda_i(f_k)
     + \frac{f_{k+1}-f_k}{f_k-f_{k-1}}
       [\lambda_i(f_k)-\lambda_i(f_{k-1})].

The first step uses the previous eigenvalue without extrapolation. The same
rule applies while traversing anchors towards lower frequencies. For an
individual-mode candidate, the assignment cost is

.. math::

   d_{ij} &= \frac{|\lambda_j(f_{k+1})-\widehat\lambda_i(f_{k+1})|}
                       {\max(1,|\lambda_i(f_k)|)},\\
   C_{ij} &= 1-O_{ij}+0.1\min(d_{ij}^2,4).

Candidates below ``tracking_overlap`` (default squared overlap 0.8) are
excluded. Additional unmatched columns have cost ``unmatched_cost`` (0.65).
A minimum-total-cost one-to-one assignment prevents two public labels from
claiming the same candidate. For each proposed individual match, the solver
repeats assignment with that edge forbidden. The increase in total cost must
be at least ``assignment_margin`` (0.02); otherwise the identity remains
unmatched. Thus a locally strong overlap is insufficient if an almost equally
good competing assignment exists.

SVD rank checks and principal-angle comparison
----------------------------------------------

For TE11, comparing only the first returned field can mistake a harmless
rotation from x/y to diagonal polarizations for a change of mode. Instead we
compare all fields that the pair can form together: their **subspace**. SVD
provides a well-conditioned basis for that comparison, independent of the
particular pair the eigensolver returned.

A degenerate eigenvalue defines a space of valid fields, rather than a unique
set of individual eigenvectors. An eigensolver may return any invertible
combination of a basis for that space, including nonorthogonal columns.
Normalizing each column alone therefore cannot make a subspace comparison
independent of the solver's basis.

Let :math:`F\in\mathbb C^{N\times r}` contain the flattened E/H field columns
for an :math:`r`-mode group. Candidate assignment uses the normalized columns
:math:`q_i` above; alignment uses the full E/H frame at its current scaling.
The reduced singular value decomposition (SVD) is

.. math::

   F=U\Sigma V^{\mathrm H},\qquad
   \Sigma=\operatorname{diag}(s_1,\ldots,s_r),\qquad
   s_1\ge\cdots\ge s_r\ge0.

The implementation requires finite singular values and
:math:`s_r>s_1/10^8`. It rejects rank loss or excessive conditioning instead
of silently dropping a partner and changing the group's dimension. The
orthonormal span basis and its transformation from the input columns are

.. math::

   Q=U,\qquad B=V\Sigma^{-1},\qquad FB=Q,\qquad Q^{\mathrm H}Q=I.

This Euclidean E/H normalization compares spans and conditions the power
check; it does not by itself give unit transported power.

For equal-rank spans at two anchors, a second SVD gives their principal angles:

.. math::

   Q_a^{\mathrm H}Q_b=L\,\operatorname{diag}(\sigma_1,\ldots,\sigma_r)
                       R^{\mathrm H},\qquad
   \sigma_i=\cos\theta_i.

The smallest singular value measures the least well matched direction. The
automatic candidate-assignment score is

.. math::

   O_{\mathrm{span}} = \sigma_{\min}^2(Q_a^{\mathrm H}Q_b).

Identical spans have score one even if their input columns have unrelated
phases, ordering, rotations, or nonorthogonal bases. A missing or substantially
changed direction lowers the smallest singular value even when the other
directions match well. Assignment compares the squared score against
``tracking_overlap`` (default 0.8). The later aligned-group continuity check
uses the unsquared :math:`\sigma_{\min}` with its separate 0.9 warning and
0.6 rejection thresholds.

Candidate clusters and adaptive frequency solves
------------------------------------------------

Candidate clusters use connected relative eigenvalue gaps no larger than
``cluster_gap`` (``1e-5``). Equal-rank spans are reserved using this score and
the predicted cluster-centre eigenvalue before assigning the remaining
individual candidates. This broad matching cluster is not a permission to
mix fields. Automatic mixing groups must satisfy the tighter ``1e-8``
eigenvalue-spread condition at the reference anchor and throughout the solved
bank; branches with resolved splitting remain separate. Field mixing requires
the stricter degeneracy, propagation-branch, positive-power, rank, and
mixed-residual checks in :ref:`eigenmode-degenerate-theory`. A crossing does
not by itself justify mixing two branches with resolved splitting.

Ambiguous intervals trigger additional candidates and midpoint solves. The
default limits are eight refinement levels, a minimum relative frequency
step of ``1e-5``, and 200 additional solves per port, shared with verification.
Required primary anchors are not removed to meet that budget. Persistent
ambiguity follows the anchor policy: automatic anchors may fall back to a
single band-centre profile; multiple explicit anchors raise an error.
For an ambiguous interval :math:`[f_a,f_b]`, the next frequency is
:math:`f_m=(f_a+f_b)/2`; subdivision stops when
:math:`(f_b-f_a)/\max(|f_m|,1\,\mathrm{Hz})\le10^{-5}` or a configured limit
is reached. A midpoint solve uses the same extra-candidate allowance and the
assignment is reevaluated across the bank. The current algorithm does not
increase that allowance automatically on each retry. Increase
``extra_candidates`` explicitly if inspection indicates missing candidates.

For independently tracked modes, the assigned fields are phase-aligned before
interpolation using their complex E/H overlap. Degenerate groups instead use
the physical-direction or unitary transport rules below. Neither procedure
authorizes interpolation through an unresolved assignment. After fallback,
the retained centre profile is a fixed-profile approximation away from that
frequency, and the unresolved interval remains recorded.

Residuals and verification evidence
-----------------------------------

Numerical validity uses eigenpair and reconstructed Maxwell-equation residuals
from the actual discrete operators, with default tolerance ``1e-8``. For the
reduced eigenproblem :math:`Av=\lambda v`, the eigenpair residual is

.. math::

   r_{\mathrm{eig}} =
   \frac{\|Av-\lambda v\|_2}{\|Av\|_2+\|\lambda v\|_2}.

The denominator is protected against zero. The reconstructed-field residual
checks the discrete Maxwell equations with the solver's numerical-dispersion
operators; the larger of the eigenpair and field residuals controls numerical
validity. A small eigenpair residual alone does not establish a correct
reconstructed E/H profile.

Full verification compares propagation constants and E/H fields against a refined
transverse mesh and, for open cross-sections, windows padded by approximately
25% and 50% of the original extent. The larger windows use available model
geometry at unchanged spacing and exclude PML. Degenerate groups are compared
as subspaces. Default acceptance thresholds are normalized beta drift
``1e-3``, verification overlap ``0.999``, and outer-edge field-norm fraction
``1e-3``; physical enclosing walls are distinguished from artificial edges.
These controls belong to ``EigenmodeTrackingConfig``.

The propagation-constant comparison uses

.. math::

   \delta_\beta = \frac{|\beta_{\mathrm{test}}-\beta_{\mathrm{base}}|}
                            {\max(|\beta_{\mathrm{base}}|,k_0)}.

For a degenerate group it uses the mean beta and compares complete spans.
Verification fields are compared on the original window after cropping a
larger solve or averaging refined cell fields back to the original cells.
The edge fraction is the sum of :math:`|\mathbf E|^2+\eta_0^2|\mathbf H|^2`
in the outer 10% strips (at least one cell per edge), divided by that sum over
the full cross-section. This is a field-norm diagnostic, not a leakage-power
measurement. A recognized physical PEC/PMC enclosure is not penalized for
field near its wall. ``verification="fast"`` evaluates residual and edge
evidence without the refinement/window comparisons.

Confinement, artifact suspicion, tracking confidence, and propagation/power
eligibility are separate diagnostics. Domain sensitivity or appreciable
field at an artificial edge can indicate leakage or a truncation/box artifact;
mesh sensitivity alone does not establish a spurious mode. Complex beta alone
is also not an artifact test. Suspect or incomplete confinement evidence
produces a warning that the usable profile remains in use and accuracy may
be reduced. Such warnings do not trim anchors, renumber modes, or force
tracking fallback.

The resolved bank is shared by injection, monitors, cached studies, and plots.
The versioned HDF5 ``mode_tracking`` group records candidate mappings,
requested/adaptive frequencies, residuals, diagnostic summaries, and unresolved
intervals without changing the existing validity-mask meanings. Degenerate
transforms are stored in ``degenerate_groups``. The modal-field plots include
tracked dispersion and diagnostic status, including in geometry-only runs.

.. _eigenmode-degenerate-theory:

Degenerate subspaces and physical alignment
-------------------------------------------

Polarization means the direction of the integrated transverse electric field,
not the direction of every local field vector:

.. math::

   \mathbf m_j(f)=\int_{\mathrm{port}}\mathbf E_{t,j}(f)\,dA.

At every anchor, the two moment columns form :math:`M`. Requested real unit
directions in the transverse plane form :math:`D`. The initial transformation
is obtained by solving a small linear system:

.. math::

   MT_0=D,\qquad
   \mathbf E'_j=\sum_i\mathbf E_i(T_0)_{ij},\qquad
   \mathbf H'_j=\sum_i\mathbf H_i(T_0)_{ij}.

Applying the same transformation to every E/H component preserves their
relative phase. For real forward powers :math:`p_j>0`, the final transform is

.. math::

   T=T_0\operatorname{diag}(p_j^{-1/2}),\qquad
   MT=D\operatorname{diag}(p_j^{-1/2}).

Thus each column has unit real power and a positive real electric moment
parallel to its requested direction; its integrated moment need not have unit
magnitude. This fixes orientation and phase consistently across ports before
interpolation.

Requested directions need not be orthogonal. The full Hermitian power matrix
is retained: for simultaneous coefficients :math:`c`, total power is
:math:`c^\dagger P c`, which can include interference terms. Continue using
multiple ``EigenmodeExcitation`` objects with ``amplitude`` and ``phase_deg``
for coherent combinations; a 90-degree relative phase drives quadrature.

Declared legacy groups and automatically detected groups use the same
alignment machinery. With physical directions, alignment is enforced
independently at every anchor; subsequent tracking diagnostics do not rotate
those directions. Without usable physical directions, a deterministic
reference basis is transported outwards using the SVD rotation below.
Legacy modes without a declared group retain independent-mode phase tracking.

Automatic tracking ignores ``degenerate`` and detects the groups from the
solved spectrum. ``mode_polarizations`` is optional: a detected two-mode group
uses the port's global transverse axes when its integrated electric moments
support them, otherwise a deterministic subspace basis. Explicit polarization
directions override this choice and are validated against the detected groups.

The native discrete eigenvalue spread must be below
:math:`10^{-8}\max(1,\max|\lambda|)`, and mixed-mode relative eigen-residuals
must be below :math:`10^{-9}`. Resolved splitting in a declared group is an
error; automatic tracking keeps split branches as independent modes.
Propagation constants are preserved individually. Moment and direction
condition numbers above :math:`10^8` are rejected. A usable electric moment
must also exceed :math:`10^{-12}` times the absolute transverse-field sample
sum multiplied by transverse cell area; this prevents cancellation noise from
defining a polarization even when its relative condition number looks benign.
Higher-order groups with vanishing integrated E can use generic tracking,
but cannot use axis/vector references.
Modes on opposite propagation branches are not mixed even if their squared
eigenvalues coincide.

Continuity is measured using the smallest principal subspace-overlap singular
value: below 0.9 warns, and below 0.6 rejects an in-band match. Legitimate
automatic guard trimming and cutoff exclusion apply to the whole group;
failed groups never silently fall back member by member. Non-propagating
group anchors are excluded from excitation and physical references.

Power conditioning and normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For group coefficients :math:`c`, define the discrete cross-power matrix
using the port's propagation direction :math:`\hat{\mathbf w}` and its Yee
field quadrature:

.. math::

   C_{ij}=\frac12\int
       (\mathbf E_j\times\mathbf H_i^*)\cdot\hat{\mathbf w}\,dA,\qquad
   P=\frac{C+C^{\mathrm H}}{2},\qquad
   P(c)=c^{\mathrm H}Pc.

The 1D cross-section uses its corresponding line quadrature and power per
unit invariant length. A change of basis :math:`F\mapsto FT` transforms this
Hermitian power form as :math:`P\mapsto T^{\mathrm H}PT`.

Testing power directly in a poorly conditioned solver basis can exaggerate
conditioning because the quadratic form contains that basis twice. The code
first uses the SVD preconditioner :math:`B=V\Sigma^{-1}` and tests
:math:`P_B=B^{\mathrm H}PB`. Its eigenvalues must be finite and satisfy
:math:`p_{\min}>p_{\max}/10^8`, ensuring an independent positive-power basis.
A small eigenvalue residual does not substitute for this power test.

For generic subspace alignment, the canonical basis described below has
transform :math:`T_c` and power matrix
:math:`P_c=T_c^{\mathrm H}PT_c=Z\operatorname{diag}(p_j)Z^{\mathrm H}`.
The implementation normalizes the complete power form:

.. math::

   T=T_cZ\operatorname{diag}(p_j^{-1/2})Z^{\mathrm H},\qquad
   T^{\mathrm H}PT=I.

Physical-direction alignment instead normalizes each column as above.
Prescribed nonorthogonal directions can leave off-diagonal power terms;
those terms are retained rather than discarded.

Deterministic basis without electric directions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When integrated electric moments cannot supply a physical orientation, the
code constructs a repeatable basis from the subspace projector
:math:`\Pi=QQ^{\mathrm H}`. Let :math:`e_\ell` be a coordinate vector in the
flattened E/H sample order, and let :math:`W` contain the canonical columns
already selected. For each unused coordinate, form

.. math::

   z_\ell=(I-WW^{\mathrm H})\Pi e_\ell.

Choose the largest :math:`\|z_\ell\|_2`, breaking an exact tie by the lowest
coordinate index. Normalize it and multiply by a complex phase so its selected
coordinate is positive real. Repeat until there are :math:`r` columns.
The implementation applies the projector through :math:`Q` without forming
the dense :math:`N\times N` matrix. A missing independent direction is an
error.

For the resulting frame :math:`F_c`, a least-squares solve determines
:math:`T_c=F^\dagger F_c`, so :math:`FT_c=F_c` up to numerical precision.
Because the projector depends on the span, this construction avoids choosing
an orientation from arbitrary raw eigenvectors. The fixed sample ordering
defines its convention; it is not a requested physical polarization.
Power normalization is then applied before transport.

Unitary SVD transport between anchors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a group without physical-direction constraints, let :math:`A_k=F_kT_k`
be its current power-normalized frame and :math:`A_p` the already aligned
frame at the preceding anchor along the tracking path. Solve the complex
orthogonal Procrustes problem

.. math::

   R_*=\underset{R^{\mathrm H}R=I}{\operatorname{argmin}}
          \|A_kR-A_p\|_{\mathrm F}.

Expanding the squared Frobenius norm shows that this is equivalent to
maximizing :math:`\operatorname{Re}\operatorname{tr}(R^{\mathrm H}A_k^{\mathrm H}A_p)`.
For the SVD

.. math::

   A_k^{\mathrm H}A_p=U_s\Sigma_sV_s^{\mathrm H},
   \qquad R_*=U_sV_s^{\mathrm H},

update :math:`T_k\leftarrow T_kR_*` and :math:`A_k\leftarrow A_kR_*`.
The rotation is applied to all electric and magnetic field components.
It preserves unit power because :math:`R_*^{\mathrm H}IR_*=I`, and it stays
inside the same degenerate subspace. Transport starts at the retained anchor
nearest the band centre (lower frequency on a tie) and proceeds separately
towards increasing and decreasing frequencies.

This transport uses the power-normalized frames themselves; the principal
angle test above uses Euclidean-orthonormalized spans. The two SVDs answer
different questions: whether the spans match, and which allowed orientation
best continues the preceding frame. A physically specified or automatically
selected electric direction is not subsequently rotated by Procrustes
transport.

After the final rotation, the code recomputes the mixed eigenpair residual
for each column against its own stored eigenvalue:

.. math::

   r_j=\frac{\|A_{\mathrm{op}}(XT)_j-\lambda_j(XT)_j\|_2}
              {\|A_{\mathrm{op}}(XT)_j\|_2+\|\lambda_j(XT)_j\|_2}
       \le 10^{-9}.

Here :math:`X` contains the raw reduced eigenvectors and
:math:`A_{\mathrm{op}}` is the actual discrete eigenproblem operator.
Propagation constants and eigenvalues are never replaced by a group average
to make mixing pass. Mixing resolved splitting or opposite propagation
branches remains forbidden; a numerical rotation cannot repair that physical
incompatibility.

The aligned bank is shared by single-anchor and broadband sources, monitors,
modal studies, virtual guides and plots. HDF5 port groups include
``degenerate_groups`` with requested directions, achieved electric moments,
transformations, power matrices, eigenvalue spread, condition numbers,
residuals, subspace overlaps and retained-anchor flags. Geometry-only modal
field exports include the same diagnostics; plot titles show assigned directions.

Confinement and suspected box-mode detection
--------------------------------------------

A finite artificial transverse window can support eigenvectors that mainly
describe the truncation box rather than the intended guide. Such a box mode
can satisfy the discrete eigenproblem accurately. Residual convergence alone
therefore cannot distinguish it from a physical guided mode. The detector
uses sensitivity to the transverse mesh/window and participation of fields
near the outer boundary as separate evidence.

The implementation reports ``artificial_boundary_or_box_suspect`` rather than
claiming a definitive spurious mode. A physical weakly confined mode can show
the same sensitivity when its evanescent tail is truncated. In particular,
large low-frequency CPW profiles can be questionable at a small aperture and
still supply usable forward-power fields. Conversely, decay along the guide
below cutoff does not imply poor transverse confinement. Complex beta alone
is never the box-mode test.

Physical walls and artificial edges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The enclosure test inspects the solver's PEC/PMC masks, not an eigenvalue
threshold. In a 2D cross-section, each outer edge must have the required
tangential component constraints for PEC or PMC to recognize a complete
physical enclosure. The reduced 1D solver uses the corresponding scalar
endpoint masks. A recognized enclosure requires the refined-mesh comparison
but skips larger-window comparisons and the outer-edge penalty.

Otherwise, the finite window is treated conservatively as open for these
diagnostics. The current edge-strip norm covers all outer strips; it does
not individually remove physical-wall strips in a partially open perimeter.
Thus an edge warning is evidence to inspect, not a resolved separation of
physical leakage and artificial truncation effects.

Refinement and larger-window comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``verification="full"``, the refined solve halves each physical transverse
spacing, expands the stored Yee constitutive arrays and PEC/PMC masks, and
rebuilds the solver operators. It retains the solve frequency, FDTD time step,
and propagation-axis spacing. This tests the represented voxel cross-section;
it does not revoxelize an original smooth circle or recover missing subvoxel
thin-sheet geometry. Surface-impedance refinement is currently unsupported
and produces unresolved verification rather than a substituted boundary model.

For open cross-sections, each transverse side is padded by
:math:`\lceil0.25N\rceil` and then :math:`\lceil0.50N\rceil` cells, where
:math:`N` is the original extent along that axis. These are per-side additions,
so the total window widths are approximately 1.5 and 2 times the originals.
Spacing is unchanged. The surrounding material and boundary data are extracted
from the actual model; no homogeneous exterior is invented. A requested
window that reaches beyond available non-PML geometry is unavailable. These
larger-window solves are also currently unavailable in domain-decomposed MPI.

Verification does not compare raw numerical mode numbers. It selects the
highest-overlap candidate after cropping a larger window to the original
cross-section or averaging refined cell fields back to the original cells.
For a degenerate group it compares equal-rank candidate spans, using
orthonormal bases and the smallest squared principal overlap. A missing
matching span leaves the comparison unresolved. The chosen candidate or group
must meet all of the following defaults:

* eigenpair and reconstructed-field residual at most ``1e-8``;
* normalized beta drift at most ``1e-3``;
* squared field/subspace overlap at least ``0.999``;
* for a padded window, its own outer-edge field-norm fraction at most ``1e-3``.

In addition, a non-enclosed primary profile must have outer-edge fraction at
most ``1e-3`` to receive the complete ``bound`` classification. Mesh sensitivity
alone records insufficient convergence evidence; it does not identify a box
mode. Failed padding comparisons or appreciable outer-edge fields supply
the boundary-sensitivity evidence used for artifact suspicion.

Classification decisions
^^^^^^^^^^^^^^^^^^^^^^^^

Full verification expects one successful comparison for a recognized physical
enclosure and three for an open cross-section. Here "complete" means all
required comparisons could be evaluated, even if some failed their thresholds.
The diagnostic decision is independent of the primary numerical-validity and
power masks:

.. list-table:: Confinement and artifact states
   :header-rows: 1
   :widths: 50 20 30

   * - Evidence
     - Confinement
     - Artifact suspicion
   * - Full checks complete and pass; primary edge test passes or enclosure is physical.
     - ``bound``
     - ``none``
   * - Full checks complete but a padding comparison fails or an artificial-edge profile exceeds the norm threshold.
     - ``unbound_suspect``
     - ``artificial_boundary_or_box_suspect``
   * - Full checks complete but only the mesh comparison fails, without boundary evidence.
     - ``unresolved``
     - ``none``
   * - Full checks incomplete: unavailable geometry/refinement, failed verification solve or comparison, or exhausted budget.
     - ``unresolved``
     - ``unresolved``
   * - Fast checks: physical enclosure recognized or primary edge fraction passes.
     - ``bound``
     - ``none``
   * - Fast checks: no recognized enclosure and primary edge fraction fails.
     - ``unbound_suspect``
     - ``artificial_boundary_or_box_suspect``

The fast path uses residual and enclosure/edge evidence only. Its ``bound``
label does not imply that mesh or window stability was verified. Even a full
``bound`` result is evidence at the tested resolutions and windows, not a
proof of continuum convergence or absence of radiation.

Verification is limited by the additional-solve budget shared with adaptive
tracking. Exhausting it leaves required primary anchors intact. Reasons such
as ``insufficient_non_pml_exterior_geometry``,
``mpi_padding_verification_unavailable``, or
``verification_solve_budget_exhausted`` identify incomplete evidence. A failed
verification eigensolve leaves the primary profile unresolved; it does not
retroactively make a numerically valid primary eigensolve fail.

Anchor eligibility, warnings, and recorded diagnostics
------------------------------------------------------

Numerical validity determines whether usable fields exist. Tracking confidence
determines whether anchors may be connected. Propagation and forward power
determine injection eligibility; generalized below-cutoff receiver references
retain their existing restrictions. Confinement and artifact suspicion are
advisory: a numerically valid, confidently tracked forward-power profile stays
eligible even when unbound or box-like behavior is suspected. Such warnings
alone do not trim anchors, break interpolation, renumber modes, or trigger
fallback. Failed primary residuals, nonfinite fields, singular reconstruction,
and rank loss remain disqualifying.

The coordinator aggregates confinement warnings for each port and public mode,
listing the states, reasons, and minimum-to-maximum affected frequency range.
This summary can span intervening frequencies with no warning; inspect the
per-anchor records for individual statuses. The message states that usable
profiles remain in use and that injection or S-parameter accuracy may be
reduced. No recommendation warning is emitted merely for selecting legacy
tracking.

The shared resolved bank is reused by sources, receivers, cached studies,
virtual guides, and plots. Geometry-only runs can inspect dispersion and E/H
profiles without time stepping. Diagnostic markers distinguish questionable
retained profiles from missing or invalid profiles; they do not alter the
underlying anchor masks.

HDF5 diagnostics are added for automatic tracking without changing existing
dataset names or validity-mask meanings. The current ``mode_tracking`` schema
is version 1:

* Attributes identify the tracking/verification policy, source revision,
  reference-anchor index, verification solve count, and detected groups.
* ``candidate_indices`` records original, zero-based solver candidate indices
  after assignment to tracked labels; public mode labels remain one-based.
  Hidden partners can appear in the internal mapping without becoming public
  measurement channels.
* ``tracking_overlaps``, ``eigenpair_residuals``, ``field_residuals``,
  ``combined_residuals``, and ``numerical_valid`` preserve assignment and
  numerical evidence.
* ``requested_frequencies``, ``adaptive_frequencies``, and the
  ``unresolved_interval_*`` datasets record sampling and remaining identity
  gaps.
* ``anchor_quality`` records public mode/frequency rows with residuals,
  edge fraction, numerical validity, confinement, artifact suspicion, and
  reason strings. Detailed per-comparison candidate, overlap, beta-drift, and
  pass/fail records are held in the runtime diagnostics; schema 1 does not
  serialize those individual verification records.
* The separate ``degenerate_groups`` hierarchy stores orientation, reference
  anchor, transforms, power matrices, residuals, overlaps, retained flags, and
  physical directions/moments when available. Its transformation convention
  is ``aligned fields = raw fields @ transform``.

Older output files without these groups remain readable. Existing
``anchor_mode_valid``, ``anchor_mode_reference_valid``, and measurement power
masks continue to control how results may be used; a quality label does not
replace them.

Legacy phase tracking
---------------------

This subsection describes independent-mode phase tracking with
``tracking="legacy"``. Declared groups use
:ref:`eigenmode-degenerate-theory`; optional automatic assignment is described
in :ref:`eigenmode-auto-tracking-theory`.

For solve frequencies :math:`f_k`, gprMax obtains fields
:math:`(\mathbf{E}_k,\mathbf{H}_k)` and effective indices :math:`n_k`.
Adjacent eigenvectors can carry unrelated arbitrary phases, so their complex
overlap is evaluated as

.. math::

   O_{k-1,k}=
   \frac{
     \langle\mathbf{E}_{k-1},\mathbf{E}_k\rangle
     +\langle\eta_0\mathbf{H}_{k-1},
              \eta_0\mathbf{H}_k\rangle
   }{
     \| (\mathbf{E}_{k-1},\eta_0\mathbf{H}_{k-1}) \|
     \| (\mathbf{E}_k,\eta_0\mathbf{H}_k) \|
   }.

Anchor ``k`` is multiplied by
``exp(-j*arg(O[k-1,k]))``. This makes interpolation follow a continuous phase
choice instead of blending arbitrary eigenvector phases. If
``abs(O) < 0.9``, gprMax warns that the mode may have crossed cut-off, become
degenerate, changed ordering, or been sampled too sparsely. If
``abs(O) < 0.6``, the ambiguity is too large for ordinary interpolation.
Multiple explicit anchors stop with an error. With automatic anchors, an
outer-guard failure trims the affected port/mode spectral tail; an in-band
failure selects the band-centre single-frequency basis only for that port and
mode. The candidate frequency list itself remains common.

Phase tracking is evaluated before the forward-power filter so that a solved
non-propagating candidate can still diagnose and represent branch continuity.
The forward-power filter produces the one-watt source/power mask, while every
successfully tracked retained candidate produces the monitor-reference mask.
A centre-only tracking fallback collapses both masks so that a rejected mode
cannot re-enter monitor interpolation. Non-propagating reference anchors are
never used by TF/SF source synthesis or treated as power waves.

Validation and reproducibility
==============================

The degenerate-mode validation report is
``testing/validation/degenerate_eigenmode_results.json``. It records CPU
comparisons for circular TE11 with x/y and diagonal directions, all three
propagation axes and both directions, randomized raw bases, both precisions,
and physical versus virtual continuations. Its recorded direction error is
``4.12e-16`` before field casting and ``9.10e-10`` after single-precision
casting; the mixed-mode residual is ``6.21e-13``. These are results for the
reported configurations, not bounds for arbitrary cross-sections.

To repeat the focused configuration, tracking, plotting, and integration checks:

.. code-block:: console

   python -m pytest tests/test_eigenmode_degenerate.py tests/test_eigenmode_degenerate_integration.py tests/test_eigenmode_config.py

The optional tracking and two-port tutorial checks are:

.. code-block:: console

   python -m pytest tests/test_eigenmode_auto_tracking.py tests/test_eigenmode_tracking_examples.py

Use ``testing/validation/degenerate_eigenmode_ports.py`` for its executable
circular-guide validation model. Ordinary source, dispersion, and
virtual-guide regressions are under ``tests/fdfd_eigenmode_solver`` and
``testing/regression/eigenmode_sources``. The recorded broad CPU run excluded
accelerator tests and skipped cases requiring MPI/parallel HDF5; it is not
GPU validation evidence.

For SIBC-backed guides, the boundary, PMC, PML-profile, and long-run evidence
belongs to :doc:`impedance_surfaces_theory`. Output layouts and modal
diagnostics are described in :doc:`output`.
