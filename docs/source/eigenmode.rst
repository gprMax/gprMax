.. _eigenmode:
.. _fdfd-eigenmode-source:

*********************************************
Eigenmode Sources, Ports, and Mode Solvers
*********************************************

Eigenmode sources launch a solved waveguide mode instead of prescribing one
field component. Every source is simultaneously a modal receiver, and
additional ``#eigenmode_rx`` planes provide explicitly numbered ports. A
single time-domain run can therefore produce multimode S-parameters and, when
the device radiates, directivity, gain, and realized gain.

Using an eigenmode source and receiver
======================================

The quickest workflow is:

1. build the geometry without time stepping and inspect every modal field;
2. run the complete model to accumulate the requested modal DFT bins;
3. read the portable S-parameter CSV or the corresponding HDF5 datasets.

This straight 2D dielectric guide uses source port 1 and receiver port 2:

.. literalinclude:: ../../examples/features/eigenmode_sources/dielectric_slab_2d_tm.in
   :language: none
   :caption: ``dielectric_slab_2d_tm.in``

The source token after the direction is
``excitation_mode[,mode_count]``. Both values are one-based. For example,
``2,4`` excites mode 2 while the source monitor calculates modes 1 through 4.
If the second value is omitted, the monitor calculates modes 1 through the
excited mode. The following integer is the explicit one-based port number.

For ``#eigenmode_rx``, the integer after the direction is only the number of
consecutive modes to calculate and measure, followed by its explicit
one-based port number. Port numbers must be unique. Whenever either command
is present, one and only one ``#eigenmode_source`` must exist.

Run the geometry-only check first:

.. code-block:: console

   python -m gprMax examples/features/eigenmode_sources/dielectric_slab_2d_tm.in --geometry-only

For 2D models this writes a TM or TE modal-field PNG. For 3D models it writes
separate transverse electric and magnetic PNGs. Check the expected
polarisation, symmetry, confinement, conducting-boundary behaviour, and mode
order. The final optional ``y`` or ``n`` on either eigenmode command forces or
suppresses these plots during a normal run.

Then run the time-domain model:

.. code-block:: console

   python -m gprMax examples/features/eigenmode_sources/dielectric_slab_2d_tm.in -outputfile examples/features/eigenmode_sources/dielectric_slab_2d_tm
   python examples/features/eigenmode_sources/plot_dielectric_slab_2d_tm.py

The source and every receiver must use identical ``dft_start``, ``dft_stop``,
and ``dft_points`` values. Each requested bin is updated once at every FDTD
time step by the modal Cython DFT kernel. This example requests 21 points from
4 to 6 GHz. The resulting
``examples/features/eigenmode_sources/dielectric_slab_2d_tm_sparameters.csv``
contains one row per frequency,
destination port, and destination mode. Source-port rows are modal S11
results; port-2 rows are modal S21 results. Complex value, magnitude, dB
magnitude, phase, coefficient magnitude squared, and validity are all
included. Coefficient magnitude squared is not an independently attributable
modal power fraction when the power matrix is non-diagonal.

The same data are stored in HDF5:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File(
       "examples/features/eigenmode_sources/dielectric_slab_2d_tm.h5", "r"
   ) as output:
       source = output["eigenmode_ports/port1"]
       receiver = output["eigenmode_ports/port2"]
       frequency = source["frequency"][...]
       s11_mode1 = source["S"][0]
       s21_mode1 = receiver["S"][0]
       s21_mode2 = receiver["S"][1]
       valid_21_mode2 = receiver["valid_S"][1].astype(bool)

   s21_mode2_db = 20 * np.log10(np.abs(s21_mode2[valid_21_mode2]))

Arrays ``incident`` and ``outgoing`` have shape
``(number_of_modes, number_of_frequencies)`` and contain generalized modal
travelling-wave coefficients. ``power_matrix`` stores the generally
non-diagonal forward-wave power form and ``electric_cross_power_matrix``
stores the total-field flux form, while ``valid``,
``power_normalization_valid``, ``power_matrix_valid``, and ``valid_S`` identify
usable results. Always apply the validity masks before plotting.

The no-argument
:download:`plot_dielectric_slab_2d_tm.py <../../examples/features/eigenmode_sources/plot_dielectric_slab_2d_tm.py>`
reads those adjacent output files and writes
``dielectric_slab_2d_tm_sparameters.png`` plus
``dielectric_slab_2d_tm_field_propagation.png`` in the same directory. The
second figure places six time-ordered ``Ez`` snapshots on one common colour
scale so that the pulse can be followed along the straight guide. Both ports
monitor modes 1 and 2 while mode 1 is excited. The 21-point sweep should give
essentially 0 dB mode-1 S21, very low S11, and negligible mode-2 conversion
across the plotted band. Values below -100 dB are placed on the plotting floor
so that numerical-zero conversion does not compress the useful traces.

Curved waveguide and mode conversion
====================================

The next example replaces the straight core with a tight 90-degree annular
bend made from two cylindrical sectors:

.. literalinclude:: ../../examples/features/eigenmode_sources/curved_dielectric_waveguide_2d_tm.in
   :language: none
   :caption: ``curved_dielectric_waveguide_2d_tm.in``

The source still excites mode 1, but both ports monitor modes 1 and 2. The
output port is normal to ``y`` because the bend rotates propagation through
90 degrees. Run and plot it with:

.. code-block:: console

   python -m gprMax examples/features/eigenmode_sources/curved_dielectric_waveguide_2d_tm.in -outputfile examples/features/eigenmode_sources/curved_dielectric_waveguide_2d_tm
   python examples/features/eigenmode_sources/plot_curved_dielectric_waveguide_2d_tm.py

The no-argument plot script reads the adjacent S-parameter CSV and plots every
valid source-port reflection and output-port transmission coefficient. It
also writes ``curved_dielectric_waveguide_2d_tm_field_propagation.png`` from
eight time-ordered ``Ez`` snapshots ending at 2.5 ns, using one colour scale
to show the wave entering and leaving the curve without late empty frames. In
contrast to the straight guide, the tight curve reduces fundamental-mode S21,
increases S11, and produces a non-zero port-2 mode-2 transmission coefficient.
That mode-2 trace is the directly measured higher-order-mode conversion.

Choosing frequency anchors
--------------------------

A *single-frequency solve* uses one modal profile at every significant
waveform frequency. Choose its solve frequency at the intended monochromatic
analysis frequency, or near the centre of a genuinely narrow useful band.
Use this path for degenerate modes, modes close to cut-off, and any band that
contains a mode crossing. Do not use broadband interpolation for degenerate
modes: an eigensolver may return a different linear combination of the
degenerate subspace at each anchor. Likewise, numerical mode ordering can
exchange at a crossing even when the integer mode index has not changed.

A *broadband solve* is appropriate only when one isolated physical mode can be
tracked smoothly. Select anchors as follows:

* cover the entire significant spectrum of the source waveform, not merely
  the requested output bins;
* include both useful-band edges and add anchors where effective index,
  confinement, loss, or field shape changes rapidly;
* inspect the modal-field plot at every anchor;
* add anchors adaptively until adjacent profiles have high overlap.

gprMax phase-aligns adjacent anchors before interpolation. An overlap below
0.9 emits a warning. An overlap below 0.6 is treated as ambiguous and raises
an error asking for a single-frequency solve. Adding anchors can resolve
under-sampling, but it cannot make a true degeneracy or mode crossing safe;
use separate single-frequency simulations in those cases.

Eigenmode-fed antenna far fields
================================

The tapered dielectric rod example uses a five-face free-space Huygens
surface around the radiating taper:

.. literalinclude:: ../../examples/features/eigenmode_sources/dielectric_rod_antenna_3d.in
   :language: none
   :caption: ``dielectric_rod_antenna_3d.in``

It intentionally uses a single-frequency solve: circular and nearly circular
cross-sections commonly support degenerate polarisation partners, for which
broadband eigenvector interpolation is not well defined.

The optional final ``x0`` on ``#ntff_surface`` omits that physical face from
the frequency-domain equivalent-current integral. The source plane is on the
feed side of the omitted face, while the uniform dielectric feed continues
without a discontinuity through the rear PML. Backward guide waves therefore
leave the Huygens volume and are absorbed instead of being re-radiated by a
finite feed termination inside the far-field surface.

One to five faces may be omitted, and only ``#ntff_frequency`` supports this
open-surface form. The Ramahi/KSIR formulation and the transient
equivalent-current transform still require a closed or symmetry-completed
surface. Every sampled face must remain outside the PML and in the homogeneous
exterior. An impressed source outside the Huygens volume may enter only through
one of the omitted faces. Multiple omissions are useful when a waveguide passes
through the surface to a passive output port, or when the surface terminates on
a PEC backplane. An open Huygens surface is not an exact mathematical closure;
check convergence by moving and enlarging its sampled faces.

The frequency-transform bins must exactly equal the eigenmode-port DFT bins.
Gain normalization currently requires a rectangular window.
``#ntff_antenna_ports`` must list every physical port, including passive modal
receivers when present. Other active sources that do not expose port power
cannot contribute to the same gain result.

.. warning::

   Eigenmode sources are incompatible with the Ramahi/KSIR formulation and
   gprMax rejects any model that combines them. Use the frequency-domain
   equivalent-current Huygens interface: ``#ntff_frequency`` with
   ``#ntff_far_field`` or ``#ntff_far_field_array``, and add
   ``#ntff_antenna_ports`` when gain or efficiency is required.

Run and plot the example with:

.. code-block:: console

   python -m gprMax examples/features/eigenmode_sources/dielectric_rod_antenna_3d.in -outputfile examples/features/eigenmode_sources/dielectric_rod_antenna_3d
   python examples/features/eigenmode_sources/plot_dielectric_rod_antenna_3d.py

This requests nine S-parameter and far-field frequencies from 6.75 to
7.25 GHz. The narrow sweep is intentional because one 7 GHz modal profile is
reused to avoid broadband interpolation of the degenerate circular-mode
subspace. It produces the source-port S11 CSV plus a full-sphere far-field
group at
``/ntff/radiation_surface/frequency/antenna_band/far_field/full_sphere``.
The no-argument
:download:`plot_dielectric_rod_antenna_3d.py <../../examples/features/eigenmode_sources/plot_dielectric_rod_antenna_3d.py>`
writes a three-panel figure beside the model: S11 versus frequency, peak
directivity/gain/realized gain versus frequency, and their 7 GHz
principal-plane patterns.

All example CSV, HDF5, VTK-HDF, modal-profile, snapshot, and result-plot files
are generated locally and ignored by Git.

.. note::

   The 1 mm reference mesh is deliberately small enough for a quick example.
   Refine the mesh, enlarge the time window, and perform convergence checks
   before using its efficiency values quantitatively.

The requested linear antenna quantities use

.. math::

   D = \frac{4\pi U}{P_{\mathrm{rad}}},\qquad
   G = \frac{4\pi U}{P_{\mathrm{acc}}},\qquad
   G_{\mathrm{realized}} = \frac{4\pi U}{P_{\mathrm{inc}}}.

Here :math:`U` is radiation intensity, :math:`P_{\mathrm{rad}}` is total
radiated power, :math:`P_{\mathrm{acc}}` is the net power accepted across all
physical ports, and :math:`P_{\mathrm{inc}}` is the externally launched
incident power. Directivity is normalised by total radiated power, so it does
not include reductions from dissipative losses or port mismatch. Gain is
normalised by net accepted port power and therefore includes radiation
efficiency, including material loss, but not mismatch loss. Realized gain is
normalised by externally driven incident power and includes both radiation
efficiency and reflection loss. For an eigenmode source, the incident
denominator is the launched mode's modal power; a passive modal receiver
contributes zero generator incident power, but its signed net modal power
remains in the multiport accepted-power balance.

The far-field group stores ``directivity``, ``gain``, and
``realized_gain`` (and their dB forms), radiation and total efficiencies, and
the associated ``port_power`` diagnostics. Modal ports are identified by the
``modal_power_waves`` representation. Their incident/outgoing coefficients,
mode indices, power matrix, and validity mask are repeated below
``port_power/modal_ports`` so the normalization can be audited without
combining unrelated output groups.

Mathematical formulation
========================

The remaining sections describe the component-sampled FDFD eigenproblems,
power normalization and modal reconstruction, followed by spectrum synthesis,
I/Q injection, Yee staggering, direct DFT reception, and multimode
decomposition.

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
    ``TM`` selects the ``E_a`` scalar problem. ``TE`` selects the ``H_a``
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

gprMax uses ``exp(+j*omega*t - j*beta*w)``. The square-root branch is
therefore chosen with positive real phase propagation, ``Re(n_eff) >= 0``.
For a passive mode ``Im(n_eff) <= 0``; a purely evanescent mode uses the
negative-imaginary branch so that it decays in positive ``w``.

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

The branch follows ``exp(+j*omega*t - j*beta*w)``: ``Re(n_eff) >= 0`` for
positive phase propagation and ``Im(n_eff) <= 0`` for passive attenuation.
When the real part is zero, the negative-imaginary branch gives evanescent
decay in positive ``w``.

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
* ``H_u`` and ``H_v`` from ``Q * Euv / sqrt(eigenvalue)``, using the branch
  ``sqrt(eigenvalue) = +j*n_eff`` selected by the propagation convention.
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
* A single-frequency source reuses one solved profile across the waveform
  spectrum. A broadband source instead phase-aligns and interpolates several
  anchor solves, but its accuracy is limited by anchor spacing and it must not
  be used through degeneracies or mode crossings.

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

Source synthesis and FDTD injection
-----------------------------------

This section connects the FDFD solvers described above to the complete
``#eigenmode_source`` workflow. It fixes the phasor and propagation signs,
shows how a solved mode becomes real FDTD update terms, and describes the
single-frequency, in-phase/quadrature (I/Q), and broadband synthesis paths.

Phasor, Fourier, and Propagation Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

With ``exp(+j*omega*t)``, Maxwell's curl equations are

.. math::

   \nabla\times\widetilde{\mathbf{E}}
   &= -j\omega\widetilde{\boldsymbol{\mu}}
      \widetilde{\mathbf{H}},\\
   \nabla\times\widetilde{\mathbf{H}}
   &= +j\omega\widetilde{\boldsymbol{\epsilon}}_c
      \widetilde{\mathbf{E}}.

Finite electric and magnetic conductivities are included through

.. math::

   \epsilon_{r,c}(\omega)
   &= \epsilon_r(\omega)
      -j\frac{\sigma}{\omega\epsilon_0},\\
   \mu_{r,c}(\omega)
   &= \mu_r(\omega)
      -j\frac{\sigma_m}{\omega\mu_0}.

For a forward passive mode,

.. math::

   \beta=k_0n_{\mathrm{eff}}=\beta_r-j\alpha,
   \qquad \alpha\geq 0,

and therefore

.. math::

   \exp(-j\beta w)
   =\exp(-j\beta_r w)\exp(-\alpha w).

The selected square-root branch consequently satisfies

.. math::

   \operatorname{Re}(n_{\mathrm{eff}})&\geq 0,\\
   \operatorname{Im}(n_{\mathrm{eff}})&\leq 0
   \quad\text{for passive propagation}.

If the real part is numerically zero, a purely evanescent mode uses
``Im(n_eff) < 0`` so it decays in positive local ``w``. The imaginary part of
the square root is never replaced by its absolute value; its sign contains
the loss or gain information.

As an example, at 5 GHz a homogeneous material with
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
   2D FDTD model or the 2D full-vector problem for a 3D FDTD model.
5. Reconstruct all modal E and H components, zero constrained degrees of
   freedom, normalize power, and choose a consistent global phase.
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

Mode Selection, Fields, and Power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
power per metre along the invariant axis. A negative initial power reverses
all H components. The fields are then scaled to one watt in 3D or one watt per
metre in 2D.

This normalization defines the scale of the modal profile at the source
plane. Multiplying the source waveform amplitude by a factor multiplies both
incident E and H by that factor; for a monochromatic mode, time-average power
therefore scales with the square of the waveform amplitude.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Broadband Anchor Solves and Phase Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
``abs(O) < 0.6``, the ambiguity is too large for reliable broadband tracking:
gprMax stops and asks the user to use a single-frequency eigenmode solve
instead.

Spectrum and Piecewise-Linear Modal Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The real waveform is sampled at the FDTD interval for the requested number of
iterations. It is zero-padded to the next power of two at least twice as long
as the simulation record, and transformed with ``numpy.fft.rfft``. For FFT
bin :math:`m`, let its frequency and coefficient be :math:`f_m` and :math:`S_m`.

Piecewise-linear weights :math:`w_{k,m}` interpolate between surrounding
anchors and satisfy

.. math::

   \sum_k w_{k,m}=1.

Below or above the anchor range, the nearest endpoint receives weight one.
This avoids a hard spectral truncation, although significant waveform energy
outside the anchor range produces a warning because endpoint extrapolation may
be inaccurate. The interpolated fields and propagation constant are

.. math::

   \mathbf{E}_m &= \sum_k w_{k,m}\mathbf{E}_k,\\
   \mathbf{H}_m &= \sum_k w_{k,m}\mathbf{H}_k,\\
   n_m &= \sum_k w_{k,m}n_k,\\
   \beta_m &= \frac{2\pi f_m}{c}n_m.

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

Thus the interpolated frequency-bin mode is renormalized rather than assuming
that linear field weights retain one-watt power. Invalid or nearly zero power
produces a warning and a finite fallback normalization.

Yee Time and Space Staggering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Electric and magnetic fields are offset by half a time step and their
tangential samples at the TF/SF surface are offset by half a normal cell. For
each bin, gprMax applies the magnetic staggering factor

.. math::

   M_m=\exp\left[j\left(
       \frac{\omega_m\Delta t}{2}
       +\frac{\beta_m\Delta w}{2}
   \right)\right].

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
record, so gprMax sets those bins to zero. Significant DC or Nyquist energy
produces a warning. Broadband and single-frequency I/Q waveforms should
therefore be band-limited and approximately zero mean.

TF/SF Injection into the FDTD Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Modal Receivers, Direct DFT, and S-parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exactly one eigenmode source must exist whenever modal ports are used. The
source owns a passive monitor at its reference plane, and ``#eigenmode_rx``
adds another passive plane. Port indices are supplied explicitly, are
one-based, and must be unique. The source token
``excitation_mode[,mode_count]`` separates the single mode being launched
from the consecutive modes 1 through ``mode_count`` monitored at that source.
An eigenmode receiver similarly monitors modes 1 through its ``mode_count``.
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
Ill-conditioned modal systems and bins with negligible incident spectrum are
retained in the output but marked invalid. Decompositions must satisfy both
the absolute condition cap :math:`\kappa<10^{10}` and the precision-aware
budget :math:`\kappa\epsilon<10^{-3}`. The small Gram systems are solved in
complex128 even when the stored FDTD fields and Gram entries use complex64.

The same Gram matrices define the Hermitian forward-wave power form

.. math::

   W=\frac{1}{2}\left(G^E+G^H\right),\qquad
   P(c)=\operatorname{Re}\{c^\mathrm{H}Wc\}.

The implementation symmetrizes :math:`W` against round-off and checks that it
is finite and positive semidefinite. Keeping the off-diagonal terms is
essential for degenerate, nearly degenerate, or merely non-orthogonal
finite-grid profiles. Individual coefficient magnitudes therefore are not
additive modal powers.

For net accepted power, let :math:`x=a+b` be the total electric coefficient
and :math:`y=a-b` the co-located total magnetic coefficient after the
half-cell correction. The direct time-average flux is

.. math::

   P_{\mathrm{acc},p}
     =\operatorname{Re}\{y_p^\mathrm{H}G^E_p x_p\}.

This reduces to :math:`P(a_p)-P(b_p)` when :math:`G^E_p` is Hermitian.
For a lossy port, its anti-Hermitian part supplies a forward/backward
interference term which must be retained.

The externally driven incident power contains only the excitation mode at the
single source. Passive modal receivers have zero generator incident power,
but their signed accepted power remains in the multiport balance used for
gain. This distinction makes realized gain use launched source power while
gain uses the net power accepted by the radiating structure.

Understanding Lossy-Mode Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a passive lossy mode, users should expect:

* positive ``Re(n_eff)`` for the selected forward phase branch;
* negative ``Im(n_eff)``;
* positive modal power in the requested source direction;
* a forward magnitude factor
  ``abs(exp(-j*k0*n_eff*d)) < 1`` for ``d > 0``;
* a nonzero complex-profile residual in general, causing I/Q injection;
* downstream attenuation to be produced by the FDTD material updates after
  the mode has been launched.

The source field plots show the solved cross-sectional profile and report the
complex effective index. A nonzero receiver field alone does not validate a
lossy mode: sign-sensitive validation must also check ``Im(n_eff)``, modal
power direction, and forward attenuation.

Accuracy Guidance and Warnings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For reliable broadband excitation:

* place frequency anchors across the significant waveform spectrum;
* add anchors near rapid dispersion, cut-off, avoided crossings, or strong
  profile changes;
* inspect adjacent-anchor overlaps below 0.9 rather than assuming a fixed mode
  index tracks the same physical branch, and use a single-frequency solve when
  an overlap is below the hard 0.6 limit;
* avoid significant DC and Nyquist content when I/Q synthesis is required;
* refine the FDTD grid until both the modal effective index and field profile
  converge;
* keep the source plane in a longitudinally uniform section of the guide so
  the transverse eigenproblem represents the structure being launched;
* compare forward and backward modal power or receiver phase when reflections
  and mode purity matter.

The current implementation is CPU-only and cannot be used with MPI. Material
dispersion is sampled at each anchor frequency, but interpolation between
anchors remains piecewise linear; additional anchors are the normal way to
resolve stronger frequency dependence.
