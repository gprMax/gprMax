2D FDFD Eigenmode Solver
========================

Overview
--------

``fdfd_2d_mode_solver.py`` contains ``FDFD_2D_mode_solver``, a 2D vector
finite-difference frequency-domain eigenmode solver used by gprMax to generate
modal fields for eigenmode sources.

The solver is designed for Yee-grid material data. The material arrays supplied
to the solver must already be sampled at the corresponding field-component
locations:

.. list-table::
   :header-rows: 1

   * - Array
     - Sampling location
   * - ``eps_r_xx``
     - ``Ex``
   * - ``eps_r_yy``
     - ``Ey``
   * - ``eps_r_zz``
     - ``Ez``
   * - ``mu_r_xx``
     - ``Hx``
   * - ``mu_r_yy``
     - ``Hy``
   * - ``mu_r_zz``
     - ``Hz``

This is the key assumption behind the solver. It does not infer where a PEC
surface should be from a cell-centred object after the fact. Instead, each
electric component is constrained exactly where that component's sampled
material data says it is inside PEC.

Inputs
------

The constructor signature is:

.. code-block:: python

   FDFD_2D_mode_solver(
       frequency,
       dx,
       dy,
       mode_index,
       eps_r_xx,
       eps_r_yy,
       eps_r_zz,
       mu_r_xx,
       mu_r_yy,
       mu_r_zz,
       pec_ex_mask=None,
       pec_ey_mask=None,
       pec_ez_mask=None,
       pmc_hx_mask=None,
       pmc_hy_mask=None,
       pmc_hz_mask=None,
   )

``frequency``
    Source frequency in Hz.

``dx``, ``dy``
    Transverse cell sizes in metres. The solver normalises finite-difference
    operators by ``k0 * dx`` and ``k0 * dy``.

``mode_index``
    Zero-based modal index requested by the caller. The solver computes
    ``mode_index + 1`` physical modes, then exposes the requested mode through
    ``modal_Ex``, ``modal_Ey``, ``modal_Ez``, ``modal_Hx``, ``modal_Hy``,
    ``modal_Hz`` and ``modal_real_neff``.

``eps_r_*`` and ``mu_r_*``
    Complex relative permittivity and permeability arrays. All arrays must have
    identical ``(Nx, Ny)`` shape and must already be sampled at Yee component
    positions.

``pec_ex_mask``, ``pec_ey_mask``, ``pec_ez_mask``
    Optional explicit boolean masks for constrained electric degrees of freedom.
    These are mainly useful for tests or non-gprMax callers. In normal gprMax
    integration, PEC is passed through non-finite electric material values.

``pmc_hx_mask``, ``pmc_hy_mask``, ``pmc_hz_mask``
    Reserved names for future magnetic conductor constraints. gprMax currently
    supports PEC materials, but not PMC materials, so passing any non-empty PMC
    mask raises ``NotImplementedError``. The names are intentionally kept in
    the solver API so a future PMC implementation can be enabled without
    changing the public argument names.

Array Ordering
--------------

The solver uses the input arrays directly as ``(Nx, Ny)`` arrays. Internally,
field vectors are flattened with Fortran order:

.. code-block:: python

   flat = array.ravel(order="F")

and reshaped back with:

.. code-block:: python

   array = vector.reshape((Nx, Ny), order="F")

There is no axis-order switch. gprMax must pass component-sampled material
slices in the same native transverse ordering used by the extracted source
plane.

PEC Handling
------------

PEC is represented as constrained electric degrees of freedom, not as a large
permittivity approximation.

The solver detects electric PEC in two ways:

1. Explicit electric masks passed through ``pec_ex_mask``, ``pec_ey_mask`` and
   ``pec_ez_mask``.
2. Non-finite values, normally ``np.inf``, in the electric material arrays.

For example:

.. code-block:: python

   eps_r_xx[pec_ex_mask] = np.inf
   eps_r_yy[pec_ey_mask] = np.inf
   eps_r_zz[pec_ez_mask] = np.inf

Each electric component is treated independently:

* ``pec_ex_mask`` constrains ``Ex`` to zero.
* ``pec_ey_mask`` constrains ``Ey`` to zero.
* ``pec_ez_mask`` constrains ``Ez`` to zero.

After masks are built, PEC material entries are replaced by finite placeholders
before matrix assembly:

.. code-block:: python

   eps_r_xx[self.pec_ex_mask] = 1.0 + 0j

This avoids artificial high-index material modes. The physics is carried by the
removed/constrained degrees of freedom, not by the placeholder value.

Large finite values such as ``1e8`` or ``1e10`` are intentionally not treated as
PEC. They are ordinary finite material values. gprMax should use ``np.inf`` for
electric PEC in eigenmode slices.

PMC and Magnetic Conductors
---------------------------

PMC is not implemented because gprMax currently supports PEC materials but does
not yet provide a PMC material workflow. The solver therefore raises
``NotImplementedError`` if:

* any ``mu_r_*`` array contains non-finite values, or
* any explicit magnetic conductor mask is supplied.

This is deliberate. Approximating PMC with a large permeability is not a
correct replacement for enforcing the magnetic tangential boundary condition.

The solver does reserve ``pmc_hx_mask``, ``pmc_hy_mask`` and ``pmc_hz_mask``
constructor arguments. These are currently fail-fast placeholders, but they
make the intended extension point explicit: once gprMax supports PMC material
sampling, the solver can implement magnetic-field constraints behind these
existing argument names.

Eigenproblem
------------

The solver constructs sparse finite-difference operators:

.. code-block:: python

   DEX = kron(Iy, Dx) / (k0 * dx)
   DEY = kron(Dy, Ix) / (k0 * dy)
   DHX = -DEX.H
   DHY = -DEY.H

The transverse electric field vector is:

.. code-block:: python

   Exy = [Ex, Ey]^T

The solver forms the standard vector FDFD ``P`` and ``Q`` matrices and solves:

.. code-block:: text

   Omega * Exy = eigenvalue * Exy
   Omega = P * Q

where the effective index is recovered from:

.. code-block:: python

   neff = sqrt(-eigenvalue)

The square-root branch is chosen so that the phase constant is positive and
attenuation is non-negative.

Degree-of-Freedom Reduction
---------------------------

PEC constraints are applied by removing constrained electric degrees of freedom
from the eigenproblem:

.. code-block:: python

   Omega = Omega[self.free_exy_mask, :][:, self.free_exy_mask]

After ARPACK returns the reduced eigenvectors, the solver expands them back to
the full field-vector size and explicitly zeros constrained fields.

The inverse ``eps_zz`` and ``mu_zz`` operators are also built only on free
degrees of freedom. Constrained entries receive zero inverse values so that no
division by ``np.inf`` or placeholder data affects the fields.

Spurious Mode Rejection
-----------------------

With PEC boundaries, ARPACK may return modes localized near or inside the PEC
region. These modes can appear in the requested eigenvalue range even when the
physical modes are correct.

By default, ``solve()`` enables spurious-mode rejection:

.. code-block:: python

   solver.solve(
       reject_spurious=True,
       extra_modes=8,
       max_pec_neighbor_energy_fraction=0.35,
   )

When rejection is enabled, the solver asks ARPACK for extra candidate modes:

.. code-block:: python

   candidate_modes = num_modes + extra_modes

It then scores each candidate by the fraction of transverse electric energy
located in a one-cell neighbourhood of the electric PEC masks:

.. code-block:: text

   score = energy_near_PEC / total_transverse_E_energy

Candidates with scores greater than
``max_pec_neighbor_energy_fraction`` are marked as spurious. The solver keeps
the first ``num_modes`` accepted candidates after eigenvalue sorting.

Diagnostic attributes are available after ``solve()``:

``spurious_scores``
    Energy-localisation score for every candidate mode.

``accepted_candidate_indices``
    Candidate indices used for the final returned modes.

``rejected_candidate_indices``
    Candidate indices whose score exceeded the rejection threshold.

``unselected_candidate_indices``
    Candidate indices not used in the final mode list.

If too few modes pass the threshold, the solver fills the missing modes from the
least PEC-localized rejected candidates. This avoids silently returning fewer
modes than requested, while still exposing the diagnostics needed to inspect the
result.

Field Reconstruction
--------------------

After selecting physical modes, the solver reconstructs:

* ``Ex`` and ``Ey`` directly from the transverse eigenvector.
* ``Hx`` and ``Hy`` from ``Q * Exy / sqrt(eigenvalue)``.
* ``Ez`` from transverse magnetic curl terms.
* ``Hz`` from transverse electric curl terms.

Magnetic fields are converted to physical A/m using ``eta0``:

.. code-block:: python

   H = 1j * H_normalized / eta0

The solver then zeroes all constrained fields to ensure the returned modal
fields satisfy the enforced constraints exactly.

Normalisation and Phase Alignment
---------------------------------

Modes are normalised to carry one watt of time-average power:

.. code-block:: text

   P = 0.5 * Re integral((E x H*) . z dA)

If a mode initially carries negative power, the magnetic field is flipped before
normalisation. After normalisation, each complex mode is phase-rotated so that
its real-valued field profile carries positive real-profile power. This makes
the plotted and injected fields easier to interpret.

gprMax Integration
------------------

``sources.py`` extracts complex material tensors from ``G.ID`` after the Yee
grid has been built. This is the correct integration point because the material
IDs are already sampled at the Yee component locations.

For electric materials:

* finite conductivity is folded into complex permittivity,
* ``se == inf`` is converted to ``np.inf + 0j``, which the solver treats as
  PEC.

For magnetic materials:

* finite magnetic conductivity is folded into complex permeability,
* ``sm == inf`` raises ``NotImplementedError`` because PMC is not supported.

Standalone Helper
-----------------

The static helper ``component_pec_masks_from_cell_mask()`` converts a
cell-centred PEC mask into approximate component masks:

.. code-block:: python

   pec_ex_mask, pec_ey_mask, pec_ez_mask = (
       FDFD_2D_mode_solver.component_pec_masks_from_cell_mask(cell_pec_mask)
   )

This helper is intended for standalone tests only. In production gprMax usage,
the component masks should come from component-sampled material IDs or from
``np.inf`` values in the component-sampled electric arrays.

Limitations
-----------

* The solver assumes all material arrays have the same ``(Nx, Ny)`` shape.
* PEC support is electric-only.
* PMC and magnetic conductor constraints are not implemented.
* Large finite permittivity is not PEC.
* The spurious-mode filter is a practical rejection heuristic, not a substitute
  for correct component-sampled PEC data.
* The finite-difference operators use the current first-order sparse difference
  construction in ``fdfd_2d_mode_solver.py``.

Recommended Usage
-----------------

For gprMax integration, prefer this path:

1. Extract ``eps_r_xx``, ``eps_r_yy`` and ``eps_r_zz`` from Yee electric
   component material IDs.
2. Mark electric PEC entries with ``np.inf``.
3. Extract finite ``mu_r_xx``, ``mu_r_yy`` and ``mu_r_zz`` from Yee magnetic
   component material IDs.
4. Construct ``FDFD_2D_mode_solver``.
5. Call ``solver.solve()`` with default spurious rejection enabled.
6. Use ``solver.modal_*`` fields for eigenmode source injection.
