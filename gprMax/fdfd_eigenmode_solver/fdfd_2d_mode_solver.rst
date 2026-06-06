2D FDFD Eigenmode Solver
========================

Overview
--------

``fdfd_2d_mode_solver.py`` contains ``FDFD_2D_mode_solver``, a 2D
full-vector finite-difference frequency-domain eigenmode solver used by gprMax
to generate modal fields for eigenmode sources.

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

Local Yee Shapes
----------------

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

Inputs
------

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
    Yee component locations listed in `Local Yee Shapes`_.

``pec_u_mask``, ``pec_v_mask``, ``pec_w_mask``
    Optional explicit boolean masks for constrained electric degrees of
    freedom. They must match the ``E_u``, ``E_v`` and ``E_w`` shapes.

``pmc_u_mask``, ``pmc_v_mask``, ``pmc_w_mask``
    Optional explicit boolean masks for constrained magnetic degrees of
    freedom. The solver can consume these masks, but gprMax does not currently
    provide a PMC material workflow; ``sources.py`` still rejects ``sm == inf``.

``guess``
    Optional ARPACK shift. If omitted, the solver chooses a conservative shift
    from the largest finite material magnitude.

Array Ordering
--------------

All component arrays are flattened with Fortran order:

.. code-block:: python

   flat = array.ravel(order="F")

and modal vectors are reshaped back with:

.. code-block:: python

   array = vector.reshape((*shape, num_modes), order="F")

There is no axis-order switch. gprMax must pass local ``u``/``v`` slices in the
same native transverse ordering used by the extracted source plane.

PEC Handling
------------

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

Eigenproblem
------------

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

Degree-of-Freedom Reduction
---------------------------

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

Field Reconstruction
--------------------

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

Normalisation and Phase Alignment
---------------------------------

Modes are normalised to carry one watt of time-average power. Power is computed
from cell-centred transverse Poynting flux by averaging the staggered
transverse fields onto local cells:

.. code-block:: text

   P = 0.5 * Re integral((E_u * H_v* - E_v * H_u*) dA)

If a mode initially carries negative power, the magnetic field is flipped
before normalisation. After normalisation, each complex mode is phase-rotated
so that its real-valued field profile carries positive real-profile power.
This makes plotted and injected real fields easier to interpret.

gprMax Integration
------------------

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
* ``sm == inf`` raises ``NotImplementedError`` because the gprMax material path
  does not currently support PMC eigenmode slices.

After solving, ``sources.py`` maps local modal fields back to global component
slots. The Cython injection kernels consume the transverse components with
their native staggered shapes; longitudinal modal fields are stored but are not
used for TF/SF source corrections.

PEC Boxes and ``constrain_all_edges``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eigenmode sources need the FDFD mode solve and the time-domain FDTD injection
to see the same PEC boundary. This is stricter than ordinary geometry
placement because the FDFD solver removes constrained electric degrees of
freedom from the modal eigenproblem, while the FDTD source later injects the
same modal fields through the Yee-grid update coefficients.

A PEC ``#box`` is specified by cell-centred extents, but electric fields live
on component-specific Yee edges. The legacy non-averaged box builder only
assigned a limited set of component IDs around each PEC cell, with selected
high-side faces handled separately. That behaviour is preserved for existing
models without eigenmode sources.

For a PEC-loaded waveguide eigenmode source, however, this limited assignment
can leave a tangential electric edge active on one side of a PEC interface. The
FDFD solve may still classify the nearby cell as PEC through the supplemental
cell mask, so the solved mode satisfies a PEC boundary there. The FDTD update,
using ``G.ID`` and ``G.updatecoeffsE``, then still treats the unmatched edge as
an active field location. The result is an impure injected modal field because
the source is adding a clean FDFD mode onto a slightly different FDTD boundary
condition.

The ``constrain_all_edges`` option fixes this mismatch for PEC boxes used with
eigenmode sources. When enabled, the box geometry path assigns every Yee
electric edge touched by each PEC voxel, including high-side edges. This makes
the component material IDs, the FDTD update coefficients, and the FDFD PEC
masks agree at the source cross-section.

The option is intentionally gated in ``gprMax/user_objects/cmds_geometry/box.py``:

.. code-block:: python

   constrain_all_edges = bool(grid.eigenmodesources) and any(
       getattr(material, "se", 0) == float("inf") for material in materials
   )

This keeps the old PEC box behaviour when no eigenmode source is present, and
only uses the stricter edge assignment for PEC boxes in models that actually
need modal FDFD/FDTD consistency. gprMax builds grid objects, including
eigenmode sources, before geometry objects, so ``Box.build()`` can safely check
``grid.eigenmodesources``.

The related implementation points are:

* ``gprMax/user_objects/cmds_geometry/box.py`` decides when
  ``constrain_all_edges`` should be enabled.
* ``gprMax/cython/geometry_primitives.pyx`` implements the stricter box build
  path by calling ``build_voxel()`` for every PEC cell when
  ``constrain_all_edges`` is true.
* ``gprMax/sources.py`` builds local source-plane PEC masks with native Yee
  electric component shapes and passes them into the FDFD solver.
* ``gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py`` consumes the explicit
  component PEC masks and removes constrained electric degrees of freedom from
  the eigenproblem.

The 2D TM artificial PEC boundaries do not use ``box.py``. They are applied by
the grid setup code for ``2D TMx``, ``2D TMy`` and ``2D TMz`` models, so the
``constrain_all_edges`` gate does not change those boundary conditions.

Limitations
-----------

* Material tensors are diagonal in the local ``u``/``v``/``w`` basis.
* Electric PEC constraints are supported and are used by gprMax eigenmode
  sources.
* The solver has magnetic constraint masks, but gprMax does not yet provide a
  PMC material workflow for eigenmode sources.
* Large finite permittivity is not PEC.
* The finite-difference operators use first-order sparse Yee-grid differences.

Recommended Usage
-----------------

For gprMax integration, use this path:

1. Extract local ``eps_r_uu``, ``eps_r_vv`` and ``eps_r_ww`` from Yee electric
   component material IDs with native staggered shapes.
2. Mark electric PEC entries with ``np.inf + 0j`` or explicit local PEC masks.
3. Extract local ``mu_r_uu``, ``mu_r_vv`` and ``mu_r_ww`` from Yee magnetic
   component material IDs with native staggered shapes.
4. Construct ``FDFD_2D_mode_solver`` using local ``du`` and ``dv``.
5. Call ``solver.solve()``.
6. Use ``solver.modal_Eu``, ``solver.modal_Ev``, ``solver.modal_Hu`` and
   ``solver.modal_Hv`` for transverse eigenmode source injection after mapping
   local components back to global gprMax components.
