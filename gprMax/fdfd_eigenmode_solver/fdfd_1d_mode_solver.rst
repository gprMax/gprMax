1D FDFD Eigenmode Solver
========================

``fdfd_1d_mode_solver.py`` supplies the scalar, Yee-staggered mode solve used
by eigenmode sources in gprMax 2D TM and TE domains.

Coordinates and Fields
----------------------

The local basis is ``(t, a, w)``:

* ``t`` is the one physical transverse coordinate;
* ``a`` is the invariant 2D axis;
* ``w`` is the in-plane propagation direction and source normal.

For ``N`` cells along ``t``, the native staggered shapes are:

.. code-block:: text

   E_t (N)      E_a (N+1)    E_w (N+1)
   H_t (N+1)    H_a (N)      H_w (N)

The TM reduction solves the node-sampled scalar field ``E_a`` and reconstructs
``H_t`` and ``H_w``. The TE reduction solves the cell-sampled scalar field
``H_a`` and reconstructs ``E_t`` and ``E_w``. No derivative is taken through
gprMax's artificial one-cell TM or two-cell TE invariant-axis thickness.

Integration
-----------

``EigenmodeSource`` samples component materials from the mode's live invariant
layer, supplies the corresponding PEC/PMC masks, and maps the returned line
profiles back into the thin 3D Yee arrays used by the CPU update kernels.
Inactive components and TE outer boundary planes are explicitly zero.

Modes are ordered using the same shift-invert convention as the full-vector
2D solver, use the passive positive effective-index branch, and are normalised
to one watt per metre of invariant-axis length. ``plot_fields`` writes one row
per mode with line plots of all three active fields, including their staggered
sample locations.
