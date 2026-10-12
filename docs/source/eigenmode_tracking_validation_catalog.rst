.. _eigenmode_tracking_validation_catalog:

Eigenmode tracking: analytical validation
=========================================

Validation has two folders: ``degenerate_subspace`` and ``branch_crossing``.
Every case compares tracked FDFD modal profiles and propagation constants
with an independent continuum solution. These are undriven eigenproblems.

Profiles show phase-aligned, separately normalized longitudinal fields.
The quantitative comparison uses all six components, with magnetic fields
scaled by the free-space impedance. For a degenerate analytical space,
the score is the squared projection onto its orthonormal span. Analytical
orientation is fitted within that space for display; numerical fields and
propagation constants are unchanged. A high subspace score does not claim a
unique individual orientation. Staircase-split cylindrical modes are also
compared with their complete smooth analytical pair. A square Cartesian
mesh retains C4v rather than continuous rotational symmetry, so not every
cylindrical pair remains numerically degenerate.

Degenerate subspace
-------------------

The cases are a C2v 2:1 rectangular PEC guide, a C4v square PEC guide,
a cylindrical PEC guide, and an open dielectric cylinder. The PEC runs
retain their original dates and source hashes. Compact figures were
regenerated from their original field archives; this is not a new PEC
simulation. The open dielectric cylinder is a new full-vector comparison.

For the vacuum-filled rectangular and square guides,

.. math::

   k_c^2=(m\pi/a)^2+(n\pi/b)^2, \qquad \beta=\sqrt{k_0^2-k_c^2}.

TE fields have :math:`H_z\propto\cos(m\pi x/a)\cos(n\pi y/b)`; TM fields have
:math:`E_z\propto\sin(m\pi x/a)\sin(n\pi y/b)`. The cylindrical PEC reference
uses :math:`J_m(k_c r)` with TE roots of :math:`J'_m(k_c a)=0` and TM roots of
:math:`J_m(k_c a)=0`. The comparison plots include every reported channel;
the compact field pages show the first sixteen channels.

.. list-table:: Cases and fine meshes
   :header-rows: 1
   :widths: 30 35 35

   * - Case
     - Geometry
     - Meshes
   * - C2v rectangular PEC
     - 20 × 10 mm; 35 reported channels
     - 64 × 32 and 96 × 48
   * - C4v square PEC
     - 20 × 20 mm; 18 reported channels
     - 64 × 64 and 96 × 96
   * - Cylindrical PEC
     - Radius 10 mm; 21 reported channels
     - 96 × 96 and 192 × 192; 24 mm window
   * - Open dielectric cylinder
     - Radius 10 mm; core εr = 4, air cladding; 16 channels
     - 192 × 192 and 256 × 256; 60 mm window

For the open dielectric cylinder, the exact reference matches all four
tangential components at the circular interface. Longitudinal fields use
:math:`J_m(ur/a)` inside and :math:`K_m(wr/a)` outside, with
:math:`u^2=a^2(\epsilon_c k_0^2-\beta^2)` and
:math:`w^2=a^2(\beta^2-\epsilon_s k_0^2)`. For hybrid modes,

.. math::

   \left[\frac{J'_m(u)}{uJ_m(u)}+\frac{K'_m(w)}{wK_m(w)}\right]
   \left[\frac{\epsilon_c J'_m(u)}{uJ_m(u)}+
   \frac{\epsilon_s K'_m(w)}{wK_m(w)}\right]
   =m^2\frac{\beta^2}{k_0^2}\left(\frac{1}{u^2}+\frac{1}{w^2}\right)^2.

The m = 0 TE and TM roots come from the first and second factors. This is a
full-vector solution; see the independent
`step-index vector-mode derivation <https://jayggg.github.io/fibermode/1_4_stepindex_vector.html>`_.
Hybrid labels give azimuthal order and characteristic-root order without
claiming an unverified HE/EH family assignment. Numerical cladding is
truncated by a distant PEC square; a larger-window check uses unchanged
spacing. Only guided modes above the air light line enter the comparison.

On the 256 × 256 mesh, minimum analytical-subspace overlap is 0.999398 and
maximum relative β error is 0.0181% over 39, 40, and 41 GHz. The 192 × 192
values are 0.993611 and 0.0276%. The retained smooth-cylinder PEC comparison
has a 0.90% maximum β error on its finer mesh; it is a different boundary
problem and includes higher modes with staircase sensitivity.

.. include:: ../../testing/validation/eigenmode_tracking/degenerate_subspace/gallery.rst

Branch crossing
---------------

A 20-mm square PEC bore is uniformly filled with
:math:`\operatorname{diag}(2,2,\epsilon_z)`. At 20 GHz the sweep varies
:math:`\epsilon_z` from 2.3 to 2.7. This is a material-parameter sweep, not a
causal dispersive frequency sweep. Independent continuum dispersion is

.. math::

   \beta_{\mathrm{TE},mn}^2=\epsilon_t k_0^2-k_{c,mn}^2, \qquad
   \beta_{\mathrm{TM},mn}^2=\epsilon_t k_0^2-
   \frac{\epsilon_t}{\epsilon_z}k_{c,mn}^2.

The persistent TE20/TE02 and TM12/TM21 doublets cross at
:math:`\epsilon_z=2.5`. Tracking keeps two doublets through the coincidence
and preserves their identities after raw eigenvalue ranks exchange. The
exact continuum crossing and displaced finite-difference crossing are
recorded separately. At each mesh's exact numerical coincidence, all four
input modes are deliberately mixed by a seeded complex unitary matrix;
the tracker must recover the two persistent pairs. Fine-mesh maximum relative β error is 0.00697%, and
minimum full-vector pair overlap is 0.999999992.

.. include:: ../../testing/validation/eigenmode_tracking/branch_crossing/gallery.rst

Reproduce
---------

Run from the repository root, using fresh output directories::

   python -m testing.validation.eigenmode_tracking.degenerate_subspace.validate_dielectric_cylinder --output results/cylindrical_dielectric --scratch results/cylindrical_dielectric_raw
   python -m testing.validation.eigenmode_tracking.branch_crossing.square_anisotropic_bore --output results/square_anisotropic_bore

Reproduce the three PEC analytical comparisons with::

   python -m testing.validation.eigenmode_tracking.degenerate_subspace.validate_pec_guides --output results/pec_guides --scratch results/pec_guides_raw

``plot_pec_comparison.py`` renders compact comparisons from a JSON report
and its original field archive. Numerical regression fixtures remain under
``testing/eigenmode_support``; they are not additional validation galleries.
Check the independent analytical fields and related tracking code with::

   python -m pytest tests/test_tracking_analytical_references.py tests/test_eigenmode_expanded_validation.py tests/test_waveguide_modal_profiles.py tests/test_cylindrical_modal_profiles.py
