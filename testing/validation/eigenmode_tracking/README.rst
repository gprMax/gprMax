Eigenmode tracking validation
=============================

Only two groups are retained:

* ``degenerate_subspace``: PEC rectangular, square, and cylindrical guides,
  plus an open dielectric cylinder. Each has modal-profile and propagation-
  constant comparisons against independent analytical fields.
* ``branch_crossing``: two persistent doublets cross in a uniformly
  anisotropic square PEC bore. Numerical and exact continuum crossings are
  recorded separately.

The Sphinx validation page embeds both galleries. PEC JSON metadata retains
its original run provenance; compact figures were regenerated from the
original field archives. Original source hashes record run-file bytes before Git text-line-ending
normalization and may refer to paths before reorganization. Numerical test helpers moved to ``testing/eigenmode_support``.
Copper and the other former tracking galleries are removed, along with
``fdfd_eigenmode_results``. Raw field archives are local scratch products.

Run the new cases from the repository root::

   python -m testing.validation.eigenmode_tracking.degenerate_subspace.validate_dielectric_cylinder --output results/cylindrical_dielectric --scratch results/cylindrical_dielectric_raw
   python -m testing.validation.eigenmode_tracking.branch_crossing.square_anisotropic_bore --output results/square_anisotropic_bore
