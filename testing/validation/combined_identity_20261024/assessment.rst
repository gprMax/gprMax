Combined mode-identity audit, 24 October 2026
===========================================

All examined square and circular PEC, copper, and dielectric cases recovered stable individual FDFD channels with the combined longitudinal-content, directional-gradient, total-gradient, and symmetry pipeline. No examined group required unresolved Procrustes transport. The earlier circular angular-pair counterexamples demonstrate a limitation of the three scalar operators alone; they do not demonstrate failure of the combined pipeline.

Scope and method
----------------

The audit requested the first 20 modes from 32 solver candidates at 39, 40, and 41 GHz, on 24-by-24 and 40-by-40 transverse meshes. Automatic completion of degenerate groups increased the retained count to 22 for square PEC and 21 for circular PEC/copper. Every retained mode was propagating. Each bank was processed once as solved and twice after independent seeded complex unitary rotations within numerically exact complex eigenspaces (relative eigenvalue tolerance 1e-11). These rotations also randomize singleton phases. The production automatic tracker, separation stages, symmetry certification, and physical validity checks ran without altered tolerances.

Square conductor guides are 20 mm wide and air filled. Circular conductor guides have a 10 mm radius in a 24 mm window, with voxel-to-Yee PEC masks or clipped Ampere surface-impedance rows. Copper conductivity is 5.96e7 S/m; the ADE fit and discrete response are recorded in each case. Dielectric cores have relative permittivity 4 in air: a 20 mm square or 10 mm radius circle, initially inside a 40 mm square PEC truncation window. The dielectric results concern hybrid discrete eigenmodes; the audit does not assign exact continuum HE/EH indices.

Results
-------

.. list-table:: Results on the 40-by-40 mesh, all three frequencies
   :header-rows: 1

   * - Case
     - Retained modes
     - Minimum independent longitudinal-template score
     - Maximum full-field residual, including rotated inputs
   * - Square PEC
     - 22
     - 1.000000
     - 5.98e-14
   * - Circular PEC
     - 21
     - 0.992102
     - 2.20e-14
   * - Square copper
     - 20
     - 0.999892
     - 6.90e-10
   * - Circular copper
     - 21
     - 0.992091
     - 1.53e-11
   * - Square dielectric
     - 20
     - Not a TE/TM template test
     - 3.74e-15
   * - Circular dielectric
     - 20
     - Not a TE/TM template test
     - 2.47e-15

Across both meshes and both randomized runs, squared individual-channel overlap with the original run exceeded 0.9999999999999. Channel ordering was also preserved; this conclusion does not rely on allowing permutations. All full power matrices were positive, every degenerate parent received C4v certification, and no separation or final transport trial was rejected. The worst full-field residual was 7.03e-10, below the production 1e-7 limit. Minimum adjacent-frequency squared channel overlap was 0.999496. Maximum baseline mirror-parity error was 8.19e-10. The corresponding raw records and verification.json retain the per-case values.

Square PEC patterns include TE/TM11, TE/TM12 and 21, TE/TM13 and 31, TE/TM23 and 32, and the directional TE02/20 pair. In particular, coincident TE and TM E copies in square PEC quartets are separated before their angular partners are oriented. Copper patterns remain close to these PEC references, with the expected finite-conductivity changes.

Circular conductor patterns include TE11, TM01, TE21, TE01, TM11, TE31, TE41, TM21, TE12, TM02, TE51, and TM31. All partners in these families were recovered consistently. On the C4v staircase, even-angular-order partners can already have distinct propagation constants; they must not be forced back into a continuum-degenerate pair. The odd-order E pairs receive a reproducible orientation. A continuum zero electric moment need not remain exactly zero on a C4v staircase, so some higher-order grid modes use electric-moment orientation and others use mirror parity. Both are checked against the complete vector fields.

Dielectric window check
-----------------------

The separate ../combined_identity_window_20261024 audit repeated both dielectric shapes at 1 mm spacing in 40 mm and 60 mm windows. The first 20 modes remained well above the air light line (minimum effective index above 1.82 in the primary audit). Comparing fields in the common 40 mm region gave minimum squared overlaps of 0.999999999999567 for the square and 0.999999999999728 for the circle. Maximum relative changes in effective index were 3.39e-13 and 2.28e-13 respectively. Thus the outer PEC window did not materially determine the identities of these selected, confined modes.

Interpretation
--------------

For these mode banks, the combined operations suffice to recover individual physical channels under the chosen axis/parity convention, without unresolved fallback. The tests establish this for the selected discrete modes and geometry/frequency range. They do not constitute a completeness theorem for arbitrarily high mode orders, exact coincidences between otherwise indistinguishable equivalent irreps, or weakly confined modes near cutoff. No such unresolved physical counterexample was observed here.

The code's decomposition of equivalent symmetry copies can use a numerical probe. Symmetry certification alone therefore does not prove a unique physical decomposition in an arbitrary repeated-irrep space. The independent template checks and raw-basis rotation checks provide additional evidence in this audit. Dielectric identification here means stable, physically valid hybrid FDFD channels, rather than a new analytic HE/EH classification system.

Reproduction and verification
----------------------------

Run from the repository root with the gprMax2 Python environment and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1::

   python -m testing.eigenmode_support.identification.combined_identity_audit --output NEW_OUTPUT_DIRECTORY
   python -m testing.eigenmode_support.identification.combined_identity_window_check --output NEW_WINDOW_OUTPUT_DIRECTORY

The manifests record source hashes, solver setup, frequencies, meshes, random seeds, timestamps, and per-case completion. JSON outputs contain the raw transforms, mode signatures, trial diagnostics, power, residual, and field-overlap measurements. run.log preserves execution output. The targeted separation, copper-symmetry, circular-operator, parity, and conditioning regression suite passed 70 tests; pytest.log records that run. This investigation adds audit scripts and results; it makes no additional runtime-pipeline changes.
