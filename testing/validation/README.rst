Validation models
=================

This directory contains reproducible, quantitative validation models. The
retained cases compare production gprMax outputs with independent analytical
solutions rather than with saved output from another gprMax version.

The main validations are:

* ``validate_plane_wave_dispersive_halfspace.py`` -- normal-incidence Fresnel
  reflection for dielectric, multipole Debye, Lorentz, and Drude materials;
* ``validate_plane_wave_realistic_materials.py`` -- Fresnel reflection for
  fresh water and Puerto Rico clay over their dispersive bands;
* ``impedance_surface/validate_copper_wall_waveguide.py`` -- a TE10 validation
  sequence for a copper-preset rectangular guide at 130--150 GHz,
  using a 120--150 GHz excitation, automatic three-pole fit, and a causally
  isolated 500 ps record; its FDFD and FDTD attenuation gates are 1% and 2%
  relative L2 error;
* ``validate_hertzian_dipole.py`` -- Hertzian-dipole far-field pattern and
  directivity, plus one analytical near-field time-domain component;
* ``planar_layered_ntff/validate_point_dipole.py`` -- the frequency-domain
  planar-layered Huygens transform for a lossy, magnetic three-layer stack,
  compared with the closed-form layered Hertzian-dipole Green function;
* ``planar_layered_ntff/validate_capoglu_grounded_time.py`` -- the direct
  time-domain transform for horizontal and vertical Hertzian dipoles in the
  PEC-grounded dielectric slab of Figure 11 in Çapoğlu's thesis, compared
  with its independent short-circuit echo series;
* ``planar_layered_ntff/validate_grounded_dipoles.py`` -- electric and
  magnetic dipoles over bare PEC against exact image theory, plus electric
  dipoles over a dielectric-coated PEC against an independently evaluated
  short-circuited TE/TM spectrum;
* ``planar_layered_ntff/validate_grounded_slab_reflection.py`` -- broadband
  axial-DPW reflection magnitude and phase from a PEC-backed dielectric slab
  against its exact short-circuited transmission-line input impedance;
* ``tests/ntff/test_layered_time.py`` -- direct time-domain planar-layered
  impulse amplitudes and delays against the dielectric half-space and
  ungrounded-slab expressions in Eqs. (53)--(55) and (71)--(75) of Çapoğlu's
  thesis, followed by random-stack and production-FDTD cross-checks;
* ``planar_layered_ntff/validate_capoglu_paper.py`` -- the complete
  nine-dipole, eight-layer Figure 2 benchmark of Çapoğlu *et al.*, including
  spectral and angular TE/TM results;
* ``planar_layered_ntff/validate_engheta_interfacial_dipoles.py`` -- vertical
  and horizontal interfacial-dipole patterns for refractive-index ratios two
  and four, compared with the closed asymptotic power expressions of Engheta,
  Papas, and Elachi;
* ``planar_layered_ntff/validate_smith_dipole_height.py`` -- E- and H-plane
  patterns of a horizontal dipole at three heights above a dielectric
  half-space, compared with Smith's asymptotic solution;
* ``planar_layered_ntff/validate_gssi_energy_convergence.py`` -- finite-radius
  pulse-energy patterns of the GSSI-like 1.5 GHz antenna over a lossless
  half-space, verifying the convergence reported by Warren and Giannopoulos
  against a broadband layered-NTFF asymptote;
* ``validate_fdfd_eigenmodes.py`` -- effective index of 1D PEC
  parallel-plate and dielectric slab modes, plus 2D rectangular and cylindrical
  PEC waveguide modes, against analytical dispersion;
* ``validate_rational_network_literature.py`` -- elementary R, C, and L
  terminal sheets and the series-RC/series-RLC loaded-guide examples of
  Pereda et al. against their exact TEM reflection coefficients;
* ``validate_dielectric_sphere_rcs.py`` -- broadband dielectric-sphere
  monostatic RCS against the homogeneous-sphere Mie series;
* ``validate_debye_sphere_averaging.py`` -- three two-pole dispersive-soil
  spheres with and without dispersive interface averaging, compared with
  the homogeneous dispersive-sphere Mie series;
* ``validate_pec_sphere_rcs.py`` -- broadband PEC-sphere monostatic RCS
  against the Mie series;
* ``validate_sar_lossy_halfspace.py`` -- local SAR in a conductive dielectric
  half space against the exact Fresnel/transmission solution;
* ``validate_sar_lossy_sphere.py`` -- total absorbed power obtained by
  integrating tagged-cell SAR in a lossy sphere, and the incident-flux
  normalised radiometric absorption cross-section, against the exact Mie
  absorption solution;
* ``validate_sar_2d_cylinder.py`` -- 2-D TM and TE local SAR plus absorbed
  power per unit length in homogeneous lossy cylinders against the exact
  cylindrical-Mie series. The muscle case uses parameters reported by
  Gasmelseed (2026), whose graphical cylinder comparison does not include
  numerical error estimates; the reported errors, fat and skin cases, and TE
  solution are explicitly additional validations performed here;
* ``plot_sar_2d_cylinder_materials.py`` -- combines completed fat, skin, and
  muscle results into 2-D analytical/FDTD/error maps, boundary-depth error
  profiles, and material/polarisation error summaries without rerunning FDTD;
* ``validate_sar_power_normalisation.py`` -- end-to-end incident- and
  accepted-port-power normalisation identities;
* ``validate_sar_spatial_averaging.py`` -- 1 g and 10 g spatial-average SAR
  against the independent STASIS implementation of IEC/IEEE 62704-1;
* ``validate_sar_star.py`` -- the complete uniform-grid 1 g and 10 g SAR Star
  distributions against the official IEC/IEEE 62704-1 supplemental data;
* ``fmcw/validate_paper_multilayer.py`` -- the short-pulse FMCW correction
  sequence of Eide *et al.* over 150--1200 MHz, compared with the exact
  normal-incidence reflection from a lossless multilayer; and
* ``rectangular_waveguide_partial_cutoff`` -- generalized TE10 transmission
  magnitude and phase across cutoff, compared with the analytical
  :math:`S_{21}=\exp(-j\beta L)` response; and
* ``eigenmode_multiport_deembedding`` -- a prescribed exact two-port matrix
  and an end-to-end rectangular-guide comparison validating the conditioned
  full incident-matrix solve used by eigenmode studies.

The ``dispersive_averaging`` subdirectory adds mixed-family validations for
half spaces, finite multilayers, construction-order sensitivity, and a
Debye-core/Lorentz-shell sphere evaluated with the Aden--Kerker series.

The ``impedance_surface`` subdirectory contains the copper-wall waveguide
validation. The driver writes CSV data, a PNG comparison figure, and a
machine-readable ``summary.json`` to its results directory. It uses dense
source-port anchors for exact FDFD validation and modal launch, but guarded
sparse anchor sets on passive propagation ports to avoid redundant mode solves.
The copper preset explicitly requests automatic fit order rather than fixing a
pole count.

``mie_pec.py`` and ``mie_dielectric.py`` supply the independent sphere series
used by both manual validation and automated tests. Behavioural and
backend-consistency suites without analytical reference solutions are kept
outside this directory under ``testing/regression`` and
``testing/backend_consistency``.

Rectangular-waveguide cutoff
----------------------------

The partial-cutoff model samples 100 frequencies across the TE10 transition.
Seven bins are below cutoff: their generalized modal coefficients remain
finite, while their separate physical-power-wave validity flags are false.
The retained plot compares gprMax directly with the analytical magnitude and
unwrapped phase; no auxiliary receiver-derived oracle is used.

.. figure:: rectangular_waveguide_partial_cutoff/rectangular_waveguide_partial_cutoff_s11_s21.png
   :alt: gprMax and analytical TE10 S21 magnitude and phase across cutoff
   :width: 100%

   Simulated and analytical TE10 transmission through a 12 mm reference-plane
   separation. Below cutoff, :math:`\beta=-j\alpha`, so the magnitude decays as
   :math:`\exp(-\alpha L)` and the phase is zero. Above cutoff, the ideal
   magnitude is 0 dB and the phase is :math:`-\beta L`.

Running validations
-------------------

Run modules from the repository root, for example::

    python -m testing.validation.validate_plane_wave_dispersive_halfspace --gpu 0
    python -m testing.validation.validate_plane_wave_realistic_materials --gpu 0
    python -m testing.validation.validate_hertzian_dipole --gpu 0
    python -m testing.validation.planar_layered_ntff.validate_point_dipole
    python -m testing.validation.planar_layered_ntff.validate_capoglu_grounded_time
    python -m testing.validation.planar_layered_ntff.validate_grounded_dipoles
    python -m testing.validation.planar_layered_ntff.validate_grounded_slab_reflection
    python -m testing.validation.planar_layered_ntff.validate_capoglu_paper --gpu 0
    python -m testing.validation.planar_layered_ntff.validate_engheta_interfacial_dipoles --gpu 0
    python -m testing.validation.planar_layered_ntff.validate_smith_dipole_height --gpu 0
    python -m testing.validation.planar_layered_ntff.validate_gssi_energy_convergence --gpu 0
    python -m testing.validation.validate_fdfd_eigenmodes
    python -m testing.validation.impedance_surface.validate_copper_wall_waveguide --threads 4
    python -m testing.validation.validate_rational_network_literature
    python -m testing.validation.validate_dielectric_sphere_rcs --gpu 0
    python -m testing.validation.validate_debye_sphere_averaging --gpu 0
    python -m testing.validation.validate_pec_sphere_rcs --gpu 0
    python -m testing.validation.validate_sar_lossy_halfspace --backend cuda --sweep
    python -m testing.validation.validate_sar_lossy_sphere --backend cuda --dl 0.000375
    python -m testing.validation.validate_sar_2d_cylinder \
        --backend cuda --material muscle
    python -m testing.validation.validate_sar_power_normalisation
    python testing/validation/validate_sar_spatial_averaging.py \
        --reference /path/to/IEC-IEEE-62704-1-spatial-average-SAR
    python -m testing.validation.validate_sar_star \
        /path/to/62704-1_supplemental_files.zip
    python -m testing.validation.fmcw.validate_paper_multilayer --gpu 0
    python -m testing.validation.dispersive_averaging.validate_multilayer_fdtd
    python -m testing.validation.dispersive_averaging.validate_core_shell_fdtd --gpu 0
    python -m gprMax \
        testing/validation/rectangular_waveguide_partial_cutoff/rectangular_waveguide_partial_cutoff.in \
        --hide-progress-bars
    python testing/validation/rectangular_waveguide_partial_cutoff/plot_partial_cutoff.py
    python -m testing.validation.eigenmode_multiport_deembedding.validate_rectangular_waveguide

Omit ``--gpu`` for CPU execution. The report-based drivers write a report,
summary, CSV data, and PNG figures. Their solver HDF5 and NumPy working data
are written below an ignored ``_cache`` directory. The partial-cutoff workflow
instead keeps its reproducible HDF5, CSV, and modal plots ignored beside the
model while retaining only the analytical comparison plot. Working data may
be retained locally for ``--reuse`` where supported, but it is not validation
evidence and must not be committed.

The spatial-average comparisons deliberately do not vendor their independent
oracles. Clone and build the Apache-2.0 STASIS reference repository separately
for the compact heterogeneous comparison. Download the official supplemental
archive from the IEC supporting documents for the full SAR Star comparison,
then pass those paths as shown above.

The complete SAR Star cases are standards-scale manual validations. On the
development server, reusable averaging geometry and compiled OpenMP processing
complete the 1 g and 10 g production calls over the 281-cubed reference grid
in approximately 20.9 and 36.9 seconds. The pre-optimisation implementation
took approximately 78.5 and 59.2 minutes, respectively. The compact STASIS
case remains the more convenient independent development check; the
exhaustive cases are retained for release validation.

Each analytical script applies conservative numerical tolerances after writing
its outputs and exits with a non-zero status if the comparison fails. The
partial-cutoff plotter applies 0.45 dB magnitude and 3-degree phase limits. The
``summary.json`` and report produced by the report-based drivers record the
individual checks and the overall ``PASS``/``FAIL`` result. The full-resolution
plane-wave and PEC-sphere cases
are intentionally long-running manual validations; the focused ``tests``
suite contains smaller end-to-end analytical checks for routine CI use.

The plots and compact numerical tables are intentionally retained so that a
developer can inspect the last full validation without rerunning every large
model. A changed solver should be assessed by regenerating these files, not by
comparing only with the previous gprMax output.
