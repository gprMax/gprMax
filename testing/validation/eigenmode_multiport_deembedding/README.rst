Eigenmode multiport de-embedding
================================

This validation separates the post-processing identity from the numerical
FDTD problem.

The prescribed analytical case defines a frequency-dependent two-port
scattering matrix :math:`S`, a non-diagonal incident matrix :math:`A`, and
forms :math:`B=SA` exactly. The production conditioned solve recovers
:math:`S` with a maximum Frobenius-norm error of
:math:`2.31\times10^{-16}`. Dividing each run only by its nominal source wave
has an error of at least :math:`2.17\times10^{-2}` for this prescribed case.

.. figure:: analytical_matrix_deembedding.png
   :alt: Exact two-port matrix recovery using full and diagonal incident matrices
   :width: 100%

   Recovery error and :math:`B=SA` residual for a known two-port response with
   deliberately non-zero passive-port incident waves.

The end-to-end case is a uniform, lossless PEC rectangular guide with two
virtual-waveguide ports. Its dominant-mode response is

.. math::

   S_{21}(f)=\exp[-j\beta(f)L], \qquad
   \beta(f)=\sqrt{(2\pi f/c)^2-(\pi/a)^2}.

Across 20--24 GHz the full solve has a maximum magnitude error of 0.001829 in
linear magnitude and a maximum circular phase error of 1.243 degrees relative
to that continuum solution. Its maximum measured network-equation residual is
:math:`2.93\times10^{-16}`; the diagonal approximation leaves a residual of
0.0379. The largest passive-to-active incident-wave ratio is 0.0215 and the
maximum incident-matrix condition number is 1.052.

.. figure:: rectangular_waveguide_deembedding.png
   :alt: Two-port gprMax de-embedding compared with analytical TE10 propagation
   :width: 100%

   TE10 transmission magnitude and phase, followed by the measured
   :math:`B=SA` residual. In this coarse model the diagonal approximation has
   an accidentally smaller magnitude error against the ideal continuum guide;
   it nevertheless fails the measured multiport network equation. This plot
   therefore validates the full de-embedding identity but does not claim that
   it removes spatial discretisation or modal-termination errors.

Run from the repository root::

    python -m testing.validation.eigenmode_multiport_deembedding.validate_rectangular_waveguide

Use ``--reuse`` to regenerate the CSV, JSON, and plots from the ignored
aggregate HDF5 result without repeating the FDTD cases.
