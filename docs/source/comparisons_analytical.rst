.. _analytical-comparisons:

**********************
Analytical comparisons
**********************

This section compares production gprMax results with independent analytical
solutions. These are correctness validations: unlike the
:ref:`inter-code studies <numerical-comparisons>`, the reference is not the
output of another numerical solver.

Two levels are provided. Compact tests in ``tests/ntff`` exercise the complete
FDTD and near-to-far-field path during routine pytest runs. The higher
resolution studies in ``testing/validation`` are deliberately run manually;
they write plots, CSV data, a machine-readable summary, and a report containing
the numerical acceptance checks. The large HDF5 solver files are reproducible
working data and are not the analytical reference.

Hertzian dipole in free space
=============================

:download:`hertzian_dipole_fs_analytical.in <../../testing/models_basic/hertzian_dipole_fs_analytical/hertzian_dipole_fs_analytical.in>`

This example is a Hertzian electric dipole, i.e. an additive electric-current
source, in free space.

.. literalinclude:: ../../testing/models_basic/hertzian_dipole_fs_analytical/hertzian_dipole_fs_analytical.in
    :language: none
    :linenos:

The function ``hertzian_dipole_fs`` in ``testing/analytical_solutions.py``
evaluates the complete free-space time-domain solution at the Yee-component
positions. The relative source and receiver positions are important because
the six field components are spatially staggered.

.. _hertzian_dipole_fs_results:

.. figure:: ../../images_shared/hertzian_dipole_fs_analytical.png
    :width: 700 px

    Electric and magnetic field histories from gprMax and the analytical
    solution (``Ref``).

.. _hertzian_dipole_fs_results_diffs:

.. figure:: ../../images_shared/hertzian_dipole_fs_analytical_diffs.png
    :width: 700 px

    Percentage differences between the numerical and analytical fields.

The largest difference is approximately one percent in the electric component
parallel to the source. The remaining electric components are within about
0.5 percent and the magnetic components within about 0.25 percent.

Plane-wave reflection from dispersive half spaces
=================================================

The normal-incidence validation calculates the incident waveform in free
space, repeats the model with a planar half space, and obtains the reflected
field by subtraction. The complex reflection coefficient is compared with the
Fresnel result

.. math::

    \Gamma(\omega) = \frac{\eta_2(\omega)-\eta_1(\omega)}
                           {\eta_2(\omega)+\eta_1(\omega)},

where the frequency-dependent wave impedance is evaluated from the same
dielectric, conductivity, Debye, Lorentz, or Drude parameters used to define
the material. Propagation phase between the receiver and the effective Yee
interface is included explicitly.

:download:`Dispersive half-space driver <../../testing/validation/validate_plane_wave_dispersive_halfspace.py>`
tests a nondispersive dielectric, one- and three-pole Debye media, and
multi-pole Lorentz and Drude media over 0.25--8 GHz.

.. figure:: ../../images_shared/plane_wave_halfspace_reflection_magnitude.png
    :width: 750 px

    Magnitude of the gprMax reflection coefficients and the corresponding
    Fresnel solutions.

Across the retained result set, magnitude RMS errors are between
:math:`3.3\times10^{-4}` and :math:`1.2\times10^{-3}`. The maximum phase RMS
error is 0.021 degrees. These comparisons exercise the discrete plane-wave
source and the complete multipole material-update path, rather than only
evaluating the material formula in isolation.

:download:`Realistic-material driver <../../testing/validation/validate_plane_wave_realistic_materials.py>`
adds fresh water at 25 degrees Celsius and Puerto Rico clay with ten percent
volumetric water content over frequency ranges selected to show their
dispersion.

.. figure:: ../../images_shared/plane_wave_realistic_material_reflection.png
    :width: 750 px

    Complex Fresnel comparison for the fresh-water and clay models.

.. _rational-network-validation:

Rational lumped networks in a parallel-plate guide
===================================================

The :download:`rational-network validation driver
<../../testing/validation/validate_rational_network_literature.py>` uses the
finite-width parallel-plate guide introduced by Pereda et al. [PER1999]_. Its
width is :math:`a=15` mm, its plate separation is :math:`b=2` mm, and the
1 mm Yee mesh gives 15 effective parallel paths, each containing two series
electric edges. For a one-edge network impedance :math:`Z(\omega)`, the exact
TEM quantities are therefore

.. math::

    Z_\mathrm{guide}=\eta_0\frac{b}{a},\qquad
    Z_\mathrm{sheet}=\frac{2Z}{15},\qquad
    \Gamma=-\frac{Z_\mathrm{guide}}
                   {2Z_\mathrm{sheet}+Z_\mathrm{guide}}.

The FDTD result is obtained from an empty-guide incident run and a loaded-guide
run. The first reflected pulse is isolated, transformed using the engineering
Fourier convention, and de-embedded to the sheet with the axial Yee numerical
wavenumber. Separate resistor, capacitor, and inductor sheets exercise the
direct :math:`G`, direct :math:`sC`, and zero-pole terms. The two literature
cases exercise a real pole for a series RC network and a conjugate pole pair
for a series RLC network.

.. figure:: ../../images_shared/rational_network_pereda_1999_reflection.png
    :width: 850 px

    Production gprMax FDTD reflection (symbols) compared with the exact
    continuous-time network result (lines) for the two examples of
    [PER1999]_.

From 1--30 GHz, the series-RC comparison has magnitude and phase RMS errors of
0.0027 and 0.15 degrees; the series-RLC errors are 0.0057 and 0.37 degrees.
The RC capacitance printed in [PER1999]_ is 0.02 pF, but both magnitude and
phase curves published in that paper correspond to 0.2 pF; the latter value
is used for the reproduced curve above.

The arbitrary one-port coupling follows the lumped-network FDTD formulations
of [PER1999]_ and [CHE2007]_. gprMax does not retain their classic PLRC time
placement unchanged. Each partial-fraction term is advanced using the
exponential recursive-convolution approach of Giannakis and Giannopoulos
[GIA2014]_, evaluated directly at :math:`n+1/2` for a linearly varying
terminal voltage. This avoids estimating the half-step network current by
averaging adjacent integer-time pole currents.

Hertzian-dipole pattern, directivity, and near field
====================================================

The :download:`Hertzian-dipole validation driver
<../../testing/validation/validate_hertzian_dipole.py>` compares both KSIR and
equivalent-current far fields with the analytical :math:`\sin\theta` field
pattern and :math:`D_\mathrm{max}=1.5` directivity. It also compares one
directly sampled and one KSIR-predicted time-domain near-field component with
the complete analytical field.

.. figure:: ../../images_shared/hertzian_dipole_far_field.png
    :width: 750 px

    Far-field pattern and directivity from both transformations compared with
    the Hertzian-dipole closed form.

.. figure:: ../../images_shared/hertzian_dipole_near_field.png
    :width: 750 px

    Direct FDTD and KSIR time-domain near fields compared with the analytical
    component at the same physical position.

The retained result has directivity-pattern RMS errors below
:math:`10^{-4}` for KSIR and :math:`2.5\times10^{-5}` for the
equivalent-current method. The significant-signal relative L2 errors of the
direct and KSIR near fields are :math:`3.4\times10^{-4}` and
:math:`1.7\times10^{-4}`, respectively.

Sphere radar cross section
==========================

The sphere studies illuminate a staircased sphere with a broadband discrete
plane wave and calculate monostatic RCS using the production NTFF path. The
reference values are evaluated independently from converged Mie series at the
actual requested frequencies.

.. figure:: ../../images_shared/pec_sphere_backscatter_rcs_validation.png
    :width: 750 px

    Broadband PEC-sphere backscatter compared with the PEC Mie series.

The :download:`PEC-sphere driver
<../../testing/validation/validate_pec_sphere_rcs.py>` covers
:math:`0.25<ka<3.02`, including the first resonances and nulls. With a 0.5 mm
grid and 16 mm radius, the retained result has 0.44 dB RMS error and 0.95 dB
maximum error across 34 independent frequency samples.

.. figure:: ../../images_shared/dielectric_sphere_backscatter_rcs_validation.png
    :width: 750 px

    Backscatter from a lossless :math:`\epsilon_r=4` sphere compared with the
    homogeneous dielectric-sphere Mie series.

The :download:`dielectric-sphere driver
<../../testing/validation/validate_dielectric_sphere_rcs.py>` uses the same
exterior geometry and frequency range. Its retained result has 0.27 dB RMS
error and 0.72 dB maximum error. This case additionally exercises dielectric
geometry averaging and the shorter wavelength inside the sphere.

Dispersive spheres and interface averaging
===========================================

Hartley's effective-permittivity treatment extends the ordinary arithmetic
electric-edge average to Debye media [HAR2020]_. For the engineering Fourier
convention used by gprMax, constituent material :math:`m` has

.. math::

    \epsilon_{r,m}(\omega) =
    \epsilon_{\infty,m}
    + \sum_p \frac{\Delta\epsilon_{m,p}}
                       {1 + j\omega\tau_{m,p}}
    + \frac{\sigma_m}{j\omega\epsilon_0}.

If the surrounding cells occupy fractions :math:`w_m` of the contour-path
area, with :math:`\sum_m w_m=1`, the effective edge response is

.. math::

    \epsilon_{r,\mathrm{eff}}(\omega)
    = \sum_m w_m\epsilon_{r,m}(\omega).

Consequently,

.. math::

    \begin{aligned}
    \epsilon_{\infty,\mathrm{eff}}
      &= \sum_m w_m\epsilon_{\infty,m}, \\
    \sigma_{\mathrm{eff}}
      &= \sum_m w_m\sigma_m, \\
    \Delta\epsilon_{\mathrm{eff},m,p}
      &= w_m\Delta\epsilon_{m,p}, \\
    \tau_{\mathrm{eff},m,p}
      &= \tau_{m,p}.
    \end{aligned}

The four cells surrounding a standard electric Yee edge each contribute
:math:`w_m=1/4`. Repeated relaxation times are combined exactly; distinct
times remain distinct poles. Thus two different single-pole Debye materials
generally produce a two-pole interface material. This construction follows
Chapter 4 of Hartley's thesis [HAR2020]_.

The implementation extends the same arithmetic response to Lorentz and Drude
media through the inclusive susceptibility used by gprMax [GIA2014]_. If
:math:`W_{m,p}` and :math:`Q_{m,p}` are the residue and location of a Debye,
Lorentz, or Drude term, the interface retains :math:`Q_{m,p}` and replaces
:math:`W_{m,p}` by :math:`w_m W_{m,p}`. Identical locations are merged exactly.
This gives the frequency-by-frequency arithmetic constitutive response without
fitting. It may, however, increase the model-wide pole count, so dispersive
averaging is an opt-in feature.

Reducing this exact union is a separate, band-limited material-fitting
problem. The validation package includes constrained fitting experiments,
building on the type of hybrid optimisation used for Debye fitting in
[KEL2007]_, but the production solver never replaces the exact interface
response automatically.

The :download:`Debye-sphere averaging driver
<../../testing/validation/validate_debye_sphere_averaging.py>` reproduces the
three curved-interface experiments from Section 4.4 of the thesis. A
:math:`200^3` grid contains a sphere of radius 60 cells made from one of three
two-pole Puerto Rico clay/loam models. A 300 MHz discrete plane wave excites
the model, and production equivalent-current NTFF output is compared from 25
to 525 MHz with an independent homogeneous dispersive-sphere Mie series.

.. figure:: ../../images_shared/debye_sphere_averaging_soil_a.png
    :width: 750 px

    Type A clay/loam (2.5% moisture): analytical Mie backscatter compared
    with dispersive-averaged and non-averaged FDTD spheres.

.. figure:: ../../images_shared/debye_sphere_averaging_soil_b.png
    :width: 750 px

    Type B clay/loam (5% moisture). Debye averaging follows the resonant
    minima more closely than the staircased interface.

.. figure:: ../../images_shared/debye_sphere_averaging_soil_c.png
    :width: 750 px

    Type C clay/loam (10% moisture). Errors in dB become very large at the
    exceptionally deep analytical null, so global relative L2 error is the
    more representative metric.

The retained 300 ns simulations give:

.. list-table:: Debye-sphere backscatter errors
    :header-rows: 1
    :widths: 16 18 18 18 18

    * - Soil
      - Averaged RMS error
      - Staircased RMS error
      - Averaged relative L2
      - Staircased relative L2
    * - A, 2.5%
      - 0.371 dB
      - 1.728 dB
      - 3.39%
      - 13.71%
    * - B, 5%
      - 1.011 dB
      - 2.476 dB
      - 3.58%
      - 11.57%
    * - C, 10%
      - 1.734 dB
      - 3.090 dB
      - 3.14%
      - 8.85%

The relative L2 error is reduced by factors of approximately 4.0, 3.2, and
2.8 for soils A, B, and C, respectively. Unlike the planar half-space test,
these curved objects expose the geometric and accumulated phase errors that
the interface average is intended to reduce.

Mixed dispersive layers and a core-shell sphere
------------------------------------------------

The reusable validation modules in
``testing/validation/dispersive_averaging`` extend the comparison to mixed
dispersion families. Planar stacks are evaluated with the normal-incidence
multilayer recursion using the complex permittivity of every layer. All FDTD
results are de-embedded to the geometrical first interface requested by the
model; no fitted half-cell displacement is applied.

.. figure:: ../../images_shared/dispersive_multilayer_reflection.png
    :width: 750 px

    A three-pole Debye layer and a two-pole Lorentz layer compared with the
    analytical multilayer reflection coefficient. Symbols are FDTD samples;
    the continuous black curve is evaluated independently.

Across the retained dielectric, Debye/dielectric, Debye/Lorentz,
Lorentz/Drude, and multipole Debye/Lorentz stacks, exact interface averaging
gives complex relative L2 errors between :math:`4.3\times10^{-4}` and
:math:`8.1\times10^{-4}`. The corresponding staircased errors are 1.85--3.70%.
The :download:`multilayer validation driver
<../../testing/validation/dispersive_averaging/validate_multilayer_fdtd.py>`
also records magnitude and phase residuals at every FDTD frequency.

The curved mixed-material test is a 60 mm Debye core surrounded by a 100 mm
outer-radius Lorentz shell. Its exact backscatter reference uses the
Aden--Kerker coated-sphere series and the complex material response at every
frequency.

.. figure:: ../../images_shared/dispersive_core_shell_sphere_averaging.png
    :width: 750 px

    Refined Debye-core/Lorentz-shell backscatter. The analytical curve is
    densely sampled; averaged and staircased FDTD results are shown as symbols
    with pointwise error bars below.

On the 2.5 mm grid, the exact pole-residue average reduces the complex
relative L2 error from 8.77% to 6.01% and the RMS logarithmic error from
0.756 dB to 0.507 dB. Run the :download:`coated-sphere validation driver
<../../testing/validation/dispersive_averaging/validate_core_shell_fdtd.py>`
to regenerate the full comparison.

Running the validation suite
============================

Run the studies from the repository root, for example:

.. code-block:: none

    python -m testing.validation.validate_plane_wave_dispersive_halfspace --gpu 0
    python -m testing.validation.validate_plane_wave_realistic_materials --gpu 0
    python -m testing.validation.validate_hertzian_dipole --gpu 0
    python -m testing.validation.validate_rational_network_literature
    python -m testing.validation.validate_pec_sphere_rcs --gpu 0
    python -m testing.validation.validate_dielectric_sphere_rcs --gpu 0
    python -m testing.validation.validate_debye_sphere_averaging --gpu 0
    python -m testing.validation.dispersive_averaging.validate_multilayer_fdtd
    python -m testing.validation.dispersive_averaging.validate_core_shell_fdtd --gpu 0

Omit ``--gpu`` for CPU execution. The full-resolution plane-wave and sphere
cases are long-running and are not part of the default pytest selection. See
:download:`the validation README <../../testing/validation/README.rst>` for
the output layout, cache policy, and acceptance criteria.

Related half-wave dipole example
================================

The :ref:`wire-dipole antenna example <example-wire-dipole>` demonstrates the
simulated reflection coefficient, input impedance, directivity, and gain of a
finite half-wave dipole. Its resonant frequency and impedance can be compared
with classical thin-wire predictions, while the more detailed MoM comparison
is documented in the :ref:`numerical-comparison section
<numerical-comparisons>`.
