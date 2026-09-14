.. _sources-ports:

*****************
Sources and ports
*****************

A source imposes an excitation; a port describes a terminal or modal reference
plane and its measured response. They are related, but not interchangeable.
This guide connects source selection to data interpretation and plotting.
The :doc:`input_hash_cmds` and :doc:`input_api` entries give syntax and
per-source output summaries; :doc:`output` owns the full file schemas.

Choose an excitation
====================

.. list-table:: Source and output families
   :header-rows: 1
   :widths: 23 37 40

   * - Source
     - Physical role
     - Port output
   * - Hertzian / magnetic dipole
     - Impressed electric / magnetic current
     - No circuit port; request receivers or NTFF
   * - Finite-resistance voltage
     - One electric-edge Thevenin generator and resistance
     - Automatic ``/ports/<id>``; received voltage and driven-port spectra
   * - Hard voltage
     - Direct electric-field prescription on one edge
     - Automatic 3-D port where the complete Ampere loop is available
   * - Transmission line
     - Explicit auxiliary 1-D line feeding an antenna
     - Automatic ``/tls/tlN``
   * - Magnetic frill
     - Coaxial continuous-wire feed through a ground plane
     - Automatic ``/frills/frillN``; not a bow-tie/dipole gap feed
   * - Rational network
     - A passive lumped admittance with optional generator
     - Explicit ``NetworkPort`` / ``#network_port``: ``/ports/<terminal ID>``
   * - Eigenmode
     - Launch/monitor guided modes at a declared cross-section
     - ``/eigenmode_ports/portN``; modal coefficients and scattering data
   * - Discrete plane wave
     - TFSF illumination for scattering or layered-media problems
     - No circuit port; total/scattered fields and incident-normalised RCS

Source amplitude units depend on the family: a voltage waveform is a
generator voltage, not necessarily the incident or terminal voltage. In
particular, a frill's incident voltage is half its Thevenin generator
waveform. A modal power specification is not a voltage amplitude.
Read the selected command's amplitude definition before comparing sources.

Raw fields, terminal signals and time axes
==========================================

A receiver records local Yee fields or an Ampere-loop current. A port
records a physical terminal or modal response and applies its own
corrections. ``Ez`` is in V/m, not V, and a raw field/source ratio is
not a power-wave S21. A gap voltage has the source convention
:math:`V=-E_{\parallel}\Delta l`.

Always use stored ``time``, ``time_current``, ``frequency`` and
timing metadata where supplied. Do not construct every port's time axis
from the receiver count or assume voltage and current occupy the same Yee
time level. Finite-resistance voltage ports use N-1 aligned half-step
samples; hard ports retain N samples, including the initial voltage at time
zero. Transmission-line currents need both temporal and spatial
de-embedding, while frill currents use adjacent-half-step averaging.
The :doc:`output` reference describes each convention.

A hard waveform equal to zero still prescribes zero field while active.
For a hard ``impulse`` at ``start=0``, choose
:math:`0<\mathrm{stop}<\Delta t` to prescribe only E(0) and release the
edge afterward. ``stop=dt`` includes a second, zero-valued clamp.
Use the owning subgrid's dt, if applicable. Releasing the clamp changes
the boundary condition: that impulse cannot synthesise arbitrary later
hard-source drives under an unchanged clamp. See :doc:`inc_ImpulseResponse`.

.. _passive-receiving-port:

Driven transmitter and passive receiver
=======================================

Use a zero-amplitude waveform with a **finite** receiver resistance. This
retains the intended passive loading and records received gap voltage
without a separate field-to-voltage reconstruction. It is not a hard source
and does not remove the load. Set the underlying edge material deliberately:
its permittivity and conductivity affect the solved gap capacitance/loss.

The :ref:`small terminal example <terminal-port-example>` supplies matching
hash and Python models. For named voltage-source objects, after the solve:

.. code-block:: python

    time = receive.result.time
    received_voltage = receive.result.total_voltage

The equivalent HDF5 quantities are ``/ports/receive/time`` and
``/ports/receive/Vtotal``. The passive voltage source's own
source-normalised S11 and impedance masks are invalid because its generator
has no incident spectrum. To characterise that input, drive it in another
case with the other ports terminated; use :ref:`PortStudy <study-port>`.
A rational-network port also measures network current and can report passive
V/I impedance where defined; this is a different measurement contract.

Interpret spectra before plotting
==================================

For ordinary terminal outputs, use:

.. code-block:: console

    python -m gprMax.toolboxes.Plotting.plot_port model.h5 --list-ports
    python -m gprMax.toolboxes.Plotting.plot_port model.h5 --port feed --validity --save

The tool supports voltage/network ports, transmission lines and frills,
including subgrids. It reads stored results rather than recomputing S11.
Modal results use the plots/readers in :doc:`eigenmode_port`; full matrices
use the study examples in :doc:`studies`.

Respect each quantity's validity mask. An S11 zero and an impedance pole
need different handling. The normal wavelength limit, weak incident
spectrum, undeveloped/undecayed pulse and finite record length can all limit
interpretation. ``spectrum_limit='nyquist'`` retains research data;
it is not a mesh-convergence certificate. Extending a frequency grid does
not replace a longer record or a finer mesh.

For a voltage-port result, Python uses ``s11``, ``zin``,
``yin``, ``valid_s11``, etc.; HDF5 uses ``S11``, ``Zin``,
``Yin``, ``valid_S11``, etc. Read the authoritative frequency array:

.. code-block:: python

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    with h5py.File('model.h5', 'r') as output:
        port = output['ports/feed']
        f = port['frequency'][...]
        z = port['Zin'][...]
        valid = port['valid_Zin'][...].astype(bool)
    plt.plot(f[valid] / 1e9, z[valid].real, label='Resistance')
    plt.plot(f[valid] / 1e9, z[valid].imag, label='Reactance')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Impedance [Ohm]')
    plt.legend()

.. _port-power-accounting:

Ports, antenna power and NTFF
=============================

Far-field electric/magnetic components and directivity do not require a
terminal feed reference. Gain, realised gain and efficiency additionally
require accepted/incident power and a rectangular-window transform:

* Associate every monitored physical port, including zero-amplitude
  terminations, using ``NTFFAntennaPorts`` / ``#ntff_antenna_ports``
  for equivalent currents, or the corresponding ``KSIRAntennaPorts``
  command for conventional terminal normalisation.
* Voltage IDs are explicit or ``portN``; line/frill IDs are ``tlN``
  and ``frillN``. Rational networks use the terminal ID and require an
  explicit NetworkPort for every excited terminal.
* Monitor and include a passive network termination too if it is intended
  to be an external port load, rather than part of the antenna's internal
  loss. This choice changes the power-accounting boundary.
* Subgrid references are ``subgrid_id/port_id``, not HDF5 paths.
  All powers use the owning grid's spatial/temporal discretisation.
* Include signed accepted powers. A passive load can receive coupled energy
  and therefore contribute negative power into the antenna. Summing only
  driven ports gives a different efficiency definition.
* For modal power-normalised gain use an equivalent-current transform and
  NTFFAntennaPorts. A closed KSIR surface can reconstruct fields around a
  virtual-guide-fed antenna, but that does not enable KSIR's conventional
  terminal gain path for active modal sources.
* Do not mix active portless dipoles/plane waves into a port-normalised
  antenna experiment. Use incident-wave normalisation for RCS instead.

Radiation efficiency compares radiated with net accepted power; total
efficiency and realised gain also include mismatch relative to incident
power. A full-sphere quadrature and valid port powers are required, not just
a plotted principal-plane cut. See :ref:`output-ntff`.
The AntennaPatterns toolbox's finite-radius integrated field patterns are
not NTFF gain/directivity; see :doc:`inc_AntennaPatterns`.

.. _voltage-port-theory:

Voltage-port equations and timing
=================================

A supported 3-D ``#voltage_source`` automatically calculates the complex reflection
coefficient and input impedance of its single-cell feed edge. The hidden field
monitor is placed at the source coordinate; a separate ``#rx`` command is not
required. Two representative forms are:

.. code-block:: none

    #voltage_source: z 0.050 0.050 0.020 50 source_wave 0 10e-9 feed 10
    #voltage_source: z 0.060 0.050 0.020 0 source_wave 0 10e-9 ideal_feed nyquist 75

The automatic port is supported in domain-decomposed MPI CPU models. The source and
its internal field monitor belong to one rank; for a hard source, magnetic
halos are synchronised before the next current sample so an Ampere loop may
cross an internal rank face or corner. Port histories are gathered and the
frequency-domain quantities are calculated once on the coordinator rank.

Finite-resistance sources on dispersive edges use the complete complex
background permittivity in the Yee-gap correction. The same correction is
used for terminal current and accepted power in antenna parameters and SAR
normalisation. Hard sources on dispersive
edges are not yet supported.

For a finite-resistance source, the voltage-source resistance is the
reference impedance :math:`Z_0`; a hard source defaults to 50 Ohms unless the
final optional reference-impedance value is supplied. At the source plane, the known generator
spectrum :math:`V_g` and sampled total gap voltage :math:`V` give

.. math::

    S_{11,\mathrm{source}} = \frac{2V-V_g}{V_g}.

No current calculation is required in this finite-resistance case. gprMax
removes the parallel capacitance and background conductance of the source Yee
edge before reporting the antenna-terminal result. With
:math:`c=Z_0Y_\mathrm{gap}`, this correction and the input impedance are

.. math::

    S_{11} =
    \frac{2S_{11,\mathrm{source}}+c(1+S_{11,\mathrm{source}})}
         {2-c(1+S_{11,\mathrm{source}})},
    \qquad
    Z_\mathrm{in}=Z_0\frac{1+S_{11}}{1-S_{11}}.

For a zero-resistance source, the gap voltage is prescribed at integer
electric-field times. gprMax calculates the Ampere-loop current from the four
surrounding magnetic components. The voltage and current samples retain their
exact Yee times,

.. math::

    V[n]=V^{n}, \qquad I_\mathrm{loop}[n]=I_\mathrm{loop}^{n-1/2},

and the engineering-convention transforms apply the corresponding
:math:`0` and :math:`-\Delta t/2` time offsets, retaining the initial voltage
sample. This corrects their
relative phase without attenuating current by interpolation. The terminal
current is then

.. math::

    I_\mathrm{terminal}=I_\mathrm{loop}-Y_\mathrm{gap}V.

For this integer-voltage/half-step-current pairing, the discrete parallel-gap
admittance is

.. math::

    Y_\mathrm{gap} = G_\mathrm{bg}\cos\left(\frac{\omega\Delta t}{2}\right)
    +j\frac{2C_\mathrm{gap}}{\Delta t}
    \sin\left(\frac{\omega\Delta t}{2}\right).

This is the FDTD analogue of an ideal delta-gap MoM excitation: voltage is
imposed and the antenna current is a solved response. The user-supplied
:math:`Z_0` (or its 50 Ohm default) defines the travelling-wave normalisation
only. The reported
quantities are calculated directly as

.. math::

    Z_\mathrm{in}=\frac{V}{I_\mathrm{terminal}},
    \qquad
    V^\pm=\frac{V\pm Z_0 I_\mathrm{terminal}}{2},
    \qquad
    S_{11}=\frac{V^-}{V^+}.

The gap capacitance and conductance use the effective electric-edge material
before any artificial source resistance is added. The appropriate discrete
admittance is used in each source mode so the correction is consistent with
the trapezoidal Yee update and the mode's voltage/current sampling times.

By default, output stops at the first native FFT bin that does not have at
least 10 cells per shortest wavelength in the model. For nonmagnetic,
lossless, nondispersive media this is the material with the largest :math:`\epsilon_r`.
A numeric ``spectrum_limit`` changes this sampling requirement; values below 10 produce
a warning and values below 3 are rejected. ``nyquist`` deliberately retains
the full spectrum but does not claim it is accurate: the lambda/10 limit and
per-bin mesh/source validity masks are still written to HDF5. The actual
stored range, native frequency resolution, Nyquist bound, and limiting
material are reported when the model is built.

.. note::

    * Multiple independently identified voltage ports are supported in 3-D
      CPU, CUDA, OpenCL and Metal models, domain-decomposed MPI CPU models,
      and CPU or CUDA HSG subgrids. The automatic port is not available in 2-D;
      excitation support and port support are distinct.
    * Sources inside PML are rejected. Finite-resistance ports support
      dispersive source edges; hard sources on those edges are rejected.
      For repeated finite-resistance port excitations use
      :class:`gprMax.PortStudy`, which validates the fixed source positions,
      resistances and per-case port histories.
      A hard source at a domain-minimum transverse boundary remains a valid
      excitation, but gprMax warns and omits its automatic port output because
      the complete current loop cannot be sampled there.
    * ``S11`` remains the primary result. ``Zin`` is singular near an open
      circuit (:math:`S_{11}=1`), so gprMax also stores ``Yin`` and separate
      validity masks.
    * A time trace that has not decayed before the end of the model window can
      contaminate the spectrum. gprMax reports a tail-level warning rather
      than hiding or clipping the result.
