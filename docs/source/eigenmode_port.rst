.. _eigenmode:
.. _eigenmode-port:
.. _fdfd-eigenmode-source:

****************************************
Eigenmode ports and S-parameter analysis
****************************************

An eigenmode port solves the transverse field pattern of a waveguide and
measures incoming and outgoing modal waves. An excitation chooses which
pattern to launch. A broadband pulse and one time record can provide results
at many frequencies, without a separate full simulation for every frequency.

This guide explains setup and interpretation for users familiar with
electromagnetic simulation. Python comes first, followed by compact hash
equivalents. The field equations, discrete operators, and numerical
algorithms are in :doc:`eigenmode_port_theory`.

.. contents:: On this page
   :local:
   :depth: 1

First two-port model
====================

The three main objects are ``gprMax.EigenmodeBand``,
``gprMax.EigenmodePort``, and ``gprMax.EigenmodeExcitation``. Add them to the
same ``gprMax.Scene`` as the geometry and materials. A band defines the
frequencies to measure, a port defines a reference plane and monitored modes,
and an excitation selects a channel to drive. A *channel* is a
``(port, mode)`` pair.

For a conventional independently driven measurement, one excitation produces
one S-matrix column. Several excitations produce a
coherent driven state and its active reflection coefficients. To recover a
complete S matrix, use ``EigenmodeStudy`` with independent excitation cases.
Excitation may be omitted only when every port has a passive
``VirtualWaveguide``; this writes raw modal spectra without S-parameters.

Creating and running a scene
----------------------------

This is a complete two-port 2D dielectric-waveguide model. Coordinates and
cell sizes are in metres, frequencies in Hz, and times in seconds.
``float("inf")`` denotes the invariant extent of a 2D model.

.. code-block:: python

   from pathlib import Path
   import gprMax

   inf = float("inf")
   scene = gprMax.Scene()
   scene.add(gprMax.DomainMode(mode="TM"))
   scene.add(gprMax.Domain(p1=(0.24, 0.08, inf)))
   scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
   scene.add(gprMax.TimeWindow(time=5e-9))
   scene.add(gprMax.PMLThickness(thickness=(5, 5, 0, 5, 5, 0)))
   scene.add(gprMax.Material(er=9, se=0, mr=1, sm=0, id="core"))
   scene.add(gprMax.Box(
       p1=(0, 0.03, 0), p2=(0.24, 0.05, inf), material_id="core",
   ))
   scene.add(gprMax.EigenmodeBand(
       id="band", fmin=4e9, fmax=6e9, points=21,
   ))
   scene.add(gprMax.EigenmodePort(
       port=1, p1=(0.02, 0.005, 0), p2=(0.02, 0.075, inf),
       direction="+", modes=(1, 2), anchors="auto",
   ))
   scene.add(gprMax.EigenmodePort(
       port=2, p1=(0.235, 0.005, 0), p2=(0.235, 0.075, inf),
       direction="-", modes=(1, 2), anchors="auto",
   ))
   scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto"))
   gprMax.run(scenes=[scene], outputfile=Path("straight_waveguide"),
              geometry_only=True)

First use ``geometry_only=True`` to build the material grid and solve the
modes without FDTD time stepping. Inspect the modal-field and waveform PNGs.
Set ``geometry_only=False`` to accumulate the spectra and write the HDF5 and
S-parameter CSV files. The example scripts below package these two operations
behind ``--geometry-only`` and the default full run, respectively.

Before interpreting S-parameters, check the termination: the port itself
does not absorb returning waves. This example extends the physical guide
through the domain PML, an absorbing layer at its boundary. An internal
feed can instead use a :ref:`virtual-waveguide`.

For a runnable script and result plots, use:

.. code-block:: console

   python examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py --geometry-only
   python examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py
   python examples/features/eigenmode_ports/example_1_straight_waveguide/plot_results.py

The full run writes ``straight_waveguide.h5`` and
``straight_waveguide_sparameters.csv`` beside the script. Expect mode-1
transmission near 0 dB, with small reflection and conversion into mode 2.
The plotter also shows the pulse travelling through the guide. See
:ref:`eigenmode-results` for mask-aware interpretation; a finite coefficient
alone is not evidence of a trustworthy power ratio.

Configuring ports
=================

EigenmodeBand arguments
-----------------------

.. code-block:: python

   gprMax.EigenmodeBand(
       id="band", fmin=4e9, fmax=6e9, points=21,
       frequencies=(4.25e9, 4.75e9),
       transition="auto", spectral_threshold=1e-3,
   )

.. include:: _includes/eigenmode_band_parameters.rstinc

For example, ``fmin=4e9``, ``fmax=6e9``, and ``points=3`` select
4, 5, and 6 GHz. Adding ``frequencies=(4.5e9, 5e9)`` gives the final list
4, 4.5, 5, and 6 GHz. The existing 5 GHz value is included only once, so the
final number of output frequencies can be larger than ``points``.

Every port measures at the same final list of output frequencies. Increasing ``points``
or adding ``frequencies`` increases output sampling; it does not request more
modal field solves. Those solves are controlled by ``EigenmodePort.anchors``.
When ``NTFFAntennaPorts`` uses modal power, every NTFF frequency must be in
this list. NTFF may use fewer of these frequencies, so a dense S-parameter sweep
can share power data with a sparse far-field calculation.

Hash command: #eigenmode_band
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   #eigenmode_band: band 4e9 6e9 21 4.25e9 4.75e9

The optional trailing values add output frequencies to the equally spaced
frequencies selected by ``fmin``, ``fmax``, and ``points``. Square brackets
denote optional arguments and are not typed into an input file.
``transition`` and ``spectral_threshold`` are Python-only controls; the hash
command uses their defaults.

EigenmodePort arguments
-----------------------

.. code-block:: python

   gprMax.EigenmodePort(
       port=1, p1=(0.02, 0.005, 0), p2=(0.02, 0.075, float("inf")),
       direction="+", modes=(1, 2), anchors="auto", plot_fields=None,
       tracking="legacy",
   )

.. include:: _includes/eigenmode_port_parameters.rstinc

Place the aperture in a longitudinally uniform section and include the whole
guided field, including evanescent tails around a dielectric core. The normal
comes from the matching coordinate: equal x coordinates give an x-normal
port, equal y coordinates a y-normal port. Mode numbers are solver ordering,
not guaranteed physical labels. Inspect the E/H profiles before identifying
a solution as TE10, quasi-TEM, or a particular guided slab mode.

Automatic mode tracking is deliberately opt-in while it gains validation across
more guide families. Set ``tracking="auto"`` to follow physical branches from
the anchor nearest the band centre, detect exact degenerate subspaces, and run
mode-quality diagnostics. ``verification="full"`` (the opt-in default) compares
the represented voxel geometry on a refined mesh and, for open apertures, on two
larger in-model windows. If surrounding non-PML geometry is unavailable, the
classification is ``unresolved`` and the usable primary mode is retained with a
warning. ``verification="fast"`` avoids those extra solves.

For an automatically detected two-mode degenerate space, gprMax first tries to
orient mode 1 and mode 2 along the port's two global transverse axes. If the
integrated electric-field moments cannot define those axes, it constructs a
repeatable basis from the complete E/H subspace. ``mode_polarizations`` remains
available when a different physical orientation is wanted; ``degenerate`` is
ignored in automatic mode because the solved spectrum defines the groups.

Confinement and numerical validity are independent. A numerically valid,
confidently tracked mode with forward real power remains an anchor when it is
unbound-suspect or unresolved; gprMax warns that injection and S-parameters may
be less accurate. Nonfinite fields, failed eigenpair residuals, rank loss, and
singular reconstruction remain unusable. This is useful for low-frequency CPW
modes whose evanescent tails have not decayed at the chosen aperture boundary.

In geometry-only runs, automatic tracking adds ``Re(n_eff)`` and
``-Im(n_eff)`` dispersion panels above each tracked modal-field figure. Red
crosses identify anchors with confinement warnings. Inspect these curves and
the E/H profiles before running the time-domain model.

.. warning::

   Automatic mode tracking and its confinement/artifact diagnostics are under
   development. Keep ``tracking="legacy"`` for established production models,
   and inspect the generated dispersion and modal-field plots before relying on
   an automatically tracked profile.

Two focused examples exercise the new path:

* :download:`automatic circular TE11 degeneracy <../../examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py>`
  omits both ``degenerate`` and ``mode_polarizations``, letting the tracker
  discover the pair, apply its default x/y directions, and measure S11/S21
  between two ports on a straight guide;
* :download:`automatic mode crossing <../../examples/features/eigenmode_ports/example_9_auto_mode_crossing/auto_mode_crossing.py>`
  uses one anisotropic guide whose polarized propagation constants cross, so
  raw eigenvalue order changes while the tracked polarization identities remain
  consistent between its two S-parameter ports.

Both scripts set ``plot_fields=True``. Running either one generates the combined
dispersion and field inspection plots; ``--geometry-only`` skips time stepping.

An excited port launches the selected modal field and measures returning
waves, but it does not absorb those waves. Continue the guide behind the
port through a domain PML, or attach a :ref:`virtual-waveguide` to provide a
matched termination. Point receivers are optional diagnostics; S-parameters
come from the modal port monitors.

Hash command: #eigenmode_port
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   #eigenmode_port: 1 0.02 0.005 0 0.02 0.075 inf + 1,2 auto
   #eigenmode_port: 1 0.02 0.005 0 0.02 0.075 inf + 1,2 auto tracking=auto verification=full
   #eigenmode_port: 2 0.235 0.005 0 0.235 0.075 inf - 1,2 4e9 5e9 6e9 y

Use ``inf`` for an invariant extent and comma-separated mode indices such as
``1,2``. Specify ``auto`` or space-separated modal anchor frequencies. The
optional final ``y`` or ``n`` forces or suppresses the port's field plots;
omitting it retains the geometry-only default.
Named hash options may follow the anchor/plot tail. Automatic tracking accepts
``tracking=auto`` and ``verification=full|fast``. Advanced names match
``EigenmodeTrackingConfig`` fields, for example
``residual_tolerance=1e-8``, ``edge_fraction_max=1e-3``, and
``max_solves=200``.

EigenmodeExcitation arguments
-----------------------------

.. code-block:: python

   gprMax.EigenmodeExcitation(
       port=1, mode=1, waveform="auto", amplitude=1.0,
       phase_deg=0.0, delay_s=0.0, plot_waveform=None,
   )

.. include:: _includes/eigenmode_excitation_parameters.rstinc

Automatic excitation requires ``fmax > fmin`` and a time window long enough
to contain the pulse. For a single-frequency band, add an explicit waveform,
for example ``gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9,
id="tone")``, and select ``waveform="tone"``. A custom broadband waveform's
exact sampled spectrum must fit the declared band's supported coverage.
Significant DC and Nyquist bins are warned about and discarded; more than
one percent of spectral power outside the allowed band is rejected. Use a
band-limited pulse rather than extending a measurement sweep over
frequencies the source cannot represent.
The automatic pulse has smooth spectral transitions outside the output band;
the anchor policy accounts for significant transition energy.

The diagnostic files are ``<output>_PortN_ModeM.png`` and
``<output>_EigenmodeExcitation.png``. Each modal anchor occupies one row with
tangential E and H vectors. With multiple drives, excitation filenames also
include ``_PortN_ModeM``. The waveform figure shows the sampled pulse, its
surrounding spectrum, the requested band, and the exact output DFT bins.

Hash command: #eigenmode_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   #eigenmode_excitation: 1 1 auto
   #eigenmode_excitation: 2 1 auto 0.5 90 0 y

The waveform, amplitude, phase, and delay default to ``auto``, 1, 0, and 0.
Supply preceding positional values when setting later values. The final
``y`` or ``n`` independently controls this drive's waveform/spectrum plot;
omitting it retains the geometry-only default. Repeated commands drive
distinct channels of one coherent state. The hash interface uses amplitude;
``power`` is a Python-only alternative.

Two-dimensional fields and mode names
-------------------------------------

In a 2D model, one spatial direction is invariant: geometry and fields are
assumed constant along it. Use ``float("inf")`` to span it in the domain,
geometry, and port aperture. ``DomainMode(mode="TM")`` retains electric
field along the invariant axis and magnetic field in the physical plane;
``"TE"`` retains the complementary components. This choice is defined
relative to the invariant axis, not the waveguide's propagation axis.
It is distinct from naming a 3D waveguide mode TE10 or TE11.

The modal solver uses a line cross-section in 2D and an area cross-section
in 3D. Its propagating profiles are normalized to one watt per metre in 2D
and one watt in 3D; the pulse amplitude scales those fields. A synthetic
invariant mesh spacing does not specify a physical guide width.
See :ref:`eigenmode-power` for the integrals and phase convention.

Placement and accuracy checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This feature is experimental. It requires an internal port, a locally
uniform, non-dispersive cross-section, and at least two cells across each
physical transverse direction. Reduced 2D TE/TM guides use the CPU solver.
SIBC walls, including surface dispersion and exact PMC, are supported with
the retained-host and extrusion restrictions described in :ref:`sibc-pml`.
Main-grid CPU, CUDA, OpenCL, Metal, and domain-decomposed MPI CPU paths are
supported; HSG subgrid virtual ports use the CPU or CUDA fine-grid update cycle.

For an antenna, enclose the entire physical antenna and feed aperture with
the NTFF box. Keep every face in the intended homogeneous background, clear
of the aperture and metal, and outside the domain PML. Both a closed
equivalent-current surface and KSIR can be used; KSIR requires all six
physical faces. For gain calculations, use a rectangular transform window
and an ``NTFFAntennaPorts`` association listing every physical port,
including passive ports. See the complete horn example for these settings.

Start with the default guide settings and inspect the geometry and modal
field plots using ``geometry_only=True``. Then repeat the full run with a
longer auxiliary guide, a thicker PML, and different source clearance,
changing one control at a time and respecting the length constraint. Also
refine the mesh. Check that S11 and any requested radiation patterns or gain
change by less than the accuracy your application needs. A finite PML is an
approximation to a matched termination, so the defaults are a starting point,
not a guarantee of negligible reflection for every guide and frequency band.

Choosing frequency anchors
--------------------------

An eigenmode is the electric and magnetic field pattern that a guide supports
at a particular frequency. An **anchor** is a frequency at which gprMax solves
that pattern directly. The field pattern, the relative sizes of its electric
and magnetic fields, and its propagation constant can change with frequency.
The propagation constant determines how quickly the wave's phase changes as
it travels along the guide.

With a single anchor, gprMax uses one solved pattern and effective index
throughout the pulse's frequency range. This is a useful approximation for a
narrow band or a mode that changes little over the band. For a wide band,
however, the pattern at the centre may describe the band edges poorly. The
source can then launch a less accurate wave, and the monitor can misidentify
how much of the measured field is incoming or outgoing.

Multiple anchors let gprMax follow these changes. It solves the guide's
cross-section at several frequencies, matches the same physical mode between
solutions, aligns their phases, and interpolates between the retained
solutions. This gives the source and monitors a reference that varies with
frequency. When the mode changes smoothly and is tracked successfully, this
can reduce reflection caused by an inaccurate injected mode and improve the
amplitude and phase of the extracted S-parameters across the band.

These are additional cross-section solves during setup; they do not require
one complete FDTD simulation per anchor. A broadband pulse still covers many
frequencies in one time-domain run. Anchors and output frequency bins have
different jobs: anchors control how accurately the mode is represented,
whereas output bins select where the measured spectrum is reported. Asking
for more output bins alone does not improve the modal reference. More anchors
increase setup cost and storage, so compare results as you refine their
spacing instead of assuming that a larger list is always better.

``anchors="auto"`` gives automatic ports a common candidate list covering the
output band and significant excitation spectrum. Candidates include band
edges, centre, spectral limits, and geometrically spaced intermediate values.
Extra candidates outside the output band cover parts of the pulse that can
still excite the model. Tracking and the decision to retain or discard an
anchor are resolved independently for each port and mode, or whole declared
degenerate group. Passive-only setups use the requested band alone.

Use explicit anchors where the mode changes rapidly, especially near cutoff,
as in example 6. Cutoff is the boundary between a travelling mode and a field
that decays along the guide. Closer anchors can resolve a smooth but rapid
change. A scalar or one-element tuple selects one fixed modal profile; use it
for a narrow band or when the mode cannot be matched uniquely across a wider
band.

Multiple anchors can fail for several different reasons:

* **The anchors are too far apart.** Their patterns may differ too much to
  identify them confidently as the same mode. Inspect the fields and try
  closer spacing in the region of rapid change.
* **Two modes become indistinguishable or exchange order.** The mode number
  alone does not guarantee the same physical pattern at every frequency.
  Interpolating unrelated patterns would create an incorrect reference.
  Declare the complete degenerate group using ``degenerate``. For circular
  TE11, also use ``mode_polarizations`` to obtain physical channel labels.
  A resolved split requires independent modes rather than degenerate mixing.
* **The mode reaches cutoff or a non-propagating gap.** A decaying, or
  *evanescent*, mode cannot supply the same one-watt travelling-wave source
  as a propagating mode. The interpolation cannot bridge a gap where that
  travelling mode ceases to exist. At exact cutoff, separating forward and
  backward waves can also become numerically ambiguous. Split disconnected
  propagating ranges into separate bands and inspect the validity masks near
  cutoff.

The tracking check measures the similarity, or **overlap**, of neighbouring
patterns. An overlap below 0.9 warns; below 0.6 the match is treated as
ambiguous. With automatic anchors, gprMax may discard a failing candidate
outside the output band and use the nearest retained endpoint there. An
in-band failure for an independently tracked mode may instead select a
single band-centre anchor, provided that it carries forward real power. This allows the run to proceed with a fixed
reference, whose accuracy can decrease away from that frequency. Multiple
explicit anchors remain strict: a tracking failure is an error that requires
revising the anchor choice.

Two anchor banks serve different purposes. ``anchor_mode_valid`` selects
propagating profiles for source injection and power normalization.
``anchor_mode_reference_valid`` also admits tracked evanescent patterns, so a
monitor can describe a decaying field even when it cannot treat that field
as a power-carrying wave. Interpolation stays within a contiguous branch and
never mixes propagating and evanescent references across cutoff. Every
requested mode, including passive monitored modes, must retain at least one
forward-real-power anchor. Inspect the profiles, ``RequestedAnchorPolicy``,
``ResolvedAnchorPolicy``, ``CandidateAnchorFrequencies``, and the retained
anchor masks in HDF5 to see which references were actually used.

Declared degenerate groups are stricter: legitimate guard trimming or
cutoff exclusion applies to the whole group, and a failed group does not
fall back one member at a time. Resolved splitting, rank loss, or failed
in-band subspace tracking is an error. See :ref:`eigenmode-degenerate-theory`
for the tests used to make that decision.

Running hash input files
------------------------

Run a hash model with ``python -m gprMax model.in --geometry-only`` to inspect
it, then ``python -m gprMax model.in -outputfile results/model`` to simulate.
The tutorial models below are Python programs and run directly with Python.

Terminating the feed
====================

.. _virtual-waveguide:

VirtualWaveguide: a matched termination
---------------------------------------

In CST or `HFSS <https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v251/en/Subsystems/HFSS/Content/HFSS/WavePortsTheory.htm>`_,
you often use a waveguide port that does two jobs: it injects a chosen
waveguide mode and absorbs waves returning to the port from the device. A
*mode* is the electric and magnetic field pattern that travels along the
guide. A *matched termination* accepts a returning wave with as little
additional reflection as possible, as though the guide continued indefinitely.

In gprMax these jobs are separate. ``EigenmodePort`` together with
``EigenmodeExcitation`` supplies the modal injection and measurement, but
**the eigenmode port itself does not absorb the returning wave**. The wave
can pass back through the source plane and continue along the feed. If it
then reaches a reflecting end, it can bounce back into the device and alter
the result.

The usual finite-difference time-domain (FDTD) solution is to extend the real
waveguide behind the port into the **domain PML**. FDTD advances electric and
magnetic fields on a grid of cells; the *domain* is the physical region
represented by that grid. A PML (*perfectly matched layer*) is an absorbing
layer at its boundary. Keeping the feed's cross-section unchanged as it
enters this layer lets returning waves leave the model with little reflection.

This arrangement can be inconvenient for an antenna. A **near-to-far-field
(NTFF) box** is an imaginary measurement surface around the antenna: fields
recorded on its faces are used to calculate radiation far away. For a closed
box in homogeneous air, we want to enclose the entire antenna and its feed
aperture without a metal or dielectric feed crossing a face. A real feed
running all the way to the domain PML can prevent that arrangement.

``VirtualWaveguide`` provides a matched termination at an internal port, so
the physical feed no longer needs to extend to the domain boundary. Despite
its name, **its practical purpose is to terminate the port**. The extra guide
is simulated in a separate, auxiliary grid, leaving room in the main domain
for the rear face of a closed NTFF box. The main domain still needs its own
PML to absorb radiation leaving the antenna.

How the virtual waveguide works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. gprMax copies the material cross-section at the port into a separate,
   straight numerical waveguide. This auxiliary grid uses the same cell
   sizes and time step as the grid containing the port, and has its own PML
   at the far end.
2. The port aperture connects the two grids. Electric and magnetic fields
   are exchanged at every time step, allowing waves to travel through the
   connection in both directions. A wave returning from the device enters
   the auxiliary guide and travels to its PML, where it is absorbed.
3. If the port has an ``EigenmodeExcitation``, gprMax places that source
   inside the auxiliary guide. The launched wave travels through the
   aperture into the physical feed. If there is no excitation on that port,
   it simply acts as a passive matched termination.

The following shows the wave paths, not the physical layout of the grids:

.. code-block:: text

   Separate auxiliary grid                  Main simulation grid
   [PML] --- [modal source] --- connection --- [port] --- [antenna]
                 launched wave -------------------------->
     <--------------------------- wave returning from antenna

The auxiliary guide does not occupy space behind the port in the main
geometry, so it does not cross the NTFF box. Its fields are still calculated
during the run, so it adds some memory and time-stepping cost. The modal
monitor remains at the physical port plane, where it measures incident and
outgoing waves for S-parameters.

A matched termination reduces reflections from the *feed termination*; it
does not remove the antenna's own mismatch or force its S11 to zero. The
antenna's reflected wave is measured at the port before travelling into the
auxiliary absorber. The approach follows Wang and Langdon [WAN2010]_; see
:ref:`eigenmode-virtual-coupling` for the field-update equations.

How to use it
^^^^^^^^^^^^^

Place an ``EigenmodePort`` at the end of a straight, uniform section of the
physical feed, inside the simulation domain. Point ``direction`` toward the
device: for an x-normal port, ``"+"`` launches toward increasing x and the
virtual continuation represents the feed behind it, toward decreasing x.
The aperture must cover the guided field and have the same material
cross-section immediately on either side of the plane. Keep bends, tapers,
and other changes away from this connection.

Add one ``VirtualWaveguide`` referring to that port number. Keep the
``EigenmodeExcitation`` if the port should transmit; omit it if the port
should only absorb. You do not draw the auxiliary guide or add its PML to
the physical geometry yourself.

For passive surface-impedance walls, CPU main-grid 3D and 2D TE/TM models
can use the same continuation, including fitted metal and infinite-resistance
PMC. Follow :ref:`sibc-pml` for uniform extrusion, retained-host restrictions,
and opaque padding. In 2D, padding is needed only along the physical
transverse axis. The general virtual-guide backend support does not extend
SIBC to accelerators, MPI, or subgrids.

For example, this is the feed configuration from `Example 3: a pyramidal
horn antenna`_. It assumes that ``scene`` already contains the 3D domain,
mesh, materials, and horn geometry from that example; this snippet alone is
not a complete model. Coordinates are in metres and frequencies are in Hz.

.. code-block:: python

   scene.add(gprMax.EigenmodeBand(
       id="eigenmode_band", fmin=8e9, fmax=12e9, points=101,
   ))
   scene.add(gprMax.EigenmodePort(
       port=1, p1=(0.012, 0.033, 0.029), p2=(0.012, 0.057, 0.041),
       direction="+", modes=(1,), anchors="auto",
   ))
   scene.add(gprMax.VirtualWaveguide(
       port=1, length_cells=30, pml_cells=12,
       source_clearance_cells=6,
   ))
   scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto"))

The physical feed starts at x = 12 mm and points toward the horn. The
example's closed NTFF box has its rear face at x = 10 mm, in air behind the
feed. The virtual continuation uses separate grid storage, so its 30-cell
length does not need to fit into that 2 mm gap. Use the complete example for
the geometry, NTFF setup, and commands to run and plot the results.

For a passive receiving port in a multiport model, add its own
``EigenmodePort`` and ``VirtualWaveguide`` but no ``EigenmodeExcitation`` for
that port. A model with no eigenmode excitations is allowed only when every
eigenmode port has a virtual guide; it records raw modal spectra without
normalized S-parameters.

VirtualWaveguide arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   gprMax.VirtualWaveguide(
       port=1, length_cells=30, pml_cells=12,
       source_clearance_cells=6, pml_profile=None,
   )

.. include:: _includes/virtual_waveguide_parameters.rstinc

Only ``port`` is required: ``gprMax.VirtualWaveguide(port=1)`` uses the
defaults above. Cell counts refer to the grid containing the port, along
the guide axis. For a 1 mm cell size, 30 cells mean a 30 mm auxiliary guide.

The remaining
``length_cells - pml_cells - source_clearance_cells`` cells separate the
source from the port connection: 12 cells with the defaults. Increasing the
PML thickness or source clearance at fixed total length reduces this
separation; increase the total length as needed. These constraints also
apply to passive guides.

Hash command: #virtual_waveguide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   #virtual_waveguide: 1 30 12 6

The required port number identifies an existing ``#eigenmode_port``. Cell
counts default to 30, 12, and 6, with the same constraints as the Python API.
The optional final token names a reusable PML profile. An unexcited virtual
port provides a passive matched termination.

Selecting modal channels
========================

Inspect the solved E/H profile before assigning a physical mode name.
Mode numbers are solver ordering, not guaranteed TE/TM labels. A near tie
can rotate the raw solver basis even in an unchanged symmetric guide.

.. _eigenmode-degeneracy-explained:

What degeneracy means for a port
--------------------------------

Two modes are **degenerate** when they have the same propagation constant
at a given frequency but independent field patterns. They accumulate the
same propagation phase along a uniform guide. This does not mean that they
are duplicate results or that only one should be kept: they represent two
independent channels that the guide can carry.

The circular TE11 pair is a familiar example. An ideal circular guide has
no preferred transverse direction. One TE11 pattern can be oriented with
its overall electric polarization along y and another along x. Rotated
combinations, such as two diagonal polarizations, describe the same pair
equally well. The chosen pair of patterns is called a **basis**. Changing
the basis changes how we label and combine the channels, not the guide.

Why mode numbers alone can be misleading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An eigensolver finds valid field patterns, but symmetry does not tell it
which one to call vertical. At one frequency it might return vertical and
horizontal patterns as modes 1 and 2; at another it might return the same
patterns in reverse order, or two diagonal combinations. Small numerical
differences can change that choice even when the physical guide is unchanged.
This is a change of basis, not evidence that the guide has converted the
wave's polarization.

Broadband ports solve several frequency anchors and interpolate their
fields. If mode 1 means vertical at one anchor and diagonal at the next,
interpolating those raw patterns would change the intended polarization
across the pulse spectrum. Tracking the complete degenerate pair lets
gprMax compare the same two-channel space before choosing its labels.

Which option should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Use ``degenerate=(1, 2)`` when you want the pair tracked together but do
  not need mode numbers to mean particular physical directions. Its initial
  basis is arbitrary; subsequent anchors follow it smoothly.
* Also use ``mode_polarizations={1: "y", 2: "x"}`` when you want mode 1
  to mean vertical and mode 2 horizontal for a z-directed guide. gprMax
  enforces those physical directions independently at each retained anchor.
  Use the same assignments at the receiving port to measure the same labels.

Declaring the pair neither excites both channels nor forces their physical
propagation constants to become equal. ``EigenmodeExcitation`` still selects
the channel to launch. A declared pair contributes two channels to a complete
S-matrix study, so each needs its own independent excitation case.

Near equality alone is not enough to justify mixing. For example, an
elliptical guide can split the circular pair into modes with different
propagation constants. Those modes accumulate different phases, and a
combination can change polarization as it travels. That is physical beating,
which should be represented using independent modes. gprMax accepts exact
or numerically unresolved degeneracy and rejects resolved splitting; the
thresholds and alignment method are in :ref:`eigenmode-degenerate-theory`.

Physically labelled degenerate modes
------------------------------------

For circular TE11 propagating along z, assign the two channels once:

.. code-block:: python

   scene.add(gprMax.EigenmodePort(
       port=1, p1=(0, 0, 0.02), p2=(0.05, 0.05, 0.02), direction="+",
       modes=(1, 2), anchors="auto", degenerate=(1, 2),
       mode_polarizations={1: "y", 2: "x"},
   ))
   scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto"))

Here mode 1 is vertically polarized (global y); changing only ``mode=1`` to
``mode=2`` launches horizontal polarization (global x). Reversing propagation
does not change these electric-field labels. Use the same assignments on the
receiving port. For diagonal directions, use
``mode_polarizations={1: (1, 1, 0), 2: (-1, 1, 0)}``.

Equivalent hash options follow the existing anchor and plotting arguments:

.. code-block:: none

   #eigenmode_port: 1 0 0 0.02 0.05 0.05 0.02 + 1,2 auto degenerate=1,2 mode_polarizations=1:y;2:x
   #eigenmode_port: 1 0 0 0.02 0.05 0.05 0.02 + 1,2 auto n degenerate=1,2 mode_polarizations=1:1,1,0;2:-1,1,0

Separate groups with semicolons (``degenerate=1,2;3,4``), and polarization
entries with semicolons. Components inside a vector use commas. Values are
parsed literally, with no expression evaluation. Partially specified pairs,
dependent directions, and directions normal to the port are errors.

Polarization describes the direction of the transverse electric field
integrated across the aperture; local arrows need not all be parallel. gprMax
aligns that direction at every retained frequency anchor. Geometry-only plots,
single-anchor and broadband sources, monitors, studies, and virtual guides
all use the same aligned fields. Set ``plot_fields=True`` to also write the
standard modal pictures during a full run. Example 7 shows the two tracked
TE11 profiles.

With only ``degenerate``, the group is tracked smoothly, but its orientations
at the reference anchor remain arbitrary. With neither argument, modes keep
the existing independent tracking. Physical axis/vector selection requires
a two-mode group in a 3D cross-section, with both directions supplied.

Declare a group only for exact or numerically unresolved degeneracy. If
asymmetry resolves the splitting, excite the independently solved modes or
correct unintended geometry asymmetry. A group with negligible integrated
electric field cannot be oriented by axis/vector references; use generic
group tracking. These failures and their numerical thresholds are derived
in :ref:`eigenmode-degenerate-theory`.

Coherent and quadrature excitation
----------------------------------

Add distinct ``EigenmodeExcitation`` channels with the same base waveform
and set their ``amplitude`` and ``phase_deg``. For a circular guide with the
assignments above, the following replaces the single excitation:

.. code-block:: python

   scene.add(gprMax.EigenmodeExcitation(
       port=1, mode=1, waveform="auto", amplitude=1, phase_deg=0,
   ))
   scene.add(gprMax.EigenmodeExcitation(
       port=1, mode=2, waveform="auto", amplitude=1, phase_deg=90,
   ))

The 90-degree relative phase produces quadrature. This is one combined
driven state, so read its active reflection outputs; it does not measure
two independent S-matrix columns. Requested polarization vectors need not
be orthogonal: use the full power matrix for their combined power rather
than summing individual squared amplitudes. See :ref:`eigenmode-results`.

Measuring a network
===================

.. _eigenmode-results:

Reading coefficients and validity
---------------------------------

A port monitor compares the simulated electric and magnetic fields with the
reference patterns for its requested modes. It reports an ``incident`` and an
``outgoing`` **coefficient** for each mode and frequency: complex numbers that
describe how much of each wave is present, including its amplitude and phase.
An S-parameter divides an outgoing coefficient by the incident coefficient of
the driven channel. For example, S11 describes reflection at the driven port,
and S21 describes the response at port 2 to a drive at port 1.

A number in an output array is not enough to tell whether it is usable. The
reference pattern may be missing or unsuitable; two waves may be too similar
to separate reliably; or the source may provide almost no signal at that
frequency. There is also a physical distinction: a decaying field can have a
meaningful coefficient without carrying forward real power on its own.
Dividing by a nearly zero incident signal can produce a large, misleading
S-parameter even when the individual coefficients are well defined.

The **validity masks** record these different checks for each mode and
frequency. A mask is an array of true/false values, stored as 1/0 in HDF5.
True means that the value passes the named check; false means that it should
not be used for that purpose. A false mask does **not** mean zero reflection,
zero transmission, or an absent field. Keep the mask alongside the data when
plotting or calculating results, including when a stored coefficient is
finite or zero.

The first two masks check the reference patterns before measuring wave
amounts. The next two check the measured coefficients. The final two check
the S-parameter ratio after division by the incident signal:

.. list-table::
   :class: api-parameters
   :header-rows: 1
   :widths: 30 18 52

   * - Port HDF5 dataset
     - Shape
     - Meaning in plain language
   * - ``reference_basis_valid``
     - ``(M, F)``
     - We have a tracked reference pattern to compare with the measured fields. This alone does not guarantee that incoming and outgoing amounts can be separated.
   * - ``power_basis_valid``
     - ``(M, F)``
     - The reference pattern supports forward real power and can be scaled to a known power. A purely decaying reference does not pass this check.
   * - ``coefficient_valid``
     - ``(M, F)``
     - The incoming and outgoing amounts can be separated without a numerically ambiguous fit. A decaying field can still pass this check.
   * - ``power_wave_valid``
     - ``(M, F)``
     - The coefficients are usable and also pass the physical power checks, including a usable modal power matrix. They can be used in power calculations with the normalization described below.
   * - ``coefficient_valid_S``
     - ``(M, F)``
     - The coefficients needed for this S ratio are usable, and the incident signal used as its denominator is strong enough to divide by.
   * - ``power_wave_valid_S``
     - ``(M, F)``
     - The S ratio also passes the physical power checks for both the driven input and the measured output. Use this mask when interpreting reflection or transmission as power.

For example, below cutoff ``coefficient_valid`` may be true while
``power_wave_valid`` is false: the monitor can describe the decaying field,
but its coefficient squared is not transported power. Near exact cutoff,
``reference_basis_valid`` may be true while ``coefficient_valid`` is false:
a pattern exists, but the forward and backward contributions are too similar
to separate reliably. At a weakly excited frequency, the individual
coefficients may pass their checks while ``coefficient_valid_S`` is false
because their ratio would divide by too little incident signal. The incident
floor is -60 dB relative to the peak within each reference-normalization
class; this is a signal-strength check, not an accuracy guarantee.

Here ``F`` is the number of output frequencies and ``M`` is the number of
monitored modes at that port. All six masks share the ``(M, F)`` shape of
``incident``, ``outgoing``, and ``S``. The mode axis follows the port's
``mode_indices`` order, and the frequency axis follows ``frequency``. Raw
spectra exist for passive-only runs, while ``S`` and its masks exist only for
a single driven channel. Multiple drives instead write ``active_S``,
``active_S_driven``, ``coefficient_valid_active_S``, and
``power_wave_valid_active_S``. Undriven entries have no active-S ratio.

Use the coefficient mask to inspect modal responses, including decaying
fields. Use the power-wave mask to select the subset that also supports a
physical power interpretation:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File("straight_waveguide.h5", "r") as output:
       port = output["eigenmode_ports/port2"]
       frequency = port["frequency"][...]
       s21 = port["S"][0]  # first entry in this port's monitored mode list
       coefficient_mask = port["coefficient_valid_S"][0].astype(bool)
       power_mask = port["power_wave_valid_S"][0].astype(bool)
       coefficient_frequency = frequency[coefficient_mask]
       coefficient_db = 20 * np.log10(np.abs(s21[coefficient_mask]))
       power_wave_frequency = frequency[power_mask]
       power_wave_db = 20 * np.log10(np.abs(s21[power_mask]))

The CSV contains one row per frequency and destination channel, with complex
S, magnitude, dB magnitude, phase, ``coefficient_magnitude_squared``,
``coefficient_valid``, and ``power_wave_valid``. CSV masks refer to the ratio
in that row. The active-S CSV uses these same two mask column names.

For an orthogonal, power-normalized set of propagating modes, a coefficient's
magnitude squared gives its power, and an S-parameter's magnitude squared
gives the corresponding power ratio. With non-orthogonal modes, the modes
also contribute power through their interaction: use ``power_matrix`` and
keep these cross terms instead of adding individual squared magnitudes.
``electric_cross_power_matrix`` additionally describes the total-field power
in lossy ports. Both matrices have shape ``(F, M, M)``;
``power_matrix_valid`` and ``condition_number`` have shape ``(F,)``.
``condition_number`` describes how sensitive the coefficient fit is to small
numerical changes; a large value indicates a more difficult separation.
Invalid ratios remain in the frequency array as NaNs.

Passing these masks is necessary for the stated interpretation, but it does
not prove that the simulation has converged. Check the mesh, run duration,
port placement, and anchor spacing as well. A usable coefficient can still
be inaccurate if the physical setup or its numerical resolution is poor.

Complete matrices with EigenmodeStudy
-------------------------------------

An S-parameter matrix describes how waves entering a device produce waves
leaving it. A **channel** is one ``(port, mode)`` pair: the port identifies a
cross-section, and the mode identifies a particular field pattern at that
cross-section. A port with two monitored modes contributes two channels.
The complete matrix includes reflection, transmission, and conversion between
all declared channels.

Why one excitation gives one column
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a device with a left port numbered 1 and a right port numbered 2,
with one propagating mode at each. At a particular frequency, let
:math:`a_1,a_2` be the complex amplitudes (magnitude and phase) of waves
travelling toward the device and :math:`b_1,b_2` those travelling away from it.
The matrix relation is

.. math::

   \begin{bmatrix}b_1\\b_2\end{bmatrix}
   =
   \begin{bmatrix}S_{11}&S_{12}\\S_{21}&S_{22}\end{bmatrix}
   \begin{bmatrix}a_1\\a_2\end{bmatrix}.

The first index of :math:`S_{ij}` names the **output** channel and the second
names the **input** channel. Driving port 1 while no wave enters from port 2
sets :math:`a_2=0`, giving :math:`b_1=S_{11}a_1` and
:math:`b_2=S_{21}a_1`. Measuring both ports therefore determines the first
column. It does not reveal how the device responds to a wave entering from
port 2.

.. list-table:: Two independent excitations for a two-channel matrix
   :class: api-parameters
   :header-rows: 1
   :widths: 25 40 35

   * - FDTD case
     - Waves observed at both ports
     - Matrix entries determined
   * - Drive port 1, mode 1
     - Reflection back to port 1; transmission to port 2
     - First column: S11 and S21
   * - Drive port 2, mode 1
     - Transmission to port 1; reflection back to port 2
     - Second column: S12 and S22

In each case the finite-difference time-domain (FDTD) solver advances the
electric and magnetic fields through time. A broadband pulse excites many
frequencies, and the port's discrete Fourier transform (DFT) extracts the
response at every requested frequency from that time record. This gives a
column at many frequencies; it still probes only one input channel. Adding
DFT bins refines the frequency sampling. Adding modal anchors refines the
frequency-dependent field profiles. Neither supplies the missing excitation
from the other channel.

Driving both ports simultaneously with one fixed combination of amplitudes
and phases gives a combined response, such as
:math:`b_1=S_{11}a_1+S_{12}a_2`. That single measurement cannot separate the
two contributions. It is useful for active reflection or array radiation,
while independent excitation cases provide the information needed for a full
matrix. Reciprocity can relate S12 and S21 under suitable assumptions, but it
does not generally determine S22 from S11; ``EigenmodeStudy`` measures every
input channel instead of assuming symmetry.

What the study runs and reuses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``EigenmodeStudy`` manages these independent FDTD cases and assembles their
results. Keep the device geometry, materials, ports, and reference planes
fixed, and schedule each declared ``(port, mode)`` channel once. The study:

1. Builds the geometry, solves the modal anchor fields for the ports, and
   runs the first excitation through time.
2. Moves the scene's one excitation object to the next scheduled port and
   mode, preparing its waveform injection from the cached modal fields.
3. Clears the electric and magnetic fields, PML absorbing-boundary histories,
   and port DFT accumulators before advancing the next case through time.
   Responses from different cases therefore do not overlap.
4. Collects incoming and outgoing waves at every monitored channel and
   assembles an S matrix at each requested frequency.

Geometry and modal solves are reused, but each excitation still requires its
own time-domain simulation. Reusing the modal basis avoids repeating the
cross-section eigenmode calculations for an unchanged device.

In the ideal two-port example above, only the driven channel has an incoming
wave, so dividing the measured outgoing waves by that incoming amplitude
directly gives one column. Real simulations can also measure incoming waves
at nominally passive channels, for example from imperfect absorbing
boundaries. The study retains these measurements. It puts the incoming
vectors from all cases into the columns of :math:`A(f)` and the outgoing
vectors into :math:`B(f)`, then solves :math:`B(f)=S(f)A(f)` at each frequency.
This separates the responses using the measured excitations. Frequencies
with missing or insufficiently independent measurements are marked invalid;
the study also retains the measured matrices and conditioning diagnostics.
This correction does not replace checking the mesh, absorbing boundaries,
and simulation duration for convergence.

Python example: excite each port in turn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After constructing a ``scene`` with geometry, materials, an ``EigenmodeBand``,
and ports 1 and 2 each declaring ``modes=(1,)``, add the following. The scene
must not already contain another ``EigenmodeExcitation``. Example 4 below
provides the complete runnable geometry and script.

.. code-block:: python

   import gprMax

   excitation = gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto")
   scene.add(excitation)

   cases = [
       gprMax.StudyCase(
           "drive_port_1",
           [gprMax.ObjectState(excitation, port=1, mode=1)],
       ),
       gprMax.StudyCase(
           "drive_port_2",
           [gprMax.ObjectState(excitation, port=2, mode=1)],
       ),
   ]
   study = gprMax.EigenmodeStudy(cases)
   gprMax.run(scenes=[scene], study=study, outputfile="two_port")

``StudyCase`` names one run. Its ``ObjectState`` selects the ``port`` and
``mode`` of the same ``excitation`` object for that run; it does not add a
second simultaneous source. Here one call to ``gprMax.run`` performs two
FDTD cases and writes the aggregate ``two_port_study.h5``.

More modes mean more channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If both ports instead declare ``modes=(1, 2)``, the four channels are
``(1, 1)``, ``(1, 2)``, ``(2, 1)``, and ``(2, 2)``. Replace the two-case
schedule above with one case for each pair:

.. code-block:: python

   channels = [(1, 1), (1, 2), (2, 1), (2, 2)]
   cases = [
       gprMax.StudyCase(
           f"drive_port_{port}_mode_{mode}",
           [gprMax.ObjectState(excitation, port=port, mode=mode)],
       )
       for port, mode in channels
   ]
   study = gprMax.EigenmodeStudy(cases)

Pass this study to ``gprMax.run`` as above. Four FDTD cases produce a
4 by 4 matrix, with 16 entries at each frequency. For example, the case
driving port 1's mode 1 measures reflection into both modes of port 1 and
transmission into both modes of port 2. Driving port 1's mode 2 is a separate
case because its field pattern is a different input. With ``C`` declared
channels and ``F`` DFT frequencies, the study schedules ``C`` cases and
returns an array of shape ``(F, C, C)``. Include every declared channel exactly
once; the study checks the schedule before running it. At a given frequency,
use the validity masks below to distinguish usable coefficients from entries
that support a propagating-power interpretation.

Reading the assembled matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``study.result`` is an ``EigenmodeStudyResult``. Its ``s``,
``coefficient_valid_s``, and ``power_wave_valid_s`` arrays use
``(frequency, output_channel, input_channel)`` order. ``channel_ports`` and
``channel_modes`` identify the channels. The aggregate
``<output>_study.h5`` stores ``S``, ``coefficient_valid_S``,
``power_wave_valid_S``, measured ``incident_matrix`` and ``outgoing_matrix``,
``coefficient_valid_wave_matrix``, ``power_wave_valid_matrix``, and the
de-embedding diagnostics. Load it later with
``gprMax.EigenmodeStudyResult.from_hdf5(path)``. The Python wave-mask attributes
have the same names as their datasets.

Use the channel metadata to look up the output row and input column. After
the two-port run, for example:

.. code-block:: python

   result = study.result
   channels = list(zip(result.channel_ports, result.channel_modes))
   left = channels.index((1, 1))
   right = channels.index((2, 1))

   s11 = result.s[:, left, left]    # reflection for excitation from the left
   s21 = result.s[:, right, left]   # transmission from left to right
   s12 = result.s[:, left, right]   # transmission from right to left
   s22 = result.s[:, right, right]  # reflection for excitation from the right
   valid_s21 = result.power_wave_valid_s[:, right, left]
   frequency = result.frequency[valid_s21]
   usable_s21 = s21[valid_s21]

The same lookup works with multiple modes: a channel is always a port/mode
pair, and the array indices are zero-based even though port and mode numbers
are one-based. ``s`` contains complex amplitude ratios, including phase;
use the coefficient mask instead when inspecting generalized below-cutoff
responses rather than propagating power waves.

Each case stores ``study/eigenmode_response/S_column`` with
``coefficient_valid_S_column`` and ``power_wave_valid_S_column``. Its measured
``incident`` and ``outgoing`` arrays use ``coefficient_valid_wave`` and
``power_wave_valid``. These case arrays have shape ``(F, C)`` for ``C`` channels.
Restart with ``gprMax.run(..., study=study, i=N)``; a compatible aggregate
retains already completed cases.

Hash command: #study
^^^^^^^^^^^^^^^^^^^^

Supply the independent excitation cases in a CSV file:

.. code-block:: none

   #study: eigenmode cases.csv

For a two-port model with one monitored mode per port, ``cases.csv`` contains:

.. code-block:: text

   case_id,object_id,port,mode
   drive_port_1,eigenmode_excitation_1,1,1
   drive_port_2,eigenmode_excitation_1,2,1

``eigenmode_excitation_1`` identifies the scene's single excitation object.
Each row selects one input channel. See :doc:`input_hash_cmds` for the full
study-command syntax.


The modal projection, full power matrices, and measured-excitation matrix
solve are derived in :ref:`eigenmode-measurement-theory`.

Applications and troubleshooting
================================

Tutorial examples
-----------------

Run these commands from the repository root with gprMax installed in the
active environment. Every model defines ``build_scene()`` without running on
import. Its ``main()`` writes beside the script by default, so the adjacent
no-argument ``plot_results.py`` can find the output. ``--output PATH`` changes
the output stem; update plotting paths if using it. ``--gpu N`` selects a CUDA
device; omit it for CPU. Other backends can be selected through ``gprMax.run``.
The FDFD setup is performed on the host before device time stepping.

Example 1: a straight waveguide
-------------------------------

Start with the uniform 2D TM dielectric slab. Both ports monitor its two
guided modes, while port 1 excites mode 1. The 25 mm free-space margins around
the 20 mm core include its evanescent tails. The guide continues through the
x-directed domain PML. Receivers and snapshots illustrate propagation but
are not used for the modal S-parameter calculation.

``Snapshot(..., fileext=".h5")`` requests the HDF5 field arrays consumed by
the plotter. Set the extension explicitly in the Python API; the snapshot
default is VTK-HDF.

:download:`Complete Python model <../../examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py>`

Use the three commands in the first two-port workflow above.

Inspect confinement and symmetry in the modal figures, then check that the
waveform fits the time window. ``straight_waveguide_sparameters.png`` should
show mode-1 S21 near 0 dB with small reflection and mode-2 conversion.
``straight_waveguide_field_propagation.png`` follows twelve ``Ez`` snapshots
through right-PML absorption. Residual ripple depends on grid dispersion,
finite recording time, discretization, and boundary reflections.

Try ``mode=2`` in the excitation, or refine the mesh and compare S11/S21.
Temporarily requesting ``modes=(1, 2, 3, 4)`` exposes artificial aperture box
modes in this geometry. Inspect their boundary interaction and sensitivity to
aperture size before including additional modes in an analysis.

Example 2: a curved waveguide
-----------------------------

Two cylindrical sectors form a tight 90-degree bend. This example measures
4--8 GHz with 81 DFT bins. Port 2 has equal y coordinates and points in the
negative y direction, into the bend; it monitors the same two guided modes.

:download:`Complete Python model <../../examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.py>`

.. code-block:: console

   python examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.py --geometry-only
   python examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.py
   python examples/features/eigenmode_ports/example_2_curved_waveguide/plot_results.py

``curved_waveguide_sparameters.png`` shows reflection and conversion into the
second output mode; ``curved_waveguide_field_propagation.png`` follows the
pulse around the bend. Compare with the straight guide over their shared
4--6 GHz range. Increase the bend radius and inspect how reflection and modal
conversion change. If modal profiles exchange character with frequency,
investigate tracking rather than assuming the integer mode label is stable.

Example 3: a pyramidal horn antenna
-----------------------------------

The hollow rectangular feed expands through nine staircased PEC sections.
The fundamental TE10-like mode is launched over 8--12 GHz, and a virtual guide
provides the matched continuation behind the internal port.

:download:`Complete Python model <../../examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.py>`

The 101-point band has 40 MHz spacing. ``frequencies`` adds four off-grid
half-GHz bins after deduplication, giving 105 modal frequencies. The NTFF
transform uses just nine of those bins. ``anchors="auto"`` separately tracks
the TE10-like field profile and significant pulse spectrum. A below-cutoff
guard anchor may warn and remain only in the monitor reference bank; it is
excluded from one-watt source synthesis.

.. code-block:: console

   python examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.py --geometry-only
   python examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.py
   python examples/features/eigenmode_ports/example_3_antenna_and_farfield/plot_results.py

Inspect the feed polarization and geometry before the 3D run. The plotter
writes ``horn_sparameters.png``, ``horn_farfield_3d.png``, and
``horn_principal_planes.png``. The main beam should point along +x. The xz
cut is the E-plane and xy is the H-plane for the launched polarization.
Directivity describes the radiation pattern; gain includes radiation
efficiency, and realized gain also includes feed mismatch. The closed surface
is possible because the region behind the physical feed is homogeneous air.
Refine the mesh and vary the NTFF surface, feed length, auxiliary PML, and
time window before using the values quantitatively.

Example 4: the complete dominant-mode S matrix
----------------------------------------------

A microstrip with a central 2 mm series gap provides nontrivial reflection
and transmission. One dominant quasi-TEM mode is monitored at each port.
``build_scene()`` returns both the scene and its two-case ``EigenmodeStudy``.
The cases change the same excitation object between ports 1 and 2.

:download:`Complete Python model <../../examples/features/eigenmode_ports/example_4_complete_s_matrix/complete_s_matrix.py>`

.. code-block:: console

   python examples/features/eigenmode_ports/example_4_complete_s_matrix/complete_s_matrix.py --geometry-only
   python examples/features/eigenmode_ports/example_4_complete_s_matrix/complete_s_matrix.py
   python examples/features/eigenmode_ports/example_4_complete_s_matrix/plot_results.py

The aggregate ``complete_s_matrix_study.h5`` contains the 2 by 2 matrix at
101 frequencies from 4 to 8 GHz. The second FDTD case reuses the first build's
geometry and modal basis. The solver retains all measured incident waves,
including those at nominally passive channels, and solves ``B = S A``.
Incomplete or ill-conditioned incident bases invalidate the affected matrix
bins while retaining the measured matrices and diagnostics.

``complete_s_matrix.png`` plots magnitude and phase for all four entries.
Compare both S21 and S12 panels: this reciprocal structure should give equal
complex transmission responses within numerical error. Its geometric symmetry
also makes S11 and S22 agree. To resume an interrupted study, pass
``--restart 2`` with the same output stem and an existing compatible aggregate.

Example 5: a phased array and active S-parameters
-------------------------------------------------

Four virtual-waveguide-fed open apertures lie 18 mm apart along y. All four
channels are driven in one run, with equal amplitudes and a constant -108
degree progressive phase. Each port uses a single 10 GHz modal anchor: this
example demonstrates array excitation with a fixed reference profile rather
than a converged broadband modal interpolation study.

:download:`Complete Python model <../../examples/features/eigenmode_ports/example_5_phased_array/phased_array.py>`

.. code-block:: console

   python examples/features/eigenmode_ports/example_5_phased_array/phased_array.py --geometry-only
   python examples/features/eigenmode_ports/example_5_phased_array/phased_array.py
   python examples/features/eigenmode_ports/example_5_phased_array/plot_results.py

Ten uniform modal bins plus exact 9, 10, and 11 GHz values produce thirteen
output frequencies. NTFF uses the five integer-GHz bins and stores the dense
one-degree xy-plane cut. Total radiated power for directivity and gain comes
from an internal full-sphere quadrature, even though only the cut is saved.

``phased_array_active_sparameters.csv`` and HDF5 ``active_S`` describe this
coherent array state. Active reflection at channel q is its outgoing wave
divided by its measured incident wave, so it depends on all drive weights.
It is not an independent column of the S matrix. Use separate study cases to
recover the matrix.

The array-factor target is approximately 30 degrees from +x toward +y at
10 GHz. A constant phase increment makes the peak angle vary with frequency
(beam squint); aperture patterns, mutual coupling, and the grid also shift
the simulated peak. Try true delays through ``delay_s`` and compare the
frequency dependence. Keep each delayed pulse within the recording window.

Example 6: near and below cutoff
--------------------------------

The 6 mm wide air-filled rectangular guide has analytical TE10 cutoff near
24.9827 GHz. Seven of its 100 requested DFT points lie below cutoff.
Explicit anchors sample every below-cutoff bin, the first nine propagating
bins, and more widely spaced points thereafter. Additional candidates cover
the automatic pulse's transition spectrum.

:download:`Complete Python model <../../examples/features/eigenmode_ports/example_6_near_cutoff/near_cutoff.py>`

.. code-block:: console

   python examples/features/eigenmode_ports/example_6_near_cutoff/near_cutoff.py --geometry-only
   python examples/features/eigenmode_ports/example_6_near_cutoff/near_cutoff.py
   python examples/features/eigenmode_ports/example_6_near_cutoff/plot_results.py

``near_cutoff_sparameters.png`` compares coefficient-valid S11/S21 and the
analytical uniform-guide attenuation and phase. Shading identifies the
below-cutoff region. There, ``coefficient_valid_S`` may be true while
``power_wave_valid_S`` is false. The decaying field coefficient is useful,
but its squared magnitude is not transported real power. At exact cutoff
the forward/backward basis coalesces: inspect conditioning, move the DFT grid,
and refine the anchor sampling to check sensitivity.

Example 7: physically aligned circular TE11
-------------------------------------------

The circular PEC guide assigns global y electric polarization to mode 1
and global x polarization to mode 2 using ``degenerate=(1, 2)`` and
``mode_polarizations={1: "y", 2: "x"}`` on both ports. Automatic broadband
anchors preserve those physical labels. The source uses a virtual guide;
the receiving port sits at the opposite longitudinal PML interface.

:download:`Complete Python model <../../examples/features/eigenmode_ports/example_7_degenerate_te11/circular_te11.py>`

.. code-block:: console

   python examples/features/eigenmode_ports/example_7_degenerate_te11/circular_te11.py --geometry-only
   python examples/features/eigenmode_ports/example_7_degenerate_te11/circular_te11.py --mode 1
   python examples/features/eigenmode_ports/example_7_degenerate_te11/plot_results.py --mode 1
   python examples/features/eigenmode_ports/example_7_degenerate_te11/circular_te11.py --mode 2
   python examples/features/eigenmode_ports/example_7_degenerate_te11/plot_results.py --mode 2

Only the excitation mode changes between the two full runs. Their separate
result plots show reflection and transmission into each labelled channel
and the centre receiver's Ex/Ey traces. The folder includes the equivalent
``circular_te11.in`` hash model and instructions for diagonal or quadrature
excitation.

Both ports set ``plot_fields=True`` (``y`` after ``auto`` in the hash model).
The standard gprMax modal plotter uses the aligned basis and includes the
requested electric direction in each title. A single run generates pictures
for both modes, regardless of which channel is excited. The unmodified
source-port pictures below show E on the left and H on the right, with one
row per retained anchor, including guard frequencies. Local u/v are global
x/y; E and H magnitudes are normalised independently.

.. figure:: ../../examples/features/eigenmode_ports/example_7_degenerate_te11/te11_mode1.png
   :alt: Mode 1 retains vertical electric polarization at every frequency anchor.
   :width: 100%

   Degenerate TE11 mode 1: global y electric polarization.

.. figure:: ../../examples/features/eigenmode_ports/example_7_degenerate_te11/te11_mode2.png
   :alt: Mode 2 retains horizontal electric polarization at every frequency anchor.
   :width: 100%

   Degenerate TE11 mode 2: global x electric polarization.

Direct eigenmode ports inside an HSG subgrid
--------------------------------------------

The same Python objects can be added to one ``SubGridHSG`` instead of the
main scene. Keep the band, all associated ports, waveform, and excitations
in that same grid. With ``autotranslate=True`` the port coordinates remain
global physical coordinates. The FDFD solver reads the final fine-grid
component material IDs and transverse cell sizes, and injection and monitoring
advance at every fine time step.

The entire aperture and its staggered Yee stencil must lie strictly within
the subgrid working region. No endpoint or adjacent normal magnetic plane may
touch the HSG coupling surface or enter the auxiliary/PML region. A modal
network cannot span different grids because its band and normalization are
local to the owning grid. Results appear under
``/subgrids/<subgrid ID>/eigenmode_ports/portN`` with fine ``dx_dy_dz`` and
``dt`` metadata; the CSV ends in ``_<subgrid ID>_sparameters.csv``.
Direct HSG ports and their optional virtual guides use the CPU or CUDA update
cycle and do not support MPI. CUDA auxiliary guides share their owner's device
context and local timestep; their fields remain on the device during updates.
See :doc:`input_api` for subgrid construction.

Supported configurations
------------------------

Ordinary eigenmode injection and monitoring support CPU, CUDA, OpenCL, and
Metal, including 2D TE/TM models. Main-grid direct ports and virtual guides
also support domain-decomposed MPI CPU models. Direct HSG ports and their
virtual guides run on CPU or CUDA and do not support MPI. An SIBC wall imposes the
stricter CPU main-grid restrictions in :doc:`impedance_surfaces`.

An ``EigenmodeStudy`` reuses one fixed model through independent cases; its
supported combinations are documented in the study reference in
:doc:`input_api`. Support for an ordinary single run does not by itself
imply support for that run inside a reusable study.

Reading lossy and near-cutoff results
-------------------------------------

With gprMax's sign convention, a passive forward lossy mode has negative
``Im(n_eff)``. Its field amplitude decays with propagation. Do not change
the sign or take an absolute value to make it resemble a different solver's
display convention; compare attenuation and power in physical directions.
See :ref:`eigenmode-conventions` for the definitions.

Near cutoff, incoming and outgoing patterns can become difficult to
separate. Below cutoff, a meaningful field coefficient need not represent
transported real power. Check masks and conditioning, move bins away from
the exact singularity, and refine anchors. Neither more frequency bins nor
a finite output number certifies accuracy.

Accuracy checklist
------------------

* Inspect geometry-only modal and waveform plots before a full solve. Check
  field confinement, symmetry, polarization, and enough room for the pulse.
* Keep port planes in uniform guide sections and include the whole guided
  field in the aperture. Move bends and discontinuities away from the plane.
* Provide a physical PML continuation or virtual termination. A reflection
  from the feed end can contaminate the device's measured reflection.
* Refine mesh spacing, anchor spacing, and recording duration independently.
  Run long enough for propagation and ring-down; more bins only sample the
  same finite record more densely.
* Use ``waveform="auto"`` or a sufficiently band-limited custom waveform.
  A warning about discarded DC or Nyquist content means the sampled pulse
  needs attention; the source cannot represent arbitrary frequencies.
* Treat overlap warnings as a reason to inspect neighbouring profiles.
  Declare genuinely degenerate groups when appropriate; do not use a
  fixed single anchor to conceal an incorrect physical channel label.

The numerical limitations, convergence considerations, and validation
evidence are collected in :doc:`eigenmode_port_theory`.

.. seealso::

   :doc:`eigenmode_port_theory` derives the modal operators, tracking,
   broadband source synthesis, and measurement equations.
