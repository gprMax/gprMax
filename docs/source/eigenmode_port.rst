.. _eigenmode:
.. _eigenmode-port:
.. _fdfd-eigenmode-source:

****************************************
Eigenmode Ports and S-parameter Analysis
****************************************

An eigenmode port solves the waveguide's transverse field profile, launches
that profile into FDTD, and separates the measured field into incident and
outgoing modal coefficients. Each Python API introduction is followed by its
equivalent hash input command. Six runnable Python examples follow, and the
final section develops the mathematics used by the solvers, sources, and
monitors.

Python API
==========

The three main objects are ``gprMax.EigenmodeBand``,
``gprMax.EigenmodePort``, and ``gprMax.EigenmodeExcitation``. Add them to the
same ``gprMax.Scene`` as the geometry and materials. A band defines the
frequencies to measure, a port defines a reference plane and monitored modes,
and an excitation selects a channel to drive. A *channel* is a
``(port, mode)`` pair.

One excitation produces one S-matrix column. Several excitations produce a
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

EigenmodeBand arguments
-----------------------

.. code-block:: python

   gprMax.EigenmodeBand(
       id="band", fmin=4e9, fmax=6e9, points=21,
       frequencies=(4.25e9, 4.75e9),
       transition="auto", spectral_threshold=1e-3,
   )

.. list-table::
   :header-rows: 1
   :widths: 25 22 53

   * - Argument
     - Default
     - Meaning and constraints
   * - ``id``
     - Required
     - Non-empty identifier without whitespace. Exactly one band is allowed per grid.
   * - ``fmin``, ``fmax``
     - Required
     - Positive, finite frequency limits with ``fmax >= fmin``.
   * - ``points``
     - Required
     - Number of equally spaced output frequencies from ``fmin`` to ``fmax``, including both endpoints. Use one point when the limits are equal and at least two when they differ.
   * - ``frequencies``
     - No extra frequencies
     - Extra output frequencies in Hz, between ``fmin`` and ``fmax`` inclusive. Supply one number or a sequence. These are added to the frequencies selected by ``points``, then sorted from low to high; repeated values appear only once.
   * - ``transition``
     - ``"auto"``
     - Automatic pulse transition width, or a positive finite width in Hz.
   * - ``spectral_threshold``
     - ``1e-3``
     - Relative amplitude threshold for significant waveform spectral support; strictly between zero and one.

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

   #eigenmode_band: id fmin fmax points [frequency ...]
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
   )

.. list-table::
   :header-rows: 1
   :widths: 25 22 53

   * - Argument
     - Default
     - Meaning and constraints
   * - ``port``
     - Required
     - Unique positive, one-based port number within the grid.
   * - ``p1``, ``p2``
     - Required
     - Opposite physical corners of an axis-aligned aperture. Exactly one finite coordinate must match, defining its normal; the invariant 2D axis is ignored in that test.
   * - ``direction``
     - Required
     - ``"+"`` or ``"-"`` along that normal. Point both port directions into the device for the usual S convention.
   * - ``modes``
     - Required
     - An increasing sequence of unique positive mode indices, or a positive integer mode count. ``modes=2`` means modes 1 and 2; ``modes=(2,)`` means only mode 2.
   * - ``anchors``
     - ``"auto"``
     - One modal solve frequency, an increasing sequence of positive finite frequencies, or automatic candidate selection.
   * - ``plot_fields``
     - ``None``
     - ``True`` forces modal-field PNGs, ``False`` suppresses them. ``None`` enables them only in geometry-only runs.

Place the aperture in a longitudinally uniform section and include the whole
guided field, including evanescent tails around a dielectric core. The normal
comes from the matching coordinate: equal x coordinates give an x-normal
port, equal y coordinates a y-normal port. Mode numbers are solver ordering,
not guaranteed physical labels. Inspect the E/H profiles before identifying
a solution as TE10, quasi-TEM, or a particular guided slab mode.

An excited port launches the selected modal field and measures returning
waves, but it does not absorb those waves. Continue the guide behind the
port through a domain PML, or attach a :ref:`virtual-waveguide` to provide a
matched termination. Point receivers are optional diagnostics; S-parameters
come from the modal port monitors.

Hash command: #eigenmode_port
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   #eigenmode_port: port x1 y1 z1 x2 y2 z2 direction modes auto|anchor [anchor ...] [y|n]
   #eigenmode_port: 1 0.02 0.005 0 0.02 0.075 inf + 1,2 auto
   #eigenmode_port: 2 0.235 0.005 0 0.235 0.075 inf - 1,2 4e9 5e9 6e9 y

Use ``inf`` for an invariant extent and comma-separated mode indices such as
``1,2``. Specify ``auto`` or space-separated modal anchor frequencies. The
optional final ``y`` or ``n`` forces or suppresses the port's field plots;
omitting it retains the geometry-only default.

EigenmodeExcitation arguments
-----------------------------

.. code-block:: python

   gprMax.EigenmodeExcitation(
       port=1, mode=1, waveform="auto", amplitude=1.0,
       phase_deg=0.0, delay_s=0.0, plot_waveform=None,
   )

.. list-table::
   :header-rows: 1
   :widths: 25 22 53

   * - Argument
     - Default
     - Meaning and constraints
   * - ``port``, ``mode``
     - Required
     - Existing port and one of its monitored modes. Each driven channel must be unique.
   * - ``waveform``
     - ``"auto"``
     - Band-adapted finite pulse, or the ID of an explicitly added ``gprMax.Waveform``. Simultaneous drives must share one base waveform.
   * - ``amplitude``
     - ``1.0``
     - Finite nonzero modal-amplitude scale. Omit the excitation to make a channel passive.
   * - ``power``
     - Not set
     - Positive finite relative incident-power scale, applying amplitude ``sqrt(power)``. Mutually exclusive with ``amplitude``; it scales the base pulse rather than specifying a constant time-domain power.
   * - ``phase_deg``
     - ``0.0``
     - Finite constant spectral phase in degrees.
   * - ``delay_s``
     - ``0.0``
     - Finite true time delay in seconds, applied as ``exp(-1j*2*pi*f*delay_s)``. Ensure the shifted pulse fits the time window.
   * - ``plot_waveform``
     - ``None``
     - Force or suppress the waveform/spectrum PNG with ``True`` or ``False``; ``None`` enables it only for geometry-only runs.

Automatic excitation requires ``fmax > fmin`` and a time window long enough
to contain the pulse. For a single-frequency band, add an explicit waveform,
for example ``gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9,
id="tone")``, and select ``waveform="tone"``. A custom broadband waveform's
exact sampled spectrum must fit the declared band's supported coverage.
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

   #eigenmode_excitation: port mode [waveform] [amplitude] [phase_deg] [delay_s] [y|n]
   #eigenmode_excitation: 1 1 auto
   #eigenmode_excitation: 2 1 auto 0.5 90 0 y

The waveform, amplitude, phase, and delay default to ``auto``, 1, 0, and 0.
Supply preceding positional values when setting later values. The final
``y`` or ``n`` independently controls this drive's waveform/spectrum plot;
omitting it retains the geometry-only default. Repeated commands drive
distinct channels of one coherent state. The hash interface uses amplitude;
``power`` is a Python-only alternative.

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
`Virtual-guide aperture coupling`_ for the field-update equations.

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

.. list-table::
   :header-rows: 1
   :widths: 25 22 53

   * - Argument
     - Default
     - Meaning and constraints
   * - ``port``
     - Required
     - Positive, one-based number of an existing ``EigenmodePort`` in the same grid. Attach at most one virtual guide per port.
   * - ``length_cells``
     - ``30``
     - Total auxiliary-guide length, **including** its PML and source clearance. A positive integer, at least ``pml_cells + source_clearance_cells + 3``.
   * - ``pml_cells``
     - ``12``
     - Thickness of the auxiliary guide's absorbing layer at the end away from the port. An integer of at least 2 cells.
   * - ``source_clearance_cells``
     - ``6``
     - Distance between the internal source plane and the start of that PML. A positive integer number of cells, at least 1.
   * - ``pml_profile``
     - ``None``
     - Name of an existing reusable PML profile. ``None`` uses the global PML formulation and absorption settings (CFS terms).

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

   #virtual_waveguide: port [length_cells] [pml_cells] [source_clearance_cells] [pml_profile]
   #virtual_waveguide: 1 30 12 6

The required port number identifies an existing ``#eigenmode_port``. Cell
counts default to 30, 12, and 6, with the same constraints as the Python API.
The optional final token names a reusable PML profile. An unexcited virtual
port provides a passive matched termination.

Placement and accuracy checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This feature is experimental. It currently requires a 3D internal port, a
locally uniform, non-dispersive cross-section, and at least two cells across
each transverse direction. Models using impedance volumes are not supported.
Main-grid CPU, CUDA, OpenCL, Metal, and domain-decomposed MPI CPU paths are
supported; HSG subgrid virtual ports use the CPU fine-grid update cycle.

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
anchor are resolved independently for each port and mode. Passive-only setups
use the requested band alone.

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
  More anchors cannot make an exactly degenerate pair uniquely identifiable;
  a narrower band or a single anchor may be needed.
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
in-band failure may instead select a single band-centre anchor, provided that
it carries forward real power. This allows the run to proceed with a fixed
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

Running hash input files
------------------------

Run a hash model with ``python -m gprMax model.in --geometry-only`` to inspect
it, then ``python -m gprMax model.in -outputfile results/model`` to simulate.
The tutorial models below are Python programs and run directly with Python.

Tutorial examples
=================

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

.. literalinclude:: ../../examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py
   :language: python
   :caption: ``example_1_straight_waveguide/straight_waveguide.py``
   :linenos:

.. code-block:: console

   python examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py --geometry-only
   python examples/features/eigenmode_ports/example_1_straight_waveguide/straight_waveguide.py
   python examples/features/eigenmode_ports/example_1_straight_waveguide/plot_results.py

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

.. literalinclude:: ../../examples/features/eigenmode_ports/example_2_curved_waveguide/curved_waveguide.py
   :language: python
   :caption: ``example_2_curved_waveguide/curved_waveguide.py``
   :linenos:

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

.. literalinclude:: ../../examples/features/eigenmode_ports/example_3_antenna_and_farfield/horn_antenna.py
   :language: python
   :caption: ``example_3_antenna_and_farfield/horn_antenna.py``
   :linenos:

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

.. literalinclude:: ../../examples/features/eigenmode_ports/example_4_complete_s_matrix/complete_s_matrix.py
   :language: python
   :caption: ``example_4_complete_s_matrix/complete_s_matrix.py``
   :linenos:

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

.. literalinclude:: ../../examples/features/eigenmode_ports/example_5_phased_array/phased_array.py
   :language: python
   :caption: ``example_5_phased_array/phased_array.py``
   :linenos:

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

.. literalinclude:: ../../examples/features/eigenmode_ports/example_6_near_cutoff/near_cutoff.py
   :language: python
   :caption: ``example_6_near_cutoff/near_cutoff.py``
   :linenos:

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
Direct HSG ports and their optional virtual guides use the CPU update cycle
and do not support MPI. See :doc:`input_api` for subgrid construction.

Mathematical formulation
========================

The remaining sections describe the component-sampled FDFD eigenproblems,
power normalization and modal reconstruction, followed by spectrum synthesis,
I/Q injection, Yee staggering, direct DFT reception, and multimode
decomposition.

Overview
--------

gprMax uses two finite-difference frequency-domain eigenmode solvers:

.. list-table::
   :header-rows: 1

   * - Time-domain model
     - Modal cross-section
     - Solver
   * - 2D TM or TE
     - One physical transverse coordinate
     - ``FDFD_1D_mode_solver``
   * - 3D
     - Two physical transverse coordinates
     - ``FDFD_2D_mode_solver``

Both solvers operate directly on component-sampled Yee grids, accept complex
permittivity and permeability, enforce PEC and PMC constraints at the
corresponding component locations, select the passive effective-index branch,
and return either real-power-normalised or E/H-balanced modal fields. Only the
propagating, real-power-normalised subset can be used by ``EigenmodeSource``.

Frequency and wavenumber convention
-----------------------------------

Both FDFD solvers distinguish the physical solve frequency from the symbols
of the FDTD differences. At each modal solve frequency,

.. math::

   \omega = 2\pi f, \qquad k_0 = \frac{\omega}{c}.

These physical quantities are exposed as ``solver.omega`` and ``solver.k0``.

Given the owning grid's time step :math:`\Delta t`, the eigenproblem and
field reconstruction use the leapfrog temporal symbol

.. math::

   \Omega = \frac{2}{\Delta t}\sin\left(\frac{\omega\Delta t}{2}\right),
   \qquad k_{0,\mathrm{operator}} = \frac{\Omega}{c}.

These are exposed as ``solver.operator_omega`` and ``solver.operator_k0``.
The transverse operators retain their component-sampled Yee differences. The eigenvalue
determines the longitudinal difference symbol :math:`K_w`, rather than the
phase propagation constant :math:`\beta` directly:

.. math::

   \lambda&=-n_{\mathrm{operator}}^2,
   \qquad n_{\mathrm{operator}}=\frac{K_w}{k_{0,\mathrm{operator}}},\\
   \beta&=\frac{2}{\Delta w}\sin^{-1}\left(\frac{K_w\Delta w}{2}\right),
   \qquad n_{\mathrm{eff}}=\frac{\beta}{k_0}.

Here :math:`\Delta w` is the normal cell spacing. The inverse-sine branch is
chosen for passive forward propagation, including decay for evanescent
modes. ``solver.beta`` stores the phase propagation constant. Field
reconstruction uses ``operator_neff``; the public ``complex_neff`` and
``modal_real_neff`` describe its effective index. This distinction keeps the
E/H amplitude relationship consistent with the discrete curls while supplying
the correct phase to source and monitor staggering.

Lossless modes at or beyond the longitudinal spatial band edge
:math:`|K_w\Delta w/2|\geq1` are retained for inspection but have
``power_valid=False`` and cannot be used for source injection.

Eigenmode sources pass the owning grid's time step and normal spacing
automatically, including a subgrid's own values. Direct low-level callers
enable the same convention with ``fdtd_dt`` and ``propagation_spacing``.
Omitting ``fdtd_dt`` uses :math:`\Omega=\omega` and makes ``operator_k0`` equal
to ``k0``; omitting ``propagation_spacing`` uses :math:`\beta=K_w`. Omitting both preserves the
continuum time and longitudinal conventions of earlier low-level calls.
Waveforms, Fourier transforms, and half-time-step phase factors always use
the physical frequency :math:`\omega`.

The source material extraction also includes the midpoint factor in static
electric and magnetic conductivity, including static electric conductivity
in dispersive media. Bulk material poles and their associated Drude or
inclusive conductivity terms are still evaluated from their analytic
physical-frequency response, rather than the exact FDTD ADE transfer.
Dispersive materials therefore retain a time-discretization mismatch in
their pole response that should be checked by convergence.

1D Scalar Solver for 2D Models
------------------------------

``fdfd_1d_mode_solver.py`` supplies the scalar, Yee-staggered mode solve used
by eigenmode sources in gprMax 2D TM and TE domains.

1D Coordinates and Yee Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The local basis is ``(t, a, w)``:

``t``
    The one physical transverse coordinate.

``a``
    The invariant 2D axis.

``w``
    The in-plane propagation direction and source normal.

For ``N`` cells along ``t``, the native staggered shapes are:

.. list-table::
   :header-rows: 1

   * - Material array
     - Field component
     - Shape
   * - ``eps_r_t``
     - ``E_t``
     - ``(N,)``
   * - ``eps_r_a``
     - ``E_a``
     - ``(N + 1,)``
   * - ``eps_r_w``
     - ``E_w``
     - ``(N + 1,)``
   * - ``mu_r_t``
     - ``H_t``
     - ``(N + 1,)``
   * - ``mu_r_a``
     - ``H_a``
     - ``(N,)``
   * - ``mu_r_w``
     - ``H_w``
     - ``(N,)``

The TM reduction solves the node-sampled scalar field ``E_a`` and reconstructs
``H_t`` and ``H_w``. The TE reduction solves the cell-sampled scalar field
``H_a`` and reconstructs ``E_t`` and ``E_w``. No derivative is taken through
gprMax's artificial one-cell TM or two-cell TE invariant-axis thickness.

1D Inputs
^^^^^^^^^

The constructor signature is:

.. code-block:: python

   FDFD_1D_mode_solver(
       frequency,
       dt,
       mode_index,
       polarization,
       eps_r_t,
       eps_r_a,
       eps_r_w,
       mu_r_t,
       mu_r_a,
       mu_r_w,
       pec_t_mask=None,
       pec_a_mask=None,
       pec_w_mask=None,
       pmc_t_mask=None,
       pmc_a_mask=None,
       pmc_w_mask=None,
       guess=None,
       *,
       fdtd_dt=None,
       propagation_spacing=None,
   )

``frequency``
    Modal solve frequency in Hz.

``dt``
    Yee-cell spacing along the physical transverse coordinate ``t``, in
    metres. Despite its name, this is a spatial step rather than a time step.

``mode_index``
    Zero-based requested mode. The solver computes ``mode_index + 1`` modes
    and exposes the selected one through ``modal_Et``, ``modal_Ea``,
    ``modal_Ew``, ``modal_Ht``, ``modal_Ha``, ``modal_Hw`` and
    ``modal_real_neff``.

``polarization``
    ``TM`` selects the ``E_a`` scalar problem. ``TE`` selects the ``H_a``
    scalar problem.

``eps_r_*`` and ``mu_r_*``
    Complex relative material arrays sampled at the component locations in
    `1D Coordinates and Yee Shapes`_.

``pec_*_mask`` and ``pmc_*_mask``
    Optional boolean masks at the matching electric or magnetic component
    locations. Non-finite values in the corresponding material arrays are
    also interpreted as constraints.

``guess``
    Optional ARPACK shift. If omitted, the solver derives a shift from the
    largest finite material magnitude.

``fdtd_dt``
    Optional keyword-only FDTD time step in seconds. A positive finite value
    enables the leapfrog frequency symbol and requires ``frequency`` below
    temporal Nyquist. This is distinct from the transverse spatial ``dt``.

``propagation_spacing``
    Optional keyword-only positive finite cell spacing along ``w``, in
    metres. Enables conversion from the longitudinal difference symbol to
    the phase propagation constant. See `Frequency and wavenumber convention`_.

1D Eigenproblem and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The solver constructs a node-to-cell difference operator ``D_nc`` and its
negative adjoint:

.. code-block:: text

   D_nc : node fields -> cell fields
   D_cn = -D_nc.H : cell fields -> node fields

Both operators are normalised by ``k0 * dt``. For TM, the scalar eigenproblem
is assembled on the ``E_a`` nodes:

.. code-block:: text

   A_TM = -mu_t [D_cn inv(mu_w) D_nc + eps_a]
   A_TM E_a = lambda E_a

For TE, it is assembled on the ``H_a`` cells:

.. code-block:: text

   A_TE = -eps_t [D_nc inv(eps_w) D_cn + mu_a]
   A_TE H_a = lambda H_a

In both cases:

.. code-block:: text

   operator_neff = sqrt(-lambda)

gprMax uses ``exp(+j*omega*t - j*beta*w)``. The square-root branch is
therefore chosen with ``Re(operator_neff) >= 0``. For a passive mode
``Im(operator_neff) <= 0``; a purely evanescent mode uses the negative-imaginary
branch so that it decays in positive ``w``. The public ``complex_neff`` is
then recovered as described in `Frequency and wavenumber convention`_.

PEC constraints remove electric scalar degrees of freedom from the TM
problem, while PMC constraints remove magnetic scalar degrees of freedom from
the TE problem. Longitudinal inverse-material operators are evaluated only on
unconstrained degrees of freedom: constrained entries receive a zero inverse.
After the reduced sparse eigenproblem is solved, the eigenvectors are expanded
back to their full node or cell arrays and every constrained field component
is explicitly zeroed.

1D Field Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^

For TM, the selected ``E_a`` eigenvector gives:

.. code-block:: text

   H_t = -operator_neff E_a / (eta0 mu_t)
   H_w = i inv(mu_w) D_nc E_a / eta0

For TE, the selected ``H_a`` eigenvector gives:

.. code-block:: text

   E_t = eta0 operator_neff H_a / eps_t
   E_w = -i eta0 inv(eps_w) D_cn H_a

The other three field components are identically zero for the selected 2D
polarization. The reconstructed electric fields are in V/m and magnetic
fields are in A/m.

1D Normalisation and Phase Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each propagating mode is normalised to one watt per metre of invariant-axis
length. A non-propagating mode instead uses the positive E/H-balanced scale
defined in `Mode Selection, Fields, and Power`_; it is retained
as a monitor reference but is not a one-watt power wave. For TM, node-sampled
``E_a`` and ``H_t`` are first averaged onto transverse cells:

.. code-block:: text

   P_TM = 0.5 Re sum(-E_a H_t*) dt

For TE, ``E_t`` and ``H_a`` already share the cell locations:

.. code-block:: text

   P_TE = 0.5 Re sum(E_t H_a*) dt

If a passive branch has negative real power along the solver's forward axis,
the propagation-constant branch is reversed and its dependent fields are
reconstructed before normalisation. Each complex mode is then phase-rotated
to a deterministic real-profile convention used for tracking and, for a
propagating source anchor, FDTD injection.

1D gprMax Integration
^^^^^^^^^^^^^^^^^^^^^

``EigenmodeSource`` samples component materials from the mode's live invariant
layer, supplies the corresponding PEC/PMC masks, and maps the returned line
profiles back into the thin 3D Yee arrays used by the FDTD source kernels.
The TM source uses the single live invariant layer; the TE source uses the
shared interior layer of its two-cell invariant thickness. Inactive components
and TE outer boundary planes are explicitly zero.

Modes are selected using the same shift-invert convention as the full-vector
2D solver. ``plot_fields`` writes one row per computed mode with line plots of
all three active fields, including their node- or cell-sampled locations.

2D Full-Vector Solver for 3D Models
-----------------------------------

``fdfd_2d_mode_solver.py`` contains ``FDFD_2D_mode_solver``, the full-vector
solver used when a gprMax eigenmode source has two physical transverse
coordinates.

2D Coordinates and Yee Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

2D Inputs
^^^^^^^^^

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
       surface_boundary=None,
       *,
       fdtd_dt=None,
       propagation_spacing=None,
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
    Yee component locations listed in `2D Coordinates and Yee Shapes`_.

``pec_u_mask``, ``pec_v_mask``, ``pec_w_mask``
    Optional explicit boolean masks for constrained electric degrees of
    freedom. They must match the ``E_u``, ``E_v`` and ``E_w`` shapes.

``pmc_u_mask``, ``pmc_v_mask``, ``pmc_w_mask``
    Optional explicit boolean masks for constrained magnetic degrees of
    freedom. Non-finite entries in the matching permeability arrays are also
    interpreted as PMC.

``guess``
    Optional ARPACK shift. If omitted, the solver chooses a conservative shift
    from the largest finite material magnitude.

``surface_boundary``
    Optional compiled impedance-volume boundary. See :doc:`impedance_surfaces`
    for retained-component masks and clipped curl rows.

``fdtd_dt``
    Optional keyword-only FDTD time step in seconds. A positive finite value
    enables the leapfrog frequency symbol and requires ``frequency`` below
    temporal Nyquist.

``propagation_spacing``
    Optional keyword-only positive finite cell spacing along ``w``, in
    metres. Enables conversion from the longitudinal difference symbol to
    the phase propagation constant. See `Frequency and wavenumber convention`_.

2D Array Ordering
^^^^^^^^^^^^^^^^^

All component arrays are flattened with Fortran order:

.. code-block:: python

   flat = array.ravel(order='F')

and modal vectors are reshaped back with:

.. code-block:: python

   array = vector.reshape((*shape, num_modes), order='F')

There is no axis-order switch. gprMax must pass local ``u``/``v`` slices in the
same native transverse ordering used by the extracted source plane.

2D PEC Handling
^^^^^^^^^^^^^^^

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

2D Eigenproblem
^^^^^^^^^^^^^^^

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

where the operator index is recovered from:

.. code-block:: python

   operator_neff = sqrt(-eigenvalue)

Here the matrix ``Omega = P * Q`` is distinct from the scalar temporal symbol
:math:`\Omega`. The branch follows ``exp(+j*omega*t - j*beta*w)``:
``Re(operator_neff) >= 0`` and ``Im(operator_neff) <= 0`` for passive forward
propagation. When the real part is zero, the negative-imaginary branch gives
evanescent decay in positive ``w``. The public ``complex_neff`` is then
recovered as described in `Frequency and wavenumber convention`_.

Because the operators connect the correct staggered Yee component grids, there
is no separate PEC-neighbour spurious-mode rejection heuristic in this solver.
The old candidate scoring/filtering path has been removed.

2D Degree-of-Freedom Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PEC constraints are applied by removing constrained transverse electric degrees
of freedom from the eigenproblem:

.. code-block:: python

   Omega = Omega[self.free_euv_mask, :][:, self.free_euv_mask]

When the reduced matrix has only one more degree of freedom than the requested
mode count, the solver uses a dense eigensolve because ARPACK requires
``k < N - 1``. Larger systems use shift-invert ARPACK and retry with a
roundoff-scale shift perturbation if the original shift produces a singular
factorisation. The reduced eigenvectors are then expanded back to the full
transverse field-vector size and constrained fields are explicitly zeroed.

The inverse ``eps_r_ww`` and ``mu_r_ww`` operators are built only on free
longitudinal degrees of freedom. Constrained entries receive zero inverse
values so that no division by ``np.inf`` or placeholder data affects the
reconstructed fields.

2D Field Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^

After solving the eigenproblem, the solver reconstructs:

* ``E_u`` and ``E_v`` directly from the transverse eigenvector.
* ``H_u`` and ``H_v`` from ``Q * Euv / sqrt(eigenvalue)``, using the branch
  ``sqrt(eigenvalue) = +j*operator_neff`` selected by the propagation convention.
* ``E_w`` from transverse magnetic curl terms.
* ``H_w`` from transverse electric curl terms.

Magnetic fields are converted to physical A/m using ``eta0``:

.. code-block:: python

   H = 1j * H_normalized / eta0

The solver then zeroes all constrained fields to ensure returned modal fields
satisfy the enforced constraints exactly.

2D Normalisation and Phase Alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Propagating modes are normalised to carry one watt of time-average power.
Non-propagating modes instead use the finite E/H-balanced scale defined in
`Mode Selection, Fields, and Power`_; they can enter the tracked
monitor-reference bank but not the source/power bank. Real power is computed
from cell-centred transverse Poynting flux by averaging the staggered
transverse fields onto local cells:

.. code-block:: text

   P = 0.5 * Re integral((E_u * H_v* - E_v * H_u*) dA)

If a passive branch carries negative real power along the solver's forward
axis, the propagation-constant branch is reversed and all dependent fields
are reconstructed before normalisation. Each complex mode is then
phase-rotated to a deterministic real-profile convention. This makes plotted,
tracked, and (for propagating source anchors) injected fields easier to
compare.

2D gprMax Integration
^^^^^^^^^^^^^^^^^^^^^

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
* ``sm == inf`` is converted to ``np.inf + 0j``, which the solver treats as
  PMC.

After solving, ``sources.py`` maps local modal fields back to global component
slots. The Cython injection kernels consume the transverse components with
their native staggered shapes; longitudinal modal fields are stored but are not
used for TF/SF source corrections.

Limitations
-----------

* Material tensors are diagonal in the local ``u``/``v``/``w`` basis.
* The finite-difference operators use first-order sparse Yee-grid differences.
* Bulk dispersive material poles use their analytic physical-frequency
  response. Matching their exact FDTD ADE transfer remains a separate step
  beyond the temporal and longitudinal difference compensation.
* A single-frequency source reuses one solved profile across the waveform
  spectrum. A broadband source instead phase-aligns and interpolates several
  anchor solves, but its accuracy is limited by anchor spacing and it must not
  be used through degeneracies or mode crossings.

Recommended 2D-Solver Usage
---------------------------

For gprMax integration, use this path:

1. Extract local ``eps_r_uu``, ``eps_r_vv`` and ``eps_r_ww`` from Yee electric
   component material IDs with native staggered shapes.
2. Mark electric PEC entries with ``np.inf + 0j`` or explicit local PEC masks.
3. Extract local ``mu_r_uu``, ``mu_r_vv`` and ``mu_r_ww`` from Yee magnetic
   component material IDs with native staggered shapes.
4. Mark magnetic PMC entries with ``np.inf + 0j`` or explicit local PMC masks.
5. Construct ``FDFD_2D_mode_solver`` using local ``du`` and ``dv``, the owning
   grid's ``fdtd_dt``, and the normal cell size as ``propagation_spacing``.
6. Call ``solver.solve()``.
7. Use ``solver.modal_Eu``, ``solver.modal_Ev``, ``solver.modal_Hu`` and
   ``solver.modal_Hv`` for transverse eigenmode source injection after mapping
   local components back to global gprMax components.

Source synthesis and FDTD injection
-----------------------------------

This section connects the FDFD solvers described above to the complete
eigenmode band/port/excitation workflow. It fixes the phasor and propagation signs,
shows how a solved mode becomes real FDTD update terms, and describes the
single-frequency, in-phase/quadrature (I/Q), and broadband synthesis paths.

Phasor, Fourier, and Propagation Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

gprMax advances real fields in time. Whenever a complex frequency-domain
field is used, it follows the electrical-engineering convention

.. math::

   \mathbf{E}(u,v,w,t)
   = \operatorname{Re}\left\{
       \widetilde{\mathbf{E}}(u,v,\omega)
       \exp(+j\omega t-j\beta w)
     \right\}.

The forward Fourier transform uses the complementary negative-sign kernel,

.. math::

   X(\omega)=\int_{-\infty}^{\infty}x(t)\exp(-j\omega t)\,dt,

so NumPy's ``rfft`` returns the positive-frequency coefficients used by this
phasor convention, and ``irfft`` reconstructs the real signal with positive
frequency time dependence ``exp(+j*omega*t)``. The FFT sign is not a second
physical convention: it is the analysis kernel paired with the phasor
synthesis sign.

With ``exp(+j*omega*t)``, the continuum Maxwell curl equations are

.. math::

   \nabla\times\widetilde{\mathbf{E}}
   &= -j\omega\widetilde{\boldsymbol{\mu}}
      \widetilde{\mathbf{H}},\\
   \nabla\times\widetilde{\mathbf{H}}
   &= +j\omega\widetilde{\boldsymbol{\epsilon}}_c
      \widetilde{\mathbf{E}}.

Their continuum electric and magnetic conductivity terms are

.. math::

   \epsilon_{r,c}(\omega)
   &= \epsilon_r(\omega)
      -j\frac{\sigma}{\omega\epsilon_0},\\
   \mu_{r,c}(\omega)
   &= \mu_r(\omega)
      -j\frac{\sigma_m}{\omega\mu_0}.

For nondispersive material extraction on an FDTD grid, the solver instead
uses the exact midpoint conductivity terms

.. math::

   \epsilon_{r,c}^{\mathrm{Yee}}
   &= \epsilon_r-j\frac{\sigma\cos(\omega\Delta t/2)}{\Omega\epsilon_0},\\
   \mu_{r,c}^{\mathrm{Yee}}
   &= \mu_r-j\frac{\sigma_m\cos(\omega\Delta t/2)}{\Omega\mu_0},

with the temporal symbol from `Frequency and wavenumber convention`_.
Low-level callers supply their own complex relative material arrays; passing
``fdtd_dt`` does not reinterpret those arrays as conductivity parameters.

For a forward passive mode,

.. math::

   \beta=k_0n_{\mathrm{eff}}=\beta_r-j\alpha,
   \qquad \alpha\geq 0,

and therefore

.. math::

   \exp(-j\beta w)
   =\exp(-j\beta_r w)\exp(-\alpha w).

The selected propagation branch consequently satisfies

.. math::

   \operatorname{Re}(n_{\mathrm{eff}})&\geq 0,\\
   \operatorname{Im}(n_{\mathrm{eff}})&\leq 0
   \quad\text{for passive propagation}.

If the real part is numerically zero, a purely evanescent mode uses
``Im(n_eff) < 0`` so it decays in positive local ``w``. The imaginary part of
the square root is never replaced by its absolute value; its sign contains
the loss or gain information.

As a continuum example, at 5 GHz a homogeneous material with
``epsilon_r=9`` and ``sigma=2 S/m`` has

.. math::

   \epsilon_{r,c}\simeq 9-j7.190,
   \qquad
   n\simeq 3.203-j1.122.

This gives :math:`\alpha=-k_0\operatorname{Im}(n)\simeq
117.6\ \mathrm{m}^{-1}`. The field magnitude after 0.5 mm is approximately
``exp(-alpha * 0.5e-3) = 0.943``.

Build-Time Workflow
^^^^^^^^^^^^^^^^^^^

The source is prepared after geometry construction, when material IDs already
refer to their final Yee-component locations. The workflow is:

1. Validate the source plane, requested direction, mode index, waveform, and
   one or more solve frequencies.
2. Choose local coordinates ``(u, v, w)``. The ``u`` and ``v`` axes lie in the
   source plane and ``w`` is normal to it. In a 2D TM or TE model, one of the
   transverse axes is the invariant axis and only one is physical.
3. Extract complex relative permittivity and permeability at every native Yee
   component position on the source plane. PEC and PMC cells become explicit
   component constraints.
4. At each requested frequency, solve either the 1D scalar TM/TE problem for a
   2D FDTD model or the 2D full-vector problem for a 3D FDTD model, using the
   owning grid's time step and normal cell spacing.
5. Reconstruct all modal E and H components, zero constrained degrees of
   freedom, apply either real-power or balanced E/H normalization, and choose
   a consistent global phase.
6. Map local modal arrays back to global x/y/z component slots. If the local
   coordinate mapping is left-handed, reverse H so that the Poynting direction
   remains correct.
7. Select real-only, single-frequency I/Q, or multi-anchor broadband
   synthesis.
8. During time stepping, apply tangential incident E and H as TF/SF
   corrections on the appropriate side of the source plane.

The modal solution describes the cross-section at the source reference plane.
Propagation away from that plane is performed by the FDTD grid itself. The
explicit propagation constant is used in the broadband E/H staggering and in
frequency interpolation, not to overwrite fields throughout the guide.

Mode Selection, Fields, and Power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The user-facing ``mode_index`` is one-based: mode 1 selects the first eigenpair
after the solver orders the computed eigenvalues. The low-level Python FDFD
solver classes retain zero-based array indices internally. Shift-invert sparse
eigensolving targets modes near a material-derived guess. Near cut-off,
degeneracy, or an eigenvalue crossing, the same numerical index can identify a
different physical mode at another frequency; broadband sources and receivers
therefore check adjacent-anchor overlap.

For a 3D model, transverse modal power is evaluated on local cells as

.. math::

   P=\frac{1}{2}\operatorname{Re}
   \sum_{u,v}
   \left(E_uH_v^*-E_vH_u^*\right)\Delta u\Delta v.

The native Yee components are averaged only as needed to place each product on
the same transverse cell. For a 2D model, the equivalent line integral gives
power per metre along the invariant axis. If a passive branch initially has
negative real power along the solver's forward axis, gprMax reverses its
propagation-constant branch and reconstructs the dependent fields. A
propagating mode is then scaled to one watt in 3D or one watt per metre in 2D.
A non-propagating mode has no independent forward real power and instead
receives a finite balanced E/H field scale for tracking and diagnostics; that
scale is not a one-watt normalization.

This normalization defines the scale of the modal profile at the source
plane. Multiplying the source waveform amplitude by a factor multiplies both
incident E and H by that factor; for a monochromatic mode, time-average power
therefore scales with the square of the waveform amplitude only where the
mode has valid real-power normalization.

The one-watt propagating profiles form the source-synthesis and real-power
bank. Modal monitoring also retains a second, tracked reference bank. Before
a reference profile is used in a generalized-only bin, both its E and H
fields are multiplied by the same factor so that

.. math::

   P_\mathrm{bal}=\frac{1}{4\eta_0}\int
   \left(|\mathbf E_t|^2+\eta_0^2|\mathbf H_t|^2\right)\,dA=1

(or the corresponding 2D line integral). This positive balanced quantity is
a field-scale convention, not transported real power. Applying it to both
propagating and evanescent monitor references prevents their original solver
scales from introducing an artificial coefficient-scale jump. It does not
license interpolating through the cutoff singularity: propagating and
evanescent references remain in separate interpolation branches.

The solver returns the positive-local-``w`` mode. A source requested in the
negative global direction retains the electric profile and reverses the
magnetic orientation in the TF/SF updates, as required for the opposite
Poynting vector.

Global Phase and the Real-Only Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An eigenvector has arbitrary complex phase. Before deciding how to inject a
single-frequency profile, gprMax finds the phase rotation that makes all
*tangential* E and impedance-scaled H components as real as possible. If
:math:`F_q` denotes every such component sample, with H samples first scaled
by :math:`\eta_0`, the rotation is

.. math::

   \phi=-\frac{1}{2}\arg\left(\sum_q F_q^2\right),
   \qquad F_q' = F_q\exp(j\phi).

Only tangential components enter this test because longitudinal mode
components can be intrinsically in quadrature while not participating in the
TF/SF correction. After rotation, the normalized imaginary residual is

.. math::

   r=\sqrt{
     \frac{\sum_q\left[\operatorname{Im}(F_q')\right]^2}
          {\sum_q\left|F_q'\right|^2}
   }.

If ``r <= 1e-8``, the tangential spatial profiles are effectively real. gprMax
stores their real parts and multiplies them directly by the requested real
waveform. The H waveform is evaluated with its Yee half-time-step and
half-normal-cell phase delay. This path avoids unnecessary FFT preparation for
ordinary lossless fixed-profile modes.

Why Complex Modes Need I/Q Injection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single real scalar waveform cannot reproduce a general complex spatial
profile. Let

.. math::

   F=F_R+jF_I,
   \qquad Z(t)=\operatorname{irfft}\{C(\omega)\}.

The desired real field is

.. math::

   \operatorname{Re}\{F C(\omega)\exp(j\omega t)\}.

gprMax realizes this with two real bases:

.. math::

   F_R\operatorname{irfft}\{C\}
   +F_I\operatorname{irfft}\{jC\}.

Because ``irfft(j*C)`` is the negative quadrature of ``irfft(C)``, this sum is
exactly the required real part. No complex values are passed to the real FDTD
update arrays.

When the single-frequency residual exceeds the tolerance, gprMax uses this
I/Q construction with one modal anchor. The same solved profile and
``n_eff`` are used for every significant FFT bin of the waveform. This is a
fixed-profile approximation: use multiple solve frequencies when the modal
shape or propagation constant varies appreciably across the waveform
bandwidth.

Broadband Anchor Solves and Phase Tracking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For solve frequencies :math:`f_k`, gprMax obtains fields
:math:`(\mathbf{E}_k,\mathbf{H}_k)` and effective indices :math:`n_k`.
Adjacent eigenvectors can carry unrelated arbitrary phases, so their complex
overlap is evaluated as

.. math::

   O_{k-1,k}=
   \frac{
     \langle\mathbf{E}_{k-1},\mathbf{E}_k\rangle
     +\langle\eta_0\mathbf{H}_{k-1},
              \eta_0\mathbf{H}_k\rangle
   }{
     \| (\mathbf{E}_{k-1},\eta_0\mathbf{H}_{k-1}) \|
     \| (\mathbf{E}_k,\eta_0\mathbf{H}_k) \|
   }.

Anchor ``k`` is multiplied by
``exp(-j*arg(O[k-1,k]))``. This makes interpolation follow a continuous phase
choice instead of blending arbitrary eigenvector phases. If
``abs(O) < 0.9``, gprMax warns that the mode may have crossed cut-off, become
degenerate, changed ordering, or been sampled too sparsely. If
``abs(O) < 0.6``, the ambiguity is too large for ordinary interpolation.
Multiple explicit anchors stop with an error. With automatic anchors, an
outer-guard failure trims the affected port/mode spectral tail; an in-band
failure selects the band-centre single-frequency basis only for that port and
mode. The candidate frequency list itself remains common.

Phase tracking is evaluated before the forward-power filter so that a solved
non-propagating candidate can still diagnose and represent branch continuity.
The forward-power filter produces the one-watt source/power mask, while every
successfully tracked retained candidate produces the monitor-reference mask.
A centre-only tracking fallback collapses both masks so that a rejected mode
cannot re-enter monitor interpolation. Non-propagating reference anchors are
never used by TF/SF source synthesis or treated as power waves.

Spectrum and Piecewise-Linear Modal Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the automatic waveform, gprMax first forms the requested smooth spectral
magnitude and finds the circular time support of its zero-phase analytic
envelope. It then applies the earliest discrete delay that places all envelope
samples above the significance threshold inside the causal record. This keeps
the requested spectrum while reserving as much of the simulation window as
possible for the packet to traverse the model and decay.

The real waveform is sampled at the FDTD interval for the requested number of
iterations. It is zero-padded to the next power of two at least twice as long
as the simulation record, and transformed with ``numpy.fft.rfft``. For FFT
bin :math:`m`, let its frequency and coefficient be :math:`f_m` and :math:`S_m`.
Non-finite samples and a zero or non-finite spectrum are errors; gprMax does
not replace the excitation with a zero-valued source.

For source synthesis, piecewise-linear weights :math:`w_{k,m}` interpolate
between surrounding propagating source anchors and satisfy

.. math::

   \sum_k w_{k,m}=1.

Below or above the anchor range, the nearest endpoint receives weight one.
This avoids a hard spectral truncation. Significant waveform energy outside
the anchor range is governed by the spectrum-coverage policy: the default is
an error, while an explicit ``warn`` policy permits endpoint extrapolation
with a warning. The source-interpolated fields and propagation constant are

.. math::

   \mathbf{E}_m &= \sum_k w_{k,m}\mathbf{E}_k,\\
   \mathbf{H}_m &= \sum_k w_{k,m}\mathbf{H}_k,\\
   n_{\mathrm{operator},m} &= \sum_k w_{k,m}n_{\mathrm{operator},k},\\
   K_m &= \frac{\Omega(f_m)}{c}n_{\mathrm{operator},m},\\
   \beta_m &= \frac{2}{\Delta w}\sin^{-1}\left(\frac{K_m\Delta w}{2}\right).

The inverse spatial difference is evaluated at each FFT bin after operator
index interpolation, including endpoint extrapolation. Single-frequency I/Q
sources, banks with only one retained anchor, and downstream solvers without
operator-index metadata retain the constant/legacy physical-index convention
:math:`\beta_m=2\pi f_m n_m/c`. In particular, a single-frequency source holds
its physical phase index constant across the pulse.

Significant source energy in the longitudinal grid stop band is an error:
refine the normal cell spacing or narrow the excitation bandwidth.
Sub-threshold tails in that stop band are discarded; the scalar waveform
reconstruction error includes their removal. DC and Nyquist bins are excluded
before this propagation calculation.

Linear interpolation of individually normalized modes does not in general
preserve unit power. gprMax constructs the cross-power matrix

.. math::

   P_{kl}=\frac{1}{2}\int
   \left(\mathbf{E}_k\times\mathbf{H}_l^*\right)
   \cdot\hat{\mathbf{w}}\,dA

(or the corresponding 2D line integral), and calculates

.. math::

   p_m=\operatorname{Re}
   \left\{\sum_{k,l}w_{k,m}P_{kl}w_{l,m}\right\},
   \qquad a_m=\frac{1}{\sqrt{p_m}}.

Thus the interpolated frequency-bin source mode is renormalized rather than
assuming that linear field weights retain one-watt power. Invalid or nearly
zero interpolated power is an error because a finite fallback would no longer
represent the requested one-watt incident wave.

Yee Time and Space Staggering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Electric and magnetic fields are offset by half a time step and their
tangential samples at the TF/SF surface are offset by half a normal cell. For
each bin, gprMax applies the magnetic staggering factor

.. math::

   M_m=\exp\left[j\left(
       \frac{\omega_m\Delta t}{2}
       +\frac{\beta_m\Delta w}{2}
   \right)\right].

Here :math:`\omega_m=2\pi f_m` is the physical angular frequency, and
:math:`\beta_m` is recovered from the longitudinal difference symbol. The
temporal difference symbol :math:`\Omega` is not a phase frequency.

The local coordinate is defined in the requested propagation direction. The
tangential H correction lies half a cell on the incident side of the electric
reference plane, giving the positive spatial phase in this relative factor.
For a lossy mode this local factor can have magnitude greater than one because
the incident-side sample precedes the reference plane; this does not represent
growth in the forward direction. Forward propagation over a positive distance
``d`` is always tested by ``exp(-j*beta*d)``, whose magnitude is below one for
a passive mode.

The spectral coefficients assigned to anchor ``k`` are therefore

.. math::

   C^E_{k,m} &= w_{k,m}S_m a_m,\\
   C^H_{k,m} &= w_{k,m}S_m a_m M_m.

Each anchor field is split into real and imaginary arrays, and each coefficient
set is inverse-transformed both normally and after multiplication by ``j``.
The FDTD source update then sums the two I/Q bases over all anchors.

DC and, for an even transform length, the Nyquist bin are self-conjugate. They
cannot carry a general complex modal coefficient while preserving a real time
record, so gprMax discards those two bins. Significant DC or Nyquist energy
produces a warning that the requested excitation has been changed. Use a
band-limited waveform; for a finite frequency band,
``EigenmodeExcitation(..., waveform='auto')`` synthesizes one automatically.

TF/SF Injection into the FDTD Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source plane is a total-field/scattered-field interface. In continuous
form, the incident mode corresponds to equivalent surface currents

.. math::

   \mathbf{J}_s &= \hat{\mathbf{n}}\times\mathbf{H}_{\mathrm{inc}},\\
   \mathbf{M}_s &= -\hat{\mathbf{n}}\times\mathbf{E}_{\mathrm{inc}}.

The discrete implementation applies the equivalent correction directly to
the curl updates adjacent to the plane:

* the magnetic-field update uses the incident tangential E profile;
* the electric-field update uses the incident tangential H profile;
* the side of the plane and correction sign depend on ``+`` or ``-`` source
  direction;
* only transverse/tangential components are injected. Longitudinal fields are
  reconstructed and retained for diagnostics but do not enter these TF/SF
  corrections.

Because modal fields are stored on their native Yee component grids, the
source does not resample every component onto a common rectangle. The update
kernels consume the component-specific array shapes documented earlier in
this page. This preserves PEC/PMC constraints, tangential staggering, and the
mode solver's discrete curl relationships.

For real-only sources, one real modal array per component is multiplied by the
waveform value. For I/Q and broadband sources, each update sums all anchor and
quadrature contributions prepared by the inverse FFT. In both cases, source
activation is clipped to the configured waveform start and stop times.

Single-frequency sources record ``single_frequency_iq_reasons`` and log the
individual selection reasons alongside the measured modal-profile residual:
``complex modal profile``, ``drive phase/delay``, and/or
``complex longitudinal staggering``. A small modal residual can therefore
coexist with I/Q injection. Negative real propagation is handled by the
signed real-only time shift and does not itself require I/Q. Eigenmode
solvers, source staggering, and monitor propagation use the active simulation's
``em_consts["c"]`` for the speed of light.

Virtual-guide aperture coupling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a virtual guide the TF/SF incident-mode correction is applied on an
internal plane of the auxiliary grid rather than the main grid. Let
:math:`w` be the port-normal coordinate and :math:`u,v` the transverse
coordinates. At the main/auxiliary split the normal magnetic sample is shared,

.. math::

   H_w^{\mathrm{aux}}\big|_{\Gamma}
   = H_w^{\mathrm{main}}\big|_{\Gamma},

and the aperture updates for :math:`E_u` and :math:`E_v` use the ordinary Yee
curl with the two normal-neighbour magnetic samples taken from different
grids. In schematic form,

.. math::

   E_u^{n+1}\big|_{\Gamma}
   &= C_A E_u^n\big|_{\Gamma}
      + C_v\,\Delta_v H_w
      - C_w\left(H_v^{\mathrm{aux}}-H_v^{\mathrm{main}}\right),\\
   E_v^{n+1}\big|_{\Gamma}
   &= C_A E_v^n\big|_{\Gamma}
      + C_w\left(H_u^{\mathrm{aux}}-H_u^{\mathrm{main}}\right)
      - C_u\,\Delta_u H_w.

The signs reverse consistently for the opposite port direction. The updated
tangential E samples are shared with the main-grid aperture, while the
duplicate main-grid continuation behind the aperture is disconnected. The
six axis/direction variants are implemented as compiled Cython kernels; no
per-cell Python work occurs during time stepping.

Modal Receivers, Direct DFT, and S-parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exactly one global band and zero or more distinct port/mode excitations may
exist whenever modal ports are used. Excitation may be omitted only when every
port is a passive virtual guide; that form writes raw modal spectra but no S
matrix. Every ``EigenmodePort`` owns one monitor at its reference plane; each
selected excitation additionally applies a TF/SF source. Port indices are
one-based and unique, and each port carries an explicit tuple of monitored
mode indices. Every excitation selects one of the modes listed by its port.
All ports accumulate the common DFT bins from ``EigenmodeBand``. Exactly one
active channel permits S-parameter normalization; a simultaneous driven state
retains its decomposed waves without constructing S.
Automatic ports share one candidate-frequency list, but tracking, retained
source/reference masks, and fallback policies are resolved independently for
each port and mode. Ports with explicit anchors retain their individually
requested candidate frequencies.
For each requested frequency :math:`f_q`, a Cython kernel applies the
recursive DFT

.. math::

   X_q^{(n+1)}=X_q^{(n)}
     +\Delta t\,x_n\exp(-j2\pi f_q n\Delta t).

The phase factor is advanced once per bin and time step. To bound complex64
recurrence drift in long simulations, both oscillators are reconstructed from
their physical times every 1024 iterations using float64 argument reduction;
the HDF5 ``PhaseReanchorInterval`` attribute records this interval. Electric
fields use integer FDTD times, while magnetic fields use their half-time-step
phase.
Transverse Yee components are averaged to common cells only inside the
projection. A passive receiver samples the magnetic reference plane half a
normal cell upstream of the electric plane. At a TF/SF source, H is sampled
half a cell downstream so that both fields are on the total-field side. The
final decomposition corrects either offset using each mode's propagation
constant.

The DFT-bin monitor basis is selected independently from source synthesis. A
power-wave-valid bin uses the interpolated one-watt propagating bank. A
generalized-only bin uses ``anchor_mode_reference_valid`` instead. It first
selects the applicable contiguous evanescent reference run inside the solved
candidate range; interpolation never crosses cutoff or spans disconnected
evanescent runs. Outside that range, it uses the nearest tracked reference
endpoint. Every selected anchor E/H pair is divided by the square root of its
``anchor_balanced_power``, then E, H, and the operator index are
interpolated with identical branch-local weights. The propagation constant is
mapped at each DFT frequency using the same discrete symbols as source
synthesis; branches retaining only one anchor keep their physical phase
index constant. The interpolated
cell-centred E/H pair is balanced once more before its Gram matrices are
formed. Keeping all three quantities on the same tracked branch is essential
below cutoff, where modal admittance becomes reactive; interpolating only the
propagation constant while retaining a propagating endpoint E/H pair would
not correctly separate forward and backward amplitudes at one plane.

A bin that enters the numerical spatial stop band after interpolation loses
power-wave validity. It retains its selected propagating-bank weights,
fields, and propagation constant, with final cell-centred balanced
normalization for generalized coefficients. It does not select a different
physical-cutoff branch. Existing conditioning and separation checks still
determine whether those coefficients can be reported.

The HDF5 port group preserves ``anchor_complex_neff`` as the physical phase
index. When available, ``anchor_operator_neff`` stores the corresponding
dimensionless operator indices with the same anchor/mode axes. ``beta``
records the actual propagation constant used at every DFT bin, in radians
per metre, with frequency/mode axes. These datasets also describe runs that
reuse cached modal anchors.

For several requested modes, independent overlaps are insufficient when the
discrete profiles are not exactly orthogonal. At each frequency gprMax forms
electric and magnetic Gram matrices,

.. math::

   G^E_{mn}
     &= \frac{1}{2}\int
        (\mathbf E_n\times\mathbf H_m^*)\cdot\hat{\mathbf w}\,dA,\\
   G^H_{mn}
     &= \frac{1}{2}\int
        (\mathbf E_m^*\times\mathbf H_n)\cdot\hat{\mathbf w}\,dA,

and solves both systems for the total electric and magnetic modal
coefficients. If :math:`x=a+b` is the electric coefficient and the staggered
magnetic coefficient is
:math:`y=p_+a-p_-b`, where :math:`p_+` and :math:`p_-` are the forward
and backward half-cell phase factors, then

.. math::

   a=\frac{y+p_-x}{p_++p_-},\qquad b=x-a.

Here :math:`a` travels in the receiver's declared direction and :math:`b`
travels in the opposite direction. With port directions defined into the
device, the single-source scattering result is

.. math::

   S_{j m,\,1 n}(f)=\frac{b_{j,m}(f)}{a_{1,n}(f)}.

Consequently the explicitly numbered source port gives S11 and a downstream
multimode port gives one S21 result for every requested destination mode.
These are generalized modal-amplitude ratios. ``reference_basis_valid`` records
only pre-solve reference eligibility. ``coefficient_valid`` then marks each
coefficient that survives both the electric and magnetic conditioned solves,
finite half-cell phase reconstruction, and finite-value checks.
``coefficient_valid_S`` additionally requires a usable source coefficient and
a -60 dB incident-spectrum floor evaluated separately for power-wave and
generalized-only source bins. Bins remain present in the arrays, but unusable
S entries are NaN. ``power_wave_valid`` further requires destination-mode
power-wave support and a valid destination power matrix. ``power_wave_valid_S`` includes
those destination gates and additionally requires the launched source mode and
its power matrix to be physically valid.

For each electric or magnetic Gram matrix, a singular value is retained only
when

.. math::

   \sigma_i > \max\left(
      \frac{\epsilon}{10^{-3}},
      \frac{\sigma_{\max}}{\kappa_{\mathrm{lim}}}
   \right),
   \qquad
   \kappa_{\mathrm{lim}}=\min\left(10^{10},\frac{10^{-3}}{\epsilon}\right),

where :math:`\epsilon` is the precision of the stored field components. If all
singular values pass, gprMax solves the complete system directly. If not, a
truncated full-system SVD fallback is considered only when the active set
contains both power-wave and generalized-only coordinates. The discarded
right-singular subspace must have Frobenius projection no larger than
:math:`10^{-3}` onto the power-wave coordinates; otherwise the complete solve
is rejected. Among accepted fallbacks, only coordinates whose ambiguity in
the discarded subspace is no larger than :math:`10^{-3}` in *both* the
electric and magnetic solves become ``coefficient_valid``. This can preserve a
power-wave coordinate when only a generalized-only coordinate is singular,
but it rejects a nullspace that mixes physical modes.

``condition_number`` reports the larger electric/magnetic full-system
condition number for a direct solve. For a successful truncated fallback it
instead reports the larger condition number of the two retained singular
subspaces, not the singular original system; it is infinite when no
coefficient survives. The small Gram systems and SVD are evaluated in
complex128 even when the stored FDTD fields and Gram entries use complex64.

A below-cutoff mode can therefore produce finite incident/outgoing
coefficients and a continuous S21 amplitude. In a uniform guide its forward
amplitude varies as :math:`\exp(-\alpha L)` for
:math:`\beta=-j\alpha`; this is the attenuation of a field coefficient, not
the transport of real power by an isolated evanescent wave. At exact cutoff,
:math:`\beta=0` and the true forward/backward eigenmode basis coalesces. An
E/H-balanced reference may approach a finite limiting coefficient there, but
the directional decomposition is not unique and must be treated as
conditioning-sensitive.

The same Gram matrices define the Hermitian forward-wave power form

.. math::

   W=\frac{1}{2}\left(G^E+G^H\right),\qquad
   P(c)=\operatorname{Re}\{c^\mathrm{H}Wc\}.

The implementation symmetrizes :math:`W` against round-off and checks that it
is finite and positive semidefinite. Keeping the off-diagonal terms is
essential for degenerate, nearly degenerate, or merely non-orthogonal
finite-grid profiles. Individual coefficient magnitudes therefore are not
additive modal powers. A pure evanescent mode has zero independent real power,
so ``coefficient_valid_S`` may be true while
``power_basis_valid`` and ``power_wave_valid_S`` are
false. In particular, :math:`|S|^2` must not be used as an evanescent power
fraction.

For net accepted power, let :math:`x=a+b` be the total electric coefficient
and :math:`y=a-b` the co-located total magnetic coefficient after the
half-cell correction. The direct time-average flux is

.. math::

   P_{\mathrm{acc},p}
     =\operatorname{Re}\{y_p^\mathrm{H}G^E_p x_p\}.

This reduces to :math:`P(a_p)-P(b_p)` when :math:`G^E_p` is Hermitian.
For a lossy port, its anti-Hermitian part supplies a forward/backward
interference term which must be retained.

At each frequency this quadratic form is evaluated on the valid propagating
mode subspace. All off-diagonal terms within that subspace are retained, but
generalized-only rows and columns are excluded. An invalid propagating mode
invalidates the accepted-power result rather than being silently omitted.

For active port :math:`p`, let :math:`D_p` select its explicitly driven modes.
The externally driven incident power is

.. math::

   P_{\mathrm{inc}}
     = \sum_{p\,\mathrm{active}}
       \operatorname{Re}\!\left\{
       a_{p,D_p}^{\mathrm{H}}W_{p,D_pD_p}a_{p,D_p}\right\}.

Thus a one-channel run retains the previous single-mode definition, while a
simultaneous multimode run keeps the cross terms between non-orthogonal driven
modes on the same physical port. Passive modal receivers have zero generator
incident power, but their signed accepted power remains in the multiport
balance used for gain. This distinction makes realized gain use launched
source power while gain uses the net power accepted by the radiating
structure. The power adapter applies the power-normalization and power-matrix
masks, so generalized below-cutoff coefficients do not enter gain,
accepted-power, or energy-balance normalization.

Understanding Lossy-Mode Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a passive lossy mode, users should expect:

* positive ``Re(n_eff)`` for the selected forward phase branch;
* negative ``Im(n_eff)``;
* positive modal power in the requested source direction;
* a forward magnitude factor
  ``abs(exp(-j*k0*n_eff*d)) < 1`` for ``d > 0``;
* a nonzero complex-profile residual in general, causing I/Q injection;
* downstream attenuation to be produced by the FDTD material updates after
  the mode has been launched.

The port-mode field plots show tangential E and H vectors for every retained
anchor and report the complex effective index. E and H magnitudes are
normalised independently in their panels. A nonzero receiver field alone does
not validate a lossy mode: sign-sensitive validation must also check
``Im(n_eff)``, modal power direction, and forward attenuation.

Accuracy Guidance and Warnings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For reliable broadband excitation:

* place frequency anchors across the significant waveform spectrum;
* add anchors near rapid dispersion, cut-off, avoided crossings, or strong
  profile changes;
* treat a below-cutoff balanced-reference S trace as a generalized amplitude
  and use ``power_wave_valid_S`` before making any power claim;
* move the DFT grid and repeat the solve when a requested bin lies at or very
  close to cutoff, where the true forward/backward basis is ill-conditioned;
* inspect adjacent-anchor overlaps below 0.9 rather than assuming a fixed mode
  index tracks the same physical branch, and use a single-frequency solve when
  an overlap is below the hard 0.6 limit;
* use a band-limited waveform when I/Q synthesis is required; significant DC
  and Nyquist bins produce a warning and are discarded, while ``waveform='auto'``
  synthesizes a suitable waveform for a finite frequency band;
* refine the FDTD grid until both the modal effective index and field profile
  converge;
* keep the source plane in a longitudinally uniform section of the guide so
  the transverse eigenproblem represents the structure being launched;
* compare forward and backward modal power or receiver phase when reflections
  and mode purity matter.

Eigenmode injection and modal monitoring run on the CPU, CUDA, OpenCL, and
Metal solvers. Domain-decomposed MPI CPU models are also supported: the modal
material plane is assembled collectively during model construction, each rank
applies only its owned TF/SF corrections, and the small distributed modal DFT
arrays are reduced once after time stepping. Material dispersion is sampled at
each anchor frequency, but interpolation between anchors remains piecewise
linear; additional anchors are the normal way to resolve stronger frequency
dependence.

Complete matrices, active reflection, and antenna normalization
---------------------------------------------------------------

For independent cases, assemble the measured incident and outgoing columns
into :math:`A(f)` and :math:`B(f)`. The complete matrix satisfies

.. math::

   B = S A.

Solving this system accounts for measured incident waves at nominally passive
channels. A missing or ill-conditioned incident basis invalidates the affected
matrix bins. For reciprocal linear, time-invariant materials and consistent
modal normalization, phase conventions, and reference planes,
:math:`S_{ij}=S_{ji}`. Reciprocity alone does not require
:math:`S_{11}=S_{22}`; that equality in example 4 additionally follows from
the device's symmetry.

A simultaneous drive q has spectral multiplier

.. math::

   w_q(f) = A_q\exp(j\phi_q)\exp(-j2\pi f\tau_q).

The API takes :math:`\phi_q` in degrees and :math:`\tau_q` in seconds.
For a coherent incident vector :math:`a`, active reflection is

.. math::

   \Gamma_{\mathrm{active},q} = \frac{b_q}{a_q}
       = \frac{\sum_j S_{qj}a_j}{a_q}.

Its coefficient and power-wave masks also require an adequately excited
denominator. It depends on the incident vector, so it cannot replace the
independent cases used to determine :math:`S`.

For the array along y in example 5, spacing d and constant progressive phase
:math:`\Delta\phi` give the ideal forward xy-plane maximum

.. math::

   \sin\phi_{\mathrm{peak}}(f)
       = -\frac{\Delta\phi}{k(f)d},\qquad k(f)=\frac{2\pi f}{c}.

The phase increment is in radians in this equation. Positive observation
angle is measured from +x toward +y. For d = 18 mm and an increment of -108
degrees, the predicted angles at 8, 9, 10, 11, and 12 GHz are approximately
38.7, 33.7, 30.0, 27.0, and 24.6 degrees. These ideal array-factor angles do
not include the aperture element pattern or mutual coupling.

For radiation intensity U, total radiated power :math:`P_{\mathrm{rad}}`,
accepted power :math:`P_{\mathrm{acc}}`, and externally driven incident power
:math:`P_{\mathrm{inc}}`, the antenna quantities are

.. math::

   D = \frac{4\pi U}{P_{\mathrm{rad}}},\qquad
   G = \frac{4\pi U}{P_{\mathrm{acc}}},\qquad
   G_{\mathrm{realized}} = \frac{4\pi U}{P_{\mathrm{inc}}}.

The power adapter uses the physical power-wave subspace and its full power
matrix. Finite below-cutoff coefficients do not enter gain or energy-balance
normalization. The ``port_power/modal_ports`` output repeats the modal waves,
mode indices, power matrix, and physical masks at the NTFF frequencies so
that the normalization can be audited.
