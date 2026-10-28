.. _eigenmode-tracking:
.. _eigenmode-mode-tracking:

************************
Eigenmode tracking guide
************************

Mode tracking connects independently solved port fields into a consistent
broadband field bank. It follows branches through crossings, aligns phase,
and selects reproducible bases for degenerate groups. This guide covers the
inputs, workflow, automatic symmetry selection, and inspection of the resulting fields.
For aperture setup and S-parameter measurement, see :doc:`eigenmode_port`;
for equations and implementation details, see :doc:`eigenmode_tracking_theory`.

.. contents:: On this page
   :local:
   :depth: 2

Enabling tracking and choosing what to inspect
==============================================

Add ``tracking="auto"`` to each participating source and receiver port.
This also enables symmetry detection and basis selection for eligible
degenerate groups on supported 3D port cross sections. Explicit
``mode_polarizations`` takes precedence when selecting a pair's directions.
Start with the default ``verification="full"`` when evaluating a model.
``verification="fast"`` skips the extra mesh and window solves, so it provides
less evidence about box modes. Examples 8 and 9 use ``fast`` to keep their
inspection runs short; change it to ``full`` to perform those extra checks.

.. code-block:: python

   scene.add(gprMax.EigenmodePort(
       port=1, p1=(0.02, 0.005, 0), p2=(0.02, 0.075, float("inf")),
       direction="+", modes=(1, 2), anchors="auto",
       tracking="auto", verification="full", plot_fields=True,
   ))

``anchors="auto"`` and ``tracking="auto"`` do different jobs: **anchors choose
where to solve; tracking chooses which solutions to connect**. You can also
provide an explicit list of anchor frequencies, as Examples 8 and 9 do.

Mode labels are assigned at the anchor nearest the band center (the lower
frequency on a tie), then followed toward both ends of the band. Thus
``mode=1`` means a tracked channel, not necessarily the first raw solution at
every frequency. When comparing with an older run, check the actual fields
at both ports rather than assuming that the same number means the same
polarization.

Run Examples 8 and 9 with ``--geometry-only`` to inspect their dispersion
curves and modal fields before the time-domain simulation. Their scripts set
``plot_fields=True``, so a full run also produces these plots. At a crossing,
check that the labels follow consistent fields. For TE11, check that the chosen
polarizations stay aligned. For an open guide, check the outer-edge fields and
any confinement warnings. Read the warning status as well as the curve: a
smooth curve can still describe a questionable profile.

Tracking does not guarantee low S11. Correctly matched fields can still be
interpolated too coarsely, especially near cutoff. Check convergence with
closer anchor spacing, a suitable aperture, mesh resolution, PML thickness,
and recording duration when small reflections matter.

Tracking input arguments
========================

Pass these arguments to ``gprMax.EigenmodePort``. The aperture, propagation
direction, and port identifier are configured as in :doc:`eigenmode_port`.
The first table covers channel selection and basis construction; the next
table lists every ``EigenmodeTrackingConfig`` control.

.. list-table:: Port arguments used by tracking
   :header-rows: 1
   :widths: 23 18 59
   :class: api-parameters

   * - Argument
     - Default
     - Use
   * - ``modes``
     - Required
     - Public channel indices, or a positive integer count. With automatic tracking these labels refer to the anchor nearest the band center, then follow the matched fields. Internal partners do not add public channels.
   * - ``anchors``
     - ``"auto"``
     - Frequencies in Hz at which to solve the port. Accepts a positive frequency, an increasing sequence, or automatic selection. Tracking may add midpoint solves to resolve ambiguous associations.
   * - ``tracking``
     - ``"legacy"``
     - ``"auto"`` enables field assignment, group detection, automatic symmetry selection, and adaptive refinement. ``"legacy"`` retains ordering and manual grouping and skips symmetry selection. Automatic tracking is experimental.
   * - ``verification``
     - ``"full"``
     - With automatic tracking, ``"full"`` adds finer-grid and open-window comparisons; ``"fast"`` checks residuals and edge fields without those extra solves.
   * - ``tracking_config``
     - ``None``
     - An ``EigenmodeTrackingConfig`` instance or mapping overriding the numerical controls in the next table. Omitted fields keep their defaults.
   * - ``mode_polarizations``
     - ``None``
     - Optional single transverse axis or real three-vector. Its orthogonal partner is inferred using the positive port normal. Applied after subgroup resolution with automatic tracking, or to declared pairs with legacy tracking. Shared direction pairs in either tracker and legacy mode/group mappings are deprecated and will be removed in gprMax 5.0.
   * - ``degenerate``
     - ``None``
     - Legacy groups such as ``(1, 2)`` or ``((1, 2), (3, 4))``. Members must be requested in ``modes``. Automatic tracking detects groups and ignores this argument.
   * - ``plot_fields``
     - ``None``
     - ``True`` writes dispersion and modal-field plots in all runs; ``False`` suppresses them; ``None`` writes them in geometry-only runs.

Symmetry detection and basis selection run automatically with ``tracking="auto"``
on supported 3D port cross sections. Legacy tracking skips symmetry selection.
Polarization vectors use the Python API. Keep the same basis convention
at both ports when comparing their channel coefficients.

Advanced thresholds and limits
------------------------------

Most users can leave these settings at their defaults. Inspect the fields and
geometry before relaxing a threshold to suppress a warning. Pass
``tracking_config=gprMax.EigenmodeTrackingConfig(...)`` or a mapping to
override the defaults below. Omitted fields keep their defaults. All thresholds
are dimensionless; integer controls count candidates, refinement steps, or
additional solves. These controls apply only to automatic tracking.
Thresholds must be finite and in ``(0, 1]``; ``extra_candidates`` and
``max_depth`` are nonnegative integers, and ``max_solves`` is a positive integer.

.. list-table:: EigenmodeTrackingConfig
   :class: api-parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Meaning
   * - ``residual_tolerance``
     - ``1e-7``
     - Common maximum relative residual for input eigenpairs, mixed reduced vectors, and available reconstructed-field checks. Eigenvalue-gap, propagation, and power checks also apply.
   * - ``beta_drift_tolerance``
     - ``1e-3``
     - Maximum normalized propagation-constant drift in mesh/window verification.
   * - ``verification_overlap``
     - ``0.999``
     - Minimum squared field overlap, or smallest squared principal subspace overlap, in verification comparisons.
   * - ``edge_fraction_max``
     - ``1e-3``
     - Maximum fraction of the squared E/H field norm in the outer edge strip (10% of each transverse extent, at least one cell) used as confinement evidence at artificial boundaries.
   * - ``tracking_overlap``
     - ``0.8``
     - Minimum squared field/subspace overlap for a candidate match. This is distinct from the stricter verification overlap.
   * - ``assignment_margin``
     - ``0.02``
     - Minimum total-cost advantage over an alternative one-to-one assignment for an individual-mode match.
   * - ``subspace_margin``
     - ``0.02``
     - Minimum separation of the squared projection scores of the last selected and first excluded members when extracting a subgroup from a larger cluster.
   * - ``unmatched_cost``
     - ``0.65``
     - Cost of leaving an individual mode unmatched rather than forcing a poor match.
   * - ``cluster_gap``
     - ``1e-5``
     - Relative eigenvalue gap for candidate clusters. It does not relax the stricter ``1e-8`` degeneracy condition for mixing fields.
   * - ``extra_candidates``
     - ``4``
     - Additional internal eigenpairs requested per solve, subject to solver size, to find competing branches or missing partners.
   * - ``max_depth``
     - ``8``
     - Maximum adaptive midpoint-refinement steps for resolving identity ambiguity; ``0`` disables these extra frequency solves.
   * - ``min_relative_step``
     - ``1e-5``
     - Stop refining an ambiguous interval when its width divided by its midpoint frequency reaches this threshold.
   * - ``max_solves``
     - ``200``
     - Budget for additional adaptive and verification solves per port. Required primary anchors are retained independently of this budget.

For example, allow more internal candidates while retaining all other defaults:

.. code-block:: python

   tracking_config = gprMax.EigenmodeTrackingConfig(extra_candidates=6)
   # Pass tracking="auto", tracking_config=tracking_config to EigenmodePort.

Compared with released 4.0.1, the general residual default increases from
``1e-8`` to ``1e-7``. Mixed-mode acceptance now uses this setting instead of
the previous fixed ``1e-9`` limit, including legacy degenerate groups. Set
``residual_tolerance=1e-9`` to retain that mixed-mode limit; this also tightens
the other residual checks. See :doc:`eigenmode_tracking_theory` for the
additional reconstructed Maxwell-residual checks.

The equivalent hash options use the field names directly, without a
``tracking_config`` prefix:

.. code-block:: none

   #eigenmode_port: 1 0.02 0.005 0 0.02 0.075 inf + 1,2 auto y tracking=auto verification=full extra_candidates=6

How automatic tracking works
============================

An **anchor** is a frequency at which the port eigenproblem is solved. A
**candidate** is one of that solve's eigenmodes. A **branch** is the sequence
of fields associated across anchors. A **degenerate group** contains independent
fields with the same propagation constant; its individual eigenvectors can
rotate while their combined span stays unchanged.

.. figure:: ../../images_shared/mode_tracking_workflow.svg
   :width: 100%
   :alt: Tracking workflow from independent solves through association and basis selection to a checked field bank.

   Association establishes continuity; basis selection supplies reproducible
   E/H fields for interpolation. Extra candidates and midpoint solves help
   resolve ambiguous matches.

1. **Solve and seed labels.** Solve the primary anchors with extra internal
   candidates. Assign public labels at the anchor nearest the band center,
   and work outward toward lower and higher frequencies.
2. **Associate fields.** For isolated modes, combine E/H overlap and eigenvalue
   prediction in a global one-to-one assignment. Degenerate candidates are
   recognized and reserved by span matching before this individual assignment.
   When a group encounters another branch or group, projection and continuity
   tests identify the persistent subgroups inside the enlarged cluster.
3. **Refine ambiguous intervals.** Insert midpoint anchors when overlap or
   assignment margins cannot resolve a match. Refinement stops at the configured
   depth, frequency spacing, or solve budget; unresolved intervals retain their
   diagnostic status.
4. **Construct a reproducible basis.** Align each isolated mode's common E/H
   phase. Within a degenerate group, first compare longitudinal electric and
   magnetic content, then directional and total field variation (Ku-Kv and
   Ku+Kv). Use symmetry afterward to refine compatible groups. Keep any
   successful separation if symmetry cannot be certified, and smoothly align
   only the blocks that remain unresolved. Apply a requested polarization
   only to eligible C4v E pairs. Every proposed change must pass the physical
   validity checks. The :ref:`step-by-step explanation
   <tracking-separation-explained>` below describes these decisions without
   requiring group theory or matrix algebra.
5. **Check and use the bank.** Record identity, residual, propagation, power,
   and boundary diagnostics separately. Eligible propagating fields supply
   excitation. Receiving references can also describe evanescent fields, with
   finite normalization and valid coefficient decomposition.

**Power whitening** normalizes a group's field combinations so their
coefficients have a consistent power scale. It provides a stable starting
basis even when the mode orientations are unresolved. After separation,
normalization stays within each retained block, preserving the families
already found. Power calculations still include any interference between
blocks. The matrix formulation is in :ref:`eigenmode-degenerate-theory`.

The candidate-cluster gap and the mixing tolerance serve different purposes.
``cluster_gap`` searches for nearby branches; the stricter eigenvalue-spread
and residual checks decide whether their fields may be combined. Increasing
``cluster_gap`` alone does not enable approximate mixing of split modes.

Why track modes?
================

A broadband pulse contains many frequencies. To launch the intended mode
across that band, the port needs to know what its field looks like at each
frequency. It solves the fields at a set of frequencies called **anchors**,
then interpolates between them.

That interpolation only makes sense if the anchor fields belong together.
Blending an x-polarized mode at one frequency with an unrelated y-polarized
mode at the next creates the wrong broadband source, even if both individual
mode solves are accurate. It can also make the receiving port measure the
wrong channel. **Mode tracking decides which fields belong to the same mode
as frequency changes.** Its verification checks also warn about profiles that
may be distorted by the finite port aperture.

When mode numbers swap: a crossing in an anisotropic guide
----------------------------------------------------------

An anisotropic dielectric can affect x- and y-polarized fields differently.
Their propagation constants can therefore cross as frequency changes, as in
:ref:`eigenmode-example-9`. The fields remain distinguishable, but their order
in the solver's list changes. For example:

.. list-table:: Why the raw solution number is not a reliable mode identity
   :class: api-parameters
   :header-rows: 1

   * - Frequency
     - First solution in the sorted list
     - Second solution in the sorted list
   * - Below a crossing
     - Mostly x-polarized
     - Mostly y-polarized
   * - Above the crossing
     - Mostly y-polarized
     - Mostly x-polarized

If the source simply took the first solution at every anchor, its polarization
would switch at the crossing. Interpolating those fields would blend two
different channels. A receiver using the same mistaken numbering could then
report an apparent change in transmission or coupling between modes.

Automatic tracking compares the **field patterns**, as well as how their
propagation constants change. In this example it connects the x-like field
below the crossing to the x-like field above it, even when its raw solution
number changes. The selected channel follows one curve *through* the crossing
instead of jumping to the other curve. Distinct modes retain their own
propagation constants; tracking does not average them together.

In the plots, follow one label across neighbouring frequencies: its field
should change consistently. A crossing in the dispersion plot is not itself
an error.

An **avoided crossing** looks different: two propagation curves approach
each other, turn apart, and leave a gap. As you follow either curve, its
field pattern can gradually change from resembling one mode family to
resembling the other. The changing pattern can be the correct solution of
the waveguide equations. Tracking follows that valid branch through the
change; forcing it to keep its original appearance can produce an invalid
field.

For example, a smooth circular dielectric guide can have an exact crossing
that becomes an avoided crossing when its boundary is represented by a
Cartesian staircase. The :ref:`TE03/HE42 validation example
<tracking-circular-avoided-crossing>` shows this effect and explains why
mixing the numerical branches back into pure analytical patterns fails the
field-equation checks. This differs from an arbitrary rotation of an exactly
degenerate pair, which can be corrected without changing the physical modes.

When polarization rotates arbitrarily: the circular TE11 pair
-------------------------------------------------------------

A circular waveguide has two TE11 modes with the same propagation constant.
One can be chosen x-polarized and the other y-polarized. Because they are
**degenerate** (they share that propagation constant), rotated combinations
are equally valid solutions.

An eigensolver might return x/y fields at one anchor and two diagonal fields
at the next. Nothing in the circular guide has rotated: the solver has simply
chosen different representatives of the same pair. Interpolating the first
returned field at each frequency would make the source polarization vary
across the pulse's band. Measured coupling into the other polarization could
then reflect this choice of fields rather than a physical effect.

Automatic tracking recognizes the pair and aligns it **as a pair**. In
:ref:`eigenmode-example-8`, it chooses x/y directions consistently at the
anchors, so a selected input polarization keeps its meaning across the band.
The pair does not need to be declared with ``degenerate``, and
``mode_polarizations`` is optional. Supply ``"y"`` when you want y/-x
directions instead of x/y, and use the same choice at both ports.

Even if you request only ``modes=(1,)``, the tracker can solve the second member
internally to align the pair. That extra solve does not add another public
input or output channel. Alignment is only allowed when the modes really can
be mixed without breaking their field equations. Two nearby modes with
different, resolved propagation constants stay separate.

When the mode belongs to the artificial box
-------------------------------------------

An open guide, such as a coplanar waveguide, has fields extending into the
surrounding air. The mode solver can only see the finite cross-section inside
the port aperture. Its artificial outer boundary can influence the fields
or support a pattern that mainly belongs to that calculation box. This is a
**box mode**; it is not the same as a legitimate mode confined by the physical
metal walls of a waveguide.

A box mode can be solved very accurately and still be a poor representation
of the intended guide. If used for injection or measurement, it can produce
results that depend strongly on the chosen aperture. Following that pattern
consistently across frequency would not, by itself, make it a good profile.

With ``verification="full"``, automatic tracking asks practical questions:

* Is much of the field still present near the artificial outer boundary?
* If the window is enlarged using the surrounding model geometry, does the
  field pattern or propagation constant change appreciably?
* Does a finer transverse grid give a consistent result?

These checks can flag a **suspected** box mode or poorly confined profile.
Mesh sensitivity alone does not prove that a mode is a box artifact. Nor can
the checks always distinguish an artificial mode from a real guided mode
whose field has simply been cut off by a small aperture.

For example, a low-frequency CPW mode can have a wide field extending far into
the air. A small aperture may truncate it and trigger a warning, even though
it is the intended mode. **A confinement or box-mode warning does not remove
the profile:** it remains available if its numerical solution, tracking, and
forward power are valid. The warning says that accuracy may be reduced.
Enlarge the aperture, leave enough surrounding non-PML geometry for the
checks, and compare the fields and S-parameters. Tracking helps identify the
problem; it cannot supply the missing space around the guide.


Degeneracy and polarization choices
===================================

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
  not need mode numbers to mean particular physical directions with legacy
  tracking. Its initial basis is arbitrary; subsequent anchors follow it smoothly.
* Also use ``mode_polarizations="y"`` when you want mode 1
  to mean vertical and mode 2 horizontal for a z-directed guide. gprMax
  enforces those physical directions independently at each retained anchor.
  Use the same assignments at the receiving port to measure the same labels.
* Use ``tracking="auto"`` to detect groups, separate their field families,
  and use symmetry to refine and orient compatible modes. The fundamental circular
  pair selects transverse directions as in :ref:`eigenmode-example-8`.
  ``degenerate`` is ignored in this mode; ``mode_polarizations`` remains useful
  for rotating or swapping the default directions.

Declaring the pair neither excites both channels nor forces their physical
propagation constants to become equal. ``EigenmodeExcitation`` still selects
the channel to launch. A declared pair contributes two channels to a complete
S-matrix study, so each needs its own independent excitation case.

Near equality alone is not enough to justify mixing. For example, an
elliptical guide can split the circular pair into modes with different
propagation constants. Those modes accumulate different phases, and a
combination can change polarization as it travels. That is physical beating,
which should be represented using independent modes. gprMax accepts exact
or numerically unresolved degeneracy for mixing. It rejects resolved splitting
in a declared group and keeps split automatic branches separate; the
thresholds and alignment method are in :ref:`eigenmode-degenerate-theory`.

.. _tracking-separation-explained:

How automatic tracking separates a degenerate group
---------------------------------------------------

At a degeneracy, the solver may return mixtures of several underlying
patterns. Their equal propagation constants give no way to tell those
mixtures apart. The tracker looks for distinguishable patterns within
their combined space. It takes new linear combinations of the returned
fields, applying the same combination to all electric and magnetic
components. This is a change of basis.

Each separation step divides a group where its field comparison can
distinguish the members. A group of one is a resolved channel; a larger
group, called a **block**, needs another criterion to choose its individual
patterns. Symmetry can also bring compatible blocks together to check how
they transform as a group. The method works from the computed fields,
without a preset list of TE, TM, HE, or EH mode shapes.

1. Separate longitudinal field families: TE/TM first. Longitudinal means
   along the guide, while transverse means across its cross-section. A TE
   mode has no longitudinal electric field; a TM mode has no longitudinal
   magnetic field. The tracker finds combinations with different balances
   of these two fields. This can recover a TE family and a TM family from
   mixtures returned at a degeneracy. Dielectric modes usually contain both
   components, so distinct balances can separate hybrid families without
   automatically assigning them HE or EH names. Equal balances leave a
   block for the next step. If the longitudinal fields are too weak or
   unreliable to compare, this entire longitudinal/gradient stage keeps
   the existing block for symmetry testing.

2. Compare variation in the two transverse directions: Ku-Kv. Call the
   two cross-section directions u and v; for a z-directed port they are x
   and y. Ku measures how strongly the longitudinal fields vary along u,
   and Kv does the same along v. Their difference distinguishes patterns
   that vary mainly in one direction from patterns that vary mainly in the
   other. For example, square-guide mixtures of TE20 and TE02 can be
   separated into their individual axis-oriented patterns. This step can
   work even when the port has no certified spatial symmetry.

3. Compare total variation: Ku+Kv. Two patterns can have the same
   directional balance but different amounts of spatial variation. Within
   a block still tied after Ku-Kv, the sum supplies an additional distinction.
   A smoother pattern can have a lower total score than one that varies
   more across the cross-section. This can help separate radial patterns,
   but it is not a radial mode number: different modes can share both
   gradient scores. A tied score keeps the block intact.

4. Use symmetry as an independent check and further separator. The
   tracker now tests rotations and reflections of the sampled geometry,
   materials, boundaries, and fields. These operations can distinguish
   patterns that the preceding comparisons leave together. For example,
   an ideal circular TE21 angular pair can share all three scores but
   have different reflection symmetries. A quarter turn can also exchange
   two partners and reverse the sign of Ku-Kv. The tracker therefore
   considers blocks in the same longitudinal family with matching absolute
   directional scores and total-variation scores together for certification.
   Matching scores identify candidates; the rotation/reflection tests must
   still confirm their symmetry. A failed symmetry trial keeps the already
   validated field separation.

5. Continue any remaining unresolved block smoothly. If the available
   criteria still cannot distinguish individual patterns, the tracker keeps
   the whole block and chooses combinations that best match the neighboring
   frequency anchor. This is called **Procrustes alignment**. It stabilizes
   interpolation within that block without mixing it with already separated
   families. An isolated channel needs only a common E/H phase adjustment.
   Smooth continuation of a block does not establish unique physical names
   for its individual members.

6. Apply the user's polarization choice to an eligible pair. A certified
   **C4v E pair** is a two-channel space whose members transform together
   under the square grid's rotations and reflections. The tracker also
   requires usable integrated transverse electric fields, called electric
   moments: the two moments must be strong enough and point in independent
   directions. Only then can ``mode_polarizations`` set a reliable electric
   direction. Higher-order pairs can have electric fields that cancel on
   integration; they retain their automatic orientation, often selected by
   reflection symmetry. An unsupported explicit request raises an error.

For example, suppose a four-mode degeneracy contains two TE and two TM
patterns. The first step can divide it into a TE block and a TM block.
Directional or total variation may then separate members of either block;
symmetry can identify any remaining compatible singlets or pairs. If a
two-mode block is still unresolved, only that block receives the final
smooth alignment. The tracker does not recombine all four modes.

Every accepted transformation must preserve a valid solution: the field
equations, propagation behavior, power, and receiving-port decomposition
must still pass their checks. A rejected trial leaves the preceding validated
basis in place. Near an avoided crossing, this means retaining the actual
numerical eigenmodes even when their resemblance to analytical patterns
changes. The criteria resolve the tested mode families, but they do not
guarantee a unique decomposition of every possible degenerate space.

For the equations and acceptance tolerances, see :ref:`longitudinal and
gradient separation <tracking-independent-separation-theory>`,
:ref:`symmetry certification <tracking-symmetry-theory>`,
:ref:`unresolved-block continuation <tracking-unresolved-transport-theory>`,
and :ref:`electric-moment orientation <tracking-moment-orientation-theory>`.

Physically labeled degenerate modes
-----------------------------------

The default ``mode_polarizations=None`` uses automatic basis selection.
Supply one transverse axis (``"x"``, ``"y"``, or ``"z"``) or one real
three-vector such as ``(1, 1, 0)`` to orient every eligible doublet. The first
channel follows that direction; the second follows the positive port-normal
axis crossed with it. For a z-normal port, ``"y"`` gives y/-x. This convention
is independent of propagation sign. Use the same setting at both ports.

With ``tracking="auto"``, the complete parent group is tracked and subdivided
by longitudinal content, then directional and total gradient variation before
spatial symmetry is detected. Unresolved blocks remain subspaces. Symmetry is
certified independently and may refine these blocks; failure retains the
validated separation. After symmetry, remaining unresolved blocks use Procrustes
transport across frequency. This final step preserves each block's span and
cannot mix it with resolved modes; resolved singlets receive only phase alignment.
The requested direction is applied only inside resolved,
certified C4v E pairs with significant, well-conditioned rank-two integrated
electric moments. Singlets and uncertified two-mode parents are not eligible.
This is the supported scope of automatic polarization, not a general prohibition
on superpositions in other degenerate subspaces.

With ``tracking="legacy"``, the direction applies to the declared two-mode
pairs, without automatic symmetry selection. In either mode, a request with
no eligible pair or a pair with zero or ill-conditioned integrated transverse
E raises an error. Higher-order doublets with vanishing moments retain their
automatic mirror-based orientation when the option is omitted.

First and second refer to the lower and higher tracked indices inside the
resolved pair, including any internally retained partner. The direction must
be finite, nonzero, real, and transverse. It is normalized; a normalized normal
component above ``1e-12`` is rejected. The inferred partner is orthogonal.

.. deprecated:: Unreleased

   Two-direction sequences remain supported with both ``tracking="legacy"``
   and ``tracking="auto"``, preserving both requested directions and signs,
   including valid nonorthogonal pairs. Per-mode/per-group mappings require
   legacy tracking. These compatibility forms emit a visible ``FutureWarning``
   and will be removed in the next major release, **gprMax 5.0**.

Prefer one global direction for every pair: it keeps channel orientation
consistent and avoids separate mode/group settings. The second direction is
inferred as the positive port normal crossed with the first direction.
For example, migrate ``mode_polarizations=("y", "x")`` or
``mode_polarizations={1: "y", 2: "x"}`` to ``mode_polarizations="y"`` for a
z-normal port. The inferred partner is **-x**, so this migration changes the
second channel's phase by 180 degrees relative to an explicit +x reference.
Use the same convention at both ports when comparing complex S-parameters.

With legacy tracking, the compatibility forms require declared two-mode
``degenerate`` pairs. With automatic tracking, a shared direction pair is
applied inside each eligible doublet after automatic subgroup selection:

.. code-block:: python

   # Deprecated compatibility forms: each emits a warning.
   mode_polarizations=("y", "x")       # Shared explicit pair.
   mode_polarizations={1: "y", 2: "x"} # Directions for both members.
   mode_polarizations={(1, 2): "y"}    # One direction for this declared pair.

   # Preferred form for legacy and automatic tracking.
   mode_polarizations="y"             # Shared y/-x pair for a z-normal port.

In hash input, the corresponding deprecated values are
``mode_polarizations=y;x``, ``mode_polarizations=1:y;2:x``, and
``mode_polarizations=1,2:y``. Replace them with ``mode_polarizations=y``.
Explicit direction pairs must be transverse and independent, with a
condition number no greater than 10. Nonorthogonal pairs cannot be represented
by one direction; retain the explicit pair through the deprecation period
when that basis is required. Legacy mappings may orient selected declared
pairs while leaving the others to generic subspace alignment; incomplete or
overlapping mappings raise an error.

For circular TE11 propagating along z, assign the channels once:

.. code-block:: python

   scene.add(gprMax.EigenmodePort(
       port=1, p1=(0, 0, 0.02), p2=(0.05, 0.05, 0.02), direction="+",
       modes=(1, 2), anchors="auto", tracking="auto",
       mode_polarizations="y",  # The partner is automatically -x.
   ))
   scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="auto"))

Here mode 1 is vertically polarized (global y); changing only ``mode=1`` to
``mode=2`` launches horizontal polarization (global -x). Reversing propagation
does not change these electric-field labels. ``anchors="auto"`` selects the
frequency anchors; it does not enable automatic mode tracking. Set
``tracking="auto"`` explicitly to discover pairs without ``degenerate``.

Equivalent hash options follow the existing anchor and plotting arguments:

.. code-block:: none

   ## One direction for every detected pair:
   #eigenmode_port: 1 0 0 0.02 0.05 0.05 0.02 + 1,2 auto tracking=auto mode_polarizations=y
   ## One diagonal direction; its orthogonal partner is inferred:
   #eigenmode_port: 1 0 0 0.02 0.05 0.05 0.02 + 1,2 auto tracking=auto mode_polarizations=1,1,0

Use one definition per port and no spaces inside hash option values. Omitting
the option corresponds to Python ``None``. Semicolon-separated directions and
colon-based mappings raise an error. Values are parsed without expression
evaluation.

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
a two-mode group in a 3D cross-section. With one direction, the second is inferred.

Declare a group only for exact or numerically unresolved degeneracy. If
asymmetry resolves the splitting, excite the independently solved modes or
correct unintended geometry asymmetry. A group with negligible integrated
electric field cannot be oriented by axis/vector references; use generic
group tracking. These failures and their numerical thresholds are derived
in :ref:`eigenmode-degenerate-theory`.

.. _tracking-symmetry-options:

Automatic symmetry detection
----------------------------

With ``tracking="auto"``, no symmetry label is required. Symmetry is tested
**after** the longitudinal and gradient comparisons in
:ref:`tracking-separation-explained`. The tests use the actual sampled port,
including materials and boundaries, rather than the intended geometric name.
They check every solved anchor and use the strongest supported symmetry
common to the bank.

.. important::

   To use ``mode_polarizations`` with ``tracking="auto"`` in a PEC or SIBC
   waveguide, keep the complete port window C4v-symmetric. Use a square
   window with equal transverse cell spacing, center the guide in it, and
   keep padding equal on opposite sides. The sampled materials and boundary
   treatment must also preserve quarter turns and reflections. Avoid
   symmetry-breaking elements within the port cross-section.

   Symmetry is checked about the center of the full port window. Even an
   extra PEC cell on just one side can prevent certification and cause an
   explicit polarization request to be rejected. For SIBC walls, the
   retained samples and surface-impedance rows must also map under these
   symmetry operations. Geometry symmetry alone is insufficient: the
   selected modes must form a certified C4v E pair with usable electric
   moments.

   C4v is required for this automatic polarization choice. Ports with lower
   symmetry can still use automatic tracking and field-family separation
   with ``mode_polarizations=None``.

* **C4v** includes quarter turns and reflections, as for a centered square
  on a square mesh with equal transverse spacing. A centered staircased
  circle can also have C4v symmetry.
* **C2v** includes a half turn and two axis reflections, as for a centered
  rectangle. It can be available when quarter-turn symmetry is absent.

The geometry test identifies possible symmetries. The field test then checks
whether rotating or reflecting a candidate group keeps its fields inside
that same group and preserves their validity. This is what **certified**
means here. Symmetry can identify single channels (singlets) or, under C4v,
two-channel E pairs (doublets). Several different pairs can share the E
label, so that label alone does not distinguish their physical families.

Within a certified E pair, the default orientation uses electric moments
when they define reliable directions. Otherwise, reflection symmetry selects
the partners. These reflection labels describe how the complete vector field
transforms; they cannot be read from the sign of just one plotted component.
The mathematics and numerical thresholds are in
:ref:`tracking-moment-orientation-theory`.

Symmetry can express an axis-oriented pattern as combinations such as
TE20 + TE02 and TE20 - TE02. A further axis-selection step can recover
the individual TE20 and TE02 patterns within compatible candidates. The
output saves both the symmetry basis and the basis used for the source;
a label such as ``A2+B2`` records contributions from both symmetry types.
See :ref:`tracking-axis-selection` for this distinction.

If a C4v trial fails, the tracker tries C2v when supported. If neither is
accepted, successful longitudinal/gradient separation remains in place,
with smooth continuation only inside remaining unresolved blocks. Copper
surface-impedance boundaries have additional symmetry checks. Any rejected
change retains the preceding validated fields. The full-field, power, and
measurement-conditioning requirements are detailed in
:ref:`tracking-symmetry-theory`.

An explicit ``mode_polarizations`` direction is applied afterward, only to
certified C4v E pairs with usable electric moments. A diagonal polarization
need not itself have a single reflection parity; its two-channel space
retains the certified symmetry. The final fields must still pass all
physical and measurement checks.

.. _tracking-hdf5-results:

Reading the saved HDF5 results
------------------------------

Two output types share the tracking diagnostics. In a simulation's HDF5
output, port 1 is under ``/eigenmode_ports/port1``. In a native modal-field
export created by ``EigenmodeFieldOutput``, it is under ``/ports/1``.
The paths in the first table are relative to either port group.

.. list-table:: Saved tracking results and how to use them
   :class: api-parameters
   :header-rows: 1
   :widths: 25 38 37

   * - Group or dataset
     - What it records
     - What to look for
   * - ``mode_tracking``
     - Anchor matching, overlaps, numerical residuals, and requested/adaptive frequencies.
     - Use ``numerical_valid`` and residuals to assess solved fields. Check ``unresolved_interval_*`` for intervals whose mode assignment remains uncertain.
   * - ``mode_tracking/anchor_quality``
     - Mode/frequency rows with confinement, artifact, edge-fraction, validity, and reason fields.
     - Inspect ``confinement``, ``artifact``, and ``reasons`` for aperture or box-mode warnings, even when the numerical solve is valid.
   * - ``mode_tracking/subspace_recovery``
     - Candidate indices and transforms used to recover persistent groups at crossings, when needed.
     - Explains how a group was recovered from a larger solver cluster. Each record's ``candidate_indices`` identifies the original zero-based solver slots used in that recovery.
   * - ``mode_separation``
     - The pre-symmetry TE/TM, Ku-Kv, and Ku+Kv decisions. Each numbered group has ``ParentModes`` and ``Blocks`` attributes; each ``anchorN`` has a ``Selection`` attribute.
     - Decode ``Selection`` as JSON to inspect longitudinal scores, gradient scores, and symmetry candidates. Root attribute ``RejectedTrials`` records unsuccessful trials.
   * - ``symmetry_detection``
     - Geometry/boundary detection and the applied choice or rejected trials for each parent group, in attributes including ``Detected`` and JSON ``Details``.
     - A detected geometry symmetry is a candidate. Check the per-group applied result and ``symmetry_subspaces`` for successful field certification.
   * - ``symmetry_subspaces``
     - Certified blocks, symmetry labels, transforms, residuals, and per-anchor orientation records before explicit user polarization.
     - ``Symmetry`` and ``Irreps`` identify certified E pairs. ``orientation`` explains moment/parity selection; ``irreducible_basis`` and ``axis_sensitive`` distinguish symmetry fields from axis-oriented source fields.
   * - ``unresolved_transport``
     - Final alignment of unresolved blocks, with ``Blocks``, ``BlockMethods``, per-anchor transforms, and rejected trials.
     - ``procrustes`` means a block was continued smoothly; it does not certify a unique identity for every member. ``phase`` means only a singleton phase was aligned.
   * - ``polarization_subspaces``
     - Requested directions and accepted pairwise transforms, moment conditioning, and channel overlaps.
     - Present for accepted explicit automatic polarization. These transforms act on the certified pair basis.
   * - ``degenerate_groups``
     - Parent groups, final orientation, per-anchor transforms, power matrices, residuals, active/retained flags, and available directions/moments.
     - These are the final parent transforms, including later orientation choices: ``aligned fields = raw fields @ transform``.
   * - ``anchor_mode_valid``, ``anchor_mode_reference_valid``, ``anchor_mode_propagating``, ``anchor_balanced_power``
     - Anchor eligibility for excitation, reference use, propagation, and power normalization.
     - Check the mask for the intended use. A valid evanescent reference is not a propagating power wave; a confinement warning is separate from these masks.

Numbered diagnostic groups such as ``mode_separation/1`` identify records,
not public mode 1. Read their ``ParentModes``, ``ModeIndices``, or ``Blocks``
attributes to find the affected channels. Public mode labels are one-based;
``anchor0`` is the first anchor. Diagnostic arrays can include internally
solved partners as well as requested channels. Optional groups appear only
when the corresponding stage has records; absence by itself does not mean
that a run failed. JSON-valued attributes are read through ``group.attrs``,
not as HDF5 datasets.

The main ``mode_tracking/candidate_indices`` array maps tracked channels to
zero-based solver slots after recovery. The indices inside a
``subspace_recovery`` record instead identify the inputs to that recovery.
Neither array assigns analytical mode names.

.. list-table:: Modal fields versus driven simulation results
   :class: api-parameters
   :header-rows: 1
   :widths: 24 38 38

   * - Output
     - Saved quantities
     - Interpretation
   * - Native field export: ``/ports/1``
     - ``frequencies``, ``mode_indices``, ``neff``, optional ``operator_neff``, and ``fields/Ex`` through ``fields/Hz``.
     - Each component stores ``values`` in anchor, mode, u, v order, plus its own ``u`` and ``v`` coordinates for the staggered grid. These are modal profiles, not driven FDTD fields or S-parameters.
   * - Simulation output: ``/eigenmode_ports/port1``
     - ``frequency``, ``incident``, ``outgoing``, and ``S`` when available; solve frequencies and mode labels are in ``CandidateAnchorFrequencies`` and ``ModeIndices`` attributes, with ``anchor_complex_neff`` as a dataset.
     - Measurement frequencies differ from solve anchors. Use ``coefficient_valid`` for modal coefficients and ``power_wave_valid`` for power waves; use the corresponding ``*_S`` masks for S-parameters.

A smooth dispersion curve or an empty ``unresolved_interval_modes`` array
does not establish that every degenerate block has a unique physical basis.
Inspect ``mode_separation``, ``symmetry_subspaces``, and
``unresolved_transport`` for that question. For valid S-parameters, also
require their validity masks and finite complex values. The detailed schema
and transform conventions are in :ref:`tracking-hdf5-theory`.

.. _tracking-hybrid-conventions:

TE/TM and HE/EH conventions
---------------------------

TE modes have zero longitudinal electric field, :math:`E_w=0`; TM modes have
zero longitudinal magnetic field, :math:`H_w=0`. Dielectric guides generally
support **hybrid modes**, with both components nonzero. We use **HE** and
**EH** for the conventional hybrid families. Their identification follows
the vector fields and dispersion of a reference waveguide; the numerical
longitudinal-content score alone does not assign an HE/EH name.

In an ideal circular dielectric guide, :math:`\mathrm{HE}_{mn}` and
:math:`\mathrm{EH}_{mn}` use azimuthal order m and radial order n. The
fundamental pair is :math:`\mathrm{HE}_{11}`. Sine/cosine angular partners
belong to the same family and share the same indices. For rectangular or
square dielectric cores, an HE-like or EH-like label refers to a stated
reference-field convention; the circular angular/radial indices do not
directly become transverse m/n indices. See the
`COMSOL vector hybrid-mode example <https://doc.comsol.com/6.4/doc/com.comsol.help.models.woptics.microstructured_optical_fiber/microstructured_optical_fiber.html>`_
and its HE/EH field comparisons.

The runtime retains stable branch labels B1, B2, and so on. These labels
identify computed channels through frequency sweeps. Field plots may add a
verified conventional name, while parity and symmetry labels describe the
computed orientation.

Analytical validation
---------------------

The :ref:`validation gallery <eigenmode_tracking_validation_catalog>` compares
modal profiles and propagation constants for PEC rectangular, square, and
cylindrical guides, an open dielectric cylinder, and persistent pairs crossing
in an anisotropic square bore. Analytical fields check the computed channels
and do not determine their runtime labels.

Warnings and inspection
-----------------------

A warning about the aperture is different from a failed mode solve or an
uncertain mode identity. Use the following distinction when deciding what to
change:

.. list-table:: What the result means for your source
   :class: api-parameters
   :header-rows: 1
   :widths: 30 35 35

   * - Result
     - What happens
     - What to inspect
   * - Suspected box mode or poor confinement
     - A numerically valid, confidently tracked mode with forward power
       stays in use, with an accuracy warning. This warning alone does not
       remove anchors or change mode numbers.
     - Check whether the field reaches the artificial edge. Enlarge the
       aperture and compare fields and S-parameters.
   * - Verification could not finish
     - The result is unresolved, rather than proved good or bad. A usable
       primary profile is retained.
     - Read the reason: surrounding geometry may be unavailable, refinement
       may be unsupported, or the extra-solve budget may be exhausted.
   * - Unsure which mode connects to the next anchor
     - The tracker tries more candidates and intermediate frequencies. If
       still unresolved, it uses the automatic-anchor fallback or raises an
       error for explicit anchors. It does not interpolate across the gap.
     - Inspect the fields on both sides of the uncertain interval and the
       anchor-selection message before trusting the broadband result.
   * - Failed numerical solution
     - Fields that fail the equation checks, contain nonfinite values, or
       cannot be reconstructed or aligned reliably cannot supply a profile.
     - Investigate the solve failure; enlarging a warning threshold does not
       repair unusable fields.

A mode below cutoff is another case: decay along the guide does not make it
a box mode. It may remain useful as a receiver reference, but a mode without
usable forward power cannot be injected as a travelling power wave.

The figures place the dispersion curves, ``Re(n_eff)`` and ``-Im(n_eff)``,
above the E/H fields. The first shows how the phase propagation changes with
frequency; the second shows attenuation for the passive forward convention.
Diagnostic markers distinguish questionable profiles that remain in use
from missing or unusable profiles. A nonzero attenuation curve alone is not
evidence of a box mode.

For detailed inspection, the HDF5 ``mode_tracking`` group records which
solutions were matched, equation errors, verification evidence, and unresolved
intervals. Existing anchor and power validity masks keep their meanings.
The equations and exact decision rules are in
:ref:`eigenmode-auto-tracking-theory`.

.. _tracking-symmetry-examples:

Simulation examples
=====================

Start with the worked simulation below to follow the complete process:
configure tracking at the source and receiving ports, inspect the modal
bank, excite a tracked channel, run the simulation, and check the modal
measurements. Afterwards, use the :ref:`common-mode galleries
<tracking-common-mode-galleries>` to explore familiar field patterns and
crossing behaviour. Independent analytical checks are in
:ref:`eigenmode_tracking_validation_catalog`.

.. _eigenmode-example-8:

Worked simulation: the circular TE11 pair
-------------------------------------------

The complete model is
``examples/features/eigenmode_ports/example_8_auto_degenerate_te11``.
It constructs a straight circular PEC guide with a source port, a receiving
port, and a virtual-waveguide feed. Both ports use ``tracking="auto"`` and
request the two TE11 channels. The tracker detects their persistent pair
and selects reproducible default global x/y polarizations. A single shared
polarization direction can rotate those axes. Requesting only ``modes=(1,)``
still solves its missing partner internally.

1. **Configure the guide and ports.** Define the 20–24 GHz eigenmode band,
   place matching source and receiving apertures, and set ``tracking="auto"``
   on both ports. Use the same mode request and polarization convention at
   each end. Explicit anchors cover the band and pulse spectral guards.
   The bore uses ``averaging="y"`` to preserve shared PEC wall samples; see
   :ref:`eigenmode-pec-wall-sampling`.
2. **Inspect a geometry-only run.** Check the dispersion and E/H profiles
   before excitation. The TE11 channels should have coincident propagation
   curves and stable orthogonal orientations. Inspect ``mode_tracking``
   diagnostics for numerical validity and unresolved intervals. Resolve
   aperture or sampling problems before continuing.
3. **Choose the launched channel.** ``EigenmodeExcitation(port=1, mode=1,
   waveform="auto")`` launches the first tracked polarization. Changing
   ``mode`` to 2 launches its partner. The virtual guide terminates the feed;
   the receiving end continues into the domain PML.
4. **Run the full simulation.** Omit the geometry-only option and use the
   same port definitions. Repeat with the second launched mode to measure
   the other polarization.
5. **Check the modal measurements.** Plot co- and cross-polarized S11/S21 and
   the receiver traces. Interpret scattering results only where the output
   power-wave validity masks accept them. Confirm that channel identities
   remain consistent at both ports, then refine anchors, mesh, aperture, or
   terminations as needed for the intended accuracy.

Run these commands from the repository root in that order:

.. code-block:: console

   python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --geometry-only
   python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --mode 1
   python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/plot_results.py --mode 1
   python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py --mode 2
   python examples/features/eigenmode_ports/example_8_auto_degenerate_te11/plot_results.py --mode 2

``plot_fields=True`` writes modal-profile figures in geometry-only and full
runs. The result plotter produces the modal S-parameter and receiver-trace
figures after each full run. This example uses ``verification="fast"`` for
inspection; when adapting the model, omit that setting or select ``"full"``
for the additional convergence checks. The double-precision CPU runs were
regenerated on 24 October 2026. Worst co-polarized S11 is -62.9 dB for either
polarization, and the largest co-polarized S21 magnitude deviation from
0 dB is 0.0000326 dB. Both launches retain valid power-wave masks, finite
scattering data, and no unresolved tracking intervals.

:download:`Complete simulation model <../../examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11.py>`
and :download:`simulation result plotter <../../examples/features/eigenmode_ports/example_8_auto_degenerate_te11/plot_results.py>`.

.. figure:: ../../examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11_mode1_results.png
   :width: 95%
   :alt: Circular TE11 mode 1 reflection, transmission, and receiver polarization.

   Launching the first automatically oriented TE11 channel.

.. figure:: ../../examples/features/eigenmode_ports/example_8_auto_degenerate_te11/auto_degenerate_te11_mode2_results.png
   :width: 95%
   :alt: Circular TE11 mode 2 reflection, transmission, and receiver polarization.

   Launching its orthogonal partner. User polarization mixing requires a
   certified C4v E pair with significant, well-conditioned rank-two electric moments.

:download:`Recorded measurements and source hashes
<../../examples/features/eigenmode_ports/example_8_auto_degenerate_te11/recorded_results.json>`.

.. _eigenmode-example-9:

Worked simulation: branch crossing in a square anisotropic guide
----------------------------------------------------------------

Example 9 under ``eigenmode_ports`` uses a 16 × 16 mm square PEC bore filled
with ``diag(2.25, 1, 1)``. The TE10 branch polarized along y and TE02 branch
polarized along x start as modes 3 and 4; their propagation curves cross
near 14.4 GHz on this mesh. These are higher-ranked channels rather than the two lowest
modes. Both ports request ``modes=(3, 4)`` and ``tracking="auto"``. Their
tracked labels remain stable as the raw eigenvalue ranks exchange.

Follow the same full process as the circular-pair example: inspect the
geometry-only dispersion and modal fields, choose one tracked channel to
launch, run the simulation, and inspect co- and cross-channel scattering
with valid power-wave masks. Explicit anchors cover the 13.1–15.8 GHz band
and its guards. Walls are built after the anisotropic fill to preserve the
modal/FDTD boundary agreement, and sixteen-cell PMLs terminate both ends.
An off-centre receiver samples both Ex and Ey; the TE02 field is zero at
the bore centre.

.. code-block:: console

   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/auto_mode_crossing.py --geometry-only
   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/auto_mode_crossing.py --mode 3
   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/plot_results.py --mode 3
   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/auto_mode_crossing.py --mode 4
   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/plot_results.py --mode 4

The result plots mark the tracked crossing and show the launched channel's
reflection and transmission alongside the interior probe polarization.
This demonstrates automatic tracking during a driven simulation; the later
crossing galleries inspect undriven modal banks.

:download:`Complete square-guide simulation <../../examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/auto_mode_crossing.py>`
and :download:`result plotter <../../examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/plot_results.py>`.

.. figure:: ../../examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/square_crossing_mode3_results.png
   :width: 95%
   :alt: Mode 3 retains Ey polarization through the crossing while reflection and cross-channel transmission remain small.

   Launching tracked mode 3: TE10 / Ey. The dotted line marks the crossing
   inferred from retained anchors. The receiver is at x = 10 mm, y = 6 mm.

.. figure:: ../../examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/square_crossing_mode4_results.png
   :width: 95%
   :alt: Mode 4 retains Ex polarization through the crossing while reflection and cross-channel transmission remain small.

   Launching tracked mode 4: TE02 / Ex. Both full runs retain valid power-wave
   masks over the measurement band and have no unresolved tracking intervals.

The double-precision CPU runs regenerated on 24 October 2026 have worst co-channel S11 of
-81.1 dB and -62.4 dB for launched modes 3 and 4. The largest co-channel
S21 magnitude deviation from 0 dB is 0.000102 dB. These results concern the
represented mesh and port sampling; they do not establish convergence to
an exact continuum boundary. :download:`Recorded measurements
<../../examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/recorded_results.json>`.

.. _tracked-mode-gallery:
.. _eigenmode_tracking_examples:
.. _tracking-common-mode-galleries:

Common-mode galleries
=======================

After following the simulation workflow, browse these galleries to recognize
tracked modes in common guides. Their scripts live under
``examples/features/eigenmode_tracking`` and perform geometry-only runs.
They export native E/H fields and modal dispersion; they do not excite a
port or produce scattering measurements. Gallery numbers match the numbered
script folders, which are separate from the driven ``eigenmode_ports``
tutorials above.

Choose a geometry and material
--------------------------------

.. list-table:: Common-mode galleries
   :header-rows: 1

   * - Gallery
     - Guide
     - Material
   * - :ref:`1 <eigenmode_tracking_example_1_C2v_rectangular_PEC>`
     - C2v rectangular, 20 × 10 mm
     - PEC
   * - :ref:`2 <eigenmode_tracking_example_2_C4v_square_PEC>`
     - C4v square, 20 × 20 mm
     - PEC
   * - :ref:`3 <eigenmode_tracking_example_3_C4v_cylindrical_PEC>`
     - Cylindrical, radius 10 mm
     - PEC
   * - :ref:`4 <eigenmode_tracking_example_4_C2v_rectangular_copper>`
     - C2v rectangular, 20 × 10 mm
     - Copper
   * - :ref:`5 <eigenmode_tracking_example_5_C4v_square_copper>`
     - C4v square, 20 × 20 mm
     - Copper
   * - :ref:`6 <eigenmode_tracking_example_6_C4v_cylindrical_copper>`
     - Cylindrical, radius 10 mm
     - Copper
   * - :ref:`7 <eigenmode_tracking_example_7_C2v_rectangular_dielectric>`
     - Rectangular core, 20 × 10 mm
     - εr = 4 in air
   * - :ref:`8 <eigenmode_tracking_example_8_C4v_square_dielectric>`
     - Square core, 20 × 20 mm
     - εr = 4 in air
   * - :ref:`9 <eigenmode_tracking_example_9_C4v_cylindrical_dielectric>`
     - Cylindrical core, radius 10 mm
     - εr = 4 in air

Each common-guide gallery shows sixteen channels over 39–41 GHz, with
real/imaginary effective-index curves and E/H fields at the middle anchor.
Colour gives field magnitude normalized per mode; arrows show transverse
fields at a common phase. The default spacing is 0.25 mm. Copper uses the
conductivity surface-impedance preset with an ADE fit over 1–120 GHz.
Dielectric apertures have distant artificial PEC rims, and cylindrical
boundaries are staircased. Detected symmetry labels describe that sampled
problem.

All eleven galleries were regenerated on 24 October 2026 with longitudinal
TE/TM separation followed by Ku-Kv and Ku+Kv, symmetry refinement, and
Procrustes transport within any remaining unresolved blocks. Every retained
gallery mode passes the numerical and reference-validity checks, with no
unresolved tracking intervals. The finite copper windows retain
``unbound_suspect`` confinement flags from artificial-edge participation;
their manifests preserve these warnings. Numerical validity alone does not
establish aperture convergence or an exact continuum modal identity.

To regenerate a gallery, run its script from the repository root, for example::

   python examples/features/eigenmode_tracking/example_1_C2v_rectangular_PEC/tracked_modes.py

``--spacing-mm`` changes the mesh, ``--modes`` changes the requested channel
count, ``--verification full`` adds convergence checks, and ``--output``
selects a fresh result directory. Recorded galleries are separate from new
outputs. Analytical field and propagation-constant comparisons remain in
:ref:`eigenmode_tracking_validation_catalog`.

Recorded common modes
-----------------------

.. _eigenmode_tracking_example_1_C2v_rectangular_PEC:

Gallery 1: PEC rectangular tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_1_C2v_rectangular_PEC/gallery/pec_rectangular_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_1_C2v_rectangular_PEC/gallery/pec_rectangular_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_1_C2v_rectangular_PEC/gallery/pec_rectangular_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_1_C2v_rectangular_PEC/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_1_C2v_rectangular_PEC/gallery/manifest.json>`

.. _eigenmode_tracking_example_2_C4v_square_PEC:

Gallery 2: PEC square tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_2_C4v_square_PEC/gallery/pec_square_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_2_C4v_square_PEC/gallery/pec_square_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_2_C4v_square_PEC/gallery/pec_square_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_2_C4v_square_PEC/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_2_C4v_square_PEC/gallery/manifest.json>`

.. _eigenmode_tracking_example_3_C4v_cylindrical_PEC:

Gallery 3: PEC cylindrical tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_3_C4v_cylindrical_PEC/gallery/pec_cylindrical_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_3_C4v_cylindrical_PEC/gallery/pec_cylindrical_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_3_C4v_cylindrical_PEC/gallery/pec_cylindrical_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_3_C4v_cylindrical_PEC/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_3_C4v_cylindrical_PEC/gallery/manifest.json>`

.. _eigenmode_tracking_example_4_C2v_rectangular_copper:

Gallery 4: COPPER rectangular tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_4_C2v_rectangular_copper/gallery/copper_rectangular_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_4_C2v_rectangular_copper/gallery/copper_rectangular_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_4_C2v_rectangular_copper/gallery/copper_rectangular_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_4_C2v_rectangular_copper/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_4_C2v_rectangular_copper/gallery/manifest.json>`

.. _eigenmode_tracking_example_5_C4v_square_copper:

Gallery 5: COPPER square tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_5_C4v_square_copper/gallery/copper_square_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_5_C4v_square_copper/gallery/copper_square_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_5_C4v_square_copper/gallery/copper_square_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_5_C4v_square_copper/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_5_C4v_square_copper/gallery/manifest.json>`

.. _eigenmode_tracking_example_6_C4v_cylindrical_copper:

Gallery 6: COPPER cylindrical tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_6_C4v_cylindrical_copper/gallery/copper_cylindrical_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_6_C4v_cylindrical_copper/gallery/copper_cylindrical_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_6_C4v_cylindrical_copper/gallery/copper_cylindrical_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_6_C4v_cylindrical_copper/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_6_C4v_cylindrical_copper/gallery/manifest.json>`

.. _eigenmode_tracking_example_7_C2v_rectangular_dielectric:

Gallery 7: DIELECTRIC rectangular tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_7_C2v_rectangular_dielectric/gallery/dielectric_rectangular_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_7_C2v_rectangular_dielectric/gallery/dielectric_rectangular_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_7_C2v_rectangular_dielectric/gallery/dielectric_rectangular_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_7_C2v_rectangular_dielectric/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_7_C2v_rectangular_dielectric/gallery/manifest.json>`

.. _eigenmode_tracking_example_8_C4v_square_dielectric:

Gallery 8: DIELECTRIC square tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_8_C4v_square_dielectric/gallery/dielectric_square_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_8_C4v_square_dielectric/gallery/dielectric_square_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_8_C4v_square_dielectric/gallery/dielectric_square_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_8_C4v_square_dielectric/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_8_C4v_square_dielectric/gallery/manifest.json>`

.. _eigenmode_tracking_example_9_C4v_cylindrical_dielectric:

Gallery 9: DIELECTRIC cylindrical tracked modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_9_C4v_cylindrical_dielectric/gallery/dielectric_cylindrical_dispersion.png
   :width: 100%
   :alt: Tracked modal dispersion.

.. figure:: ../../examples/features/eigenmode_tracking/example_9_C4v_cylindrical_dielectric/gallery/dielectric_cylindrical_E_modes.png
   :width: 100%
   :alt: Electric fields for every tracked channel.

.. figure:: ../../examples/features/eigenmode_tracking/example_9_C4v_cylindrical_dielectric/gallery/dielectric_cylindrical_H_modes.png
   :width: 100%
   :alt: Magnetic fields for every tracked channel.

:download:`Executable script <../../examples/features/eigenmode_tracking/example_9_C4v_cylindrical_dielectric/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_9_C4v_cylindrical_dielectric/gallery/manifest.json>`

.. _tracking-crossing-examples:

Crossing-mode galleries
-------------------------

Gallery 10 follows two polarized branches crossing in frequency in an
anisotropic rectangular PEC bore. Gallery 11 shows two persistent square-
guide doublets crossing as axial permittivity changes at fixed 20 GHz.
These focused cases use two and eight channels. Gallery 11 combines
geometry-only native port snapshots with an undriven material-parameter
FDFD sweep and analytical comparisons. At the exact numerical coincidence,
all four input modes are deliberately mixed, and tracking recovers the two
persistent pairs.

In the regenerated 64- and 96-cell parameter sweeps, the maximum relative
propagation-constant error is below 0.000157, the minimum analytical profile
subspace overlap exceeds 0.9999999, and the maximum Maxwell residual is
below 2.2e-12. The recorded sweep includes the deliberately mixed exact
crossing at each mesh's numerical coincidence.

Regenerate them with::

   python examples/features/eigenmode_tracking/example_10_anisotropic_frequency_crossing/tracked_modes.py
   python examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/tracked_modes.py

.. _eigenmode_tracking_example_10_anisotropic_frequency_crossing:

Gallery 10: Two polarized branches crossing in frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_10_anisotropic_frequency_crossing/gallery/frequency_crossing_dispersion.png
   :width: 100%
   :alt: frequency crossing dispersion.png

.. figure:: ../../examples/features/eigenmode_tracking/example_10_anisotropic_frequency_crossing/gallery/frequency_crossing_E_modes.png
   :width: 100%
   :alt: frequency crossing E modes.png

.. figure:: ../../examples/features/eigenmode_tracking/example_10_anisotropic_frequency_crossing/gallery/frequency_crossing_H_modes.png
   :width: 100%
   :alt: frequency crossing H modes.png

:download:`Executable script <../../examples/features/eigenmode_tracking/example_10_anisotropic_frequency_crossing/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_10_anisotropic_frequency_crossing/gallery/manifest.json>`

.. _eigenmode_tracking_example_11_C4v_square_persistent_pair_crossing:

Gallery 11: Persistent doublets crossing in a square guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/gallery/propagation_constants.png
   :width: 100%
   :alt: propagation constants.png

.. figure:: ../../examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/gallery/modal_profiles.png
   :width: 100%
   :alt: modal profiles.png

.. figure:: ../../examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/gallery/before_crossing_E_modes.png
   :width: 100%
   :alt: before crossing E modes.png

.. figure:: ../../examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/gallery/after_crossing_E_modes.png
   :width: 100%
   :alt: after crossing E modes.png

:download:`Executable script <../../examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/tracked_modes.py>`

:download:`Recorded run metadata <../../examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/gallery/manifest.json>`

.. figure:: ../../examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/gallery/before_crossing_H_modes.png
   :width: 100%
   :alt: Magnetic fields before the crossing.

.. figure:: ../../examples/features/eigenmode_tracking/example_11_C4v_square_persistent_pair_crossing/gallery/after_crossing_H_modes.png
   :width: 100%
   :alt: Magnetic fields after the crossing.
