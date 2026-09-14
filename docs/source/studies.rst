.. _studies:

****************
Reusable studies
****************

A study schedules repeated field solves with one built geometry. Choose a
family by the objects and outputs required, not by the application's name.

.. list-table:: Choose a study
   :header-rows: 1
   :widths: 20 35 45

   * - Family / CSV type
     - Typical task
     - What changes
   * - :ref:`GPRStudy <study-gpr>` / ``gpr``
     - B-scans, multistatic surveys, point-source acquisitions
     - Supported point-source/receiver positions and source waveforms/timing
   * - :ref:`SourceStudy <study-source>` / ``source``
     - Multiple-feed antennas and weighted terminal drives
     - TL/frill/network generators; not the physical loads or positions
   * - :ref:`PortStudy <study-port>` / ``port``
     - Complete finite-resistance voltage-port S matrix
     - Exactly one driven voltage port per case; other ports remain loaded
   * - :ref:`EigenmodeStudy <study-eigenmode>` / ``eigenmode``
     - Multimode S matrices, embedded patterns and coherent array states
     - One declared port/mode channel per solve; optional post-solve codebook
   * - :ref:`PlaneWaveStudy <study-plane-wave>` / ``plane_wave``
     - Incidence-angle/polarisation sweeps and RCS
     - One DPW template's direction, polarisation and waveform

Geometry/material parameter sweeps require fresh Scenes: do not change a
material, source resistance or network in a reused study. Moving a complete
toolbox antenna requires moving/rebuilding its geometry, not a GPRStudy
point-source override. See :doc:`inc_Optimisation`. For many waveforms on
the same linear system, :doc:`inc_ImpulseResponse` can replace repeated
solves with convolution if the feed boundary condition remains unchanged.
SFCW/FMCW processing is described in :doc:`inc_SFCW` and :doc:`inc_FMCW`.

.. _study-compatibility:

Execution and reuse
===================

Studies run sequential cases on the selected solver. CPU, CUDA, OpenCL and
Metal may be used where the selected source, dimensionality and outputs are
supported; a study does not extend a feature's backend support. MPI domain
decomposition and task farming are unsupported for every study family.

* GPRStudy, SourceStudy and PortStudy manage main-grid objects. GPR studies
  can use supported reduced 2-D point-source models; terminal ports need 3-D.
* EigenmodeStudy can manage ports on the main grid or an owning CPU or CUDA HSG
  subgrid. Ordinary dimensionality and virtual-guide restrictions still
  apply; see :doc:`eigenmode_port`.
* PlaneWaveStudy uses a main-grid TFSF source and may enclose complete
  subgrids, but those subgrids cannot contain another excitation.
* Use exactly one Scene or one hash input. The case count supplies ``n``
  automatically. Do not add ``src_steps``/``rx_steps`` acquisition
  controls to a study.
* Geometry and discretisation stay fixed. Fields, PML state, source
  histories and output accumulators reset per case. Modal bases are cached;
  direction-dependent auxiliary plane waves are rebuilt. Reuse does not
  mean every backend allocation or every source setup is free.
* Source, Port, Eigenmode and PlaneWave studies require a solve and reject
  ``geometry_only``. Inspect a modal scene with ordinary
  ``geometry_only=True`` before attaching its study.

Case IDs and restart
--------------------

Give each case a unique ID. A CSV row describes one object state, not
necessarily a whole case: all rows with the same case ID form one case, in
order of first appearance of that ID. An object may be listed only once per
case. Keep related rows together for readability. Object IDs
identify declarations, not receiver group numbers. Prefer explicit IDs where
supported and keep them stable across files.

Use Python ``i=N`` or command-line ``-i N`` to resume at one-based
case N. Output numbering remains absolute. This is a restart at a case
boundary, not a checkpoint within a time step. Port and eigenmode studies
reuse compatible saved aggregate data; use a separate output prefix for a
changed experiment.

Read results by task
--------------------

Each solve produces ``<prefix>1.h5``, ``<prefix>2.h5``, etc., with
requested receiver/port/NTFF outputs and ``/study`` provenance. Only
PortStudy and EigenmodeStudy additionally produce ``<prefix>_study.h5``
with an aggregate S matrix.

* GPR: merge matching receiver identities and plot a B-scan; see the
  :ref:`CSV acquisition example <bscan_csv_study>`.
* Source: compare per-case terminal histories and NTFF patterns.
  ``plot_port`` reads local terminal results, not a full S matrix.
* Port: read ``S[frequency, output_port, input_port]`` with its validity
  mask and port-ID array, or use ``results['study']`` after the solve.
* Eigenmode: follow the channel port/mode arrays and distinguish coefficient
  validity from propagating power-wave validity.
* PlaneWave: the observation requests stay fixed. Select the backscatter
  direction appropriate to each incidence for a monostatic curve; a fixed
  observation direction is not generally monostatic.

See :ref:`study-output` for schemas and :ref:`study-worked-examples` for
small runnable models. The scheduling snippets below assume an existing
``scene``, its geometry and a ``pulse`` waveform.

.. _study-gpr:

GPR acquisitions
================

A :class:`gprMax.Study` runs an ordered set of source and receiver states while
reusing one built geometry. It is intended for arbitrary GPR acquisition
patterns and underlies the specialised multiport, antenna-array, and
plane-wave workflows. Every case restores the original object state before its
overrides are applied, so parameters cannot accidentally accumulate between
runs.

General GPR studies support top-level
:class:`gprMax.HertzianDipole`, :class:`gprMax.MagneticDipole`, and
:class:`gprMax.Rx` objects on the main grid. A state can refer directly to its
Python object or use its deterministic ID. Sources omitted from a case are
inactive; receivers omitted from a case keep their original position and are
recorded.

.. code-block:: python

    source = gprMax.HertzianDipole(
        polarisation='z', p1=(0.10, 0.05, 0.03), waveform_id='pulse'
    )
    receiver = gprMax.Rx(p1=(0.14, 0.05, 0.03), id='measurement')
    scene.add(source)
    scene.add(receiver)

    study = gprMax.GPRStudy([
        gprMax.StudyCase('trace_1', [
            gprMax.ObjectState(source, position=(0.10, 0.05, 0.03), scale=1.0),
            gprMax.ObjectState(receiver, position=(0.14, 0.05, 0.03)),
        ]),
        gprMax.StudyCase('trace_2', [
            gprMax.ObjectState(source, position=(0.102, 0.052, 0.03), scale=0.8),
            gprMax.ObjectState('measurement', position=(0.145, 0.052, 0.03)),
        ]),
    ])

    gprMax.run(scenes=[scene], study=study, outputfile='survey')

The available source overrides are ``active``, ``position``,
``waveform_id``, ``start``, ``stop``, and the dimensionless amplitude
``scale``. Receivers currently accept ``position`` and ``record=True``. The
study determines the run count automatically; pass ``i=N`` to restart at the
one-based case number ``N``. For a text input model the equivalent
``#study`` command reads the same information from CSV.

Study ``start``/``stop`` overrides must be finite. ``active`` and ``record``
accept Python or NumPy boolean scalars; strings, numeric flags and boolean
arrays are rejected rather than interpreted by truthiness. These checks also
apply before reusing a study in another run. ``record=False`` remains
unsupported and is rejected for both boolean types.

For a complete acquisition that users can edit in a spreadsheet, see the
:ref:`CSV B-scan example <bscan_csv_study>`. A Python model can use that same
schedule with ``gprMax.Study.from_csv('gpr', 'path/to/cases.csv')`` and pass
the returned object as the ``study`` argument to ``gprMax.run``.



.. _study-source:

Fixed-topology terminal-source studies
======================================

A :class:`gprMax.SourceStudy` reuses a model containing stateful terminal
sources. It supports main-grid :class:`gprMax.TransmissionLine`,
:class:`gprMax.MagneticFrillSource`, and
:class:`gprMax.NetworkExcitation` objects. Their positions and physical
definitions remain fixed, but each case may change ``waveform_id``, ``start``,
``stop``, and the dimensionless generator ``scale``. ``active=false`` is
equivalent to zero generator drive.

This is deliberately different from an S-parameter study: any number of
terminals may be active in one case, which is useful for phased-array and
multiple-feed antenna patterns. A source omitted from a case is not removed.
Its transmission-line resistance, coaxial-frill termination, or rational
network remains coupled to the Yee grid as a passive load.

.. code-block:: python

    scene.add(gprMax.RationalNetwork(id='load50', conductance=1 / 50))
    scene.add(gprMax.NetworkTerminal(
        p1=(0.040, 0.050, 0.030), polarisation='z',
        network_id='load50', id='port1'
    ))
    scene.add(gprMax.NetworkTerminal(
        p1=(0.060, 0.050, 0.030), polarisation='z',
        network_id='load50', id='port2'
    ))
    scene.add(gprMax.NetworkPort('port1'))
    scene.add(gprMax.NetworkPort('port2'))
    feed1 = gprMax.NetworkExcitation('port1', 'pulse')
    feed2 = gprMax.NetworkExcitation('port2', 'pulse')
    scene.add(feed1)
    scene.add(feed2)

    study = gprMax.SourceStudy([
        gprMax.StudyCase('feed_1_only', [
            gprMax.ObjectState(feed1, scale=1),
        ]),
        gprMax.StudyCase('equal_feeds', [
            gprMax.ObjectState(feed1, scale=1),
            gprMax.ObjectState(feed2, scale=1),
        ]),
        gprMax.StudyCase('weighted_feeds', [
            gprMax.ObjectState(feed1, scale=1),
            gprMax.ObjectState(feed2, scale=-1),
        ]),
    ])

    gprMax.run(scenes=[scene], study=study, outputfile='fed_array')

Before every case gprMax reconstructs the selected source waveform and clears
all transmission-line voltage/current and ABC state, magnetic-frill recurrence
and histories, rational-network pole state, receiver histories, and derived
port results. Declarative NTFF monitors are recompiled with new accumulators,
so every case may safely produce an independent antenna pattern. SourceStudy
uses the normal CPU, CUDA, OpenCL, or Metal implementation of each terminal.
It does not currently support source objects inside a subgrid, MPI execution,
or task farming.


.. _study-port:

Finite-resistance voltage-port studies
======================================

A :class:`gprMax.PortStudy` calculates a complete multiport S matrix without
rebuilding the antenna geometry. Every finite-resistance
:class:`gprMax.VoltageSource` remains on its original electric edge in every
case. Exactly one source is driven, while omitted sources have zero generator
voltage but retain their resistance and therefore act as passive matched
terminations. Every source automatically owns a port monitor with a unique
explicit or automatically assigned ID; explicit IDs are recommended.

.. code-block:: python

    port1_source = gprMax.VoltageSource(
        p1=(0.040, 0.050, 0.030), polarisation='z', resistance=50,
        waveform_id='pulse', id='port1'
    )
    port2_source = gprMax.VoltageSource(
        p1=(0.060, 0.050, 0.030), polarisation='z', resistance=50,
        waveform_id='pulse', id='port2'
    )
    scene.add(port1_source)
    scene.add(port2_source)

    study = gprMax.PortStudy([
        gprMax.StudyCase('drive_port1', [
            gprMax.ObjectState(port1_source, scale=1.0),
        ]),
        gprMax.StudyCase('drive_port2', [
            gprMax.ObjectState(port2_source, scale=1.0),
        ]),
    ])

    results = gprMax.run(scenes=[scene], study=study, outputfile='array')
    smatrix = results['study'].s

The returned and stored matrix uses
``S[frequency, output_port, input_port]``. Voltage waves are converted to
power-wave normalisation, so ports may use different positive real reference
impedances. The individual source gaps contain numerical background
capacitance and conductance. These are removed from the complete admittance
matrix:

.. math::

    \begin{aligned}
    \overline{\mathbf{Y}}_{\mathrm{s}}
      &= (\mathbf{I}-\mathbf{S}_{\mathrm{s}})
         (\mathbf{I}+\mathbf{S}_{\mathrm{s}})^{-1}, \\
    \overline{\mathbf{Y}}
      &= \overline{\mathbf{Y}}_{\mathrm{s}}
         - \operatorname{diag}(Z_{0,p}Y_{\mathrm{gap},p}), \\
    \mathbf{S}
      &= (\mathbf{I}+\overline{\mathbf{Y}})^{-1}
         (\mathbf{I}-\overline{\mathbf{Y}}).
    \end{aligned}

This matrix operation is important: applying the scalar one-port correction
independently to off-diagonal elements is not mathematically valid. The
per-case files contain the raw source-plane column, and ``array_study.h5``
contains both ``S_source`` and the corrected ``S`` matrix. Restarting with
``i=N`` reuses compatible columns already present in this aggregate file.

Source position and resistance are immutable because both affect the built
electric-edge material. The permitted case parameters are ``active``,
``waveform_id``, ``start``, ``stop``, and ``scale``. Hard voltage sources are
rejected because zero drive would impose a zero electric field rather than a
matched passive termination.



.. _study-eigenmode:

Eigenmode-port studies and array synthesis
==========================================

An :class:`gprMax.EigenmodeStudy` constructs the complete modal S matrix by
exciting one declared ``(port, mode)`` channel per case. The geometry, Yee
arrays, FDFD modal solutions, phase-aligned anchor fields, and modal power
normalisation are prepared once. Between cases gprMax clears the main and
virtual-waveguide fields, PML histories, modal DFT accumulators, recursive DFT
phases, and derived S data before selecting the next cached modal basis.

Every declared mode on every :class:`gprMax.EigenmodePort` must appear in
exactly one case. This deliberate one-active-channel policy gives ordinary
S-parameters; exciting several ports in one solve would yield only the active
relation :math:`b(f)=S(f)a(f)`, not all columns of :math:`S`.

The nominally passive ports can still contain a small measured incident wave
because their finite terminations are not mathematically perfect. gprMax does
not assume that the incident-wave matrix is diagonal. For the independent
cases it assembles

.. math::

   A(f) = [a^{(1)}(f)\;\cdots\;a^{(N)}(f)], \qquad
   B(f) = [b^{(1)}(f)\;\cdots\;b^{(N)}(f)],

and obtains the authoritative aggregate matrix from

.. math::

   B(f)=S(f)A(f), \qquad S(f)=B(f)A(f)^{-1}.

The implementation uses a conditioned linear solve, not an explicit inverse.
Bins with an incomplete, invalid, or ill-conditioned incident basis are
marked invalid. The measured :math:`A` and :math:`B`, their validity masks,
condition number, and solve-valid flag are retained in the aggregate HDF5
file for audit and restart.

.. code-block:: python

    excitation = gprMax.EigenmodeExcitation(
        port=1, mode=1, waveform='auto', plot_waveform=False
    )
    scene.add(excitation)

    study = gprMax.EigenmodeStudy([
        gprMax.StudyCase('p1m1', [
            gprMax.ObjectState(excitation, port=1, mode=1),
        ]),
        gprMax.StudyCase('p2m1', [
            gprMax.ObjectState(excitation, port=2, mode=1),
        ]),
    ])

    results = gprMax.run(scenes=[scene], study=study, outputfile='array')
    modal_s = results['study'].s

The corresponding hash command selects an eigenmode study and its case CSV:

.. code-block:: none

    #study: eigenmode cases.csv

For the two channels above, ``cases.csv`` contains:

.. code-block:: text

    case_id,object_id,port,mode
    p1m1,eigenmode_excitation_1,1,1
    p2m1,eigenmode_excitation_1,2,1

The matrix convention is
``S[frequency, output_channel, input_channel]``. ``channel_ports`` and
``channel_modes`` define both channel axes. ``power_wave_valid_s`` and
``coefficient_valid_s`` are the physical power-wave and modal-coefficient
masks on ``EigenmodeStudyResult``. HDF5 uses ``power_wave_valid_S`` and
``coefficient_valid_S``. An evanescent coefficient is not a propagating power
wave. Compatible columns in an existing ``<output>_study.h5`` are retained
when restarting with ``i=N``. More precisely, the compatible raw excitation
cases are retained and the full de-embedding solve is repeated using all
available cases.

Embedded far fields from the individual cases can be retained and combined
without a new FDTD solve. Select an existing frequency-domain KSIR or
equivalent-current far-field request, then define named states in an
:class:`gprMax.ArrayCodebook`:

.. code-block:: python

    codebook = gprMax.ArrayCodebook(
        states=[
            gprMax.ArrayState('broadside', [
                gprMax.ModalWeight(port=1, mode=1, power=1),
                gprMax.ModalWeight(port=2, mode=1, power=1),
            ]),
            gprMax.ArrayState('steered', [
                gprMax.ModalWeight(port=1, mode=1, power=1),
                gprMax.ModalWeight(port=2, mode=1, power=1, phase_deg=90),
            ]),
        ],
        embedded_far_fields=[
            gprMax.EmbeddedFarFieldSpec('antenna_band', 'pattern'),
        ],
    )
    study = gprMax.EigenmodeStudy(cases, codebook=codebook)
    results = gprMax.run(scenes=[scene], study=study, outputfile='array')
    steered = results['study'].evaluate_array_state(codebook.states[1])

The same versioned definition can be loaded from JSON with
``ArrayCodebook.from_json`` and serialized again with ``to_json``. For
hash-command models use
``#array_codebook: file.json`` alongside ``#study: eigenmode ...``. An
existing aggregate can be reopened without another solve:

.. code-block:: python

    study_result = gprMax.EigenmodeStudyResult.from_hdf5('array_study.h5')
    codebook = gprMax.ArrayCodebook.from_json('array_states.json')
    states = study_result.evaluate_codebook(codebook)

Here ``power`` is incident modal power in watts, so the power-wave magnitude
is its square root. With the engineering Fourier convention a constant phase
uses :math:`\exp(+j\phi)`, whereas a delay uses
:math:`\exp(-j2\pi f\tau)`. Constant phase produces ordinary narrowband
beam steering and beam squint; true time delay preserves steering over
bandwidth. In a lower-level embedded response array, the first axis is
frequency and the selected channel axis follows ``channel_ports`` and
``channel_modes``; use ``combine_embedded_modal_responses`` to combine such
an array directly.

For state :math:`q`, gprMax forms the incident vector

.. math::

   a_{q,p}(f)=\sqrt{P_{q,p}}\,
   \exp\!\left(j\phi_{q,p}-j2\pi f\tau_{q,p}\right),
   \qquad b_q(f)=S(f)a_q(f).

It reports the active reflection coefficient :math:`b_{q,p}/a_{q,p}` for
each driven channel and

.. math::

   \mathrm{TARC}_q(f)=
   \sqrt{\frac{\sum_p |b_{q,p}(f)|^2}
                    {\sum_p |a_{q,p}(f)|^2}}.

Complex embedded :math:`E_\theta` and :math:`E_\phi` fields use the same full
incident-matrix de-embedding. If :math:`F_{\mathrm{runs}}` contains the raw
field from each case, gprMax solves
:math:`F_{\mathrm{runs}}=F_{\mathrm{emb}}A` for the embedded modal basis.
A retained full-sphere quadrature is treated in the same way, so
radiated power, directivity, gain, realized gain, and efficiencies include
the coherent cross terms. Only physical propagating power-wave bins marked by
``power_wave_valid_S`` are used for these power metrics; generalized evanescent
coefficients remain available in ``S`` but are not treated as watts.
No embedded-field storage or full-sphere evaluation is performed unless a
codebook explicitly selects a far-field output. With a selection, storage is
proportional to the number of frequencies, quadrature directions, and modal
channels. Both the raw case fields needed for restart/audit and the
de-embedded modal basis are retained, approximately doubling the complex field
storage compared with keeping only one representation. The synthesis assumes
a linear, time-invariant antenna and feed
model; a nonlinear or state-dependent feed network requires new driven
simulations rather than post-processing.







The study methods above return structured result objects rather than scene
commands. Their array shapes, channel ordering, units, and validity masks are
documented in :doc:`input_api`. These objects can also be reconstructed from a saved study
file, allowing array states to be evaluated without rerunning FDTD.






.. _study-plane-wave:

Plane-wave and RCS studies
==========================

A :class:`gprMax.PlaneWaveStudy` evaluates several incident plane waves while
building the main Yee geometry only once. The Scene contains exactly one
discrete-plane-wave object, which acts as the reusable template, and each case
changes its direction, polarisation, timing, waveform, or amplitude. Other
active source types are rejected so that scattered-field and RCS results have
an unambiguous incident wave.

.. code-block:: python

    plane_wave = gprMax.DiscretePlaneWaveAngles(
        p1=(0.03, 0.03, 0.03),
        p2=(0.07, 0.07, 0.07),
        theta=90,
        phi=0,
        psi=90,
        waveform_id='pulse',
    )
    scene.add(plane_wave)

    study = gprMax.PlaneWaveStudy([
        gprMax.StudyCase('x_incidence', [
            gprMax.ObjectState(plane_wave, theta=90, phi=0, psi=90),
        ]),
        gprMax.StudyCase('y_incidence', [
            gprMax.ObjectState(plane_wave, theta=90, phi=90, psi=90),
        ]),
    ])

    gprMax.run(scenes=[scene], study=study, outputfile='angular_rcs')

The TFSF box, background material, and angular-approximation tolerance remain
fixed because they define the reusable source topology. The parameters which
may change depend on the template:

* :class:`gprMax.DiscretePlaneWaveAngles`: ``theta``, ``phi``, and ``psi``;
* :class:`gprMax.DiscretePlaneWaveVector`: ``m_vec`` and ``psi``;
* :class:`gprMax.DiscretePlaneWaveAxial`: ``axis`` and ``psi``.

All three forms also accept per-case ``waveform_id``, ``start``, ``stop``, and
non-zero dimensionless ``scale``. The principal Yee arrays and material IDs
are retained, but the small auxiliary one-dimensional DPW grid is rebuilt for
each case. This is necessary because its length, rational integer mapping,
field projections, material profile, and PML state depend on the propagation
direction.

Declarative NTFF transforms are also reconstructed for every case. Their
surface geometry is reused, while all time/frequency accumulators and the
incident-wave DFT are new. Consequently an RCS result cannot contain state
from an earlier direction. Each numbered HDF5 file records the requested
study case under ``/study`` and the actual rationalised plane-wave parameters
under the frequency transform's ``plane_wave`` group. A complete subgrid may
be enclosed by the fixed TFSF and NTFF surfaces, subject to the normal
enclosure rules, but it cannot contain another excitation. Far-field
observation directions are part of the fixed output definition rather than a
case parameter. Request every direction needed by the study (for example a
complete angular sweep), then select the appropriate monostatic or bistatic
direction from each case file.




.. note::

    MPI/task-farm studies are not yet enabled. General GPR and SourceStudy
    objects remain main-grid only. Eigenmode studies support the owning main
    grid or subgrid and reset direct and virtual-waveguide modal state
    explicitly. Plane-wave studies use a main-grid TFSF source but may enclose
    complete subgrids.

.. _study-worked-examples:

Worked examples: model to plot
==============================

The small models below demonstrate scheduling, file schemas and plotting,
not converged antenna designs or scattering benchmarks. The port models use
two bare Yee-edge terminals in free space; no commercial antenna geometry
or calibration data are involved. Use a longer record, finer mesh and
larger domain/PML when assessing quantitative antenna performance.

Download the shared :download:`Python model builder
<../../examples/features/studies/run_study.py>` and
:download:`headless result plotter
<../../examples/features/studies/plot_results.py>`.
Run the commands below from the repository root. The Python builder accepts
``--gpu 0`` for CUDA device zero; the hash command equivalent is ``-gpu 0``.

.. _terminal-port-example:

Passive receiving voltage port
------------------------------

:download:`passive.in <../../examples/features/studies/passive.in>`
runs once, without a study. Both gaps have 50 Ohm resistance and a free-space
background; only the receiver's generator waveform has zero amplitude.

.. literalinclude:: ../../examples/features/studies/passive.in
   :language: none

.. code-block:: console

    mkdir -p study_results
    python -m gprMax examples/features/studies/passive.in -o study_results/passive_hash
    python examples/features/studies/run_study.py passive --output study_results/passive_api
    python examples/features/studies/plot_results.py passive study_results/passive_api
    python -m gprMax.toolboxes.Plotting.plot_port study_results/passive_api.h5 --port feed --port receive --validity --save

The comparison PNG plots ``/ports/receive/Vtotal`` against its stored
time axis. Its own source-normalised S11 is invalid, but the received
voltage is useful. See :ref:`passive-receiving-port`.

Fixed-network generator study
-----------------------------

Download :download:`source.in <../../examples/features/studies/source.in>`
and :download:`source.csv <../../examples/features/studies/source.csv>`
into the same directory. Two fixed rational 50 Ohm terminations are driven
first from one end and then with opposite-polarity waveforms:

.. literalinclude:: ../../examples/features/studies/source.csv
   :language: text

.. code-block:: console

    python -m gprMax examples/features/studies/source.in -o study_results/source_hash
    python examples/features/studies/run_study.py source --output study_results/source_api
    python examples/features/studies/plot_results.py source study_results/source_api

Two numbered files contain received voltages and terminal results. The PNG
compares the receiver voltage in both cases. This is not a complete S matrix.
Transmission-line/frill SourceStudy cases use the same scheduling pattern,
with the correct family-specific objects and passive terminations.

Complete voltage-port matrix
----------------------------

Download :download:`port.in <../../examples/features/studies/port.in>`
and :download:`port.csv <../../examples/features/studies/port.csv>`.
Both terminals remain at fixed positions and resistance:

.. literalinclude:: ../../examples/features/studies/port.csv
   :language: text

.. code-block:: console

    python -m gprMax examples/features/studies/port.in -o study_results/port_hash
    python examples/features/studies/run_study.py port --output study_results/port_api
    python examples/features/studies/plot_results.py port study_results/port_api

The two case files are supplemented by ``port_api_study.h5``. The plotter
selects matrix elements using ``port_ids``, masks them with ``valid_S``,
and plots all four magnitudes. The complete builder is:

.. literalinclude:: ../../examples/features/studies/run_study.py
   :language: python
   :pyobject: build_model

Plane-wave incidence and RCS
----------------------------

Download :download:`plane_wave.in <../../examples/features/studies/plane_wave.in>`
and :download:`plane_wave.csv <../../examples/features/studies/plane_wave.csv>`.
A small PEC sphere is illuminated from +x and +y, with fixed -x and -y
observation requests:

.. literalinclude:: ../../examples/features/studies/plane_wave.csv
   :language: text

.. code-block:: console

    python -m gprMax examples/features/studies/plane_wave.in -o study_results/plane_wave_hash
    python examples/features/studies/run_study.py plane_wave --output study_results/plane_wave_api
    python examples/features/studies/plot_results.py plane_wave study_results/plane_wave_api

The plot compares the two appropriate backscatter observations at 5 GHz:
``back_x`` from case 1 and ``back_y`` from case 2. Both requests are
stored in both files; the unused request in each case is bistatic. Read
``ntff/surface/frequency/band/frequencies`` and
``far_field/<id>/fields/rcs`` within that transform, not a port FFT axis.

GPR acquisitions and multimode/array workflows
----------------------------------------------

The :ref:`CSV B-scan tutorial <bscan_csv_study>` provides a full point-source
acquisition, merge and plot workflow. Its Python equivalent can load the
same schedule using ``Study.from_csv('gpr', csv_path)``.

For complete hash and Python modal models, plots and array-state examples,
use examples 4 and 5 in :doc:`eigenmode_port`. Keep modal coefficient
masks separate from power-wave masks and use the aggregate result for a
complete matrix. The same chapter explains geometry-only mode inspection,
virtual-guide constraints, cutoff and degenerate-mode tracking.
