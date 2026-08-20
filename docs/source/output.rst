.. _output:

************
Model Output
************

Field(s) output
===============

gprMax produces an output file that primarily contains time history data for electromagnetic field outputs (receivers) in the model. The output file has the same name as the input file but with ``.h5`` appended, and therefore uses the widely-supported `HDF5 <https://www.hdfgroup.org/HDF5/>`_ format which was designed to store and organize large amounts of numerical data. There are a number of free tools available to read HDF5 files. Also MATLAB has high- and low-level functions for reading and writing HDF5 files, i.e. ``h5info`` and ``h5disp`` are useful for returning information and displaying the contents of HDF5 files respectively. gprMax includes some Python modules (in the ``toolboxes/plotting`` package) to help you view output data, which are documented in the README.rst file for that package.

File structure
--------------

The output file has the following HDF5 attributes at the root (``/``):

- ``gprMax`` is the version number of gprMax used to create the output
- ``Title`` is the title of the model
- ``Iterations`` is the number of iterations for the time window of the model
- ``nx_ny_nz`` is a tuple containing the number of cells in each direction of the model
- ``dx_dy_dz`` is a tuple containing the spatial discretisation, i.e. :math:`\Delta x`, :math:`\Delta y`, :math:`\Delta z`
- ``dt`` is the time step of the model, i.e. :math:`\Delta t`
- ``srcsteps`` is the spatial increment used to move all sources between model runs.
- ``rxsteps`` is the spatial increment used to move all receivers between model runs.
- ``nsrc`` is the total number of sources in the model.
- ``nrx`` is the total number of receievers in the model.
- ``nports`` is the number of voltage-source S11/impedance outputs.
- ``neigenmodeports`` is the number of eigenmode source/receiver port
  monitors.
- ``SourceExcitationSchemaVersion`` identifies the schema used for exact
  source histories stored below local source groups.
- ``ReceiverTimingSchemaVersion`` identifies the schema used for receiver
  dataset time offsets.

The output file contains HDF5 groups for sources (``srcs``), transmission lines
(``tls``), magnetic frill sources (``frills``), receivers (``rxs``),
voltage-source ports (``ports``), SAR outputs (``sar``), radiometric
absorption outputs (``radiometry``), and KSIR outputs (``ntff``) when
requested. Eigenmode sources and receivers add ``eigenmode_ports``. Within these are
further groups for each named or numbered output.

.. _sar-output:

Absorbed power density and specific absorption rate
---------------------------------------------------

A requested main-grid SAR output adds ``/sar/<id>``; an output attached to a
subgrid adds ``/subgrids/<subgrid ID>/sar/<id>``. gprMax transforms only the unique
Yee electric edges required by the selected tagged cells and only at the
requested frequencies; it does not store a full time history for every cell.
The serial CPU solver uses a Cython/OpenMP sparse transform. CUDA, OpenCL, and
Metal accumulate the same sparse transforms in device memory and download only
the completed complex edge spectra. Local SAR and any requested spatial mass
averages are then evaluated on the host using the common implementation.
With MPI domain decomposition, every rank transforms only the sparse electric
edges needed near its selected tagged cells. After time stepping, uniquely
owned edge spectra and owned-cell metadata are gathered once. The coordinator
then collocates the global Yee-edge spectra, applies source or port
normalisation, and performs any mass averaging on the complete tag volume.
This avoids additional collectives inside the timestep loop and permits both
tag regions and averaging cubes to cross rank boundaries. It also avoids
depending on diagonal MPI halo values, which the field solver does not require
and therefore does not exchange.
The transform uses the engineering convention

.. math::

   \tilde{E}(f)=\int E(t)\exp(-j2\pi ft)\,\mathrm{d}t.

The three-dimensional Yee geometry is retained in both 3-D and reduced 2-D
models. In 3-D, the four parallel Yee edges of each electric component are
averaged as complex phasors to the cell centre. A 2-D TM model uses only the
electric component parallel to the invariant axis, at invariant index zero;
its four edges surround the cell in the transverse cross-section. A 2-D TE
model uses the two electric components tangential to the invariant axis on
the live central plane at invariant index one. Each TE component is averaged
from its two genuine tangential edges. The second TE cell exists only to
preserve the 3-D Yee staggering and is not a second physical SAR plane. Thus,
for example,

.. math::

   \begin{aligned}
   E_{z,c}^{\mathrm{TMz}} &= \frac{1}{4}
       \left[E_z(i,j,0)+E_z(i+1,j,0)+E_z(i,j+1,0)+E_z(i+1,j+1,0)\right],\\
   E_{x,c}^{\mathrm{TEz}} &= \frac{1}{2}
       \left[E_x(i,j,1)+E_x(i,j+1,1)\right],\\
   E_{y,c}^{\mathrm{TEz}} &= \frac{1}{2}
       \left[E_y(i,j,1)+E_y(i+1,j,1)\right].
   \end{aligned}

The corresponding coordinate permutations apply to invariant x and y
models. All inactive components remain zero. Two-dimensional TE/TM FDTD SAR
is useful for rapid invariant-model dosimetry and polarisation studies
[LWI2019]_. With peak (not RMS) phasors, the local time-average absorbed
power density and SAR are [IEEE62704-3]_

.. math::

   \begin{aligned}
   p_{\mathrm{abs}}(f) &= \frac{1}{2}\sigma_{\mathrm{eff}}(f)
       \left(|E_x(f)|^2+|E_y(f)|^2+|E_z(f)|^2\right),\\
   \mathrm{SAR}(f) &= \frac{p_{\mathrm{abs}}(f)}{\rho},\\
   \sigma_{\mathrm{eff}}(f) &= -\omega\epsilon_0
       \operatorname{Im}\{\epsilon_r(f)\}.
   \end{aligned}

Thus ``absorbed_power_density`` is in W/m\ :sup:`3`, ``density`` is in
kg/m\ :sup:`3`, and ``sar`` is in W/kg. This evaluates dielectric/conductive
loss directly and does not explicitly form a current-density array. The cell
material supplies :math:`\sigma_{\mathrm{eff}}` and :math:`\rho`; dielectric
smoothing of Yee-edge update coefficients does not average mass density or
tag membership.

Because a finite transient spectrum is not itself a continuous-wave
excitation, waveform normalisation divides the field DFT by the DFT of the
selected source waveform and multiplies it by ``TargetAmplitude``. Exactly
one active source may use that waveform. For a Hertzian electric dipole,
``current_moment`` divides by the product of waveform spectrum and source-edge
length, so ``TargetAmplitude`` is :math:`I\ell` in A m. For a discrete plane
wave, the source waveform is the incident electric-field magnitude in V/m.
It can either be used directly or converted to an incident power flux using

.. math::

   S_{\mathrm{inc}}(f)=\frac{1}{2}|E_{\mathrm{inc}}(f)|^2
       \operatorname{Re}\!\left\{\frac{1}{\eta(f)}\right\},
   \qquad
   \eta(f)=\sqrt{\frac{\mu(f)}{\epsilon(f)}}.

The ``incident_flux`` mode scales to ``TargetFlux`` in W/m\ :sup:`2`; the
complex permittivity and permeability of the plane wave's source-side
material are used at every requested frequency. Alternatively, incident- or
accepted-power normalisation multiplies the field DFT by

.. math::

   C_P(f)=\sqrt{\frac{P_{\mathrm{target}}}{P_{\mathrm{port}}(f)}}.

The requested port power is obtained through the same terminal/modal power
adapter used for antenna gain. Non-positive, non-finite, or invalid port
powers produce invalid SAR bins rather than unbounded scaling. Frequencies
below ``SourceFloorDB`` are likewise marked invalid and stored as NaN.
Source and port normalisation are resolved across the complete model rather
than only the grid containing the SAR output. Consequently, a source on the
main grid can illuminate and normalise tagged tissue sampled at every fine
timestep inside a subgrid. A subgrid port uses the model-wide qualified
reference ``<subgrid ID>/<port ID>``.

The datasets include ``frequency``, ``cell_indices``, ``tag_id``,
``material_id``, ``density``, ``source_spectrum``, ``source_relative_db``,
``source_valid``, ``incident_power``, ``incident_flux``, ``mesh_valid``,
``valid``, ``cells_per_wavelength``,
``limiting_material``, ``absorbed_power_density``, and ``sar``. Each
``tags/<name>`` subgroup contains cell count, integrated mass and absorbed
power, mass-average SAR, and peak voxel SAR.
``CellIndexOrigin`` and ``CellCentreOffset`` map the integer
``cell_indices`` to physical cell-centre positions. ``CellIndexFrame`` is
``main-grid`` for a normal output and ``subgrid-local`` for an HSG output.
MPI output uses globally sorted main-grid cell indices and the same HDF5 schema
as the serial solver. The gathered sparse spectra and completed frequency-by-
cell arrays reside on the coordinator during finalisation; for extremely large
tag volumes this coordinator memory is the present scalability limit.

In a 2-D model, local ``absorbed_power_density`` and ``sar`` retain their
ordinary W/m\ :sup:`3` and W/kg units. Integrated quantities describe the
invariant geometry per unit length:

.. math::

   P'_{\mathrm{abs}}(f) = \int_A p_{\mathrm{abs}}(f)\,\mathrm{d}A,
   \qquad
   m' = \int_A \rho\,\mathrm{d}A,
   \qquad
   \overline{\mathrm{SAR}} = \frac{P'_{\mathrm{abs}}}{m'}.

Consequently, a 2-D tag subgroup contains ``MassPerLength`` in kg/m and
``absorbed_power_per_length`` in W/m. Its ``mass_average_sar`` remains in
W/kg. ``Dimensionality``, ``ModelMode``, ``Polarisation``, ``InvariantAxis``,
``LiveInvariantIndex``, and ``ActiveElectricComponents`` make the convention
explicit. Port powers used for 2-D normalisation are likewise interpreted in
W/m rather than W.

The TMz and TEz cell collocation, local SAR, and absorbed power per unit
length are compared with exact lossy-cylinder series in
:ref:`sar-2d-cylinder-validation`.

Spatial averaging is not performed by default. This keeps the application-
neutral local quantities available without imposing a bioelectromagnetic
1 g/10 g convention or its potentially substantial post-processing cost.
When one or more target masses are explicitly requested, the output also
writes ``spatial_average/<mass>g``. The standard 1 g and 10 g outputs are
requested with ``averaging_masses=(0.001, 0.01)``. They use the two-step
cubical mass-averaging procedure of IEC/IEEE 62704-1 [IEEE62704-1]_: first a
cube is expanded symmetrically about each tissue voxel; remaining boundary
voxels are then evaluated with six face-centred cubes. During centred-cube
growth every complete voxel shell is checked to ensure that each of its six
faces touches or intersects tissue. A cell is marked as covered when more
than 99.9% of its volume lies in a valid centred cube. Fractional
boundary-voxel volumes are used to reach the target mass. Density is
piecewise constant within each tagged FDTD cell: a fractional contribution
has mass

.. math::

   \Delta m_i = q_i\,\rho_i\,\Delta x\Delta y\Delta z,

where :math:`0\leq q_i\leq1` is only the geometrical fraction inside the
averaging cube. Density is not interpolated or Yee-edge averaged. The selected
tags collectively define tissue; cells outside them are background. Each mass
group contains ``sar``, ``status``, ``averaging_mass``,
``averaging_volume``, ``orientation``, ``peak_sar``, and ``peak_cell``.
Status values are 0 invalid/background, 1 unused boundary, 2 covered by a
valid centred cube, and 3 a valid centred-cube location. Orientation 7 is a
centred cube and 1--6 identify the face-centred directions.

The density-, tag-, and grid-dependent averaging-cube geometry is constructed
once for each requested mass and reused for every frequency. The compiled
OpenMP implementation therefore avoids repeating the expensive target-mass
search for a frequency sweep.

Spatial 1 g/10 g averaging is currently restricted to 3-D models. A reduced
2-D field solution is invariant, so such an average would require a deliberate
virtual extrusion of the density, tags, and fields along the invariant axis;
the one-cell TM and two-cell TE storage thicknesses must not be treated as a
finite anatomical volume. Until that invariant-extrusion procedure is added
and separately validated, a 2-D ``SAR`` request must leave
``averaging_masses=()``.

Cells inside either a domain-boundary PML or an internal PML slab are always
excluded, even when their geometry tag is selected. PML attenuation is a
numerical coordinate stretching and must not be counted as physical material
absorption, tissue mass, or SAR. The group attributes ``PMLCellPolicy`` and
``ExcludedPMLCellCount`` record this operation.

The requested frequencies must be below temporal Nyquist. By default they
must also have at least ten cells per shortest wavelength in every model
material; a lambda/8 criterion may be selected. The ``nyquist`` research mode
retains frequencies outside that spatial criterion and records the mesh
validity metadata, but does not make those results physically reliable.

.. _radiometry-output:

Radiometric absorption weighting
--------------------------------

A ``Radiometry`` request writes ``/radiometry/<id>`` on the main grid, or
``/subgrids/<subgrid ID>/radiometry/<id>`` on a subgrid. It uses the same
sparse electric-field transforms, cell-centre collocation,
:math:`\sigma_{\mathrm{eff}}`, source/port normalisation, PML exclusion, and
mesh-validity checks as SAR. It deliberately does not use mass density and
does not calculate SAR or a 1 g/10 g average. This makes it suitable for
lossy geological and planetary materials for which density is unavailable or
irrelevant to the electromagnetic absorption calculation.

The principal local quantities are ``absorbed_power_density`` and
``normalised_absorption_density``. The latter has a definition appropriate to
the selected excitation:

.. math::

   w(\mathbf r,f)=
   \begin{cases}
   p_{\mathrm{abs}}(\mathbf r,f)/P_{\mathrm{ref}},
       & \text{incident or accepted port power},\\
   p_{\mathrm{abs}}(\mathbf r,f)/S_{\mathrm{inc}},
       & \text{incident plane-wave flux},\\
   p_{\mathrm{abs}}(\mathbf r,f)/|q_{\mathrm{src}}|^2,
       & \text{portless source amplitude}.
   \end{cases}

This is the active-mode reciprocity construction used for microwave
radiometry. At one frequency the cell weight for an incident-power reference
is

.. math::

   C_n(f)=\frac{P_{\mathrm{abs},n}(f)}{P_{\mathrm{inc}}(f)},

and Wu and Nieh's finite-band weighting factor is

.. math::

   \overline{C}_n=\frac{1}{B}
   \int_{f_0-B/2}^{f_0+B/2}
   \frac{P_{\mathrm{abs},n}(f)}{P_{\mathrm{inc}}(f)}\,\mathrm{d}f.

The ``normalised_absorption`` values are the per-frequency integrand; a
receiver-specific quadrature over frequency remains post-processing because
its bandshape is not an FDTD property. The equivalent continuous weighting
function commonly used in radiometric forward models is

.. math::

   W(\mathbf r,f)=\frac{p_{\mathrm{abs}}(\mathbf r,f)}
   {\int_V p_{\mathrm{abs}}(\mathbf r,f)\,\mathrm{d}V},
   \qquad
   p_{\mathrm{abs}}=\frac{1}{2}\sigma_{\mathrm{eff}}|\mathbf E|^2,

which integrates to unity over the chosen receiving volume [WU1995]_ and
[ROD2013]_. It can be formed from
``normalised_absorption_density`` without rerunning the model. Normalising to
accepted rather than incident power removes mismatch loss from the spatial
weight but does not otherwise change its shape.

For a 3-D power-normalised port, :math:`w` has units m\ :sup:`-3` and its
volume integral is the fraction of the reference power absorbed in a tag. For
plane-wave flux, it has units m\ :sup:`-1` and its volume integral is an
absorption cross section in m\ :sup:`2`. A 2-D invariant model instead
integrates over area and reports, respectively, a dimensionless absorbed
fraction or an absorption cross section per unit invariant length in metres.
For a portless source the integral is absorbed power per squared native source
amplitude; examples include V/m for a plane-wave waveform, A m for Hertzian
current moment, and V for a voltage source.

Each ``tags/<name>`` subgroup stores the integrated physical absorbed power
and ``normalised_absorption``. These tag-resolved weighting quantities are the
electromagnetic part needed by radiometer forward models. gprMax does not yet
combine them with physical temperature, receiver bandwidth, receiver noise,
or a radiative-transfer model to produce brightness or antenna temperature;
those are separate measurement-model inputs rather than FDTD field
quantities.

Reusable parameter studies add a root ``study`` group. Its attributes are
``Type``, ``CaseID``, ``CaseIndex`` (one based), ``CaseCount``,
``GeometryReused``, and, for CSV studies, ``SourcePath``. The ``source``
dataset preserves the complete CSV text. ``definition`` stores the normalised
full study as JSON, while ``resolved_case`` contains the post-validation values
actually applied to every study-managed object, including baseline receivers
and implicitly disabled sources.
Study-managed source and receiver groups also carry a ``StudyID`` attribute,
including the ``tls`` and ``frills`` groups used by a source study. This
provides an unambiguous link back to the case table.

For a plane-wave study, ``resolved_case`` records the requested case together
with the actual rationalised propagation angles and integer DPW mapping. Each
case remains an ordinary numbered model output; no aggregate matrix is needed
because RCS patterns from different illuminations are independent results.

For a source study, ``resolved_case`` records every stateful terminal,
including sources omitted from the sparse case table. ``active=false`` and
``scale=0`` identify a passive physical termination rather than an object
removed from the model. Transmission-line, magnetic-frill, rational-network,
receiver, port, and declarative NTFF datasets in each numbered file contain
only that case's reset state.

Every case in a finite-resistance voltage-port study additionally contains
``study/port_response``. ``DrivenSourceID`` and ``DrivenPortID`` identify its
input port; ``S_source_column`` contains the power-normalised raw source-plane
response for every output port. ``gap_correction_c`` stores the dimensionless
parallel-gap admittance :math:`Z_{0,p}Y_{\mathrm{gap},p}` used by the matrix
correction.

After all requested cases finish, gprMax writes ``<output>_study.h5``. Its
principal datasets are:

- ``frequency``: common frequency axis;
- ``port_ids`` and ``source_ids``: output-port and deterministic source order;
- ``case_ids``: case contributing each input-port column;
- ``reference_impedance``: positive real wave-reference impedance per port;
- ``gap_correction_c``: numerical gap shunt correction for every frequency and
  port;
- ``S_source``: uncorrected source-plane S matrix;
- ``S``: matrix after removing all numerical gap admittances;
- ``valid_S_source`` and ``valid_S``: element validity masks.

Both matrices use axis order
``S[frequency, output_port, input_port]``. The ``Complete`` attribute states
whether every input-port column is present. A restarted study loads compatible
columns from an existing aggregate file and replaces the newly evaluated
columns; an incompatible existing file is rejected rather than mixed with new
results.

Every eigenmode-study case instead contains
``study/eigenmode_response``. ``InputPort`` and ``InputMode`` identify the
nominal incident modal channel. ``incident`` and ``outgoing`` contain the
complete measured modal vectors for that run, ordered by ``channel_ports``
and ``channel_modes``; ``valid_wave`` and ``generalized_valid_wave`` are their
validity masks. ``S_column`` is the single-source-normalized per-run
diagnostic retained alongside those raw waves. It is not the authoritative
column when passive ports have residual incident waves.

The eigenmode aggregate ``<output>_study.h5`` contains ``frequency``,
``channel_ports``, ``channel_modes``, ``case_ids``, ``incident_matrix``,
``outgoing_matrix``, ``valid_wave_matrix``,
``generalized_valid_wave_matrix``, ``deembedding_condition_number``,
``deembedding_valid``, ``S``, ``valid_S``, and ``generalized_valid_S``. Its convention is
``S[frequency, output_channel, input_channel]``. Modal channels can represent
different mode counts on different ports; the two channel-index datasets are
therefore authoritative and should be used instead of assuming one mode per
port. The aggregate :math:`S` satisfies :math:`B=SA` for the stored measured
incident matrix :math:`A` and outgoing matrix :math:`B`; it is calculated by
a conditioned solve rather than independent scalar division. A restarted
study retains the raw case columns and repeats this complete solve.
Both run matrices use
``[frequency, modal_channel, excitation_case]`` axis order.

When an array codebook is attached, the same aggregate may also contain:

.. code-block:: none

    /array_codebook/
        definition [canonical JSON, always present]
        source [original JSON text, when loaded from disk]
    /embedded_far_fields/<transform_id>/<output_id>/
        frequency
        theta
        phi
        Etheta                         [frequency, direction, channel]
        Ephi                           [frequency, direction, channel]
        valid                          [frequency, channel]
        sphere/
            theta
            phi
            weights
            Etheta                     [frequency, sphere_direction, channel]
            Ephi                       [frequency, sphere_direction, channel]
        raw_runs/
            Etheta                     [frequency, direction, excitation_case]
            Ephi                       [frequency, direction, excitation_case]
            valid                      [frequency, excitation_case]
            sphere/{Etheta,Ephi}
    /array_states/<state_id>/
        drives/{port,mode,power_w,phase_deg,delay_s}
        incident
        outgoing
        active_reflection
        incident_power
        reflected_power
        accepted_power
        tarc
        valid
        far_fields/<transform_id>/<output_id>/
            Etheta
            Ephi
            radiation_intensity
            radiated_power
            directivity
            directivity_dbi
            gain
            gain_dbi
            realized_gain
            realized_gain_dbi
            radiation_efficiency
            total_efficiency
            maximum_directivity
            maximum_directivity_dbi
            maximum_theta
            maximum_phi
            valid

The embedded fields have units of range-normalized field per square root watt
of incident modal power. The requested-direction and full-sphere arrays retain
complex fields, rather than per-channel directivity or power, so coherent
cross terms are included when a state is synthesized. The sphere ``weights``
integrate radiation intensity over solid angle. ``valid`` is false wherever
an active input does not have a physical propagating power-wave S coefficient;
such bins are stored as NaN for power-derived state results.
``definition`` is a complete portable representation even when the codebook
was constructed with the Python API; ``source`` preserves the user's original
text when it came from a JSON file.

.. code-block:: none

    /
        study/ [optional]
            source [CSV studies]
            definition
            resolved_case
            port_response/ [port studies]
                port_ids
                source_ids
                frequency
                reference_impedance
                gap_correction_c
                S_source_column
                valid_S_source_column
            eigenmode_response/ [eigenmode studies]
                channel_ports
                channel_modes
                frequency
                S_column
                valid_S_column
                generalized_valid_S_column
                incident
                outgoing
                valid_wave
                generalized_valid_wave
        rxs/
            rx1/
                Name
                Position
                Ex
                Ey
                Ez
                Hx
                Hy
                Hz
                Ix [optional]
                Iy [optional]
                Iz [optional]
            rx2/
                ...
        srcs/
            src1/
                Type
                Position
                ID
                GridPosition
                Polarisation
                excitation/
                    samples
            src2/
                ...

        tls/
            tl1/
                Position
                Resistance
                dl
                Vinc
                Iinc
                Vtotal
                Itotal
                frequency
                S11
                Zin
                Yin
                S11_current
                Zin_current
                ...
            tl2/
                ...
        frills/ [optional]
            frill1/
                Position
                Polarisation
                Z0
                Mirror1Face
                Mirror2Face
                Mirror1
                Mirror2
                Vinc
                Vtotal
                Itot
                frequency
                S11
                Zin
                Yin
                ...
            frill2/
                ...
        ntff/ [optional]
            <monitor name>/
                ...
        ports/ [optional]
            <port ID>/
                frequency
                S11
                Zin
                Yin
                ...
        eigenmode_ports/ [optional]
            port1/
                frequency
                incident
                outgoing
                electric_cross_power_matrix
                power_matrix
                anchor_mode_valid
                anchor_mode_reference_valid
                anchor_mode_propagating
                anchor_balanced_power
                decomposition_valid
                generalized_valid
                power_normalization_valid
                power_matrix_valid
                valid
                condition_number
                S
                generalized_valid_S
                valid_S
                power_wave_valid_S

Within each individual ``rx`` group are the following attributes:

* ``Name`` is the name of the receiver if specified. Otherwise 'Rx(x,y,z)', where x,y,z is the position of the receiver, is used.
* ``Position`` is the x, y, z position (in metres) of the receiver in the model.
* ``GridPosition`` is the integer x, y, z position of the receiver on its
  owning grid.

Within each individual ``rx`` group can be the following datasets:

* ``Ex`` is an array containing the time history (for the model time window) of the values of the x component of the electric field at that receiver position.
* ``Ey`` is an array containing the time history (for the model time window) of the values of the y component of the electric field at that receiver position.
* ``Ez`` is an array containing the time history (for the model time window) of the values of the z component of the electric field at that receiver position.
* ``Hx`` is an array containing the time history (for the model time window) of the values of the x component of the magnetic field at that receiver position.
* ``Hy`` is an array containing the time history (for the model time window) of the values of the y component of the magnetic field at that receiver position.
* ``Hz`` is an array containing the time history (for the model time window) of the values of the z component of the magnetic field at that receiver position.
* ``Ix`` is an optional array containing the time history (for the model time window) of the values of the x component of current (calculated around a single cell loop) at that receiver position.
* ``Iy`` is an optional array containing the time history (for the model time window) of the values of the y component of current (calculated around a single cell loop) at that receiver position.
* ``Iz`` is an optional array containing the time history (for the model time window) of the values of the z component of current (calculated around a single cell loop) at that receiver position.

Every receiver component dataset has ``SampleInterval``, ``TimeSampleOffset``,
and ``Quantity`` attributes. Electric fields have ``TimeSampleOffset=0`` and
represent :math:`E^n`. Magnetic fields and magnetic-loop currents have
``TimeSampleOffset=-\Delta t/2`` and represent :math:`H^{n-1/2}` at stored
sample index :math:`n`. These explicit physical times are important when
combining electric and magnetic quantities or deconvolving a source.

Within each individual ``src`` group are the following attributes:

* ``Type`` is the type of source, e.g. Hertzian dipole, voltage source etc...
* ``Position`` is the x, y, z position (in metres) of the source in the model.
* ``ID``, ``GridPosition``, and ``Polarisation`` identify the source and its
  electric or magnetic Yee component.

Each supported local source also contains an ``excitation`` group. Its
``samples`` dataset is the exact scalar history consumed by the solver, with
the update look-ahead sample removed. ``SampleInterval`` and
``TimeSampleOffset`` define the physical time of sample :math:`n` as
:math:`t_n=n\Delta t+t_0`; ``DrivingQuantity``, ``Units``, ``SpatialScale``,
``UpdateLattice``, and waveform attributes describe the excitation. A
``WaveformEvaluationTimeOffset`` attribute separately records the time used
to evaluate the waveform function for source sample :math:`n`. A
resistive voltage source and Hertzian electric dipole, for example, use
:math:`t_0=\Delta t/2`, while a hard voltage source is imposed on
:math:`E^{n+1}` and uses :math:`t_0=\Delta t`. Transmission-line sources store
their two staggered histories as ``samples_whole`` and ``samples_half``;
``samples`` is a non-duplicating HDF5 link to the whole-step generator-voltage
reference.

This timing-aware source schema is used by the
:ref:`impulse-response waveform-synthesis toolbox <impulse_response>`, the
:ref:`SFCW toolbox <sfcw>`, and the :ref:`FMCW toolbox <fmcw>`, and is also
available for other file-based deconvolution and post-processing.

Within each individual ``tl`` group are the following attributes:

* ``Position`` is the x, y, z position (in metres) of the source in the model.
* ``Resistance`` is the resistance of the transmission line.
* ``dl`` is the spatial discretisation of the transmission line.
* ``ReferenceImpedance`` is the real reference impedance used for S11 and is
  equal to ``Resistance``.
* ``SpectrumLimitMode``, ``MinimumWavelengthCells``,
  ``MeshFrequencyLimit``, ``NyquistFrequency``, and ``LimitingMaterial``
  describe the automatically selected frequency band.
* ``ZinPrimaryMethod`` identifies the voltage-wave S11 result as the primary
  impedance calculation. ``CurrentCheckMethod`` identifies the independent
  discrete-line current-wave calculation.

Within each individual ``tl`` group are the following datasets:

* ``Vinc`` is an array containing the time history (for the model time window) of the values of the incident voltage in the transmission line.
* ``Iinc`` is an array containing the time history (for the model time window) of the values of the incident current in the transmission line.
* ``Vtotal`` is an array containing the time history (for the model time window) of the values of the total (field) voltage in the transmission line.
* ``Itotal`` is an array containing the time history (for the model time window) of the values of the total (field) current in the transmission line.
* ``frequency`` is the authoritative non-negative frequency axis in Hz.
* ``Vincident_spectrum``, ``Vreflected_spectrum``, and ``Vtotal_spectrum``
  are the complex voltage spectra.
* ``Iincident_spectrum`` and ``Itotal_spectrum`` are the complex current
  spectra after accounting for the current's half-time-step staggering.
* ``S11`` is the voltage-wave reflection coefficient
  :math:`(V_\mathrm{total}-V_\mathrm{inc})/V_\mathrm{inc}`.
* ``Zin`` and ``Yin`` are the input impedance and admittance derived from
  ``S11`` and the line resistance.
* ``S11_current`` and ``Zin_current`` are independent checks calculated from
  the line current. The current is displaced from the voltage node by both
  half a time step and half a line cell; gprMax removes both offsets using the
  discrete 1D transmission-line dispersion relation. A simple phase shift of
  the total current alone is not sufficient because incident and reflected
  current waves require opposite spatial corrections.
* ``valid_S11``, ``valid_Zin``, ``valid_Yin``, ``valid_S11_current``, and
  ``valid_Zin_current`` are per-bin integer masks. ``source_valid``,
  ``mesh_valid``, ``line_propagation_valid``, ``incident_relative_dB``, and
  ``cells_per_minimum_wavelength`` provide the corresponding diagnostics.

Within each individual ``frill`` group are the following attributes:

* ``Position`` is the x, y, z position (in metres) of the feed point in the model.
* ``Polarisation`` is the antenna axis the source drives current along
  (``x``, ``y``, or ``z``).
* ``Z0`` is the user-supplied characteristic impedance (``Zcoax``), and is
  also the reference impedance used for S11.
* ``InnerConductorRadius`` is the radius :math:`a` obtained from the mandatory
  co-located ``#thin_wire``.
* ``CurrentTimeApproximation`` is ``average`` for Hyun's recommended average
  of the adjacent magnetic half-step currents.
* ``FeedSelfAdmittance`` is the precomputed Cartesian feed-cell coefficient
  :math:`G_f` (Siemens) used to solve the current feedback in closed form.
* ``TimeOffset`` is zero: ``Vinc``, ``Vtotal``, and the averaged ``Itot`` are
  all centred at integer electric-field times.
* ``Mirror1Face`` and ``Mirror2Face`` name the two domain faces transverse to
  ``Polarisation`` (for example ``x0``/``y0`` for a z-polarised source).
  ``Mirror1`` and ``Mirror2`` record whether the feed point actually sits on
  a symmetry-plane corner declared with ``#symmetry_boundary`` at that face.
* ``SpectrumLimitMode``, ``MinimumWavelengthCells``,
  ``MeshFrequencyLimit``, ``NyquistFrequency``, and ``LimitingMaterial``
  describe the automatically selected frequency band, as for ``tl`` groups.
* ``ZinPrimaryMethod`` identifies the voltage-wave S11 result as the primary
  impedance calculation - there is no separate current-based cross-check
  here, unlike a transmission line, since the frill's voltage and current
  histories are solved together at the same instant, not staggered.

Within each individual ``frill`` group are the following datasets:

* ``Vinc`` is an array containing the time history of the incident voltage
  (half the generator waveform).
* ``Vtotal`` is an array containing the time history of the total terminal
  voltage :math:`V_\mathrm{ab}`.
* ``Itot`` is an array containing the time history of the total current at
  the feed point, averaged from the adjacent magnetic half steps and
  generalised for a symmetry-mirrored feed point.
* ``frequency``, ``S11``, ``Zin``, ``Yin``, and their associated
  ``valid_S11``/``valid_Zin``/``valid_Yin``/``source_valid``/``mesh_valid``/
  ``incident_relative_dB``/``cells_per_minimum_wavelength`` diagnostics have
  the same meaning as the corresponding ``tl`` datasets.

Transmission-line S11 and impedance output
-------------------------------------------

S11, input impedance, and input admittance are generated automatically for
every transmission-line source; no additional receiver is required. With the
real line resistance :math:`Z_0` as the
reference impedance, the primary results are

.. math::

    S_{11}=\frac{V_\mathrm{total}-V_\mathrm{inc}}{V_\mathrm{inc}},
    \qquad
    Z_\mathrm{in}=Z_0\frac{1+S_{11}}{1-S_{11}},
    \qquad
    Y_\mathrm{in}=\frac{1-S_{11}}{Z_0(1+S_{11})}.

This S11-based ``Zin`` is the primary result because direct division by the
terminal current becomes ill-conditioned at frequencies where that current
approaches zero. The independently de-embedded current result remains in
``Zin_current`` so that the line coupling and the voltage-wave result can be
checked. Algebraically undefined open- or short-circuit quantities are stored
as complex NaNs and identified by their validity masks.

The stored spectrum uses the native resolution :math:`\Delta f=1/T` and is
capped automatically at the lambda/10 mesh limit of the most restrictive
material in the model. The original ``Vinc``, ``Iinc``, ``Vtotal``, and
``Itotal`` histories remain available for users who need to reprocess a wider
research band.

For example, the valid S11 and impedance bins can be read directly:

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File('model.h5', 'r') as output:
        line = output['tls/tl1']
        frequency = line['frequency'][...]
        s11 = line['S11'][...]
        zin = line['Zin'][...]
        valid = (
            line['valid_S11'][...].astype(bool)
            & line['valid_Zin'][...].astype(bool)
        )

    s11_db = 20 * np.log10(np.abs(s11[valid]))
    resistance = zin.real[valid]
    reactance = zin.imag[valid]

Magnetic-frill-source S11 and impedance output
-----------------------------------------------

S11, input impedance, and input admittance are generated automatically for
every ``#magnetic_frill_source``, exactly as for a transmission line; no
additional receiver is required (the electric field
component along the polarisation axis is identically zero at the feed point
by construction, so no field sample is possible there in the first place).
Main-grid results are stored at ``/frills/frillN``. A Python API frill inside
a subgrid is stored at ``/subgrids/<subgrid ID>/frills/frillN``; its
``Position`` is reported in global physical coordinates, while its histories,
frequency axis, and validity limits use the owning subgrid's finer ``dt`` and
``dl``.
With the frill's user-supplied characteristic impedance :math:`Z_0` as the
reference impedance,

.. math::

    S_{11}=\frac{V_\mathrm{ab}-V_\mathrm{inc}}{V_\mathrm{inc}},
    \qquad
    Z_\mathrm{in}=Z_0\frac{1+S_{11}}{1-S_{11}},
    \qquad
    Y_\mathrm{in}=\frac{1-S_{11}}{Z_0(1+S_{11})}.

Unlike a transmission line, there is no independent current-based
cross-check: :math:`V_\mathrm{inc}`, :math:`V_\mathrm{ab}`, and
:math:`I_\mathrm{tot}` are centred at the same integer time. The current is
Hyun's average of the preceding and following magnetic half-step values; the
following value depends on the voltage applied in that update, so gprMax
solves the resulting feed-cell relation analytically. There is therefore no
remaining leapfrog phase shift to de-embed. :math:`I_\mathrm{tot}` is formed
directly from the mandatory thin wire's stored Mäkinen-projected H edges,
consistent with the :math:`F k_H` source deposit. At every sample the stored
histories satisfy

.. math::

    V_\mathrm{ab}=2V_\mathrm{inc}-Z_0 I_\mathrm{tot}.

Set ``spectrum_limit`` directly on the magnetic-frill source when a limit
other than the default lambda/10 band is required.

Rational-network port output
----------------------------

The ``#network_port`` command and ``NetworkPort`` Python object write a
rational terminal beneath ``/ports/<terminal ID>``. The external-network
current is defined positive from the FDTD gap into the network,

.. math::

    I_N(s)=Y(s)\left[V(s)-V_g(s)\right],
    \qquad
    Y(s)=G+sC+\sum_m\frac{r_m}{s-p_m}.

Here :math:`V_g=0` for a passive terminal. For antenna and accepted-power
calculations, the current entering the electromagnetic structure removes the
numerical parallel Yee-gap admittance,

.. math::

    I_\mathrm{terminal}=-I_N-Y_\mathrm{gap}V,
    \qquad
    V^\pm=\frac{V\pm Z_0I_\mathrm{terminal}}{2}.

For each rational term, :math:`\dot{x}_m=p_mx_m+r_mu`, where
:math:`u=V-V_g`. Its value at any fraction :math:`\theta` of a time step is
calculated without a finite-difference approximation to :math:`x_m`,

.. math::

    x_m^{n+\theta}=e^{\theta p_m\Delta t}x_m^n
    +a_{m,\theta}u^{n+1}+b_{m,\theta}u^n,

with the coefficients obtained from the exact convolution of an exponential
pole with a linearly varying voltage. The implementation uses
:math:`\theta=1/2` directly for the locally implicit Ampere update and
:math:`\theta=1` to advance stored state. This is the relevant
Giannakis--Giannopoulos improvement [GIA2014]_ to the classic PLRC
lumped-network treatment of [PER1999]_ and [CHE2007]_: the half-step pole
current is obtained directly rather than by averaging adjacent integer-time
values.

The reported quantities are :math:`S_{11}=V^-/V^+`,
:math:`Z_\mathrm{in}=V/I_\mathrm{terminal}`, and
:math:`Y_\mathrm{in}=I_\mathrm{terminal}/V`. Their validity is recorded
separately, so a passive receiving port may retain valid impedance while its
source-normalised S11 is invalid.

The group stores ``poles`` and ``residues`` together with ``Conductance`` and
``Capacitance``; time histories ``Vgenerator``, ``Vtotal``, and ``Inetwork``;
frequency-domain ``Vgenerator_spectrum``, ``Vtotal_spectrum``,
``Inetwork_spectrum``, ``Iterminal_spectrum``, ``Vincident_spectrum``, and
``Vreflected_spectrum``; and the usual ``S11``, ``Zin``, ``Yin``, mesh,
source, gap, and cells-per-wavelength validity datasets. The transform uses
the engineering convention and all three histories are sampled at the
electric half-step. ``NetworkModelID`` identifies the reusable model and
``PortMode`` is ``rational_network``.

The complete production update is compared with the exact reflection from
series-RC and series-RLC sheets in a parallel-plate guide in
:ref:`rational-network-validation`.

A terminal owned by a subgrid is stored beneath
``/subgrids/<subgrid ID>/ports/<terminal ID>`` using the subgrid's fine time
and space increments.


Voltage-source S11 and impedance output
---------------------------------------

Every 3-D ``#voltage_source`` or ``VoltageSource`` Python object writes one
source-owned group per main-grid port at ``/ports/<port ID>``. A source added
to a Python API subgrid is
stored at ``/subgrids/<subgrid ID>/ports/<port ID>`` and uses the owning
subgrid's spatial step, time step, iteration count, material edge, and field
histories. For a finite-resistance source, its resistance is the reference
impedance :math:`Z_0`; the source-plane reflection coefficient is calculated
from the known generator voltage and sampled total gap voltage.
In a domain-decomposed MPI model the same schema is written after source,
receiver, and port histories have been gathered on the coordinator. All
``Position`` and ``GridPosition`` metadata use the global main-grid frame.
For a zero-resistance hard source, :math:`Z_0` is supplied by the voltage
source (50 Ohms by default) and
the terminal quantities are calculated from the prescribed voltage and
time-centred Ampere-loop current. Both paths remove the effective Yee-edge
background capacitance and conductance before reporting ``S11``, ``Zin``, and
``Yin``:

.. math::

    Z_\mathrm{in}=Z_0\frac{1+S_{11}}{1-S_{11}},
    \qquad
    Y_\mathrm{in}=\frac{1-S_{11}}{Z_0(1+S_{11})}.

For a finite-resistance source in a dispersive background, the shunt
admittance uses the material's complete complex relative permittivity,
including every Debye, Lorentz, Drude, or inclusive pole,

.. math::

    \widetilde{\omega}=\frac{2}{\Delta t}
    \tan\!\left(\frac{\omega\Delta t}{2}\right),
    \qquad
    Y_\mathrm{gap}=j\widetilde{\omega}\varepsilon_0
    \varepsilon_r(\widetilde{\omega})\frac{A}{\Delta l}.

This reduces to :math:`G_\mathrm{bg}+j\widetilde{\omega}C_\mathrm{gap}`
for a nondispersive conductive material. Hard voltage ports use their sampled
Ampere-loop current and are not yet supported on dispersive source edges.

Important attributes include:

* ``PortMode``, ``ReferenceImpedance``, ``ReferenceImpedanceSource``,
  ``Polarisation``, ``Position``, and ``GridPosition``;
* ``BackgroundMaterial``, ``GapCapacitance``, and
  ``BackgroundConductance``;
* ``SpectrumLimitMode``, ``MinimumWavelengthCells``,
  ``MeshFrequencyLimit``, ``NyquistFrequency``, and ``LimitingMaterial``;
* ``FrequencyRange`` (the first and last stored bins),
  ``ValidFrequencyRange`` (a convenience summary), and
  ``IndependentFrequencyResolution``;
* ``phasor_time_sign=exp(+j*omega*t)`` and
  ``forward_transform_sign=exp(-j*omega*t)``.

``Position`` is always expressed in the global physical coordinate frame,
including for a subgrid port, and therefore matches the associated source's
``Position`` attribute. ``GridPosition`` is the integer index in the owning
grid and can include the subgrid's internal boundary padding.

The principal datasets are:

* ``frequency``: the authoritative plotting axis in Hz;
* ``S11``, ``Zin``, and ``Yin``: corrected complex terminal quantities;
* ``S11_source`` and ``Zin_source``: uncorrected source-plane quantities;
* ``Vincident_spectrum``, ``Vreflected_source_spectrum``, and
  ``Vtotal_spectrum``: complex voltage spectra;
* ``time``, ``Vgenerator``, and ``Vtotal``: aligned voltage audit histories;
* ``time_current``, ``Iloop``, ``Iloop_spectrum``, and
  ``Iterminal_spectrum``: additional hard-source current data. ``Iloop`` is
  sampled at magnetic half-step times; its spectrum includes the Yee-gap
  admittance, while the terminal spectrum has that admittance removed;
* ``valid_S11``, ``valid_Zin``, ``valid_Yin``, ``source_valid``,
  ``mesh_valid``, and ``gap_correction_valid``: per-bin integer masks;
* ``incident_relative_dB`` and ``cells_per_minimum_wavelength``: diagnostics
  behind the masks.

The normal output is capped by the requested cells-per-wavelength criterion.
With ``spectrum_limit='nyquist'``, all native non-negative bins are retained
for research, including finite values outside the recommended band, while the
validity masks and lambda/10 advisory ceiling remain present. Algebraically
undefined values are stored as complex NaNs; finite invalid values are not
silently clipped.

The frequency dataset and validity mask can be plotted directly:

.. code-block:: python

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np

    with h5py.File('model.h5', 'r') as output:
        port = output['ports/feed']
        frequency = port['frequency'][...]
        s11 = port['S11'][...]
        valid = port['valid_S11'][...].astype(bool)

    plt.plot(frequency[valid], 20 * np.log10(np.abs(s11[valid])))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'$|S_{11}|$ [dB]')

For routine inspection, the port plotter reads these stored values and their
diagnostics directly, and can save PNG, PDF, or SVG figures without repeating
the terminal calculation:

.. code-block:: console

    python -m toolboxes.Plotting.plot_port model.h5 --port feed --validity --save

The same command supports rational-network, transmission-line, and
magnetic-frill port groups, including ports owned by subgrids. Multiple
``--port`` options may be supplied, or omitted to plot every discovered port.

Eigenmode-port and S-parameter output
-------------------------------------

Each explicitly numbered main-grid eigenmode source or receiver is stored
below ``/eigenmode_ports/portN``. A direct port owned by an HSG subgrid is
stored below ``/subgrids/<subgrid ID>/eigenmode_ports/portN`` and uses that
grid's fine ``dx_dy_dz``, ``dt``, and iteration count. ``frequency`` contains
the direct-DFT bins.
``incident`` and ``outgoing`` have shape ``(nmodes, nfrequencies)``, with
rows ordered by the one-based ``ModeIndices`` attribute. With exactly one
active channel, ``S`` is the outgoing coefficient divided by that source-mode
incident coefficient, so the source group contains modal S11 and other groups
contain S21, S31, and modal-conversion terms.

Each mode has two audited anchor banks. ``anchor_mode_valid`` selects the
propagating, one-watt profiles used by TF/SF source synthesis and real-power
normalization. ``anchor_mode_reference_valid`` selects the successfully
tracked profiles used only by modal monitoring and can include evanescent
anchors. For generalized-only bins, the applicable contiguous evanescent
reference run is converted to a common E/H-balanced scale before its E, H,
and effective index are interpolated with matching branch-local weights.
Propagating and evanescent anchors are never mixed across cutoff, and
disconnected evanescent runs are not bridged. ``anchor_mode_propagating``
records the raw forward-power classification, and ``anchor_balanced_power``
records each raw profile's positive balanced E/H power; its inverse square
root supplies the reference scale.

The ``AnchorFrequencies`` attribute remains the union of the retained
propagating source/power anchors, while ``ReferenceAnchorFrequencies`` is the
union of retained monitor-reference anchors. ``CandidateAnchorFrequencies``
lists every solved candidate; use it together with
``anchor_mode_reference_valid`` to recover the monitor-reference frequencies
for each mode rather than only their union. Non-propagating reference anchors
never drive the source and never become power waves.

The complex ``electric_cross_power_matrix`` and Hermitian ``power_matrix``
both have shape
``(nfrequencies, nmodes, nmodes)``. For a modal coefficient vector
:math:`c`, its spectral power is

.. math::

    P(c)=\operatorname{Re}\{c^\mathrm{H}Wc\}.

This full quadratic form is required when finite-grid modal profiles are not
exactly orthogonal. The incident and outgoing arrays are generalized modal
travelling-wave coefficients; an individual coefficient magnitude squared is
not an additive power when ``power_matrix`` is non-diagonal. The electric
cross-power matrix is retained to reconstruct total-field flux in lossy
ports. ``decomposition_valid`` has shape ``(nfrequencies, nmodes)`` and records
pre-solve reference eligibility. ``generalized_valid`` has shape
``(nmodes, nfrequencies)`` and records conditioned incident/outgoing
coefficients. Legacy ``valid`` has the same latter shape and is the stricter
physical power-wave mask. ``power_normalization_valid`` has shape
``(nfrequencies, nmodes)``, while ``power_matrix_valid`` has shape
``(nfrequencies,)``. The anchor masks have shape ``(ncandidates, nmodes)``.

For scattering output, ``generalized_valid_S``, ``valid_S``, and
``power_wave_valid_S`` all have shape ``(nmodes, nfrequencies)``.
``generalized_valid_S`` identifies conditioned generalized coefficient
ratios, including the source-spectrum check. ``valid_S`` preserves the legacy
physical meaning, and ``power_wave_valid_S`` is its explicit exact alias. A
below-cutoff coefficient can therefore have ``generalized_valid_S=1`` while
``valid_S=power_wave_valid_S=0``. Ill-conditioned or weakly excited bins
remain in the file but must not be plotted as valid results; generalized
coefficients must not be used for power accounting unless the corresponding
physical mask is also true.

A run with one eigenmode source also writes
``<output>_sparameters.csv`` with one row per frequency, destination port,
and destination mode. For a source in a subgrid the filename is
``<output>_<subgrid ID>_sparameters.csv``. Ports in different owning grids are
finalised independently and do not form one cross-grid S matrix.

When every eigenmode port has a passive virtual waveguide and no
``EigenmodeExcitation`` is defined, the same HDF5 groups contain the raw
``incident`` and ``outgoing`` modal spectra. No ``S`` datasets or
``<output>_sparameters.csv`` file are written because there is no active
incident spectrum by which to normalize them.

When two or more modal channels are excited simultaneously, the groups also
contain the raw ``incident`` and ``outgoing`` spectra and have
``ResponseType=driven``. Source groups record ``ExcitationModes``,
``DriveAmplitudes``, ``DrivePowers``, ``DrivePhasesDegrees``, ``DriveDelays``,
and ``DriveWaveformIDs``. No ``S`` datasets or S-parameter CSV are written:
normalizing a superposed response by one incident channel would not be an
independently excited S-matrix column. Use an ``EigenmodeStudy`` for that
matrix.

.. _output-ntff:

NTFF field-transformation output
--------------------------------

Reusable KSIR and equivalent-current outputs are stored in the normal model
file under their surface and transform IDs:

MPI domain-decomposition runs use this same schema and write the completed
datasets from the coordinator rank. A ``collection_backend`` value beginning
with ``mpi_`` records that the integration surface was partitioned between
ranks; it does not change field normalisation or array ordering.

.. code-block:: none

    /ntff/<surface_id>/
        time/<rx_id>/
            points
            times
            time_origins
            valid_lengths
            fully_supported_lengths
            terminal_field_ratios
            terminal_decay_ok
            spherical_coordinates [spherical commands only]
            fields/<output>
        time_far_field/<output_id>/
            times
            theta
            phi
            directions
            terminal_field_ratios
            terminal_decay_ok
            fields/<output>
        frequency/<transform_id>/
            frequencies
            layered_background/ [planar-layered transforms only]
                interfaces
                material_ids
                relative_permittivity
                relative_permeability
            surface_dft/<component>/
                field
                normal_derivative
                patch_positions
                patch_normals
                area_weights
            receivers/<rx_id>/
                points
                spherical_coordinates [spherical commands only]
                fields/<output>
            far_field/<output_id>/
                theta
                phi
                directions
                radiated_power [directivity/efficiency requests]
                maximum_directivity
                maximum_directivity_dbi
                maximum_directivity_theta
                maximum_directivity_phi
                port_power/ [gain/efficiency requests]
                    port_ids
                    source_types
                    representations
                    reference_impedances
                    incident_voltage_per_port
                    terminal_voltage_per_port
                    terminal_current_per_port
                    incident_power_per_port
                    accepted_power_per_port
                    incident_power
                    accepted_power
                    reflected_power
                    incident_relative_db
                    mesh_valid
                    terminal_valid
                    gain_valid
                    realized_gain_valid
                    modal_ports/ [eigenmode ports only]
                        port1/
                            mode_indices
                            incident
                            outgoing
                            electric_cross_power_matrix
                            power_matrix
                            valid
                fields/<output>

The surface group records logical bounds, physical reference origin, closure
status, omitted faces, boundary types/coordinates, and image count.
``closure=huygens_open`` with one to five values in ``omitted_faces`` identifies
an open frequency-domain Huygens surface; ``closure=symmetry`` identifies
faces restored by PEC/PMC image completion.
The frequency transform group records its ``ksir``, ``equivalent_current``,
or ``planar_layered_equivalent_current`` formulation, window, configured
precision and collection backend, plus the engineering convention:
``exp(+j*omega*t)`` phasors, ``exp(-j*omega*t)`` forward transform, and
``exp(-j*k*R)`` outgoing Green function.

For a planar-layered transform, ``layered_background`` records the stack
axis, positive-to-negative-axis material ordering, physical interface
coordinates, and the complex relative permittivity and permeability evaluated
at every transform frequency. The transform-level scalar ``wave_speed`` and
``impedance`` are collection metadata for the positive-axis exterior; the
far-field group uses the appropriate observation half-space for every
direction. Its normalization attribute is therefore
``r * exp(+j*k_observation_halfspace*r) * field``.

Exact frequency receiver groups have ``range_normalized=False``. They contain
physical finite-distance phasors with every ``1/R`` and ``1/R**2`` term.
Far-field groups have ``range_normalized=True`` and a ``normalization``
attribute specifying ``r * exp(+j*k*r) * field``. Their radius is intentionally
absent. Complex datasets use the complex type paired with the configured
gprMax real precision.

The ``time_far_field`` group is produced by ``#ntff_time_far_field`` or its
array form. It records ``formulation=equivalent_current_1997``, linear
fractional-delay interpolation, CPU/Cython collection, and the normalization
``r * field at reduced time t - r/c``. Its real field arrays have shape
``(ndirections, ntimes)``. ``Er`` and ``Hr`` are identically zero in the
far-zone model; magnetic components are derived from
:math:`\mathbf H=(\hat{\mathbf r}\times\mathbf E)/\eta`.
The complete definitions and the distinction between the retained Yee-time
staggering and fractional-delay interpolation are given in
:ref:`ntff-formulations`.

Far-field derived antenna quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``radiation_intensity`` has shape ``(nfrequencies, ndirections)``. If
``directivity`` or an efficiency is requested, the five full-sphere summary
datasets shown above have shape ``(nfrequencies,)``. The far-field group also
records the Gauss--Legendre/periodic quadrature orders and the completed
surface's bounding radius in metres. These summaries come from an internal
full-sphere calculation even if the stored user directions form only a cut;
the internal directional fields are not written to the output file. The
maximum is refined using the directions in that particular output whenever
they sample a larger value than the internal quadrature.

The linear definitions are

.. math::

    U=\frac{|F_\theta|^2+|F_\phi|^2}{2\eta_o(\hat{\mathbf r})},\qquad
    D=\frac{4\pi U}{P_\mathrm{rad}},\qquad
    P_\mathrm{rad}=\int_{4\pi}U\,\mathrm{d}\Omega.

For homogeneous transforms :math:`\eta_o` is constant. For a planar stack it
is the real impedance of the positive- or negative-axis lossless exterior,
selected by the observation direction.

``directivity_dbi``, ``gain_dbi``, and ``realized_gain_dbi`` are
``10*log10`` of their corresponding linear datasets. The linear gain and
efficiency definitions are

.. math::

    G=\frac{4\pi U}{P_\mathrm{acc}},\qquad
    G_\mathrm{realized}=\frac{4\pi U}{P_\mathrm{inc}},\qquad
    \eta_\mathrm{rad}=\frac{P_\mathrm{rad}}{P_\mathrm{acc}},\qquad
    \eta_\mathrm{total}=\frac{P_\mathrm{rad}}{P_\mathrm{inc}}.

The two efficiencies are frequency-only quantities and therefore have shape
``(nfrequencies,)`` even though they are stored in ``fields`` with the other
requested outputs.

For a ``#ksir_antenna_ports`` or ``#ntff_antenna_ports`` association, per-port
powers have shape ``(nports, nfrequencies)``. Total powers and all validity
masks have shape ``(nfrequencies,)``. Eigenmode ports are supported only by
the equivalent-current ``#ntff_antenna_ports`` path; an eigenmode source
cannot be combined with Ramahi/KSIR. Conventional terminal ports use

.. math::

    P_{\mathrm{acc},p}=\frac{1}{2}\Re\{V_p I_p^*\},\qquad
    P_{\mathrm{inc},p}=\frac{|V_p^+|^2}{2Z_{0p}}.

For a modal port with incident and outgoing vectors :math:`a_p` and
:math:`b_p`, define :math:`x_p=a_p+b_p` and :math:`y_p=a_p-b_p`. With
electric cross-power matrix :math:`G^E_p`, the co-located total-field flux
is

.. math::

    P_{\mathrm{acc},p}
      = \Re\{y_p^\mathrm{H}G^E_px_p\}.

The incident power of an eigenmode source is the quadratic power of its
externally excited mode only. A passive eigenmode receiver has zero generator
incident power. Totals are

.. math::

    P_\mathrm{acc}=\sum_p P_{\mathrm{acc},p},\qquad
    P_\mathrm{inc}=\sum_p P_{\mathrm{inc},p},\qquad
    P_\mathrm{refl}=P_\mathrm{inc}-P_\mathrm{acc}.

These are spectral power-normalisation quantities: the voltage and field
Fourier transforms carry a common time scale, which cancels in gain and
efficiency. Their HDF5 attributes consequently give voltage spectra in
``V s``, current spectra in ``A s``, and spectral powers in ``W s**2``. The
exact complex terminal and incident spectra are retained so that every
derived power can be checked independently. For modal ports those terminal
arrays and reference impedance are NaN because no artificial voltage/current
equivalent is introduced; ``representations`` identifies them as
``modal_power_waves`` (the retained schema identifier), and ``modal_ports``
retains the generalized modal amplitudes, electric cross-power matrix, and
Hermitian forward-wave power matrix. Where the real-power masks are true,
modal amplitudes have units ``sqrt(W) s`` and their dimensionless quadratic
matrices produce ``W s**2`` spectral power. A finite coefficient outside that
mask is a generalized reference-basis amplitude only; its magnitude squared
is not power. A zero-amplitude
conventional source remains a terminated port with zero incident voltage; its
terminal voltage/current and signed accepted power can still be non-zero
through mutual coupling.
The ``port_ids`` dataset stores main-grid IDs unchanged and qualifies subgrid
ports as ``<subgrid ID>/<local port ID>``. Although the grouped antenna result
is written below the main-grid NTFF group, each port spectrum is calculated
with the spatial step, time step, and trace length of the grid that owns it.

The legacy name ``terminal_valid`` is also the combined modal-power validity
mask when a modal port is present. The validity datasets should always be
applied before plotting. The default
gain bandwidth includes frequencies whose total incident spectrum is within
``40 dB`` of its peak and for which the mesh and port reconstruction are
valid. Invalid derived results are stored as ``NaN`` rather than a plausible
but ill-conditioned value.

When requested, ``far_field/<output_id>/fields/rcs`` contains real, linear
bistatic radar cross section in square metres. It is not stored in dBsm. For
the range-normalized scattered electric field
:math:`F_\mathrm{s}=r\exp(+jkr)E_\mathrm{s}`, gprMax calculates

.. math::

    \sigma(\theta,\phi,f)
    = 4\pi
      \frac{|F_{\mathrm{s},\theta}|^2+|F_{\mathrm{s},\phi}|^2}
      {|E_{\mathrm{inc},x}|^2+|E_{\mathrm{inc},y}|^2
       +|E_{\mathrm{inc},z}|^2}.

The denominator is obtained from the actual field history in the associated
discrete-plane-wave grid, accumulated at the transform frequencies with the
same time window as the surface data. It therefore includes the numerical
plane-wave amplitude and its configured start and stop times. A zero incident
spectrum produces ``NaN`` RCS. Very small incident values are mathematically
non-zero but can give unreliable results, so the requested frequencies should
remain within the useful excitation bandwidth.

When a frequency transform is associated with a plane wave, its
``plane_wave`` subgroup stores the source index, TFSF corners, waveform and
material IDs, actual angles, polarisation, rational integer mapping, start and
stop times, and the incident-field reference positions. This metadata is the
authoritative description of the illumination used for that transform.

For plotting in dBsm, convert the stored values explicitly:

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File('model.h5', 'r') as output:
        far = output['ntff/surface/frequency/transform/far_field/backscatter']
        rcs = far['fields/rcs'][...]  # linear RCS in m**2
        rcs_dbsm = 10 * np.log10(rcs)

Here ``theta`` and ``phi`` in the same group define the observation direction.
Monostatic RCS is the value in the direction opposite to plane-wave
propagation; all other directions are bistatic RCS. Use separate simulations
for different incident plane waves because selecting an association does not
separate simultaneously accumulated scattered fields. A
:class:`gprMax.PlaneWaveStudy` automates those separate illuminations while
reusing the unchanged Yee geometry.

Exact finite-distance KSIR time-domain fields retain the complete raw retarded
buffer and have shape ``(npoints, max(valid_lengths))``. Its final bins may contain only part of the
closed-surface history because different surface patches have different
propagation delays. For normal use, point ``q`` must therefore be sliced with
``fully_supported_lengths``:

.. code-block:: python

    length = fully_supported_lengths[q]
    physical_time = time_origins[q] + times[:length]
    trace = fields[output][q, :length]

``valid_lengths`` remains available for research access to every stored bin.
Those additional bins are not a complete field reconstruction unless the
surface fields had already decayed to zero. ``terminal_field_ratios`` gives,
for each point, the largest ratio between the final 32 fully supported samples
and the trace peak, evaluated independently for each underlying Cartesian
component. ``terminal_decay_ok`` is true when this is no greater than
``terminal_decay_threshold`` (stored as a group attribute). gprMax emits a
warning when the test fails; increasing the model time window is the correct
remedy.

With ``time_origin=simulation`` every origin is zero. With
``time_origin=first_arrival`` each origin retains its absolute propagation
time without storing the potentially large guaranteed leading-zero prefix.
For this origin policy, ``fully_supported_lengths`` normally equals the number
of FDTD iterations.

Equivalent-current transient far fields use a stricter output policy. Their
``times`` dataset already contains only the common interval for which every
surface patch has an available retarded-time sample; no partially supported
tail is written. The time coordinate is reduced time
:math:`\tau=t-r/c_b`, referenced to the surface origin, so increasing a
hypothetical observation radius would not prepend zeros. The
``terminal_field_ratios`` and ``terminal_decay_ok`` datasets are per direction.
A false value means the FDTD time window should be increased.

The surface DFT datasets are present by default and allow later angular or
point evaluation without rerunning FDTD. They can be large: their leading
dimensions are frequency and surface patch. An NTFF output creates the normal
model output file even when there are no conventional receivers or
transmission lines.

NTFF surfaces must strictly enclose every impressed source. For plane-wave
scattering models, the associated total-field/scattered-field box must be
strictly enclosed by the NTFF surface. The integration surface then samples
the scattered-field region outside the TFSF box, while the associated
numerical plane wave supplies the incident-field normalization used for RCS.
Sources and scatterers may reside on an HSG subgrid enclosed by these
main-grid surfaces. Neither an NTFF surface nor a TFSF correction surface may
touch or cut the HSG outer coupling surface; an overlapping surface must
strictly enclose the complete subgrid coupling region.


.. _outputs-snaps:

Snapshots
---------

Snapshot files contain a snapshot of the electromagnetic field values of a specified volume of the model domain at a specified point in time during the simulation. By default, snapshot files use the open source `Visualization ToolKit (VTK) <http://www.vtk.org>`_ format which can be viewed in many free readers, such as `Paraview <http://www.paraview.org>`_. Paraview is an open-source, multi-platform data analysis and visualization application. It is available for Linux, macOS, and Windows. You can optionally output snapshot files using the HDF5 format if desired. HDF5 snapshots include ``origin`` (global x, y, z coordinates), ``dx_dy_dz``, ``nx_ny_nz``, and ``time`` attributes. A snapshot owned by an HSG subgrid is sampled at the fine-grid time step and is still positioned in this global coordinate frame.

For a reduced 2-D model, a snapshot contains only the genuine invariant-axis
field plane: index zero for TM and index one for TE. Bounds spanning the full
invariant extent are collapsed automatically to that plane; bounds excluding
it are rejected. Inactive components may still be requested for a consistent
file schema, but contain zero. The file origin places the centre of the single
output plane at the physical Yee-field plane, rather than at either forced-zero
TE padding wall.

.. tip::
    You can take advantage of our Python API to easily create a series of snapshots. For example, to create 30 snapshots starting at time 0.1ns until 3ns in intervals of 0.1ns, use the following code snippet in your input file. Replace ``x, y, z, dl, fn`` accordingly.

    .. code-block:: none

        import gprMax

        for i in range(1, 31):
            s = gprMax.Snapshot(p1=(0, 0, 0), p2=(x, y, z), dl=(dl, dl, dl),
                                time=(i/10) * 1e-9,
                                filename=fn.with_suffix('').parts[-1] + '_' + str(i))
            scene.add(s)

The following are steps to get started with viewing snapshot files in Paraview:

#. **Open the file** either from the File menu or toolbar. Paraview should recognise the time series based on the file name and load in all the files.
#. Click the **Apply** button in the Properties panel. You should see an outline of the snapshot volume.
#. Use the **Coloring** drop down menu to select the field component you want to visual, e.g. **Ex**, **Ey**, **Ez**, **Hx**, **Hy**, **Hz**.
#. From the **Representation** drop down menu select **Surface**.
#. You can step through or play as an animation the time steps using the **time controls** in the toolbar.

.. tip::
    * Turn on the Animation View (View->Animation View menu) to control the speed and start/stop points of the animation.

    * Use the Color Map Editor to adjust the Color Scaling.


Geometry output
===============

Geometry files use the open source `Visualization ToolKit (VTK) <http://www.vtk.org>`_ format (specifically VTKHDF) which can be viewed in many free readers, such as `Paraview <http://www.paraview.org>`_. Paraview is an open-source, multi-platform data analysis and visualization application. It is available for Linux, Mac OS X, and Windows.

The ``#geometry_view:`` command produces either ImageData for a per-cell geometry view, or UnstructuredGrid for a per-cell-edge geometry view. The following are steps to get started with viewing geometry files in Paraview:

If the model contains semantic geometry tags, a normal per-cell geometry view
also contains the integer cell array ``TagID``. VTKHDF field data stores the
matching ``geometry_tag_ids`` and ``geometry_tag_names`` arrays; ID zero means
``untagged``. The tag array records the final voxelised geometry after ordered
overwrites and is not affected by dielectric smoothing. It is omitted
entirely for models without tags. Per-cell-edge geometry views do not export
tags because zero-thickness edge and plate objects do not yet have tag
semantics.

.. _pv_toolbar:

.. figure:: ../../images_shared/paraview_toolbar.png

    Paraview toolbar showing ``gprMax`` macro button.

#. **Open the file** either from the File menu or the toolbar.
#. Click the **Apply** button in the Properties panel. You should see an outline of the volume of the geometry view.
#. Install the ``gprMax.py`` Python script, that comes with the gprMax source code (in the ``toolboxes/Utilities/Paraview`` directory), as a macro in Paraview. This script makes it quick and easy to view the different materials in a geometry file. To add the script as a macro in Paraview choose the file from the Macros->Add new macro menu. It will then appear as a shortcut button in the toolbar as shown in :numref:`pv_toolbar`. You only need to do this once, the macro will be kept in Paraview for future use.
#. Click the ``gprMax`` shortcut button. All the materials in the model should appear in the Pipeline Browser as Threshold items as shown in :numref:`pv_pipeline`.

.. _pv_pipeline:

.. figure:: ../../images_shared/paraview_pipeline.png
    :width: 350 px

    Paraview Pipeline Browser showing list of materials in an example model.

.. tip::
    * You can turn on and off the visibility of materials using the eye icon in the Pipeline Browser. You can select multiple materials using the Shift key, and by shift-clicking the eye icon, turn the visibility of multiple materials on and off.

    * You can set the Color and Opacity of materials from the Properties panel.
