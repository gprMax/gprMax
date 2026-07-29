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

The output file contains HDF5 groups for sources (``srcs``), transmission lines
(``tls``), receivers (``rxs``), voltage-source ports (``ports``), and KSIR
outputs (``ntff``) when requested. Within these are further groups for each
named or numbered output.

.. code-block:: none

    /
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

Within each individual ``rx`` group are the following attributes:

* ``Name`` is the name of the receiver if specified. Otherwise 'Rx(x,y,z)', where x,y,z is the position of the receiver, is used.
* ``Position`` is the x, y, z position (in metres) of the receiver in the model.

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

Within each individual ``src`` group are the following attributes:

* ``Type`` is the type of source, e.g. Hertzian dipole, voltage source etc...
* ``Position`` is the x, y, z position (in metres) of the source in the model.

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

Transmission-line S11 and impedance output
-------------------------------------------

S11, input impedance, and input admittance are generated automatically for
every transmission-line source; no ``#rx_port`` command or additional
receiver is required. With the real line resistance :math:`Z_0` as the
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

Voltage-source S11 and impedance output
---------------------------------------

The ``#rx_port`` command and ``RxPort`` Python object write one group per port at
``/ports/<port ID>``. The source resistance is the reference impedance
:math:`Z_0`. The source-plane reflection coefficient is calculated directly
from the known generator voltage and sampled total gap voltage; the reported
``S11`` then removes the effective Yee-edge background capacitance and
conductance. ``Zin`` and ``Yin`` are derived from that corrected result:

.. math::

    Z_\mathrm{in}=Z_0\frac{1+S_{11}}{1-S_{11}},
    \qquad
    Y_\mathrm{in}=\frac{1-S_{11}}{Z_0(1+S_{11})}.

Important attributes include:

* ``ReferenceImpedance``, ``Polarisation``, ``Position``, and ``GridPosition``;
* ``BackgroundMaterial``, ``GapCapacitance``, and
  ``BackgroundConductance``;
* ``SpectrumLimitMode``, ``MinimumWavelengthCells``,
  ``MeshFrequencyLimit``, ``NyquistFrequency``, and ``LimitingMaterial``;
* ``FrequencyRange`` (the first and last stored bins),
  ``ValidFrequencyRange`` (a convenience summary), and
  ``IndependentFrequencyResolution``;
* ``phasor_time_sign=exp(+j*omega*t)`` and
  ``forward_transform_sign=exp(-j*omega*t)``.

The principal datasets are:

* ``frequency``: the authoritative plotting axis in Hz;
* ``S11``, ``Zin``, and ``Yin``: corrected complex terminal quantities;
* ``S11_source`` and ``Zin_source``: uncorrected source-plane quantities;
* ``Vincident_spectrum``, ``Vreflected_source_spectrum``, and
  ``Vtotal_spectrum``: complex voltage spectra;
* ``time``, ``Vgenerator``, and ``Vtotal``: half-time-step-aligned audit
  histories;
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

KSIR field-transformation output
--------------------------------

Reusable KSIR outputs are stored in the normal model file under their surface
and transform IDs:

.. code-block:: none

    /ntff/<surface_id>/
        time/<rx_id>/
            points
            times
            time_origins
            valid_lengths
            spherical_coordinates [spherical commands only]
            fields/<output>
        frequency/<transform_id>/
            frequencies
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
                fields/<output>

The surface group records logical bounds, physical reference origin, closure
status, omitted symmetry faces, boundary types/coordinates, and image count.
The frequency transform group records the window, inferred wave speed and impedance,
configured precision and collection backend, plus the engineering convention:
``exp(+j*omega*t)`` phasors, ``exp(-j*omega*t)`` forward transform, and
``exp(-j*k*R)`` outgoing Green function.

Exact frequency receiver groups have ``range_normalized=False``. They contain
physical finite-distance phasors with every ``1/R`` and ``1/R**2`` term.
Far-field groups have ``range_normalized=True`` and a ``normalization``
attribute specifying ``r * exp(+j*k*r) * field``. Their radius is intentionally
absent. Complex datasets use the complex type paired with the configured
gprMax real precision.

Time-domain fields have shape ``(npoints, max(valid_lengths))``. For point
``q``, the physical time vector and valid trace are:

.. code-block:: python

    physical_time = time_origins[q] + times[:valid_lengths[q]]
    trace = fields[output][q, :valid_lengths[q]]

With ``time_origin=simulation`` every origin is zero. With
``time_origin=first_arrival`` each origin retains its absolute propagation
time without storing the potentially large guaranteed leading-zero prefix.

The surface DFT datasets are present by default and allow later angular or
point evaluation without rerunning FDTD. They can be large: their leading
dimensions are frequency and surface patch. A KSIR output creates the normal
model output file even when there are no conventional receivers or
transmission lines.

KSIR surfaces must strictly enclose every impressed source. For plane-wave
scattering models, the associated total-field/scattered-field box must be
strictly enclosed by the KSIR surface so that incident-field subtraction can
be applied consistently.


.. _outputs-snaps:

Snapshots
---------

Snapshot files contain a snapshot of the electromagnetic field values of a specified volume of the model domain at a specified point in time during the simulation. By default, snapshot files use the open source `Visualization ToolKit (VTK) <http://www.vtk.org>`_ format which can be viewed in many free readers, such as `Paraview <http://www.paraview.org>`_. Paraview is an open-source, multi-platform data analysis and visualization application. It is available for Linux, macOS, and Windows. You can optionally output snapshot files using the HDF5 format if desired.

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
