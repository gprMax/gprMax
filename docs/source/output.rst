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

The output file contains HDF5 groups for sources (``srcs``), transmission lines
(``tls``), receivers (``rxs``), and KSIR outputs (``ntff``) when requested.
Within these are further groups for each named or numbered output.

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
            tl2/
                ...
        ntff/ [optional]
            <monitor name>/
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

Within each individual ``tl`` group are the following datasets:

* ``Vinc`` is an array containing the time history (for the model time window) of the values of the incident voltage in the transmission line.
* ``Iinc`` is an array containing the time history (for the model time window) of the values of the incident current in the transmission line.
* ``Vtotal`` is an array containing the time history (for the model time window) of the values of the total (field) voltage in the transmission line.
* ``Itotal`` is an array containing the time history (for the model time window) of the values of the total (field) current in the transmission line.

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
