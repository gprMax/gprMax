.. _output:

***********
Output file
***********

gprMax produces an output file that has the same name as the input file but with ``.out`` appended. The output file uses the widely-supported HDF5 (https://www.hdfgroup.org/HDF5/) format which was designed to store and organize large amounts of numerical data.


File structure
==============

The output file has the following HDF5 attributes at the root (``/``):

* ``Title`` is the title of the model
* ``Iterations`` is the number of iterations for the time window of the model
* ``nx, ny, nz`` is a tuple containing the number of cells in each direction of the model
* ``dx, dy, dz`` is a tuple containing the spatial discretisation, i.e. :math:`\Delta x`, :math:`\Delta y`, :math:`\Delta z`
* ``dt`` is the time step of the model, i.e. :math:`\Delta t`
* ``srcsteps`` is the spatial increment used to move all sources between model runs.
* ``rxsteps`` is the spatial increment used to move all receivers between model runs.
* ``nsrc`` is the total number of sources in the model.
* ``nrx`` is the total number of receievers in the model.

The output file contains HDF5 groups for sources (``srcs``), transmission lines (``tls``), and receivers (``rxs``). Within each group are further groups that correspond to individual sources/transmission lines/receivers, e.g. ``src1``, ``src2`` etc...

.. code-block:: none

    /
        rxs/
            rx1/
                Name [optional]
                Position
                Ex
                Ey
                Ez
                Hx
                Hy
                Hz
                Ix
                Iy
                Iz
            rx2/
                ...
        srcs/
            src1/
                Position
            src2/
                ...

        tls/
            tl1/
                Position
            tl22/
                ...

Within each individual ``rx`` group are the following attributes:

* ``Name`` is optional if a name for the receiver is given in the model.
* ``Position`` is the x, y, z position (in metres) of the receiver in the model.

Within each individual ``rx`` group can be the following datasets:

* ``Ex`` is an array containing the time history (for the model time window) of the values of the x component of the electric field at that receiver position.
* ``Ey`` is an array containing the time history (for the model time window) of the values of the y component of the electric field at that receiver position.
* ``Ez`` is an array containing the time history (for the model time window) of the values of the z component of the electric field at that receiver position.
* ``Hx`` is an array containing the time history (for the model time window) of the values of the x component of the magnetic field at that receiver position.
* ``Hy`` is an array containing the time history (for the model time window) of the values of the y component of the magnetic field at that receiver position.
* ``Hz`` is an array containing the time history (for the model time window) of the values of the z component of the magnetic field at that receiver position.
* ``Ix`` is an array containing the time history (for the model time window) of the values of the x component of current (calculated around a single cell loop) at that receiver position.
* ``Iy`` is an array containing the time history (for the model time window) of the values of the y component of current (calculated around a single cell loop) at that receiver position.
* ``Iz`` is an array containing the time history (for the model time window) of the values of the z component of current (calculated around a single cell loop) at that receiver position.

Within each individual ``src`` group are the following attributes:

* ``Type`` is the type of source, e.g. Hertzian dipole, voltage source etc...
* ``Position`` is the x, y, z position (in metres) of the source in the model.

Within each individual ``tl`` group are the following attributes:

* ``Position`` is the x, y, z position (in metres) of the source in the model.
* ``Resistance`` is the resistance of the transmission line.
* ``dl`` is the spatial discretisation of the transmission line.

Within each individual ``tl`` group are the following datasets:

* ``Vinc`` is an array containing the time history (for the model time window) of the values of the incident voltage in the transmission line.
* ``Vscat`` is an array containing the time history (for the model time window) of the values of the scattered (field) voltage in the transmission line.
* ``Iscat`` is an array containing the time history (for the model time window) of the values of the scattered (field) current in the transmission line.
* ``Vtot`` is an array containing the time history (for the model time window) of the values of the total (field) voltage in the transmission line.
* ``Itot`` is an array containing the time history (for the model time window) of the values of the total (field) current in the transmission line.


Viewing output
==============

There are a number of free tools available to read HDF5 files. Also MATLAB has high- and low-level functions for reading and writing HDF5 files, i.e. ``h5info`` and ``h5disp`` are useful for returning information and displaying the contents of HDF5 files respectively. gprMax includes some Python modules (in the ``tools`` package) to help you view output data. These are documented in the :ref:`tools section <plotting>`.


