User libraries is a sub-package where useful Python modules contributed by users are stored.

******************
GPR antenna models
******************

Information
===========

**Author/Contact**: Craig Warren (Craig.Warren@ed.ac.uk), University of Edinburgh

**License**: `Creative Commons Attribution-ShareAlike 4.0 International License <http://creativecommons.org/licenses/by-sa/4.0/>`_

**Attribution/cite**: Warren, C., Giannopoulos, A. (2011). Creating finite-difference time-domain models of commercial ground-penetrating radar antennas using Taguchi's optimization method. *Geophysics*, 76(2), G37-G47. (http://dx.doi.org/10.1190/1.3548506)

The module currently features models of antennas similar to commercial GPR antennas:

* `Geophysical Survey Systems, Inc. (GSSI) <http://www.geophysical.com>`_ 1.5 GHz (Model 5100) antenna. The dimensions of the GSSI 1.5GHz antenna model are: 170x108x45mm.
* `MALA Geoscience <http://www.malags.com/>`_ 1.2 GHz antenna. The dimensions of the MALA 1.2GHz antenna model are: 184x109x46mm.

A description of how the models were created can be found at the reference given by the aforementioned attribution/cite.

Module overview
===============

* ``antennas.py`` is a module containing the descriptions of the antennas.


How to use the module
=====================

The antenna models can be accessed from within a block of Python code in an input file. The models are inserted at location x,y,z. The coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction. The models must be used with cubic spatial resolutions of either 1mm (default) or 2mm by setting the keyword argument, e.g. ``resolution=0.002``. The antenna models can be rotated 90 degrees counter-clockwise (CCW) in the x-y plane by setting the keyword argument ``rotate90=True``.

Example
-------

To include an antenna model similar to a GSSI 1.5 GHz antenna at a location 0.125m, 0.094m, 0.100m (x,y,z) using a 2mm cubic spatial resolution:

.. code-block:: none

    #python:
    from user_libs.antennas import antenna_like_GSSI_1500
    antenna_like_GSSI_1500(0.125, 0.094, 0.100, resolution=0.002)
    #end_python:

.. figure:: images/antenna_like_GSSI_1500.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a GSSI 1.5 GHz antenna (skid removed for illustrative purposes).

.. figure:: images/antenna_like_MALA_1200.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a MALA 1.2GHz antenna (skid removed for illustrative purposes).
