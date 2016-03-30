.. _user-libs:

*************
User libaries
*************

The user libraries sub-package is where useful modules contributed by users are stored.

antennas.py
===========

This library currently features models of antennas similar to:

* a Geophysical Survey Systems, Inc. (GSSI) 1.5 GHz (Model 5100) antenna (http://www.geophysical.com)
* a MALA Geoscience 1.2 GHz antenna (http://www.malags.com/)

These antenna models can be accessed from within a block of Python code in your simulation. For example, to use Python to include an antenna model similar to a GSSI 1.5 GHz antenna at a location 0.125m, 0.094m, 0.100m (x,y,z) using a 1mm spatial resolution:

.. code-block:: none

    #python:
    from user_libs.antennas import antenna_like_GSSI_1500
    antenna_like_GSSI_1500(0.125, 0.094, 0.100, 0.001)
    #end_python:

.. figure:: images/antenna_like_GSSI_1500.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a GSSI 1.5   GHz antenna (skid removed for illustrative purposes).

.. figure:: images/antenna_like_MALA_1200.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a MALA 1.2GHz antenna (skid removed for illustrative purposes).
