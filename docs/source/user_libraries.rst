.. _user-libs:

**************
User libraries
**************

User libraries is a sub-package where useful Python modules contributed by users are stored.

antennas.py
===========

.. code-block:: python

    # Copyright (C) 2015-2016, Craig Warren
    #
    # This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
    # To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
    #
    # Please use the attribution at http://dx.doi.org/10.1190/1.3548506

The module currently features models of antennas similar to:

* a Geophysical Survey Systems, Inc. (GSSI) 1.5 GHz (Model 5100) antenna (http://www.geophysical.com)
* a MALA Geoscience 1.2 GHz antenna (http://www.malags.com/)

A description of how the models were created can be found at http://dx.doi.org/10.1190/1.3548506.

The antenna models can be accessed from within a block of Python code in your simulation. The models must be used with cubic spatial resolutions of either 1mm (default) or 2mm. For example, to use Python to include an antenna model similar to a GSSI 1.5 GHz antenna at a location 0.125m, 0.094m, 0.100m (x,y,z):

.. code-block:: none

    #python:
    from user_libs.antennas import antenna_like_GSSI_1500
    antenna_like_GSSI_1500(0.125, 0.094, 0.100)
    #end_python:

.. figure:: images/antenna_like_GSSI_1500.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a GSSI 1.5 GHz antenna (skid removed for illustrative purposes).

.. figure:: images/antenna_like_MALA_1200.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a MALA 1.2GHz antenna (skid removed for illustrative purposes).
