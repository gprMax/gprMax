***********************
Overview of source code
***********************

This section provides an overview of the source code modules and describes each of the classes and methods used in the gprMax package. The following licensing information applies to all source files unless otherwise stated::

    Copyright (C) 2015, The University of Edinburgh.

    Authors: Craig Warren and Antonis Giannopoulos

    gprMax is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gprMax is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gprMax.  If not, see <http://www.gnu.org/licenses/>.


constants.py
============

Defines constants:

* Speed of light in vacuum :math:`c=2.9979245 \times 10^8` m/s
* Permittivity of free space :math:`\epsilon_0=8.854187 \times 10^{-12}` F/m
* Permeability of free space :math:`\mu_0=1.256637 \times 10^{-6}` H/m
* Impedance of free space :math:`z_0=376.7303134` Ohms

Defines data types:

* Solid and ID arrays use 32-bit integers (0 to 4294967295)
* Rigid arrays use 8-bit integers (the smallest available numpy type to store booleans - true/false)
* Fractal and dispersive coefficient arrays use complex numbers (:code:`complextype`) which are represented as two :code:`floattype`
* Main field arrays use floats (:code:`floattype`) and complex numbers (:code:`complextype`)
* :code:`floattype` and :code:`complextype` are set to use 32-bit floats but can be changed to use 64-bit double precision if required.

.. automodule:: gprMax.constants


exceptions.py
=============

.. automodule:: gprMax.exceptions


fields_update.pyx
=================

.. automodule:: gprMax.fields_update


fractals.py
===========

.. automodule:: gprMax.fractals


geometry_primitives.pyx
=======================

.. automodule:: gprMax.geometry_primitives


geometry_views.py
=================

.. automodule:: gprMax.geometry_views


gprMax.py
===========

.. automodule:: gprMax.gprMax

grid.py
=======

.. automodule:: gprMax.grid


input_cmds_file.py
==================

.. automodule:: gprMax.input_cmds_file


input_cmds_geometry.py
======================

.. automodule:: gprMax.input_cmds_geometry


input_cmds_multiuse.py
======================

.. automodule:: gprMax.input_cmds_multiuse


input_cmds_singleuse.py
=======================

.. automodule:: gprMax.input_cmds_singleuse


materials.py
============

.. automodule:: gprMax.materials


output.py
=========

.. automodule:: gprMax.output


pml_1order_update.pyx
=====================

.. automodule:: gprMax.pml_1order_update


pml_2order_update.pyx
=====================

.. automodule:: gprMax.pml_2order_update


pml_call_updates.py
===================

.. automodule:: gprMax.pml_call_updates


pml.py
======

.. automodule:: gprMax.pml


receivers.py
============

.. automodule:: gprMax.receivers


snapshots.py
============

.. automodule:: gprMax.snapshots


sources.py
==========

.. automodule:: gprMax.sources


user_libs.antennas.py
=====================

This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License::

    Copyright (C) 2015, Craig Warren

    This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
    To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.

    Please use the attribution at http://dx.doi.org/10.1190/1.3548506

.. automodule:: user_libs.antennas


utilities.py
============

.. automodule:: gprMax.utilities


waveforms.py
============

.. automodule:: gprMax.waveforms


yee_cell_build.pyx
==================

.. automodule:: gprMax.yee_cell_build


yee_cell_setget_rigid.pyx
=========================

.. automodule:: gprMax.yee_cell_setget_rigid

