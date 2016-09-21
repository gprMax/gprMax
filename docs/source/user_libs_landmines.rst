User libraries is a sub-package where useful Python modules contributed by users are stored.

*********
Landmines
*********

Information
===========

**Author/Contact**: Iraklis Giannakis (I.Giannakis@ed.ac.uk), University of Edinburgh

**License**: Creative Commons Attribution-ShareAlike 4.0 International License (http://creativecommons.org/licenses/by-sa/4.0/)

**Attribution/cite**: Giannakis, I., Giannopoulos, A., Warren, C. (2016). A Realistic FDTD Numerical Modeling Framework of Ground Penetrating Radar for Landmine Detection. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 9(1), 37-51. (http://dx.doi.org/10.1109/JSTARS.2015.2468597)

The module currently features models of different anti-personnel (AP) landmines and a metal can which can be used as a false target. They are:

* **PMA-1**: a blast AP landmine with minimum metal content, manufactured in the former Yugoslavia. It is possible to detect the PMA-1 with a metal detector because it contains a metal fuse, but there are reports of types of PMA-1 with plastic fuses. The PMA-1 contains 200g of high explosive (TNT). The dimensions of the PMA-1 model are: 140x64x34mm.
* **PMN**: one of the oldest and most widely used landmines, it is a palm shaped cylindrical blast AP landmine, manufactured in Russia. Similar to PMA-1, the PMN contains a large amount, 240g, of high explosive (TNT). It has a minimum metal content which can make it detectable with a metal detector. The dimensions of the PMN model are: 116x156x50mm.
* **TS-50**: a blast AP landmine with minimum metal content, manufactured in Italy. The dimensions of the TS-50 model are: 90x90x44mm.
* **Metal can**: a cylindrical metal can which is can be useful as a false target. The dimensions of the metal can model are: 76x76x108mm.

The landmine models and the metal can be used with a cubic spatial resolution of either 1mm or 2mm.

The dielectric properties of the landmines were obtained through an iterative process of matching numerical and laboratory measurements of scattered electromagnetic fields in free space. A full description of how the models were created can be found at http://dx.doi.org/10.1109/JSTARS.2015.2468597.

Package overview
================

.. code-block:: none

    can_1x1x1.h5
    can_gprMax_materials.txt
    PMA_1x1x1.h5
    PMA_gprMax_materials.txt
    PMN_1x1x1.h5
    PMN_gprMax_materials.txt
    TS50_1x1x1.h5
    TS50_gprMax_materials.txt

* ``can_1x1x1.h5`` is a HDF5 file containing a description of the geometry of the metal can (false target) with a cubic spatial resolution of 1mm
* ``can_gprMax_materials.txt`` is a text file containing material properties associated with the metal can
* ``PMA_1x1x1.h5`` is a HDF5 file containing a description of the geometry of the PMA landmine with a cubic spatial resolution of 1mm
* ``PMA_gprMax_materials.txt`` is a text file containing material properties associated with the PMA landmine
* ``PMN_1x1x1.h5`` is a HDF5 file containing a description of the geometry of the PMN landmine with a cubic spatial resolution of 1mm
* ``PMN_gprMax_materials.txt`` is a text file containing material properties associated with the PMN landmine
* ``TS50_1x1x1.h5`` is a HDF5 file containing a description of the geometry of the TS-50 landmine with a cubic spatial resolution of 1mm
* ``TS50_gprMax_materials.txt`` is a text file containing material properties associated with the TS-50 landmine

How to use the models
=====================

To insert any of the landmine models or metal can into a simulation use the ``#geometry_objects_file`` command.

Example
-------

The input file for inserting the PMN landmine, with the lower left corner 10mm from the origin of the domain, into an empty domain (free-space) would be:

.. code-block:: none

    #title: PMN landmine (116x156x50mm) in free space
    #domain: 0.136 0.176 0.070
    #dx_dy_dz: 0.001 0.001 0.001
    #time_window: 5e-9
    #geometry_objects_file: 0.010 0.010 0.010 ../user_libs/landmines/PMN_1x1x1.h5 ../user_libs/landmines/PMN_gprMax_materials.txt
    #geometry_view: 0 0 0 0.136 0.176 0.070 0.001 0.001 0.001 landmine_PMN_fs n

For further information on the ``#geometry_objects_file`` see the section on object contruction commands in the :ref:`Input commands section <commands>`.

.. figure:: images/user_libs/PMA.png
    :width: 600 px

    FDTD geometry mesh showing the PMA-1 landmine model.

.. figure:: images/user_libs/PMN.png
    :width: 600 px

    FDTD geometry mesh showing the PMN landmine model.