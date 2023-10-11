User libraries is a sub-package where useful Python modules contributed by users are stored.

*********************
AustinMan/AustinWoman
*********************

Information
===========

**Authors**: Jackson W. Massey, Cemil S. Geyik, Jungwook Choi, Hyun-Jae Lee, Natcha Techachainiran, Che-Lun Hsu, Robin Q. Nguyen, Trevor Latson, Madison Ball, and Ali E. Yılmaz

**Contact**: Ali E. Yılmaz (ayilmaz@mail.utexas.edu), The University of Texas at Austin

**License**: `Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License <http://creativecommons.org/licenses/by-nc-nd/3.0/>`_

**Attribution/cite**: Please follow the instructions at http://web.corral.tacc.utexas.edu/AustinManEMVoxels/AustinMan/citing_the_model/index.html

`AustinMan and AustinWoman <https://web.corral.tacc.utexas.edu/AustinManEMVoxels/AustinMan/index.html>`_ are open source electromagnetic voxel models of the human body, which are developed by the `Computational Electromagnetics Group <http://www.ece.utexas.edu/research/areas/electromagnetics-acoustics>`_ at `The University of Texas at Austin <http://www.utexas.edu>`_. The models are based on data from the `National Library of Medicine’s Visible Human Project <https://www.nlm.nih.gov/research/visible/visible_human.html>`_.

.. figure:: images/user_libs/AustinMan_head.png
    :width: 600 px

    FDTD geometry mesh showing the head of the AustinMan model (2x2x2mm :math:`^3`).

The following whole body models are available.

=========== ========================== ==================
Model       Resolution (mm :math:`^3`) Dimensions (cells)
=========== ========================== ==================
AustinMan   8x8x8                      86 x 47 x 235
AustinMan   4x4x4                      171 x 94 x 470
AustinMan   2x2x2                      342 x 187 x 939
AustinMan   1x1x1                      683 x 374 x 1877
AustinWoman 8x8x8                      86 x 47 x 217
AustinWoman 4x4x4                      171 x 94 x 433
AustinWoman 2x2x2                      342 x 187 x 865
AustinWoman 1x1x1                      683 x 374 x 1730
=========== ========================== ==================

Package overview
================

.. code-block:: none

    AustinManWoman_materials.txt
    AustinManWoman_materials_dispersive.txt
    head_only_h5.py

* ``AustinManWoman_materials.txt`` is a text file containing `non-dispersive material properties at 900 MHz <http://niremf.ifac.cnr.it/tissprop/>`_.
* ``AustinManWoman_materials_dispersive.txt`` is a text file containing `dispersive material properties using a 3-pole Debye model <http://dx.doi.org/10.1109/LMWC.2011.2180371>`_.

.. note::

    * The main body tissues are described using a 3-pole Debye model, but not all materials have a dispersive description.
    * The dispersive material properties can only be used with the 1x1x1mm or 2x2x2mm AustinMan/Woman models. This is because the time step of the model must always be less than any of the relaxation times of the poles of the Debye models used for the dispersive material properties.

* ``head_only_h5.py`` is a script to assist with creating a model of only the head from a full body AustinMan/Woman model.

How to use the models
=====================

The AustinMan and AustinWoman models themselves are not included in the user libraries sub-package.

* `Download a HDF5 file (.h5) of AustinMan or AustinWoman <https://web.corral.tacc.utexas.edu/AustinManEMVoxels/AustinMan/download/index.html>`_ at the resolution you wish to use

To insert either AustinMan or AustinWoman models into a simulation use the ``#geometry_objects_read``.

Example
-------

To insert a 2x2x2mm :math:`^3` AustinMan with the lower left corner 40mm from the origin of the domain, using disperive material properties, and with no dielectric smoothing, use the command:

.. code-block:: none

    #geometry_objects_read: 0.04 0.04 0.04 ../user_libs/AustinManWoman/AustinMan_v2.3_2x2x2.h5 ../user_libs/AustinManWoman/AustinManWoman_materials_dispersive.txt

For further information on the ``#geometry_objects_read`` see the section on object contruction commands in the :ref:`Input commands section <commands>`.

.. figure:: images/user_libs/AustinMan.png
    :width: 300 px

    FDTD geometry mesh showing the AustinMan body model (2x2x2mm :math:`^3`).
