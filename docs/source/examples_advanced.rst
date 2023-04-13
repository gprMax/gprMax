*****************
Advanced features
*****************

This section provides example models of some of the more advanced features of gprMax. Each example comes with an input file which you can download and run.

Building a heterogeneous soil
=============================

:download:`heterogeneous_soil.in <../../examples/heterogeneous_soil.in>`

This example demonstrates how to build a more realistic soil model using a stochastic distribution of dielectric properties. A mixing model for soils proposed by Peplinski (http://dx.doi.org/10.1109/36.387598) is used to define a series of dispersive material properties for the soil.

.. literalinclude:: ../../examples/heterogeneous_soil.in
    :language: none
    :linenos:

.. figure:: ../../images_shared/heterogeneous_soil.png
    :width: 600 px

    FDTD geometry mesh showing a heterogeneous soil model with a rough surface.

Line 10 defines a series of dispersive materials to represent a soil with sand fraction 0.5, clay fraction 0.5, bulk density :math:`2~g/cm^3`, sand particle density of :math:`2.66~g/cm^3`, and a volumetric water fraction range of 0.001 - 0.25. The volumetric water fraction is given as a range which is what defines a series of dispersive materials.

These materials can then be distributed stochastically over a volume using the ``#fractal_box`` command. Line 11 defines a volume, a fractal dimension, a number of materials, and a mixing model to use. The fractal dimension, 1.5, controls how the materials are stochastically distributed. The fractal weightings, 1, 1, 1, weight the fractal in the x, y, and z directions. The number of materials, 50, specifies how many dispersive materials to create using the mixing model (``my_soil``).

Adding rough surfaces
---------------------

A rough surface can be added to any side of ``#fractal_box`` using,

.. code-block:: none

    #add_surface_roughness: 0 0 0.070 0.15 0.15 0.070 1.5 1 1 0.065 0.080 my_soil_box

which defines one of the surfaces of the ``#fractal_box``, a fractal dimension, and minimum and maximum values for the height of the roughness (relative to the original ``#fractal_box`` volume). In this example the roughness will be stochastically distributed with troughs up to 5mm deep, and peaks up to 10mm high.

More information, including adding surface water and vegetation, can be found in the :ref:`section on using the fractal box command <fractals>`.


Using subgrid(s)
================

Including finely detailed objects or regions of high dielectric strength in FDTD modeling can dramatically increase the computational burden of the method. This is because the conditionally stable nature of the algorithm requires a minimum time step for a given spatial discretization. Thus, when the spatial discretization is lowered, either to reduce numerical dispersion or include small-sized features, the time step must be reduced. Also, the number of spatial cells is increased. One approach to reducing the overall computational cost is to introduce local finely discretized regions into a coarser finite-difference grid. This approach is known as subgridding. The computing time is reduced since there are fewer cells to solve. Also, there are fewer iterations since the coarse time step is maintained in the coarse region. gprMax uses a new Huygens subgridding (HSG) algorithm with a novel artificial loss mechanism called the switched Huygens subgridding (SHSG). For a detailed description of subgridding and the SHSG method please read [HAR2021]_.

Subgridding functionality requires using our :ref:`Python API <input-api>`.

.. _examples-subgrid:

High dielectric example
-----------------------

:download:`cylinder_fs.py <../../examples/subgrids/cylinder_fs.py>`

This example is a basic demonstration of how to use subgrids. The geometry is 3D (required for any use of subgrids) and is of a water-filled (high dielectric constant) cylindrical object in freespace. The subgrid encloses the cylindrical object using a fine spatial discretisation (1mm), and a courser spatial discretisation (5mm) is used in the rest of the model (main grid). A simple Hertzian dipole source is used with a waveform shaped as the first derivative of a gaussian.

.. literalinclude:: ../../examples/subgrids/cylinder_fs.py
    :language: none
    :linenos:


Antenna modelling example
-------------------------

:download:`gssi_400_over_fractal_subsurface.py <../../examples/subgrids/gssi_400_over_fractal_subsurface.py>`

This example....

.. literalinclude:: ../../examples/subgrids/gssi_400_over_fractal_subsurface.py
    :language: none
    :linenos:







