*****************
Advanced features
*****************

This section provides example models of some of the more advanced features of gprMax. Each example comes with an input file which you can download and run.

Building a heterogeneous soil
=============================

:download:`heterogeneous_soil.in <../../user_models/heterogeneous_soil.in>`

This example demonstrates how to build a more realistic soil model using a stochastic distribution of dielectric properties. A mixing model for soils proposed by Peplinski (http://dx.doi.org/10.1109/36.387598) is used to define a series of dispersive material properties for the soil.

.. literalinclude:: ../../user_models/heterogeneous_soil.in
    :language: none
    :linenos:

.. figure:: images/heterogeneous_soil.png
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








