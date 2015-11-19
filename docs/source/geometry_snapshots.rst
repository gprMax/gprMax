***************************
Geometry and Snapshot files
***************************

The geometry and snapshot files use the open source Visualization ToolKit (VTK) (http://www.vtk.org) format which can be viewed in many free readers, such as Paraview (http://www.paraview.org). Paraview is an open-source, multi-platform data analysis and visualization application. It is available for Linux, Mac OS X, and Windows.

Geometry files
==============

The ``#geometry_view:`` command produces either ImageData (.vti) for a per-cell geometry view, or PolygonalData (.vtp) for a per-cell-edge geometry view. The following are steps to get started with viewing geometry files in Paraview:

.. _pv_toolbar:

.. figure:: images/paraview_toolbar.png

    Paraview toolbar showing ``gprMax_materials`` macro button.

#. **Open the file** either from the File menu or toolbar.
#. Click the **Apply** button in the Properties panel. You should see an outline of the volume of the geometry view.
#. Install the ``gprMax_materials.py`` Python script, that comes with the gprMax source code (in the ``tools/Paraview macros`` directory), as a macro in Paraview. This script makes it quick and easy to view the different materials in a geometry file. To add the script as a macro in Paraview choose the file from the Macros->Add new macro menu. It will then appear as a shortcut button in the toolbar as shown in :numref:`pv_toolbar`. You only need to do this once, the macro will be kept in Paraview for future use.
#. Click the ``gprMax_materials`` shortcut button. All the materials in the model should appear in the Pipeline Browser as Threshold items as shown in :numref:`pv_pipeline`.

.. _pv_pipeline:

.. figure:: images/paraview_pipeline.png
    :width: 350 px

    Paraview Pipeline Browser showing list of materials in an example model.

.. tip::
    * You can turn on and off the visibility of materials using the eye icon in the Pipeline Browser. You can select multiple materials using the Shift key, and by shift-clicking the eye icon, turn the visibility of multiple materials on and off.

    * You can set the Color and Opacity of materials from the Properties panel.


Snapshot files
==============

The ``#snapshot:`` command produces an ImageData (.vti) snapshot file for each time instance requested.

.. tip::
    You can take advantage of Python scripting to easily create a series of snapshots. For example, to create 30 snapshots starting at time 0.1ns until 3ns in intervals of 0.1ns, insert the following into your input file.

    .. code-block:: none

        #python:
        for i in range(1, 31):
            print('#snapshot: x1 y1 z1 x2 y2 z2 dx dy dz {} snapshot{}'.format((i/10)*1e-9, i))
        #end_python:

The following are steps to get started with viewing snapshot files in Paraview:

#. **Open the file** either from the File menu or toolbar. Paraview should recognise the time series based on the file name and load in all the files.
#. Click the **Apply** button in the Properties panel. You should see an outline of the snapshot volume.
#. Use the **Coloring** drop down menu to select either **E-field** or **H-field**, and the further drop down menu to select either **Magnitude**, **x**, **y** or **z** component.
#. From the **Representation** drop down menu select **Surface**.
#. You can step through or play as an animation the time steps using the **time controls** in the toolbar.

.. tip::

    * Turn on the Animation View (View->Animation View menu) to control the speed and start/stop points of the animation.

    * Use the Color Map Editor to adjust the Color Scaling.


