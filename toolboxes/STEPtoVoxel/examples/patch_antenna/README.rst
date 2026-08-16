Probe-fed patch antenna
=======================

``PROBE_FED.stp`` is an AP242 CAD assembly containing the patch,
substrate, ground plane, coaxial inner conductor, coaxial dielectric, outer
conductor, and a zero-volume port reference face.

Run the complete conversion and gprMax geometry-only inspection from the
repository root:

.. code-block:: console

    python toolboxes/STEPtoVoxel/examples/patch_antenna/patch_antenna_geometry.py

Generated HDF5, VTI, VTK-HDF, JSON and cache files should not be committed.
The conversion recognises ``port1`` as a non-physical CAD port marker and
writes its centre, aperture bounds and normal axis to ``markers.json``. The
example translates these coordinates to the imported gprMax geometry and
prints the nearest grid-aligned port plane. It also writes
``reference_geometry_gprmax.vtp``, translated for direct overlay on the
gprMax GeometryView. It does not create an excitation.

After running the conversion, open ``output/geometry.vti`` and
``output/reference_geometry_cad.vtp`` together in ParaView. The latter
contains the original CAD ``port1`` plane for checking the source location
without adding it to the electromagnetic material grid.

The documented views can also be regenerated with PyVista:

.. code-block:: console

    python toolboxes/STEPtoVoxel/examples/patch_antenna/plot_patch_geometry.py

``reference_geometry_cad.vtp`` overlays the converter's ``geometry.vti``
because both use CAD coordinates. Use ``reference_geometry_gprmax.vtp`` with
the gprMax GeometryView because both then use model coordinates. The example
creates the latter only after ``import_origin`` is known; gprMax itself does
not write these reference files.

``probe_fed_gprmax_patch_view.svg`` and
``probe_fed_gprmax_coax_view.svg`` are ParaView exports of the actual gprMax
VTK-HDF geometry view produced after ``GeometryObjectsRead``.

The dielectric properties in ``materials.csv`` are illustrative and must be
replaced by the actual antenna material properties before running an
electromagnetic simulation.
