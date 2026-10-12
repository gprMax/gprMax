Eigenmode tracking validation
=============================

Studies are classified by purpose: subspaces, symmetry, identification,
dielectric modes, conductor loss, analytical profiles, and driven ports.
Each result directory contains its JSON reports and plots. Field NPZ archives
are local run products and are not included in Git.

Provenance
----------

``retained_runs.json`` records the SHA-256 hashes and original directories of
historical runs retained during the reorganization. Their original metadata,
source hashes, and recorded paths are preserved; they are not new simulations.
``relocations.json`` maps moved source and figure paths. The persistent,
expanded, and classification studies were rerun with the reorganized modules.
Plots added to historical hybrid, circular, and C2v TE/TM studies were generated
from their original field archives using ``plot_results.py``. Field panels show
the magnitude of the strongest electric component, normalized separately for
each mode; they do not measure analytical agreement.

Run scripts as modules from the repository root, for example::

   python -m testing.validation.eigenmode_tracking.subspaces.expanded_mode_tracking --output results/eigenmode_tracking/expanded

Regenerate field and tracking plots from a retained local archive::

   python -m testing.validation.eigenmode_tracking.plot_results path/to/result_directory

Study galleries
---------------

* `subspaces: expanded <subspaces/results/expanded/README.rst>`_ (4 plots)
* `subspaces: general <subspaces/results/general/README.rst>`_ (18 plots)
* `subspaces: persistent <subspaces/results/persistent/README.rst>`_ (1 plots)
* `symmetry: automatic <symmetry/results/automatic/README.rst>`_ (15 plots)
* `symmetry: fine mesh <symmetry/results/fine_mesh/README.rst>`_ (12 plots)
* `symmetry: moment parity <symmetry/results/moment_parity/README.rst>`_ (16 plots)
* `symmetry: polygons <symmetry/results/polygons/README.rst>`_ (2 plots)
* `symmetry: waveguides <symmetry/results/waveguides/README.rst>`_ (24 plots)
* `identification: C2v TE TM <identification/results/C2v_TE_TM/README.rst>`_ (2 plots)
* `identification: classification <identification/results/classification/README.rst>`_ (14 plots)
* `identification: higher circular <identification/results/higher_circular/README.rst>`_ (4 plots)
* `identification: longitudinal <identification/results/longitudinal/README.rst>`_ (4 plots)
* `dielectric: hybrid control <dielectric/results/hybrid_control/README.rst>`_ (2 plots)
* `dielectric: separation <dielectric/results/separation/README.rst>`_ (3 plots)
* `conductors: copper symmetry <conductors/results/copper_symmetry/README.rst>`_ (1 plots)
* `profiles: C2v rectangle 2to1 copper <profiles/results/C2v_rectangle_2to1_copper/README.rst>`_ (8 plots)
* `profiles: C2v rectangle 2to1 PEC <profiles/results/C2v_rectangle_2to1_PEC/README.rst>`_ (7 plots)
* `profiles: C4v cylindrical copper <profiles/results/C4v_cylindrical_copper/README.rst>`_ (7 plots)
* `profiles: C4v cylindrical PEC <profiles/results/C4v_cylindrical_PEC/README.rst>`_ (6 plots)
* `profiles: C4v square copper <profiles/results/C4v_square_copper/README.rst>`_ (5 plots)
* `profiles: C4v square PEC <profiles/results/C4v_square_PEC/README.rst>`_ (4 plots)
* `ports: figures <ports/figures/README.rst>`_ (2 plots)
* `overview: figures <overview/figures/README.rst>`_ (4 plots)
