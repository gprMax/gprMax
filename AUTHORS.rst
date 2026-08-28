.. _authors:

Authors
*******

The principal authors of gprMax, responsible for its creation, foundational
original developments, or major codebase rewrites, are:

* Antonis Giannopoulos — Creator of the first version of the original gprMax C code and designer of the code's numerical foundation. Developed the original input file scheme, the RIPML and the improved time-domain NTFF formulations — University of Edinburgh
* Craig Warren — Lead developer. Created the new open source gprMax codebase in Python and Cython. Built the GPR antenna toolboxes, the gprMax website and the new documentation — Northumbria University
* Iraklis Giannakis — Advanced algorithmic numerical development. Created the new PLRC formulation used in all material modelling and network ports. Developed the fractal soil models and the landmine toolboxes — Macau University of Science and Technology
* John Hartley — Initial developer of the new version 4 codebase and creator and developer of the subgrid functionality and of the dispersive material averaging — University of Edinburgh
* Nathan Mannall — Major refactoring of the version 4 codebase and developer of the MPI domain decomposition solver — Edinburgh Parallel Computing Centre

Core contributors
------------------

The following people have made substantial contributions to gprMax, such as
major features or subsystems:

* Qifeng Shen — Designed and developed the 2D FDFD eigensolver and the new waveport capabilities and impedance boundaries. Leading developer for antenna and RF simulation capabilities — University of Edinburgh

Contributors
------------

We are grateful to the following people for their contributions, including
bug fixes, documentation, testing, toolboxes and model libraries, and smaller
features:

* Abhishek Kumar — GPU backend for DWP plane-wave sources — GSoC 2026
* Sahibjot Singh — Comprehensive test suite — GSoC 2026
* Gaurav Sharma — Marimo notebooks and dashboards for gprMax data — GSoC 2026
* Mahdee Abir — Initial development of STEP file voxelisation toolbox for gprMax — University of Edinburgh, 2026
* Manobhav Sachan — Apple Metal port — GSoC 2025
* Quyen "Bianca" Pham — Apple Metal port — GSoC 2024
* Adittya Pal — Initial development of the Discrete Plane Wave code — GSoC 2023
* Sylwia Majchrowska — Initial work on the DebyeFit toolbox — GSoC 2021
* Kartik Bansal — Initial development of STL file voxelisation toolbox for gprMax — GSoC 2021
* Ourania Patsia — Development of the 2000 MHz palm GSSI antenna surrogate model — University of Edinburgh
* Sam Stadler — Development of the 400 MHz GSSI antenna surrogate model — Leibniz Institute for Applied Geophysics
* Nectaria Diamanti — Early ADI-FDTD subgridding research for gprMax that influenced the current HSG formulation, and advanced testing of gprMax and its GPR antenna modelling capabilities — Aristotle University of Thessaloniki
* Tobias Schruff-Wieneke — Initial generic MPI executor and accompanying CLI integration, subsequently incorporated into the gprMax MPI task-farm implementation — PR #233, 2019–2020

For a complete and up-to-date list of contributions, see the
`GitHub contributor graph <https://github.com/gprMax/gprMax/graphs/contributors>`_.

Citing gprMax
-------------

If you use gprMax in your research, please currently cite the software using
the information in ``CITATION.cff``, or the primary paper of version 3 until
an updated publication becomes available:

.. code-block:: text

    Warren, C., Giannopoulos, A., & Giannakis, I. (2016). gprMax: Open source
    software to simulate electromagnetic wave propagation for Ground
    Penetrating Radar. Computer Physics Communications.

Acknowledgements
-----------------

We would like to thank the following people and organisations for supporting
the development of gprMax over the years:

* Google Inc. for supporting the development of the initial GPU solver and for including gprMax in their Google Summer of Code programme
* Dstl, UK for supporting PhD research at the University of Edinburgh that led to the development of advanced gprMax functionality and most of the initial Version 3 codebase
* BRE, UK for initial funding support to develop the first version of gprMax in 1997
* COST Action TU1208, ``Civil Engineering Applications of Ground Penetrating Radar``, to which gprMax contributed
