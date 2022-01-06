"""
======
gprMax
======

Electromagnetic wave propagation simulation software.

"""

import gprMax.config as config

from ._version import __version__
from .cmds_geometry.add_grass import AddGrass
from .cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from .cmds_geometry.add_surface_water import AddSurfaceWater
from .cmds_geometry.box import Box
from .cmds_geometry.cylinder import Cylinder
from .cmds_geometry.cylindrical_sector import CylindricalSector
from .cmds_geometry.edge import Edge
from .cmds_geometry.fractal_box import FractalBox
from .cmds_geometry.geometry_objects_read import GeometryObjectsRead
from .cmds_geometry.plate import Plate
from .cmds_geometry.sphere import Sphere
from .cmds_geometry.triangle import Triangle
from .cmds_multiuse import (PMLCFS, AddDebyeDispersion, AddDrudeDispersion,
                            AddLorentzDispersion, GeometryObjectsWrite,
                            GeometryView, HertzianDipole, MagneticDipole,
                            Material, Rx, RxArray, Snapshot, SoilPeplinski,
                            TransmissionLine, VoltageSource, Waveform)
from .cmds_singleuse import (Discretisation, Domain, ExcitationFile,
                             OMPThreads, PMLCells, RxSteps, SrcSteps,
                             TimeStepStabilityFactor, TimeWindow, Title)
from .gprMax import run as run
from .hash_cmds_file import user_libs_fn_to_scene_obj
from .scene import Scene
from .subgrids.user_objects import ReferenceRx, SubGridHSG

__name__ = 'gprMax'
