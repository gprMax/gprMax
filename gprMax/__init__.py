"""
======
gprMax
======

Electromagnetic wave propagation simulation software.

"""

import gprMax.config as config

from ._version import __version__
from .gprMax import run as run
from .scene import Scene
from .subgrids.user_objects import SubGridHSG
from .user_objects.cmds_geometry.add_grass import AddGrass
from .user_objects.cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from .user_objects.cmds_geometry.add_surface_water import AddSurfaceWater
from .user_objects.cmds_geometry.box import Box
from .user_objects.cmds_geometry.cone import Cone
from .user_objects.cmds_geometry.cylinder import Cylinder
from .user_objects.cmds_geometry.cylindrical_sector import CylindricalSector
from .user_objects.cmds_geometry.edge import Edge
from .user_objects.cmds_geometry.ellipsoid import Ellipsoid
from .user_objects.cmds_geometry.fractal_box import FractalBox
from .user_objects.cmds_geometry.geometry_objects_read import GeometryObjectsRead
from .user_objects.cmds_geometry.plate import Plate
from .user_objects.cmds_geometry.sphere import Sphere
from .user_objects.cmds_geometry.triangle import Triangle
from .user_objects.cmds_multiuse import (
    PMLCFS,
    AddDebyeDispersion,
    AddDrudeDispersion,
    AddLorentzDispersion,
    ExcitationFile,
    HertzianDipole,
    MagneticDipole,
    Material,
    MaterialList,
    MaterialRange,
    Rx,
    RxArray,
    SoilPeplinski,
    TransmissionLine,
    VoltageSource,
    Waveform,
)
from .user_objects.cmds_output import GeometryObjectsWrite, GeometryView, Snapshot
from .user_objects.cmds_singleuse import (
    Discretisation,
    Domain,
    OMPThreads,
    OutputDir,
    PMLProps,
    RxSteps,
    SrcSteps,
    TimeStepStabilityFactor,
    TimeWindow,
    Title,
)

__name__ = "gprMax"
