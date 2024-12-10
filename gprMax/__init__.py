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
from .cmds_geometry.cone import Cone
from .cmds_geometry.cylinder import Cylinder
from .cmds_geometry.cylindrical_sector import CylindricalSector
from .cmds_geometry.edge import Edge
from .cmds_geometry.ellipsoid import Ellipsoid
from .cmds_geometry.fractal_box import FractalBox
from .cmds_geometry.geometry_objects_read import GeometryObjectsRead
from .cmds_geometry.plate import Plate
from .cmds_geometry.sphere import Sphere
from .cmds_geometry.triangle import Triangle
from .gprMax import run as run
from .scene import Scene
from .subgrids.user_objects import SubGridHSG
from .user_objects.cmds_multiuse import (
    PMLCFS,
    AddDebyeDispersion,
    AddDrudeDispersion,
    AddLorentzDispersion,
    ExcitationFile,
    GeometryObjectsWrite,
    GeometryView,
    HertzianDipole,
    MagneticDipole,
    Material,
    MaterialList,
    MaterialRange,
    Rx,
    RxArray,
    Snapshot,
    SoilPeplinski,
    Subgrid,
    TransmissionLine,
    VoltageSource,
    Waveform,
)
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
