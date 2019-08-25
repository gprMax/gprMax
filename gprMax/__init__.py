"""
======
gprMax
======

Electromagnetic wave propagation simulation software.

"""

from ._version import __version__
from .cmds_single_use import Discretisation
from .cmds_single_use import Domain
from .cmds_single_use import TimeWindow
from .cmds_single_use import Messages
from .cmds_single_use import Title
from .cmds_single_use import NumThreads
from .cmds_single_use import TimeStepStabilityFactor
from .cmds_single_use import PMLCells
from .cmds_single_use import SrcSteps
from .cmds_single_use import RxSteps
from .cmds_single_use import ExcitationFile
from .cmds_multiple import Waveform
from .cmds_multiple import VoltageSource
from .cmds_multiple import HertzianDipole
from .cmds_multiple import MagneticDipole
from .cmds_multiple import TransmissionLine
from .cmds_multiple import Rx
from .cmds_multiple import RxArray
from .cmds_multiple import Snapshot
from .cmds_multiple import Material
from .cmds_multiple import AddDebyeDispersion
from .cmds_multiple import AddLorentzDispersion
from .cmds_multiple import AddDrudeDispersion
from .cmds_multiple import SoilPeplinski
from .cmds_multiple import GeometryView
from .cmds_multiple import GeometryObjectsWrite
from .cmds_multiple import PMLCFS
from .subgrids.user_objects import SubGridHSG
from .subgrids.user_objects import ReferenceRx

# import geometry commands
from .cmds_geometry.edge import Edge
from .cmds_geometry.plate import Plate
from .cmds_geometry.triangle import Triangle
from .cmds_geometry.box import Box
from .cmds_geometry.cylinder import Cylinder
from .cmds_geometry.cylindrical_sector import CylindricalSector
from .cmds_geometry.sphere import Sphere
from .cmds_geometry.fractal_box import FractalBox
from .cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from .cmds_geometry.add_surface_water import AddSurfaceWater
from .cmds_geometry.add_grass import AddGrass

from .scene import Scene
from .gprMax import run as run

import gprMax.config as config

__name__ = 'gprMax'
