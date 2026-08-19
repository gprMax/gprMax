"""
======
gprMax
======

Electromagnetic wave propagation simulation software.

"""

import gprMax.config as config

from ._version import __version__
from .gprMax import run as run
from .ntff import (
    ExperimentalMask,
    SymmetryCompletion,
    evaluate_saved_surface_dft,
    spherical_observation_points,
)
from .scene import Scene
from .studies import (
    EigenmodeStudy,
    EigenmodeStudyResult,
    GPRStudy,
    ModalWeight,
    ObjectState,
    PlaneWaveStudy,
    PortStudy,
    PortStudyResult,
    SourceStudy,
    Study,
    StudyCase,
    combine_embedded_modal_responses,
    modal_array_weights,
)
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
from .user_objects.cmds_geometry.magnetic_edge import MagneticEdge
from .user_objects.cmds_geometry.plate import Plate
from .user_objects.cmds_geometry.sphere import Sphere
from .user_objects.cmds_geometry.thin_wire import ThinWire
from .user_objects.cmds_geometry.triangle import Triangle
from .user_objects.cmds_multiuse import (
    PMLCFS,
    AddDebyeDispersion,
    AddDrudeDispersion,
    AddLorentzDispersion,
    DiscretePlaneWaveAngles,
    DiscretePlaneWaveAxial,
    DiscretePlaneWaveVector,
    EigenmodeBand,
    EigenmodeExcitation,
    EigenmodePort,
    ExcitationFile,
    HertzianDipole,
    MagneticDipole,
    MagneticFrillSource,
    Material,
    MaterialCrim,
    MaterialDensity,
    MaterialFromDatabase,
    MaterialList,
    MaterialRange,
    NetworkExcitation,
    NetworkTerminal,
    PMLSlab,
    RationalNetwork,
    Rx,
    RxArray,
    SoilPeplinski,
    SymmetryBoundary,
    TransmissionLine,
    VirtualWaveguide,
    VoltageSource,
    Waveform,
)
from .user_objects.cmds_output import (
    SAR,
    GeometryObjectsWrite,
    GeometryView,
    KSIRAntennaPorts,
    KSIRFarField,
    KSIRFarFieldArray,
    KSIRFrequencyRx,
    KSIRFrequencyRxArray,
    KSIRFrequencyRxSpherical,
    KSIRFrequencyTransform,
    KSIRTimeRx,
    KSIRTimeRxArray,
    KSIRTimeRxSpherical,
    NetworkPort,
    NTFFAntennaPorts,
    NTFFFarField,
    NTFFFarFieldArray,
    NTFFFrequencyTransform,
    NTFFSurface,
    NTFFTimeFarField,
    NTFFTimeFarFieldArray,
    Radiometry,
    Snapshot,
)
from .user_objects.cmds_singleuse import (
    Discretisation,
    DispersiveAveraging,
    Domain,
    DomainMode,
    MagneticAveraging,
    OMPThreads,
    OutputDir,
    PMLFormulation,
    PMLProps,
    PMLThickness,
    RxSteps,
    SrcSteps,
    TimeStepStabilityFactor,
    TimeWindow,
    Title,
)

__name__ = "gprMax"
