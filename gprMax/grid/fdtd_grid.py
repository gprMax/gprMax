import decimal
from collections import OrderedDict

import numpy as np

from gprMax import config
from gprMax.pml import PML
from gprMax.utilities.utilities import round_value


class FDTDGrid:
    """Holds attributes associated with entire grid. A convenient way for
    accessing regularly used parameters.
    """

    def __init__(self):
        self.title = ""
        self.name = "main_grid"
        self.mem_use = 0

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dt = 0
        self.dt_mod = None  # Time step stability factor
        self.iteration = 0  # Current iteration number
        self.iterations = 0  # Total number of iterations
        self.timewindow = 0

        # PML parameters - set some defaults to use if not user provided
        self.pmls = {}
        self.pmls["formulation"] = "HORIPML"
        self.pmls["cfs"] = []
        self.pmls["slabs"] = []
        # Ordered dictionary required so *updating* the PMLs always follows the
        # same order (the order for *building* PMLs does not matter). The order
        # itself does not matter, however, if must be the same from model to
        # model otherwise the numerical precision from adding the PML
        # corrections will be different.
        self.pmls["thickness"] = OrderedDict((key, 10) for key in PML.boundaryIDs)

        self.materials = []
        self.mixingmodels = []
        self.averagevolumeobjects = True
        self.fractalvolumes = []
        self.geometryviews = []
        self.geometryobjectswrite = []
        self.waveforms = []
        self.voltagesources = []
        self.hertziandipoles = []
        self.magneticdipoles = []
        self.transmissionlines = []
        self.rxs = []
        self.srcsteps = [0, 0, 0]
        self.rxsteps = [0, 0, 0]
        self.snapshots = []
        self.subgrids = []

    def within_bounds(self, p):
        if p[0] < 0 or p[0] > self.nx:
            raise ValueError("x")
        if p[1] < 0 or p[1] > self.ny:
            raise ValueError("y")
        if p[2] < 0 or p[2] > self.nz:
            raise ValueError("z")

    def discretise_point(self, p):
        x = round_value(float(p[0]) / self.dx)
        y = round_value(float(p[1]) / self.dy)
        z = round_value(float(p[2]) / self.dz)
        return (x, y, z)

    def round_to_grid(self, p):
        p = self.discretise_point(p)
        p_r = (p[0] * self.dx, p[1] * self.dy, p[2] * self.dz)
        return p_r

    def within_pml(self, p):
        if (
            p[0] < self.pmls["thickness"]["x0"]
            or p[0] > self.nx - self.pmls["thickness"]["xmax"]
            or p[1] < self.pmls["thickness"]["y0"]
            or p[1] > self.ny - self.pmls["thickness"]["ymax"]
            or p[2] < self.pmls["thickness"]["z0"]
            or p[2] > self.nz - self.pmls["thickness"]["zmax"]
        ):
            return True
        else:
            return False

    def initialise_geometry_arrays(self):
        """Initialise an array for volumetric material IDs (solid);
            boolean arrays for specifying whether materials can have dielectric
            smoothing (rigid); and an array for cell edge IDs (ID).
        Solid and ID arrays are initialised to free_space (one);
            rigid arrays to allow dielectric smoothing (zero).
        """
        self.solid = np.ones((self.nx, self.ny, self.nz), dtype=np.uint32)
        self.rigidE = np.zeros((12, self.nx, self.ny, self.nz), dtype=np.int8)
        self.rigidH = np.zeros((6, self.nx, self.ny, self.nz), dtype=np.int8)
        self.ID = np.ones((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)
        self.IDlookup = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}

    def initialise_field_arrays(self):
        """Initialise arrays for the electric and magnetic field components."""
        self.Ex = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=config.sim_config.dtypes["float_or_double"])
        self.Ey = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=config.sim_config.dtypes["float_or_double"])
        self.Ez = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=config.sim_config.dtypes["float_or_double"])
        self.Hx = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=config.sim_config.dtypes["float_or_double"])
        self.Hy = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=config.sim_config.dtypes["float_or_double"])
        self.Hz = np.zeros((self.nx + 1, self.ny + 1, self.nz + 1), dtype=config.sim_config.dtypes["float_or_double"])

    def initialise_std_update_coeff_arrays(self):
        """Initialise arrays for storing update coefficients."""
        self.updatecoeffsE = np.zeros((len(self.materials), 5), dtype=config.sim_config.dtypes["float_or_double"])
        self.updatecoeffsH = np.zeros((len(self.materials), 5), dtype=config.sim_config.dtypes["float_or_double"])

    def initialise_dispersive_arrays(self):
        """Initialise field arrays when there are dispersive materials present."""
        self.Tx = np.zeros(
            (config.get_model_config().materials["maxpoles"], self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.Ty = np.zeros(
            (config.get_model_config().materials["maxpoles"], self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.Tz = np.zeros(
            (config.get_model_config().materials["maxpoles"], self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.get_model_config().materials["dispersivedtype"],
        )

    def initialise_dispersive_update_coeff_array(self):
        """Initialise array for storing update coefficients when there are dispersive
        materials present.
        """
        self.updatecoeffsdispersive = np.zeros(
            (len(self.materials), 3 * config.get_model_config().materials["maxpoles"]),
            dtype=config.get_model_config().materials["dispersivedtype"],
        )

    def reset_fields(self):
        """Clear arrays for field components and PMLs."""
        # Clear arrays for field components
        self.initialise_field_arrays()
        if config.get_model_config().materials["maxpoles"] > 0:
            self.initialise_dispersive_arrays()

        # Clear arrays for fields in PML
        for pml in self.pmls["slabs"]:
            pml.initialise_field_arrays()

    def mem_est_basic(self):
        """Estimates the amount of memory (RAM) required for grid arrays.

        Returns:
            mem_use: int of memory (bytes).
        """

        solidarray = self.nx * self.ny * self.nz * np.dtype(np.uint32).itemsize

        # 12 x rigidE array components + 6 x rigidH array components
        rigidarrays = (12 + 6) * self.nx * self.ny * self.nz * np.dtype(np.int8).itemsize

        # 6 x field arrays + 6 x ID arrays
        fieldarrays = (
            (6 + 6)
            * (self.nx + 1)
            * (self.ny + 1)
            * (self.nz + 1)
            * np.dtype(config.sim_config.dtypes["float_or_double"]).itemsize
        )

        # PML arrays
        pmlarrays = 0
        for k, v in self.pmls["thickness"].items():
            if v > 0:
                if "x" in k:
                    pmlarrays += (v + 1) * self.ny * (self.nz + 1)
                    pmlarrays += (v + 1) * (self.ny + 1) * self.nz
                    pmlarrays += v * self.ny * (self.nz + 1)
                    pmlarrays += v * (self.ny + 1) * self.nz
                elif "y" in k:
                    pmlarrays += self.nx * (v + 1) * (self.nz + 1)
                    pmlarrays += (self.nx + 1) * (v + 1) * self.nz
                    pmlarrays += (self.nx + 1) * v * self.nz
                    pmlarrays += self.nx * v * (self.nz + 1)
                elif "z" in k:
                    pmlarrays += self.nx * (self.ny + 1) * (v + 1)
                    pmlarrays += (self.nx + 1) * self.ny * (v + 1)
                    pmlarrays += (self.nx + 1) * self.ny * v
                    pmlarrays += self.nx * (self.ny + 1) * v

        mem_use = int(fieldarrays + solidarray + rigidarrays + pmlarrays)

        return mem_use

    def mem_est_dispersive(self):
        """Estimates the amount of memory (RAM) required for dispersive grid arrays.

        Returns:
            mem_use: int of memory (bytes).
        """

        mem_use = int(
            3
            * config.get_model_config().materials["maxpoles"]
            * (self.nx + 1)
            * (self.ny + 1)
            * (self.nz + 1)
            * np.dtype(config.get_model_config().materials["dispersivedtype"]).itemsize
        )
        return mem_use

    def mem_est_fractals(self):
        """Estimates the amount of memory (RAM) required to build any objects
            which use the FractalVolume/FractalSurface classes.

        Returns:
            mem_use: int of memory (bytes).
        """

        mem_use = 0

        for vol in self.fractalvolumes:
            mem_use += vol.nx * vol.ny * vol.nz * vol.dtype.itemsize
            for surface in vol.fractalsurfaces:
                surfacedims = surface.get_surface_dims()
                mem_use += surfacedims[0] * surfacedims[1] * surface.dtype.itemsize

        return mem_use

    def tmx(self):
        """Add PEC boundaries to invariant direction in 2D TMx mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ey & Ez components
        self.ID[1, 0, :, :] = 0
        self.ID[1, 1, :, :] = 0
        self.ID[2, 0, :, :] = 0
        self.ID[2, 1, :, :] = 0

    def tmy(self):
        """Add PEC boundaries to invariant direction in 2D TMy mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ex & Ez components
        self.ID[0, :, 0, :] = 0
        self.ID[0, :, 1, :] = 0
        self.ID[2, :, 0, :] = 0
        self.ID[2, :, 1, :] = 0

    def tmz(self):
        """Add PEC boundaries to invariant direction in 2D TMz mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ex & Ey components
        self.ID[0, :, :, 0] = 0
        self.ID[0, :, :, 1] = 0
        self.ID[1, :, :, 0] = 0
        self.ID[1, :, :, 1] = 0

    def calculate_dt(self):
        """Calculate time step at the CFL limit."""
        if config.get_model_config().mode == "2D TMx":
            self.dt = 1 / (config.sim_config.em_consts["c"] * np.sqrt((1 / self.dy**2) + (1 / self.dz**2)))
        elif config.get_model_config().mode == "2D TMy":
            self.dt = 1 / (config.sim_config.em_consts["c"] * np.sqrt((1 / self.dx**2) + (1 / self.dz**2)))
        elif config.get_model_config().mode == "2D TMz":
            self.dt = 1 / (config.sim_config.em_consts["c"] * np.sqrt((1 / self.dx**2) + (1 / self.dy**2)))
        else:
            self.dt = 1 / (
                config.sim_config.em_consts["c"] * np.sqrt((1 / self.dx**2) + (1 / self.dy**2) + (1 / self.dz**2))
            )

        # Round down time step to nearest float with precision one less than
        # hardware maximum. Avoids inadvertently exceeding the CFL due to
        # binary representation of floating point number.
        self.dt = round_value(self.dt, decimalplaces=decimal.getcontext().prec - 1)
