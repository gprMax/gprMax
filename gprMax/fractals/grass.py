import numpy as np

from gprMax import config
from gprMax.utilities.utilities import round_value

np.seterr(divide="raise")


class Grass:
    """Geometry information for blades of grass."""

    def __init__(self, numblades, seed):
        """
        Args:
            numblades: int for the number of blades of grass.
            seed: int for seed value for random number generator.
        """

        self.numblades = numblades
        self.geometryparams = np.zeros(
            (self.numblades, 6), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.seed = seed
        self.set_geometry_parameters()

    def set_geometry_parameters(self):
        """Sets randomly defined parameters that will be used to calculate
        blade and root geometries.
        """

        self.R1 = np.random.default_rng(seed=self.seed)
        self.R2 = np.random.default_rng(seed=self.seed)
        self.R3 = np.random.default_rng(seed=self.seed)
        self.R4 = np.random.default_rng(seed=self.seed)
        self.R5 = np.random.default_rng(seed=self.seed)
        self.R6 = np.random.default_rng(seed=self.seed)

        for i in range(self.numblades):
            self.geometryparams[i, 0] = 10 + 20 * self.R1.random()
            self.geometryparams[i, 1] = 10 + 20 * self.R2.random()
            self.geometryparams[i, 2] = self.R3.choice([-1, 1])
            self.geometryparams[i, 3] = self.R4.choice([-1, 1])

    def calculate_blade_geometry(self, blade, height):
        """Calculates the x and y coordinates for a given height of grass blade.

        Args:
            blade: int for the numeric ID of grass blade.
            height: float for the height of grass blade.

        Returns:
            x, y: floats for the x and y coordinates of grass blade.
        """

        x = (
            self.geometryparams[blade, 2]
            * (height / self.geometryparams[blade, 0])
            * (height / self.geometryparams[blade, 0])
        )
        y = (
            self.geometryparams[blade, 3]
            * (height / self.geometryparams[blade, 1])
            * (height / self.geometryparams[blade, 1])
        )
        x = round_value(x)
        y = round_value(y)

        return x, y

    def calculate_root_geometry(self, root, depth):
        """Calculates the x and y coordinates for a given depth of grass root.

        Args:
            root: int for the umeric ID of grass root.
            depth: float for the depth of grass root.

        Returns:
            x, y: floats for the x and y coordinates of grass root.
        """

        self.geometryparams[root, 4] += -1 + 2 * self.R5.random()
        self.geometryparams[root, 5] += -1 + 2 * self.R6.random()
        x = round(self.geometryparams[root, 4])
        y = round(self.geometryparams[root, 5])

        return x, y
