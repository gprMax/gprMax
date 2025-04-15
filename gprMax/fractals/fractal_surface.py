import numpy as np
import numpy.typing as npt
from scipy import fftpack

from gprMax import config
from gprMax.cython.fractals_generate import generate_fractal2D

np.seterr(divide="raise")


class FractalSurface:
    """Fractal surfaces."""

    surfaceIDs = ["xminus", "xplus", "yminus", "yplus", "zminus", "zplus"]

    def __init__(self, xs, xf, ys, yf, zs, zf, dimension, seed):
        """
        Args:
            xs, xf, ys, yf, zs, zf: floats for the extent of the fractal surface
                                        (one pair of coordinates must be equal
                                        to correctly define a surface).
            dimension: float for the fractal dimension that controls the fractal
                        distribution.
            seed: int for seed value for random number generator.
        """

        self.ID = None
        self.surfaceID = None
        self.start = np.array([xs, ys, zs], dtype=np.int32)
        self.stop = np.array([xf, yf, zf], dtype=np.int32)
        self.dtype = np.dtype(np.complex128)
        self.seed = seed
        self.dimension = (
            dimension  # Fractal dimension from: http://dx.doi.org/10.1017/CBO9781139174695
        )
        self.weighting = np.array([1, 1], dtype=np.float64)
        self.fractalrange = (0, 0)
        self.filldepth = 0
        self.grass = []

    @property
    def xs(self) -> int:
        return self.start[0]

    @xs.setter
    def xs(self, value: int):
        self.start[0] = value

    @property
    def ys(self) -> int:
        return self.start[1]

    @ys.setter
    def ys(self, value: int):
        self.start[1] = value

    @property
    def zs(self) -> int:
        return self.start[2]

    @zs.setter
    def zs(self, value: int):
        self.start[2] = value

    @property
    def xf(self) -> int:
        return self.stop[0]

    @xf.setter
    def xf(self, value: int):
        self.stop[0] = value

    @property
    def yf(self) -> int:
        return self.stop[1]

    @yf.setter
    def yf(self, value: int):
        self.stop[1] = value

    @property
    def zf(self) -> int:
        return self.stop[2]

    @zf.setter
    def zf(self, value: int):
        self.stop[2] = value

    @property
    def size(self) -> npt.NDArray[np.int32]:
        return self.stop - self.start

    @property
    def nx(self) -> int:
        return self.xf - self.xs

    @property
    def ny(self) -> int:
        return self.yf - self.ys

    @property
    def nz(self) -> int:
        return self.zf - self.zs

    def get_surface_dims(self):
        """Gets the dimensions of the fractal surface based on surface plane."""

        if self.xs == self.xf:
            surfacedims = (self.ny, self.nz)
        elif self.ys == self.yf:
            surfacedims = (self.nx, self.nz)
        elif self.zs == self.zf:
            surfacedims = (self.nx, self.ny)

        return surfacedims

    def generate_fractal_surface(self):
        """Generate a 2D array with a fractal distribution."""

        surfacedims = self.get_surface_dims()

        self.fractalsurface = np.zeros(surfacedims, dtype=self.dtype)

        # Positional vector at centre of array, scaled by weighting
        v1 = np.array(
            [
                self.weighting[0] * (surfacedims[0]) / 2,
                self.weighting[1] * (surfacedims[1]) / 2,
            ]
        )

        # 2D array of random numbers to be convolved with the fractal function
        rng = np.random.default_rng(seed=self.seed)
        A = rng.standard_normal(size=(surfacedims[0], surfacedims[1]))

        # 2D FFT
        A = fftpack.fftn(A)

        # Generate fractal
        generate_fractal2D(
            surfacedims[0],
            surfacedims[1],
            0,
            0,
            surfacedims[0],
            surfacedims[1],
            config.get_model_config().ompthreads,
            self.dimension,
            self.weighting,
            v1,
            A,
            self.fractalsurface,
        )

        # Set DC component of FFT to zero
        self.fractalsurface[0, 0] = 0
        # Take the real part (numerical errors can give rise to an imaginary part)
        #  of the IFFT, and convert type to floattype. N.B calculation of fractals
        # must always be carried out at double precision, i.e. float64, complex128
        self.fractalsurface = np.real(fftpack.ifftn(self.fractalsurface)).astype(
            config.sim_config.dtypes["float_or_double"], copy=False
        )
        # Scale the fractal volume according to requested range
        fractalmin = np.amin(self.fractalsurface)
        fractalmax = np.amax(self.fractalsurface)
        fractalrange = fractalmax - fractalmin
        self.fractalsurface = (
            self.fractalsurface * ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange)
            + self.fractalrange[0]
            - ((self.fractalrange[1] - self.fractalrange[0]) / fractalrange) * fractalmin
        )
