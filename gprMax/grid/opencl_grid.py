from importlib import import_module

import numpy as np

from gprMax import config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.utilities.utilities import fft_power, round_value


class OpenCLGrid(FDTDGrid):
    """Additional grid methods for solving on compute device using OpenCL."""

    def __init__(self):
        super().__init__()

        self.clarray = import_module("pyopencl.array")

    def htod_geometry_arrays(self, queue):
        """Initialise an array for cell edge IDs (ID) on compute device.

        Args:
            queue: pyopencl queue.
        """

        self.ID_dev = self.clarray.to_device(queue, self.ID)

    def htod_field_arrays(self, queue):
        """Initialise field arrays on compute device.

        Args:
            queue: pyopencl queue.
        """

        self.Ex_dev = self.clarray.to_device(queue, self.Ex)
        self.Ey_dev = self.clarray.to_device(queue, self.Ey)
        self.Ez_dev = self.clarray.to_device(queue, self.Ez)
        self.Hx_dev = self.clarray.to_device(queue, self.Hx)
        self.Hy_dev = self.clarray.to_device(queue, self.Hy)
        self.Hz_dev = self.clarray.to_device(queue, self.Hz)

    def htod_dispersive_arrays(self, queue):
        """Initialise dispersive material coefficient arrays on compute device.

        Args:
            queue: pyopencl queue.
        """

        self.updatecoeffsdispersive_dev = self.clarray.to_device(queue, self.updatecoeffsdispersive)
        # self.updatecoeffsdispersive_dev = self.clarray.to_device(queue, np.ones((95,95,95), dtype=np.float32))
        self.Tx_dev = self.clarray.to_device(queue, self.Tx)
        self.Ty_dev = self.clarray.to_device(queue, self.Ty)
        self.Tz_dev = self.clarray.to_device(queue, self.Tz)


def dispersion_analysis(G):
    """Analysis of numerical dispersion (Taflove et al, 2005, p112) -
        worse case of maximum frequency and minimum wavelength

    Args:
        G: FDTDGrid class describing a grid in a model.

    Returns:
        results: dict of results from dispersion analysis.
    """

    # deltavp: physical phase velocity error (percentage)
    # N: grid sampling density
    # material: material with maximum permittivity
    # maxfreq: maximum significant frequency
    # error: error message
    results = {"deltavp": None, "N": None, "material": None, "maxfreq": [], "error": ""}

    # Find maximum significant frequency
    if G.waveforms:
        for waveform in G.waveforms:
            if waveform.type in ["sine", "contsine"]:
                results["maxfreq"].append(4 * waveform.freq)

            elif waveform.type == "impulse":
                results["error"] = "impulse waveform used."

            elif waveform.type == "user":
                results["error"] = "user waveform detected."

            else:
                # Time to analyse waveform - 4*pulse_width as using entire
                # time window can result in demanding FFT
                waveform.calculate_coefficients()
                iterations = round_value(4 * waveform.chi / G.dt)
                iterations = min(iterations, G.iterations)
                waveformvalues = np.zeros(G.iterations)
                for iteration in range(G.iterations):
                    waveformvalues[iteration] = waveform.calculate_value(iteration * G.dt, G.dt)

                # Ensure source waveform is not being overly truncated before attempting any FFT
                if np.abs(waveformvalues[-1]) < np.abs(np.amax(waveformvalues)) / 100:
                    # FFT
                    freqs, power = fft_power(waveformvalues, G.dt)
                    # Get frequency for max power
                    freqmaxpower = np.where(np.isclose(power, 0))[0][0]

                    # Set maximum frequency to a threshold drop from maximum power, ignoring DC value
                    try:
                        freqthres = (
                            np.where(
                                power[freqmaxpower:] < -config.get_model_config().numdispersion["highestfreqthres"]
                            )[0][0]
                            + freqmaxpower
                        )
                        results["maxfreq"].append(freqs[freqthres])
                    except ValueError:
                        results["error"] = (
                            "unable to calculate maximum power "
                            + "from waveform, most likely due to "
                            + "undersampling."
                        )

                # Ignore case where someone is using a waveform with zero amplitude, i.e. on a receiver
                elif waveform.amp == 0:
                    pass

                # If waveform is truncated don't do any further analysis
                else:
                    results["error"] = (
                        "waveform does not fit within specified " + "time window and is therefore being truncated."
                    )
    else:
        results["error"] = "no waveform detected."

    if results["maxfreq"]:
        results["maxfreq"] = max(results["maxfreq"])

        # Find minimum wavelength (material with maximum permittivity)
        maxer = 0
        matmaxer = ""
        for x in G.materials:
            if x.se != float("inf"):
                er = x.er
                # If there are dispersive materials calculate the complex
                # relative permittivity at maximum frequency and take the real part
                if x.__class__.__name__ == "DispersiveMaterial":
                    er = x.calculate_er(results["maxfreq"])
                    er = er.real
                if er > maxer:
                    maxer = er
                    matmaxer = x.ID
        results["material"] = next(x for x in G.materials if x.ID == matmaxer)

        # Minimum velocity
        minvelocity = config.c / np.sqrt(maxer)

        # Minimum wavelength
        minwavelength = minvelocity / results["maxfreq"]

        # Maximum spatial step
        if "3D" in config.get_model_config().mode:
            delta = max(G.dx, G.dy, G.dz)
        elif "2D" in config.get_model_config().mode:
            if G.nx == 1:
                delta = max(G.dy, G.dz)
            elif G.ny == 1:
                delta = max(G.dx, G.dz)
            elif G.nz == 1:
                delta = max(G.dx, G.dy)

        # Courant stability factor
        S = (config.c * G.dt) / delta

        # Grid sampling density
        results["N"] = minwavelength / delta

        # Check grid sampling will result in physical wave propagation
        if int(np.floor(results["N"])) >= config.get_model_config().numdispersion["mingridsampling"]:
            # Numerical phase velocity
            vp = np.pi / (results["N"] * np.arcsin((1 / S) * np.sin((np.pi * S) / results["N"])))

            # Physical phase velocity error (percentage)
            results["deltavp"] = (((vp * config.c) - config.c) / config.c) * 100

        # Store rounded down value of grid sampling density
        results["N"] = int(np.floor(results["N"]))

    return results
