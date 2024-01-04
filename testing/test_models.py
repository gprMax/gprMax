# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import logging
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import gprMax
from testing.analytical_solutions import hertzian_dipole_fs

logger = logging.getLogger(__name__)

if sys.platform == "linux":
    plt.switch_backend("agg")


"""Compare field outputs

    Usage:
        cd gprMax
        python -m testing.test_models
"""

# Specify directory with set of models to test
modelset = "models_basic"
# modelset += 'models_advanced'

basepath = Path(__file__).parents[0] / modelset

# List of available basic test models
testmodels = [
    "hertzian_dipole_fs_analytical",
    "2D_ExHyHz",
    "2D_EyHxHz",
    "2D_EzHxHy",
    "cylinder_Ascan_2D",
    "hertzian_dipole_fs",
    "hertzian_dipole_hs",
    "hertzian_dipole_dispersive",
    "magnetic_dipole_fs",
]

# List of available advanced test models
# testmodels = ['antenna_GSSI_1500_fs', 'antenna_MALA_1200_fs']

# Select a specific model if desired
# testmodels = [testmodels[0]]
testresults = dict.fromkeys(testmodels)
path = "/rxs/rx1/"

# Minimum value of difference to plot (dB)
plotmin = -160

for i, model in enumerate(testmodels):
    testresults[model] = {}

    # Run model
    file = basepath / model / model
    gprMax.run(inputfile=file.with_suffix(".in"), gpu=None, opencl=None)

    # Special case for analytical comparison
    if model == "hertzian_dipole_fs_analytical":
        # Get output for model file
        filetest = h5py.File(file.with_suffix(".h5"), "r")
        testresults[model]["Test version"] = filetest.attrs["gprMax"]

        # Get available field output component names
        outputstest = list(filetest[path].keys())

        # Arrays for storing time
        float_or_double = filetest[path + outputstest[0]].dtype
        timetest = (
            np.linspace(0, (filetest.attrs["Iterations"] - 1) * filetest.attrs["dt"], num=filetest.attrs["Iterations"])
            / 1e-9
        )
        timeref = timetest

        # Arrays for storing field data
        datatest = np.zeros((filetest.attrs["Iterations"], len(outputstest)), dtype=float_or_double)
        for ID, name in enumerate(outputstest):
            datatest[:, ID] = filetest[path + str(name)][:]
            if np.any(np.isnan(datatest[:, ID])):
                logger.exception("Test data contains NaNs")
                raise ValueError

        # Tx/Rx position to feed to analytical solution
        rxpos = filetest[path].attrs["Position"]
        txpos = filetest["/srcs/src1/"].attrs["Position"]
        rxposrelative = ((rxpos[0] - txpos[0]), (rxpos[1] - txpos[1]), (rxpos[2] - txpos[2]))

        # Analytical solution of a dipole in free space
        dataref = hertzian_dipole_fs(
            filetest.attrs["Iterations"], filetest.attrs["dt"], filetest.attrs["dx_dy_dz"], rxposrelative
        )
        filetest.close()

    else:
        # Get output for model and reference files
        fileref = f"{file.stem}_ref"
        fileref = file.parent / Path(fileref)
        fileref = h5py.File(fileref.with_suffix(".h5"), "r")
        filetest = h5py.File(file.with_suffix(".h5"), "r")
        testresults[model]["Ref version"] = fileref.attrs["gprMax"]
        testresults[model]["Test version"] = filetest.attrs["gprMax"]

        # Get available field output component names
        outputsref = list(fileref[path].keys())
        outputstest = list(filetest[path].keys())
        if outputsref != outputstest:
            logger.exception("Field output components do not match reference solution")
            raise ValueError

        # Check that type of float used to store fields matches
        float_or_doubleref = fileref[path + outputsref[0]].dtype
        float_or_doubletest = filetest[path + outputstest[0]].dtype
        if float_or_doubleref != float_or_doubletest:
            logger.warning(
                f"Type of floating point number in test model "
                f"({float_or_doubletest}) does not "
                f"match type in reference solution ({float_or_doubleref})\n"
            )

        # Arrays for storing time
        timeref = np.zeros((fileref.attrs["Iterations"]), dtype=float_or_doubleref)
        timeref = (
            np.linspace(0, (fileref.attrs["Iterations"] - 1) * fileref.attrs["dt"], num=fileref.attrs["Iterations"])
            / 1e-9
        )
        timetest = np.zeros((filetest.attrs["Iterations"]), dtype=float_or_doubletest)
        timetest = (
            np.linspace(0, (filetest.attrs["Iterations"] - 1) * filetest.attrs["dt"], num=filetest.attrs["Iterations"])
            / 1e-9
        )

        # Arrays for storing field data
        dataref = np.zeros((fileref.attrs["Iterations"], len(outputsref)), dtype=float_or_doubleref)
        datatest = np.zeros((filetest.attrs["Iterations"], len(outputstest)), dtype=float_or_doubletest)
        for ID, name in enumerate(outputsref):
            dataref[:, ID] = fileref[path + str(name)][:]
            datatest[:, ID] = filetest[path + str(name)][:]
            if np.any(np.isnan(datatest[:, ID])):
                logger.exception("Test data contains NaNs")
                raise ValueError

        fileref.close()
        filetest.close()

    # Diffs
    datadiffs = np.zeros(datatest.shape, dtype=np.float64)
    for i in range(len(outputstest)):
        maxi = np.amax(np.abs(dataref[:, i]))
        datadiffs[:, i] = np.divide(
            np.abs(dataref[:, i] - datatest[:, i]), maxi, out=np.zeros_like(dataref[:, i]), where=maxi != 0
        )  # Replace any division by zero with zero

        # Calculate power (ignore warning from taking a log of any zero values)
        with np.errstate(divide="ignore"):
            datadiffs[:, i] = 20 * np.log10(datadiffs[:, i])
        # Replace any NaNs or Infs from zero division
        datadiffs[:, i][np.invert(np.isfinite(datadiffs[:, i]))] = 0

    # Store max difference
    maxdiff = np.amax(np.amax(datadiffs))
    testresults[model]["Max diff"] = maxdiff

    # Plot datasets
    fig1, ((ex1, hx1), (ey1, hy1), (ez1, hz1)) = plt.subplots(
        nrows=3,
        ncols=2,
        sharex=False,
        sharey="col",
        subplot_kw=dict(xlabel="Time [ns]"),
        num=model + ".in",
        figsize=(20, 10),
        facecolor="w",
        edgecolor="w",
    )
    ex1.plot(timetest, datatest[:, 0], "r", lw=2, label=model)
    ex1.plot(timeref, dataref[:, 0], "g", lw=2, ls="--", label=f"{model}(Ref)")
    ey1.plot(timetest, datatest[:, 1], "r", lw=2, label=model)
    ey1.plot(timeref, dataref[:, 1], "g", lw=2, ls="--", label=f"{model}(Ref)")
    ez1.plot(timetest, datatest[:, 2], "r", lw=2, label=model)
    ez1.plot(timeref, dataref[:, 2], "g", lw=2, ls="--", label=f"{model}(Ref)")
    hx1.plot(timetest, datatest[:, 3], "r", lw=2, label=model)
    hx1.plot(timeref, dataref[:, 3], "g", lw=2, ls="--", label=f"{model}(Ref)")
    hy1.plot(timetest, datatest[:, 4], "r", lw=2, label=model)
    hy1.plot(timeref, dataref[:, 4], "g", lw=2, ls="--", label=f"{model}(Ref)")
    hz1.plot(timetest, datatest[:, 5], "r", lw=2, label=model)
    hz1.plot(timeref, dataref[:, 5], "g", lw=2, ls="--", label=f"{model}(Ref)")
    ylabels = [
        "$E_x$, field strength [V/m]",
        "$H_x$, field strength [A/m]",
        "$E_y$, field strength [V/m]",
        "$H_y$, field strength [A/m]",
        "$E_z$, field strength [V/m]",
        "$H_z$, field strength [A/m]",
    ]
    for i, ax in enumerate(fig1.axes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, np.amax(timetest))
        ax.grid()
        ax.legend()

    # Plot diffs
    fig2, ((ex2, hx2), (ey2, hy2), (ez2, hz2)) = plt.subplots(
        nrows=3,
        ncols=2,
        sharex=False,
        sharey="col",
        subplot_kw=dict(xlabel="Time [ns]"),
        num="Diffs: " + model + ".in",
        figsize=(20, 10),
        facecolor="w",
        edgecolor="w",
    )
    ex2.plot(timeref, datadiffs[:, 0], "r", lw=2, label="Ex")
    ey2.plot(timeref, datadiffs[:, 1], "r", lw=2, label="Ey")
    ez2.plot(timeref, datadiffs[:, 2], "r", lw=2, label="Ez")
    hx2.plot(timeref, datadiffs[:, 3], "r", lw=2, label="Hx")
    hy2.plot(timeref, datadiffs[:, 4], "r", lw=2, label="Hy")
    hz2.plot(timeref, datadiffs[:, 5], "r", lw=2, label="Hz")
    ylabels = [
        "$E_x$, difference [dB]",
        "$H_x$, difference [dB]",
        "$E_y$, difference [dB]",
        "$H_y$, difference [dB]",
        "$E_z$, difference [dB]",
        "$H_z$, difference [dB]",
    ]
    for i, ax in enumerate(fig2.axes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, np.amax(timetest))
        ax.set_ylim([plotmin, np.amax(np.amax(datadiffs))])
        ax.grid()

    # Save a PDF/PNG of the figure
    filediffs = f"{file.stem}_diffs"
    filediffs = file.parent / Path(filediffs)
    # fig1.savefig(file.with_suffix('.pdf'), dpi=None, format='pdf',
    #              bbox_inches='tight', pad_inches=0.1)
    # fig2.savefig(savediffs.with_suffix('.pdf'), dpi=None, format='pdf',
    #              bbox_inches='tight', pad_inches=0.1)
    fig1.savefig(file.with_suffix(".png"), dpi=150, format="png", bbox_inches="tight", pad_inches=0.1)
    fig2.savefig(filediffs.with_suffix(".png"), dpi=150, format="png", bbox_inches="tight", pad_inches=0.1)

# Summary of results
for name, data in sorted(testresults.items()):
    if "analytical" in name:
        logger.info(
            f"Test '{name}.in' using v.{data['Test version']} compared "
            f"to analytical solution. Max difference {data['Max diff']:.2f}dB."
        )
    else:
        logger.info(
            f"Test '{name}.in' using v.{data['Test version']} compared to "
            f"reference solution using v.{data['Ref version']}. Max difference "
            f"{data['Max diff']:.2f}dB."
        )
