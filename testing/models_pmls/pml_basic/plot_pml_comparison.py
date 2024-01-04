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

import itertools
import logging
from operator import add
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from testing.diff_output_files import diff_output_files

logger = logging.getLogger(__name__)

# Create/setup plot figure
# Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colorIDs = ["#79c72e", "#5774ff", "#ff7c2c", "#4b4e80", "#d7004e", "#007545"]
colors = itertools.cycle(colorIDs)
lines = itertools.cycle(("--", ":", "-.", "-"))
markers = ["o", "d", "^", "s", "*"]

fn = Path(__file__)
basename = "pml_basic"
PMLIDs = ["off", "x0", "y0", "z0", "xmax", "ymax", "zmax"]
maxerrors = []

for x, PMLID in enumerate(PMLIDs):
    file1 = fn.parent.joinpath(basename + str(x + 1) + "_CPU.h5")
    file2 = fn.parent.joinpath(basename + str(x + 1) + "_GPU.h5")
    time, datadiffs = diff_output_files(file1, file2)

    # Print maximum error value
    start = 150
    maxerrors.append(f": {np.amax(np.amax(datadiffs[start::, :])):.1f} [dB]")
    print(f"{PMLID}: Max. error {maxerrors[x]}")

    # Plot diffs
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)
    fig, ax = plt.subplots(figsize=(20, 10), facecolor="w", edgecolor="w")
    ax.remove()
    fig.suptitle(f"{PMLID}")

    outputs = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    for i, output in enumerate(outputs):
        if i < 3:
            ax = plt.subplot(gs[i, 0])
        else:
            ax = plt.subplot(gs[i - 3, 1])
        ax.plot(time[start::], datadiffs[start::, i], color=next(colors), lw=2)
        ax.set_xticks(np.arange(0, 1800, step=200))
        ax.set_xlim([0, 1600])
        ax.set_yticks(np.arange(-400, 80, step=40))
        ax.set_ylim([-400, 40])
        ax.set_axisbelow(True)
        ax.grid(color=(0.75, 0.75, 0.75), linestyle="dashed")
        ax.set_xlabel("Time [iterations]")
        ax.set_ylabel(f"{output} error [dB]")

    # Save a PDF/PNG of the figure
    fig.savefig(basename + "_diffs_" + PMLID + ".pdf", dpi=None, format="pdf", bbox_inches="tight", pad_inches=0.1)
    # fig.savefig(basename + "_diffs_" + PMLID + ".png", dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()
