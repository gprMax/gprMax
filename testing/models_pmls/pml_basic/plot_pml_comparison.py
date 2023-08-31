# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
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

import matplotlib.pyplot as plt
import numpy as np

from testing.diff_output_files import diff_output_files

logger = logging.getLogger(__name__)

# Create/setup plot figure
# Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colorIDs = ["#79c72e", "#5774ff", "#ff7c2c", "#4b4e80", "#d7004e", "#007545", "#ff83ec"]
colors = itertools.cycle(colorIDs)
lines = itertools.cycle(("--", ":", "-.", "-"))
markers = ["o", "d", "^", "s", "*"]

fn = Path(__file__)
basename = "pml_basic"
PMLIDs = ["off", "x0", "y0", "z0", "xmax", "ymax", "zmax"]
maxerrors = []

for x in range(len(PMLIDs)):
    file1 = fn.parent.joinpath(basename + str(x + 1) + "_CPU.h5")
    file2 = fn.parent.joinpath(basename + str(x + 1) + "_GPU.h5")
    time, datadiffs = diff_output_files(file1, file2)

    # Print maximum error value
    start = 0
    maxerrors.append(f": {np.amax(datadiffs[start::, 1]):.1f} [dB]")
    logger.info(f"{file1.name} - {file2.name}: Max. error {maxerrors[x]}")

    fig, ax = plt.subplots(
    subplot_kw=dict(xlabel="Iterations", ylabel="Error [dB]"), figsize=(20, 10), facecolor="w", edgecolor="w")

    # Plot diffs (select column to choose field component, 0-Ex, 1-Ey etc..)
    ax.plot(time[start::], datadiffs[start::, 1], color=next(colors), lw=2, ls=next(lines), label=f"{file1.name} - {file2.name}")
    ax.set_xticks(np.arange(0, 2200, step=100))
    ax.set_xlim([0, 2100])
    ax.set_yticks(np.arange(-160, 0, step=20))
    ax.set_ylim([-160, -20])
    ax.set_axisbelow(True)
    ax.grid(color=(0.75, 0.75, 0.75), linestyle="dashed")

mylegend = list(map(add, PMLIDs, maxerrors))
legend = ax.legend(mylegend, loc=1, fontsize=14)
frame = legend.get_frame()
frame.set_edgecolor("white")
frame.set_alpha(0)

plt.show()

# Save a PDF/PNG of the figure
# fig.savefig(basepath + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
# fig.savefig(savename + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
