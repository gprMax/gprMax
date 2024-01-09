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

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from pytest_easyMPI import mpi_parallel

import gprMax

from gprMax.utilities.logging import logging_config

logger = logging.getLogger(__name__)
logging_config(name=__name__)

if sys.platform == "linux":
    plt.switch_backend("agg")


"""Compare field outputs

    Usage:
        cd gprMax
        pytest tests/test_models.py
"""

# Specify directory containing basic models to test
BSCAN_MODELS_DIRECTORY = Path(__file__).parent / "data" / "models_bscan"

# List of available basic test models
BSCAN_MODELS = [
    "cylinder_Bscan_2D",
]

FIELD_COMPONENTS_BASE_PATH = "/rxs/rx1/"


def run_test(model_name, input_base, data_directory, analytical_func=None, gpu=None, opencl=None):
    input_filepath = input_base.with_suffix(".in")
    reference_filepath = Path(f"{input_base}_ref.h5")
    
    output_base = data_directory / model_name
    output_filepath = output_base.with_suffix(".h5")

    # Run model
    gprMax.run(inputfile=input_filepath, outputfile=output_filepath, gpu=gpu, opencl=opencl, n=31, mpi=True)


@pytest.mark.parametrize("model", BSCAN_MODELS)
@pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
@mpi_parallel("n")
def test_bscan_models(model, datadir, n):

    base_filepath = Path(BSCAN_MODELS_DIRECTORY, model, model)
    run_test(model, base_filepath, datadir)
