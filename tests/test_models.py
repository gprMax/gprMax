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

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

import gprMax
from testing.analytical_solutions import hertzian_dipole_fs
from tests.utilities.data import get_data_from_h5_file, calculate_diffs
from tests.utilities.plotting import plot_dataset_comparison, plot_diffs

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
BASIC_MODELS_DIRECTORY = Path(__file__).parent / "data" / "models_basic"

# List of available basic test models
BASIC_MODELS = [
    "2D_ExHyHz",
    "2D_EyHxHz",
    "2D_EzHxHy",
    "cylinder_Ascan_2D",
    "hertzian_dipole_fs",
    "hertzian_dipole_hs",
    "hertzian_dipole_dispersive",
    "magnetic_dipole_fs",
]

# Specify directory containing analytical models to test
ANALYTICAL_MODELS_DIRECTORY = Path(__file__).parent / "data" / "models_analytical"

# List of available analytical models
ANALYTICAL_MODELS = ["hertzian_dipole_fs_analytical"]

FIELD_COMPONENTS_BASE_PATH = "/rxs/rx1/"


def create_ascan_comparison_plots(test_time, test_data, ref_time, ref_data, model_name, output_base):
    fig1 = plot_dataset_comparison(test_time, test_data, ref_time, ref_data, model_name)
    fig1.savefig(output_base.with_suffix(".png"), dpi=150, format="png", bbox_inches="tight", pad_inches=0.1)

    # Required to correctly calculate diffs
    assert test_time.shape == ref_time.shape
    assert np.all(test_time == ref_time)
    assert test_data.shape == ref_data.shape

    data_diffs = calculate_diffs(test_data, ref_data)

    fig2 = plot_diffs(test_time, data_diffs)
    fig2.savefig(Path(f"{output_base}_diffs.png"), dpi=150, format="png", bbox_inches="tight", pad_inches=0.1)

    logger.info(f"Output data folder: {output_base.parent}")



def run_test(model_name, input_base, data_directory, analytical_func=None, gpu=None, opencl=None):
    input_filepath = input_base.with_suffix(".in")
    reference_filepath = Path(f"{input_base}_ref.h5")
    
    output_base = data_directory / model_name
    output_filepath = output_base.with_suffix(".h5")

    # Run model
    gprMax.run(inputfile=input_filepath, outputfile=output_filepath, gpu=gpu, opencl=opencl)

    test_time, test_data = get_data_from_h5_file(output_filepath)

    if analytical_func is not None:
        ref_time = test_time
        ref_data = analytical_func(output_filepath)
    else:
        ref_time, ref_data = get_data_from_h5_file(reference_filepath)

    create_ascan_comparison_plots(test_time, test_data, ref_time, ref_data, model_name, output_base)
    
    data_diffs = calculate_diffs(test_data, ref_data)
    max_diff = round(np.max(data_diffs), 2)
    assert max_diff <= 0


def run_regression_test(request, ndarrays_regression, model_name, input_base, data_directory, gpu=None, opencl=None):
    input_filepath = input_base.with_suffix(".in")
    
    output_dir = data_directory / request.node.name
    output_dir.mkdir(exist_ok=True)
    output_base = output_dir / model_name
    output_filepath = output_base.with_suffix(".h5")
    reference_filepath = output_base.with_suffix(".npz")

    # Run model
    gprMax.run(inputfile=input_filepath, outputfile=output_filepath, gpu=gpu, opencl=opencl)

    test_time, test_data = get_data_from_h5_file(output_filepath)

    # May not exist if first time running the regression test
    if os.path.exists(reference_filepath):
        reference_file = np.load(reference_filepath)

        ref_time = reference_file["time"]
        ref_data = reference_file["data"]

        create_ascan_comparison_plots(test_time, test_data, ref_time, ref_data, model_name, output_base)

    ndarrays_regression.check({"time": test_time, "data": test_data}, basename=os.path.relpath(output_base, data_directory))


def calc_hertzian_dipole_fs_analytical_solution(filepath):
    with h5py.File(filepath, "r") as file:
        # Tx/Rx position to feed to analytical solution
        rx_pos = file[FIELD_COMPONENTS_BASE_PATH].attrs["Position"]
        tx_pos = file["/srcs/src1/"].attrs["Position"]
        rx_pos_relative = ((rx_pos[0] - tx_pos[0]), (rx_pos[1] - tx_pos[1]), (rx_pos[2] - tx_pos[2]))

        # Analytical solution of a dipole in free space
        data = hertzian_dipole_fs(
            file.attrs["Iterations"], file.attrs["dt"], file.attrs["dx_dy_dz"], rx_pos_relative
        )
    return data


@pytest.mark.parametrize("model", BASIC_MODELS)
def test_basic_models(model, datadir):

    base_filepath = Path(BASIC_MODELS_DIRECTORY, model, model)
    run_test(model, base_filepath, datadir)


@pytest.mark.parametrize("model", ANALYTICAL_MODELS)
def test_analyitical_models(datadir, model):

    base_filepath = Path(ANALYTICAL_MODELS_DIRECTORY, model)
    run_test(model, base_filepath, datadir, analytical_func=calc_hertzian_dipole_fs_analytical_solution)


@pytest.mark.parametrize("model", BASIC_MODELS)
def test_basic_models_regression(request, ndarrays_regression, datadir, model):

    base_filepath = Path(BASIC_MODELS_DIRECTORY, model, model)
    run_regression_test(request, ndarrays_regression, model, base_filepath, datadir)


@pytest.mark.parametrize("model", ANALYTICAL_MODELS)
def test_analytical_models_regression(request, ndarrays_regression, datadir, model):

    base_filepath = Path(ANALYTICAL_MODELS_DIRECTORY, model)
    run_regression_test(request, ndarrays_regression, model, base_filepath, datadir)