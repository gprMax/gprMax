import logging

import h5py
import numpy as np

from gprMax.utilities.logging import logging_config

logger = logging.getLogger(__name__)
logging_config(name=__name__)


FIELD_COMPONENTS_BASE_PATH = "/rxs/rx1/"


def get_data_from_h5_file(h5_filepath):
    with h5py.File(h5_filepath, "r") as h5_file:
        # Get available field output component names and datatype
        field_components = list(h5_file[FIELD_COMPONENTS_BASE_PATH].keys())
        dtype = h5_file[FIELD_COMPONENTS_BASE_PATH + field_components[0]].dtype
        shape = h5_file[FIELD_COMPONENTS_BASE_PATH + str(field_components[0])].shape

        # Arrays for storing field data
        if len(shape) == 1:
            data = np.zeros((h5_file.attrs["Iterations"], len(field_components)), dtype=dtype)
        else:  # Merged B-scan data
            data = np.zeros((h5_file.attrs["Iterations"], len(field_components), shape[1]), dtype=dtype)
        for index, field_component in enumerate(field_components):
            data[:, index] = h5_file[FIELD_COMPONENTS_BASE_PATH + str(field_component)]
            if np.any(np.isnan(data[:, index])):
                logger.exception("Data contains NaNs")
                raise ValueError

        max_time = (h5_file.attrs["Iterations"] - 1) * h5_file.attrs["dt"] / 1e-9
        time = np.linspace(0, max_time, num=h5_file.attrs["Iterations"])

    return time, data


def calculate_diffs(test_data, ref_data):
    diffs = np.zeros(test_data.shape, dtype=np.float64)
    for i in range(test_data.shape[1]):
        maxi = np.amax(np.abs(ref_data[:, i]))
        diffs[:, i] = np.divide(
            np.abs(ref_data[:, i] - test_data[:, i]), maxi, out=np.zeros_like(ref_data[:, i]), where=maxi != 0
        )  # Replace any division by zero with zero

        # Calculate power (ignore warning from taking a log of any zero values)
        with np.errstate(divide="ignore"):
            diffs[:, i] = 20 * np.log10(diffs[:, i])
        # Replace any NaNs or Infs from zero division
        diffs[:, i][np.invert(np.isfinite(diffs[:, i]))] = 0

    return diffs
