# Copyright (C) 2015, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import numpy as np
import h5py

from gprMax.constants import floattype


def fitness_max(filename, outputnames):
    """Return the maximum value from specific outputs in a file.
        
    Args:
        filename (str): Name of output file
        outputnames (list): Names (IDs) of outputs (rxs) from input file
        
    Returns:
        maxvalue (float): Maximum value from specific outputs
    """

    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']

    for rx in range(1, nrx + 1):
        tmp = f['/rxs/rx' + str(rx) + '/']
        if tmp.attrs['Name'] in outputnames:
            fieldname = list(tmp.keys())[0]
            maxvalue = np.amax(tmp[fieldname])

    return maxvalue

    



    