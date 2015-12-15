# Copyright (C) 2015, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import os
import numpy as np
from collections import OrderedDict

import h5py

from gprMax.constants import floattype
from gprMax.exceptions import CmdInputError

moduledirectory = os.path.dirname(os.path.abspath(__file__))


def select_OA(optparams):
    """Load an orthogonal array (OA) from a numpy file. Configure and return OA and properties of OA.
        
    Args:
        optparams (dict): Dictionary containing name of parameters to optimise and their initial ranges
        
    Returns:
        OA (array): Orthogonal array
        N (int): Number of experiments in OA
        k (int): Number of parameters to optimise in OA
        s (int): Number of levels in OA
        t (int): Strength of OA
    """

    # Load the appropriate OA
    if len(optparams) <= 4:
        OA = np.load(os.path.join(moduledirectory, 'OA_9_4_3_2.npy'))
    elif len(optparams) <= 7:
        OA = np.load(os.path.join(moduledirectory, 'OA_18_7_3_2.npy'))
    else:
        raise CmdInputError('Too many parameters to optimise for the available orthogonal arrays (OA). Please find and load a bigger, suitable OA.')

    # Cut down OA columns to number of parameters to optimise
    OA = OA[:, 0:len(optparams)]

    # Number of experiments
    N = OA.shape[0]

    # Number of parameters to optimise
    k = OA.shape[1]

    # Number of levels
    s = 3

    # Strength
    t = 2

    return OA, N, k, s


def calculate_ranges_experiments(optparams, optparamsinit, levels, levelsopt, levelsdiff, OA, N, k, s, i):
    """Calculate values for parameters to optimise for a set of experiments.
        
    Args:
        optparams (dict): Ordered dictionary containing name of parameters to optimise and their values
        optparamsinit (list): Initial ranges for parameters to optimise
        levels (array): Lower, central, and upper values for each parameter
        levelsopt (array): Optimal level for each parameter from previous iteration
        levelsdiff (array): Difference used to set values in levels array
        OA (array): Orthogonal array
        N (int): Number of experiments in OA
        k (int): Number of parameters to optimise in OA
        s (int): Number of levels in OA
        i (int): Iteration number
        
    Returns:
        optparams (dict): Ordered dictionary containing name of parameters to optimise and their values
        levels (array): Lower, central, and upper values for each parameter
        levelsdiff (array): Difference used to set values in levels array
    """

    # Reducing function used for calculating levels
    RR = np.exp(-(i/18)**2)

    # Calculate levels for each parameter
    for p in range(0, k):
        # Central levels - for first iteration set to midpoint of initial range and don't use RR
        if i == 0:
            levels[1, p] = ((optparamsinit[p][1][1] - optparamsinit[p][1][0]) / 2) + optparamsinit[p][1][0]
            levelsdiff[p] = (optparamsinit[p][1][1] - optparamsinit[p][1][0]) / (s + 1)
        # Central levels - set to optimum from previous iteration
        else:
            levels[1, p] = levels[levelsopt[p], p]
            levelsdiff[p] = RR * levelsdiff[p]

        # Lower levels set using central level and level differences values; and check they are not outwith initial ranges
        if levels[1, p] - levelsdiff[p] < optparamsinit[p][1][0]:
            levels[0, p] = optparamslist[p][1][0]
        else:
            levels[0, p] = levels[1, p] - levelsdiff[p]

        # Upper levels set using central level and level differences values; and check they are not outwith initial ranges
        if levels[1, p] + levelsdiff[p] > optparamsinit[p][1][1]:
            levels[2, p] = optparamslist[p][1][1]
        else:
            levels[2, p] = levels[1, p] + levelsdiff[p]

    # Update dictionary of parameters to optimise with lists of new values; clear dictionary first
    optparams = OrderedDict((key, list()) for key in optparams)
    p = 0
    for key, value in optparams.items():
        for exp in range(0, N):
            if OA[exp, p] == 0:
                optparams[key].append(levels[0, p])
            elif OA[exp, p] == 1:
                optparams[key].append(levels[1, p])
            elif OA[exp, p] == 2:
                optparams[key].append(levels[2, p])
        p += 1

    return optparams, levels, levelsdiff



def calculate_optimal_levels(optparams, levels, levelsopt, fitness, OA, N, k):
    """Calculate optimal levels from results of fitness metric by building a response table.
        
    Args:
        optparams (dict): Ordered dictionary containing name of parameters to optimise and their values
        levels (array): Lower, central, and upper values for each parameter
        levelsopt (array): Optimal level for each parameter from previous iteration
        fitness (array): Values from results of fitness metric
        OA (array): Orthogonal array
        N (int): Number of experiments in OA
        k (int): Number of parameters to optimise in OA
        
    Returns:
        optparams (dict): Ordered dictionary containing name of parameters to optimise and their values
        levelsopt (array): Optimal level for each parameter from previous iteration
    """

    # Build a table of responses based on the results of the fitness metric
    for p in range(0, k):
        responses = np.zeros(3, dtype=floattype)

        cnt1 = 0
        cnt2 = 0
        cnt3 = 0

        for exp in range(1, N):
            if OA[exp, p] == 0:
                responses[0] += fitness[exp]
                cnt1 += 1
            elif OA[exp, p] == 1:
                responses[1] += fitness[exp]
                cnt2 += 1
            elif OA[exp, p] == 2:
                responses[2] += fitness[exp]
                cnt3 += 1

        responses[0] /= cnt1
        responses[1] /= cnt2
        responses[2] /= cnt3

        # Calculate optimal level from table of responses
        tmp = np.where(responses == np.amax(responses))[0]

        # If there is more than one level found use the first
        if len(tmp) > 1:
            tmp = tmp[0]

        levelsopt[p] = tmp

    # Update dictionary of parameters to optimise with lists of new values; clear dictionary first
    optparams = OrderedDict((key, list()) for key in optparams)
    p = 0
    for key, value in optparams.items():
        optparams[key].append(levels[levelsopt[p], p])
        p += 1

    return optparams, levelsopt


def fitness_max(filename, outputnames):
    """Return the maximum value from specific outputs in a file.
        
    Args:
        filename (dict): Ordered dictionary containing name of parameters to optimise and their values
        outputnames (list): Names (IDs) of outputs (rxs)
        
    Returns:
        maxvalue (array): Maximum value(s) from specific outputs
    """

    maxvalue = np.zeros(len(outputnames), dtype=floattype)
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']

    i = 0
    for rx in range(1, nrx + 1):
        tmp = f['/rxs/rx' + str(rx) + '/']
        if tmp.attrs['Name'] in outputnames:
            fieldname = list(tmp.keys())[0]
            maxvalue[i] = np.amax(tmp[fieldname])
            i += 1

    return maxvalue

    



    