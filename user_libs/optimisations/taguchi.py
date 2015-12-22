# Copyright (C) 2015, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import os
from collections import OrderedDict

import numpy as np
import h5py

from gprMax.constants import floattype
from gprMax.exceptions import CmdInputError

moduledirectory = os.path.dirname(os.path.abspath(__file__))


def taguchi_code_blocks(inputfile, taguchinamespace):
    """Looks for and processes a Taguchi code block (containing Python code) in the input file. It will ignore any lines that are comments, i.e. begin with a double hash (##), and any blank lines.
        
    Args:
        inputfile (str): Name of the input file to open.
        taguchinamespace (dict): Namespace that can be accessed by user a Taguchi code block in input file.
        
    Returns:
        processedlines (list): Input commands after Python processing.
    """
        
    with open(inputfile, 'r') as f:
        # Strip out any newline characters and comments that must begin with double hashes
        inputlines = [line.rstrip() for line in f if(not line.startswith('##') and line.rstrip('\n'))]
    
    x = 0
    while(x < len(inputlines)):
        if(inputlines[x].startswith('#taguchi:')):
            # String to hold Python code to be executed
            taguchicode = ''
            x += 1
            while not inputlines[x].startswith('#end_taguchi:'):
                # Add all code in current code block to string
                taguchicode += inputlines[x] + '\n'
                x += 1
                if x == len(inputlines):
                    raise CmdInputError('Cannot find the end of the Taguchi code block, i.e. missing #end_taguchi: command.')
        
            # Compile code for faster execution
            taguchicompiledcode = compile(taguchicode, '<string>', 'exec')

            # Execute code block & make available only usernamespace
            exec(taguchicompiledcode, taguchinamespace)
    
        x += 1
    
    return taguchinamespace


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
            levels[0, p] = optparamsinit[p][1][0]
        else:
            levels[0, p] = levels[1, p] - levelsdiff[p]

        # Upper levels set using central level and level differences values; and check they are not outwith initial ranges
        if levels[1, p] + levelsdiff[p] > optparamsinit[p][1][1]:
            levels[2, p] = optparamsinit[p][1][1]
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


def calculate_optimal_levels(optparams, levels, levelsopt, fitnessvalues, OA, N, k):
    """Calculate optimal levels from results of fitness metric by building a response table.
        
    Args:
        optparams (dict): Ordered dictionary containing name of parameters to optimise and their values
        levels (array): Lower, central, and upper values for each parameter
        levelsopt (array): Optimal level for each parameter from previous iteration
        fitnessvalues (list): Values from results of fitness metric
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
                responses[0] += fitnessvalues[exp]
                cnt1 += 1
            elif OA[exp, p] == 1:
                responses[1] += fitnessvalues[exp]
                cnt2 += 1
            elif OA[exp, p] == 2:
                responses[2] += fitnessvalues[exp]
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


def plot_optimisation_history(fitnessvalueshist, optparamshist, optparamsinit):
    """Plot the history of fitness values and each optimised parameter values for the optimisation.
        
    Args:
        fitnessvalueshist (list): History of fitness values
        optparamshist (dict): Name of parameters to optimise and history of their values
    """

    import matplotlib.pyplot as plt
    
    # Plot history of fitness values
    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Iterations', ylabel='Fitness value'), num='History of fitness values', figsize=(20, 10), facecolor='w', edgecolor='w')
    iterations = np.arange(1, len(fitnessvalueshist) + 1)
    ax.plot(iterations, fitnessvalueshist, 'r', marker='.', ms=15, lw=1)
    ax.grid()

    # Plot history of optimisation parameters
    p = 0
    for key, value in optparamshist.items():
        fig, ax = plt.subplots(subplot_kw=dict(xlabel='Iterations', ylabel='Parameter value'), num='History of ' + key + ' parameter', figsize=(20, 10), facecolor='w', edgecolor='w')
        ax.plot(iterations, optparamshist[key], 'r', marker='.', ms=15, lw=1)
        ax.set_ylim(optparamsinit[p][1][0], optparamsinit[p][1][1])
        ax.grid()
        p += 1
    plt.show()


    