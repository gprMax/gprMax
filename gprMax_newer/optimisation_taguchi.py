# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

from collections import OrderedDict
import datetime
from importlib import import_module
import os
import pickle
import sys
from time import perf_counter

from colorama import init, Fore, Style
init()
import numpy as np

from gprMax.constants import floattype
from gprMax.exceptions import CmdInputError
from gprMax.gprMax import run_std_sim
from gprMax.gprMax import run_mpi_sim
from gprMax.utilities import get_terminal_width
from gprMax.utilities import open_path_file


def run_opt_sim(args, inputfile, usernamespace):
    """Run a simulation using Taguchi's optmisation process.

    Args:
        args (dict): Namespace with command line arguments
        inputfile (object): File object for the input file.
        usernamespace (dict): Namespace that can be accessed by user
                in any Python code blocks in input file.
    """

    tsimstart = perf_counter()

    if args.n > 1:
        raise CmdInputError('When a Taguchi optimisation is being carried out the number of model runs argument is not required')

    inputfileparts = os.path.splitext(inputfile.name)

    # Default maximum number of iterations of optimisation to perform (used
    # if the stopping criterion is not achieved)
    maxiterations = 20

    # Process Taguchi code blocks in the input file; pass in ordered
    # dictionary to hold parameters to optimise
    tmp = usernamespace.copy()
    tmp.update({'optparams': OrderedDict()})
    taguchinamespace = taguchi_code_blocks(inputfile, tmp)

    # Extract dictionaries and variables containing initialisation parameters
    optparams = taguchinamespace['optparams']
    fitness = taguchinamespace['fitness']
    if 'maxiterations' in taguchinamespace:
        maxiterations = taguchinamespace['maxiterations']

    # Store initial parameter ranges
    optparamsinit = list(optparams.items())

    # Dictionary to hold history of optmised values of parameters
    optparamshist = OrderedDict((key, list()) for key in optparams)

    # Import specified fitness function
    fitness_metric = getattr(import_module('user_libs.optimisation_taguchi.fitness_functions'), fitness['name'])

    # Select OA
    OA, N, cols, k, s, t = construct_OA(optparams)

    taguchistr = '\n--- Taguchi optimisation'
    print('{} {}\n'.format(taguchistr, '-' * (get_terminal_width() - 1 - len(taguchistr))))
    print('Orthogonal array: {:g} experiments per iteration, {:g} parameters ({:g} will be used), {:g} levels, and strength {:g}'.format(N, cols, k, s, t))
    tmp = [(k, v) for k, v in optparams.items()]
    print('Parameters to optimise with ranges: {}'.format(str(tmp).strip('[]')))
    print('Output name(s) from model: {}'.format(fitness['args']['outputs']))
    print('Fitness function "{}" with stopping criterion {:g}'.format(fitness['name'], fitness['stop']))
    print('Maximum iterations: {:g}'.format(maxiterations))

    # Initialise arrays and lists to store parameters required throughout optimisation
    # Lower, central, and upper values for each parameter
    levels = np.zeros((s, k), dtype=floattype)
    # Optimal lower, central, or upper value for each parameter
    levelsopt = np.zeros(k, dtype=np.uint8)
    # Difference used to set values for levels
    levelsdiff = np.zeros(k, dtype=floattype)
    # History of fitness values from each confirmation experiment
    fitnessvalueshist = []

    iteration = 0
    while iteration < maxiterations:
        # Reset number of model runs to number of experiments
        args.n = N
        usernamespace['number_model_runs'] = N

        # Fitness values for each experiment
        fitnessvalues = []

        # Set parameter ranges and define experiments
        optparams, levels, levelsdiff = calculate_ranges_experiments(optparams, optparamsinit, levels, levelsopt, levelsdiff, OA, N, k, s, iteration)

        # Run model for each experiment
        # Mixed mode MPI with OpenMP or CUDA - MPI task farm for models with
        # each model parallelised with OpenMP (CPU) or CUDA (GPU)
        if args.mpi:
            run_mpi_sim(args, inputfile, usernamespace, optparams)
        # Standard behaviour - models run serially with each model parallelised
        # with OpenMP (CPU) or CUDA (GPU)
        else:
            run_std_sim(args, inputfile, usernamespace, optparams)

        # Calculate fitness value for each experiment
        for experiment in range(1, N + 1):
            outputfile = inputfileparts[0] + str(experiment) + '.out'
            fitnessvalues.append(fitness_metric(outputfile, fitness['args']))
            os.remove(outputfile)

        taguchistr = '\n--- Taguchi optimisation, iteration {}: {} initial experiments with fitness values {}.'.format(iteration + 1, N, fitnessvalues)
        print('{} {}\n'.format(taguchistr, '-' * (get_terminal_width() - 1 - len(taguchistr))))

        # Calculate optimal levels from fitness values by building a response
        # table; update dictionary of parameters with optimal values
        optparams, levelsopt = calculate_optimal_levels(optparams, levels, levelsopt, fitnessvalues, OA, N, k)

        # Update dictionary with history of parameters with optimal values
        for key, value in optparams.items():
            optparamshist[key].append(value[0])

        # Run a confirmation experiment with optimal values
        args.n = 1
        usernamespace['number_model_runs'] = 1
        # Mixed mode MPI with OpenMP or CUDA - MPI task farm for models with
        # each model parallelised with OpenMP (CPU) or CUDA (GPU)
        if args.mpi:
            run_mpi_sim(args, inputfile, usernamespace, optparams)
        # Standard behaviour - models run serially with each model parallelised
        # with OpenMP (CPU) or CUDA (GPU)
        else:
            run_std_sim(args, inputfile, usernamespace, optparams)

        # Calculate fitness value for confirmation experiment
        outputfile = inputfileparts[0] + '.out'
        fitnessvalueshist.append(fitness_metric(outputfile, fitness['args']))

        # Rename confirmation experiment output file so that it is retained for each iteraction
        os.rename(outputfile, os.path.splitext(outputfile)[0] + '_final' + str(iteration + 1) + '.out')

        taguchistr = '\n--- Taguchi optimisation, iteration {} completed. History of optimal parameter values {} and of fitness values {}'.format(iteration + 1, dict(optparamshist), fitnessvalueshist)
        print('{} {}\n'.format(taguchistr, '-' * (get_terminal_width() - 1 - len(taguchistr))))
        iteration += 1

        # Stop optimisation if stopping criterion has been reached
        if fitnessvalueshist[iteration - 1] > fitness['stop']:
            taguchistr = '\n--- Taguchi optimisation stopped as fitness criteria reached: {:g} > {:g}'.format(fitnessvalueshist[iteration - 1], fitness['stop'])
            print('{} {}\n'.format(taguchistr, '-' * (get_terminal_width() - 1 - len(taguchistr))))
            break

        # Stop optimisation if successive fitness values are within a percentage threshold
        fitnessvaluesthres = 0.1
        if iteration > 2:
            fitnessvaluesclose = (np.abs(fitnessvalueshist[iteration - 2] - fitnessvalueshist[iteration - 1]) / fitnessvalueshist[iteration - 1]) * 100
            if fitnessvaluesclose < fitnessvaluesthres:
                taguchistr = '\n--- Taguchi optimisation stopped as successive fitness values within {}%'.format(fitnessvaluesthres)
                print('{} {}\n'.format(taguchistr, '-' * (get_terminal_width() - 1 - len(taguchistr))))
                break

    tsimend = perf_counter()

    # Save optimisation parameters history and fitness values history to file
    opthistfile = inputfileparts[0] + '_hist.pickle'
    with open(opthistfile, 'wb') as f:
        pickle.dump(optparamshist, f)
        pickle.dump(fitnessvalueshist, f)
        pickle.dump(optparamsinit, f)

    taguchistr = '\n=== Taguchi optimisation completed in [HH:MM:SS]: {} after {} iteration(s)'.format(datetime.timedelta(seconds=int(tsimend - tsimstart)), iteration)
    print('{} {}\n'.format(taguchistr, '=' * (get_terminal_width() - 1 - len(taguchistr))))
    print('History of optimal parameter values {} and of fitness values {}\n'.format(dict(optparamshist), fitnessvalueshist))


def taguchi_code_blocks(inputfile, taguchinamespace):
    """
    Looks for and processes a Taguchi code block (containing Python code) in
    the input file. It will ignore any lines that are comments, i.e. begin
    with a double hash (##), and any blank lines.

    Args:
        inputfile (object): File object for the input file.
        taguchinamespace (dict): Namespace that can be accessed by user a
                Taguchi code block in input file.

    Returns:
        processedlines (list): Input commands after Python processing.
    """

    # Strip out any newline characters and comments that must begin with double hashes
    inputlines = [line.rstrip() for line in inputfile if(not line.startswith('##') and line.rstrip('\n'))]

    # Rewind input file in preparation for passing to standard command reading function
    inputfile.seek(0)

    # Store length of dict
    taglength = len(taguchinamespace)

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

    # Check if any Taguchi code blocks were found
    if len(taguchinamespace) == taglength:
        raise CmdInputError('No #taguchi and #end_taguchi code blocks found.')

    return taguchinamespace


def construct_OA(optparams):
    """
    Load an orthogonal array (OA) from a numpy file. Configure and
    return OA and properties of OA.

    Args:
        optparams (dict): Dictionary containing name of parameters to
                optimise and their initial ranges

    Returns:
        OA (array): Orthogonal array
        N (int): Number of experiments in OA
        cols (int): Number of columns in OA
        k (int): Number of columns in OA cut down to number of parameters to optimise
        s (int): Number of levels in OA
        t (int): Strength of OA
    """

    oadirectory = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'user_libs', 'optimisation_taguchi')
    oadirectory = os.path.abspath(oadirectory)

    # Properties of the orthogonal array (OA)
    # Strength
    t = 2

    # Number of levels
    s = 3

    # Number of parameters to optimise
    k = len(optparams)

    # Load the appropriate OA
    if k <= 4:
        OA = np.load(os.path.join(oadirectory, 'OA_9_4_3_2.npy'))

        # Number of experiments
        N = OA.shape[0]

        # Number of columns of OA before cut down
        cols = OA.shape[1]

        # Cut down OA columns to number of parameters to optimise
        OA = OA[:, 0:k]

    elif k <= 7:
        OA = np.load(os.path.join(oadirectory, 'OA_18_7_3_2.npy'))

        # Number of experiments
        N = OA.shape[0]

        # Number of columns of OA before cut down
        cols = OA.shape[1]

        # Cut down OA columns to number of parameters to optimise
        OA = OA[:, 0:k]

    else:
        # THIS CASE NEEDS FURTHER TESTING
        print(Fore.RED + 'WARNING: Optimising more than 7 parameters is currently an experimental feature!' + Style.RESET_ALL)

        p = int(np.ceil(np.log(k * (s - 1) + 1) / np.log(s)))

        # Number of experiments
        N = s**p

        # Number of columns
        cols = int((N - 1) / (s - 1))

        # Algorithm to construct OA from:
        # http://ieeexplore.ieee.org/xpl/articleDetails.jsp?reload=true&arnumber=6812898
        OA = np.zeros((N + 1, cols + 1), dtype=np.int8)

        # Construct basic columns
        for ii in range(1, p + 1):
            col = int((s**(ii - 1) - 1) / (s - 1) + 1)
            for row in range(1, N + 1):
                OA[row, col] = np.mod(np.floor((row - 1) / (s**(p - ii))), s)

        # Construct non-basic columns
        for ii in range(2, p + 1):
            col = int((s**(ii - 1) - 1) / (s - 1) + 1)
            for jj in range(1, col):
                for kk in range(1, s):
                    OA[:, col + (jj - 1) * (s - 1) + kk] = np.mod(OA[:, jj] * kk + OA[:, col], s)

        # First row and first columns are unneccessary, only there to
        # match algorithm, and cut down columns to number of parameters to optimise
        OA = OA[1:, 1:k + 1]

    return OA, N, cols, k, s, t


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

    # Gaussian reduction function used for calculating levels
    T = 18  # Usually values between 15 - 20
    RR = np.exp(-(i / T)**2)

    # Calculate levels for each parameter
    for p in range(k):
        # Set central level for first iteration to midpoint of initial range and don't use RR
        if i == 0:
            levels[1, p] = ((optparamsinit[p][1][1] - optparamsinit[p][1][0]) / 2) + optparamsinit[p][1][0]
            levelsdiff[p] = (optparamsinit[p][1][1] - optparamsinit[p][1][0]) / (s + 1)
        # Set central level to optimum from previous iteration
        else:
            levels[1, p] = levels[levelsopt[p], p]
            levelsdiff[p] = RR * levelsdiff[p]

        # Set levels if below initial range
        if levels[1, p] - levelsdiff[p] < optparamsinit[p][1][0]:
            levels[0, p] = optparamsinit[p][1][0]
            levels[1, p] = optparamsinit[p][1][0] + levelsdiff[p]
            levels[2, p] = optparamsinit[p][1][0] + 2 * levelsdiff[p]
        # Set levels if above initial range
        elif levels[1, p] + levelsdiff[p] > optparamsinit[p][1][1]:
            levels[0, p] = optparamsinit[p][1][1] - 2 * levelsdiff[p]
            levels[1, p] = optparamsinit[p][1][1] - levelsdiff[p]
            levels[2, p] = optparamsinit[p][1][1]
        # Set levels normally
        else:
            levels[0, p] = levels[1, p] - levelsdiff[p]
            levels[2, p] = levels[1, p] + levelsdiff[p]

    # Update dictionary of parameters to optimise with lists of new values; clear dictionary first
    optparams = OrderedDict((key, list()) for key in optparams)
    p = 0
    for key, value in optparams.items():
        for exp in range(N):
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
    for p in range(k):
        responses = np.zeros(3, dtype=floattype)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0

        for exp in range(N):
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
        optlevel = np.where(responses == np.amax(responses))[0]

        # If 2 experiments produce the same fitness value pick first level
        # (this shouldn't happen if the fitness function is designed correctly)
        if len(optlevel) > 1:
            optlevel = optlevel[0]

        levelsopt[p] = optlevel

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
    ax.set_xlim(1, len(fitnessvalueshist))
    ax.grid()

    # Plot history of optimisation parameters
    p = 0
    for key, value in optparamshist.items():
        fig, ax = plt.subplots(subplot_kw=dict(xlabel='Iterations', ylabel='Parameter value'), num='History of ' + key + ' parameter', figsize=(20, 10), facecolor='w', edgecolor='w')
        ax.plot(iterations, optparamshist[key], 'r', marker='.', ms=15, lw=1)
        ax.set_xlim(1, len(fitnessvalueshist))
        ax.set_ylim(optparamsinit[p][1][0], optparamsinit[p][1][1])
        ax.grid()
        p += 1
    plt.show()
