# Copyright (C) 2015-2023, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import argparse
import os
import pickle

from gprMax.optimisation_taguchi import plot_optimisation_history

"""Plots the results (pickled to file) from a Taguchi optimisation process."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots the results (pickled to file) from a Taguchi optimisation process.', usage='cd gprMax; python -m user_libs.optimisation_taguchi.plot_results picklefile')
parser.add_argument('picklefile', help='name of file including path')
args = parser.parse_args()

f = open(args.picklefile, 'rb')
optparamshist = pickle.load(f)
fitnessvalueshist = pickle.load(f)
optparamsinit = pickle.load(f)

print('Optimisations summary for: {}'.format(os.path.split(args.picklefile)[1]))
print('Number of iterations: {:g}'.format(len(fitnessvalueshist)))
print('History of fitness values: {}'.format(fitnessvalueshist))
print('Initial parameter values:')
for item in optparamsinit:
    print('\t{}: {}'.format(item[0], item[1]))
print('History of parameter values:')
for key, value in optparamshist.items():
    print('\t{}: {}'.format(key, value))


# Plot the history of fitness values and each optimised parameter values for the optimisation
plot_optimisation_history(fitnessvalueshist, optparamshist, optparamsinit)
