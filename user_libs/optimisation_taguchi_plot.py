# Copyright (C) 2015, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import argparse
import numpy as np

from gprMax.optimisation_taguchi import plot_optimisation_history

"""Plots the results (stored in a NumPy archive) from a Taguchi optimisation process."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots the results (stored in a NumPy archive) from a Taguchi optimisation process.', usage='cd gprMax; python -m user_libs.optimisation_taguchi_plot numpyfile')
parser.add_argument('numpyfile', help='name of NumPy archive including path')
args = parser.parse_args()

results = np.load(args.numpyfile)  
    
# Plot the history of fitness values and each optimised parameter values for the optimisation
plot_optimisation_history(results[fitnessvalueshist], results[optparamshist], results[optparamsinit])