# Copyright (C) 2015, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import h5py
import numpy as np
from scipy import signal

"""This module contains fitness metric functions that can be used with the Taguchi optimisation method.
    
    All fitness functions must take two arguments and return a single fitness value. 
    The first argument should be the name of the output file 
    The second argument is a list which can contain any number of additional arguments, e.g. names (IDs) of outputs (rxs) from input file
"""


def fitness_max(filename, args):
    """Maximum value from a response.
        
    Args:
        filename (str): Name of output file
        args (dict): 'outputs' key with a list of names (IDs) of outputs (rxs) from input file
        
    Returns:
        maxvalue (float): Maximum value from specific outputs
    """

    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']

    for rx in range(1, nrx + 1):
        tmp = f['/rxs/rx' + str(rx) + '/']
        if tmp.attrs['Name'] in args['outputs']:
            fieldname = list(tmp.keys())[0]
            maxvalue = np.amax(tmp[fieldname])

    return maxvalue


def fitness_xcorr(filename, args):
    """Maximum value of a cross-correlation between a response and a reference response.
        
    Args:
        filename (str): Name of output file
        args (dict): 'refresp' key with path & filename of reference response (time, amp) stored in a text file; 'outputs' key with a list of names (IDs) of outputs (rxs) from input file
        
    Returns:
        xcorrmax (float): Maximum value from specific outputs
    """

    # Load (from text file) and normalise the reference response
    with open(args['refresp'], 'r') as f:
        refdata = np.loadtxt(f)
    reftime = refdata[:,0] * 1e-9
    refresp = refdata[:,1]
    refresp /= np.amax(np.abs(refresp))

    # Load response from output file
    print(filename)
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    modeltime = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt'])
    
    for rx in range(1, nrx + 1):
        tmp = f['/rxs/rx' + str(rx) + '/']
        if tmp.attrs['Name'] in args['outputs']:
            fieldname = list(tmp.keys())[0]
            modelresp = tmp[fieldname]
            # Convert field value (V/m) to voltage
            if fieldname == 'Ex':
                modelresp *= -1 * f.attrs['dx, dy, dz'][0]
            elif fieldname == 'Ey':
                modelresp *= -1 * f.attrs['dx, dy, dz'][1]
            if fieldname == 'Ez':
                modelresp *= -1 * f.attrs['dx, dy, dz'][2]

    # Normalise respose from output file
    modelresp /= np.amax(np.abs(modelresp))

    # Make both responses the same length in time
    if reftime[-1] > modeltime[-1]:
        reftime = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], reftime[-1] / len(reftime))
        refresp = refresp[0:len(reftime)]
    elif modeltime[-1] > reftime[-1]:
        modeltime = np.arange(0, reftime[-1], f.attrs['dt'])
        modelresp = modelresp[0:len(modeltime)]

    # Downsample the response with the higher sampling rate
    if len(modeltime) < len(reftime):
        refresp = signal.resample(refresp, len(modelresp))
    elif len(reftime) < len(modeltime):
        modelresp = signal.resample(modelresp, len(refresp))

    # Plots responses for checking
#    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Iterations', ylabel='Voltage [V]'), figsize=(20, 10), facecolor='w', edgecolor='w')
#    ax.plot(refresp,'r', lw=2, label='refresp')
#    ax.plot(modelresp,'b', lw=2, label='modelresp')
#    ax.grid()
#    plt.show()

    # Calculate cross-correlation
    xcorr = signal.correlate(refresp, modelresp)
    # Plot cross-correlation for checking
#    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Iterations', ylabel='Voltage [V]'), figsize=(20, 10), facecolor='w', edgecolor='w')
#    ax.plot(xcorr,'r', lw=2, label='xcorr')
#    ax.grid()
#    plt.show()
    xcorrmax = np.amax(xcorr) / 100

    return xcorrmax


def fitness_diff_dB(filename, args):
    """Sum of the differences (in dB) between a response and a reference response.
        
    Args:
        filename (str): Name of output file
        args (dict): 'refresp' key with path & filename of reference response; 'outputs' key with a list of names (IDs) of outputs (rxs) from input file
        
    Returns:
        diffdB (float): Sum of the differences (in dB) between a response and a reference response
    """

    # Load (from gprMax output file) the reference response
    f = h5py.File(args['refresp'], 'r')
    tmp = f['/rxs/rx1/']
    fieldname = list(tmp.keys())[0]
    refresp = np.array(tmp[fieldname])

    # Load (from gprMax output file) the response
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    
    for rx in range(1, nrx + 1):
        tmp = f['/rxs/rx' + str(rx) + '/']
        if tmp.attrs['Name'] in args['outputs']:
            fieldname = list(tmp.keys())[0]
            modelresp = np.array(tmp[fieldname])

    # Calculate sum of differences
    diffdB = np.abs(modelresp - refresp) / np.amax(np.abs(refresp))
    diffdB = 20 * np.log10(np.sum(diffdB))

    return 1 / diffdB












    