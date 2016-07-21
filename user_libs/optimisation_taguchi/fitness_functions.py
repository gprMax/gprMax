# Copyright (C) 2015-2016, Craig Warren
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1190/1.3548506

import sys
import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

np.seterr(divide='ignore')

"""This module contains fitness metric functions that can be used with the Taguchi optimisation method.
    
    All fitness functions must take two arguments and return a single fitness value. 
    The first argument should be the name of the output file 
    The second argument is a dictionary which can contain any number of additional arguments, e.g. names (IDs) of outputs (rxs) from input file
"""

def minvalue(filename, args):
    """Minimum value from a response.
        
    Args:
        filename (str): Name of output file
        args (dict): 'outputs' key with a list of names (IDs) of outputs (rxs) from input file
        
    Returns:
        minvalue (float): Magnitude of minimum value from specific outputs
    """

    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']

    for rx in range(1, nrx + 1):
        output = f['/rxs/rx' + str(rx) + '/']
        if output.attrs['Name'] in args['outputs']:
            outputname = list(output.keys())[0]
            minvalue = np.amin(output[outputname])

    return np.abs(minvalue)


def maxvalue(filename, args):
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
        output = f['/rxs/rx' + str(rx) + '/']
        if output.attrs['Name'] in args['outputs']:
            outputname = list(output.keys())[0]
            maxvalue = np.amax(output[outputname])

    return maxvalue


def maxabsvalue(filename, args):
    """Maximum absolute value from a response.
        
    Args:
        filename (str): Name of output file
        args (dict): 'outputs' key with a list of names (IDs) of outputs (rxs) from input file
        
    Returns:
        maxabsvalue (float): Maximum absolute value from specific outputs
    """

    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']

    for rx in range(1, nrx + 1):
        output = f['/rxs/rx' + str(rx) + '/']
        if output.attrs['Name'] in args['outputs']:
            outputname = list(output.keys())[0]
            maxabsvalue = np.amax(np.abs(output[outputname]))

    return maxabsvalue


def xcorr(filename, args):
    """Maximum value of a cross-correlation between a response and a reference response.
        
    Args:
        filename (str): Name of output file
        args (dict): 'refresp' key with path & filename of reference response (time, amp) stored in a text file; 'outputs' key with a list of names (IDs) of outputs (rxs) from input file
        
    Returns:
        xcorrmax (float): Maximum value from specific outputs
    """

    # Load (from text file) the reference response
    with open(args['refresp'], 'r') as f:
        refdata = np.loadtxt(f)
    reftime = refdata[:,0] * 1e-9
    refresp = refdata[:,1]

    # Load response from output file
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    modeltime = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt'])
    
    for rx in range(1, nrx + 1):
        output = f['/rxs/rx' + str(rx) + '/']
        if output.attrs['Name'] in args['outputs']:
            outputname = list(output.keys())[0]
            modelresp = output[outputname]
            # Convert field value (V/m) to voltage
            if outputname == 'Ex':
                modelresp *= -f.attrs['dx, dy, dz'][0]
            elif outputname == 'Ey':
                modelresp *= -f.attrs['dx, dy, dz'][1]
            elif outputname == 'Ez':
                modelresp *= -f.attrs['dx, dy, dz'][2]

    # Normalise reference respose and response from output file
#    refresp /= np.amax(np.abs(refresp))
#    modelresp /= np.amax(np.abs(modelresp))

    # Make both responses the same length in time
#    if reftime[-1] > modeltime[-1]:
#        reftime = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], reftime[-1] / len(reftime))
#        refresp = refresp[0:len(reftime)]
#    elif modeltime[-1] > reftime[-1]:
#        modeltime = np.arange(0, reftime[-1], f.attrs['dt'])
#        modelresp = modelresp[0:len(modeltime)]
#
#    # Downsample the response with the higher sampling rate
#    if len(modeltime) < len(reftime):
#        refresp = signal.resample(refresp, len(modelresp))
#    elif len(reftime) < len(modeltime):
#        modelresp = signal.resample(modelresp, len(refresp))

    # Prepare data for normalized cross-correlation
    refresp = (refresp - np.mean(refresp)) / (np.std(refresp) * len(refresp))
    modelresp = (modelresp - np.mean(modelresp)) / np.std(modelresp)

    # Plots responses for checking
    #fig, ax = plt.subplots(subplot_kw=dict(xlabel='Iterations', ylabel='Voltage [V]'), figsize=(20, 10), facecolor='w', edgecolor='w')
    #ax.plot(refresp,'r', lw=2, label='refresp')
    #ax.plot(modelresp,'b', lw=2, label='modelresp')
    #ax.grid()
    #plt.show()

    # Calculate cross-correlation
    xcorr = signal.correlate(refresp, modelresp)
    
    # Set any NaNs to zero
    xcorr = np.nan_to_num(xcorr)

    # Plot cross-correlation for checking
#    fig, ax = plt.subplots(subplot_kw=dict(xlabel='Iterations', ylabel='Voltage [V]'), figsize=(20, 10), facecolor='w', edgecolor='w')
#    ax.plot(xcorr,'r', lw=2, label='xcorr')
#    ax.grid()
#    plt.show()

    xcorrmax = np.amax(xcorr)

    return xcorrmax


def min_sum_diffs(filename, args):
    """Sum of the differences (in dB) between responses and a reference response.
        
    Args:
        filename (str): Name of output file
        args (dict): 'refresp' key with path & filename of reference response; 'outputs' key with a list of names (IDs) of outputs (rxs) from input file
        
    Returns:
        diffdB (float): Sum of the differences (in dB) between responses and a reference response
    """

    # Load (from gprMax output file) the reference response
    f = h5py.File(args['refresp'], 'r')
    tmp = f['/rxs/rx1/']
    fieldname = list(tmp.keys())[0]
    refresp = np.array(tmp[fieldname])

    # Load (from gprMax output file) the response
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    
    diffdB = 0
    outputs = 0
    for rx in range(1, nrx + 1):
        output = f['/rxs/rx' + str(rx) + '/']
        if output.attrs['Name'] in args['outputs']:
            outputname = list(output.keys())[0]
            modelresp = np.array(output[outputname])
            # Calculate sum of differences
            tmp = 20 * np.log10(np.abs(modelresp - refresp) / np.amax(np.abs(refresp)))
            tmp = np.abs(np.sum(tmp[-np.isneginf(tmp)])) / len(tmp[-np.isneginf(tmp)])
            diffdB += tmp
            outputs += 1

    return diffdB / outputs


def compactness(filename, args):
    """A measure of the compactness of a time domain signal.
        
    Args:
        filename (str): Name of output file
        args (dict): 'outputs' key with a list of names (IDs) of outputs (rxs) from input file
        
    Returns:
        compactness (float): Compactness value of signal
    """
    
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']
    iterations = f.attrs['Iterations']
    time = np.linspace(0, 1, iterations)
    time *= (iterations * dt)

    for rx in range(1, nrx + 1):
        output = f['/rxs/rx' + str(rx) + '/']
        if output.attrs['Name'] in args['outputs']:
            outputname = list(output.keys())[0]
            outputdata = output[outputname]

            # Get absolute maximum value in signal
            peak = np.amax([np.amax(outputdata), np.abs(np.amin(outputdata))])

            # Get peaks and troughs in signal
            delta = peak / 50 # Considered a peak/trough if it has the max/min value, and was preceded (to the left) by a value lower by delta
            maxtab, mintab = peakdet(outputdata, delta)
            peaks = maxtab + mintab
            peaks.sort()
            
            # Remove any peaks smaller than a threshold
            thresholdpeak = 1e-3
            peaks = [peak for peak in peaks if np.abs(outputdata[peak]) > thresholdpeak]
            
            # Amplitude ratio of the 1st to 3rd peak - hopefully be a measure of a compact envelope
            compactness = np.abs(outputdata[peaks[0]]) / np.abs(outputdata[peaks[2]])
            
#            # Percentage of maximum value to measure compactness of signal
#            durationthreshold = 2
#            # Check if there is a peak/trough smaller than threshold
#            durationthresholdexist = np.where(np.abs(outputdata[peaks]) < (peak * (durationthreshold / 100)))[0]
#            if durationthresholdexist.size == 0:
#                compactness = time[peaks[-1]]
#            else:
#                time2threshold = time[peaks[durationthresholdexist[0]]]
#                compactness = time2threshold - time[min(peaks)]

    return compactness



######################################
# Helper methods for signal analyses #
######################################

def spectral_centroid(x, samplerate):
    """Calculate the spectral centroid of a signal.
        
    Args:
        x (float): 1D array containing time domain signal
        samplerate (float): Sample rate of signal (Hz)
        
    Returns:
        centroid (float): Weighted mean of the frequencies present in the signal
    """
    
    magnitudes = np.abs(np.fft.rfft(x))
    length = len(x)
    
    # Positive frequencies
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1])

    centroid = np.sum(magnitudes*freqs) / np.sum(magnitudes)
    return centroid


def zero_crossings(x):
    """Find location of zero crossings in 1D data array.
        
    Args:
        x (float): 1D array
        
    Returns:
        indexzeros (int): Array of indices of zero crossing locations
    """
    
    pos = x > 0
    npos = ~pos
    indexzeros = ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
    return indexzeros
                                
                                
def peakdet(v, delta, x = None):
    """Detect peaks and troughs in a vector (adapted from MATLAB script at http://billauer.co.il/peakdet.html). 
        A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by delta.
        
        Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        This function is released to the public domain; Any use is allowed.
    
    Args:
        v (float): 1D array
        delta (float): threshold for determining peaks/troughs
        
    Returns:
        maxtab, mintab (list): Indices of peak/trough locations
    """
                                
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                if int(mxpos) != 0:
                    maxtab.append(int(mxpos))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
        else:
            if this > mn+delta:
                if int(mnpos) != 0:
                    mintab.append(int(mnpos))
                    mx = this
                    mxpos = x[i]
                    lookformax = True

    return maxtab, mintab












    