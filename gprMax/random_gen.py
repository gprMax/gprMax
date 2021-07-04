import numpy as np
import logging
import pickle
import sys
import h5py
from pathlib import Path
from gprMax.receivers import Rx

logger = logging.getLogger(__name__)


def rand_param_create(distr, p1, p2, bounds=None, cmd=None):

    if distr=="u":
        """ Generate a random number from a Uniform Distribution
            p1 - Lower bound
            p2 - Upper bound
        """
        result = np.random.uniform(p1, p2)

    elif distr=="n":
        """ Generate a random number from a Normal Distribution
            p1 - Mean
            p2 - Standard Deviation
        """
        result = np.random.normal(p1, p2)
    
    elif distr=="ln":
        """ Generate a random number from a Log-Normal Distribution
            p1 - Mean
            p2 - Standard Deviation
        """
        result = np.random.lognormal(p1, p2)
    
    elif distr=="lg":
        """ Generate a random number from a Logistic Distribution
            p1 - Mean
            p2 - Standard Deviation
        """
        result = np.random.logistic(p1, p2)

    elif distr=="lp":
        """ Generate a random number from a Laplace (double exponential) Distribution
            p1 - Mean
            p2 - Standard Deviation
        """
        result = np.random.laplace(p1, p2)
    
    elif distr=="b":
        """ Generate a random number from a Beta Distribution
            p1 - alpha parameter
            p2 - beta parameter
        """
        result = np.random.beta(p1, p2)
    
    else:
        logger.exception('Invalid input for distribution \nAllowed values: "u" - Uniform | "n" - Normal | "ln" - Log-Normal | "lg" - Logistic | "lp" - Laplace | "b" - Beta\n\n')
        raise ValueError
    
    if bounds:
        if result < bounds[0]:
            logger.warning(f"'{cmd}' value '{result}' is smaller than the lower bound '{bounds[0]}' for this model. Changing to '{bounds[0]}'")
            result = bounds[0]
        elif result > bounds[1]:
            logger.warning(f"'{cmd}' value '{result}' is greater than the upper bound '{bounds[1]}' for this model. Changing to '{bounds[1]}'")
            result = bounds[1]

    return result


def check_upper_greater(p1, p2, cmd):

    """ Check if upper coordinate is greater than the lower coordinate.
        If not, increment the upper coordinate to just exceed the lower coordinate.

    Args:
        p1 (tuple): Lower Coordinate
        p2 (tuple): Upper Coordinate
        cmd (str): The hash command
    """

    for i in range(len(p1)):
        if p2[i] < p1[i]:
            upper_new = p1[i] + sys.float_info.epsilon
            logger.warning(f"'{cmd}' upper coordinate '{str(p2[i])}' is smaller than the lower coordinate '{str(p1[i])}'. Changing the upper coordinate to '{str(upper_new)}'")
            p2 = list(p2)
            p2[i] = upper_new

    return tuple(p2)


def make_data_label(dat, cmd, labels):

    """ Make data labels corresponding to the random parameters

    Args:
        dat (list): The list to which labels are appended
        cmd (str): The hash command
        labels (list): Labels to be appended
    """

    for i in labels:
        dat.append(cmd+i)

    return dat


def save_params(rand_params, outputfile):

    """ Write all the randomly generated parameters to a Pickle (.pkl) file.

    Args:
        rand_params (list): Generated Random Parameters.
        outputfile (str): Name of the output file.
    """

    with open(outputfile, "ab+") as f:
        pickle.dump(np.array(rand_params), f)


def compress_pkl_dat(inputfile, outputfile):

    """ Remove all redundant features and save compressed data file 
        containing randomly generated parameters

    Args:
        inputfile: The original .pkl file (with saved random parameters)
        outputfile: The compressed .pkl file
    """
    
    # Load data from input .pkl file
    data = []
    with open(inputfile, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass

    data = np.array(data)

    # Remove redundant columns from data
    for i in range(data.shape[1]-1, -1, -1):
        if np.all(data[:,i] == data[0,i]):
            data = np.delete(data, i, axis=1)

    # Write data to output .pkl file
    with open(outputfile, "wb+") as f:
        pickle.dump(np.squeeze(data), f)
    
    logger.basic(f'\n -> All Random Parameters saved to: {inputfile}')
    logger.basic(f' -> Removed redundant features and saved compressed data to: {outputfile} \n')


def save_h5_to_pkl(filename, pkl_output):

    """ Extracts A-scans from all receiver points in the given output .h5 file 
        and saves them to a .pkl file

    Args:
        filename (string): Filename (including path) of .h5 output file.
        pkl_output (string): The .pkl file to which field outputs are saved.

    Note:
        The output data dimensions are - (no. of receivers, 
                                          no. of field outputs [Ex, Ey, Ez, Hx, Hy, Hz], 
                                          no. of elements in each field output)
    """

    data = []
    outputs = Rx.defaultoutputs

    # Open output file and read iterations
    file = Path(filename)
    f = h5py.File(file, 'r')

    # Paths to grid(s) to traverse for outputs
    paths = ['/']

    # Check if any subgrids and add path(s)
    is_subgrids = "/subgrids" in f
    if is_subgrids:
        paths = paths + ['/subgrids/' + path + '/' for path in f['/subgrids'].keys()]

    # Get number of receivers in grid(s)
    nrxs = []
    for path in paths:
        if f[path].attrs['nrx'] > 0:
            nrxs.append(f[path].attrs['nrx'])
        else:
            paths.remove(path)

    # Check there are any receivers
    if not paths:
        logger.exception(f'No receivers found in {file}')
        raise ValueError

    # Loop through all grids
    for path in paths:
        nrx = f[path].attrs['nrx']

        # Save A-scan for each receiver
        for rx in range(1, nrx + 1):
            rxpath = path + 'rxs/rx' + str(rx) + '/'
            availableoutputs = list(f[rxpath].keys())

            data.append([])
            print(f"\n -> Saving the following field outputs for Receiver {rx} to: {pkl_output} \n", availableoutputs)

            for output in outputs:
                # Check for polarity of output
                if output[-1] == 'm':
                    polarity = -1
                    output = output[0:-1]
                else:
                    polarity = 1

                outputdata = f[rxpath + output][:] * polarity

                data[rx-1].append(outputdata)
        
    f.close()
    data = np.array(data)

    with open(pkl_output, "ab+") as f:
        pickle.dump(data, f)