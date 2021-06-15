import numpy as np
import logging

logger = logging.getLogger(__name__)

def Rand_Create(distr, p1, p2):

    if distr=="u":
        """ 
            Generate a random number from a Uniform Distribution
            p1 - Lower bound
            p2 - Upper bound
        """
        return np.random.uniform(p1, p2)

    elif distr=="n":
        """ 
            Generate a random number from a Normal Distribution
            p1 - Mean
            p2 - Standard Deviation
        """
        return np.random.normal(p1, p2)
    
    elif distr=="ln":
        """ 
            Generate a random number from a Log-Normal Distribution
            p1 - Mean
            p2 - Standard Deviation
        """
        return np.random.lognormal(p1, p2)
    
    else:
        logger.exception('Invalid input for distribution \nAllowed values:\n"u" - Uniform\n"n" - Normal')
        raise ValueError