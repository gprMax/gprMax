# Authors: Iraklis Giannakis, and Sylwia Majchrowska
# E-mail: i.giannakis@ed.ac.uk
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

import numpy as np
from matplotlib import pylab as plt
import scipy.optimize
from tqdm import tqdm


class Optimizer(object):
    """
    Create particle swarm optimisation object.

    :param maxiter: The maximum number of iterations for the
                    optimizer (Default: 1000).
    :type maxiter: int, optional
    :param seed: Seed for RandomState. Must be convertible to 32 bit
                 unsigned integers (Default: None).
    :type seed: int, NoneType, optional
    """
    def __init__(self, maxiter=1000, seed=None):
        self.maxiter = maxiter
        self.seed = seed

    def fit(self):
        """
        Call the optimization function that tries to find an optimal set
        of relaxation times that minimise the error
        between the actual and the approximated electric permittivity.
        """
        raise NotImplementedError()

    @staticmethod
    def plot(x, y):
        """
        Dynamically plot the error as the optimisation takes place.

        Args:
            x (array): The number of current iterations.
            y (array): The objective value at for all x points.
        """
        plt.plot(x, y, "b-", linewidth=1.0)
        plt.ylim(min(y) - 0.1 * min(y),
                 max(y) + 0.1 * max(y))
        plt.xlim(min(x) - 0.1, max(x) + 0.1)
        plt.grid(b=True, which="major", color="k",
                 linewidth=0.2, linestyle="--")
        plt.suptitle("Debye fitting process")
        plt.xlabel("Iteration")
        plt.ylabel("Average Error")
        plt.pause(0.0001)


class Particle_swarm(Optimizer):
    """
    Create particle swarm optimisation object with predefined parameters.

    :param swarmsize: The number of particles in the swarm (Default: 40).
    :type swarmsize: int, optional
    :param maxiter: The maximum number of iterations for the swarm
                    to search (Default: 50).
    :type maxiter: int, optional
    :param omega: Particle velocity scaling factor (Default: 0.9).
    :type omega: float, optional
    :param phip: Scaling factor to search away from the particle's
                 best known position (Default: 0.9).
    :type phip: float, optional
    :param phig: Scaling factor to search away from the swarm's
                 best known position (Default: 0.9).
    :type phig: float, optional
    :param minstep: The minimum stepsize of swarm's best position
                    before the search terminates (Default: 1e-8).
    :type minstep: float, optional
    :param minfun: The minimum change of swarm's best objective value
                   before the search terminates (Default: 1e-8)
    :type minfun: float, optional
    :param pflag: if True will plot the actual and the approximated
                  value during optimization process (Default: False).
    :type pflag: bool, optional
    """
    def __init__(self, swarmsize=40, maxiter=50,
                 omega=0.9, phip=0.9, phig=0.9,
                 minstep=1e-8, minfun=1e-8,
                 pflag=False, seed=None):

        super(Particle_swarm, self).__init__(maxiter, seed)
        self.swarmsize = swarmsize
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.minstep = minstep
        self.minfun = minfun
        self.pflag = pflag

    def fit(self, func, lb, ub, funckwargs={}):
        """
        A particle swarm optimisation that tries to find an optimal set
        of relaxation times that minimise the error
        between the actual and the approximated electric permittivity.
        The current code is a modified edition of the pyswarm package
        which can be found at https://pythonhosted.org/pyswarm/

        Args:
            func (function): The function to be minimized.
            lb (ndarray): The lower bounds of the design variable(s).
            ub (ndarray): The upper bounds of the design variable(s).
            funckwargs (dict): Additional keyword arguments passed to
                               objective and constraint function
                               (Default: empty dict).

        Returns:
            g (ndarray): The swarm's best known position (optimal design).
            fg (float): The objective value at ``g``.
        """
        np.random.seed(self.seed)
        # check input parameters
        assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values'

        vhigh = np.abs(ub - lb)
        vlow = -vhigh

        # Initialize objective function
        obj = lambda x: func(x=x, **funckwargs)

        # Initialize the particle swarm
        d = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(self.swarmsize, d)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fp = np.zeros(self.swarmsize)  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # artificial best swarm position starting value

        for i in range(self.swarmsize):
            # Initialize the particle's position
            x[i, :] = lb + x[i, :] * (ub - lb)
            # Initialize the particle's best known position
            p[i, :] = x[i, :]
            # Calculate the objective's value at the current particle's
            fp[i] = obj(p[i, :])
            # At the start, there may not be any feasible starting point,
            # so just
            # give it a temporary "best" point since it's likely to change
            if i == 0:
                g = p[0, :].copy()
            # If the current particle's position is better than the swarm's,
            # update the best swarm position
            if fp[i] < fg:
                fg = fp[i]
                g = p[i, :].copy()
            # Initialize the particle's velocity
            v[i, :] = vlow + np.random.rand(d) * (vhigh - vlow)

        # Iterate until termination criterion met
        for it in tqdm(range(self.maxiter), desc='Debye fitting'):
            rp = np.random.uniform(size=(self.swarmsize, d))
            rg = np.random.uniform(size=(self.swarmsize, d))
            for i in range(self.swarmsize):
                # Update the particle's velocity
                v[i, :] = self.omega * v[i, :] + self.phip * rp[i, :] * \
                          (p[i, :] - x[i, :]) + \
                          self.phig * rg[i, :] * (g - x[i, :])
                # Update the particle's position,
                # correcting lower and upper bound
                # violations, then update the objective function value
                x[i, :] = x[i, :] + v[i, :]
                mark1 = x[i, :] < lb
                mark2 = x[i, :] > ub
                x[i, mark1] = lb[mark1]
                x[i, mark2] = ub[mark2]
                fx = obj(x[i, :])
                # Compare particle's best position
                if fx < fp[i]:
                    p[i, :] = x[i, :].copy()
                    fp[i] = fx
                    # Compare swarm's best position to current
                    # particle's position
                    if fx < fg:
                        tmp = x[i, :].copy()
                        stepsize = np.sqrt(np.sum((g - tmp) ** 2))
                        if np.abs(fg - fx) <= self.minfun:
                            print(f'Stopping search: Swarm best objective '
                                  f'change less than {self.minfun}')
                            return tmp, fx
                        elif stepsize <= self.minstep:
                            print(f'Stopping search: Swarm best position '
                                  f'change less than {self.minstep}')
                            return tmp, fx
                        else:
                            g = tmp.copy()
                            fg = fx

            # Dynamically plot the error as the optimisation takes place
            if self.pflag:
                if it == 0:
                    xpp = [it]
                    ypp = [fg]
                else:
                    xpp.append(it)
                    ypp.append(fg)
                Particle_swarm.plot(xpp, ypp)
        return g, fg


class Dual_annealing(Optimizer):
    """
    Create dual annealing object with predefined parameters.

    :param maxiter: The maximum number of iterations for the swarm
                    to search (Default: 1000).
    :type maxiter: int, optional
    :param local_search_options: Extra keyword arguments to be passed
                                 to the local minimizer, reffer to
                                 scipy.optimize.minimize() function
                                 (Default: empty dict).
    :type local_search_options: dict, optional
    :param initial_temp (float): The initial temperature, use higher values to
                                facilitates a wider search of the energy
                                landscape, allowing dual_annealing to escape
                                local minima that it is trapped in.
                                Range is (0.01, 5.e4] (Default: 5230).
    :type initial_temp: float, optional
    :param restart_temp_ratio: During the annealing process,
                               temperature is decreasing, when it
                               reaches initial_temp * restart_temp_ratio,
                               the reannealing process is triggered.
                               Range is (0, 1) (Default: 2e-5).
    :type restart_temp_ratio: float, optional
    :param visit: Parameter for visiting distribution. The value range is (1, 3]
                  (Default: 2.62).
    :type visit: float, optional
    :param accept: Parameter for acceptance distribution. It is used to control
                   the probability of acceptance. The lower the acceptance parameter,
                   the smaller the probability of acceptance. The value range (-1e4, -5]
                   (Default: -5.0).
    :type accept: float, optional
    :param no_local_search (bool): If no_local_search is set to True, a traditional
                                   Generalized Simulated Annealing will be performed
                                   with no local search strategy applied (Default: False).
    :type no_local_search: bool, optional
    :param maxfun: Soft limit for the number of objective function calls.
                        (Default: 1e7).
    :type maxfun: int, optional
    :param callback: A callback function with signature callback(x, f, context),
                     which will be called for all minima found.
                     x and f are the coordinates and function value of
                     the latest minimum found, and context has value in [0, 1, 2],
                     with the following meaning:
                     0: minimum detected in the annealing process.
                     1: detection occurred in the local search process.
                     2: detection done in the dual annealing process.
                     If the callback implementation returns True,
                     the algorithm will stop.
    :type callback: None, callable, optional
    :param x0: Coordinates of a single N-D starting point, shape(n,).
                        (Default: None).
    :type x0: None, ndarray, optional
    :param seed: Specify seed for repeatable minimizations.
                 The random numbers generated with this seed only
                 affect the visiting distribution function and
                 new coordinates generation (Default: None).
    :type seed: None, int, optional
    """
    def __init__(self, maxiter=1000,
                 local_search_options={}, initial_temp=5230.0,
                 restart_temp_ratio=2e-05, visit=2.62, accept=-5.0,
                 maxfun=1e7, no_local_search=False,
                 callback=None, x0=None, seed=None):
        super(Dual_annealing, self).__init__(maxiter, seed)
        self.local_search_options = local_search_options
        self.initial_temp = initial_temp
        self.restart_temp_ratio = restart_temp_ratio
        self.visit = visit
        self.accept = accept
        self.maxfun = maxfun
        self.no_local_search = no_local_search
        self.callback = callback
        self.x0 = x0

    def fit(self, func, lb, ub, funckwargs={}):
        """
        Find the global minimum of a function using Dual Annealing.
        The current class is a modified edition of the scipy.optimize
        package which can be found at:
        https://docs.scipy.org/doc/scipy/reference/generated/
        scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing

        Args:
            func (function): The function to be minimized
            lb (array): The lower bounds of the design variable(s)
            ub (array): The upper bounds of the design variable(s)
            funckwargs (dict): Additional keyword arguments passed to
                               objective and constraint function
                               (Default: empty dict)

        Returns:
            g (array): The solution array (optimal design).
            fg (float): The objective value at the solution.
        """
        np.random.seed(self.seed)
        result = scipy.optimize.dual_annealing(func,
                                               bounds=list(zip(lb, ub)),
                                               args=funckwargs.values(),
                                               maxiter=self.maxiter,
                                               local_search_options=self.local_search_options,
                                               initial_temp=self.initial_temp,
                                               restart_temp_ratio=self.restart_temp_ratio,
                                               visit=self.visit,
                                               accept=self.accept,
                                               maxfun=self.maxfun,
                                               no_local_search=self.no_local_search,
                                               callback=self.callback,
                                               x0=self.x0)
        return result.x, result.fun


def DLS(rl, im, logt, freq):
    """
    Find the weights using a non-linear least squares (LS) method,
    the Levenberg–Marquardt algorithm (LMA or just LM),
    also known as the damped least-squares (DLS) method.

    Args:
        rl (ndarray): Real parts of chosen relaxation function
                      for given frequency points.
        im (ndarray): Imaginary parts of chosen relaxation function
                      for given frequency points.
        logt (ndarray): The best known position form optimization module (optimal design),
                        the logarithm with base 10 of relaxation times of the Debyes poles.
        freq (ndarray): The frequencies vector for defined grid.

    Returns:
        cost_i (float): Mean absolute error between the actual and
                        the approximated imaginary part.
        cost_r (float): Mean absolute error between the actual and
                        the approximated real part (plus average error).
        x (ndarray): Resulting optimised weights for the given relaxation times.
        ee (float): Average error between the actual and the approximated real part.
        rp (ndarray): The real part of the permittivity for the optimised relaxation
                      times and weights for the frequnecies included in freq.
        ip (ndarray): The imaginary part of the permittivity for the optimised
                      relaxation times and weights for the frequnecies included in freq.
    """
    # The relaxation time of the Debyes are given at as logarithms
    # logt=log10(t0) for efficiency during the optimisation
    # Here they are transformed back t0=10**logt
    tt = 10**logt
    # y = Ax , here the A matrix for the real and the imaginary part is builded
    d = 1 / (1 + 1j * 2 * np.pi * np.repeat(
             freq, len(tt)).reshape((-1, len(tt))) * tt)
    # Adding dumping (Levenberg–Marquardt algorithm)
    # Solving the overdetermined system y=Ax
    x = np.abs(np.linalg.lstsq(d.imag, im, rcond=-1)[0])  # absolute damped least-squares solution
    rp, ip = np.matmul(d.real, x[np.newaxis].T).T[0], np.matmul(d.imag, x[np.newaxis].T).T[0]
    cost_i = np.sum(np.abs(ip-im))/len(im)
    ee = np.mean(rl - rp)
    if ee < 1:
        ee = 1
    cost_r = np.sum(np.abs(rp - rl + ee))/len(im)
    return cost_i, cost_r, x, ee, rp, ip


def cost_function(x, rl_g, im_g, freq_g):
    """
    The cost function is the average error between
    the actual and the approximated electric permittivity.

    Args:
        x (ndarray): The logarithm with base 10 of relaxation times of the Debyes poles.
        rl_g (ndarray): Real parts of chosen relaxation function
                        for given frequency points.
        im_g (ndarray): Imaginary parts of chosen relaxation function
                        for given frequency points.
        freq (ndarray): The frequencies vector for defined grid.

    Returns:
        cost (float): Sum of mean absolute errors for real and imaginary part.
    """
    cost_i, cost_r, _, _, _, _ = DLS(rl_g, im_g, x, freq_g)
    return cost_i + cost_r
