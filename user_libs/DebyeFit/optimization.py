# Author: Iraklis Giannakis, Sylwia Majchrowska
# E-mail: i.giannakis@ed.ac.uk
#
# Copyright (c) 2021 gprMax
# All rights reserved.
#
# Redistribution and use in source and binary forms are permitted
# provided that the above copyright notice and this paragraph are
# duplicated in all such forms and that any documentation,
# advertising materials, and other materials related to such
# distribution and use acknowledge that the software was developed
# as part of gprMax. The name of gprMax may not be used to
# endorse or promote products derived from this software without
# specific prior written permission.
# THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

import numpy as np
from matplotlib import pylab as plt
import scipy.optimize
from tqdm import tqdm


class Optimizer(object):

    def __init__(self, maxiter=1000, seed=None):
        """
        Create particle swarm optimisation object.

        Args:
            maxiter (int): The maximum number of iterations for the swarm
                        to search (Default: 1000).
            seed (int): Seed for RandomState.
                        Must be convertible to 32 bit unsigned integers.
        """
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
        plt.rcParams["axes.facecolor"] = "black"
        plt.plot(x, y, "b-", linewidth=3.0)
        plt.ylim(min(y) - 0.1 * min(y),
                 max(y) + 0.1 * max(y))
        plt.xlim(min(x), max(x))
        plt.grid(b=True, which="major", color="w",
                 linewidth=0.2, linestyle="--")
        plt.suptitle("Debye fitting process")
        plt.xlabel("Iteration")
        plt.ylabel("Average Error")
        plt.pause(0.0001)


class Particle_swarm(Optimizer):
    def __init__(self, swarmsize=40, maxiter=50,
                 omega=0.9, phip=0.9, phig=0.9,
                 minstep=1e-8, minfun=1e-8,
                 pflag=False, seed=None):
        """
        Create particle swarm optimisation object with predefined parameters.

        Args:
            swarmsize (int): The number of particles in the swarm (Default: 40).
            maxiter (int): The maximum number of iterations for the swarm
                        to search (Default: 50).
            omega (float): Particle velocity scaling factor (Default: 0.9).
            phip (float): Scaling factor to search away from the particle's
                        best known position (Default: 0.9).
            phig (float):  Scaling factor to search away from the swarm's
                        best known position (Default: 0.9).
            minstep (float): The minimum stepsize of swarm's best position
                             before the search terminates (Default: 1e-8).
            minfun (float): The minimum change of swarm's best objective value
                             before the search terminates (Default: 1e-8)
            pflag (bool): if True will plot the actual and the approximated
                          value during optimization process (Default: False).
        """
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
        The current class is a modified edition of the pyswarm package
        which can be found at https://pythonhosted.org/pyswarm/

        Args:
            func (function): The function to be minimized.
            lb (array): The lower bounds of the design variable(s).
            ub (array): The upper bounds of the design variable(s).
            funckwargs (dict): Additional keyword arguments passed to
                               objective and constraint function
                               (Default: empty dict).

        Returns:
            g (array): The swarm's best known position (optimal design).
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
    def __init__(self, maxiter=100,
                 local_search_options={}, initial_temp=5230.0,
                 restart_temp_ratio=2e-05, visit=2.62, accept=- 5.0,
                 maxfun=1e7, no_local_search=False,
                 callback=None, x0=None, seed=None):
        """
        Create dual annealing object with predefined parameters.

        Args:
            maxiter (int): The maximum number of iterations for the swarm
                           to search (Default: 100).
            local_search_options (dict): Extra keyword arguments to be passed
                                         to the local minimizer, reffer to
                                         scipy.optimize.minimize() function
                                         (Default: empty dict).
            initial_temp (float): The initial temperature, use higher values to
                                  facilitates a wider search of the energy
                                  landscape, allowing dual_annealing to escape
                                  local minima that it is trapped in.
                                  Range is (0.01, 5.e4] (Default: 5230).
            restart_temp_ratio (float): During the annealing process,
                                        temperature is decreasing, when it
                                        reaches initial_temp * restart_temp_ratio,
                                        the reannealing process is triggered.
                                        Range is (0, 1) (Default: 2e-5).
            visit (float): Parameter for visiting distribution. The value range is (1, 3]
                           (Default: 2.62).
            accept (float): Parameter for acceptance distribution. It is used to control
                            the probability of acceptance. The lower the acceptance parameter,
                            the smaller the probability of acceptance. The value range (-1e4, -5]
                            (Default: -5.0).
            no_local_search (bool):
            maxfun (int): Soft limit for the number of objective function calls.
                          (Default: 1e7).
            callback (callable): A callback function with signature callback(x, f, context),
                                 which will be called for all minima found.
                                 x and f are the coordinates and function value of
                                 the latest minimum found, and context has value in [0, 1, 2],
                                 with the following meaning:
                                 0: minimum detected in the annealing process.
                                 1: detection occurred in the local search process.
                                 2: detection done in the dual annealing process.
                                 If the callback implementation returns True,
                                 the algorithm will stop.
            x0 (ndarray): Coordinates of a single N-D starting point, shape(n,).
                          (Default: None).
            seed (None, int): Specify seed for repeatable minimizations.
                              The random numbers generated with this seed only
                              affect the visiting distribution function and
                              new coordinates generation (Default: None).
            pflag (bool): if True will plot the actual and the approximated
                          value during optimization process (Default: False).
        """
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
        #self.pflag = pflag

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


def cost_function(x, rl_g, im_g, freq_g):
    """
    The cost function is the average error between
    the actual and the approximated electric permittivity.

    Returns:
        cost: The final error
    """
    cost1, cost2, _, _, _, _ = linear(rl_g, im_g, x, freq_g)
    cost = cost1 + cost2
    return cost


def linear(rl, im, logt, freq):
    """
    Returns:
        cost1: Error (?)
        cost2: Error (?)
        x: Resulting optimised weights for the given relaxation times
        ee: Average error between the actual and the approximated real part
        rp: The real part of the permittivity for the optimised relaxation
            times and weights for the frequnecies included in freq
        ip: The imaginary part of the permittivity for the optimised
            relaxation times and weights for the frequnecies included in freq
    """
    # The relaxation time of the Debyes are given at as logarithms
    # logt=log10(t0) for efficiency during the optimisation
    # Here they are transformed back t0=10**logt
    tt = [10**logt[i] for i in range(0, len(logt))]
    # y = Ax , here the A matrix for the real and the imaginary part is builded
    d_r = np.array(
        [[calc([1, 1, 0, 1, tt[i]], [freq[j]])[0]
         for i in range(0, len(tt))] for j in
         range(0, len(freq))])
    d = np.array(
        [[calc([1, 1, 0, 1, tt[i]], [freq[j]])[1]
         for i in range(0, len(tt))] for j in
         range(0, len(freq))])

    # Adding dumping (Marquart least squares)
    # Solving the overdetermined system y=Ax
    x = np.abs(np.linalg.lstsq(d, im)[0])
    mx, my, my2 = np.matrix(x), np.matrix(d), np.matrix(d_r)
    rp, ip = my2 * np.transpose(mx), my * np.transpose(mx)
    cost1 = np.sum([np.abs(ip[i]-im[i]) for i in range(0, len(im))])/len(im)
    ee = (np.mean(rl - rp))
    if ee < 1:
        ee = 1
    cost2 = np.sum([np.abs(rp[i] - rl[i] + ee)
                    for i in range(0, len(im))])/len(im)
    return cost1, cost2, x, ee, rp, ip


def calc(cal_inputs, freq):
    # Calculates the Havriliak-Negami function for the given cal_inputs
    q = [cal_inputs[2] + cal_inputs[3] / (np.array(1 + np.array(
         1j * 2 * np.pi * f * cal_inputs[4]) ** cal_inputs[0]
         ) ** cal_inputs[1]) for f in freq]
    # Return the real and the imaginary part of the relaxation function
    if len(q) > 1:
        rl = [q[i].real for i in range(0, len(q))]
        im = [q[i].imag for i in range(0, len(q))]
    else:
        rl = q[0].real
        im = q[0].imag
    return rl, im
