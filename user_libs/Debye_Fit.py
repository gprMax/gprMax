# Author: Iraklis Giannakis
# E-mail: i.giannakis@ed.ac.uk
#
# Copyright (c) 2017 Iraklis Giannakis
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
import os
import math
from matplotlib import pylab as plt
import sys
import scipy.interpolate
from tqdm import tqdm


class Optimizer(object):
    
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
                 minstep=1e-8, pflag=False):
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
            pflag (bool): if True will plot the actual and the approximated
                          value during optimization process (Default: False).
        """
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.minstep = minstep
        self.pflag = pflag

    def fit(self, func, lb, ub, funckwargs={}):
        """
        A particle swarm optimisation that tries to find an optimal set
        of relaxation times that minimise the error
        between the actual and the approximated electric permittivity.
        The current class is a modified edition of the pyswarm package
        which can be found at https://pythonhosted.org/pyswarm/

        Args:
            func (function): The function to be minimized
            lb (array): The lower bounds of the design variable(s)
            ub (array): The upper bounds of the design variable(s)
            funckwargs (dict): Additional keyword arguments passed to
                               objective and constraint function
                               (Default: empty dict)

        Returns:
            g (float): The swarm's best known position (optimal design).
            fg (float): The objective value at ``g``.
        """
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
        for it in tqdm(range(2, self.maxiter + 2), desc='Debye fitting'):
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
                # (if constraints are satisfied)
                if fx < fp[i]:
                    p[i, :] = x[i, :].copy()
                    fp[i] = fx
                    # Compare swarm's best position to current
                    # particle's position
                    # (Can only get here if constraints are satisfied)
                    if fx < fg:
                        tmp = x[i, :].copy()
                        stepsize = np.sqrt(np.sum((g - tmp) ** 2))
                        if stepsize <= self.minstep:
                            print(f'Stopping search: Swarm best position change less than {self.minstep}')
                            return tmp, fx
                        else:
                            g = tmp.copy()
                            fg = fx

            # Dynamically plot the error as the optimisation takes place
            if self.pflag:
                if it == 2:
                    xpp = [0]
                    ypp = [fg]
                else:
                    xpp.append(it - 1)
                    ypp.append(fg)
                Particle_swarm.plot(xpp, ypp)

        return g, fg


class Relaxation(object):

    def __init__(self, number_of_debye_poles,
                 sigma, mu, mu_sigma,
                 material_name, plot=True,
                 optimizer=Particle_swarm,
                 optimizer_options={'pflag': True,
                                    'swarmsize': 40,
                                    'maxiter': 50,
                                    'omega': 0.9,
                                    'phip': 0.9,
                                    'phig': 0.9,
                                    'minstep': 1e-8}):
        """
        Create Relaxation function object for complex material.

        Args:
            number_of_debye_poles (int): Number of Debye functions used to
                                         approximate the given electric
                                         permittivity.
            sigma (float): Conductivity.
            mu (float): Relative permabillity.
            mu_sigma (float): Magnetic looses.
            material_name (str): A string containing the given name of
                                 the material (e.g. "Clay").
            plot (bool): if True will plot the actual and the approximated
                         permittivity (it can be neglected).
                         The argument is optional and if neglected plot=False.
            pso (list): A vector which contains 5 parameters [a1,a2,a3,a4,a5].
                        a1 denotes the number of particles to be used in
                        the particle swarm optimisation. a2 denotes the number
                        of iterations. a3 is the inertia component.
                        a4 is the cognitive, a5 - social scaling parameters.
                        By default pso = [40, 50, 0.9, 0.9, 0.9]
        """
        self.number_of_debye_poles = number_of_debye_poles
        self.sigma = sigma
        self.mu = mu
        self.mu_sigma = mu_sigma
        self.material_name = material_name
        self.plot = plot
        self.optimizer = optimizer(**optimizer_options)

    def run(self):
        """
        Solve the problem described by the given relaxation function
        (Havriliak-Negami function, Crim, Jonscher)
        or data given from a text file.
        """
        # Check the validity of the inputs
        self.check_inputs()
        # Print information about chosen approximation settings
        self.print_info()
        # Calculate both real and imaginary parts
        # for the frequencies included in the vector freq
        q = self.calculation()
        # Set the real and the imaginary part of the relaxation function
        self.rl, self.im = q.real, q.imag
        # Calling the main optimisation module
        self.optimize()

    def check_inputs(self):
        """
        Check the validity of the inputs.
        """
        try:
            d = [float(i) for i in
                 [self.number_of_debye_poles,
                  self.sigma, self.mu, self.mu_sigma]]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if not isinstance(self.number_of_debye_poles, int):
            sys.exit("The number of Debye poles must be integer.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")

    def calculation(self):
        """
        Approximate the given relaxation function
        (Havriliak-Negami function, Crim, Jonscher) or based on raw data.
        """
        raise NotImplementedError()

    def print_info(self):
        """
        Print information about chosen approximation settings.
        """
        raise NotImplementedError()

    def optimize(self):
        """
        Calling the main optimisation module.
        """
        # Define the lower and upper boundaries of search
        lb = np.full(self.number_of_debye_poles,
                     -np.log10(np.max(self.freq)) - 3)
        ub = np.full(self.number_of_debye_poles,
                     -np.log10(np.min(self.freq)) + 3)
        # Call particle swarm optimisation to minimize the cost function.
        xmp, _ = self.optimizer.fit(func=cost_function,
                                    lb=lb, ub=ub,
                                    funckwargs={'rl_g':self.rl,
                                                'im_g': self.im,
                                                'freq_g': self.freq}
                                    )
        _, _, mx, ee, rp, ip = linear(self.rl, self.im, xmp, self.freq)
        # if one of the weights is negative increase the stabiliser
        # and repeat the optimisation
        # Print the results in gprMax format style
        self.print_output(xmp, mx, ee)
        # Plot the actual and the approximate dielectric properties
        if self.plot:
            self.plot_result(rp + ee, ip)

    def print_output(self, xmp, mx, ee):
        """Print out the resulting Debye parameters in a gprMax format"""
        print("Debye expansion parameters : ")
        print("         |     e_inf     |       De      |         log(t0)        |")
        print("__________________________________________________________________")
        for i in range(0, len(xmp)):

            print("Debye {0:}:|"
                  .format(i + 1), "  {0:s}    |    {1:s}    |         {2:s}        | "
                  .format(str(ee/len(xmp))[0:7], str(mx[i])[0:7], str(xmp[i])[0:7]))
            print("__________________________________________________________________\n")
        print("\n")

        # Print the Debye expnasion in a gprMax format
        print("#material: {} {} {} {} {}".format(ee, self.sigma, self.mu, self.mu_sigma, self.material_name))
        out_t = "#add_dispersion_debye: {} {} {}".format(len(xmp), mx[0], 10**xmp[0])
        for i in range(1, len(xmp)):
            out_t += " {} {}".format(mx[i], 10**xmp[i])
        out_t += " {}".format(self.material_name)
        print(out_t)

    def plot_result(self, rl_exp, im_exp):
        """Plot the actual and the approximated electric permittivity using a semilogarithm X axes"""
        plt.close("all")
        plt.rcParams["axes.facecolor"] = "black"
        plt.semilogx(self.freq * 1e-6, rl_exp, "b-", linewidth=2.0, label="Debye Expansion: Real")
        plt.semilogx(self.freq * 1e-6, -im_exp, "w-", linewidth=2.0, label="Debye Expansion: Imaginary")
        plt.semilogx(self.freq * 1e-6, self.rl, "ro", linewidth=2.0, label="Chosen Function: Real")
        plt.semilogx(self.freq * 1e-6, -self.im, "go", linewidth=2.0, label="Chosen Function: Imaginary")

        plt.rcParams["axes.facecolor"] = "white"
        plt.grid(b=True, which="major", color="w", linewidth=0.2, linestyle="--")
        axes = plt.gca()
        axes.set_xlim([np.min(self.freq * 1e-6), np.max(self.freq * 1e-6)])
        axes.set_ylim([-1, np.max(np.concatenate([self.rl,-self.im])) + 1])
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Relative permittivity")
        plt.show()


class HavriliakNegami(Relaxation):

    def __init__(self, number_of_debye_poles,
                 freq1, freq2, alfa, bita, einf, de, t0,
                 sigma, mu, mu_sigma,
                 material_name, plot=False,
                 optimizer=Particle_swarm,
                 optimizer_options={'pflag': True,
                                    'swarmsize': 40,
                                    'maxiter': 50,
                                    'omega': 0.9,
                                    'phip': 0.9,
                                    'phig': 0.9,
                                    'minstep': 1e-8}):
        """
        Approximate a given Havriliak-Negami function
        Havriliak-Negami function = einf + de / (1 + (1j * 2 * math.pi * f *t0 )**alfa )**bita,
                                    where f is the frequency in Hz

        Args:
            number_of_debye_poles (int): Number of Debye functions used to
                                         approximate the given electric
                                         permittivity.
            freq1 (float): Define the first bound of the frequency range
                           used to approximate the given function (Hz).
            freq2 (float): Define the second bound of the frequency range
                           used to approximate the given function (Hz).
                           freq1 and freq2 can be either freq1 > freq2
                           or freq1 < freq2 but not freq1 = freq2.
            einf (float): The real relative permittivity at infinity frequency
            alfa (float): Havriliak-Negami parameter. Real positive float number
                          which varies 0 < alfa < 1. For alfa = 1 and bita !=0 & bita !=1
                          Havriliak-Negami transforms to Cole-Davidson function.
            bita (float): Havriliak-Negami parameter. Real positive float number
                          which varies 0 < bita < 1. For bita = 1 and alfa !=0 & alfa !=1
                          Havriliak-Negami transforms to Cole-Cole function.
            de (float): Havriliak-Negami parameter. Real positive float number.
                        de is the relative permittivity at infinite frequency
                        minus the relative permittivity at zero frequency.
            t0 (float): Havriliak_Negami parameter. Real positive float number.
                        t0 is the relaxation time.
            sigma (float): Conductivity.
            mu (float): Relative permabillity.
            mu_sigma (float): Magnetic looses.
            material_name (str): A string containing the given name of
                                 the material (e.g. "Clay").
            plot (bool): if True will plot the actual and the approximated
                         permittivity (it can be neglected).
                         The argument is optional and if neglected plot=False.
            pso (list): A vector which contains 5 parameters [a1,a2,a3,a4,a5].
                        a1 denotes the number of particles to be used in
                        the particle swarm optimisation. a2 denotes the number
                        of iterations. a3 is the inertia component.
                        a4 is the cognitive, a5 - social scaling parameters.
                        By default pso = [40, 50, 0.9, 0.9, 0.9]
        """
        super(HavriliakNegami, self).__init__(number_of_debye_poles,
                                              sigma, mu, mu_sigma,
                                              material_name, plot, optimizer, optimizer_options)
        # Place the lower frequency bound at fr1 and the upper frequency bound at fr2
        if freq1 > freq2:
            self.freq1, self.freq2 = freq2, freq1
        else:
            self.freq1, self.freq2 = freq1, freq2
        # Choosing 50 frequencies logarithmicaly equally spaced between the bounds given
        self.freq = np.logspace(np.log10(freq1), np.log10(freq2), 50)
        self.einf, self.alfa, self.bita, self.de, self.t0 = einf, alfa, bita, de, t0

    def check_inputs(self):
        """
        Check the validity of the inputs.
        """
        super(HavriliakNegami, self).check_inputs()
        try:
            d = [float(i) for i in
                 [self.freq1, self.freq2, self.alfa,
                  self.bita, self.einf, self.de, self.t0]]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")
        if self.alfa > 1:
            sys.exit("Alfa value must range between 0-1 (0 <= Alfa <= 1)")
        if self.bita > 1:
            sys.exit("Beta value must range between 0-1 (0 <= Beta <= 1)")
        if self.freq1 == self.freq2:
            sys.exit("Null frequency range")

    def print_info(self):
        """Print information about chosen approximation settings."""
        print(f"Approximating Havriliak-Negami function"
              f" using {self.number_of_debye_poles} Debye poles")
        print("Havriliak-Negami parameters : ")
        print("De     =   {} \ne_inf  =   {} \nt0     =   {} \nalfa   =   {} \nbita   =   {}  "
              .format(self.de, self.einf, self.t0, self.alfa, self.bita))

    def calculation(self):
        """Calculates the Havriliak-Negami function for the given parameters."""
        return self.einf + self.de / (np.array(1 + np.array(1j * 2 * math.pi *
               self.freq * self.t0) ** self.alfa) ** self.bita)


class Jonscher(Relaxation):
    def __init__(self, number_of_debye_poles,
                 freq1, freq2, einf, ap, omegap, n_p,
                 sigma, mu, mu_sigma,
                 material_name, plot=False, 
                 optimizer=Particle_swarm,
                 optimizer_options={'pflag': True,
                                    'swarmsize': 40,
                                    'maxiter': 50,
                                    'omega': 0.9,
                                    'phip': 0.9,
                                    'phig': 0.9,
                                    'minstep': 1e-8}):
        """
        Approximate a given Johnsher function
        Jonscher function = einf - ap*( -1j * 2 * math.pi * f / omegap ) ** n_p,
                            where f is the frequency in Hz

        Args:
            number_of_debye_poles (int): Number of Debye functions used to
                                         approximate the given electric
                                         permittivity.
            freq1 (float): Define the first bound of the frequency range
                           used to approximate the given function (Hz).
            freq2 (float): Define the second bound of the frequency range
                           used to approximate the given function (Hz).
                           freq1 and freq2 can be either freq1 > freq2
                           or freq1 < freq2 but not freq1 = freq2.
            einf (float): The real relative permittivity at infinity frequency
            ap (float): Jonscher parameter. Real positive float number.
            omegap (float): Jonscher parameter. Real positive float number.
            n_p (float): Jonscher parameter.
                         Real positive float number which varies 0 < n_p < 1.
            sigma (float): Conductivity.
            mu (float): Relative permabillity.
            mu_sigma (float): Magnetic looses.
            material_name (str): A string containing the given name of
                                 the material (e.g. "Clay").
            plot (bool): if True will plot the actual and the approximated
                         permittivity (it can be neglected).
                         The argument is optional and if neglected plot=False.
            pso (list): A vector which contains 5 parameters [a1,a2,a3,a4,a5].
                        a1 denotes the number of particles to be used in
                        the particle swarm optimisation. a2 denotes the number
                        of iterations. a3 is the inertia component.
                        a4 is the cognitive, a5 - social scaling parameters.
                        By default pso = [40, 50, 0.9, 0.9, 0.9]
        """
        super(Jonscher, self).__init__(number_of_debye_poles,
                                       sigma, mu, mu_sigma,
                                       material_name, plot, optimizer, optimizer_options)
        # Place the lower frequency bound at fr1 and the upper frequency bound at fr2
        if freq1 > freq2:
            self.freq1, self.freq2 = freq2, freq1
        else:
            self.freq1, self.freq2 = freq1, freq2
        # Choosing 50 frequencies logarithmicaly equally spaced between the bounds given
        self.freq = np.logspace(np.log10(freq1), np.log10(freq2), 50)
        self.einf, self.ap, self.omegap, self.n_p = einf, ap, omegap, n_p

    def check_inputs(self):
        """
        Check the validity of the inputs.
        """
        super(Jonscher, self).check_inputs()
        try:
            d = [float(i) for i in
                 [self.freq1, self.freq2, self.n_p,
                  self.einf, self.omegap, self.ap]]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")
        if self.n_p > 1:
            sys.exit("n_p value must range between 0-1 (0 <= n_p <= 1)")
        if self.freq1 == self.freq2:
            sys.exit("Error: Null frequency range")

    def print_info(self):
        """
        Print information about chosen approximation settings
        """
        print(f"Approximating Jonsher function"
              f" using {self.number_of_debye_poles} Debye poles")
        print("Jonhser function parameters : ")
        print(f"omega_p =   {self.omegap}\n"
              f"e_inf   =   {self.einf}\n"
              f"n_p     =   {self.n_p}\n"
              f"A_p     =   {self.ap}")

    def calculation(self):
        """Calculates the Q function for the given parameters"""
        return self.einf + (self.ap * np.array(2*math.pi*self.freq / self.omegap
               )**(self.n_p-1)) * (1 - 1j/math.tan(self.n_p * math.pi/2))


class Crim(Relaxation):

    def __init__(self, number_of_debye_poles,
                 freq1, freq2, a, f1, e1, sigma,
                 mu, mu_sigma, material_name, plot=False,
                 optimizer=Particle_swarm,
                 optimizer_options={'pflag': True,
                                    'swarmsize': 40,
                                    'maxiter': 50,
                                    'omega': 0.9,
                                    'phip': 0.9,
                                    'phig': 0.9,
                                    'minstep': 1e-8}):
        """
        Approximate a given CRIM function
        CRIM = (sum([volumetric_fraction[i]*(material[i][0] + material[i][1] /
               (1 + (1j * 2 * math.pi * f *material[i][2])))**m_param
               for i in range(0,len(material))]))**1/m_param

        Args:
            number_of_debye_poles (int): Number of Debye functions used to
                                         approximate the given electric
                                         permittivity.
            freq1 (float): Define the first bound of the frequency range
                           used to approximate the given function (Hz).
            freq2 (float): Define the second bound of the frequency range
                           used to approximate the given function (Hz).
                           freq1 and freq2 can be either freq1 > freq2
                           or freq1 < freq2 but not freq1 = freq2.
            a (float): shape factor
            f1 (list): volumetric fraction
            e1 (list): materials
            sigma (float): Conductivity.
            mu (float): Relative permabillity.
            mu_sigma (float): Magnetic looses.
            material_name (str): A string containing the given name of
                                 the material (e.g. "Clay").
            plot (bool): if True will plot the actual and the approximated
                         permittivity (it can be neglected).
                         The argument is optional and if neglected plot=False.
            pso (list): A vector which contains 5 parameters [a1,a2,a3,a4,a5].
                        a1 denotes the number of particles to be used in
                        the particle swarm optimisation. a2 denotes the number
                        of iterations. a3 is the inertia component.
                        a4 is the cognitive, a5 - social scaling parameters.
                        By default pso = [40, 50, 0.9, 0.9, 0.9]
        """
        super(Crim, self).__init__(number_of_debye_poles,
                                   sigma, mu, mu_sigma,
                                   material_name, plot, optimizer, optimizer_options)
        # Place the lower frequency bound at fr1 and the upper frequency bound at fr2
        if freq1 > freq2:
            self.freq1, self.freq2 = freq2, freq1
        else:
            self.freq1, self.freq2 = freq1, freq2
        # Choosing 50 frequencies logarithmicaly equally spaced between the bounds given
        self.freq = np.logspace(np.log10(freq1), np.log10(freq2), 50)
        self.a, self.f1, self.e1 = a, f1, e1

    def check_inputs(self):
        """
        Check the validity of the inputs.
        """
        super(Crim, self).check_inputs()
        try:
            d = [float(i) for i in
                 [self.freq1, self.freq2, self.a]]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")
        if len(self.f1) != len(self.e1):
            sys.exit("Number of volumetric volumes does not match the dielectric properties")
        # Check if the materials are at least two
        if len(self.f1) < 2:
            sys.exit("The materials should be at least 2")
        # Check if the frequency range is null
        if self.freq1 == self.freq2:
            sys.exit("Null frequency range")
        # Check if the inputs are positive
        f = [i for i in self.f1 if i < 0]
        if len(f) != 0:
            sys.exit("Error: The inputs should be positive")
        for i in range(0, len(self.f1)):
            f = [i for i in self.e1[i][:] if i < 0]
            if len(f) != 0:
                sys.exit("Error: The inputs should be positive")
        # Check if the summation of the volumetric fractions equal to one
        if np.sum(self.f1) != 1:
            sys.exit("Error: The summation of volumetric volumes should be equal to 1")

    def print_info(self):
        """
        Print information about chosen approximation settings
        """
        print(f"Approximating Complex Refractive Index Model (CRIM)"
              f" using {self.number_of_debye_poles} Debye poles")
        print("CRIM parameters : ")
        for i in range(0, len(self.f1)):
            print("Material {} :".format(i+1))
            print("---------------------------------")
            print("           Vol. fraction   = {}".format(self.f1[i]))
            print("                   e_inf   = {}".format(self.e1[i][0]))
            print("                   De      = {}".format(self.e1[i][1]))
            print("                   log(t0) = {}".format(np.log10(self.e1[i][2])))

    def calculation(self):
        """Calculates the Crim function for the given parameters"""
        q = np.zeros(len(self.freq))
        for i in range(len(self.f1)):
            q = q + self.f1[i]*np.array(
                [self.e1[i][0] + self.e1[i][1] /
                 (np.array(1 + np.array(1j * 2 * math.pi * f * self.e1[i][2])))
                 for f in self.freq])**self.a
        return q**(1 / self.a)


class Rawdata(Relaxation):

    def __init__(self, number_of_debye_poles,
                 filename,
                 sigma, mu, mu_sigma,
                 material_name, plot=False,
                 optimizer=Particle_swarm,
                 optimizer_options={'pflag': True,
                                    'swarmsize': 40,
                                    'maxiter': 50,
                                    'omega': 0.9,
                                    'phip': 0.9,
                                    'phig': 0.9,
                                    'minstep': 1e-8}):
        """
        Interpolate data given from a text file

        Args:
            number_of_debye_poles (int): Number of Debye functions used to
                                         approximate the given electric
                                         permittivity.
            filename (str): text file which contains three columns:
                            frequency (Hz),Real,Imaginary (separated by comma).
            sigma (float): Conductivity.
            mu (float): Relative permabillity.
            mu_sigma (float): Magnetic looses.
            material_name (str): A string containing the given name of
                                 the material (e.g. "Clay").
            plot (bool): if True will plot the actual and the approximated
                         permittivity (it can be neglected).
                         The argument is optional and if neglected plot=False.
            pso (list): A vector which contains 5 parameters [a1,a2,a3,a4,a5].
                        a1 denotes the number of particles to be used in
                        the particle swarm optimisation. a2 denotes the number
                        of iterations. a3 is the inertia component.
                        a4 is the cognitive, a5 - social scaling parameters.
                        By default pso = [40, 50, 0.9, 0.9, 0.9]
        """
        super(Rawdata, self).__init__(number_of_debye_poles,
                                      sigma, mu, mu_sigma,
                                      material_name, plot,
                                      optimizer, optimizer_options)
        self.filename = filename

    def check_inputs(self):
        """
        Check the validity of the inputs.
        """
        super(Rawdata, self).check_inputs()
        if not os.path.isfile(self.filename):
            sys.exit("File doesnt exists!")

    def print_info(self):
        """
        Print information about chosen approximation settings
        """
        print(f"Approximating the function given"
              f" from file name {self.filename}"
              f" using {self.number_of_debye_poles} Debye poles")

    def calculation(self):
        """Interpolate real and imaginary part from data.
         Column framework of the input file three columns comma-separated
         Frequency(Hz),Real,Imaginary
        """
        # Read the file
        with open(self.filename) as f:
            try:
                array = np.array(
                    [[float(x) for x in line.split(",")] for line in f]
                    )
            except ValueError:
                sys.exit("Error: The inputs should be numeric")

        # Interpolate using 50 equally logarithmicaly spaced frequencies
        self.freq = np.logspace(np.log10(min(array[:, 0])) + 0.00001,
                                np.log10(max(array[:, 0])) - 0.00001,
                                50)
        rl_interp = scipy.interpolate.interp1d(array[:, 0], array[:, 1])
        im_interp = scipy.interpolate.interp1d(array[:, 0], array[:, 2])
        return rl_interp(self.freq) + 1j * im_interp(self.freq)


def cost_function(x, rl_g, im_g, freq_g):
    """
    The cost function is the average error between
    the actual and the approximated electric permittivity.

    Returns:
        cost: The final error
    """
    cost, cost2, _, _, _, _ = linear(rl_g, im_g, x, freq_g)
    cost = cost + cost2
    return cost


def linear(rl, im, logt, freq):
    """
    Returns:
        x: Resulting optimised weights for the given relaxation times
        cost: The final error
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
    cost = np.sum([np.abs(ip[i]-im[i]) for i in range(0, len(im))])/len(im)
    ee = (np.mean(rl - rp))
    if ee < 1:
        ee = 1
    cost2 = np.sum([np.abs(rp[i] - rl[i] + ee)
                    for i in range(0, len(im))])/len(im)
    return cost, cost2, x, ee, rp, ip


def calc(cal_inputs, freq):
    # Calculates the Havriliak-Negami function for the given cal_inputs
    q = [cal_inputs[2] + cal_inputs[3] / (np.array(1 + np.array(
         1j * 2 * math.pi * f * cal_inputs[4]) ** cal_inputs[0]
         ) ** cal_inputs[1]) for f in freq]
    # Return the real and the imaginary part of the relaxation function
    if len(q) > 1:
        rl = [q[i].real for i in range(0, len(q))]
        im = [q[i].imag for i in range(0, len(q))]
    else:
        rl = q[0].real
        im = q[0].imag
    return rl, im


if __name__ == "__main__":
    np.random.seed(111)
    setup = Rawdata(3, "Test.txt", 0.1, 1, 0.1, "M1", plot=True)
    setup.run()
    setup = HavriliakNegami(6, 1*10**12, 10**-3, 0.5, 1, 10, 5,
                            10**-6, 0.1, 1, 0, "M2", plot=True)
    setup.run()
    setup = Jonscher(4, 10**6, 10**-5, 50, 1, 10**5, 0.7,
                     0.1, 1, 0.1, "M3", plot=True)
    setup.run()
    f = [0.5, 0.5]
    material1 = [3, 25, 10**6]
    material2 = [3, 0, 10**3]
    materials = [material1, material2]
    setup = Crim(2, 1*10**-1, 10**-9, 0.5, f, materials, 0.1,
                 1, 0, "M4", plot=True)
    setup.run()
