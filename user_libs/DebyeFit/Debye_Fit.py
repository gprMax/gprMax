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
import os
from matplotlib import pylab as plt
import sys
import scipy.interpolate

from optimization import *


class Relaxation(object):

    def __init__(self, number_of_debye_poles,
                 sigma, mu, mu_sigma,
                 material_name, plot=True, save=True,
                 optimizer=Particle_swarm,
                 optimizer_options={}):
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
            save (bool): if True will save approximated material parameters
                         The argument is optional and if neglected save=False.
            optimizer (Optimizer class): chosen optimization method:
                                         Particle Swarm, Genetic or Dual Annealing.
            optimizer_options (dict): Additional keyword arguments passed to
                                      optimizer class (Default: empty dict).
        """
        self.number_of_debye_poles = number_of_debye_poles
        self.sigma = sigma
        self.mu = mu
        self.mu_sigma = mu_sigma
        self.material_name = material_name
        self.plot = plot
        self.save = save
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
        # if one of the weights is negative increase the stabiliser
        # and repeat the optimisation
        xmp, mx, ee, rp, ip = self.optimize()
        # Print the results in gprMax format style
        properties = self.print_output(xmp, mx, ee)
        if self.save:
            self.save_result(properties)
        # Plot the actual and the approximate dielectric properties
        if self.plot:
            self.plot_result(rp + ee, ip)

    def set_freq(self, f_min, f_max, n=50):
        """
        Interpolate frequency vector using n
        equally logarithmicaly spaced frequencies.

        Args:
            f_min (float): First bound of the frequency range
                           used to approximate the given function (Hz).
            f_max (float): Second bound of the frequency range
                           used to approximate the given function (Hz).
            n (int): Number of frequency points in frequency grid.
        Note:
            f_min and f_max must satisfied f_min < f_max
        """
        # diff_freq = np.log10(f_max) - np.log10(f_min)
        self.freq = np.logspace(np.log10(f_min) + 0.00001,
                                np.log10(f_max) - 0.00001,
                                int(n))
                                # int(n * diff_freq))
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

        Returns:
            xpm (array): The best known position form optimization module
                         (optimal design).
            mx (array): Resulting optimised weights for the given relaxation times
            ee (float): Average error between the actual and the approximated real part
            rp (matrix): The real part of the permittivity for the optimised relaxation
                         times and weights for the frequnecies included in freq
            ip (matrix): The imaginary part of the permittivity for the optimised
                         relaxation times and weights for the frequnecies included in freq
        """
        # Define the lower and upper boundaries of search
        lb = np.full(self.number_of_debye_poles,
                     -np.log10(np.max(self.freq)) - 3)
        ub = np.full(self.number_of_debye_poles,
                     -np.log10(np.min(self.freq)) + 3)
        # Call particle swarm optimisation to minimize the cost function.
        xmp, _ = self.optimizer.fit(func=cost_function,
                                    lb=lb, ub=ub,
                                    funckwargs={'rl_g': self.rl,
                                                'im_g': self.im,
                                                'freq_g': self.freq}
                                    )
        _, _, mx, ee, rp, ip = linear(self.rl, self.im, xmp, self.freq)
        return xmp, mx, ee, rp, ip

    def print_output(self, xmp, mx, ee):
        """
        Print out the resulting Debye parameters in a gprMax format.

        Args:
            xpm (array): The best known position form optimization module
                         (optimal design).
            mx (): Resulting optimised weights for the given relaxation times.
            ee (): Average error between the actual and the approximated real part.
        """
        print("Debye expansion parameters: ")
        print(f"        |{'e_inf':^14s}|{'De':^14s}|{'log(t0)':^25s}|")
        print("_" * 65)
        for i in range(0, len(xmp)):
            print("Debye {0:}:|{1:^14.5f}|{2:^14.5f}|{3:^25.5f}|"
                  .format(i + 1, ee/len(xmp), mx[i],
                          xmp[i]))
            print("_" * 65)

        # Print the Debye expnasion in a gprMax format
        material_prop = "#material: {} {} {} {} {}".format(ee, self.sigma,
                                                           self.mu,
                                                           self.mu_sigma,
                                                           self.material_name)
        print(material_prop)
        material_prop = [material_prop + '\n']
        dispersion_prop = "#add_dispersion_debye: {} {} {}".format(len(xmp),
                                                                   mx[0],
                                                                   10**xmp[0])
        for i in range(1, len(xmp)):
            dispersion_prop += " {} {}".format(mx[i], 10**xmp[i])
        dispersion_prop += " {}".format(self.material_name)
        print(dispersion_prop)
        material_prop.append(dispersion_prop + '\n')
        return material_prop

    def plot_result(self, rl_exp, im_exp):
        """
        Plot the actual and the approximated electric permittivity
        using a semilogarithm X axes.

        Args:
            rl_exp (array): Real parts of optimised Debye expansion
                            for given frequency points (plus average error).
            im_exp (array): Imaginary parts of optimised Debye expansion
                            for given frequency points.
        """
        plt.close("all")
        plt.rcParams["axes.facecolor"] = "black"
        plt.semilogx(self.freq * 1e-6, rl_exp, "b-", linewidth=2.0,
                     label="Debye Expansion: Real")
        plt.semilogx(self.freq * 1e-6, -im_exp, "w-", linewidth=2.0,
                     label="Debye Expansion: Imaginary")
        plt.semilogx(self.freq * 1e-6, self.rl, "ro",
                     linewidth=2.0, label="Chosen Function: Real")
        plt.semilogx(self.freq * 1e-6, -self.im, "go", linewidth=2.0,
                     label="Chosen Function: Imaginary")

        plt.rcParams["axes.facecolor"] = "white"
        plt.grid(b=True, which="major", color="w", linewidth=0.2,
                 linestyle="--")
        axes = plt.gca()
        axes.set_xlim([np.min(self.freq * 1e-6), np.max(self.freq * 1e-6)])
        axes.set_ylim([-1, np.max(np.concatenate([self.rl, -self.im])) + 1])
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Relative permittivity")
        plt.show()

    @staticmethod
    def save_result(output, fdir="../materials"):
        """
        Save the resulting Debye parameters in a gprMax format

        Args:
            output (str): Material and resulting Debye parameters
                          in a gprMax format.
            fdir (str): Path to saving directory.
        """
        if fdir != "../materials" and os.path.isdir(fdir):
            file_path = os.path.join(fdir, "my_materials.txt")
        elif os.path.isdir("../materials"):
            file_path = os.path.join("../materials",
                                     "my_materials.txt")
        elif os.path.isdir("materials"):
            file_path = os.path.join("materials",
                                     "my_materials.txt")
        elif os.path.isdir("user_libs/materials"):
            file_path = os.path.join("user_libs", "materials",
                                     "my_materials.txt")
        else:
            sys.exit("Cannot save material properties "
                     f"in {os.path.join(fdir, 'my_materials.txt')}!")
        fileH = open(file_path, "a")
        fileH.write(f"## {output[0].split(' ')[-1]}")
        fileH.writelines(output)
        fileH.write("\n")
        fileH.close()
        print(f"Material properties save at: {file_path}")


class HavriliakNegami(Relaxation):

    def __init__(self, number_of_debye_poles,
                 f_min, f_max, alfa, bita, einf, de, t0,
                 sigma, mu, mu_sigma,
                 material_name, plot=False, save=True,
                 optimizer=Particle_swarm,
                 optimizer_options={}):
        """
        Approximate a given Havriliak-Negami function
        Havriliak-Negami function = einf + de / (1 + (1j * 2 * pi * f *t0)**alfa )**bita,
                                    where f is the frequency in Hz

        Args:
            number_of_debye_poles (int): Number of Debye functions used to
                                         approximate the given electric
                                         permittivity.
            f_min (float): Define the first bound of the frequency range
                           used to approximate the given function (Hz).
            f_max (float): Define the second bound of the frequency range
                           used to approximate the given function (Hz).
                           Note: f_min and f_max can be either f_min > f_max
                           or f_min < f_max but not f_min = f_max.
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
                         permittivity. The argument is optional and
                         if neglected plot=False.
            save (bool): if True will save approximated material parameters
                         The argument is optional and if neglected save=False.
            optimizer (Optimizer class): chosen optimization method:
                                         Particle Swarm, Genetic or Dual Annealing.
                                         (Default: Partocle_swarm)
            optimizer_options (dict): Additional keyword arguments passed to
                                      optimizer class (Default: empty dict).
        """
        super(HavriliakNegami, self).__init__(number_of_debye_poles,
                                              sigma, mu, mu_sigma,
                                              material_name, plot, save,
                                              optimizer, optimizer_options)
        # Place the lower frequency bound at f_min and the upper frequency bound at f_max
        if f_min > f_max:
            self.f_min, self.f_max = f_max, f_min
        else:
            self.f_min, self.f_max = f_min, f_max
        # Choosing n frequencies logarithmicaly equally spaced between the bounds given
        self.set_freq(self.f_min, self.f_max)
        self.einf, self.alfa, self.bita, self.de, self.t0 = einf, alfa, bita, de, t0

    def check_inputs(self):
        """
        Check the validity of the inputs.
        """
        super(HavriliakNegami, self).check_inputs()
        try:
            d = [float(i) for i in
                 [self.f_min, self.f_max, self.alfa,
                  self.bita, self.einf, self.de, self.t0]]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")
        if self.alfa > 1:
            sys.exit("Alfa value must range between 0-1 (0 <= Alfa <= 1)")
        if self.bita > 1:
            sys.exit("Beta value must range between 0-1 (0 <= Beta <= 1)")
        if self.f_min == self.f_max:
            sys.exit("Null frequency range")

    def print_info(self):
        """Print information about chosen approximation settings."""
        print(f"Approximating Havriliak-Negami function"
              f" using {self.number_of_debye_poles} Debye poles")
        print("Havriliak-Negami parameters : ")
        print("De     =   {} \ne_inf  =   {} \nt0     =   {} \nalfa   =   {} \nbita   =   {}  "
              .format(self.de, self.einf, self.t0, self.alfa, self.bita))

    def calculation(self):
        """Calculates the Havriliak-Negami function for
        the given parameters."""
        return self.einf + self.de / (np.array(
                1 + np.array(1j * 2 * np.pi *
                             self.freq * self.t0
                            ) ** self.alfa)**self.bita)


class Jonscher(Relaxation):
    def __init__(self, number_of_debye_poles,
                 f_min, f_max, einf, ap, omegap, n_p,
                 sigma, mu, mu_sigma,
                 material_name, plot=False, save=True,
                 optimizer=Particle_swarm,
                 optimizer_options={}):
        """
        Approximate a given Johnsher function
        Jonscher function = einf - ap * ( -1j * 2 * pi * f / omegap)**n_p,
                            where f is the frequency in Hz

        Args:
            number_of_debye_poles (int): Number of Debye functions used to
                                         approximate the given electric
                                         permittivity.
            f_min (float): Define the first bound of the frequency range
                           used to approximate the given function (Hz).
            f_max (float): Define the second bound of the frequency range
                           used to approximate the given function (Hz).
                           f_min and f_max can be either f_min > f_max
                           or f_min < f_max but not f_min = f_max.
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
                         permittivity. The argument is optional and
                         if neglected plot=False.
            save (bool): if True will save approximated material parameters
                         The argument is optional and if neglected save=False.
            optimizer (Optimizer class): chosen optimization method:
                                         Particle Swarm, Genetic or Dual Annealing.
            optimizer_options (dict): Additional keyword arguments passed to
                                      optimizer class (Default: empty dict).
        """
        super(Jonscher, self).__init__(number_of_debye_poles,
                                       sigma, mu, mu_sigma,
                                       material_name, plot, save,
                                       optimizer, optimizer_options)
        # Place the lower frequency bound at f_min and the upper frequency bound at f_max
        if f_min > f_max:
            self.f_min, self.f_max = f_max, f_min
        else:
            self.f_min, self.f_max = f_min, f_max
        # Choosing n frequencies logarithmicaly equally spaced between the bounds given
        self.set_freq(self.f_min, self.f_max)
        self.einf, self.ap, self.omegap, self.n_p = einf, ap, omegap, n_p

    def check_inputs(self):
        """
        Check the validity of the inputs.
        """
        super(Jonscher, self).check_inputs()
        try:
            d = [float(i) for i in
                 [self.f_min, self.f_max, self.n_p,
                  self.einf, self.omegap, self.ap]]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")
        if self.n_p > 1:
            sys.exit("n_p value must range between 0-1 (0 <= n_p <= 1)")
        if self.f_min == self.f_max:
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
        return self.einf + (self.ap * np.array(
                            2 * np.pi * self.freq / self.omegap
                            )**(self.n_p-1)) * (
                            1 - 1j / np.tan(self.n_p * np.pi/2))


class Crim(Relaxation):

    def __init__(self, number_of_debye_poles,
                 f_min, f_max, a, f1, e1, sigma,
                 mu, mu_sigma, material_name, plot=False, save=True,
                 optimizer=Particle_swarm,
                 optimizer_options={}):
        """
        Approximate a given CRIM function
        CRIM = (sum([volumetric_fraction[i]*(material[i][0] + material[i][1] /
               (1 + (1j * 2 * pi * f *material[i][2])))**m_param
               for i in range(0,len(material))]))**1/m_param

        Args:
            number_of_debye_poles (int): Number of Debye functions used to
                                         approximate the given electric
                                         permittivity.
            f_min (float): Define the first bound of the frequency range
                           used to approximate the given function (Hz).
            f_max (float): Define the second bound of the frequency range
                           used to approximate the given function (Hz).
                           f_min and f_max can be either f_min > f_max
                           or f_min < f_max but not f_min = f_max.
            a (float): shape factor
            f1 (list): volumetric fraction
            e1 (list): materials
            sigma (float): Conductivity.
            mu (float): Relative permabillity.
            mu_sigma (float): Magnetic looses.
            material_name (str): A string containing the given name of
                                 the material (e.g. "Clay").
            plot (bool): if True will plot the actual and the approximated
                         permittivity. The argument is optional and
                         if neglected plot=False.
            save (bool): if True will save approximated material parameters
                         The argument is optional and if neglected save=False.
            optimizer (Optimizer class): chosen optimization method:
                                         Particle Swarm, Genetic or Dual Annealing.
            optimizer_options (dict): Additional keyword arguments passed to
                                      optimizer class (Default: empty dict).
        """
        super(Crim, self).__init__(number_of_debye_poles,
                                   sigma, mu, mu_sigma,
                                   material_name, plot, save,
                                   optimizer, optimizer_options)
        # Place the lower frequency bound at f_min and the upper frequency bound at f_max
        if f_min > f_max:
            self.f_min, self.f_max = f_max, f_min
        else:
            self.f_min, self.f_max = f_min, f_max
        # Choosing n frequencies logarithmicaly equally spaced between the bounds given
        self.set_freq(self.f_min, self.f_max)
        self.a, self.f1, self.e1 = a, f1, e1

    def check_inputs(self):
        """
        Check the validity of the inputs.
        """
        super(Crim, self).check_inputs()
        try:
            d = [float(i) for i in
                 [self.f_min, self.f_max, self.a]]
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
        if self.f_min == self.f_max:
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
                 (np.array(1 + np.array(1j * 2 * np.pi * f * self.e1[i][2])))
                 for f in self.freq])**self.a
        return q**(1 / self.a)


class Rawdata(Relaxation):

    def __init__(self, number_of_debye_poles,
                 filename,
                 sigma, mu, mu_sigma,
                 material_name, plot=False, save=True,
                 optimizer=Particle_swarm,
                 optimizer_options={}):
        """
        Interpolate data given from a text file.

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
            save (bool): if True will save approximated material parameters
                         The argument is optional and if neglected save=False.
            optimizer (Optimizer class): chosen optimization method:
                                         Particle Swarm, Genetic or Dual Annealing.
            optimizer_options (dict): Additional keyword arguments passed to
                                      optimizer class (Default: empty dict).
        """
        super(Rawdata, self).__init__(number_of_debye_poles,
                                      sigma, mu, mu_sigma,
                                      material_name, plot, save,
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
        """
        Interpolate real and imaginary part from data.
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

        self.set_freq(min(array[:, 0]), max(array[:, 0]))

        rl_interp = scipy.interpolate.interp1d(array[:, 0], array[:, 1])
        im_interp = scipy.interpolate.interp1d(array[:, 0], array[:, 2])
        return rl_interp(self.freq) - 1j * im_interp(self.freq)


if __name__ == "__main__":
    setup = Rawdata(3, "Test.txt", 0.1, 1, 0.1, "M1", plot=True,
                    optimizer_options={'seed':111,
                                       'pflag':True})
    setup.run()
    setup = HavriliakNegami(6, 1e12, 1e-3, 0.5, 1, 10, 5,
                            1e-6, 0.1, 1, 0, "M2", plot=True,
                            optimizer=Dual_annealing,
                            optimizer_options={'seed':111,
                                               'maxiter':50})
    setup.run()
    setup = Jonscher(4, 1e6, 1e-5, 50, 1, 1e5, 0.7,
                     0.1, 1, 0.1, "M3", plot=True,
                     optimizer_options={'seed':111})
    setup.run()
    f = [0.5, 0.5]
    material1 = [3, 25, 1e6]
    material2 = [3, 0, 1e3]
    materials = [material1, material2]
    setup = Crim(2, 1*1e-1, 1e-9, 0.5, f, materials, 0.1,
                 1, 0, "M4", plot=True,
                 optimizer_options={'seed':111})
    setup.run()
