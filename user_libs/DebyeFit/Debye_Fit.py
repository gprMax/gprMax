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
import os
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
import sys
import scipy.interpolate
import warnings

from optimization import *


class Relaxation(object):
    """ Create Relaxation function object for complex material.

    :param sigma: The conductivity (Siemens/metre).
    :type sigma: float, non-optional
    :param mu: The relative permeability.
    :type mu: float, non-optional
    :param mu_sigma: The magnetic loss.
    :type mu_sigma: float, non-optional
    :param material_name: A string containing the given name of
                          the material (e.g. "Clay").
    :type material_name: str, non-optional
    :param: number_of_debye_poles: Number of Debye functions used to
                                   approximate the given electric
                                   permittivity.
    :type number_of_debye_poles: int, optional
    :param: fn: Number of frequency points in frequency grid.
    :type fn: int, optional (Default: 50)
    :param plot: if True will plot the actual and the approximated
                 permittivity at the end (neglected as default: False).
    :type plot: bool, optional, default:False
    :param save: if True will save approximated material parameters
                        (not neglected as default: True).
    :type save: bool, optional, default:True
    :param optimizer: chosen optimization method:
                                        Hybrid Particle Swarm-Damped Least-Squares,
                                        Genetic or Dual Annealing (DA)
                                        (Default: PSO_DLS).
    :type optimizer: Optimizer class, optional
    :param optimizer_options: Additional keyword arguments passed to
                                     optimizer class (Default: empty dict).
    :type optimizer_options: dict, optional, default: empty dict
    """
    
    def __init__(self, sigma, mu, mu_sigma,
                 material_name, f_n=50,
                 number_of_debye_poles=-1,
                 plot=True, save=True,
                 optimizer=PSO_DLS,
                 optimizer_options={}):
        self.name = 'Relaxation function'
        self.params = {}
        self.number_of_debye_poles = number_of_debye_poles
        self.f_n = f_n
        self.sigma = sigma
        self.mu = mu
        self.mu_sigma = mu_sigma
        self.material_name = material_name
        self.plot = plot
        self.save = save
        self.optimizer = optimizer(**optimizer_options)

    def set_freq(self, f_min, f_max, f_n=50):
        """ Interpolate frequency vector using n equally logarithmicaly spaced frequencies.

        Args:
            f_min (float): First bound of the frequency range
                           used to approximate the given function (Hz).
            f_max (float): Second bound of the frequency range
                           used to approximate the given function (Hz).
            f_n (int): Number of frequency points in frequency grid
                       (Default: 50).
        Note:
            f_min and f_max must satisfied f_min < f_max
        """
        if abs(f_min - f_max) > 1e12:
            warnings.warn(f'The chosen range is realy big. '
                          f'Consider setting greater number of points '
                          f'on the frequency grid!')
        self.freq = np.logspace(np.log10(f_min),
                                np.log10(f_max),
                                int(f_n))

    def check_inputs(self):
        """ Check the validity of the inputs. """
        try:
            d = [float(i) for i in
                 [self.number_of_debye_poles,
                  self.sigma, self.mu, self.mu_sigma]]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if self.number_of_debye_poles <= 0:
            sys.exit("The number of Debye poles must be positive.")
        if not isinstance(self.number_of_debye_poles, int):
            sys.exit("The number of Debye poles must be integer.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")

    def calculation(self):
        """ Approximate the given relaxation function
        (Havriliak-Negami function, Crim, Jonscher) or based on raw data.
        """
        raise NotImplementedError()

    def print_info(self):
        """Readable string of parameters for given approximation settings.

        Returns:
            s (str): Info about chosen function and its parameters.
        """
        print(f"Approximating {self.name}"
              f" using {self.number_of_debye_poles} Debye poles")
        print(f"{self.name} parameters: ")
        s = ''
        for k, v in self.params.items():
            s += f"{k:10s} = {v}\n"
        print(s)
        return f'{self.name}:\n{s}'

    def optimize(self):
        """ Calling the main optimisation module with defined lower and upper boundaries of search.

        Returns:
            tau (ndarray): The optimised relaxation times.
            weights (ndarray): Resulting optimised weights for the given relaxation times.
            ee (float): Average error between the actual and the approximated real part.
            rl (ndarray): Real parts of chosen relaxation function
                          for given frequency points.
            im (ndarray): Imaginary parts of chosen relaxation function
                          for given frequency points.
        """
        # Define the lower and upper boundaries of search
        lb = np.full(self.number_of_debye_poles,
                     -np.log10(np.max(self.freq)) - 3)
        ub = np.full(self.number_of_debye_poles,
                     -np.log10(np.min(self.freq)) + 3)
        # Call optimizer to minimize the cost function
        tau, weights, ee, rl, im = self.optimizer.fit(func=self.optimizer.cost_function,
                                                      lb=lb, ub=ub,
                                                      funckwargs={'rl': self.rl,
                                                                  'im': self.im,
                                                                  'freq': self.freq}
                                                     )
        return tau, weights, ee, rl, im

    def run(self):
        """ Solve the problem described by the given relaxation function
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
        err_real, err_imag = self.error(rp + ee, ip)
        print(f'The average fractional error for:\n'
              f'- real part: {err_real}\n'
              f'- imaginary part: {err_imag}\n')
        if self.save:
            self.save_result(properties)
        # Plot the actual and the approximate dielectric properties
        if self.plot:
            self.plot_result(rp + ee, ip)

    def print_output(self, xmp, mx, ee):
        """ Print out the resulting Debye parameters in a gprMax format.

        Args:
            xpm (ndarray): The best known position form optimization module
                           (optimal design).
            mx (ndarray): Resulting optimised weights for the given relaxation times.
            ee (float): Average error between the actual and the approximated real part.

        Returns:
            material_prop (list(str)): Given material nad Debye expnasion parameters
                                       in a gprMax format.
        """
        print("Debye expansion parameters: ")
        print(f"       |{'e_inf':^14s}|{'De':^14s}|{'log(tau_0)':^25s}|")
        print("_" * 65)
        for i in range(0, len(xmp)):
            print("Debye {0:}|{1:^14.5f}|{2:^14.5f}|{3:^25.5f}|"
                  .format(i + 1, ee/len(xmp), mx[i],
                          xmp[i]))
            print("_" * 65)

        # Print the Debye expnasion in a gprMax format
        material_prop = []
        material_prop.append("#material: {} {} {} {} {}\n".format(ee, self.sigma,
                                                                  self.mu,
                                                                  self.mu_sigma,
                                                                  self.material_name))
        print(material_prop[0], end="")
        dispersion_prop = "#add_dispersion_debye: {}".format(len(xmp))
        for i in range(len(xmp)):
            dispersion_prop += " {} {}".format(mx[i], 10**xmp[i])
        dispersion_prop += " {}".format(self.material_name)
        print(dispersion_prop)
        material_prop.append(dispersion_prop + '\n')
        return material_prop

    def plot_result(self, rl_exp, im_exp):
        """ Plot the actual and the approximated electric permittivity,
        along with relative error for real and imaginary parts
        using a semilogarithm X axes.

        Args:
            rl_exp (ndarray): Real parts of optimised Debye expansion
                              for given frequency points (plus average error).
            im_exp (ndarray): Imaginary parts of optimised Debye expansion
                              for given frequency points.
        """
        plt.close("all")
        fig = plt.figure(figsize=(16,8), tight_layout=True)
        gs = gridspec.GridSpec(2, 1)
        ax = fig.add_subplot(gs[0])
        ax.grid(b=True, which="major", linewidth=0.2, linestyle="--")
        ax.semilogx(self.freq * 1e-6, rl_exp, "b-", linewidth=2.0,
                    label="Debye Expansion: Real part")
        ax.semilogx(self.freq * 1e-6, -im_exp, "k-", linewidth=2.0,
                    label="Debye Expansion: Imaginary part")
        ax.semilogx(self.freq * 1e-6, self.rl, "r.",
                    linewidth=2.0, label=f"{self.name}: Real part")
        ax.semilogx(self.freq * 1e-6, -self.im, "g.", linewidth=2.0,
                    label=f"{self.name}: Imaginary part")
        ax.set_ylim([-1, np.max(np.concatenate([self.rl, -self.im])) + 1])
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Relative permittivity")

        ax = fig.add_subplot(gs[1])
        ax.grid(b=True, which="major", linewidth=0.2, linestyle="--")
        ax.semilogx(self.freq * 1e-6, (rl_exp - self.rl)/self.rl * 100, "b-", linewidth=2.0,
                    label="Real part")
        ax.semilogx(self.freq * 1e-6, (-im_exp + self.im)/self.rl * 100, "k-", linewidth=2.0,
                    label="Imaginary part")
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Approximation error (%)")
        plt.show()

    def error(self, rl_exp, im_exp):
        """ Calculate the average fractional error separately for
        relative permittivity (real part) and conductivity (imaginary part)

        Args:
            rl_exp (ndarray): Real parts of optimised Debye expansion
                              for given frequency points (plus average error).
            im_exp (ndarray): Imaginary parts of optimised Debye expansion
                              for given frequency points.
        Returns:
            avg_err_real (float): average fractional error
                                  for relative permittivity (real part)
            avg_err_imag (float): average fractional error
                                  for conductivity (imaginary part)
        """
        avg_err_real = np.sum(np.abs((rl_exp - self.rl)/self.rl) * 100)/len(rl_exp)
        avg_err_imag = np.sum(np.abs((im_exp - self.im)/self.im) * 100)/len(im_exp)
        return avg_err_real, avg_err_imag

    @staticmethod
    def save_result(output, fdir="../materials"):
        """ Save the resulting Debye parameters in a gprMax format.

        Args:
            output (list(str)): Material and resulting Debye parameters
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
    """ Approximate a given Havriliak-Negami function
    Havriliak-Negami function = ε_∞ + Δ‎ε / (1 + (2πfjτ)**α)**β,
                                where f is the frequency in Hz.

    :param f_min: First bound of the frequency range
                  used to approximate the given function (Hz).
    :type f_min: float
    :param f_max: Second bound of the frequency range
                  used to approximate the given function (Hz).
    :type f_max: float
    :param e_inf: The real relative permittivity at infinity frequency
    :type e_inf: float
    :param alpha: Real positive float number which varies 0 < alpha < 1.
                 For alpha = 1 and beta !=0 & beta !=1 Havriliak-Negami
                 transforms to Cole-Davidson function.
    :type alpha: float
    :param beta: Real positive float number which varies 0 < beta < 1.
                 For beta = 1 and alpha !=0 & alpha !=1 Havriliak-Negami
                 transforms to Cole-Cole function.
    :type beta: float
    :param de: The difference of relative permittivity at infinite frequency 
               and the relative permittivity at zero frequency.
    :type de: float
    :param tau_0: Real positive float number, tau_0 is the relaxation time.
    :type tau_0: float
    """
    def __init__(self, f_min, f_max,
                 alpha, beta, e_inf, de, tau_0,
                 sigma, mu, mu_sigma, material_name,
                 number_of_debye_poles=-1, f_n=50,
                 plot=False, save=True,
                 optimizer=PSO_DLS,
                 optimizer_options={}):
        super(HavriliakNegami, self).__init__(sigma=sigma, mu=mu, mu_sigma=mu_sigma,
                                              material_name=material_name, f_n=f_n,
                                              number_of_debye_poles=number_of_debye_poles,
                                              plot=plot, save=save,
                                              optimizer=optimizer,
                                              optimizer_options=optimizer_options)
        self.name = 'Havriliak-Negami function'
        # Place the lower frequency bound at f_min and the upper frequency bound at f_max
        if f_min > f_max:
            self.f_min, self.f_max = f_max, f_min
        else:
            self.f_min, self.f_max = f_min, f_max
        # Choosing n frequencies logarithmicaly equally spaced between the bounds given
        self.set_freq(self.f_min, self.f_max, self.f_n)
        self.e_inf, self.alpha, self.beta, self.de, self.tau_0 = e_inf, alpha, beta, de, tau_0
        self.params = {'f_min':self.f_min, 'f_max':self.f_max,
                       'eps_inf':self.e_inf, 'Delta_eps':self.de, 'tau_0':self.tau_0,
                       'alpha':self.alpha, 'beta':self.beta}

    def check_inputs(self):
        """ Check the validity of the Havriliak Negami model's inputs. """
        super(HavriliakNegami, self).check_inputs()
        try:
            d = [float(i) for i in self.params.values()]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")
        if self.alpha > 1:
            sys.exit("Alpha value must range between 0-1 (0 <= alpha <= 1)")
        if self.beta > 1:
            sys.exit("Beta value must range between 0-1 (0 <= beta <= 1)")
        if self.f_min == self.f_max:
            sys.exit("Null frequency range")

    def calculation(self):
        """Calculates the Havriliak-Negami function for
           the given parameters."""
        return self.e_inf + self.de / (
                     1 + (1j * 2 * np.pi *
                     self.freq * self.tau_0)**self.alpha
                                     )**self.beta

class Jonscher(Relaxation):
    """ Approximate a given Johnsher function
    Jonscher function = ε_∞ - ap * (-1j * 2πf / omegap)**n_p,
                        where f is the frequency in Hz

    :param f_min: First bound of the frequency range
                  used to approximate the given function (Hz).
    :type f_min: float
    :param f_max: Second bound of the frequency range
                  used to approximate the given function (Hz).
    :type f_max: float
    :params e_inf: The real relative permittivity at infinity frequency.
    :type e_inf: float, non-optional
    :params a_p: Jonscher parameter. Real positive float number.
    :type a_p: float, non-optional
    :params omega_p: Jonscher parameter. Real positive float number.
    :type omega_p: float, non-optional
    :params n_p: Jonscher parameter, 0 < n_p < 1.
    :type n_p: float, non-optional
    """
    def __init__(self, f_min, f_max,
                 e_inf, a_p, omega_p, n_p,
                 sigma, mu, mu_sigma,
                 material_name, number_of_debye_poles=-1,
                 f_n=50, plot=False, save=True,
                 optimizer=PSO_DLS,
                 optimizer_options={}):
        super(Jonscher, self).__init__(sigma=sigma, mu=mu, mu_sigma=mu_sigma,
                                       material_name=material_name, f_n=f_n,
                                       number_of_debye_poles=number_of_debye_poles,
                                       plot=plot, save=save,
                                       optimizer=optimizer,
                                       optimizer_options=optimizer_options)
        self.name = 'Jonsher function'
        # Place the lower frequency bound at f_min and the upper frequency bound at f_max
        if f_min > f_max:
            self.f_min, self.f_max = f_max, f_min
        else:
            self.f_min, self.f_max = f_min, f_max
        # Choosing n frequencies logarithmicaly equally spaced between the bounds given
        self.set_freq(self.f_min, self.f_max, self.f_n)
        self.e_inf, self.a_p, self.omega_p, self.n_p = e_inf, a_p, omega_p, n_p
        self.params = {'f_min':self.f_min, 'f_max':self.f_max,
                       'eps_inf':self.e_inf, 'n_p':self.n_p,
                       'omega_p':self.omega_p, 'a_p':self.a_p}

    def check_inputs(self):
        """ Check the validity of the inputs. """
        super(Jonscher, self).check_inputs()
        try:
            d = [float(i) for i in self.params.values()]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")
        if self.n_p > 1:
            sys.exit("n_p value must range between 0-1 (0 <= n_p <= 1)")
        if self.f_min == self.f_max:
            sys.exit("Error: Null frequency range!")

    def calculation(self):
        """Calculates the Q function for the given parameters"""
        return self.e_inf + (self.a_p * (2 * np.pi * 
                      self.freq / self.omega_p)**(self.n_p-1)) * (
                      1 - 1j / np.tan(self.n_p * np.pi/2))


class Crim(Relaxation):
    """ Approximate a given CRIM function
    CRIM = (Σ frac_i * (‎ε_∞_i + Δ‎ε_i/(1 + 2πfj*τ_i))^a)^(1/a)

    :param f_min: First bound of the frequency range
                  used to approximate the given function (Hz).
    :type f_min: float
    :param f_max: Second bound of the frequency range
                  used to approximate the given function (Hz).
    :type f_max: float
    :param a: Shape factor.
    :type a: float, non-optional
    :param: volumetric_fractions: Volumetric fraction for each material.
    :type volumetric_fractions: ndarray, non-optional
    :param materials: Arrays of materials properties, for each material [e_inf, de, tau_0].
    :type materials: ndarray, non-optional
    """

    def __init__(self, f_min, f_max, a, volumetric_fractions,
                 materials, sigma, mu, mu_sigma, material_name, 
                 number_of_debye_poles=-1, f_n=50,
                 plot=False, save=True,
                 optimizer=PSO_DLS,
                 optimizer_options={}):

        super(Crim, self).__init__(sigma=sigma, mu=mu, mu_sigma=mu_sigma,
                                   material_name=material_name, f_n=f_n,
                                   number_of_debye_poles=number_of_debye_poles,
                                   plot=plot, save=save,
                                   optimizer=optimizer,
                                   optimizer_options=optimizer_options)
        self.name = 'CRIM function'
        # Place the lower frequency bound at f_min and the upper frequency bound at f_max
        if f_min > f_max:
            self.f_min, self.f_max = f_max, f_min
        else:
            self.f_min, self.f_max = f_min, f_max
        # Choosing n frequencies logarithmicaly equally spaced between the bounds given
        self.set_freq(self.f_min, self.f_max, self.f_n)
        self.a = a
        self.volumetric_fractions = volumetric_fractions
        self.materials = materials
        self.params = {'f_min':self.f_min, 'f_max':self.f_max,
                       'a':self.a, 'volumetric_fractions':self.volumetric_fractions,
                       'materials':self.materials}

    def check_inputs(self):
        """ Check the validity of the inputs. """
        super(Crim, self).check_inputs()
        try:
            d = [float(i) for i in
                 [self.f_min, self.f_max, self.a]]
        except ValueError:
            sys.exit("The inputs should be numeric.")
        if (np.array(d) < 0).sum() != 0:
            sys.exit("The inputs should be positive.")
        if len(self.volumetric_fractions) != len(self.materials):
            sys.exit("Number of volumetric volumes does not match the dielectric properties")
        # Check if the materials are at least two
        if len(self.volumetric_fractions) < 2:
            sys.exit("The materials should be at least 2")
        # Check if the frequency range is null
        if self.f_min == self.f_max:
            sys.exit("Null frequency range")
        # Check if the inputs are positive
        f = [i for i in self.volumetric_fractions if i < 0]
        if len(f) != 0:
            sys.exit("Error: The inputs should be positive")
        for i in range(len(self.volumetric_fractions)):
            f = [i for i in self.materials[i][:] if i < 0]
            if len(f) != 0:
                sys.exit("Error: The inputs should be positive")
        # Check if the summation of the volumetric fractions equal to one
        if np.sum(self.volumetric_fractions) != 1:
            sys.exit("Error: The summation of volumetric volumes should be equal to 1")

    def print_info(self):
        """ Print information about chosen approximation settings """
        print(f"Approximating Complex Refractive Index Model (CRIM)"
              f" using {self.number_of_debye_poles} Debye poles")
        print("CRIM parameters: ")
        for i in range(len(self.volumetric_fractions)):
            print("Material {}.:".format(i+1))
            print("---------------------------------")
            print(f"{'Vol. fraction':>27s} = {self.volumetric_fractions[i]}")
            print(f"{'e_inf':>27s} = {self.materials[i][0]}")
            print(f"{'De':>27s} = {self.materials[i][1]}")
            print(f"{'log(tau_0)':>27s} = {np.log10(self.materials[i][2])}")

    def calculation(self):
        """Calculates the Crim function for the given parameters"""
        return np.sum(np.repeat(self.volumetric_fractions, len(self.freq)
                        ).reshape((-1, len(self.materials)))*(
               self.materials[:, 0] + self.materials[:, 1] / (
                   1 + 1j * 2 * np.pi * np.repeat(self.freq, len(self.materials)
                   ).reshape((-1, len(self.materials))) * self.materials[:, 2]))**self.a,
               axis=1)**(1 / self.a)


class Rawdata(Relaxation):
    """ Interpolate data given from a text file.

    :param filename: text file which contains three columns:
                     frequency (Hz),Real,Imaginary (separated by comma).
    :type filename: str, non-optional
    :param delimiter: separator for three data columns
    :type delimiter: str, optional (Deafult: ',')
    """
    def __init__(self, filename,
                 sigma, mu, mu_sigma,
                 material_name, number_of_debye_poles=-1,
                 f_n=50, delimiter =',',
                 plot=False, save=True,
                 optimizer=PSO_DLS,
                 optimizer_options={}):

        super(Rawdata, self).__init__(sigma=sigma, mu=mu, mu_sigma=mu_sigma,
                                      material_name=material_name, f_n=f_n,
                                      number_of_debye_poles=number_of_debye_poles,
                                      plot=plot, save=save,
                                      optimizer=optimizer,
                                      optimizer_options=optimizer_options)
        self.delimiter = delimiter
        self.filename = filename
        self.params = {'filename':self.filename}

    def check_inputs(self):
        """ Check the validity of the inputs. """
        super(Rawdata, self).check_inputs()
        if not os.path.isfile(self.filename):
            sys.exit("File doesn't exists!")

    def calculation(self):
        """ Interpolate real and imaginary part from data.
        Column framework of the input file three columns comma-separated
        Frequency(Hz),Real,Imaginary
        """
        # Read the file
        with open(self.filename) as f:
            try:
                array = np.array(
                    [[float(x) for x in line.split(self.delimiter)] for line in f]
                    )
            except ValueError:
                sys.exit("Error: The inputs should be numeric")

        self.set_freq(min(array[:, 0]), max(array[:, 0]), self.f_n)

        rl_interp = scipy.interpolate.interp1d(array[:, 0], array[:, 1])
        im_interp = scipy.interpolate.interp1d(array[:, 0], array[:, 2])
        return rl_interp(self.freq) - 1j * im_interp(self.freq)


if __name__ == "__main__":
    ### Kelley et al. parameters
    setup = HavriliakNegami(f_min=1e7, f_max=1e11,
                            alpha=1-0.09, beta=0.45,
                            e_inf=2.7, de=8.6-2.7, tau_0=9.4e-10,
                            sigma=0, mu=0, mu_sigma=0,
                            material_name="Kelley",
                            number_of_debye_poles=5, f_n=100,
                            plot=True, save=False,
                            optimizer_options={'swarmsize':30,
                                               'maxiter':100,
                                               'omega':0.5,
                                               'phip':1.4,
                                               'phig':1.4,
                                               'minstep':1e-8,
                                               'minfun':1e-8,
                                               'seed': 111,
                                               'pflag': True})
    setup.run()
    setup = HavriliakNegami(f_min=1e7, f_max=1e11,
                            alpha=1-0.09, beta=0.45,
                            e_inf=2.7, de=8.6-2.7, tau_0=9.4e-10,
                            sigma=0, mu=0, mu_sigma=0,
                            material_name="Kelley",
                            number_of_debye_poles=5, f_n=100,
                            plot=True, save=False,
                            optimizer=DA,
                            optimizer_options={'seed': 111})
    setup.run()
    setup = HavriliakNegami(f_min=1e7, f_max=1e11,
                            alpha=1-0.09, beta=0.45,
                            e_inf=2.7, de=8.6-2.7, tau_0=9.4e-10,
                            sigma=0, mu=0, mu_sigma=0,
                            material_name="Kelley",
                            number_of_debye_poles=5, f_n=100,
                            plot=True, save=False,
                            optimizer=DE,
                            optimizer_options={'seed': 111})
    setup.run()
    '''### Testing setup
    setup = Rawdata("Test.txt", 0.1, 1, 0.1, "M1", 3, plot=True,
                    optimizer_options={'seed': 111,
                                       'pflag': True})
    setup.run()
    np.random.seed(111)
    setup = HavriliakNegami(1e12, 1e-3, 0.5, 1, 10, 5,
                            1e-6, 0.1, 1, 0, "M2", 6, plot=True)
    setup.run()
    setup = Jonscher(1e6, 1e-5, 50, 1, 1e5, 0.7,
                     0.1, 1, 0.1, "M3", 4, plot=True)
    setup.run()
    f = np.array([0.5, 0.5])
    material1 = [3, 25, 1e6]
    material2 = [3, 0, 1e3]
    materials = np.array([material1, material2])
    setup = Crim(1*1e-1, 1e-9, 0.5, f, materials, 0.1,
                 1, 0, "M4", 2, plot=True)
    setup.run()'''
