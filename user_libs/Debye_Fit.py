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
#
# -----------------
# Callable objects:
# -----------------
#         Rawdata(number_of_debye_poles, filename, sigma, mu, mu_sigma, material_name, plot, pso)
#         Johnsher(number_of_debye_poles, freq1, freq2, einf, ap, omegap, n_p, sigma, mu, mu_sigma, material_name, \
#                  plot, pso)
# HavriliakNegami(number_of_debye_poles, freq1, freq2, alfa, bita, einf, de, t0, sigma, mu, mu_sigma, material_name, \
#                 plot, pso):
#            Crim(number_of_debye_poles, freq1, freq2, m_par, volumetric_fractions, materials, sigma, mu, mu_sigma, \
#                 material_name, plot, pso)
# -------------------
# Objects parameters:
# -------------------
# number_of_debye_poles = Number of Debye functions used to approximate the given electric permittivity.
#
#              filename = .txt file which contains four columns | frequency (Hz) | Real{e} | frequency (Hz) |  Im{e}  |.
#
#                 sigma = Conductivity.
#
#                    mu = Relative permabillity.
#
#              mu_sigma = Magnetic looses.
#
#         material_name = A string containing the given name of the material (e.g. "Clay").
#
#                  plot = "plot=True" will plot the actual and the approximated permittivity (it can be neglected).
#                          The argument is optional and if neglected plot=False.
#
#                 freq1 = Define the first bound of the frequency range used to approximate the given function (Hz).
#
#                 freq2 = Define the second bound of the frequency range used to approximate the given function (Hz).
#                         freq1 and freq2 can be either freq1 > freq2 or freq1 < freq2 but not freq1 = freq2.
#
#                  einf = The real relative permittivity at infinity frequency for the given Havriliak-Negami or
#                         Jonscher function.
#
#                    ap = Jonscher parameter. Real positive float number.
#
#                omegap = Jonscher parameter. Real positive float number.
#
#                   n_p = Jonscher parameter. Real positive float number which varies 0 < n_p < 1.
#
#                  alfa = Havriliak-Negami parameter. Real positive float number which varies 0 < alfa < 1. For
#                         alfa = 1 and bita !=0 & bita !=1 Havriliak-Negami transforms to Cole-Davidson function.
#
#                  bita = Havriliak-Negami parameter. Real positive float number which varies 0 < bita < 1.
#                         For bita = 1 and alfa !=0 & alfa !=1 Havriliak-Negami transforms to Cole-Cole function.
#
#                    de = Havriliak-Negami parameter. Real positive float number. de is the relative permittivity
#                         at infinite frequency minus the relative permittivity at zero frequency.
#
#                    t0 = Havriliak_Negami parameter. Real positive float number. t0 is the relaxation time.
#
#  volumetric_fractions = Crim parameter. volumetric_fractions is a vector (e.g. [0.1, 0.4, 0.5]) which defines
#                         the volumetric fractions of the given materials. The volumetric fractions should add to one.
#
#             materials = Crim parameter. materials is a matrix which contain the Debye parameters of the media
#                         involved in the CRIM. Each row contains [einf, De, t0] where e_inf is the relative
#                         permittivity at infinity frequency, De is the difference between the relative permittivity
#                         at infinity and zero frequency, t0 is the relaxation time.
#                         Example:
#                         material1 = [2, 25, 10**8]
#                         material2 = [5, 10, 10**7]
#                         material3 = [10, 0, 10]
#                         materials = [material1, material2, material3]
#                         volumetric_fractions = [0.1, 0.4, 0.5]
#
#                m_par = Crim parameter. Real float number.
#
#                  pso = A vector which contains 5 parameters [a1, a2, a3, a4, a5]. a1 denotes the number of particles
#                        to be used in the particle swarm optimisation. a2 denotes the number of iterations. a3 is the
#                        inertia component. a4 and a5 are the cognitive adn social scaling parameters.
#                        The argument is optional and if neglected the default is pso = [40, 50, 0.9, 0.9, 0.9]
# ----------------------------------------------------------------------------------------------------------------------
# Havriliak-Negami function = einf + de / (1 + (1j * 2 * math.pi * f *t0 )**alfa )**bita, where f is the frequency in Hz
#
# Jonscher function = einf - ap*( -1j * 2 * math.pi * f / omegap ) ** n_p ,               where f is the frequency in Hz
#
# CRIM = (sum([volumetric_fraction[i]*(material[i][0] + material[i][1] / (1 + (1j * 2 * math.pi * \
#         f *material[i][2])))**m_param for i in range(0,len(material))]))**1/m_param
# ----------------------------------------------------------------------------------------------------------------------
#
# -----------------
# Modules required:
# -----------------
#  1) numpy
#  2) os
#  3) colorama
#  4) math
#  5) matplotlib
#  6) sys
#  7) scipy
#  8) tqdm
#
#                                             -----------
#                                              Example :
#                                             -----------
#   _________________________________________________________________________________________________
#  | from Debye_Fit import HavriliakNegami, Jonscher, Rawdata, Crim                                  |
#  |                                                                                                 |
#  |                                                                                                 |
#  | Rawdata(3, "/data.txt",0.1, 1, 0.1, "M1", plot=True)                                            |
#  |                                                                                                 |
#  | HavriliakNegami(6, 1*10**12, 10**-3, 0.5, 1, 10, 5, 10**-6, 0.1, 1, 0, "M2", plot=True)         |
#  |                                                                                                 |
#  | Jonscher(4, 10**6, 10**-5, 50, 1, 10**5, 0.7, 0.1, 1, 0.1, "M3", plot=True)                     |
#  |                                                                                                 |
#  | f = [0.5, 0.5]                                                                                  |
#  | material1 = [3, 25, 10**6]                                                                      |
#  | material2 = [3 ,0, 10**3]                                                                       |
#  | materials = [material1, material2]                                                              |
#  | Crim(2, 1*10**-1, 10**-9, 0.5, f, materials, 0.1, 1, 0, "M4", plot=True)                        |
#  |                                                                                                 |
#   -------------------------------------------------------------------------------------------------
import numpy as np
import os
from colorama import Fore
import math
from matplotlib import pylab as plt
import sys
import scipy.interpolate
from tqdm import tqdm


class Crim:
    def __init__(self, number_of_debye_poles, fr1, fr2, a, f1, e1, sigma, mu, mu_sigma, material_name, plot=False,
                 pso=[40, 50, 0.9, 0.9, 0.9]):
        # Check the validity of the inputs
        CheckInputs2(number_of_debye_poles, fr1, fr2, a, f1, e1, sigma, mu, mu_sigma)
        # Place the lower frequency bound at fr1 and the upper frequency bound at fr2
        if fr1 > fr2:
            fr1, fr2 = fr2, fr1
        # Choosing 50 frequencies logarithmicaly equally spaced between the bounds given
        freq = np.logspace(np.log10(fr1), np.log10(fr2), 50)
        # Calculate CRIM (both real and imaginary parts) for the frequencies included in the vector freq
        rl, im = Calc3(a, f1, e1, freq).rl, np.array(Calc3(a, f1, e1, freq).im)

        print(Fore.CYAN)
        print("Approximating Complex Refractive Index Model (CRIM) using {} Debye poles".format(number_of_debye_poles))
        print(Fore.RESET)
        print(Fore.MAGENTA)
        print("CRIM parameters : ")
        print(Fore.RESET)
        for i in range(0, len(f1)):
            print("Material {} :".format(i+1))
            print("---------------------------------")
            print("           Vol. fraction   = {}".format(f1[i]))
            print("                   e_inf   = {}".format(e1[i][0]))
            print("                   De      = {}".format(e1[i][1]))
            print("                   log(t0) = {}".format(np.log10(e1[i][2])))
        # Calling the main optimisation module
        Results(number_of_debye_poles, rl, im, freq, sigma, mu, mu_sigma, material_name, plot, pso)


# Approximate data given from a .txt file
class Rawdata:
    def __init__(self, number_of_debye_poles, filename, sigma, mu, mu_sigma, material_name, plot=False,
                 pso=[40, 50, 0.9, 0.9, 0.9]):
        # Check if file exists
        if os.path.isfile(filename):
            with open(filename) as fn:
                ffn = fn.readlines()
                freq1 = np.zeros(len(ffn))
                freq2 = np.zeros(len(ffn))
                rl1 = np.zeros(len(ffn))
                im1 = np.zeros(len(ffn))
                for index, line in enumerate(ffn):
                    tt = line.split()
                    t = np.zeros(4)
                    t[0:len(tt)] = [tt[i] for i in range(len(tt))]
                    # Column framework of the input file
                    #       Real part         |      Imaginary part
                    # frequency | Real part   | frequency | Imaginary part
                    freq1[index], rl1[index] = float(t[0]), float(t[1]) 
                    freq2[index], im1[index] = float(t[2]), float(t[3])
            # Interpolate using 40 equally logarithmicaly spaced frequencies

            mif, maf = max(min(freq1), min(freq2)), min(max(freq1), max(freq2))

            freq = np.logspace(np.log10(mif)+0.00001, np.log10(maf)-0.00001, 50)
            rl_interp = scipy.interpolate.interp1d(freq1, rl1)
            im_interp = scipy.interpolate.interp1d(freq2, im1)
            rl = rl_interp(freq)
            im = im_interp(freq)

            print(Fore.CYAN)
            print("Approximating the function given from file name: .\{} using {} Debye poles".format(
                os.path.basename(filename), number_of_debye_poles))
            print(Fore.RESET)
            # Calling the main optimisation module
            Results(number_of_debye_poles, rl, -im, freq, sigma, mu, mu_sigma, material_name, plot, pso)

        else:
            ErrorMsg("Error: File doesnt exists ")
            sys.exit(0)


# Approximate a given Johnsher function
class Johnscher:
    def __init__(self, number_of_debye_poles, freq1, freq2, einf, ap, omegap, n_p, sigma, mu, mu_sigma,
                material_name, plot=False, pso = [40, 50, 0.9, 0.9, 0.9]):
        # Check if the inputs are valid (e.g. numeric, positive, 0 < n_p < 1 ...)
        inputs = CheckInputs(
            [number_of_debye_poles, freq1, freq2, n_p, n_p, einf, omegap, ap, sigma, mu, mu_sigma]).D
        # Choosing 40 frequencies logarithmicaly equally spaced between the bounds given
        freq = np.logspace(np.log10(inputs[1]), np.log10(inputs[2]), 50)
        # Calculate the Jonscher function for the frequencies included in the vector freq
        rl = Calc2([inputs[f] for f in range(4, 8)], freq).rl
        im = np.array(Calc2([inputs[f] for f in range(4, 8)], freq).im)

        print(Fore.CYAN)
        print("Approximating Jonsher function using {} Debye poles".format(number_of_debye_poles))
        print(Fore.RESET)
        print(Fore.MAGENTA)
        print("Jonhser function parameters : ")
        print(Fore.RESET)
        print("omega_p =   {} \ne_inf   =   {} \nn_p     =   {} \nA_p     =   {}".format(omegap, einf, n_p, ap))
        # Calling the main optimisation module
        Results(number_of_debye_poles, rl, im, freq, sigma, mu, mu_sigma, material_name, plot, pso)


# Approximate a given Havriliak-Negami function
class HavriliakNegami:
    def __init__(self, number_of_debye_poles, freq1, freq2, alfa, bita, einf, de, t0, sigma, mu, mu_sigma,
                 material_name, plot=False, pso=[40, 50, 0.9, 0.9, 0.9]):
        # Check if the inputs are valid (e.g. numeric, positive, 0 < alfa, bita < 1 ...)
        inputs = CheckInputs([number_of_debye_poles, freq1, freq2, alfa, bita, einf, de, t0, sigma, mu, mu_sigma]).D
        # Choosing 40 frequencies logarithmicaly equally spaced between the bounds given
        freq = np.logspace(np.log10(inputs[1]), np.log10(inputs[2]), 50)
        # Calculate the Havriliak-Negami function (both real and imaginary parts) for the frequencies included in the
        # vector freq
        rl, im = Calc([inputs[f] for f in range(3, 8)], freq).rl,\
            np.array(Calc([inputs[f] for f in range(3, 8)], freq).im)

        print(Fore.CYAN)
        print("Approximating Havriliak-Negami function using {} Debye poles".format(number_of_debye_poles))
        print(Fore.RESET)
        print(Fore.MAGENTA)
        print("Havriliak-Negami parameters : ")
        print(Fore.RESET)
        print("De     =   {} \ne_inf  =   {} \nt0     =   {} \nalfa   =   {} \nbita   =   {}  "
              .format(de, einf, t0, alfa, bita))
        # Calling the main optimisation module
        Results(number_of_debye_poles, rl, im, freq, sigma, mu, mu_sigma, material_name, plot, pso)


class Results:
    def __init__(self, number_of_debye_poles, rl, im, freq, sigma, mu, mu_sigma, material_name, plot, pso):

        # The variables below are declared as global due to the fact that they will be used at the Class.cost function
        global rl_g, im_g, freq_g
        rl_g = rl
        im_g = im
        freq_g = freq

        # Define the lower and upper boundaries of search
        lb = np.full(number_of_debye_poles, -np.log10(np.max(freq_g)) - 3)
        ub = np.full(number_of_debye_poles, -np.log10(np.min(freq_g)) + 3)
        # Call particle swarm optimisation to minimize the Cost_function.
        xmp, ob = Pso.pso(Cost.function, lb, ub, plot, pso)  # xmp : The resulting optimised relaxation times of the
        # Debye poles
        # ob : The final error
        # For the given relaxation times calculate the optimised weights
        # mx  : Resulting optimised weights for the given relaxation times
        # cost: The final error
        # ee  : Average error between the actual and the approximated real part
        # rp  : The real part of the permittivity for the optimised relaxation times and weights for the frequnecies
        # included in freq
        # ip  : The imaginary part of the permittivity for the optimised relaxation times and weights for the
        # frequnecies included in freq
        mx = Linear(rl, im, xmp, freq).x
        # cost = Linear(rl, im, xmp, freq).cost
        ee = Linear(rl, im, xmp, freq).ee
        rp = Linear(rl, im, xmp, freq).rp
        ip = Linear(rl, im, xmp, freq).ip

        # if one of the weights is negative increase the stabiliser and repeat the optimisation

        # Print the results in gprMax format style
        OutCom(xmp, mx, ee, sigma, mu, mu_sigma, material_name)
        # Plot the actual and the approximate dielectric properties
        if plot:
            Plot(freq, rl, im, rp + ee, ip)


class Cost:
    def function(self):
        # The cost function is the average error between the actual and the approximated electric permittivity.
        cost = Linear(rl_g, im_g, self, freq_g).cost + Linear(rl_g, im_g, self, freq_g).cost2
        return cost


class OutCom:
    def __init__(self, xmp, mx, ee, sigma, mu, mu_sigma, material_name):
        # Print out the resulting Debye parameters
        print(Fore.MAGENTA)
        print("Debye expansion parameters : ")
        print(Fore.RESET)
        print(Fore.GREEN)
        print('         |     e_inf     |       De      |         log(t0)        | ')
        print('__________________________________________________________________')
        print(Fore.RESET)
        for i in range(0, len(xmp)):

            print(Fore.GREEN, 'Debye {0:}:|'
                  .format(i + 1), Fore.RESET, '  {0:s}    |    {1:s}    |         {2:s}        | '
                  .format(str(ee/len(xmp))[0:7], str(mx[i])[0:7], str(xmp[i])[0:7]))
            print('__________________________________________________________________\n')
        print("\n")

        # Print the Debye expnasion in a gprMax format
        print('#material: {} {} {} {} {}'.format(ee, sigma, mu, mu_sigma, material_name))
        out_t = '#add_dispersion_debye: {} {} {}'.format(len(xmp), mx[0], 10**xmp[0])
        for i in range(1, len(xmp)):
            out_t += ' {} {}'.format(mx[i], 10**xmp[i])
        out_t += ' {}'.format(material_name)
        print(out_t)


class Calc:
    def __init__(self, cal_inputs, freq):
        # Calculates the Havriliak-Negami function for the given cal_inputs
        q = [cal_inputs[2] + cal_inputs[3] / (np.array(1 + np.array(1j * 2 * math.pi *
                                              f * cal_inputs[4]) ** cal_inputs[0]) ** cal_inputs[1]) for f in freq]
        # Return the real and the imaginary part of the Havriliak-Negami function
        if len(q) > 1:
            self.rl = [q[i].real for i in range(0, len(q))]
            self.im = [q[i].imag for i in range(0, len(q))]
        else:
            self.rl = q[0].real
            self.im = q[0].imag


class Calc2:
    def __init__(self, cal_inputs, freq):
        # Calculates the Q function for the given cal_inputs
        q = [cal_inputs[1] + (cal_inputs[3]*np.array(2*math.pi*f/cal_inputs[2])**(cal_inputs[0]-1))
             * (1 - 1j/math.tan(cal_inputs[0]*math.pi/2)) for f in freq]
        # Return the real and the imaginary part of the Jonscher function
        if len(q) > 1:
            self.rl = [q[i].real for i in range(0, len(q))]
            self.im = [q[i].imag for i in range(0, len(q))]
        else:
            self.rl = q[0].real
            self.im = q[0].imag


class Calc3:
    def __init__(self, a, f1, e1, freq):
        # Calculates the Crim function for the given cal_inputs
        q = np.zeros(len(freq))
        for i in range(0, len(f1)):
            q = q + f1[i]*np.array(
                [e1[i][0] + e1[i][1] / (np.array(1 + np.array(1j * 2 * math.pi * f * e1[i][2]))) for f in freq])**a
        q = q**(1/a)
        # Return the real and the imaginary part of the Havriliak-Negami function
        if len(q) > 1:
            self.rl = [q[i].real for i in range(0, len(q))]
            self.im = [q[i].imag for i in range(0, len(q))]
        else:
            self.rl = q[0].real
            self.im = q[0].imag


class Plot:
    def __init__(self, freq, rl, im, rl1, im1):
        # Plot the actual and the approximated electric permittivity using a semilogarithm X axes
        plt.close("all")
        plt.rcParams['axes.facecolor'] = 'black'
        plt.semilogx(freq / 10 ** 6, rl1, "b-", linewidth=2.0, label="Debye Expansion: Real")
        plt.semilogx(freq / 10 ** 6, -im1, "w-", linewidth=2.0, label="Debye Expansion: Imaginary")
        plt.semilogx(freq / 10 ** 6, rl, "ro", linewidth=2.0, label="Chosen Function: Real")
        plt.semilogx(freq / 10 ** 6, -im, "go", linewidth=2.0, label="Chosen Function: Imaginary")

        plt.rcParams['axes.facecolor'] = 'white'
        plt.grid(b=True, which='major', color='w', linewidth=0.2, linestyle='--')
        axes = plt.gca()
        axes.set_xlim([np.min(freq / (10 ** 6)), np.max(freq / (10 ** 6))])
        axes.set_ylim([-1, np.max([np.max(rl), np.max(-im)]) + 1])
        plt.legend()
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Relative permittivity")
        plt.show()


class CheckInputs:
    def __init__(self, d):
        # Check the inputs validity
        try:
            d = [float(i) for i in d]
        except:
            ErrorMsg("Error: The inputs should be numeric")
            sys.exit(0)
        f = [i for i in d if i < 0]
        if len(f) != 0:
            ErrorMsg("Error: The inputs should be positive")
            sys.exit(0)
        if d[3] > 1:
            ErrorMsg("Error: Alfa value must range between 0-1 (0 < Alfa <1)")
            sys.exit(0)
        if d[4] > 1:
            ErrorMsg("Error: Beta value must range between 0-1 (0 < Beta <1)")
            sys.exit(0)
        if d[0] != int(d[0]):
            ErrorMsg("Error: The number of Debye poles must be integer")
            sys.exit(0)
        if d[1] == d[2]:
            ErrorMsg("Error: Null frequency range")
            sys.exit(0)
        if d[1] > d[2]:
            d[1], d[2] = d[2], d[1]

        self.D = d


class CheckInputs2:
    def __init__(self, number_of_debye_poles, fr1, fr2, a, f1, e1, sigma, mu, mu_sigma):
        if len(f1) != len(e1):
            ErrorMsg("Error: Volumetric volumes does not match the dielectric properties")
            sys.exit(0)
        # Check if the materials are at least two
        if len(f1) < 2:
            ErrorMsg("Error: The materials should be at least 2")
            sys.exit(0)
        # Check if the frequency range is null
        if fr1 == fr2:
            ErrorMsg("Error: Null frequency range")
            sys.exit(0)
        # Check if the inputs are positive
        f = [i for i in [number_of_debye_poles, fr1, fr2, a, sigma, mu, mu_sigma] if i < 0]
        if len(f) != 0:
            ErrorMsg("Error: The inputs should be positive")
            sys.exit(0)
        f = [i for i in f1 if i < 0]
        if len(f) != 0:
            ErrorMsg("Error: The inputs should be positive")
            sys.exit(0)
        for i in range(0, len(f1)):
            f = [i for i in e1[i][:] if i < 0]
            if len(f) != 0:
                ErrorMsg("Error: The inputs should be positive")
                sys.exit(0)

        # Check if the number_of_debye_poles is an integer
        if number_of_debye_poles != int(number_of_debye_poles):
            ErrorMsg("Error: The number of Debye poles must be integer")
            sys.exit(0)
        # Check if the summation of the volumetric fractions equal to one
        if np.sum(f1) != 1:
            ErrorMsg("Error: The summation of volumetric volumes should be equal to 1")
            sys.exit(0)


class CheckInputs3:
    def __init__(self, number_of_debye_poles, fr1, fr2, rb, rs, fw, s, c):
        # Check if the inputs are numeric
        try:
            d = [float(i) for i in [number_of_debye_poles, fr1, fr2, rb, rs, fw, s, c]]
        except:
            ErrorMsg("Error: The inputs should be numeric")
            sys.exit(0)

        if fw > 1 or s > 1 or c > 1:
            ErrorMsg("Error: The volumetric fractions should be less than one")
            sys.exit(0)

        # Check if the frequency range is null
        if fr1 == fr2:
            ErrorMsg("Error: Null frequency range")
            sys.exit(0)
        # Check if the inputs are positive
        f = [i for i in [number_of_debye_poles, fr1, fr2, rb, rs, fw, s, c] if i < 0]
        if len(f) != 0:
            ErrorMsg("Error: The inputs should be positive")
            sys.exit(0)

        # Check if the number_of_debye_poles is an integer
        if number_of_debye_poles != int(number_of_debye_poles):
            ErrorMsg("Error: The number of Debye poles must be integer")
            sys.exit(0)
        # Check if the summation of the volumetric fractions equal to one
        if s + c != 1:
            ErrorMsg("Error: The summation of sand and clay fraction should be equal to 1")
            sys.exit(0)


class ErrorMsg(ValueError):
    def __init__(self, text):
        # Print error texts
        print(Fore.RED)
        print(text)
        print(Fore.RESET)


class Linear:
    def __init__(self, rl, im, logt, freq):
        # The relaxation time of the Debyes are given at as logarithms logt=log10(t0) for efficiency during the
        # optimisation
        # Here they are transformed back t0=10**logt
        tt = [10**logt[i] for i in range(0, len(logt))]
        # y = Ax , here the A matrix for the real and the imaginary part is builded
        d_r = np.array(
            [[Calc([1, 1, 0, 1, tt[i]], [freq[j]]).rl for i in range(0, len(tt))] for j in
             range(0, len(freq))])
        d = np.array(
            [[Calc([1, 1, 0, 1, tt[i]], [freq[j]]).im for i in range(0, len(tt))] for j in
             range(0, len(freq))])
        # Adding dumping (Marquart least squares)
        # Solving the overdetermined system y=Ax
        self.x = np.abs(np.linalg.lstsq(d, im)[0])
        mx, my, my2 = np.matrix(self.x), np.matrix(d), np.matrix(d_r)
        self.rp, self.ip = my2 * np.transpose(mx), my * np.transpose(mx)
        self.cost = np.sum([np.abs(self.ip[i]-im[i]) for i in range(0, len(im))])/len(im)
        self.ee = (np.mean(rl - self.rp))
        if self.ee < 1:
            self.ee = 1
        self.cost2 = np.sum([np.abs(self.rp[i]-rl[i]+self.ee) for i in range(0, len(im))])/len(im)


class Pso:
    def pso(func, lb, ub, plot, pso):
        print('\n')
        # A particle swarm optimisation that tries to find an optimal set of relaxation times that minimise the error
        # between the actual adn the approximated electric permittivity.
        # The current class is a modified edition of the pyswarm package which can be found at
        # https://pythonhosted.org/pyswarm/

        # Predefine the parameters of the particle swarm optimisation
        omega, phip, phig = pso[2], pso[3], pso[4]
        s = pso[1]

        obj = lambda x: func(x)
        plt.close("all")
        lb = np.array(lb)
        ub = np.array(ub)
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
        # Initialize the particle swarm ############################################
        d = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(s, d)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fp = np.zeros(s)  # best particle function values
        g = []  # best swarm position
        fg = 1e100  # artificial best swarm position starting value

        for i in range(s):
            # Initialize the particle's position
            x[i, :] = lb + x[i, :] * (ub - lb)

            # Initialize the particle's best known position
            p[i, :] = x[i, :]

            # Calculate the objective's value at the current particle's
            fp[i] = obj(p[i, :])

            # At the start, there may not be any feasible starting point, so just
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

        # Iterate until termination criterion met ##################################
        for it in tqdm(range(2, pso[0]+2), desc='Debye fitting'):
            rp = np.random.uniform(size=(s, d))
            rg = np.random.uniform(size=(s, d))
            for i in range(s):

                # Update the particle's velocity
                v[i, :] = omega * v[i, :] + phip * rp[i, :] * (p[i, :] - x[i, :]) + \
                          phig * rg[i, :] * (g - x[i, :])

                # Update the particle's position, correcting lower and upper bound
                # violations, then update the objective function value
                x[i, :] = x[i, :] + v[i, :]
                mark1 = x[i, :] < lb
                mark2 = x[i, :] > ub
                x[i, mark1] = lb[mark1]
                x[i, mark2] = ub[mark2]
                fx = obj(x[i, :])

                # Compare particle's best position (if constraints are satisfied)
                if fx < fp[i]:
                    p[i, :] = x[i, :].copy()
                    fp[i] = fx

                    # Compare swarm's best position to current particle's position
                    # (Can only get here if constraints are satisfied)
                    if fx < fg:
                        tmp = x[i, :].copy()
                        stepsize = np.sqrt(np.sum((g - tmp) ** 2))
                        if stepsize <= 1e-8:
                            return tmp, fx
                        else:
                            g = tmp.copy()
                            fg = fx

            # Dynamically plot the error as the optimisation takes place
            if plot:
                if it == 2:
                    xpp = [0]
                    ypp = [fg]
                    plt.rcParams['axes.facecolor'] = 'black'

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    line1, = ax.plot(xpp, ypp, 'b-', linewidth=3.0)
                else:
                    xpp.append(it - 1)
                    ypp.append(fg)

                    line1.set_ydata(ypp)
                    line1.set_xdata(xpp)
                    plt.ylim(min(ypp) - 0.1 * min(ypp), max(ypp) + 0.1 * max(ypp))
                    plt.xlim(min(xpp), max(xpp))
                    plt.grid(b=True, which='major', color='w', linewidth=0.2, linestyle='--')
                    plt.suptitle('Debye fitting process')
                    fig.canvas.draw()

                    plt.xlabel("Iteration")
                    plt.ylabel("Average Error")
                    plt.pause(0.0001)

        return g, fg
