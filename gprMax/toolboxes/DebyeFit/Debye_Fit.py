# Copyright (C) 2015-2024, Iraklis Giannakis and Sylwia Majchrowska
#
# This module is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/.
#
# Please use the attribution at http://dx.doi.org/10.1109/TAP.2014.2308549

import os
import warnings
from numbers import Integral
from pathlib import Path

import matplotlib.gridspec as gridspec
import numpy as np
import scipy.interpolate
from matplotlib import pylab as plt

try:
    from .optimization import DA_DLS, DE_DLS, PSO_DLS
except ImportError:  # Support running the legacy examples directly.
    from optimization import DA_DLS, DE_DLS, PSO_DLS


def _scalar(value, name, lower=0, *, strict=False):
    """Validate a real, finite scalar before any numerical evaluation."""
    try:
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            raise ValueError
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(value) or (value <= lower if strict else value < lower):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {relation} {lower}")
    return value


def _frequency_count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
        raise ValueError("f_n must be an integer of at least 2")
    return int(value)


class Relaxation(object):
    """Create Relaxation function object for complex material.

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
    :param save: if True will save approximated material parameters.
    :type save: bool, optional, default:False
    :param optimizer: chosen optimization method:
                      Hybrid Particle Swarm-Damped Least-Squares (PSO_DLS),
                      Dual Annealing (DA) or Differential Evolution (DE)
                      (Default: PSO_DLS).
    :type optimizer: Optimizer class, optional
    :param optimizer_options: Additional keyword arguments passed to
                                     optimizer class (Default: empty dict).
    :type optimizer_options: dict, optional, default: empty dict
    """

    def __init__(
        self,
        sigma,
        mu,
        mu_sigma,
        material_name,
        f_n=50,
        number_of_debye_poles=-1,
        plot=True,
        save=False,
        optimizer=PSO_DLS,
        optimizer_options=None,
    ):
        self.name = "Relaxation function"
        self.params = {}
        self.number_of_debye_poles = number_of_debye_poles
        self.f_n = f_n
        self.sigma = sigma
        self.mu = mu
        self.mu_sigma = mu_sigma
        self.material_name = material_name
        self.plot = plot
        self.save = save
        if optimizer_options is None:
            optimizer_options = {}
        self.optimizer = optimizer(**optimizer_options)

    def set_freq(self, f_min, f_max, f_n=50):
        """Interpolate frequency vector using n equally logarithmicaly spaced frequencies.

        Args:
            f_min (float): First bound of the frequency range
                           used to approximate the given function (Hz).
            f_max (float): Second bound of the frequency range
                           used to approximate the given function (Hz).
            f_n (int): Number of frequency points in frequency grid
                       (Default: 50).
        Note:
            Bounds must be positive and distinct; reversed bounds are accepted.
        """
        f_min = _scalar(f_min, "f_min", strict=True)
        f_max = _scalar(f_max, "f_max", strict=True)
        f_n = _frequency_count(f_n)
        if f_min == f_max:
            raise ValueError("Frequency bounds must be distinct")
        self.f_min, self.f_max = sorted((f_min, f_max))
        if self.f_max - self.f_min > 1e12:
            warnings.warn(
                f"The chosen frequency range is very wide. "
                f"Consider setting a greater number of points "
                f"on the frequency grid!"
            )
        self.freq = np.logspace(np.log10(self.f_min), np.log10(self.f_max), f_n)
        # Preserve exact endpoints, avoiding interpolation outside the data interval.
        self.freq[0], self.freq[-1] = self.f_min, self.f_max

    def check_inputs(self):
        """Check the validity of the inputs."""
        poles = self.number_of_debye_poles
        if (
            isinstance(poles, (bool, np.bool_))
            or not isinstance(poles, Integral)
            or (poles != -1 and poles < 1)
        ):
            raise ValueError("number_of_debye_poles must be -1 (automatic) or a positive integer")
        self.f_n = _frequency_count(self.f_n)
        self.sigma = _scalar(self.sigma, "sigma")
        self.mu = _scalar(self.mu, "mu", lower=1)
        self.mu_sigma = _scalar(self.mu_sigma, "mu_sigma")
        self._check_material_name()

    def _check_material_name(self):
        from gprMax.impedance_surfaces import is_reserved_impedance_id
        from gprMax.materials import validate_user_material_id

        name = self.material_name
        if not isinstance(name, str) or not name or any(c.isspace() for c in name):
            raise ValueError("material_name must be a non-empty string without whitespace")
        validate_user_material_id(name)
        if name in ("pec", "pmc", "free_space") or is_reserved_impedance_id(name):
            raise ValueError(f"material_name {name!r} is reserved by gprMax")

    def calculation(self):
        """Approximate the given relaxation function
        (Havriliak-Negami function, Crim, Jonscher) or based on raw data.
        """
        raise NotImplementedError()

    def print_info(self):
        """Readable string of parameters for given approximation settings.

        Returns:
            s (str): Info about chosen function and its parameters.
        """
        print(f"Approximating {self.name} using {self.number_of_debye_poles} Debye poles")
        print(f"{self.name} parameters: ")
        s = "".join(f"{k:10s} = {v}\n" for k, v in self.params.items())
        print(s)
        return f"{self.name}:\n{s}"

    def optimize(self):
        """Calling the main optimisation module with defined lower and upper boundaries of search.

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
        lb = np.full(self.number_of_debye_poles, -np.log10(np.max(self.freq)) - 3)
        ub = np.full(self.number_of_debye_poles, -np.log10(np.min(self.freq)) + 3)
        # Call optimizer to minimize the cost function
        tau, weights, ee, rl, im = self.optimizer.fit(
            func=self.optimizer.cost_function,
            lb=lb,
            ub=ub,
            funckwargs={"rl": self.rl, "im": self.im, "freq": self.freq},
        )
        return tau, weights, ee, rl, im

    def run(self):
        """Solve the problem described by the given relaxation function
        (Havriliak-Negami function, Crim, Jonscher)
        or data given from a text file.

        Returns:
            avg_err (float): Sum of the mean absolute normalised real and loss
                             errors, expressed as a percentage.
            properties (list(str)): Material and Debye expansion hash commands.
        """
        # Check the validity of the inputs
        self.check_inputs()
        # Print information about chosen approximation settings
        self.print_info()
        # Calculate both real and imaginary parts
        # for the frequencies included in the vector freq
        q = np.asarray(self.calculation())
        if (
            q.shape != self.freq.shape
            or not np.all(np.isfinite(q))
            or np.any(q.real < 1)
            or np.any(q.imag > 0)
        ):
            raise ValueError("The target must be finite with real permittivity >= 1 and loss >= 0")
        # Set the real and the imaginary part of the relaxation function
        self.rl, self.im = q.real, q.imag

        if self.number_of_debye_poles == -1:
            print(
                "\n#########",
                "Try to automaticaly fit number of Debye poles, up to 20!",
                "##########\n",
                sep="",
            )
            self.number_of_debye_poles = 1
            iteration = 1
            # stop increasing number of Debye poles if error is smaller then 5%
            # or 20 debye poles is reached
            while True:
                # Calling the main optimisation module
                tau, weights, ee, rl, im = self.optimize()
                err_real, err_imag = self.error(rl + ee, im)
                error = err_real + err_imag
                if error <= 5 or iteration >= 20:
                    if error > 5:
                        warnings.warn(
                            f"Automatic fitting reached 20 poles with {error:g}% combined "
                            "mean error, above the 5% target. Inspect the residuals.",
                            UserWarning,
                            stacklevel=2,
                        )
                    break
                self.number_of_debye_poles += 1
                iteration += 1
        else:
            # Fit with the requested number of trial poles.
            tau, weights, ee, rl, im = self.optimize()
            err_real, err_imag = self.error(rl + ee, im)

        # Print the results in gprMax format style
        properties = self.print_output(tau, weights, ee)
        print(
            f"The average fractional error for:\n"
            f"- real part: {err_real}\n"
            f"- imaginary part: {err_imag}\n"
        )
        if self.save:
            self.save_result(properties)
        # Plot the actual and the approximate dielectric properties
        if self.plot:
            self.plot_result(rl + ee, im)
        return err_real + err_imag, properties

    def print_output(self, tau, weights, ee):
        """Print out the resulting Debye parameters in a gprMax format.

        Args:
            tau (ndarray): The best known position form optimization module
                           (optimal design).
            weights (ndarray): Resulting optimised weights for the given relaxation times.
            ee (float): Fitted relative permittivity at infinite frequency
                        (e_inf), written into the returned #material command.

        Returns:
            material_prop (list(str)): Given material nad Debye expnasion parameters
                                       in a gprMax format.
        """
        # Export only poles the solver can accept; do not discard small *positive*
        # strengths or silently turn a negative/non-finite fit into a material.
        self._check_material_name()
        self.sigma = _scalar(self.sigma, "sigma")
        self.mu = _scalar(self.mu, "mu", lower=1)
        self.mu_sigma = _scalar(self.mu_sigma, "mu_sigma")
        ee = _scalar(ee, "fitted e_inf", lower=1)
        if np.iscomplexobj(tau) or np.iscomplexobj(weights):
            raise ValueError("Fitted poles and weights must be real")
        tau, weights = np.asarray(tau, dtype=float), np.asarray(weights, dtype=float)
        if (
            tau.ndim != 1
            or weights.shape != tau.shape
            or not np.all(np.isfinite(tau))
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
        ):
            raise ValueError(
                "Fitted poles must be matching finite vectors with non-negative weights"
            )
        keep = weights > 0
        tau, weights = tau[keep], weights[keep]
        with np.errstate(over="ignore", under="ignore"):
            times = 10.0**tau
        if not np.all(np.isfinite(times)) or np.any(times <= 0):
            raise ValueError("Fitted relaxation times must be finite and positive")
        self.fitted_pole_count = len(tau)
        print("Debye expansion parameters: ")
        print(f"       |{'e_inf':^14s}|{'De':^14s}|{'log(tau_0)':^25s}|")
        print("_" * 65)
        for i in range(0, len(tau)):
            print(f"Debye {i + 1}|{ee:^14.5f}|{weights[i]:^14.5f}|{tau[i]:^25.5f}|")
            print("_" * 65)

        # Print the Debye expnasion in a gprMax format
        material_prop = []
        material_prop.append(
            f"#material: {ee} {self.sigma} {self.mu} {self.mu_sigma} {self.material_name}\n"
        )
        print(material_prop[0], end="")
        if not len(tau):
            return material_prop
        dispersion_prop = f"#add_dispersion_debye: {len(tau)}"
        for i in range(len(tau)):
            dispersion_prop += f" {weights[i]} {times[i]}"
        dispersion_prop += f" {self.material_name}"
        print(dispersion_prop)
        material_prop.append(dispersion_prop + "\n")
        return material_prop

    def plot_result(self, rl_exp, im_exp):
        """Plot the actual and the approximated electric permittivity,
        along with relative error for real and imaginary parts
        using a semilogarithm X axes.

        Args:
            rl_exp (ndarray): Real parts of optimised Debye expansion
                              for given frequency points (plus average error).
            im_exp (ndarray): Imaginary parts of optimised Debye expansion
                              for given frequency points.
        """
        plt.close("all")
        fig = plt.figure(figsize=(16, 8), tight_layout=True)
        gs = gridspec.GridSpec(2, 1)
        ax = fig.add_subplot(gs[0])
        ax.grid(visible=True, which="major", linewidth=0.2, linestyle="--")
        ax.semilogx(
            self.freq * 1e-6,
            rl_exp,
            "b-",
            linewidth=2.0,
            label="Debye Expansion: Real part",
        )
        ax.semilogx(
            self.freq * 1e-6,
            -im_exp,
            "k-",
            linewidth=2.0,
            label="Debye Expansion: Imaginary part",
        )
        ax.semilogx(
            self.freq * 1e-6,
            self.rl,
            "r.",
            linewidth=2.0,
            label=f"{self.name}: Real part",
        )
        ax.semilogx(
            self.freq * 1e-6,
            -self.im,
            "g.",
            linewidth=2.0,
            label=f"{self.name}: Imaginary part",
        )
        # Keep poor fits visible too; target-only limits can hide overshoot.
        ax.set_ylim([-1, np.max(np.concatenate([self.rl, -self.im, rl_exp, -im_exp])) + 1])
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Relative permittivity")

        ax = fig.add_subplot(gs[1])
        ax.grid(visible=True, which="major", linewidth=0.2, linestyle="--")
        ax.semilogx(
            self.freq * 1e-6,
            (rl_exp - self.rl) / (self.rl + 1),
            "b-",
            linewidth=2.0,
            label="Real part",
        )
        ax.semilogx(
            self.freq * 1e-6,
            (-im_exp + self.im) / (np.abs(self.im) + 1),
            "k-",
            linewidth=2.0,
            label="Imaginary part",
        )
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Relative approximation error")
        plt.show()

    def error(self, rl_exp, im_exp):
        """Calculate the average fractional error separately for
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
        avg_err_real = np.sum(np.abs((rl_exp - self.rl) / (self.rl + 1)) * 100) / len(rl_exp)
        im_rel_err = np.abs((-im_exp + self.im) / (np.abs(self.im) + 1))
        avg_err_imag = np.sum(im_rel_err * 100) / len(im_exp)
        return avg_err_real, avg_err_imag

    @staticmethod
    def save_result(output, fdir=None):
        """Save the resulting Debye parameters in a gprMax format.

        Args:
            output (list(str)): Material and resulting Debye parameters
                                in a gprMax format.
            fdir (str or Path, optional): Existing output directory. If omitted,
                search the legacy relative material directories.
        """
        if fdir is not None:
            if not os.path.isdir(fdir):
                raise FileNotFoundError(f"Material output directory does not exist: {fdir}")
            file_path = os.path.join(fdir, "my_materials.txt")
        elif os.path.isdir("../materials"):
            file_path = os.path.join("../materials", "my_materials.txt")
        elif os.path.isdir("materials"):
            file_path = os.path.join("materials", "my_materials.txt")
        elif os.path.isdir("user_libs/materials"):
            file_path = os.path.join("user_libs", "materials", "my_materials.txt")
        else:
            raise FileNotFoundError(
                "No material output directory found. Create one and pass it as fdir."
            )
        with open(file_path, "a") as fileH:
            fileH.write(f"## {output[0].split(' ')[-1]}")
            fileH.writelines(output)
            fileH.write("\n")
        print(f"Material properties save at: {file_path}")


class HavriliakNegami(Relaxation):
    """Approximate a given Havriliak-Negami function
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
    :param alpha: Real positive float number which varies 0 < alpha <= 1.
                 For alpha = 1 and beta !=0 & beta !=1 Havriliak-Negami
                 transforms to Cole-Davidson function.
    :type alpha: float
    :param beta: Real positive float number which varies 0 < beta <= 1.
                 For beta = 1 and alpha !=0 & alpha !=1 Havriliak-Negami
                 transforms to Cole-Cole function.
    :type beta: float
    :param de: The difference of relative permittivity at infinite frequency
               and the relative permittivity at zero frequency.
    :type de: float
    :param tau_0: Real positive float number, tau_0 is the relaxation time.
    :type tau_0: float
    """

    def __init__(
        self,
        f_min,
        f_max,
        alpha,
        beta,
        e_inf,
        de,
        tau_0,
        sigma,
        mu,
        mu_sigma,
        material_name,
        number_of_debye_poles=-1,
        f_n=50,
        plot=False,
        save=False,
        optimizer=PSO_DLS,
        optimizer_options=None,
    ):
        super(HavriliakNegami, self).__init__(
            sigma=sigma,
            mu=mu,
            mu_sigma=mu_sigma,
            material_name=material_name,
            f_n=f_n,
            number_of_debye_poles=number_of_debye_poles,
            plot=plot,
            save=save,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
        )
        self.name = "Havriliak-Negami function"
        self.set_freq(f_min, f_max, self.f_n)
        self.e_inf, self.alpha, self.beta, self.de, self.tau_0 = (
            e_inf,
            alpha,
            beta,
            de,
            tau_0,
        )
        self.params = {
            "f_min": self.f_min,
            "f_max": self.f_max,
            "eps_inf": self.e_inf,
            "Delta_eps": self.de,
            "tau_0": self.tau_0,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def check_inputs(self):
        """Check the validity of the Havriliak Negami model's inputs."""
        super(HavriliakNegami, self).check_inputs()
        self.set_freq(self.f_min, self.f_max, self.f_n)
        self.e_inf = _scalar(self.e_inf, "e_inf", lower=1)
        self.de = _scalar(self.de, "de")
        self.tau_0 = _scalar(self.tau_0, "tau_0", strict=True)
        self.alpha = _scalar(self.alpha, "alpha", strict=True)
        self.beta = _scalar(self.beta, "beta", strict=True)
        if self.alpha > 1 or self.beta > 1:
            raise ValueError("alpha and beta must satisfy 0 < value <= 1")
        self.params = dict(
            f_min=self.f_min,
            f_max=self.f_max,
            eps_inf=self.e_inf,
            Delta_eps=self.de,
            tau_0=self.tau_0,
            alpha=self.alpha,
            beta=self.beta,
        )

    def calculation(self):
        """Calculates the Havriliak-Negami function for
        the given parameters."""
        return (
            self.e_inf
            + self.de / (1 + (1j * 2 * np.pi * self.freq * self.tau_0) ** self.alpha) ** self.beta
        )


class Jonscher(Relaxation):
    """Approximate a given Jonsher function
    Jonscher function = ε_∞ + ap * (2πf / omegap)**(n_p - 1) * (1 - j*cot(n_p*π/2)),
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

    def __init__(
        self,
        f_min,
        f_max,
        e_inf,
        a_p,
        omega_p,
        n_p,
        sigma,
        mu,
        mu_sigma,
        material_name,
        number_of_debye_poles=-1,
        f_n=50,
        plot=False,
        save=False,
        optimizer=PSO_DLS,
        optimizer_options=None,
    ):
        super(Jonscher, self).__init__(
            sigma=sigma,
            mu=mu,
            mu_sigma=mu_sigma,
            material_name=material_name,
            f_n=f_n,
            number_of_debye_poles=number_of_debye_poles,
            plot=plot,
            save=save,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
        )
        self.name = "Jonsher function"
        self.set_freq(f_min, f_max, self.f_n)
        self.e_inf, self.a_p, self.omega_p, self.n_p = e_inf, a_p, omega_p, n_p
        self.params = {
            "f_min": self.f_min,
            "f_max": self.f_max,
            "eps_inf": self.e_inf,
            "n_p": self.n_p,
            "omega_p": self.omega_p,
            "a_p": self.a_p,
        }

    def check_inputs(self):
        """Check the validity of the inputs."""
        super(Jonscher, self).check_inputs()
        self.set_freq(self.f_min, self.f_max, self.f_n)
        self.e_inf = _scalar(self.e_inf, "e_inf", lower=1)
        self.a_p = _scalar(self.a_p, "a_p")
        self.omega_p = _scalar(self.omega_p, "omega_p", strict=True)
        self.n_p = _scalar(self.n_p, "n_p", strict=True)
        if self.n_p >= 1:
            raise ValueError("n_p must satisfy 0 < n_p < 1")
        self.params = dict(
            f_min=self.f_min,
            f_max=self.f_max,
            eps_inf=self.e_inf,
            n_p=self.n_p,
            omega_p=self.omega_p,
            a_p=self.a_p,
        )

    def calculation(self):
        """Calculates the Q function for the given parameters"""
        return self.e_inf + (
            self.a_p * (2 * np.pi * self.freq / self.omega_p) ** (self.n_p - 1)
        ) * (1 - 1j / np.tan(self.n_p * np.pi / 2))


class Crim(Relaxation):
    """Approximate a given CRIM function
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

    def __init__(
        self,
        f_min,
        f_max,
        a,
        volumetric_fractions,
        materials,
        sigma,
        mu,
        mu_sigma,
        material_name,
        number_of_debye_poles=-1,
        f_n=50,
        plot=False,
        save=False,
        optimizer=PSO_DLS,
        optimizer_options=None,
    ):
        super(Crim, self).__init__(
            sigma=sigma,
            mu=mu,
            mu_sigma=mu_sigma,
            material_name=material_name,
            f_n=f_n,
            number_of_debye_poles=number_of_debye_poles,
            plot=plot,
            save=save,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
        )
        self.name = "CRIM function"
        self.set_freq(f_min, f_max, self.f_n)
        self.a = a
        self.volumetric_fractions = np.array(volumetric_fractions)
        self.materials = np.array(materials)
        self.params = {
            "f_min": self.f_min,
            "f_max": self.f_max,
            "a": self.a,
            "volumetric_fractions": self.volumetric_fractions,
            "materials": self.materials,
        }

    def check_inputs(self):
        """Check the validity of the inputs."""
        super(Crim, self).check_inputs()
        self.set_freq(self.f_min, self.f_max, self.f_n)
        self.a = _scalar(self.a, "a", strict=True)
        if np.iscomplexobj(self.volumetric_fractions) or np.iscomplexobj(self.materials):
            raise ValueError("CRIM fractions and material parameters must be real")
        self.volumetric_fractions = np.asarray(self.volumetric_fractions, dtype=float)
        self.materials = np.asarray(self.materials, dtype=float)
        if self.volumetric_fractions.ndim != 1 or len(self.volumetric_fractions) < 2:
            raise ValueError("volumetric_fractions must be a vector of at least two fractions")
        if self.materials.shape != (len(self.volumetric_fractions), 3):
            raise ValueError("materials must contain one [e_inf, de, tau_0] row per fraction")
        if (
            not np.all(np.isfinite(self.volumetric_fractions))
            or np.any(self.volumetric_fractions < 0)
            or np.any(self.volumetric_fractions > 1)
        ):
            raise ValueError("volumetric_fractions must be finite and between zero and one")
        if (
            not np.all(np.isfinite(self.materials))
            or np.any(self.materials[:, 0] < 1)
            or np.any(self.materials[:, 1] < 0)
            or np.any(self.materials[:, 2] <= 0)
        ):
            raise ValueError("materials require finite e_inf >= 1, de >= 0 and tau_0 > 0")
        if not np.isclose(np.sum(self.volumetric_fractions), 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("The summation of volumetric volumes should be equal to 1")
        self.params = dict(
            f_min=self.f_min,
            f_max=self.f_max,
            a=self.a,
            volumetric_fractions=self.volumetric_fractions,
            materials=self.materials,
        )

    def print_info(self):
        """Print information about chosen approximation settings"""
        print(
            f"Approximating Complex Refractive Index Model (CRIM)"
            f" using {self.number_of_debye_poles} Debye poles"
        )
        print("CRIM parameters: ")
        for i in range(len(self.volumetric_fractions)):
            print(f"Material {i + 1}.:")
            print("---------------------------------")
            print(f"{'Vol. fraction':>27s} = {self.volumetric_fractions[i]}")
            print(f"{'e_inf':>27s} = {self.materials[i][0]}")
            print(f"{'De':>27s} = {self.materials[i][1]}")
            print(f"{'log(tau_0)':>27s} = {np.log10(self.materials[i][2])}")

    def calculation(self):
        """Calculates the Crim function for the given parameters"""
        return np.sum(
            self.volumetric_fractions
            * (
                self.materials[:, 0]
                + self.materials[:, 1]
                / (
                    1
                    + 1j
                    * 2
                    * np.pi
                    * np.repeat(self.freq, len(self.materials)).reshape((-1, len(self.materials)))
                    * self.materials[:, 2]
                )
            )
            ** self.a,
            axis=1,
        ) ** (1 / self.a)


class Rawdata(Relaxation):
    """Interpolate data given from a text file.

    :param filename: text file which contains three columns:
                     frequency (Hz),real relative permittivity,positive loss epsilon''.
                     sigma is an additional conductivity, not subtracted from these data.
    :type filename: str, non-optional
    :param delimiter: separator for three data columns
    :type delimiter: str, optional (Deafult: ',')
    """

    def __init__(
        self,
        filename,
        sigma,
        mu,
        mu_sigma,
        material_name,
        number_of_debye_poles=-1,
        f_n=50,
        delimiter=",",
        plot=False,
        save=False,
        optimizer=PSO_DLS,
        optimizer_options=None,
    ):
        super(Rawdata, self).__init__(
            sigma=sigma,
            mu=mu,
            mu_sigma=mu_sigma,
            material_name=material_name,
            f_n=f_n,
            number_of_debye_poles=number_of_debye_poles,
            plot=plot,
            save=save,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
        )
        self.delimiter = delimiter
        self.filename = Path(filename).absolute()
        self.params = {"filename": self.filename}

    def check_inputs(self):
        """Check the validity of the inputs."""
        super(Rawdata, self).check_inputs()

        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"Dielectric data file does not exist: {self.filename}")

    def calculation(self):
        """Interpolate real and imaginary part from data.
        Column framework of the input file three columns comma-separated
        Frequency(Hz),Real,Imaginary
        """
        if self.delimiter is not None and (
            not isinstance(self.delimiter, str) or not self.delimiter
        ):
            raise ValueError("delimiter must be a non-empty string or None for whitespace")
        try:
            # Preserve support for multi-character separators such as ', '.
            with open(self.filename) as stream:
                array = np.asarray(
                    [
                        [float(value) for value in line.split(self.delimiter)]
                        for line in stream
                        if line.strip() and not line.lstrip().startswith("#")
                    ],
                    dtype=float,
                )
        except ValueError as exc:
            raise ValueError(
                "Rawdata requires three numeric columns: frequency, real, positive loss"
            ) from exc
        if array.ndim != 2 or array.shape[1] != 3 or len(array) < 2:
            raise ValueError("Rawdata requires at least two rows and exactly three columns")
        if (
            not np.all(np.isfinite(array))
            or np.any(array[:, 0] <= 0)
            or np.any(array[:, 1] < 1)
            or np.any(array[:, 2] < 0)
        ):
            raise ValueError(
                "Rawdata requires finite frequencies > 0, real permittivity >= 1, and positive loss >= 0"
            )
        array = array[np.argsort(array[:, 0])]
        if np.any(np.diff(array[:, 0]) == 0):
            raise ValueError(
                "Rawdata frequencies must be distinct; combine duplicate measurements first"
            )
        if self.sigma:
            warnings.warn(
                "Rawdata fits the supplied loss without subtracting conductivity. sigma adds "
                "extra loss to the exported material and is excluded from the reported fit error. "
                "Use sigma=0 when the supplied data already include all conduction loss.",
                UserWarning,
                stacklevel=2,
            )
        self.set_freq(array[0, 0], array[-1, 0], self.f_n)
        rl_interp = scipy.interpolate.interp1d(array[:, 0], array[:, 1], bounds_error=True)
        im_interp = scipy.interpolate.interp1d(array[:, 0], array[:, 2], bounds_error=True)
        return rl_interp(self.freq) - 1j * im_interp(self.freq)


if __name__ == "__main__":
    # Kelley et al. parameters
    setup = HavriliakNegami(
        f_min=1e7,
        f_max=1e11,
        alpha=0.91,
        beta=0.45,
        e_inf=2.7,
        de=8.6 - 2.7,
        tau_0=9.4e-10,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="Kelley",
        f_n=100,
        number_of_debye_poles=6,
        plot=True,
        save=False,
        optimizer_options={
            "swarmsize": 30,
            "maxiter": 100,
            "omega": 0.5,
            "phip": 1.4,
            "phig": 1.4,
            "minstep": 1e-8,
            "minfun": 1e-8,
            "seed": 111,
            "pflag": True,
        },
    )
    setup.run()
    setup = HavriliakNegami(
        f_min=1e7,
        f_max=1e11,
        alpha=0.91,
        beta=0.45,
        e_inf=2.7,
        de=8.6 - 2.7,
        tau_0=9.4e-10,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="Kelley",
        f_n=100,
        number_of_debye_poles=6,
        plot=True,
        save=False,
        optimizer=DA_DLS,
        optimizer_options={"seed": 111},
    )
    setup.run()
    setup = HavriliakNegami(
        f_min=1e7,
        f_max=1e11,
        alpha=0.91,
        beta=0.45,
        e_inf=2.7,
        de=8.6 - 2.7,
        tau_0=9.4e-10,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="Kelley",
        f_n=100,
        number_of_debye_poles=6,
        plot=True,
        save=False,
        optimizer=DE_DLS,
        optimizer_options={"seed": 111},
    )
    setup.run()
    # Testing setup
    setup = Rawdata(
        "examples/Test.txt",
        0,
        1,
        0.1,
        "M1",
        number_of_debye_poles=3,
        plot=True,
        optimizer_options={"seed": 111},
    )
    setup.run()
    np.random.seed(111)
    setup = HavriliakNegami(
        1e12,
        1e-3,
        0.5,
        1,
        10,
        5,
        1e-6,
        0.1,
        1,
        0,
        "M2",
        number_of_debye_poles=6,
        plot=True,
    )
    setup.run()
    setup = Jonscher(
        1e6,
        1e-5,
        50,
        1,
        1e5,
        0.7,
        0.1,
        1,
        0.1,
        "M3",
        number_of_debye_poles=4,
        plot=True,
    )
    setup.run()
    f = np.array([0.5, 0.5])
    material1 = [3, 25, 1e6]
    material2 = [3, 0, 1e3]
    materials = np.array([material1, material2])
    setup = Crim(
        1 * 1e-1,
        1e-9,
        0.5,
        f,
        materials,
        0.1,
        1,
        0,
        "M4",
        number_of_debye_poles=2,
        plot=True,
    )
    setup.run()
