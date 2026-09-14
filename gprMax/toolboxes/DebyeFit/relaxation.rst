Relaxation classes for multi-Debye fitting
------------------------------------------

This class is designed for modelling different relaxation functions, like Havriliak-Negami (```HavriliakNegami```), Jonscher (```Jonscher```), Complex Refractive Index Mixing (```CRIM```) models, and arbitrary dielectric data derived experimentally or calculated using some other function (```Rawdata```).

Supported relaxation classes:

- [x] Havriliak-Negami,
- [x] Jonscher,
- [x] Complex Refractive Index Mixing,
- [x] Experimental data,

Methods
^^^^^^^

1. __constructor__ - is called in all child classes, creates Relaxation function object for complex material.

    It takes the following arguments:
    - ``sigma`` is a conductivity (Siemens/metre),
    - ``mu`` is a relative permeability,
    - ``mu_sigma`` is a magnetic loss (Ohms/metre),
    - ``material_name`` is definition of material name,
    - ``number_of_debye_poles`` is choosen number of Debye poles,
    - ``f_n`` is the chosen number of frequences,
    - ``plot`` is a switch to turn on the plotting,
    - ``save`` is a switch to turn on the saving final material properties,
    - ``optimizer`` is a chosen optimizer to fit model to dielectric data,
    - ``optimizer_options`` is a dict for options of chosen optimizer.

    Additional parameters:
    - ``rl`` calculated real part of chosen relaxation function for given frequency points,
    - ``im`` calculated imaginary part of chosen relaxation function for given frequency points.

2. __set_freq__ - is inherited by all child classes, interpolates frequency vector using n equally logarithmicaly spaced frequencies.
    It takes the following arguments:
    - `f_min_`: first bound of the frequency range used to approximate the given function (Hz),
    - `f_max`: second bound of the frequency range used to approximate the given function (Hz),
    - `f_n`: the number of frequency points in frequency grid (Default: 50).

3. __run__ - is inherited by all child classes, solves the problem described by the given relaxation function (main operational method).
    It consists of following steps:
    1) Check the validity of the inputs using ```check_inputs``` method.
    2) Print information about chosen approximation settings using ```print_info``` method.
    3) Calculate both real and imaginary parts using ```calculation``` method, and then set ```self.rl``` and ```self.im``` properties.
    4) Calling the main optimisation module using ```optimize``` method and calculate error based on ```error``` method.
       a) [OPTIONAL] If number of debye poles is set to -1 optimization procedure is repeated until the percentage error is les than 5% or 20 Debye poles is reached.
    5) Print the results in gprMax format style using ```print_output``` method.
    6) [OPTIONAL] Save results in gprMax style using ```save_result``` method.
    7) [OPTIONAL] Plot the actual and the approximate dielectric properties using ```plot_result``` method.

4. __check_inputs__ - is called in all child classes, finds an optimal set of relaxation times that minimise an objective function using appropriate optimization procedure.

5. __calculation__ - is inherited by all child classes, should be definied in all new chil classes, approximates the given relaxation function.

6. __print_info__ - is inherited by all child classes, prints readable string of parameters for given approximation settings.

7. __optimize__ - is inherited by all child classes, calls the main optimisation module with defined lower and upper boundaries of search.

8. __print_output__ - is inherited by all child classes, prints out the resulting Debye parameters in a gprMax format.

9. __plot_result__ - is inherited by all child classes, plots the actual and the approximated electric permittivity, along with relative error for real and imaginary parts using a semilogarithm X axes.

10. __save_result__ - is inherited by all child classes, saves the resulting Debye parameters in a gprMax format.

11. __error__ -is inherited by all child classes, calculates the average fractional error separately for relative permittivity (real part) and conductivity (imaginary part).

Each new class of relaxation object should:

- define constructor with appropriate arguments,
- define __check_inputs__ method to check relaxation class specific parameters,
- overload __calculation__ method.
