Optimization methods of multi-Debye fitting
-------------------------------------------

The ``Optimizer`` class supports global optimization algorithms (particle swarm, dual annealing, evolutionary algorithms) for finding an optimal set of relaxation times that minimise the error between the actual and the approximated electric permittivity, and calculates optimised weights for the given relaxation times.
Code written here is mainly based on external libraries, like ```scipy``` and ```pyswarm```.

Supported methods:
- [x] hybrid Particle Swarm-Damped Least Squares
- [x] hybrid Dual Annealing-Damped Least Squares
- [x] hybrid Differential Evolution-Damped Least Squares

Methods
^^^^^^^

1. __constructor__ - is called in all child classes.

    It takes the following arguments:
    - `maxiter`: maximum number of iterations for the optimizer,
    - `seed`: Seed for RandomState.

    In constructor the attributes:
    - `maxiter`,
    - `seed`,
    - `calc_weights` (used to fit weight, non-linear least squares (LS) method is used as a default)
    are set.

2. __fit__ - is inherited by all children classes. It calls the optimization function that tries to find an optimal set of relaxation times that minimise the error between the actual and the approximated electric permittivity and calculate optimised weights for the given relaxation times.
    It takes the following arguments:
    - `func`: objective function to be optimized,
    - `lb`: the lower bounds of the design variable(s),
    - `ub`: the upper bounds of the design variable(s),
    - `funckwargs`: optional arguments takien by objective function.

3. __cost_function__ - is inherited by all child classes. It calculates the cost function as the average error between the actual and the approximated electric permittivity (sum of real and imaginary part).
    It takes the following arguments:
    - `x`: the logarithm with base 10 of relaxation times of the Debyes poles,
    - `rl`: real parts of chosen relaxation function for given frequency points,
    - `im`: imaginary parts of chosen relaxation function for given frequency points,
    - `freq`: the frequencies vector for defined grid.

4. __calc_relaxation_times__ - it finds an optimal set of relaxation times that minimise an objective function using appropriate optimization procedure.
    It takes the following arguments:
    - `func`: objective function to be optimized,
    - `lb`: the lower bounds of the design variable(s),
    - `ub`: the upper bounds of the design variable(s),
    - `funckwargs`: optional arguments takien by objective function.

Each new class of optimizer should:
- define constructor with appropriate arguments,
- overload __calc_relaxation_times__ method (and optional define __calc_weights__ function in case of hybrid optimization procedure).
