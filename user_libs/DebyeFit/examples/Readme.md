# Getting started

Please see provided examples for the basic usage of ``DebyeFit`` module. We provide jupyter tutorials, and full guidance for quick run with existing relaxation functions and optimizers:

* ```example_DebyeFitting.ipynb```: simple cases of using all available implemented relaxation functions,
* ```example_BiologicalTissues.ipynb```: simple cases of using Cole-Cole function for biological tissues,
* ```example_ColeCole.py```: simple cases of using Cole-Cole function in case of 3, 5 and automatically chosen number of Debye poles.

Main usage of the specific relaxation fucntion based on creation of choosen relaxation model and then calling run method.

```python

    # set Havrilak-Negami function with initial parameters
    setup = HavriliakNegami(f_min=1e4, f_max=1e11,
                            alpha=0.3, beta=1,
                            e_inf=3.4, de=2.7, tau_0=.8e-10,
                            sigma=0.45e-3, mu=1, mu_sigma=0,
                            material_name="dry_sand", f_n=100,
                            plot=True, save=False,
                            number_of_debye_poles=3,
                            optimizer_options={'swarmsize':30,
                                               'maxiter':100,
                                               'omega':0.5,
                                               'phip':1.4,
                                               'phig':1.4,
                                               'minstep':1e-8,
                                               'minfun':1e-8,
                                               'seed': 111,
                                               'pflag': True})
    # run optimization
    setup.run()

```
