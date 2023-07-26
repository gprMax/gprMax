# I. Giannakis, A. Giannopoulos and N. Davidson,
# "Incorporating dispersive electrical properties in FDTD GPR models
# using a general Cole-Cole dispersion function,"
# 2012 14th International Conference on Ground Penetrating Radar (GPR), 2012, pp. 232-236
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Debye_Fit import HavriliakNegami

if __name__ == "__main__":
    # set Havrilak-Negami function with initial parameters
    setup = HavriliakNegami(
        f_min=1e4,
        f_max=1e11,
        alpha=0.3,
        beta=1,
        e_inf=3.4,
        de=2.7,
        tau_0=0.8e-10,
        sigma=0.45e-3,
        mu=1,
        mu_sigma=0,
        material_name="dry_sand",
        f_n=100,
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
    ### Dry Sand in case of 3, 5
    # and automatically set number of Debye poles (-1)
    for number_of_debye_poles in [3, 5, -1]:
        setup.number_of_debye_poles = number_of_debye_poles
        setup.run()

    ### Moist sand
    # set Havrilak-Negami function parameters
    setup.material_name = "moist_sand"
    setup.alpha = 0.25
    setup.beta = 1
    setup.e_inf = 5.6
    setup.de = 3.3
    setup.tau_0 = (1.1e-10,)
    setup.sigma = 2e-3
    # calculate for different number of Debye poles
    for number_of_debye_poles in [3, 5, -1]:
        setup.number_of_debye_poles = number_of_debye_poles
        setup.run()
