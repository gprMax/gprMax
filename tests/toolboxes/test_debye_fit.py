# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

import numpy as np

from toolboxes.DebyeFit.Debye_Fit import HavriliakNegami


def test_short_havriliak_negami_fit_produces_gprmax_material_commands():
    model = HavriliakNegami(
        f_min=1e6,
        f_max=1e9,
        alpha=1,
        beta=1,
        e_inf=3,
        de=5,
        tau_0=1e-9,
        sigma=0,
        mu=1,
        mu_sigma=0,
        material_name="smoke_test",
        number_of_debye_poles=1,
        f_n=12,
        plot=False,
        save=False,
        optimizer_options={"swarmsize": 4, "maxiter": 2, "seed": 1},
    )

    error, properties = model.run()

    assert np.isfinite(error)
    assert any(line.startswith("#material:") for line in properties)
    assert any(line.startswith("#add_dispersion_debye:") for line in properties)
