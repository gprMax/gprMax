# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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

import logging

import numpy as np

import gprMax.config as config

logger = logging.getLogger(__name__)


def validate_lorentz_pole(frequency_hz: float, damping_per_s: float, dt: float) -> None:
    """Validate a Lorentz pole for the recursive material formulation.

    Lorentz resonance frequencies are supplied in hertz, whereas the
    coefficient construction uses the angular frequency ``2 * pi * f``.
    The simple conjugate-pole representation used by gprMax is valid only for
    an underdamped pole.
    """

    if frequency_hz <= 0:
        raise ValueError("Lorentz resonance frequency must be positive")
    if damping_per_s < 0:
        raise ValueError("Lorentz damping coefficient must be non-negative")
    if frequency_hz >= 1.0 / dt:
        raise ValueError("Lorentz resonance frequency must be below 1 / dt")
    if damping_per_s >= 1.0 / dt:
        raise ValueError("Lorentz damping coefficient must be below 1 / dt")

    angular_frequency = 2.0 * np.pi * frequency_hz
    if damping_per_s >= angular_frequency or np.isclose(
        damping_per_s,
        angular_frequency,
        rtol=1e-12,
        atol=0.0,
    ):
        raise ValueError("Lorentz damping coefficient must be below 2 * pi * resonance frequency")


def validate_drude_pole(frequency_hz: float, collision_per_s: float, dt: float) -> None:
    """Validate a Drude pole for the recursive material formulation."""

    if frequency_hz <= 0:
        raise ValueError("Drude plasma frequency must be positive")
    if collision_per_s <= 0:
        raise ValueError("Drude collision frequency must be positive")
    if frequency_hz >= 1.0 / dt:
        raise ValueError("Drude plasma frequency must be below 1 / dt")
    if collision_per_s >= 1.0 / dt:
        raise ValueError("Drude collision frequency must be below 1 / dt")


class Material:
    """Super-class to describe generic, non-dispersive materials,
    their properties and update coefficients.
    """

    def __init__(self, numID: int, ID: str):
        """
        Args:
            numID: int for numeric I of the material.
            ID: string for name of the material.
        """

        self.numID = numID
        self.ID = ID
        self.type = ""
        # Default material averaging
        self.averagable = True

        # Default material constitutive parameters (free_space)
        self.er = 1.0
        self.se = 0.0
        self.mr = 1.0
        self.sm = 0.0

        # Optional cell-centred physical mass density in SI units. Density is
        # deliberately not part of the electromagnetic update coefficients or
        # dielectric smoothing: it belongs to the final volumetric material
        # assignment and is consumed only by derived quantities such as SAR.
        self.mass_density = None

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Material):
            return self.ID == value.ID
        else:
            raise TypeError(
                f"'==' not supported between instances of 'Material' and '{type(value)}'"
            )

    def __lt__(self, value: "Material") -> bool:
        """Less than comparator for two Materials.

        Only non-compound materials (i.e. default or user added
        materials) are guaranteed to have the same numID for the same
        material across MPI ranks. Therefore compound materials are
        sorted by ID and non-compound materials are always less than
        compound materials.
        """
        if not isinstance(value, Material):
            raise TypeError(
                f"'<' not supported between instances of 'Material' and '{type(value)}'"
            )
        elif self.is_compound_material() and value.is_compound_material():
            return self.ID < value.ID
        elif not self.is_compound_material() and not value.is_compound_material():
            return self.numID < value.numID
        else:
            return value.is_compound_material()

    def __gt__(self, value: "Material") -> bool:
        """Greater than comparator for two Materials.

        Only non-compound materials (i.e. default or user added
        materials) are guaranteed to have the same numID for the same
        material across MPI ranks. Therefore compound materials are
        sorted by ID and are always greater than non-compound materials.
        """
        if not isinstance(value, Material):
            raise TypeError(
                f"'>' not supported between instances of 'Material' and '{type(value)}'"
            )
        elif self.is_compound_material() and value.is_compound_material():
            return self.ID > value.ID
        elif not self.is_compound_material() and not value.is_compound_material():
            return self.numID > value.numID
        else:
            return self.is_compound_material()

    def is_compound_material(self) -> bool:
        """Check if a material is a compound material.

        The ID of a compound material comprises of the component
        material IDs joined by a '+' symbol. Therefore we check for a
        compound material by looking for a '+' symbol in the material
        ID.

        Returns:
            is_compound_material: True if material is a compound
                material. False otherwise.
        """
        return self.ID.count("+") > 0

    @property
    def is_pec(self) -> bool:
        """Check if a material is electrically a perfect conductor (PEC).

        Matches the builtin 'pec' material by name, or any user-defined
        material with infinite electric conductivity (se == inf) - the same
        criterion already used in calculate_update_coeffsE() and in
        Material.build()'s averagable check, so a custom PEC-equivalent
        material is treated consistently everywhere.

        Returns:
            is_pec: True if material is PEC or PEC-equivalent.
        """
        return self.ID == "pec" or self.se == float("inf")

    @property
    def is_pmc(self) -> bool:
        """Check if a material is magnetically a perfect conductor (PMC).

        This mirrors :attr:`is_pec`: the builtin material name and a
        user-defined infinite magnetic conductivity are equivalent to the
        magnetic-field update equations.

        Returns:
            is_pmc: True if material is PMC or PMC-equivalent.
        """
        return self.ID == "pmc" or self.sm == float("inf")

    @staticmethod
    def create_compound_id(*materials: "Material") -> str:
        """Create a compound ID from existing materials.

        The new ID will be the IDs of the existing materials joined by a
        '+' symbol. The component IDs will be sorted alphabetically and
        if two materials are provided, the compound ID will contain each
        material twice.

        Args:
            *materials: Materials to use to create the compound ID.

        Returns:
            compound_id: New compound id.
        """
        if len(materials) == 2:
            materials += materials
        return "+".join(sorted([material.ID for material in materials]))

    def calculate_update_coeffsH(self, G):
        """Calculates the magnetic update coefficients of the material.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        HA = (config.m0 * self.mr / G.dt) + 0.5 * self.sm
        HB = (config.m0 * self.mr / G.dt) - 0.5 * self.sm

        if self.ID == "pmc" or self.sm == float("inf"):
            self.DA = 0
            self.DBx = 0
            self.DBy = 0
            self.DBz = 0
            self.srcm = 0
        else:
            self.DA = HB / HA
            self.DBx = (1 / G.dx) * 1 / HA
            self.DBy = (1 / G.dy) * 1 / HA
            self.DBz = (1 / G.dz) * 1 / HA
            self.srcm = 1 / HA

    def calculate_update_coeffsE(self, G):
        """Calculates the electric update coefficients of the material.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        EA = (config.sim_config.em_consts["e0"] * self.er / G.dt) + 0.5 * self.se
        EB = (config.sim_config.em_consts["e0"] * self.er / G.dt) - 0.5 * self.se

        if self.ID == "pec" or self.se == float("inf"):
            self.CA = 0
            self.CBx = 0
            self.CBy = 0
            self.CBz = 0
            self.srce = 0
        else:
            self.CA = EB / EA
            self.CBx = (1 / G.dx) * 1 / EA
            self.CBy = (1 / G.dy) * 1 / EA
            self.CBz = (1 / G.dz) * 1 / EA
            self.srce = 1 / EA

    def calculate_er(self, freq):
        """Calculates the complex relative permittivity of the material at a
            specific frequency.

        Args:
            freq: float for frequency used to calculate complex relative
                    permittivity.

        Returns:
            er: float for complex relative permittivity.
        """

        return self.er


class DispersiveMaterial(Material):
    """Class to describe materials with frequency dependent properties, e.g.
    Debye, Drude, Lorenz.
    """

    # Properties of water from: http://dx.doi.org/10.1109/TGRS.2006.873208
    waterer = 80.1
    watereri = 4.9
    waterdeltaer = waterer - watereri
    watertau = 9.231e-12

    # Properties of grass from: http://dx.doi.org/10.1007/BF00902994
    grasser = 18.5087
    grasseri = 12.7174
    grassdeltaer = grasser - grasseri
    grasstau = 1.0793e-11

    def __init__(self, numID, ID):
        super().__init__(numID, ID)
        self.poles = 0
        self.deltaer = []
        self.tau = []
        self.alpha = []
        # Compound interface materials use the inclusive susceptibility
        # representation of Giannakis and Giannopoulos (2014). It permits
        # Debye, Lorentz, and Drude terms to coexist in one recursive update.
        self.inclusive_w = []
        self.inclusive_q = []
        self.inclusive_conductivity = 0.0

    def calculate_update_coeffsE(self, G):
        """Calculates the electric update coefficients of the material.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # The implementation of the dispersive material modelling comes from the
        # derivation in: http://dx.doi.org/10.1109/TAP.2014.2308549
        self.w = np.zeros(
            config.get_model_config().materials["maxpoles"],
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.q = np.zeros(
            config.get_model_config().materials["maxpoles"],
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.zt = np.zeros(
            config.get_model_config().materials["maxpoles"],
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.zt2 = np.zeros(
            config.get_model_config().materials["maxpoles"],
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.eqt = np.zeros(
            config.get_model_config().materials["maxpoles"],
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.eqt2 = np.zeros(
            config.get_model_config().materials["maxpoles"],
            dtype=config.get_model_config().materials["dispersivedtype"],
        )

        # The Drude susceptibility contains a constant causal term whose
        # time derivative is equivalent to an electric conductivity. Keep
        # that numerical conductivity local: ``self.se`` is the physical
        # conductivity supplied by the user and must neither be overwritten
        # nor accumulated if coefficients are calculated more than once.
        effective_se = self.se

        for x in range(self.poles):
            if self.inclusive_w:
                self.w[x] = self.inclusive_w[x]
                self.q[x] = self.inclusive_q[x]
            elif "debye" in self.type:
                self.w[x] = self.deltaer[x] / self.tau[x]
                self.q[x] = -1 / self.tau[x]
            elif "lorentz" in self.type:
                # tau for Lorentz materials are pole frequencies
                # alpha for Lorentz materials are the damping coefficients
                wp2 = (2 * np.pi * self.tau[x]) ** 2
                self.w[x] = -1j * ((wp2 * self.deltaer[x]) / np.sqrt(wp2 - self.alpha[x] ** 2))
                self.q[x] = -self.alpha[x] + (1j * np.sqrt(wp2 - self.alpha[x] ** 2))
            elif "drude" in self.type:
                # tau for Drude materials are pole frequencies
                # alpha for Drude materials are the inverse of relaxation times
                wp2 = (2 * np.pi * self.tau[x]) ** 2
                effective_se += config.sim_config.em_consts["e0"] * wp2 / self.alpha[x]
                self.w[x] = -(wp2 / self.alpha[x])
                self.q[x] = -self.alpha[x]

            self.eqt[x] = np.exp(self.q[x] * G.dt)
            self.eqt2[x] = np.exp(self.q[x] * (G.dt / 2))
            self.zt[x] = (self.w[x] / self.q[x]) * (1 - self.eqt[x]) / G.dt
            self.zt2[x] = (self.w[x] / self.q[x]) * (1 - self.eqt2[x])

        effective_se += self.inclusive_conductivity
        EA = (
            (config.sim_config.em_consts["e0"] * self.er / G.dt)
            + 0.5 * effective_se
            - (config.sim_config.em_consts["e0"] / G.dt) * np.sum(self.zt2.real)
        )
        EB = (
            (config.sim_config.em_consts["e0"] * self.er / G.dt)
            - 0.5 * effective_se
            - (config.sim_config.em_consts["e0"] / G.dt) * np.sum(self.zt2.real)
        )

        # Matches the same guard in the base Material.calculate_update_coeffsE -
        # without it, se=inf (the literal 'pec' material, or any user-defined
        # material with infinite conductivity) makes EA and EB both contain a
        # 0.5*inf term, so EB/EA is the indeterminate form -inf/inf, evaluating
        # to NaN rather than the intended 0 - NaN then propagates through the
        # whole simulation from the very first update.
        if self.ID == "pec" or self.se == float("inf"):
            self.CA = 0
            self.CBx = 0
            self.CBy = 0
            self.CBz = 0
            self.srce = 0
        else:
            self.CA = EB / EA
            self.CBx = (1 / G.dx) * 1 / EA
            self.CBy = (1 / G.dy) * 1 / EA
            self.CBz = (1 / G.dz) * 1 / EA
            self.srce = 1 / EA

    def calculate_er(self, freq):
        """Calculates the complex relative permittivity of the material at a
            specific frequency.

        Args:
            freq: float for frequency used to calculate complex relative
                    permittivity.

        Returns:
            er: float for complex relative permittivity.
        """

        # Permittivity at infinite frequency if the material is dispersive
        er = self.er

        w = 2 * np.pi * freq
        er += self.se / (1j * w * config.e0)
        if self.inclusive_w:
            er += self.inclusive_conductivity / (1j * w * config.e0)
            for pole_w, pole_q in zip(self.inclusive_w, self.inclusive_q):
                if np.isclose(np.imag(pole_w), 0.0) and np.isclose(np.imag(pole_q), 0.0):
                    er += np.real(pole_w) / (1j * w - np.real(pole_q))
                else:
                    er += 0.5 * (
                        pole_w / (1j * w - pole_q)
                        + np.conjugate(pole_w) / (1j * w - np.conjugate(pole_q))
                    )
        elif "debye" in self.type:
            for pole in range(self.poles):
                er += self.deltaer[pole] / (1 + 1j * w * self.tau[pole])
        elif "lorentz" in self.type:
            for pole in range(self.poles):
                pole_omega = 2 * np.pi * self.tau[pole]
                er += (self.deltaer[pole] * pole_omega**2) / (
                    pole_omega**2 + 2j * w * self.alpha[pole] - w**2
                )
        elif "drude" in self.type:
            for pole in range(self.poles):
                pole_omega = 2 * np.pi * self.tau[pole]
                er -= pole_omega**2 / (w**2 - 1j * w * self.alpha[pole])

        return er


def create_electric_average_material(numID, ID, materials):
    """Create the arithmetic electric-edge average of four materials.

    The ordinary constitutive properties are averaged exactly as in the
    existing generalised-Yee construction. For dispersive constituents, each
    inclusive pole residue is scaled by its cell weight while its pole
    location is retained. Identical pole locations are merged exactly. This
    applies to Debye, Lorentz, and Drude terms and permits different
    dispersion families on opposite sides of an interface.

    This is the grid-aligned contour-path formulation developed in Chapter 4
    of Hartley (2020), "On Finite-Difference Time-Domain Sub-Gridding
    Algorithms for Efficient Modelling of Ground-Penetrating Radar".

    Args:
        numID: Numeric identifier for the compound material.
        ID: Deterministic compound material identifier.
        materials: Four materials surrounding an electric Yee edge. Repeated
            entries provide the normal quarter-cell weighting used by gprMax.

    Returns:
        The effective :class:`Material` or :class:`DispersiveMaterial`.
    """

    if len(materials) != 4:
        raise ValueError("Electric-edge averaging requires exactly four materials")

    dispersive = [material for material in materials if getattr(material, "poles", 0) > 0]
    if dispersive:
        averaged = DispersiveMaterial(numID, ID)
        source_types = {
            kind
            for material in dispersive
            for kind in ("debye", "lorentz", "drude")
            if kind in material.type
        }
        averaged.type = "dielectric-smoothed, inclusive, " + ", ".join(sorted(source_types))

        # Each surrounding cell contributes one quarter of its susceptibility.
        # Sorting makes equivalent compound IDs deterministic on MPI ranks.
        pole_strengths = {}
        weight = 1.0 / len(materials)
        for material in materials:
            if not getattr(material, "poles", 0):
                continue
            terms, extra_conductivity = _inclusive_material_terms(material)
            averaged.inclusive_conductivity += weight * extra_conductivity
            for pole_w, pole_q in terms:
                key = complex(pole_q)
                pole_strengths[key] = pole_strengths.get(key, 0j) + weight * complex(pole_w)

        for pole_q in sorted(pole_strengths, key=lambda value: (value.real, value.imag)):
            pole_w = pole_strengths[pole_q]
            if source_types == {"debye"}:
                averaged.inclusive_q.append(pole_q.real)
                averaged.inclusive_w.append(pole_w.real)
            else:
                averaged.inclusive_q.append(pole_q)
                averaged.inclusive_w.append(pole_w)
        averaged.poles = len(averaged.inclusive_q)

        # Preserve familiar Debye parameters for material reporting and for
        # compatibility with tools that inspect deltaer/tau.
        if source_types == {"debye"}:
            for pole_w, pole_q in zip(averaged.inclusive_w, averaged.inclusive_q):
                averaged.tau.append(-1.0 / pole_q)
                averaged.deltaer.append(-pole_w / pole_q)
        averaged.averagable = config.get_model_config().dispersive_averaging

        if averaged.poles > config.get_model_config().materials["maxpoles"]:
            config.get_model_config().materials["maxpoles"] = averaged.poles
    else:
        averaged = Material(numID, ID)
        averaged.type = "dielectric-smoothed"

    averaged.er = np.mean([material.er for material in materials], axis=0)
    averaged.se = np.mean([material.se for material in materials], axis=0)
    averaged.mr = np.mean([material.mr for material in materials], axis=0)
    averaged.sm = np.mean([material.sm for material in materials], axis=0)

    return averaged


def _inclusive_material_terms(material):
    """Return inclusive ``(W, Q)`` terms and Drude-equivalent conductivity."""

    if getattr(material, "inclusive_w", None):
        return (
            list(zip(material.inclusive_w, material.inclusive_q)),
            material.inclusive_conductivity,
        )

    terms = []
    extra_conductivity = 0.0
    if "debye" in material.type:
        for deltaer, tau in zip(material.deltaer, material.tau):
            terms.append((deltaer / tau, -1.0 / tau))
    elif "lorentz" in material.type:
        for deltaer, frequency, damping in zip(material.deltaer, material.tau, material.alpha):
            omega_0_squared = (2 * np.pi * frequency) ** 2
            beta = np.sqrt(omega_0_squared - damping**2)
            terms.append((-1j * omega_0_squared * deltaer / beta, -damping + 1j * beta))
    elif "drude" in material.type:
        for frequency, collision in zip(material.tau, material.alpha):
            omega_p_squared = (2 * np.pi * frequency) ** 2
            extra_conductivity += config.e0 * omega_p_squared / collision
            terms.append((-omega_p_squared / collision, -collision))
    return terms, extra_conductivity


class PeplinskiSoil:
    """Soil objects that are characterised according to a mixing model
    by Peplinski (http://dx.doi.org/10.1109/36.387598).
    """

    def __init__(
        self, ID, sandfraction, clayfraction, bulkdensity, sandpartdensity, watervolfraction
    ):
        """
        Args:
            ID: string for name of the soil.
            sandfraction: float of sand fraction of the soil.
            clayfraction: float of clay fraction of the soil.
            bulkdensity: float of bulk density of the soil (g/cm3).
            sandpartdensity: float of density of the sand particles in the
                                soil (g/cm3).
            watervolfraction: tuple of floats of two numbers that specify a
                                range for the volumetric water fraction of the
                                soil.
        """

        self.ID = ID
        self.S = sandfraction
        self.C = clayfraction
        self.rb = bulkdensity
        self.rs = sandpartdensity
        self.mu = watervolfraction
        # Store all of the material IDs which allows for more general mixing models.
        self.matID = []

    def calculate_properties(self, nbins, G):
        """Calculates the real and imaginery part of a Debye model for the soil
        as well as a conductivity. It uses an approximation to a semi-empirical
        model (http://dx.doi.org/10.1109/36.387598).

        Args:
            nbins: int for number of bins to use to create the different materials.
            G: FDTDGrid class describing a grid in a model.
        """

        # Debye model properties of water at 25C & zero salinity
        T = 25
        S = 0
        watereri, waterer, watertau, watersig = calculate_water_properties(T, S)
        f = 1.3e9
        w = 2 * np.pi * f
        waterdeltaer = waterer - watereri
        erealw = watereri + (waterdeltaer / (1 + (w * watertau) ** 2))

        a = 0.65  # Experimentally derived constant
        es = (1.01 + 0.44 * self.rs) ** 2 - 0.062  #  Relative permittivity of sand particles
        b1 = 1.2748 - 0.519 * self.S - 0.152 * self.C
        b2 = 1.33797 - 0.603 * self.S - 0.166 * self.C

        # For frequencies in the range 0.3GHz to 1.3GHz
        sigf = 0.0467 + 0.2204 * self.rb - 0.411 * self.S + 0.6614 * self.C
        # For frequencies in the range 1.4GHz to 18GHz
        # sigf = -1.645 + 1.939 * self.rb - 2.25622 * self.S + 1.594 * self.C

        # Generate a set of bins based on the given volumetric water fraction
        # values. Changed to make sure mid points are contained completely within the ranges.
        # The limiting values of the ranges are not included in this.

        mubins = np.linspace(self.mu[0], self.mu[1], nbins + 1)
        # Generate a range of volumetric water fraction values the mid-point of
        # each bin to make materials from
        mumaterials = 0.5 * (mubins[1 : nbins + 1] + mubins[0:nbins])

        # Create an iterator
        muiter = np.nditer(mumaterials, flags=["c_index"])
        while not muiter.finished:
            # Real part for frequencies in the range 1.4GHz to 18GHz
            er = (
                1
                + (self.rb / self.rs) * ((es**a) - 1)
                + (muiter[0] ** b1 * erealw**a)
                - muiter[0]
            ) ** (1 / a)
            # Real part for frequencies in the range 0.3GHz to 1.3GHz (linear
            # correction to 1.4-18GHz value)
            er = 1.15 * er - 0.68

            # Permittivity at infinite frequency
            eri = er - (muiter[0] ** (b2 / a) * waterdeltaer)

            # Effective conductivity
            sig = muiter[0] ** (b2 / a) * ((sigf * (self.rs - self.rb)) / (self.rs * muiter[0]))

            # Create individual materials
            m = DispersiveMaterial(len(G.materials), None)
            m.type = "debye"
            m.averagable = config.get_model_config().dispersive_averaging
            m.poles = 1
            if m.poles > config.get_model_config().materials["maxpoles"]:
                config.get_model_config().materials["maxpoles"] = m.poles
            m.er = eri
            m.se = sig
            m.deltaer.append(er - eri)
            m.tau.append(watertau)
            m.ID = f"|{float(m.er):.4f}+{float(m.se):.4f}+{float(m.mr):.4f}+{float(m.sm):.4f}|"
            G.materials.append(m)
            self.matID.append(m.numID)

            muiter.iternext()


class RangeMaterial:
    """Material objects defined by a given range of their parameters to be used
    for fractal spatial distributions.
    """

    def __init__(self, ID, er_range, se_range, mr_range, sm_range):
        """
        Args:
            ID: string for name of the material range.
            er_range: tuple of floats for relative permittivity range of the
                        materials.
            se_range: tuple of floats for electric conductivity range of the
                        materials.
            mr_range: tuple of floats for magnetic permeability of materials.
            sm_range: tuple of floats for magnetic loss range of materials.
        """

        self.ID = ID
        self.er = er_range
        self.sig = se_range
        self.mu = mr_range
        self.ro = sm_range
        # Store all of the material IDs which allows for more general mixing models.
        self.matID = []

    def calculate_properties(self, nbins, G):
        """Calculates the specific properties of each of the materials.

        Args:
            nbins: int for number of bins to use to create the different materials.
            G: FDTDGrid class describing a grid in a model.
        """

        # Generate a set of relative permittivity bins based on the given range
        erbins = np.linspace(self.er[0], self.er[1], nbins + 1)

        # Generate a range of relative permittivity values the mid-point of
        # each bin to make materials from
        ermaterials = 0.5 * (erbins[1 : nbins + 1] + erbins[0:nbins])

        # Generate a set of conductivity bins based on the given range
        sigmabins = np.linspace(self.sig[0], self.sig[1], nbins + 1)

        # Generate a range of conductivity values the mid-point of
        # each bin to make materials from
        sigmamaterials = 0.5 * (sigmabins[1 : nbins + 1] + sigmabins[0:nbins])

        # Generate a set of magnetic permeability bins based on the given range
        mubins = np.linspace(self.mu[0], self.mu[1], nbins + 1)

        # Generate a range of magnetic permeability values the mid-point of
        # each bin to make materials from
        mumaterials = 0.5 * (mubins[1 : nbins + 1] + mubins[0:nbins])

        # Generate a set of magnetic loss bins based on the given range
        robins = np.linspace(self.ro[0], self.ro[1], nbins + 1)

        # Generate a range of magnetic loss values the mid-point of each bin to
        # make materials from
        romaterials = 0.5 * (robins[1 : nbins + 1] + robins[0:nbins])

        # Iterate over the bins
        for iter in np.arange(nbins):
            # Relative permittivity
            er = ermaterials[iter]
            # Effective conductivity
            se = sigmamaterials[iter]
            # Magnetic permeability
            mr = mumaterials[iter]
            # Magnetic loss
            sm = romaterials[iter]

            # Check to see if the material already exists before creating a new one
            requiredID = f"|{float(er):.4f}+{float(se):.4f}+{float(mr):.4f}+{float(sm):.4f}|"
            material = next((x for x in G.materials if x.ID == requiredID), None)
            # `self.matID` must gain exactly one entry per bin, regardless
            # of whether this bin reuses an existing material or needs a
            # new one - the previous `iter == 0` guard only appended a
            # reused material's ID on the first bin, so any later bin
            # that happened to reuse an existing material appended
            # nothing at all, leaving matID shorter than nbins and every
            # subsequent bin's index into it wrong (see fractal_box.py's
            # `mixingmodel.matID[int(numberinbin)]` lookup).
            if material:
                self.matID.append(material.numID)
            else:
                m = Material(len(G.materials), requiredID)
                m.type = ""
                m.averagable = True
                m.er = er
                m.se = se
                m.mr = mr
                m.sm = sm
                G.materials.append(m)
                self.matID.append(m.numID)


class ListMaterial:
    """A list of predefined materials to be used for fractal spatial distributions.
    This class does not create new materials but collects them to be used
    in a stochastic distribution by a fractal box.
    """

    def __init__(self, ID, listofmaterials):
        """
        Args:
            ID: string for name of the material list.
            listofmaterials: list of material IDs.
        """

        self.ID = ID
        self.mat = listofmaterials
        # Store all of the material IDs which allows for more general mixing models.
        self.matID = []

    def calculate_properties(self, nbins, G):
        """Calculates the properties of the materials.

        Args:
            nbins: int for number of bins to use to create the different materials.
            G: FDTDGrid class describing a grid in a model.
        """

        # Iterate over the bins
        for iter in np.arange(nbins):
            requiredID = self.mat[iter]
            # Check if the material already exists before creating a new one
            material = next((x for x in G.materials if x.ID == requiredID), None)

            if not material:
                logger.exception(self.__str__() + f" material(s) {requiredID} do not exist")
                raise ValueError

            self.matID.append(material.numID)


def create_built_in_materials(G):
    """Creates pre-defined (built-in) materials.

    Args:
        G: FDTDGrid class describing a grid in a model.
    """

    m = Material(0, "pec")
    m.se = float("inf")
    m.type = "builtin"
    m.averagable = False
    G.materials.append(m)

    m = Material(1, "pmc")
    m.sm = float("inf")
    m.type = "builtin"
    m.averagable = False
    G.materials.append(m)

    m = Material(2, "free_space")
    m.type = "builtin"
    G.materials.append(m)


def calculate_water_properties(T=25, S=0):
    """Get extended Debye model properties for water.

    Args:
        T: float for temperature of water (degrees centigrade).
        S: float for salinity of water (part per thousand).

    Returns:
        eri: float for relative permittivity at infinite frequency.
        er: float for static relative permittivity.
        tau: float for relaxation time (s).
        sig: float for conductivity (Siemens/m).
    """

    # Properties of water from: https://doi.org/10.1109/JOE.1977.1145319
    eri = 4.9
    er = 88.045 - 0.4147 * T + 6.295e-4 * T**2 + 1.075e-5 * T**3
    tau = (1 / (2 * np.pi)) * (1.1109e-10 - 3.824e-12 * T + 6.938e-14 * T**2 - 5.096e-16 * T**3)

    delta = 25 - T
    beta = (
        2.033e-2
        + 1.266e-4 * delta
        + 2.464e-6 * delta**2
        - S * (1.849e-5 - 2.551e-7 * delta + 2.551e-8 * delta**2)
    )
    sig_25s = S * (0.182521 - 1.46192e-3 * S + 2.09324e-5 * S**2 - 1.28205e-7 * S**3)
    sig = sig_25s * np.exp(-delta * beta)

    return eri, er, tau, sig


def create_water(G, T=25, S=0):
    """Creates single-pole Debye model for water with specified temperature and
        salinity.

    Args:
        T: float for temperature of water (degrees centigrade).
        S: float for salinity of water (part per thousand).
        G: FDTDGrid class describing a grid in a model.
    """

    eri, er, tau, sig = calculate_water_properties(T, S)

    m = DispersiveMaterial(len(G.materials), "water")
    m.averagable = config.get_model_config().dispersive_averaging
    m.type = "builtin, debye"
    m.poles = 1
    m.er = eri
    m.se = sig
    m.deltaer.append(er - eri)
    m.tau.append(tau)
    G.materials.append(m)
    if config.get_model_config().materials["maxpoles"] == 0:
        config.get_model_config().materials["maxpoles"] = 1


def create_grass(G):
    """Creates single-pole Debye model for grass

    Args:
        G: FDTDGrid class describing a grid in a model.
    """

    # Properties of grass from: http://dx.doi.org/10.1007/BF00902994
    er = 18.5087
    eri = 12.7174
    tau = 1.0793e-11
    sig = 0

    m = DispersiveMaterial(len(G.materials), "grass")
    m.averagable = config.get_model_config().dispersive_averaging
    m.type = "builtin, debye"
    m.poles = 1
    m.er = eri
    m.se = sig
    m.deltaer.append(er - eri)
    m.tau.append(tau)
    G.materials.append(m)
    if config.get_model_config().materials["maxpoles"] == 0:
        config.get_model_config().materials["maxpoles"] = 1


def process_materials(G):
    """Processes complete list of materials - calculates update coefficients,
        stores in arrays, and builds text list of materials/properties

    Args:
        G: FDTDGrid class describing a grid in a model.

    Returns:
        materialsdata: list of material IDs, names, and properties to
                        print a table.
    """

    if config.get_model_config().materials["maxpoles"] == 0:
        materialsdata = [
            [
                "\nID",
                "\nName",
                "\nType",
                "\neps_r",
                "sigma\n[S/m]",
                "\nmu_r",
                "sigma*\n[Ohm/m]",
                "Dielectric\nsmoothable",
            ]
        ]
    else:
        materialsdata = [
            [
                "\nID",
                "\nName",
                "\nType",
                "\neps_r",
                "sigma\n[S/m]",
                "Delta\neps_r",
                "tau\n[s]",
                "omega\n[Hz]",
                "delta\n[Hz]",
                "gamma\n[Hz]",
                "\nmu_r",
                "sigma*\n[Ohm/m]",
                "Dielectric\nsmoothable",
            ]
        ]

    for material in G.materials:
        # Calculate update coefficients for specific material
        material.calculate_update_coeffsE(G)
        material.calculate_update_coeffsH(G)

        # Add update coefficients to overall storage for all materials
        G.updatecoeffsE[material.numID, :] = (
            material.CA,
            material.CBx,
            material.CBy,
            material.CBz,
            material.srce,
        )
        G.updatecoeffsH[material.numID, :] = (
            material.DA,
            material.DBx,
            material.DBy,
            material.DBz,
            material.srcm,
        )

        # Add update coefficients to overall storage for dispersive materials
        if hasattr(material, "poles"):
            z = 0
            for pole in range(config.get_model_config().materials["maxpoles"]):
                G.updatecoeffsdispersive[material.numID, z : z + 3] = (
                    config.sim_config.em_consts["e0"] * material.eqt2[pole],
                    material.eqt[pole],
                    material.zt[pole],
                )
                z += 3

        # Construct information on material properties for printing table
        materialtext = [
            str(material.numID),
            material.ID[:50] if len(material.ID) > 50 else material.ID,
            material.type,
            f"{material.er:g}",
            f"{material.se:g}",
        ]
        if config.get_model_config().materials["maxpoles"] > 0:
            if "debye" in material.type:
                materialtext.append("\n".join(f"{deltaer:g}" for deltaer in material.deltaer))
                materialtext.append("\n".join(f"{tau:g}" for tau in material.tau))
                materialtext.extend(["", "", ""])
            elif "lorentz" in material.type:
                materialtext.append(", ".join(f"{deltaer:g}" for deltaer in material.deltaer))
                materialtext.append("")
                materialtext.append(", ".join(f"{tau:g}" for tau in material.tau))
                materialtext.append(", ".join(f"{alpha:g}" for alpha in material.alpha))
                materialtext.append("")
            elif "drude" in material.type:
                materialtext.extend(["", ""])
                materialtext.append(", ".join(f"{tau:g}" for tau in material.tau))
                materialtext.append("")
                materialtext.append(", ".join(f"{alpha:g}" for alpha in material.alpha))
            else:
                materialtext.extend(["", "", "", "", ""])

        materialtext.extend((f"{material.mr:g}", f"{material.sm:g}", material.averagable))
        materialsdata.append(materialtext)

    return materialsdata
