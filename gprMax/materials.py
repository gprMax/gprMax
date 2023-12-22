# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

from gprMax.constants import e0
from gprMax.constants import m0
from gprMax.constants import complextype


class Material(object):
    """Materials, their properties and update coefficients."""

    # Maximum number of dispersive material poles in a model
    maxpoles = 0

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
        """
        Args:
            numID (int): Numeric identifier of the material.
            ID (str): Name of the material.
        """

        self.numID = numID
        self.ID = ID
        self.type = ''
        # Default material averaging
        self.averagable = True

        # Default material constitutive parameters (free_space)
        self.er = 1.0
        self.se = 0.0
        self.mr = 1.0
        self.sm = 0.0

        # Parameters for dispersive materials
        self.poles = 0
        self.deltaer = []
        self.tau = []
        self.alpha = []

    def calculate_update_coeffsH(self, G):
        """Calculates the magnetic update coefficients of the material.

        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        HA = (m0 * self.mr / G.dt) + 0.5 * self.sm
        HB = (m0 * self.mr / G.dt) - 0.5 * self.sm
        self.DA = HB / HA
        self.DBx = (1 / G.dx) * 1 / HA
        self.DBy = (1 / G.dy) * 1 / HA
        self.DBz = (1 / G.dz) * 1 / HA
        self.srcm = 1 / HA

    def calculate_update_coeffsE(self, G):
        """Calculates the electric update coefficients of the material.

        Args:
            G (class): Grid class instance - holds essential parameters
                    describing the model.
        """

        # The implementation of the dispersive material modelling comes from the
        # derivation in: http://dx.doi.org/10.1109/TAP.2014.2308549
        if self.maxpoles > 0:
            self.w = np.zeros(self.maxpoles, dtype=complextype)
            self.q = np.zeros(self.maxpoles, dtype=complextype)
            self.zt = np.zeros(self.maxpoles, dtype=complextype)
            self.zt2 = np.zeros(self.maxpoles, dtype=complextype)
            self.eqt = np.zeros(self.maxpoles, dtype=complextype)
            self.eqt2 = np.zeros(self.maxpoles, dtype=complextype)

            for x in range(self.poles):
                if 'debye' in self.type:
                    self.w[x] = self.deltaer[x] / self.tau[x]
                    self.q[x] = -1 / self.tau[x]
                elif 'lorentz' in self.type:
                    # tau for Lorentz materials are pole frequencies
                    # alpha for Lorentz materials are the damping coefficients
                    wp2 = (2 * np.pi * self.tau[x])**2
                    self.w[x] = -1j * ((wp2 * self.deltaer[x]) / np.sqrt(wp2 - self.alpha[x]**2))
                    self.q[x] = -self.alpha[x] + (1j * np.sqrt(wp2 - self.alpha[x]**2))
                elif 'drude' in self.type:
                    # tau for Drude materials are pole frequencies
                    # alpha for Drude materials are the inverse of relaxation times
                    wp2 = (2 * np.pi * self.tau[x])**2
                    self.se += wp2 / self.alpha[x]
                    self.w[x] = - (wp2 / self.alpha[x])
                    self.q[x] = - self.alpha[x]

                self.eqt[x] = np.exp(self.q[x] * G.dt)
                self.eqt2[x] = np.exp(self.q[x] * (G.dt / 2))
                self.zt[x] = (self.w[x] / self.q[x]) * (1 - self.eqt[x]) / G.dt
                self.zt2[x] = (self.w[x] / self.q[x]) * (1 - self.eqt2[x])

            EA = (e0 * self.er / G.dt) + 0.5 * self.se - (e0 / G.dt) * np.sum(self.zt2.real)
            EB = (e0 * self.er / G.dt) - 0.5 * self.se - (e0 / G.dt) * np.sum(self.zt2.real)

        else:
            EA = (e0 * self.er / G.dt) + 0.5 * self.se
            EB = (e0 * self.er / G.dt) - 0.5 * self.se

        if self.ID == 'pec' or self.se == float('inf'):
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
        """
        Calculates the complex relative permittivity of the material at a specific frequency.

        Args:
            freq (float): Frequency used to calculate complex relative permittivity.

        Returns:
            er (float): Complex relative permittivity.
        """

        # Permittivity at infinite frequency if the material is dispersive
        er = self.er

        if self.poles > 0:
            w = 2 * np.pi * freq
            er += self.se / (1j * w * e0)
            if 'debye' in self.type:
                for pole in range(self.poles):
                    er += self.deltaer[pole] / (1 + 1j * w * self.tau[pole])
            elif 'lorentz' in self.type:
                for pole in range(self.poles):
                    er += (self.deltaer[pole] * self.tau[pole]**2) / (self.tau[pole]**2 + 2j * w * self.alpha[pole] - w**2)
            elif 'drude' in self.type:
                ersum = 0
                for pole in range(self.poles):
                    ersum += self.tau[pole]**2 / (w**2 - 1j * w * self.alpha[pole])
                    er -= ersum

        return er


def process_materials(G):
    """
    Process complete list of materials - calculate update coefficients,
        store in arrays, and build text list of materials/properties

    Args:
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        materialsdata (list): List of material IDs, names, and properties to print a table.
    """

    if Material.maxpoles == 0:
        materialsdata = [['\nID', '\nName', '\nType', '\neps_r', 'sigma\n[S/m]', '\nmu_r', 'sigma*\n[Ohm/m]', 'Dielectric\nsmoothable']]
    else:
        materialsdata = [['\nID', '\nName', '\nType', '\neps_r', 'sigma\n[S/m]', 'Delta\neps_r', 'tau\n[s]', 'omega\n[Hz]', 'delta\n[Hz]', 'gamma\n[Hz]', '\nmu_r', 'sigma*\n[Ohm/m]', 'Dielectric\nsmoothable']]

    for material in G.materials:
        # Calculate update coefficients for material
        material.calculate_update_coeffsE(G)
        material.calculate_update_coeffsH(G)

        # Store all update coefficients together
        G.updatecoeffsE[material.numID, :] = material.CA, material.CBx, material.CBy, material.CBz, material.srce
        G.updatecoeffsH[material.numID, :] = material.DA, material.DBx, material.DBy, material.DBz, material.srcm

        # Store coefficients for any dispersive materials
        if Material.maxpoles > 0:
            z = 0
            for pole in range(Material.maxpoles):
                G.updatecoeffsdispersive[material.numID, z:z + 3] = e0 * material.eqt2[pole], material.eqt[pole], material.zt[pole]
                z += 3

        # Construct information on material properties for printing table
        materialtext = []
        materialtext.append(str(material.numID))
        materialtext.append(material.ID[:50] if len(material.ID) > 50 else material.ID)
        materialtext.append(material.type)
        materialtext.append('{:g}'.format(material.er))
        materialtext.append('{:g}'.format(material.se))
        if Material.maxpoles > 0:
            if 'debye' in material.type:
                materialtext.append('\n'.join('{:g}'.format(deltaer) for deltaer in material.deltaer))
                materialtext.append('\n'.join('{:g}'.format(tau) for tau in material.tau))
                materialtext.extend(['', '', ''])
            elif 'lorentz' in material.type:
                materialtext.append(', '.join('{:g}'.format(deltaer) for deltaer in material.deltaer))
                materialtext.append('')
                materialtext.append(', '.join('{:g}'.format(tau) for tau in material.tau))
                materialtext.append(', '.join('{:g}'.format(alpha) for alpha in material.alpha))
                materialtext.append('')
            elif 'drude' in material.type:
                materialtext.extend(['', ''])
                materialtext.append(', '.join('{:g}'.format(tau) for tau in material.tau))
                materialtext.append('')
                materialtext.append(', '.join('{:g}'.format(alpha) for alpha in material.alpha))
            else:
                materialtext.extend(['', '', '', '', ''])

        materialtext.append('{:g}'.format(material.mr))
        materialtext.append('{:g}'.format(material.sm))
        materialtext.append(material.averagable)
        materialsdata.append(materialtext)

    return materialsdata


class PeplinskiSoil(object):
    """
    Soil objects that are characterised according to a mixing
    model by Peplinski (http://dx.doi.org/10.1109/36.387598).
    """

    def __init__(self, ID, sandfraction, clayfraction, bulkdensity, sandpartdensity, watervolfraction):
        """
        Args:
            ID (str): Name of the soil.
            sandfraction (float): Sand fraction of the soil.
            clayfraction (float): Clay fraction of the soil.
            bulkdensity (float): Bulk density of the soil (g/cm3).
            sandpartdensity (float): Density of the sand particles in the soil (g/cm3).
            watervolfraction (float): Two numbers that specify a range for the volumetric water fraction of the soil.
        """

        self.ID = ID
        self.S = sandfraction
        self.C = clayfraction
        self.rb = bulkdensity
        self.rs = sandpartdensity
        self.mu = watervolfraction
        self.startmaterialnum = 0

    def calculate_debye_properties(self, nbins, G, fractalboxname):
        """
        Calculates the real and imaginery part of a Debye model for the soil as
        well as a conductivity. It uses an approximation to a semi-empirical model (http://dx.doi.org/10.1109/36.387598).

        Args:
            nbins (int): Number of bins to use to create the different materials.
            G (class): Grid class instance - holds essential parameters describing the model.
            fractalboxname (str): Name of the fractal box for which the materials are being created.
        """
        self.startmaterialnum = len(G.materials)

        # Debye model properties of water
        f = 1.3e9
        w = 2 * np.pi * f
        erealw = Material.watereri + ((Material.waterdeltaer) / (1 + (w * Material.watertau)**2))

        a = 0.65  # Experimentally derived constant
        es = (1.01 + 0.44 * self.rs)**2 - 0.062  #  Relative permittivity of sand particles
        b1 = 1.2748 - 0.519 * self.S - 0.152 * self.C
        b2 = 1.33797 - 0.603 * self.S - 0.166 * self.C

        # For frequencies in the range 0.3GHz to 1.3GHz
        sigf = 0.0467 + 0.2204 * self.rb - 0.411 * self.S + 0.6614 * self.C
        # For frequencies in the range 1.4GHz to 18GHz
        # sigf = -1.645 + 1.939 * self.rb - 2.25622 * self.S + 1.594 * self.C

        # Generate a set of bins based on the given volumetric water fraction values
        mubins = np.linspace(self.mu[0], self.mu[1], nbins)
        # Generate a range of volumetric water fraction values the mid-point of each bin to make materials from
        mumaterials = mubins + (mubins[1] - mubins[0]) / 2

        # Create an iterator
        muiter = np.nditer(mumaterials, flags=['c_index'])
        while not muiter.finished:
            # Real part for frequencies in the range 1.4GHz to 18GHz
            er = (1 + (self.rb / self.rs) * ((es**a) - 1) + (muiter[0]**b1 * erealw**a) - muiter[0]) ** (1 / a)
            # Real part for frequencies in the range 0.3GHz to 1.3GHz (linear correction to 1.4-18GHz value)
            er = 1.15 * er - 0.68

            # Permittivity at infinite frequency
            eri = er - (muiter[0]**(b2 / a) * Material.waterdeltaer)

            # Effective conductivity
            sig = muiter[0]**(b2 / a) * ((sigf * (self.rs - self.rb)) / (self.rs * muiter[0]))

            # Check to see if the material already exists before creating a new one
            materialID = '|{:.4f}_{}|'.format(float(muiter[0]), fractalboxname)
            m = Material(len(G.materials), materialID)
            m.type = 'debye'
            m.averagable = False
            m.poles = 1
            if m.poles > Material.maxpoles:
                Material.maxpoles = m.poles
            m.er = eri
            m.se = sig
            m.deltaer.append(er - eri)
            m.tau.append(Material.watertau)
            G.materials.append(m)

            muiter.iternext()
