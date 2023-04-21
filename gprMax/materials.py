# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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
import logging


import gprMax.config as config


logger = logging.getLogger(__name__)

class Material:
    """Super-class to describe generic, non-dispersive materials,
        their properties and update coefficients.
    """

    def __init__(self, numID, ID):
        """
        Args:
            numID: int for numeric I of the material.
            ID: string for name of the material.
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

    def calculate_update_coeffsH(self, G):
        """Calculates the magnetic update coefficients of the material.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        HA = (config.m0 * self.mr / G.dt) + 0.5 * self.sm
        HB = (config.m0 * self.mr / G.dt) - 0.5 * self.sm
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

        EA = (config.sim_config.em_consts['e0'] * self.er / G.dt) + 0.5 * self.se
        EB = (config.sim_config.em_consts['e0'] * self.er / G.dt) - 0.5 * self.se

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

    def calculate_update_coeffsE(self, G):
        """Calculates the electric update coefficients of the material.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # The implementation of the dispersive material modelling comes from the
        # derivation in: http://dx.doi.org/10.1109/TAP.2014.2308549
        self.w = np.zeros(config.get_model_config().materials['maxpoles'],
                          dtype=config.get_model_config().materials['dispersivedtype'])
        self.q = np.zeros(config.get_model_config().materials['maxpoles'],
                          dtype=config.get_model_config().materials['dispersivedtype'])
        self.zt = np.zeros(config.get_model_config().materials['maxpoles'],
                           dtype=config.get_model_config().materials['dispersivedtype'])
        self.zt2 = np.zeros(config.get_model_config().materials['maxpoles'],
                            dtype=config.get_model_config().materials['dispersivedtype'])
        self.eqt = np.zeros(config.get_model_config().materials['maxpoles'],
                            dtype=config.get_model_config().materials['dispersivedtype'])
        self.eqt2 = np.zeros(config.get_model_config().materials['maxpoles'],
                            dtype=config.get_model_config().materials['dispersivedtype'])

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

        EA = ((config.sim_config.em_consts['e0'] * self.er / G.dt) + 0.5 * self.se -
             (config.sim_config.em_consts['e0'] / G.dt) * np.sum(self.zt2.real))
        EB = ((config.sim_config.em_consts['e0'] * self.er / G.dt) - 0.5 * self.se -
             (config.sim_config.em_consts['e0'] / G.dt) * np.sum(self.zt2.real))

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
        er += self.se / (1j * w * config.sim_config.em_consts['e0'])
        if 'debye' in self.type:
            for pole in range(self.poles):
                er += self.deltaer[pole] / (1 + 1j * w * self.tau[pole])
        elif 'lorentz' in self.type:
            for pole in range(self.poles):
                er += ((self.deltaer[pole] * self.tau[pole]**2)
                       / (self.tau[pole]**2 + 2j * w * self.alpha[pole] - w**2))
        elif 'drude' in self.type:
            ersum = 0
            for pole in range(self.poles):
                ersum += self.tau[pole]**2 / (w**2 - 1j * w * self.alpha[pole])
                er -= ersum

        return er


def process_materials(G):
    """Processes complete list of materials - calculates update coefficients,
        stores in arrays, and builds text list of materials/properties

    Args:
        G: FDTDGrid class describing a grid in a model.

    Returns:
        materialsdata: list of material IDs, names, and properties to
                        print a table.
    """

    if config.get_model_config().materials['maxpoles'] == 0:
        materialsdata = [['\nID', '\nName', '\nType', '\neps_r', 'sigma\n[S/m]',
                          '\nmu_r', 'sigma*\n[Ohm/m]', 'Dielectric\nsmoothable']]
    else:
        materialsdata = [['\nID', '\nName', '\nType', '\neps_r', 'sigma\n[S/m]',
                          'Delta\neps_r', 'tau\n[s]', 'omega\n[Hz]', 'delta\n[Hz]',
                          'gamma\n[Hz]', '\nmu_r', 'sigma*\n[Ohm/m]', 'Dielectric\nsmoothable']]

    for material in G.materials:
        # Calculate update coefficients for specific material
        material.calculate_update_coeffsE(G)
        material.calculate_update_coeffsH(G)

        # Add update coefficients to overall storage for all materials
        G.updatecoeffsE[material.numID, :] = material.CA, material.CBx, material.CBy, material.CBz, material.srce
        G.updatecoeffsH[material.numID, :] = material.DA, material.DBx, material.DBy, material.DBz, material.srcm

        # Add update coefficients to overall storage for dispersive materials
        if hasattr(material, 'poles'):
            z = 0
            for pole in range(config.get_model_config().materials['maxpoles']):
                G.updatecoeffsdispersive[material.numID, z:z + 3] = (config.sim_config.em_consts['e0'] *
                                                                     material.eqt2[pole], material.eqt[pole], material.zt[pole])
                z += 3

        # Construct information on material properties for printing table
        materialtext = []
        materialtext.append(str(material.numID))
        materialtext.append(material.ID[:50] if len(material.ID) > 50 else material.ID)
        materialtext.append(material.type)
        materialtext.append(f'{material.er:g}')
        materialtext.append(f'{material.se:g}')
        if config.get_model_config().materials['maxpoles'] > 0:
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

        materialtext.append(f'{material.mr:g}')
        materialtext.append(f'{material.sm:g}')
        materialtext.append(material.averagable)
        materialsdata.append(materialtext)

    return materialsdata


class PeplinskiSoil:
    """Soil objects that are characterised according to a mixing model
        by Peplinski (http://dx.doi.org/10.1109/36.387598).
    """

    def __init__(self, ID, sandfraction, clayfraction, bulkdensity, 
                 sandpartdensity, watervolfraction):
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
        self.startmaterialnum = 0 #This is not used anymore and code that uses it can be removed
        # store all of the material IDs in a list instead of storing only the first number of the material 
        # and assume that all must be sequentially numbered. This allows for more general mixing models
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
        erealw = watereri + ((waterer - watereri) / (1 + (w * watertau)**2))

        a = 0.65  # Experimentally derived constant
        es = (1.01 + 0.44 * self.rs)**2 - 0.062  #  Relative permittivity of sand particles
        b1 = 1.2748 - 0.519 * self.S - 0.152 * self.C
        b2 = 1.33797 - 0.603 * self.S - 0.166 * self.C

        # For frequencies in the range 0.3GHz to 1.3GHz
        sigf = 0.0467 + 0.2204 * self.rb - 0.411 * self.S + 0.6614 * self.C
        # For frequencies in the range 1.4GHz to 18GHz
        # sigf = -1.645 + 1.939 * self.rb - 2.25622 * self.S + 1.594 * self.C

        # Generate a set of bins based on the given volumetric water fraction 
        # values. Changed to make sure mid points are contained completely within the ranges. 
        # The limiting values of the ranges are not included in this.

        #mubins = np.linspace(self.mu[0], self.mu[1], nbins)
        mubins = np.linspace(self.mu[0], self.mu[1], nbins+1)
        # Generate a range of volumetric water fraction values the mid-point of 
        # each bin to make materials from
        #mumaterials = mubins + (mubins[1] - mubins[0]) / 2
        mumaterials = 0.5*(mubins[1:nbins+1] + mubins[0:nbins])


        # Create an iterator
        muiter = np.nditer(mumaterials, flags=['c_index'])
        while not muiter.finished:
            # Real part for frequencies in the range 1.4GHz to 18GHz
            er = (1 + (self.rb / self.rs) * ((es**a) - 1) + (muiter[0]**b1 * erealw**a)
                  - muiter[0]) ** (1 / a)
            # Real part for frequencies in the range 0.3GHz to 1.3GHz (linear 
            # correction to 1.4-18GHz value)
            er = 1.15 * er - 0.68

            # Permittivity at infinite frequency
            eri = er - (muiter[0]**(b2 / a) * DispersiveMaterial.waterdeltaer)

            # Effective conductivity
            sig = muiter[0]**(b2 / a) * ((sigf * (self.rs - self.rb)) / (self.rs * muiter[0]))

            # Check to see if the material already exists before creating a new one
            requiredID = '|{:.4f}|'.format(float(muiter[0]))
            material = next((x for x in G.materials if x.ID == requiredID), None)
            if muiter.index == 0:
                if material:
                    self.startmaterialnum = material.numID
                    self.matID.append(material.numID)                    
                else:
                    self.startmaterialnum = len(G.materials)                
            if not material:
                m = DispersiveMaterial(len(G.materials), requiredID)
                m.type = 'debye'
                m.averagable = False
                m.poles = 1
                if m.poles > config.get_model_config().materials['maxpoles']:
                    config.get_model_config().materials['maxpoles'] = m.poles
                m.er = eri
                m.se = sig
                m.deltaer.append(er - eri)
                m.tau.append(DispersiveMaterial.watertau)
                G.materials.append(m)
                self.matID.append(m.numID)

            muiter.iternext()
      


class RangeMaterial:
    """Material objects defined by a given range of their parameters to be used for 
       factal spatial disttibutions. 
    """

    def __init__(self, ID, er_range, se_range, mr_range, sm_range):
        """
        Args:
            ID: string for name of the material range.
            er_range: tuple of floats for relative permittivity range of the materials.
            se_range: tuple of floats for electric conductivity range of the materials.
            mr_range: tuple of floats for magnetic permeability of materials. 
            sm_range: tuple of floats for magnetic loss range of materials. 
        """

        self.ID = ID
        self.er = er_range
        self.sig = se_range
        self.mu = mr_range 
        self.ro = sm_range
        self.startmaterialnum = 0 #This is not really needed anymore and code that uses it can be removed.
         # store all of the material IDs in a list instead of storing only the first number of the material 
        # and assume that all must be sequentially numbered. This allows for more general mixing models
        self.matID = [] 
       

    def calculate_properties(self, nbins, G):
        """Calculates the specific properties of each of the materials. 

        Args:
            nbins: int for number of bins to use to create the different materials.
            G: FDTDGrid class describing a grid in a model.
        """

        # Generate a set of relative permittivity bins based on the given range
        erbins = np.linspace(self.er[0], self.er[1], nbins+1)
        # Generate a range of relative permittivity values the mid-point of 
        # each bin to make materials from
        #ermaterials = erbins + np.abs((erbins[1] - erbins[0])) / 2
        ermaterials = 0.5*(erbins[1:nbins+1] + erbins[0:nbins])

        # Generate a set of conductivity bins based on the given range
        sigmabins = np.linspace(self.sig[0], self.sig[1], nbins+1)
        # Generate a range of conductivity values the mid-point of 
        # each bin to make materials from
        #sigmamaterials = sigmabins + (sigmabins[1] - sigmabins[0]) / 2
        sigmamaterials = 0.5*(sigmabins[1:nbins+1] + sigmabins[0:nbins])

        # Generate a set of magnetic permeability bins based on the given range
        mubins = np.linspace(self.mu[0], self.mu[1], nbins+1)
        # Generate a range of magnetic permeability values the mid-point of 
        # each bin to make materials from
        #mumaterials = mubins + np.abs((mubins[1] - mubins[0])) / 2
        mumaterials = 0.5*(mubins[1:nbins+1] + mubins[0:nbins])
        
        # Generate a set of magnetic loss bins based on the given range
        robins = np.linspace(self.ro[0], self.ro[1], nbins+1)
        # Generate a range of magnetic loss values the mid-point of 
        # each bin to make materials from
        #romaterials = robins + np.abs((robins[1] - robins[0])) / 2
        romaterials = 0.5*(robins[1:nbins+1] + robins[0:nbins])


        # Iterate over the bins
        for iter in np.arange(0,nbins):
        
            # Relative permittivity 
            er = ermaterials[iter]

            # Effective conductivity
            se = sigmamaterials[iter] 

            # magnetic permeability
            mr = mumaterials[iter]

            # magnetic loss
            sm = romaterials[iter]

            # Check to see if the material already exists before creating a new one
            requiredID = '|{:.4f}+{:.4f}+{:.4f}+{:.4f}|'.format(float(er),float(se),float(mr),float(sm))
            material = next((x for x in G.materials if x.ID == requiredID), None)
            if iter == 0:
                if material:
                    self.startmaterialnum = material.numID
                    self.matID.append(material.numID)
                else:
                    self.startmaterialnum = len(G.materials)
            if not material:
                m = Material(len(G.materials), requiredID)
                m.type = ''
                m.averagable = True
                m.er = er
                m.se = se
                m.mr = mr
                m.sm = sm
                G.materials.append(m)
                self.matID.append(m.numID)



class ListMaterial:
    """A list of predefined materials that are to be used for 
       factal spatial disttibutions. No new materials are created but the ones specified are
       grouped together to be used in fractal spatial distributions.
    """

    def __init__(self, ID, listofmaterials):
        """
        Args:
            ID: string for name of the material list.
            listofmaterials: A list of material IDs.
            
        """

        self.ID = ID
        self.mat = listofmaterials
        self.startmaterialnum = 0 #This is not really needed anymore
        # store all of the material IDs in a list instead of storing only the first number of the material 
        # and assume that all must be sequentially numbered. This allows for more general mixing models
        # this is important here as this model assumes predefined materials.
        self.matID = [] 
        

    def calculate_properties(self, nbins, G):
        """Calculates the properties of the materials. No Debye is used but name kept the same as used in other 
           class that needs Debye

        Args:
            nbins: int for number of bins to use to create the different materials.
            G: FDTDGrid class describing a grid in a model.
        """

       
        # Iterate over the bins
        for iter in np.arange(0,nbins):
        
            # Check to see if the material already exists before creating a new one
            #requiredID = '|{:}_in_{:}|'.format((self.mat[iter]),(self.ID))
            requiredID = self.mat[iter]
            material = next((x for x in G.materials if x.ID == requiredID), None)
            self.matID.append(material.numID)
 
            #if iter == 0:
            #    if material:
            #        self.startmaterialnum = material.numID
            #    else:
            #        self.startmaterialnum = len(G.materials)

            #if not material:
            #    temp = next((x for x in G.materials if x.ID == self.mat[iter]), None) 
            #    m = copy.deepcopy(temp) #This needs to import copy in order to work
            #    m.ID = requiredID
            #    m.numID = len(G.materials)
            #    G.materials.append(m)
       
            
            if not material:
                logger.exception(self.__str__() + f' material(s) {material} do not exist')
                raise ValueError
            
            



def create_built_in_materials(G):
    """Creates pre-defined (built-in) materials.

    Args:
        G: FDTDGrid class describing a grid in a model.
    """

    G.n_built_in_materials = len(G.materials)

    m = Material(0, 'pec')
    m.se = float('inf')
    m.type = 'builtin'
    m.averagable = False
    G.materials.append(m)

    m = Material(1, 'free_space')
    m.type = 'builtin'
    G.materials.append(m)

    G.n_built_in_materials = len(G.materials)


def calculate_water_properties(T=25, S=0):
    """Get extended Debye model properties for water.

    Args:
        T: float for emperature of water (degrees centigrade).
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
    tau = (1 / (2 * np.pi)) * (1.1109e-10 - 3.824e-12 * T + 6.938e-14 * T**2 - 
                               5.096e-16 * T**3)

    delta = 25 - T
    beta = (2.033e-2 + 1.266e-4 * delta + 2.464e-6 * delta**2 - S * 
           (1.849e-5 - 2.551e-7 * delta + 2.551e-8 * delta**2))
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
    
    G.n_built_in_materials = len(G.materials)

    m = DispersiveMaterial(len(G.materials), 'water')
    m.averagable = False
    m.type = 'builtin, debye'
    m.poles = 1
    m.er = eri
    m.se = sig
    m.deltaer.append(er - eri)
    m.tau.append(tau)
    G.materials.append(m)
    if config.get_model_config().materials['maxpoles'] == 0:
        config.get_model_config().materials['maxpoles'] = 1

    G.n_built_in_materials = len(G.materials)


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

    G.n_built_in_materials = len(G.materials)

    m = DispersiveMaterial(len(G.materials), 'grass')
    m.averagable = False
    m.type = 'builtin, debye'
    m.poles = 1
    m.er = eri
    m.se = sig
    m.deltaer.append(er - eri)
    m.tau.append(tau)
    G.materials.append(m)
    if config.get_model_config().materials['maxpoles'] == 0:
        config.get_model_config().materials['maxpoles'] = 1

    G.n_built_in_materials = len(G.materials)