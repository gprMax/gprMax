# Copyright (C) 2015-2016: The University of Edinburgh
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

from gprMax.constants import e0, z0, floattype
from gprMax.pml_1order_update import update_pml_1order_ey_xminus, update_pml_1order_ez_xminus, update_pml_1order_ey_xplus, update_pml_1order_ez_xplus, update_pml_1order_ex_yminus, update_pml_1order_ez_yminus, update_pml_1order_ex_yplus, update_pml_1order_ez_yplus, update_pml_1order_ex_zminus, update_pml_1order_ey_zminus, update_pml_1order_ex_zplus, update_pml_1order_ey_zplus, update_pml_1order_hy_xminus, update_pml_1order_hz_xminus, update_pml_1order_hy_xplus, update_pml_1order_hz_xplus, update_pml_1order_hx_yminus, update_pml_1order_hz_yminus, update_pml_1order_hx_yplus, update_pml_1order_hz_yplus, update_pml_1order_hx_zminus, update_pml_1order_hy_zminus, update_pml_1order_hx_zplus, update_pml_1order_hy_zplus
from gprMax.pml_2order_update import update_pml_2order_ey_xminus, update_pml_2order_ez_xminus, update_pml_2order_ey_xplus, update_pml_2order_ez_xplus, update_pml_2order_ex_yminus, update_pml_2order_ez_yminus, update_pml_2order_ex_yplus, update_pml_2order_ez_yplus, update_pml_2order_ex_zminus, update_pml_2order_ey_zminus, update_pml_2order_ex_zplus, update_pml_2order_ey_zplus, update_pml_2order_hy_xminus, update_pml_2order_hz_xminus, update_pml_2order_hy_xplus, update_pml_2order_hz_xplus, update_pml_2order_hx_yminus, update_pml_2order_hz_yminus, update_pml_2order_hx_yplus, update_pml_2order_hz_yplus, update_pml_2order_hx_zminus, update_pml_2order_hy_zminus, update_pml_2order_hx_zplus, update_pml_2order_hy_zplus


class CFSParameter(object):
    """Individual CFS parameter (e.g. alpha, kappa, or sigma)."""

    # Allowable scaling profiles and directions
    scalingprofiles = {'constant': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4, 'quintic': 5, 'sextic': 6}
    scalingdirections = ['forward', 'reverse']

    def __init__(self, ID=None, scaling='polynomial', scalingprofile=None, scalingdirection='forward', min=0, max=0):
        """
        Args:
            ID (str): Identifier for CFS parameter, can be: 'alpha', 'kappa' or 'sigma'.
            scaling (str): Type of scaling, can be: 'polynomial'.
            scalingprofile (str): Type of scaling profile from scalingprofiles.
            scalingdirection (str): Direction of scaling profile from scalingdirections.
            min (float): Minimum value for parameter.
            max (float): Maximum value for parameter.
        """

        self.ID = ID
        self.scaling = scaling
        self.scalingprofile = scalingprofile
        self.scalingdirection = scalingdirection
        self.min = min
        self.max = max


class CFS(object):
    """CFS term for PML."""

    def __init__(self):
        """
        Args:
            alpha (CFSParameter): alpha parameter for CFS.
            kappa (CFSParameter): kappa parameter for CFS.
            sigma (CFSParameter): sigma parameter for CFS.
        """

        self.alpha = CFSParameter(ID='alpha', scalingprofile='constant')
        self.kappa = CFSParameter(ID='kappa', scalingprofile='constant', min=1, max=1)
        self.sigma = CFSParameter(ID='sigma', scalingprofile='quartic', min=0, max=None)

    def calculate_sigmamax(self, direction, er, mr, G):
        """Calculates an optimum value for sigma max based on underlying material properties.

        Args:
            direction (str): Direction of PML slab
            er (float): Average permittivity of underlying material.
            mr (float): Average permeability of underlying material.
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        if direction[0] == 'x':
            d = G.dx
        elif direction[0] == 'y':
            d = G.dy
        elif direction[0] == 'z':
            d = G.dz

        # Calculation of the maximum value of sigma from http://dx.doi.org/10.1109/8.546249
        m = CFSParameter.scalingprofiles[self.sigma.scalingprofile]
        self.sigma.max = (0.8 * (m + 1)) / (z0 * d * np.sqrt(er * mr))

    def scaling_polynomial(self, order, Evalues, Hvalues):
        """Applies the polynomial to be used for the scaling profile for electric and magnetic PML updates.

        Args:
            order (int): Order of polynomial for scaling profile.
            Evalues (float): numpy array holding scaling profile values for electric PML update.
            Hvalues (float): numpy array holding scaling profile values for magnetic PML update.

        Returns:
            Evalues (float): numpy array holding scaling profile values for electric PML update.
            Hvalues (float): numpy array holding scaling profile values for magnetic PML update.
        """

        tmp = (np.linspace(0, (len(Evalues) - 1) + 0.5, num=2 * len(Evalues)) / (len(Evalues) - 1)) ** order
        Evalues = tmp[0:-1:2]
        Hvalues = tmp[1::2]
        return Evalues, Hvalues

    def calculate_values(self, thickness, parameter):
        """Calculates values for electric and magnetic PML updates based on profile type and minimum and maximum values.

        Args:
            thickness (int): Thickness of PML in cells.
            parameter (CFSParameter): Instance of CFSParameter

        Returns:
            Evalues (float): numpy array holding profile value for electric PML update.
            Hvalues (float): numpy array holding profile value for magnetic PML update.
        """

        Evalues = np.zeros(thickness + 1, dtype=floattype)
        Hvalues = np.zeros(thickness + 1, dtype=floattype)

        if parameter.scalingprofile == 'constant':
            Evalues += parameter.max
            Hvalues += parameter.max

        elif parameter.scaling == 'polynomial':
            Evalues, Hvalues = self.scaling_polynomial(CFSParameter.scalingprofiles[parameter.scalingprofile], Evalues, Hvalues)
            if parameter.ID == 'alpha':
                pass
            elif parameter.ID == 'kappa':
                Evalues = Evalues * (self.kappa.max - 1) + 1
                Hvalues = Hvalues * (self.kappa.max - 1) + 1
            elif parameter.ID == 'sigma':
                Evalues *= self.sigma.max
                Hvalues *= self.sigma.max

        if parameter.scalingdirection == 'reverse':
            Evalues = Evalues[::-1]
            Hvalues = Hvalues[::-1]

        return Evalues, Hvalues


class PML(object):
    """PML - the implementation comes from the derivation in: http://dx.doi.org/10.1109/TAP.2011.2180344"""

    directions = {0: 'xminus', 1: 'yminus', 2: 'zminus', 3: 'xplus', 4: 'yplus', 5: 'zplus'}

    def __init__(self, G, direction=None, xs=0, xf=0, ys=0, yf=0, zs=0, zf=0):
        """
        Args:
            xs, xf, ys, yf, zs, zf (float): Extent of the PML volume.
            cfs (list): CFS class instances associated with the PML.
        """

        self.direction = direction
        self.xs = xs
        self.xf = xf
        self.ys = ys
        self.yf = yf
        self.zs = zs
        self.zf = zf
        self.nx = xf - xs
        self.ny = yf - ys
        self.nz = zf - zs
        self.CFS = G.cfs
        if not self.CFS:
            self.CFS = [CFS()]
        self.initialise_field_arrays()

    def initialise_field_arrays(self):
        """Initialise arrays to store fields in PML."""

        # Subscript notation, e.g. 'EPhiyxz' means the electric field Phi vector, of which the
        # component being corrected is y, the stretching direction is x, and field derivative
        # is z direction.
        if self.direction[0] == 'x':
            self.thickness = self.nx
            self.EPhiyxz = np.zeros((len(self.CFS), self.nx + 1, self.ny, self.nz + 1), dtype=floattype)
            self.EPhizxy = np.zeros((len(self.CFS), self.nx + 1, self.ny + 1, self.nz), dtype=floattype)
            self.HPhiyxz = np.zeros((len(self.CFS), self.nx, self.ny + 1, self.nz), dtype=floattype)
            self.HPhizxy = np.zeros((len(self.CFS), self.nx, self.ny, self.nz + 1), dtype=floattype)
        elif self.direction[0] == 'y':
            self.thickness = self.ny
            self.EPhixyz = np.zeros((len(self.CFS), self.nx, self.ny + 1, self.nz + 1), dtype=floattype)
            self.EPhizyx = np.zeros((len(self.CFS), self.nx + 1, self.ny + 1, self.nz), dtype=floattype)
            self.HPhixyz = np.zeros((len(self.CFS), self.nx + 1, self.ny, self.nz), dtype=floattype)
            self.HPhizyx = np.zeros((len(self.CFS), self.nx, self.ny, self.nz + 1), dtype=floattype)
        elif self.direction[0] == 'z':
            self.thickness = self.nz
            self.EPhixzy = np.zeros((len(self.CFS), self.nx, self.ny + 1, self.nz + 1), dtype=floattype)
            self.EPhiyzx = np.zeros((len(self.CFS), self.nx + 1, self.ny, self.nz + 1), dtype=floattype)
            self.HPhixzy = np.zeros((len(self.CFS), self.nx + 1, self.ny, self.nz), dtype=floattype)
            self.HPhiyzx = np.zeros((len(self.CFS), self.nx, self.ny + 1, self.nz), dtype=floattype)

    def calculate_update_coeffs(self, er, mr, G):
        """Calculates electric and magnetic update coefficients for the PML.

        Args:
            er (float): Average permittivity of underlying material
            mr (float): Average permeability of underlying material
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        self.ERA = np.zeros((len(self.CFS), self.thickness + 1), dtype=floattype)
        self.ERB = np.zeros((len(self.CFS), self.thickness + 1), dtype=floattype)
        self.ERE = np.zeros((len(self.CFS), self.thickness + 1), dtype=floattype)
        self.ERF = np.zeros((len(self.CFS), self.thickness + 1), dtype=floattype)
        self.HRA = np.zeros((len(self.CFS), self.thickness + 1), dtype=floattype)
        self.HRB = np.zeros((len(self.CFS), self.thickness + 1), dtype=floattype)
        self.HRE = np.zeros((len(self.CFS), self.thickness + 1), dtype=floattype)
        self.HRF = np.zeros((len(self.CFS), self.thickness + 1), dtype=floattype)

        for x, cfs in enumerate(self.CFS):
            if not cfs.sigma.max:
                cfs.calculate_sigmamax(self.direction, er, mr, G)
            Ealpha, Halpha = cfs.calculate_values(self.thickness, cfs.alpha)
            Ekappa, Hkappa = cfs.calculate_values(self.thickness, cfs.kappa)
            Esigma, Hsigma = cfs.calculate_values(self.thickness, cfs.sigma)

            # Electric PML update coefficients
            tmp = (2 * e0 * Ekappa) + G.dt * (Ealpha * Ekappa + Esigma)
            self.ERA[x, :] = (2 * e0 + G.dt * Ealpha) / tmp
            self.ERB[x, :] = (2 * e0 * Ekappa) / tmp
            self.ERE[x, :] = ((2 * e0 * Ekappa) - G.dt * (Ealpha * Ekappa + Esigma)) / tmp
            self.ERF[x, :] = (2 * Esigma * G.dt) / (Ekappa * tmp)

            # Magnetic PML update coefficients
            tmp = (2 * e0 * Hkappa) + G.dt * (Halpha * Hkappa + Hsigma)
            self.HRA[x, :] = (2 * e0 + G.dt * Halpha) / tmp
            self.HRB[x, :] = (2 * e0 * Hkappa) / tmp
            self.HRE[x, :] = ((2 * e0 * Hkappa) - G.dt * (Halpha * Hkappa + Hsigma)) / tmp
            self.HRF[x, :] = (2 * Hsigma * G.dt) / (Hkappa * tmp)


def build_pmls(G):
    """This function builds instances of the PML and calculates the initial parameters and coefficients including setting profile
        (based on underlying material er and mr from solid array).
    """

    if G.messages:
        print('')

    for index, pmlthickness in enumerate(G.pmlthickness):
        if pmlthickness > 0:
            sumer = 0  # Sum of relative permittivities in PML slab
            summr = 0  # Sum of relative permeabilities in PML slab
            pmldirection = PML.directions[index]

            if pmldirection[0] == 'x':
                if pmldirection == 'xminus':
                    pml = PML(G, direction=pmldirection, xf=pmlthickness, yf=G.ny, zf=G.nz)
                elif pmldirection == 'xplus':
                    pml = PML(G, direction=pmldirection, xs=G.nx - pmlthickness, xf=G.nx, yf=G.ny, zf=G.nz)
                G.pmls.append(pml)
                for j in range(G.ny):
                    for k in range(G.nz):
                        numID = G.solid[pml.xs, j, k]
                        material = next(x for x in G.materials if x.numID == numID)
                        sumer += material.er
                        summr += material.mr
                averageer = sumer / (G.ny * G.nz)
                averagemr = summr / (G.ny * G.nz)

            elif pmldirection[0] == 'y':
                if pmldirection == 'yminus':
                    pml = PML(G, direction=pmldirection, yf=pmlthickness, xf=G.nx, zf=G.nz)
                elif pmldirection == 'yplus':
                    pml = PML(G, direction=pmldirection, ys=G.ny - pmlthickness, xf=G.nx, yf=G.ny, zf=G.nz)
                G.pmls.append(pml)
                for i in range(G.nx):
                    for k in range(G.nz):
                        numID = G.solid[i, pml.ys, k]
                        material = next(x for x in G.materials if x.numID == numID)
                        sumer += material.er
                        summr += material.mr
                averageer = sumer / (G.nx * G.nz)
                averagemr = summr / (G.nx * G.nz)

            elif pmldirection[0] == 'z':
                if pmldirection == 'zminus':
                    pml = PML(G, direction=pmldirection, zf=pmlthickness, xf=G.nx, yf=G.ny)
                elif pmldirection == 'zplus':
                    pml = PML(G, direction=pmldirection, zs=G.nz - pmlthickness, xf=G.nx, yf=G.ny, zf=G.nz)
                G.pmls.append(pml)
                for i in range(G.nx):
                    for j in range(G.ny):
                        numID = G.solid[i, j, pml.zs]
                        material = next(x for x in G.materials if x.numID == numID)
                        sumer += material.er
                        summr += material.mr
                averageer = sumer / (G.nx * G.ny)
                averagemr = summr / (G.nx * G.ny)

            if G.messages and G.pmlthickness.count(pmlthickness) != len(G.pmlthickness):
                print('Absorbing boundary: PML ({} slab) using {:g} cells'.format(pml.direction, pml.thickness))

            pml.calculate_update_coeffs(averageer, averagemr, G)

    # Where all the thicknesses of all the PML slabs are equal
    if G.messages and G.pmlthickness.count(G.pmlthickness[0]) == len(G.pmlthickness):
        if G.pmlthickness[0] == 0:
            print('Absorbing boundaries are all switched off')
        else:
            print('Absorbing boundaries: PML with (all slabs) using {:g} cells'.format(G.pmlthickness[0]))


def update_electric_pml(G):
    """This functions updates electric field components with the PML correction."""

    for pml in G.pmls:
        if len(pml.CFS) == 1:
            if pml.direction == 'xminus':
                update_pml_1order_ey_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hz, pml.EPhiyxz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
                update_pml_1order_ez_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hy, pml.EPhizxy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
            elif pml.direction == 'xplus':
                update_pml_1order_ey_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hz, pml.EPhiyxz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
                update_pml_1order_ez_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hy, pml.EPhizxy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
            elif pml.direction == 'yminus':
                update_pml_1order_ex_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hz, pml.EPhixyz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
                update_pml_1order_ez_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hx, pml.EPhizyx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
            elif pml.direction == 'yplus':
                update_pml_1order_ex_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hz, pml.EPhixyz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
                update_pml_1order_ez_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hx, pml.EPhizyx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
            elif pml.direction == 'zminus':
                update_pml_1order_ex_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hy, pml.EPhixzy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
                update_pml_1order_ey_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hx, pml.EPhiyzx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
            elif pml.direction == 'zplus':
                update_pml_1order_ex_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hy, pml.EPhixzy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
                update_pml_1order_ey_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hx, pml.EPhiyzx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)

        elif len(pml.CFS) == 2:
            if pml.direction == 'xminus':
                update_pml_2order_ey_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hz, pml.EPhiyxz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
                update_pml_2order_ez_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hy, pml.EPhizxy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
            elif pml.direction == 'xplus':
                update_pml_2order_ey_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hz, pml.EPhiyxz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
                update_pml_2order_ez_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hy, pml.EPhizxy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dx)
            elif pml.direction == 'yminus':
                update_pml_2order_ex_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hz, pml.EPhixyz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
                update_pml_2order_ez_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hx, pml.EPhizyx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
            elif pml.direction == 'yplus':
                update_pml_2order_ex_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hz, pml.EPhixyz, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
                update_pml_2order_ez_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ez, G.Hx, pml.EPhizyx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dy)
            elif pml.direction == 'zminus':
                update_pml_2order_ex_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hy, pml.EPhixzy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
                update_pml_2order_ey_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hx, pml.EPhiyzx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
            elif pml.direction == 'zplus':
                update_pml_2order_ex_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Hy, pml.EPhixzy, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)
                update_pml_2order_ey_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsE, G.ID, G.Ey, G.Hx, pml.EPhiyzx, pml.ERA, pml.ERB, pml.ERE, pml.ERF, G.dz)


def update_magnetic_pml(G):
    """This functions updates magnetic field components with the PML correction."""

    for pml in G.pmls:
        if len(pml.CFS) == 1:
            if pml.direction == 'xminus':
                update_pml_1order_hy_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ez, pml.HPhiyxz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
                update_pml_1order_hz_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ey, pml.HPhizxy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
            elif pml.direction == 'xplus':
                update_pml_1order_hy_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ez, pml.HPhiyxz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
                update_pml_1order_hz_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ey, pml.HPhizxy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
            elif pml.direction == 'yminus':
                update_pml_1order_hx_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ez, pml.HPhixyz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
                update_pml_1order_hz_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ex, pml.HPhizyx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
            elif pml.direction == 'yplus':
                update_pml_1order_hx_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ez, pml.HPhixyz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
                update_pml_1order_hz_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ex, pml.HPhizyx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
            elif pml.direction == 'zminus':
                update_pml_1order_hx_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ey, pml.HPhixzy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
                update_pml_1order_hy_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ex, pml.HPhiyzx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
            elif pml.direction == 'zplus':
                update_pml_1order_hx_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ey, pml.HPhixzy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
                update_pml_1order_hy_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ex, pml.HPhiyzx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)

        elif len(pml.CFS) == 2:
            if pml.direction == 'xminus':
                update_pml_2order_hy_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ez, pml.HPhiyxz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
                update_pml_2order_hz_xminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ey, pml.HPhizxy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
            elif pml.direction == 'xplus':
                update_pml_2order_hy_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ez, pml.HPhiyxz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
                update_pml_2order_hz_xplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ey, pml.HPhizxy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dx)
            elif pml.direction == 'yminus':
                update_pml_2order_hx_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ez, pml.HPhixyz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
                update_pml_2order_hz_yminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ex, pml.HPhizyx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
            elif pml.direction == 'yplus':
                update_pml_2order_hx_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ez, pml.HPhixyz, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
                update_pml_2order_hz_yplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hz, G.Ex, pml.HPhizyx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dy)
            elif pml.direction == 'zminus':
                update_pml_2order_hx_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ey, pml.HPhixzy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
                update_pml_2order_hy_zminus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ex, pml.HPhiyzx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
            elif pml.direction == 'zplus':
                update_pml_2order_hx_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hx, G.Ey, pml.HPhixzy, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
                update_pml_2order_hy_zplus(pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf, G.nthreads, G.updatecoeffsH, G.ID, G.Hy, G.Ex, pml.HPhiyzx, pml.HRA, pml.HRB, pml.HRE, pml.HRF, G.dz)
