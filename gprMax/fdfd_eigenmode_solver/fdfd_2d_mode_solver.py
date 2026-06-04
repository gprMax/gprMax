import math

import matplotlib
import numpy as np

import gprMax.config as config

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.sparse import bmat, diags, eye, kron
from scipy.sparse.linalg import eigs


class FDFD_2D_mode_solver:
    """2D vector FDFD mode solver using component-sampled material arrays.

    This solver assumes the supplied material arrays are already sampled at the
    corresponding Yee component locations:

        eps_r_xx -> Ex locations
        eps_r_yy -> Ey locations
        eps_r_zz -> Ez locations
        mu_r_xx  -> Hx locations
        mu_r_yy  -> Hy locations
        mu_r_zz  -> Hz locations

    PEC handling is therefore automatic: electric DOFs are constrained wherever
    the matching electric component material is marked PEC. No PEC side or
    orientation information is required.
    """

    def __init__(
            self,
            frequency,
            dx,
            dy,
            mode_index,
            eps_r_xx,
            eps_r_yy,
            eps_r_zz,
            mu_r_xx,
            mu_r_yy,
            mu_r_zz,
            pec_ex_mask=None,
            pec_ey_mask=None,
            pec_ez_mask=None,
            pmc_hx_mask=None,
            pmc_hy_mask=None,
            pmc_hz_mask=None,
    ):
        self.epsilon0 = config.sim_config.em_consts["e0"]
        self.mu0 = config.sim_config.em_consts["m0"]
        self.c = config.sim_config.em_consts["c"]
        self.eta0 = config.sim_config.em_consts["z0"]
        self.omega = 2 * np.pi * frequency
        self.k0 = self.omega / self.c

        self.frequency = frequency
        self.dx = dx
        self.normalized_dx = self.k0 * dx
        self.dy = dy
        self.normalized_dy = self.k0 * dy

        self.eps_r_xx = self._to_solver_array(eps_r_xx)
        self.eps_r_yy = self._to_solver_array(eps_r_yy)
        self.eps_r_zz = self._to_solver_array(eps_r_zz)
        self.mu_r_xx = self._to_solver_array(mu_r_xx)
        self.mu_r_yy = self._to_solver_array(mu_r_yy)
        self.mu_r_zz = self._to_solver_array(mu_r_zz)
        self.Nx, self.Ny = self.eps_r_xx.shape
        self._validate_material_shapes()
        self._validate_supported_magnetic_materials()

        self.pec_ex_mask = self._component_pec_mask(self.eps_r_xx, pec_ex_mask)
        self.pec_ey_mask = self._component_pec_mask(self.eps_r_yy, pec_ey_mask)
        self.pec_ez_mask = self._component_pec_mask(self.eps_r_zz, pec_ez_mask)
        self.pmc_hx_mask = self._component_pec_mask(self.mu_r_xx, pmc_hx_mask, default=False)
        self.pmc_hy_mask = self._component_pec_mask(self.mu_r_yy, pmc_hy_mask, default=False)
        self.pmc_hz_mask = self._component_pec_mask(self.mu_r_zz, pmc_hz_mask, default=False)
        if np.any(self.pmc_hx_mask) or np.any(self.pmc_hy_mask) or np.any(self.pmc_hz_mask):
            raise NotImplementedError("PMC or magnetic conductor constraints are not supported by this solver.")

        self.free_ex_mask = ~self.pec_ex_mask.ravel(order="F")
        self.free_ey_mask = ~self.pec_ey_mask.ravel(order="F")
        self.free_ez_mask = ~self.pec_ez_mask.ravel(order="F")
        self.free_hx_mask = ~self.pmc_hx_mask.ravel(order="F")
        self.free_hy_mask = ~self.pmc_hy_mask.ravel(order="F")
        self.free_hz_mask = ~self.pmc_hz_mask.ravel(order="F")
        self.free_exy_mask = np.concatenate((self.free_ex_mask, self.free_ey_mask))
        self.free_hxy_mask = np.concatenate((self.free_hx_mask, self.free_hy_mask))

        # Prevent artificial high-index PEC material modes. The masks carry the
        # PEC physics, so the corresponding material entries are harmless
        # finite placeholders for matrix assembly.
        self.eps_r_xx[self.pec_ex_mask] = 1.0 + 0j
        self.eps_r_yy[self.pec_ey_mask] = 1.0 + 0j
        self.eps_r_zz[self.pec_ez_mask] = 1.0 + 0j
        self.mu_r_xx[self.pmc_hx_mask] = 1.0 + 0j
        self.mu_r_yy[self.pmc_hy_mask] = 1.0 + 0j
        self.mu_r_zz[self.pmc_hz_mask] = 1.0 + 0j

        self.mode_index = mode_index
        self.num_modes = self.mode_index + 1
        self.guess = -max(
            self._max_magnitude(arr)
            for arr in [self.eps_r_xx, self.eps_r_yy, self.eps_r_zz,
                        self.mu_r_xx, self.mu_r_yy, self.mu_r_zz]
        )

        self.eigenvalues = None
        self.eigenvectors = None
        self.gammas = None
        self.complex_neff = None
        self.real_neff = None
        self.powers = None
        self.spurious_scores = None
        self.accepted_candidate_indices = None
        self.rejected_candidate_indices = None
        self.unselected_candidate_indices = None

        self._init_operators()

    def _to_solver_array(self, values):
        return np.asarray(values, dtype=np.complex128).copy()

    def _field_to_array(self, field):
        return np.asarray(field).reshape((self.Nx, self.Ny), order="F")

    def _validate_material_shapes(self):
        expected = (self.Nx, self.Ny)
        for name in ("eps_r_yy", "eps_r_zz", "mu_r_xx", "mu_r_yy", "mu_r_zz"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} shape {getattr(self, name).shape} does not match {expected}.")

    def _validate_supported_magnetic_materials(self):
        for name in ("mu_r_xx", "mu_r_yy", "mu_r_zz"):
            values = getattr(self, name)
            if np.any(~np.isfinite(values)):
                raise NotImplementedError("PMC or magnetic conductor constraints are not supported by this solver.")

    def _component_pec_mask(self, values, explicit_mask, default=True):
        mask = np.zeros(values.shape, dtype=bool)
        if default:
            mask |= ~np.isfinite(values)
        if explicit_mask is not None:
            explicit_mask = np.asarray(explicit_mask, dtype=bool)
            if explicit_mask.shape != values.shape:
                raise ValueError(f"PEC mask shape {explicit_mask.shape} does not match component shape {values.shape}.")
            mask |= explicit_mask
        return mask

    @staticmethod
    def component_pec_masks_from_cell_mask(cell_pec_mask):
        """Approximate Yee component PEC masks from a cell-centred PEC mask.

        This helper is only for standalone tests. In gprMax integration the
        preferred source is the already component-sampled material IDs.
        """
        pec = np.asarray(cell_pec_mask, dtype=bool)

        # Ex is tangential on y-normal PEC faces.
        ex_mask = pec.copy()
        ex_mask[:, 1:] |= pec[:, :-1] & ~pec[:, 1:]
        ex_mask[:, :-1] |= pec[:, 1:] & ~pec[:, :-1]

        # Ey is tangential on x-normal PEC faces.
        ey_mask = pec.copy()
        ey_mask[1:, :] |= pec[:-1, :] & ~pec[1:, :]
        ey_mask[:-1, :] |= pec[1:, :] & ~pec[:-1, :]

        # Ez is tangential to every side wall.
        ez_mask = ex_mask | ey_mask

        return ex_mask, ey_mask, ez_mask

    def _init_operators(self):
        def diff_operator(n):
            e = np.ones(n)
            data = np.array([-e, e])
            offsets = np.array([0, 1])
            return diags(data, offsets, shape=(n, n)).tocsr()

        Ix, Iy = eye(self.Nx), eye(self.Ny)
        self.DEX = kron(Iy, diff_operator(self.Nx)) / self.normalized_dx
        self.DEY = kron(diff_operator(self.Ny), Ix) / self.normalized_dy
        self.DHX = -self.DEX.conj().T
        self.DHY = -self.DEY.conj().T

    def solve(
            self,
            reject_spurious=True,
            extra_modes=8,
            max_pec_neighbor_energy_fraction=0.35,
    ):
        eps_r_xx_diag = diags(self.eps_r_xx.ravel(order="F"))
        eps_r_yy_diag = diags(self.eps_r_yy.ravel(order="F"))
        mu_r_xx_diag = diags(self.mu_r_xx.ravel(order="F"))
        mu_r_yy_diag = diags(self.mu_r_yy.ravel(order="F"))

        eps_r_zz_diag_inv = diags(self._inverse_on_free(self.eps_r_zz, self.free_ez_mask))
        mu_r_zz_diag_inv = diags(self._inverse_on_free(self.mu_r_zz, self.free_hz_mask))

        P11 = self.DEX @ eps_r_zz_diag_inv @ self.DHY
        P12 = -(self.DEX @ eps_r_zz_diag_inv @ self.DHX + mu_r_yy_diag)
        P21 = self.DEY @ eps_r_zz_diag_inv @ self.DHY + mu_r_xx_diag
        P22 = -self.DEY @ eps_r_zz_diag_inv @ self.DHX
        P = bmat([[P11, P12], [P21, P22]])

        Q11 = self.DHX @ mu_r_zz_diag_inv @ self.DEY
        Q12 = -(self.DHX @ mu_r_zz_diag_inv @ self.DEX + eps_r_yy_diag)
        Q21 = self.DHY @ mu_r_zz_diag_inv @ self.DEY + eps_r_xx_diag
        Q22 = -self.DHY @ mu_r_zz_diag_inv @ self.DEX
        Q = bmat([[Q11, Q12], [Q21, Q22]])

        P_reduced = P[:, self.free_hxy_mask]
        Q_reduced = Q[self.free_hxy_mask, :]
        Omega = P_reduced @ Q_reduced
        Omega = Omega[self.free_exy_mask, :][:, self.free_exy_mask]
        if Omega.shape[0] <= self.num_modes:
            raise ValueError(
                f"Not enough unconstrained electric DOFs ({Omega.shape[0]}) to solve {self.num_modes} modes."
            )

        candidate_modes = self.num_modes
        if reject_spurious:
            candidate_modes += max(0, int(extra_modes))
        candidate_modes = min(candidate_modes, Omega.shape[0] - 1)
        if candidate_modes < self.num_modes:
            raise ValueError(
                f"Not enough unconstrained electric DOFs ({Omega.shape[0]}) to solve {self.num_modes} modes."
            )

        eigenvalues, reduced_eigenvectors = eigs(Omega, k=candidate_modes, sigma=self.guess)
        eigenvectors = np.zeros((2 * self.Nx * self.Ny, candidate_modes), dtype=np.complex128)
        eigenvectors[self.free_exy_mask, :] = reduced_eigenvectors

        sort_indices = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        if reject_spurious:
            keep_indices = self._select_physical_candidates(
                eigenvectors,
                self.num_modes,
                max_pec_neighbor_energy_fraction,
            )
        else:
            keep_indices = np.arange(self.num_modes)
            self.spurious_scores = np.zeros(candidate_modes, dtype=np.float64)
            self.accepted_candidate_indices = keep_indices.copy()
            self.rejected_candidate_indices = np.array([], dtype=int)
            self.unselected_candidate_indices = np.array([], dtype=int)

        self.eigenvalues = eigenvalues[keep_indices]
        self.eigenvectors = eigenvectors[:, keep_indices]
        self.complex_neff = self._passive_positive_neff(-self.eigenvalues)
        self.real_neff = np.real(self.complex_neff)

        self._calculate_fields(Q_reduced, eps_r_zz_diag_inv, mu_r_zz_diag_inv)
        self._normalize_modes_to_power()
        self._align_modes_for_real_profile_power()
        self._set_modal_fields()

    def _select_physical_candidates(self, eigenvectors, num_modes, max_pec_neighbor_energy_fraction):
        scores = self._pec_neighbor_energy_scores(eigenvectors)
        candidate_indices = np.arange(eigenvectors.shape[1])
        accepted = candidate_indices[scores <= max_pec_neighbor_energy_fraction]
        rejected = candidate_indices[scores > max_pec_neighbor_energy_fraction]
        if accepted.size < num_modes:
            # If the cutoff is too strict for a geometry, fill from the least
            # PEC-localized rejected candidates rather than silently returning
            # fewer modes.
            rejected_by_score = rejected[np.argsort(scores[rejected])]
            accepted = np.concatenate((accepted, rejected_by_score[:num_modes - accepted.size]))
        keep_indices = np.sort(accepted[:num_modes])
        self.spurious_scores = scores
        self.accepted_candidate_indices = keep_indices.copy()
        self.rejected_candidate_indices = rejected.copy()
        self.unselected_candidate_indices = np.setdiff1d(candidate_indices, keep_indices, assume_unique=True)
        return keep_indices

    def _pec_neighbor_energy_scores(self, eigenvectors):
        Nx, Ny = self.Nx, self.Ny
        ex = eigenvectors[: Nx * Ny, :]
        ey = eigenvectors[Nx * Ny:, :]
        total_energy = np.sum(np.abs(ex) ** 2, axis=0) + np.sum(np.abs(ey) ** 2, axis=0)
        near_ex = self._dilate_mask(self.pec_ex_mask).ravel(order="F")
        near_ey = self._dilate_mask(self.pec_ey_mask).ravel(order="F")
        pec_neighbor_energy = np.sum(np.abs(ex[near_ex, :]) ** 2, axis=0)
        pec_neighbor_energy += np.sum(np.abs(ey[near_ey, :]) ** 2, axis=0)
        scores = np.zeros(eigenvectors.shape[1], dtype=np.float64)
        valid = total_energy > 1e-300
        scores[valid] = np.real(pec_neighbor_energy[valid] / total_energy[valid])
        scores[~valid] = np.inf
        return scores

    @staticmethod
    def _dilate_mask(mask):
        mask = np.asarray(mask, dtype=bool)
        dilated = mask.copy()
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        return dilated

    def _inverse_on_free(self, values, free_mask):
        flat = values.ravel(order="F")
        inverse = np.zeros_like(flat, dtype=np.complex128)
        inverse[free_mask] = 1.0 / flat[free_mask]
        return inverse

    def _calculate_fields(self, Q_reduced, eps_r_zz_diag_inv, mu_r_zz_diag_inv):
        Nx, Ny = self.Nx, self.Ny
        eigenvalues_inv = diags(np.sqrt(self.eigenvalues)).power(-1)
        Exy = self.eigenvectors
        self.Ex = np.asarray(Exy[: Nx * Ny, :], dtype=np.complex128)
        self.Ey = np.asarray(Exy[Nx * Ny:, :], dtype=np.complex128)

        Hxy_reduced = Q_reduced @ Exy @ eigenvalues_inv
        Hxy = np.zeros((2 * Nx * Ny, self.num_modes), dtype=np.complex128)
        Hxy[self.free_hxy_mask, :] = Hxy_reduced
        Hx_norm = Hxy[: Nx * Ny, :]
        Hy_norm = Hxy[Nx * Ny:, :]
        Hz_norm = mu_r_zz_diag_inv @ (self.DEX @ self.Ey - self.DEY @ self.Ex)
        self.Ez = np.asarray(eps_r_zz_diag_inv @ (self.DHX @ Hy_norm - self.DHY @ Hx_norm), dtype=np.complex128)
        self.Hx = np.asarray(1j * Hx_norm / self.eta0, dtype=np.complex128)
        self.Hy = np.asarray(1j * Hy_norm / self.eta0, dtype=np.complex128)
        self.Hz = np.asarray(1j * Hz_norm / self.eta0, dtype=np.complex128)
        self._zero_constrained_fields()

    def _zero_constrained_fields(self):
        self.Ex[~self.free_ex_mask, :] = 0.0
        self.Ey[~self.free_ey_mask, :] = 0.0
        self.Ez[~self.free_ez_mask, :] = 0.0
        self.Hx[~self.free_hx_mask, :] = 0.0
        self.Hy[~self.free_hy_mask, :] = 0.0
        self.Hz[~self.free_hz_mask, :] = 0.0

    def _set_modal_fields(self):
        self.modal_Ex = self.Ex[:, self.mode_index]
        self.modal_Ey = self.Ey[:, self.mode_index]
        self.modal_Ez = self.Ez[:, self.mode_index]
        self.modal_Hx = self.Hx[:, self.mode_index]
        self.modal_Hy = self.Hy[:, self.mode_index]
        self.modal_Hz = self.Hz[:, self.mode_index]
        self.modal_complex_neff = self.complex_neff[self.mode_index]
        self.modal_real_neff = self.real_neff[self.mode_index]
        self.modal_power = self.powers[self.mode_index]

    def _normalize_modes_to_power(self, target_power=1.0):
        self.powers = np.zeros(self.num_modes, dtype=np.float64)
        cell_area = self.dx * self.dy
        for mode in range(self.num_modes):
            power = self._calculate_mode_power(mode, cell_area)
            if not np.isfinite(power) or abs(power) < 1e-300:
                raise ValueError(f"Cannot normalize mode {mode}: modal power is {power}.")
            if power < 0:
                self.Hx[:, mode] = -self.Hx[:, mode]
                self.Hy[:, mode] = -self.Hy[:, mode]
                self.Hz[:, mode] = -self.Hz[:, mode]
                power = -power
            scale = np.sqrt(target_power / power)
            for field in (self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz):
                field[:, mode] *= scale
            self.powers[mode] = self._calculate_mode_power(mode, cell_area)

    def _align_modes_for_real_profile_power(self):
        cell_area = self.dx * self.dy
        for mode in range(self.num_modes):
            e_real = (np.real(self.Ex[:, mode]), np.real(self.Ey[:, mode]))
            e_imag = (np.imag(self.Ex[:, mode]), np.imag(self.Ey[:, mode]))
            h_real = (np.real(self.Hx[:, mode]), np.real(self.Hy[:, mode]))
            h_imag = (np.imag(self.Hx[:, mode]), np.imag(self.Hy[:, mode]))
            paa = self._calculate_real_profile_power(e_real, h_real, cell_area)
            pbb = self._calculate_real_profile_power(e_imag, h_imag, cell_area)
            pab = self._calculate_real_profile_power(e_real, h_imag, cell_area)
            pba = self._calculate_real_profile_power(e_imag, h_real, cell_area)
            phase = 0.5 * np.arctan2(-0.5 * (pab + pba), 0.5 * (paa - pbb))
            self._rotate_mode(mode, phase)
            aligned_power = self._calculate_real_profile_power(
                (np.real(self.Ex[:, mode]), np.real(self.Ey[:, mode])),
                (np.real(self.Hx[:, mode]), np.real(self.Hy[:, mode])),
                cell_area,
            )
            if aligned_power < 0:
                self._rotate_mode(mode, 0.5 * np.pi)

    def _rotate_mode(self, mode, phase):
        phase_factor = np.exp(1j * phase)
        for field in (self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz):
            field[:, mode] = phase_factor * field[:, mode]

    def _calculate_real_profile_power(self, e_fields, h_fields, cell_area=None):
        if cell_area is None:
            cell_area = self.dx * self.dy
        ex, ey = e_fields
        hx, hy = h_fields
        return math.fsum(np.ravel(np.real(ex * hy - ey * hx))) * cell_area

    def _calculate_mode_power(self, mode, cell_area=None):
        if cell_area is None:
            cell_area = self.dx * self.dy
        poynting_z = self.Ex[:, mode] * np.conj(self.Hy[:, mode]) - self.Ey[:, mode] * np.conj(self.Hx[:, mode])
        return 0.5 * math.fsum(np.real(poynting_z)) * cell_area

    def plot_e_fields(self, output_path="fdfd_modes_ex_ey.png"):
        fig, axes = plt.subplots(self.num_modes, 2, figsize=(8, 3 * self.num_modes), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for mode in range(self.num_modes):
            ex = np.real(self._field_to_array(self.Ex[:, mode]))
            ey = np.real(self._field_to_array(self.Ey[:, mode]))
            for ax, field, component in [(axes[mode, 0], ex, "Ex"), (axes[mode, 1], ey, "Ey")]:
                image = ax.imshow(field.T, origin="lower", cmap="RdBu_r", aspect="auto")
                ax.set_title(f"Mode {mode} {component}, neff={self.complex_neff[mode]:.6g}")
                ax.set_xlabel("x index")
                ax.set_ylabel("y index")
                fig.colorbar(image, ax=ax)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    def plot_h_fields(self, output_path="fdfd_modes_hx_hy.png"):
        fig, axes = plt.subplots(self.num_modes, 2, figsize=(8, 3 * self.num_modes), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for mode in range(self.num_modes):
            hx = np.real(self._field_to_array(self.Hx[:, mode]))
            hy = np.real(self._field_to_array(self.Hy[:, mode]))
            for ax, field, component in [(axes[mode, 0], hx, "Hx"), (axes[mode, 1], hy, "Hy")]:
                image = ax.imshow(field.T, origin="lower", cmap="RdBu_r", aspect="auto")
                ax.set_title(f"Mode {mode} {component}, neff={self.complex_neff[mode]:.6g}")
                ax.set_xlabel("x index")
                ax.set_ylabel("y index")
                fig.colorbar(image, ax=ax)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    def plot_pec_component_masks(self, output_path="fdfd_yee_sampled_pec_masks.png"):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
        masks = [
            (self.pec_ex_mask.astype(float), "PEC Ex"),
            (self.pec_ey_mask.astype(float), "PEC Ey"),
            (self.pec_ez_mask.astype(float), "PEC Ez"),
        ]
        for ax, (mask, title) in zip(axes, masks):
            image = ax.imshow(mask.T, origin="lower", cmap="gray_r", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")
            fig.colorbar(image, ax=ax)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    def _passive_positive_neff(self, neff_squared):
        sqrt = np.sqrt(neff_squared)
        neff = np.where(np.real(sqrt) < 0, -sqrt, sqrt)
        real = np.real(neff)
        imag = np.imag(neff)
        tolerance = 1e-12 * np.maximum(1.0, np.abs(neff))
        real = np.where(np.abs(real) <= tolerance, 0.0, real)
        imag = np.where(np.abs(imag) <= tolerance, 0.0, np.abs(imag))
        return real + 1j * imag

    def _max_magnitude(self, x):
        finite = np.isfinite(x)
        if not np.any(finite):
            return 1.0
        return float(np.max(np.abs(x[finite])))

