import math

import matplotlib
import numpy as np

import gprMax.config as config

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.sparse import bmat, coo_matrix, diags
from scipy.sparse.linalg import eigs


class FDFD_2D_mode_solver:
    """2D full-vector FDFD mode solver on a local staggered Yee grid.

    The solver works in a local coordinate system:

        u, v: transverse source-plane axes
        w: propagation-normal axis

    Material arrays are sampled directly at the matching Yee component
    locations.  For a transverse cell region with shape ``(Nu, Nv)`` the
    expected component shapes are:

        eps_r_uu, E_u, H_v: (Nu,     Nv + 1)
        eps_r_vv, E_v, H_u: (Nu + 1, Nv)
        eps_r_ww, E_w:      (Nu + 1, Nv + 1)
        mu_r_uu, H_u:       (Nu + 1, Nv)
        mu_r_vv, H_v:       (Nu,     Nv + 1)
        mu_r_ww, H_w:       (Nu,     Nv)

    Electric PEC masks constrain the corresponding electric component DOFs.
    Non-finite electric material entries are also interpreted as PEC and are
    replaced by finite placeholders after the masks have been built.
    """

    def __init__(
            self,
            frequency,
            du,
            dv,
            mode_index,
            eps_r_uu,
            eps_r_vv,
            eps_r_ww,
            mu_r_uu,
            mu_r_vv,
            mu_r_ww,
            pec_u_mask=None,
            pec_v_mask=None,
            pec_w_mask=None,
            pmc_u_mask=None,
            pmc_v_mask=None,
            pmc_w_mask=None,
            guess=None,
    ):
        self.epsilon0 = config.sim_config.em_consts["e0"]
        self.mu0 = config.sim_config.em_consts["m0"]
        self.c = config.sim_config.em_consts["c"]
        self.eta0 = config.sim_config.em_consts["z0"]
        self.omega = 2 * np.pi * frequency
        self.k0 = self.omega / self.c

        self.frequency = frequency
        self.du = du
        self.dv = dv
        self.normalized_du = self.k0 * du
        self.normalized_dv = self.k0 * dv
        self.mode_index = int(mode_index)
        self.num_modes = self.mode_index + 1

        self.eps_r_uu = self._to_solver_array(eps_r_uu)
        self.eps_r_vv = self._to_solver_array(eps_r_vv)
        self.eps_r_ww = self._to_solver_array(eps_r_ww)
        self.mu_r_uu = self._to_solver_array(mu_r_uu)
        self.mu_r_vv = self._to_solver_array(mu_r_vv)
        self.mu_r_ww = self._to_solver_array(mu_r_ww)

        self.Nu, self.Nv = self.mu_r_ww.shape
        if self.Nu <= 0 or self.Nv <= 0:
            raise ValueError("Local transverse Yee cell shape must be positive.")
        self.shape_cell = (self.Nu, self.Nv)
        self.shape_eu = (self.Nu, self.Nv + 1)
        self.shape_ev = (self.Nu + 1, self.Nv)
        self.shape_ew = (self.Nu + 1, self.Nv + 1)
        self.shape_hu = self.shape_ev
        self.shape_hv = self.shape_eu
        self.shape_hw = self.shape_cell

        self.n_eu = int(np.prod(self.shape_eu))
        self.n_ev = int(np.prod(self.shape_ev))
        self.n_ew = int(np.prod(self.shape_ew))
        self.n_hu = int(np.prod(self.shape_hu))
        self.n_hv = int(np.prod(self.shape_hv))
        self.n_hw = int(np.prod(self.shape_hw))
        self.n_e_transverse = self.n_eu + self.n_ev
        self.n_h_transverse = self.n_hu + self.n_hv

        self._validate_material_shapes()
        self.pec_u_mask = self._component_constraint_mask(self.eps_r_uu, pec_u_mask, self.shape_eu)
        self.pec_v_mask = self._component_constraint_mask(self.eps_r_vv, pec_v_mask, self.shape_ev)
        self.pec_w_mask = self._component_constraint_mask(self.eps_r_ww, pec_w_mask, self.shape_ew)
        self.pmc_u_mask = self._component_constraint_mask(self.mu_r_uu, pmc_u_mask, self.shape_hu, default=False)
        self.pmc_v_mask = self._component_constraint_mask(self.mu_r_vv, pmc_v_mask, self.shape_hv, default=False)
        self.pmc_w_mask = self._component_constraint_mask(self.mu_r_ww, pmc_w_mask, self.shape_hw, default=False)

        self.eps_r_uu[self.pec_u_mask] = 1.0 + 0j
        self.eps_r_vv[self.pec_v_mask] = 1.0 + 0j
        self.eps_r_ww[self.pec_w_mask] = 1.0 + 0j
        self.mu_r_uu[self.pmc_u_mask] = 1.0 + 0j
        self.mu_r_vv[self.pmc_v_mask] = 1.0 + 0j
        self.mu_r_ww[self.pmc_w_mask] = 1.0 + 0j

        self.free_eu_mask = ~self.pec_u_mask.ravel(order="F")
        self.free_ev_mask = ~self.pec_v_mask.ravel(order="F")
        self.free_ew_mask = ~self.pec_w_mask.ravel(order="F")
        self.free_hu_mask = ~self.pmc_u_mask.ravel(order="F")
        self.free_hv_mask = ~self.pmc_v_mask.ravel(order="F")
        self.free_hw_mask = ~self.pmc_w_mask.ravel(order="F")
        self.free_euv_mask = np.concatenate((self.free_eu_mask, self.free_ev_mask))
        self.free_huv_mask = np.concatenate((self.free_hu_mask, self.free_hv_mask))

        self.guess = guess if guess is not None else self._default_guess()
        self.eigenvalues = None
        self.eigenvectors = None
        self.complex_neff = None
        self.real_neff = None
        self.powers = None

        self._init_operators()

    @staticmethod
    def _to_solver_array(values):
        return np.asarray(values, dtype=np.complex128).copy()

    def _validate_material_shapes(self):
        expected = {
            "eps_r_uu": self.shape_eu,
            "eps_r_vv": self.shape_ev,
            "eps_r_ww": self.shape_ew,
            "mu_r_uu": self.shape_hu,
            "mu_r_vv": self.shape_hv,
            "mu_r_ww": self.shape_hw,
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} shape {actual} does not match expected local Yee shape {shape}.")

    def _component_constraint_mask(self, values, explicit_mask, expected_shape, default=True):
        mask = np.zeros(expected_shape, dtype=bool)
        if default:
            mask |= ~np.isfinite(values)
        if explicit_mask is not None:
            explicit_mask = np.asarray(explicit_mask, dtype=bool)
            if explicit_mask.shape != expected_shape:
                raise ValueError(
                    f"Constraint mask shape {explicit_mask.shape} does not match expected shape {expected_shape}."
                )
            mask |= explicit_mask
        return mask

    @staticmethod
    def _flat_index(i, j, nu):
        return i + j * nu

    def _difference_matrix_u(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nu, _ = in_shape
        out_nu, out_nv = out_shape
        for j in range(out_nv):
            for i in range(out_nu):
                row = self._flat_index(i, j, out_nu)
                entries = ((i + 1, j, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i - 1, j, -1.0))
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nu))
                        data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(out_nu * out_nv, in_shape[0] * in_shape[1])).tocsr()

    def _difference_matrix_v(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nu, _ = in_shape
        out_nu, out_nv = out_shape
        for j in range(out_nv):
            for i in range(out_nu):
                row = self._flat_index(i, j, out_nu)
                entries = ((i, j + 1, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i, j - 1, -1.0))
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nu))
                        data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(out_nu * out_nv, in_shape[0] * in_shape[1])).tocsr()

    def _init_operators(self):
        du = self.normalized_du
        dv = self.normalized_dv

        self.DEU_EW_TO_EU = self._difference_matrix_u(self.shape_ew, self.shape_eu, du, forward=True)
        self.DEV_EW_TO_EV = self._difference_matrix_v(self.shape_ew, self.shape_ev, dv, forward=True)
        self.DEU_EV_TO_HW = self._difference_matrix_u(self.shape_ev, self.shape_hw, du, forward=True)
        self.DEV_EU_TO_HW = self._difference_matrix_v(self.shape_eu, self.shape_hw, dv, forward=True)

        self.DHU_HV_TO_EW = -self.DEU_EW_TO_EU.conj().T
        self.DHV_HU_TO_EW = -self.DEV_EW_TO_EV.conj().T
        self.DHU_HW_TO_HU = -self.DEU_EV_TO_HW.conj().T
        self.DHV_HW_TO_HV = -self.DEV_EU_TO_HW.conj().T

    @staticmethod
    def _diag(values):
        return diags(values.ravel(order="F"), format="csr")

    def _inverse_diag_on_free(self, values, free_mask):
        flat = values.ravel(order="F")
        inverse = np.zeros_like(flat, dtype=np.complex128)
        inverse[free_mask] = 1.0 / flat[free_mask]
        return diags(inverse, format="csr")

    def solve(self):
        eps_uu_diag = self._diag(self.eps_r_uu)
        eps_vv_diag = self._diag(self.eps_r_vv)
        mu_uu_diag = self._diag(self.mu_r_uu)
        mu_vv_diag = self._diag(self.mu_r_vv)
        eps_ww_inv = self._inverse_diag_on_free(self.eps_r_ww, self.free_ew_mask)
        mu_ww_inv = self._inverse_diag_on_free(self.mu_r_ww, self.free_hw_mask)

        P11 = self.DEU_EW_TO_EU @ eps_ww_inv @ self.DHV_HU_TO_EW
        P12 = -(self.DEU_EW_TO_EU @ eps_ww_inv @ self.DHU_HV_TO_EW + mu_vv_diag)
        P21 = self.DEV_EW_TO_EV @ eps_ww_inv @ self.DHV_HU_TO_EW + mu_uu_diag
        P22 = -self.DEV_EW_TO_EV @ eps_ww_inv @ self.DHU_HV_TO_EW
        P = bmat([[P11, P12], [P21, P22]], format="csr")

        Q11 = self.DHU_HW_TO_HU @ mu_ww_inv @ self.DEV_EU_TO_HW
        Q12 = -(self.DHU_HW_TO_HU @ mu_ww_inv @ self.DEU_EV_TO_HW + eps_vv_diag)
        Q21 = self.DHV_HW_TO_HV @ mu_ww_inv @ self.DEV_EU_TO_HW + eps_uu_diag
        Q22 = -self.DHV_HW_TO_HV @ mu_ww_inv @ self.DEU_EV_TO_HW
        Q = bmat([[Q11, Q12], [Q21, Q22]], format="csr")

        P_reduced = P[:, self.free_huv_mask]
        Q_reduced = Q[self.free_huv_mask, :]
        omega_matrix = P_reduced @ Q_reduced
        omega_matrix = omega_matrix[self.free_euv_mask, :][:, self.free_euv_mask]
        if omega_matrix.shape[0] <= self.num_modes:
            raise ValueError(
                f"Not enough unconstrained electric DOFs ({omega_matrix.shape[0]}) to solve {self.num_modes} modes."
            )

        eigenvalues, reduced_eigenvectors = eigs(omega_matrix, k=self.num_modes, sigma=self.guess)
        eigenvectors = np.zeros((self.n_e_transverse, self.num_modes), dtype=np.complex128)
        eigenvectors[self.free_euv_mask, :] = reduced_eigenvectors

        order = np.argsort(np.real(eigenvalues))
        self.eigenvalues = eigenvalues[order]
        self.eigenvectors = eigenvectors[:, order]
        self.complex_neff = self._passive_positive_neff(-self.eigenvalues)
        self.real_neff = np.real(self.complex_neff)

        self._calculate_fields(Q_reduced, eps_ww_inv, mu_ww_inv)
        self._normalize_modes_to_power()
        self._align_modes_for_real_profile_power()
        self._set_modal_fields()

    def _calculate_fields(self, Q_reduced, eps_ww_inv, mu_ww_inv):
        sqrt_eigenvalues = np.sqrt(self.eigenvalues)
        if np.any(np.abs(sqrt_eigenvalues) < 1e-300):
            raise ValueError("Encountered a near-zero eigenvalue while reconstructing magnetic fields.")
        eigenvalues_inv = diags(1.0 / sqrt_eigenvalues, format="csr")

        eu_flat = np.asarray(self.eigenvectors[:self.n_eu, :], dtype=np.complex128)
        ev_flat = np.asarray(self.eigenvectors[self.n_eu:, :], dtype=np.complex128)
        huv_reduced = Q_reduced @ self.eigenvectors @ eigenvalues_inv
        huv_flat = np.zeros((self.n_h_transverse, self.num_modes), dtype=np.complex128)
        huv_flat[self.free_huv_mask, :] = huv_reduced
        hu_norm = np.asarray(huv_flat[:self.n_hu, :], dtype=np.complex128)
        hv_norm = np.asarray(huv_flat[self.n_hu:, :], dtype=np.complex128)
        ew_flat = np.asarray(eps_ww_inv @ (self.DHU_HV_TO_EW @ hv_norm - self.DHV_HU_TO_EW @ hu_norm), dtype=np.complex128)
        hw_norm = np.asarray(mu_ww_inv @ (self.DEU_EV_TO_HW @ ev_flat - self.DEV_EU_TO_HW @ eu_flat), dtype=np.complex128)

        self.Eu = self._unflatten_modes(eu_flat, self.shape_eu)
        self.Ev = self._unflatten_modes(ev_flat, self.shape_ev)
        self.Ew = self._unflatten_modes(ew_flat, self.shape_ew)
        self.Hu = self._unflatten_modes(1j * hu_norm / self.eta0, self.shape_hu)
        self.Hv = self._unflatten_modes(1j * hv_norm / self.eta0, self.shape_hv)
        self.Hw = self._unflatten_modes(1j * hw_norm / self.eta0, self.shape_hw)
        self._zero_constrained_fields()

    @staticmethod
    def _unflatten_modes(flat_modes, shape):
        return np.asarray(flat_modes, dtype=np.complex128).reshape((*shape, flat_modes.shape[1]), order="F")

    def _zero_constrained_fields(self):
        self.Eu[self.pec_u_mask, :] = 0.0
        self.Ev[self.pec_v_mask, :] = 0.0
        self.Ew[self.pec_w_mask, :] = 0.0
        self.Hu[self.pmc_u_mask, :] = 0.0
        self.Hv[self.pmc_v_mask, :] = 0.0
        self.Hw[self.pmc_w_mask, :] = 0.0

    def _set_modal_fields(self):
        self.modal_Eu = self.Eu[:, :, self.mode_index]
        self.modal_Ev = self.Ev[:, :, self.mode_index]
        self.modal_Ew = self.Ew[:, :, self.mode_index]
        self.modal_Hu = self.Hu[:, :, self.mode_index]
        self.modal_Hv = self.Hv[:, :, self.mode_index]
        self.modal_Hw = self.Hw[:, :, self.mode_index]
        self.modal_complex_neff = self.complex_neff[self.mode_index]
        self.modal_real_neff = self.real_neff[self.mode_index]
        self.modal_power = self.powers[self.mode_index]

    def _field_to_cells(self, field, component):
        if component in ("u", "hv"):
            return 0.5 * (field[:, :self.Nv] + field[:, 1:])
        if component in ("v", "hu"):
            return 0.5 * (field[:self.Nu, :] + field[1:, :])
        if component == "w_e":
            return 0.25 * (
                field[:self.Nu, :self.Nv]
                + field[1:, :self.Nv]
                + field[:self.Nu, 1:]
                + field[1:, 1:]
            )
        if component == "w_h":
            return field
        raise ValueError(f"Unknown component {component!r}.")

    def _calculate_mode_power(self, mode):
        eu = self._field_to_cells(self.Eu[:, :, mode], "u")
        ev = self._field_to_cells(self.Ev[:, :, mode], "v")
        hu = self._field_to_cells(self.Hu[:, :, mode], "hu")
        hv = self._field_to_cells(self.Hv[:, :, mode], "hv")
        poynting_w = eu * np.conj(hv) - ev * np.conj(hu)
        return 0.5 * math.fsum(np.ravel(np.real(poynting_w))) * self.du * self.dv

    def _real_profile_power_from_fields(self, eu_field, ev_field, hu_field, hv_field):
        eu = self._field_to_cells(np.real(eu_field), "u")
        ev = self._field_to_cells(np.real(ev_field), "v")
        hu = self._field_to_cells(np.real(hu_field), "hu")
        hv = self._field_to_cells(np.real(hv_field), "hv")
        poynting_w = eu * hv - ev * hu
        return math.fsum(np.ravel(np.real(poynting_w))) * self.du * self.dv

    def _calculate_real_profile_power(self, mode):
        return self._real_profile_power_from_fields(
            self.Eu[:, :, mode],
            self.Ev[:, :, mode],
            self.Hu[:, :, mode],
            self.Hv[:, :, mode],
        )

    def _normalize_modes_to_power(self, target_power=1.0):
        self.powers = np.zeros(self.num_modes, dtype=np.float64)
        for mode in range(self.num_modes):
            power = self._calculate_mode_power(mode)
            if not np.isfinite(power) or abs(power) < 1e-300:
                raise ValueError(f"Cannot normalize mode {mode}: modal power is {power}.")
            if power < 0:
                self.Hu[:, :, mode] = -self.Hu[:, :, mode]
                self.Hv[:, :, mode] = -self.Hv[:, :, mode]
                self.Hw[:, :, mode] = -self.Hw[:, :, mode]
                power = -power
            scale = np.sqrt(target_power / power)
            for field in (self.Eu, self.Ev, self.Ew, self.Hu, self.Hv, self.Hw):
                field[:, :, mode] *= scale
            self.powers[mode] = self._calculate_mode_power(mode)

    def _align_modes_for_real_profile_power(self):
        for mode in range(self.num_modes):
            eu = self.Eu[:, :, mode]
            ev = self.Ev[:, :, mode]
            hu = self.Hu[:, :, mode]
            hv = self.Hv[:, :, mode]
            e_real_h_real = self._real_profile_power_from_fields(np.real(eu), np.real(ev), np.real(hu), np.real(hv))
            e_imag_h_imag = self._real_profile_power_from_fields(np.imag(eu), np.imag(ev), np.imag(hu), np.imag(hv))
            e_real_h_imag = self._real_profile_power_from_fields(np.real(eu), np.real(ev), np.imag(hu), np.imag(hv))
            e_imag_h_real = self._real_profile_power_from_fields(np.imag(eu), np.imag(ev), np.real(hu), np.real(hv))
            phase = 0.5 * np.arctan2(
                -0.5 * (e_real_h_imag + e_imag_h_real),
                0.5 * (e_real_h_real - e_imag_h_imag),
            )
            self._rotate_mode(mode, phase)
            if self._calculate_real_profile_power(mode) < 0:
                self._rotate_mode(mode, 0.5 * np.pi)

    def _rotate_mode(self, mode, phase):
        phase_factor = np.exp(1j * phase)
        for field in (self.Eu, self.Ev, self.Ew, self.Hu, self.Hv, self.Hw):
            field[:, :, mode] *= phase_factor

    def plot_e_fields(self, output_path="fdfd_modes_eu_ev.png"):
        fig, axes = plt.subplots(self.num_modes, 2, figsize=(8, 3 * self.num_modes), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for mode in range(self.num_modes):
            for ax, field, component in (
                    (axes[mode, 0], np.real(self.Eu[:, :, mode]), "E_u"),
                    (axes[mode, 1], np.real(self.Ev[:, :, mode]), "E_v"),
            ):
                image = ax.imshow(field.T, origin="lower", cmap="RdBu_r", aspect="auto")
                ax.set_title(f"Mode {mode} {component}, neff={self.complex_neff[mode]:.6g}")
                ax.set_xlabel("u index")
                ax.set_ylabel("v index")
                fig.colorbar(image, ax=ax)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    def plot_h_fields(self, output_path="fdfd_modes_hu_hv.png"):
        fig, axes = plt.subplots(self.num_modes, 2, figsize=(8, 3 * self.num_modes), constrained_layout=True)
        axes = np.atleast_2d(axes)
        for mode in range(self.num_modes):
            for ax, field, component in (
                    (axes[mode, 0], np.real(self.Hu[:, :, mode]), "H_u"),
                    (axes[mode, 1], np.real(self.Hv[:, :, mode]), "H_v"),
            ):
                image = ax.imshow(field.T, origin="lower", cmap="RdBu_r", aspect="auto")
                ax.set_title(f"Mode {mode} {component}, neff={self.complex_neff[mode]:.6g}")
                ax.set_xlabel("u index")
                ax.set_ylabel("v index")
                fig.colorbar(image, ax=ax)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    def plot_pec_component_masks(self, output_path="fdfd_yee_uvw_pec_masks.png"):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
        masks = [
            (self.pec_u_mask.astype(float), "PEC E_u"),
            (self.pec_v_mask.astype(float), "PEC E_v"),
            (self.pec_w_mask.astype(float), "PEC E_w"),
        ]
        for ax, (mask, title) in zip(axes, masks):
            image = ax.imshow(mask.T, origin="lower", cmap="gray_r", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("u index")
            ax.set_ylabel("v index")
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

    def _default_guess(self):
        return -max(
            self._max_magnitude(arr)
            for arr in [self.eps_r_uu, self.eps_r_vv, self.eps_r_ww, self.mu_r_uu, self.mu_r_vv, self.mu_r_ww]
        )

    @staticmethod
    def _max_magnitude(values):
        finite = np.isfinite(values)
        if not np.any(finite):
            return 1.0
        return float(np.max(np.abs(values[finite])))
