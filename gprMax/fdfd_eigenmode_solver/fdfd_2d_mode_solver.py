import numpy as np
import matplotlib
import math

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.sparse import diags, kron, eye, bmat
from scipy.sparse.linalg import eigs


class FDFD_2D_mode_solver:
    def __init__(self, frequency, dx, dy, mode_index, eps_r_xx, eps_r_yy, eps_r_zz, mu_r_xx, mu_r_yy, mu_r_zz):
        self.epsilon0 = 8.85e-12
        self.mu0 = 1.26e-6
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.eta0 = np.sqrt(self.mu0 / self.epsilon0)
        self.omega = 2 * np.pi * frequency
        self.k0 = self.omega / self.c

        self.frequency = frequency
        self.dx = dx
        self.normalized_dx = self.k0 * dx
        self.dy = dy
        self.normalized_dy = self.k0 * dy
        self.eps_r_xx = eps_r_xx
        self.eps_r_yy = eps_r_yy
        self.eps_r_zz = eps_r_zz
        self.mu_r_xx = mu_r_xx
        self.mu_r_yy = mu_r_yy
        self.mu_r_zz = mu_r_zz
        self.Nx = self.eps_r_xx.shape[0]
        self.Ny = self.eps_r_xx.shape[1]

        self.mode_index = mode_index
        self.num_modes = self.mode_index + 1

        self.guess = -max(
            self._max_finite_magnitude(arr)
            for arr in [eps_r_xx, eps_r_yy, eps_r_zz, mu_r_xx, mu_r_yy, mu_r_zz]
        )

        self.eigenvectors = None
        self.gammas = None
        self.powers = None

        self._init_operators()

    def _init_operators(self):
        def diff_operator(n):
            e = np.ones(n)

            data = np.array([-e, e])
            offsets = np.array([0, 1])
            D = diags(data, offsets, shape=(n, n)).tolil()
            return D.tocsr()

        Ix, Iy = eye(self.Nx), eye(self.Ny)
        self.DEX = kron(Iy, diff_operator(self.Nx)) / self.normalized_dx  # Differentiation matrix for Ex
        self.DEY = kron(diff_operator(self.Ny), Ix) / self.normalized_dy  # Differentiation matrix for Ey
        self.DHX = -self.DEX.conj().T  # Differentiation matrix for Hx
        self.DHY = -self.DEY.conj().T  # Differentiation matrix  for Hy

    def solve(self):
        eps_r_xx_diag = diags(self.eps_r_xx.ravel(order="F"))
        eps_r_yy_diag = diags(self.eps_r_yy.ravel(order="F"))
        eps_r_zz_diag = diags(self.eps_r_zz.ravel(order="F"))
        mu_r_xx_diag = diags(self.mu_r_xx.ravel(order="F"))
        mu_r_yy_diag = diags(self.mu_r_yy.ravel(order="F"))
        mu_r_zz_diag = diags(self.mu_r_zz.ravel(order="F"))

        eps_r_zz_diag_inv = eps_r_zz_diag.power(-1)
        mu_r_zz_diag_inv = mu_r_zz_diag.power(-1)

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

        Omega = P @ Q
        self.eigenvalues, self.eigenvectors = eigs(Omega, k=self.num_modes, sigma=self.guess)
        sort_indices = np.argsort(np.real(self.eigenvalues))
        self.eigenvalues = self.eigenvalues[sort_indices]
        self.eigenvectors = self.eigenvectors[:, sort_indices]
        self.complex_neff = self._passive_positive_neff(-self.eigenvalues)
        self.real_neff = np.real(self.complex_neff)

        # Calculate fields
        Exy = self.eigenvectors
        Nx, Ny = self.Nx, self.Ny
        eigenvalues_inv = diags(np.sqrt(self.eigenvalues)).power(-1)
        self.Ex = np.asarray(Exy[: Nx * Ny, :], dtype=np.complex128)
        self.Ey = np.asarray(Exy[Nx * Ny:, :], dtype=np.complex128)
        Hxy = Q @ Exy @ eigenvalues_inv
        Hx_norm = Hxy[: Nx * Ny, :]
        Hy_norm = Hxy[Nx * Ny:, :]
        Hz_norm = mu_r_zz_diag_inv @ (self.DEX @ self.Ey - self.DEY @ self.Ex)
        self.Ez = np.asarray(eps_r_zz_diag_inv @ (self.DHX @ Hy_norm - self.DHY @ Hx_norm), dtype=np.complex128)
        self.Hx = np.asarray(1j * Hx_norm / self.eta0, dtype=np.complex128)
        self.Hy = np.asarray(1j * Hy_norm / self.eta0, dtype=np.complex128)
        self.Hz = np.asarray(1j * Hz_norm / self.eta0, dtype=np.complex128)

        self._normalize_modes_to_power()
        self._align_modes_for_real_profile_power()

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
        """Normalize all solved modes to carry `target_power` watts.

        The transverse power is calculated from the time-average Poynting flux
        in the solver's longitudinal direction:

            P = 0.5 Re integral((E x H*) . z dA)

        The H fields returned by the FDFD matrices are converted to physical
        A/m before this method is called.
        """
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
            self.Ex[:, mode] = self.Ex[:, mode] * scale
            self.Ey[:, mode] = self.Ey[:, mode] * scale
            self.Ez[:, mode] = self.Ez[:, mode] * scale
            self.Hx[:, mode] = self.Hx[:, mode] * scale
            self.Hy[:, mode] = self.Hy[:, mode] * scale
            self.Hz[:, mode] = self.Hz[:, mode] * scale

            for _ in range(5):
                normalized_power = self._calculate_mode_power(mode, cell_area)
                if np.isclose(normalized_power, target_power, rtol=1e-12, atol=1e-15):
                    break
                correction = np.sqrt(target_power / normalized_power)
                self.Ex[:, mode] = self.Ex[:, mode] * correction
                self.Ey[:, mode] = self.Ey[:, mode] * correction
                self.Ez[:, mode] = self.Ez[:, mode] * correction
                self.Hx[:, mode] = self.Hx[:, mode] * correction
                self.Hy[:, mode] = self.Hy[:, mode] * correction
                self.Hz[:, mode] = self.Hz[:, mode] * correction

            normalized_power = self._calculate_mode_power(mode, cell_area)
            self.powers[mode] = normalized_power

    def _align_modes_for_real_profile_power(self):
        """Rotate each mode so its real-valued field profile carries power."""
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

            cos_coeff = 0.5 * (paa - pbb)
            sin_coeff = -0.5 * (pab + pba)
            phase = 0.5 * np.arctan2(sin_coeff, cos_coeff)
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
        self.Ex[:, mode] = phase_factor * self.Ex[:, mode]
        self.Ey[:, mode] = phase_factor * self.Ey[:, mode]
        self.Ez[:, mode] = phase_factor * self.Ez[:, mode]
        self.Hx[:, mode] = phase_factor * self.Hx[:, mode]
        self.Hy[:, mode] = phase_factor * self.Hy[:, mode]
        self.Hz[:, mode] = phase_factor * self.Hz[:, mode]

    def _calculate_real_profile_power(self, e_fields, h_fields, cell_area=None):
        if cell_area is None:
            cell_area = self.dx * self.dy
        ex, ey = e_fields
        hx, hy = h_fields
        poynting_z = ex * hy - ey * hx
        return math.fsum(np.ravel(np.real(poynting_z))) * cell_area

    def _calculate_mode_power(self, mode, cell_area=None):
        if cell_area is None:
            cell_area = self.dx * self.dy
        poynting_z = (
                self.Ex[:, mode] * np.conj(self.Hy[:, mode])
                - self.Ey[:, mode] * np.conj(self.Hx[:, mode])
        )
        return 0.5 * math.fsum(np.real(poynting_z)) * cell_area

    def plot_e_fields(self, output_path="fdfd_modes_ex_ey.png"):
        fig, axes = plt.subplots(self.num_modes, 2, figsize=(8, 3 * self.num_modes), constrained_layout=True)
        axes = np.atleast_2d(axes)

        for mode in range(self.num_modes):
            ex = np.real(self.Ex[:, mode]).reshape((self.Nx, self.Ny), order="F")
            ey = np.real(self.Ey[:, mode]).reshape((self.Nx, self.Ny), order="F")

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
            hx = np.real(self.Hx[:, mode]).reshape((self.Nx, self.Ny), order="F")
            hy = np.real(self.Hy[:, mode]).reshape((self.Nx, self.Ny), order="F")

            for ax, field, component in [(axes[mode, 0], hx, "Hx"), (axes[mode, 1], hy, "Hy")]:
                image = ax.imshow(field.T, origin="lower", cmap="RdBu_r", aspect="auto")
                ax.set_title(f"Mode {mode} {component}, neff={self.complex_neff[mode]:.6g}")
                ax.set_xlabel("x index")
                ax.set_ylabel("y index")
                fig.colorbar(image, ax=ax)

        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    def _passive_positive_neff(self, neff_squared):
        """Return neff as positive phase constant plus positive attenuation."""
        sqrt = np.sqrt(neff_squared)
        neff = np.where(np.real(sqrt) < 0, -sqrt, sqrt)

        real = np.real(neff)
        imag = np.imag(neff)
        tolerance = 1e-12 * np.maximum(1.0, np.abs(neff))
        real = np.where(np.abs(real) <= tolerance, 0.0, real)
        imag = np.where(np.abs(imag) <= tolerance, 0.0, np.abs(imag))
        return real + 1j * imag

    def _max_finite_magnitude(self, x):
        finite = np.isfinite(x)
        if not np.any(finite):
            return 1.0
        return float(np.max(np.abs(x[finite])))


if __name__ == "__main__":
    dx = 0.1e-3
    dy = 0.1e-3
    frequency = 100e9
    mode_index = 5
    eps_r_xx = np.ones([120, 100], dtype=complex)
    eps_r_yy = np.ones([120, 100], dtype=complex)
    eps_r_zz = np.ones([120, 100], dtype=complex)
    mu_r_xx = np.ones([120, 100], dtype=complex)
    mu_r_yy = np.ones([120, 100], dtype=complex)
    mu_r_zz = np.ones([120, 100], dtype=complex)

    eps_r_xx[30:60, 40:60] = 4 - 3j
    eps_r_yy[30:60, 40:60] = 5 - 1j
    eps_r_zz[30:60, 40:60] = 6 - 3j

    solver = FDFD_2D_mode_solver(frequency=frequency, dx=dx, dy=dy, mode_index=mode_index, eps_r_xx=eps_r_xx,
                                 eps_r_yy=eps_r_yy, eps_r_zz=eps_r_zz, mu_r_xx=mu_r_xx, mu_r_yy=mu_r_yy,
                                 mu_r_zz=mu_r_zz)

    solver.solve()
    print("Eigenvalues:")
    for index, eigenvalue in enumerate(solver.eigenvalues):
        print(f"  mode {index}: {eigenvalue}")

    print("Effective indices:")
    for index, neff in enumerate(solver.complex_neff):
        print(f"  mode {index}: {neff}")

    print("Real indices:")
    for index, real_neff in enumerate(solver.real_neff):
        print(f"  mode {index}: {real_neff}")
    e_output_path = solver.plot_e_fields()
    print(f"Saved Ex/Ey modal field plot to {e_output_path}")
    h_output_path = solver.plot_h_fields()
    print(f"Saved Hx/Hy modal field plot to {h_output_path}")
