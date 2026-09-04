# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.linalg import eig
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigs

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.numerical_dispersion import (
    discrete_angular_frequency,
    phase_propagation_constant,
    positive_finite,
    spatially_resolved,
)


class FDFD_1D_mode_solver:
    """Scalar 1D FDFD mode solver on a staggered Yee grid.

    The local coordinate system is ``(t, a, w)`` where ``t`` is the one
    physical transverse coordinate, ``a`` is the invariant axis, and ``w`` is
    the propagation direction. The solver assumes that this local basis is
    right handed. Callers using a left-handed global mapping must reverse all
    magnetic fields after mapping.

    The native field shapes are::

        E_t: (N,)       E_a: (N + 1,)   E_w: (N + 1,)
        H_t: (N + 1,)   H_a: (N,)       H_w: (N,)

    ``TM`` means the gprMax 2D TM reduction, whose scalar field is ``E_a``.
    ``TE`` means the gprMax 2D TE reduction, whose scalar field is ``H_a``.

    ``dt`` is the transverse cell spacing, in metres. Optional ``fdtd_dt``
    (seconds) and ``propagation_spacing`` (metres) match the time and normal
    spatial difference symbols to FDTD. ``omega`` and ``k0`` are the physical
    angular frequency and vacuum wavenumber; ``operator_omega`` and
    ``operator_k0`` are the temporal difference symbol and its normalization
    by c. ``operator_neff`` controls field reconstruction; public
    ``complex_neff`` is phase beta / k0.

    After :meth:`solve`, ``raw_powers`` contains the complex Poynting power
    before normalization. ``forward_power_metrics`` is its real part divided
    by a positive, E/H-balanced transverse field norm; ``power_valid`` applies
    the class tolerance to that signed, scale-independent ratio.
    """

    FIELD_SHAPES = {
        "eps_r_t": "cell",
        "eps_r_a": "node",
        "eps_r_w": "node",
        "mu_r_t": "node",
        "mu_r_a": "cell",
        "mu_r_w": "cell",
        "pec_t_mask": "cell",
        "pec_a_mask": "node",
        "pec_w_mask": "node",
        "pmc_t_mask": "node",
        "pmc_a_mask": "cell",
        "pmc_w_mask": "cell",
    }
    FORWARD_POWER_METRIC_TOLERANCE = 1e-8

    def __init__(
        self,
        frequency,
        dt,
        mode_index,
        polarization,
        eps_r_t,
        eps_r_a,
        eps_r_w,
        mu_r_t,
        mu_r_a,
        mu_r_w,
        pec_t_mask=None,
        pec_a_mask=None,
        pec_w_mask=None,
        pmc_t_mask=None,
        pmc_a_mask=None,
        pmc_w_mask=None,
        guess=None,
        *,
        fdtd_dt=None,
        propagation_spacing=None,
    ):
        self.epsilon0 = config.sim_config.em_consts["e0"]
        self.mu0 = config.sim_config.em_consts["m0"]
        self.c = config.sim_config.em_consts["c"]
        self.eta0 = config.sim_config.em_consts["z0"]
        self.frequency = positive_finite(frequency, "frequency")
        self.fdtd_dt = None if fdtd_dt is None else positive_finite(fdtd_dt, "fdtd_dt")
        self.propagation_spacing = (
            None if propagation_spacing is None else positive_finite(propagation_spacing, "propagation_spacing")
        )
        self.omega = 2 * np.pi * self.frequency
        self.k0 = self.omega / self.c
        self.operator_omega = discrete_angular_frequency(self.frequency, self.fdtd_dt)
        self.operator_k0 = self.operator_omega / self.c
        self.dt = positive_finite(dt, "dt")
        self.normalized_dt = self.operator_k0 * self.dt
        self.mode_index = int(mode_index)
        self.num_modes = self.mode_index + 1
        self.polarization = str(polarization).upper()
        if self.mode_index < 0:
            raise ValueError("mode_index must be zero or greater.")
        if self.polarization not in ("TM", "TE"):
            raise ValueError("polarization must be 'TM' or 'TE'.")

        material_names = ("eps_r_t", "eps_r_a", "eps_r_w", "mu_r_t", "mu_r_a", "mu_r_w")
        for name in material_names:
            setattr(self, name, np.asarray(locals()[name], dtype=np.complex128).copy())

        self.N = self.eps_r_t.size
        if self.N <= 0:
            raise ValueError("The 1D transverse Yee grid must contain at least one cell.")
        self.shape_cell = (self.N,)
        self.shape_node = (self.N + 1,)

        mask_values = {
            "pec_t_mask": pec_t_mask,
            "pec_a_mask": pec_a_mask,
            "pec_w_mask": pec_w_mask,
            "pmc_t_mask": pmc_t_mask,
            "pmc_a_mask": pmc_a_mask,
            "pmc_w_mask": pmc_w_mask,
        }
        self._validate_material_shapes()
        for name, values in mask_values.items():
            material_name = ("eps_r_" if name.startswith("pec") else "mu_r_") + name[4]
            expected = self.shape_cell if self.FIELD_SHAPES[name] == "cell" else self.shape_node
            mask = ~np.isfinite(getattr(self, material_name))
            if values is not None:
                values = np.asarray(values, dtype=bool)
                if values.shape != expected:
                    raise ValueError(
                        f"{name} shape {values.shape} does not match expected shape {expected}."
                    )
                mask |= values
            setattr(self, name, mask)
            getattr(self, material_name)[mask] = 1.0 + 0j

        self.guess = guess if guess is not None else self._default_guess()
        self.eigenvalues = None
        self.eigenvectors = None
        self.operator_neff = None
        self.beta = None
        self.complex_neff = None
        self.real_neff = None
        self.raw_powers = None
        self.forward_power_metrics = None
        self.power_valid = None
        self.powers = None
        self._init_operators()

    def _validate_material_shapes(self):
        for name in ("eps_r_t", "mu_r_a", "mu_r_w"):
            actual = getattr(self, name).shape
            if actual != self.shape_cell:
                raise ValueError(
                    f"{name} shape {actual} does not match expected shape {self.shape_cell}."
                )
        for name in ("eps_r_a", "eps_r_w", "mu_r_t"):
            actual = getattr(self, name).shape
            if actual != self.shape_node:
                raise ValueError(
                    f"{name} shape {actual} does not match expected shape {self.shape_node}."
                )

    def _init_operators(self):
        rows = np.repeat(np.arange(self.N), 2)
        cols = np.column_stack((np.arange(1, self.N + 1), np.arange(self.N))).ravel()
        data = np.tile((1.0 / self.normalized_dt, -1.0 / self.normalized_dt), self.N)
        self.D_NODE_TO_CELL = coo_matrix((data, (rows, cols)), shape=(self.N, self.N + 1)).tocsr()
        self.D_CELL_TO_NODE = -self.D_NODE_TO_CELL.conj().T

    @staticmethod
    def _diag(values):
        return diags(np.asarray(values).ravel(), format="csr")

    @staticmethod
    def _inverse_diag_on_free(values, constrained_mask):
        inverse = np.zeros_like(values, dtype=np.complex128)
        free = ~np.asarray(constrained_mask, dtype=bool)
        inverse[free] = 1.0 / values[free]
        return diags(inverse, format="csr")

    def _solve_reduced(self, operator, free_mask):
        reduced = operator[free_mask, :][:, free_mask]
        size = reduced.shape[0]
        if size <= self.num_modes:
            raise ValueError(
                f"Not enough unconstrained scalar DOFs ({size}) to solve {self.num_modes} modes."
            )

        if size <= self.num_modes + 1:
            values, vectors = eig(reduced.toarray())
            order = np.argsort(np.abs(values - self.guess))[: self.num_modes]
            values = values[order]
            vectors = vectors[:, order]
        else:
            # ARPACK otherwise creates an implicit random starting vector.
            # That makes an identical port solve depend on unrelated prior
            # uses of the process-wide random-number generator and can lead
            # to small run/restart differences after modal normalisation.
            # Use a local deterministic vector without mutating NumPy's
            # global random state.
            v0 = np.random.default_rng(0).standard_normal(size)
            try:
                values, vectors = eigs(
                    reduced,
                    k=self.num_modes,
                    sigma=self.guess,
                    v0=v0,
                )
            except RuntimeError:
                # A homogeneous fundamental mode can lie exactly at the
                # material-derived default shift (for example sigma=-1).
                # Move the shift by roundoff rather than failing LU
                # factorisation of A - sigma I.
                shifted_guess = self.guess * (1.0 + 1e-9) - 1e-12
                values, vectors = eigs(
                    reduced,
                    k=self.num_modes,
                    sigma=shifted_guess,
                    v0=v0,
                )

        order = np.argsort(np.real(values))
        values = values[order]
        vectors = vectors[:, order]
        expanded = np.zeros((free_mask.size, self.num_modes), dtype=np.complex128)
        expanded[free_mask, :] = vectors
        return values, expanded

    def solve(self):
        if self.polarization == "TM":
            longitudinal_inverse = self._inverse_diag_on_free(self.mu_r_w, self.pmc_w_mask)
            operator = -self._diag(self.mu_r_t) @ (
                self.D_CELL_TO_NODE @ longitudinal_inverse @ self.D_NODE_TO_CELL
                + self._diag(self.eps_r_a)
            )
            free_scalar = ~self.pec_a_mask
        else:
            longitudinal_inverse = self._inverse_diag_on_free(self.eps_r_w, self.pec_w_mask)
            operator = -self._diag(self.eps_r_t) @ (
                self.D_NODE_TO_CELL @ longitudinal_inverse @ self.D_CELL_TO_NODE
                + self._diag(self.mu_r_a)
            )
            free_scalar = ~self.pmc_a_mask

        self.eigenvalues, self.eigenvectors = self._solve_reduced(operator, free_scalar)
        self.operator_neff = self._passive_positive_neff(-self.eigenvalues)
        self._calculate_fields(longitudinal_inverse)
        self._zero_constrained_fields()
        self._orient_backward_modes_to_forward_power(longitudinal_inverse)
        wavenumber = self.operator_k0 * self.operator_neff
        self.beta = phase_propagation_constant(wavenumber, self.propagation_spacing)
        self.complex_neff = self.beta / self.k0
        self.real_neff = np.real(self.complex_neff)
        self.spatially_resolved = spatially_resolved(wavenumber, self.propagation_spacing)
        self._normalize_modes_to_power()
        self._align_modes_for_real_profile_power()
        self._set_modal_fields()

    def _calculate_fields(self, longitudinal_inverse):
        shape_cell_modes = (self.N, self.num_modes)
        shape_node_modes = (self.N + 1, self.num_modes)
        self.Et = np.zeros(shape_cell_modes, dtype=np.complex128)
        self.Ea = np.zeros(shape_node_modes, dtype=np.complex128)
        self.Ew = np.zeros(shape_node_modes, dtype=np.complex128)
        self.Ht = np.zeros(shape_node_modes, dtype=np.complex128)
        self.Ha = np.zeros(shape_cell_modes, dtype=np.complex128)
        self.Hw = np.zeros(shape_cell_modes, dtype=np.complex128)

        for mode in range(self.num_modes):
            neff = self.operator_neff[mode]
            if self.polarization == "TM":
                self.Ea[:, mode] = self.eigenvectors[:, mode]
                self.Ht[:, mode] = -neff * self.Ea[:, mode] / (self.eta0 * self.mu_r_t)
                self.Hw[:, mode] = np.asarray(
                    1j
                    * (longitudinal_inverse @ (self.D_NODE_TO_CELL @ self.Ea[:, mode]))
                    / self.eta0
                ).ravel()
            else:
                self.Ha[:, mode] = self.eigenvectors[:, mode]
                self.Et[:, mode] = self.eta0 * neff * self.Ha[:, mode] / self.eps_r_t
                self.Ew[:, mode] = np.asarray(
                    -1j
                    * self.eta0
                    * (longitudinal_inverse @ (self.D_CELL_TO_NODE @ self.Ha[:, mode]))
                ).ravel()

    def _zero_constrained_fields(self):
        self.Et[self.pec_t_mask, :] = 0.0
        self.Ea[self.pec_a_mask, :] = 0.0
        self.Ew[self.pec_w_mask, :] = 0.0
        self.Ht[self.pmc_t_mask, :] = 0.0
        self.Ha[self.pmc_a_mask, :] = 0.0
        self.Hw[self.pmc_w_mask, :] = 0.0

    @staticmethod
    def _nodes_to_cells(field):
        return 0.5 * (field[:-1] + field[1:])

    def _calculate_mode_complex_power(self, mode):
        if self.polarization == "TM":
            ea = self._nodes_to_cells(self.Ea[:, mode])
            ht = self._nodes_to_cells(self.Ht[:, mode])
            flux = -ea * np.conj(ht)
        else:
            flux = self.Et[:, mode] * np.conj(self.Ha[:, mode])
        return (
            0.5
            * self.dt
            * complex(
                math.fsum(np.ravel(np.real(flux))),
                math.fsum(np.ravel(np.imag(flux))),
            )
        )

    def _calculate_mode_power(self, mode):
        return float(np.real(self._calculate_mode_complex_power(mode)))

    def _calculate_mode_balanced_power(self, mode):
        """Return a positive E/H-balanced field scale with units of power."""
        if self.polarization == "TM":
            electric = self._nodes_to_cells(self.Ea[:, mode])
            magnetic = self._nodes_to_cells(self.Ht[:, mode])
        else:
            electric = self.Et[:, mode]
            magnetic = self.Ha[:, mode]
        density = (np.square(np.abs(electric)) + self.eta0**2 * np.square(np.abs(magnetic))) / (
            4.0 * self.eta0
        )
        return math.fsum(np.ravel(density)) * self.dt

    def _orient_backward_modes_to_forward_power(self, longitudinal_inverse):
        """Select the passive beta branch whose real power points forward.

        Negative-index media can carry energy opposite to their phase vector.
        Reversing ``neff`` and reconstructing every dependent field selects
        the forward-energy solution without the inconsistent H-only flip used
        by the former normalization guard.
        """

        reverse = np.zeros(self.num_modes, dtype=bool)
        for mode in range(self.num_modes):
            balanced_power = self._calculate_mode_balanced_power(mode)
            if not np.isfinite(balanced_power) or balanced_power <= 0:
                continue
            metric = np.real(self._calculate_mode_complex_power(mode)) / balanced_power
            candidate = -self.operator_neff[mode]
            tolerance = 1e-12 * max(1.0, abs(candidate))
            reverse[mode] = (
                np.isfinite(metric)
                and metric < -self.FORWARD_POWER_METRIC_TOLERANCE
                and np.imag(candidate) <= tolerance
            )
        if not np.any(reverse):
            return

        self.operator_neff[reverse] *= -1
        self._calculate_fields(longitudinal_inverse)
        self._zero_constrained_fields()

    def _real_profile_power_from_fields(self, mode):
        if self.polarization == "TM":
            ea = self._nodes_to_cells(np.real(self.Ea[:, mode]))
            ht = self._nodes_to_cells(np.real(self.Ht[:, mode]))
            flux = -ea * ht
        else:
            flux = np.real(self.Et[:, mode]) * np.real(self.Ha[:, mode])
        return math.fsum(np.ravel(flux)) * self.dt

    def _normalize_modes_to_power(self, target_power_per_metre=1.0):
        """Power-normalize forward modes and safely L2-normalize the rest."""
        self.raw_powers = np.zeros(self.num_modes, dtype=np.complex128)
        self.forward_power_metrics = np.full(self.num_modes, np.nan, dtype=np.float64)
        self.power_valid = np.zeros(self.num_modes, dtype=bool)
        self.powers = np.zeros(self.num_modes, dtype=np.float64)
        for mode in range(self.num_modes):
            raw_power = self._calculate_mode_complex_power(mode)
            balanced_power = self._calculate_mode_balanced_power(mode)
            self.raw_powers[mode] = raw_power
            if np.isfinite(raw_power) and np.isfinite(balanced_power) and balanced_power > 0:
                metric = float(np.real(raw_power) / balanced_power)
                self.forward_power_metrics[mode] = metric
                self.power_valid[mode] = (
                    np.isfinite(metric)
                    and metric > self.FORWARD_POWER_METRIC_TOLERANCE
                    and self.spatially_resolved[mode]
                )
            if not np.isfinite(balanced_power) or balanced_power <= 0:
                raise ValueError(
                    f"Cannot normalize mode {mode}: balanced field power is {balanced_power}."
                )

            normalization_power = (
                float(np.real(raw_power)) if self.power_valid[mode] else balanced_power
            )
            scale = np.sqrt(target_power_per_metre / normalization_power)
            for field in (self.Et, self.Ea, self.Ew, self.Ht, self.Ha, self.Hw):
                field[:, mode] *= scale
            self.powers[mode] = self._calculate_mode_power(mode)

    def _align_modes_for_real_profile_power(self):
        active = (self.Ea, self.Ht) if self.polarization == "TM" else (self.Et, self.Ha)
        for mode in range(self.num_modes):
            values = np.concatenate((active[0][:, mode].ravel(), active[1][:, mode].ravel()))
            phase = -0.5 * np.angle(np.sum(values**2))
            factor = np.exp(1j * phase)
            for field in (self.Et, self.Ea, self.Ew, self.Ht, self.Ha, self.Hw):
                field[:, mode] *= factor
            if self._real_profile_power_from_fields(mode) < 0:
                for field in (self.Et, self.Ea, self.Ew, self.Ht, self.Ha, self.Hw):
                    field[:, mode] *= 1j
            self._canonicalize_mode_sign(mode)

    def _canonicalize_mode_sign(self, mode):
        """Fix the remaining plus/minus gauge using tangential electric fields."""
        pivot_vector = np.concatenate((self.Et[:, mode].ravel(), self.Ea[:, mode].ravel()))
        pivot = pivot_vector[np.argmax(np.abs(pivot_vector))]
        tolerance = 1e-12 * max(1.0, abs(pivot))
        if np.real(pivot) < -tolerance or (abs(np.real(pivot)) <= tolerance and np.imag(pivot) < 0):
            for field in (self.Et, self.Ea, self.Ew, self.Ht, self.Ha, self.Hw):
                field[:, mode] *= -1

    def _set_modal_fields(self):
        mode = self.mode_index
        self.modal_Et = self.Et[:, mode]
        self.modal_Ea = self.Ea[:, mode]
        self.modal_Ew = self.Ew[:, mode]
        self.modal_Ht = self.Ht[:, mode]
        self.modal_Ha = self.Ha[:, mode]
        self.modal_Hw = self.Hw[:, mode]
        self.modal_complex_neff = self.complex_neff[mode]
        self.modal_real_neff = self.real_neff[mode]
        self.modal_raw_power = self.raw_powers[mode]
        self.modal_forward_power_metric = self.forward_power_metrics[mode]
        self.modal_power_valid = self.power_valid[mode]
        self.modal_power = self.powers[mode]

    def plot_fields(self, output_path="fdfd_1d_modes.png"):
        """Plot the three active fields for every solved mode as line profiles."""
        if self.eigenvalues is None:
            raise RuntimeError("solve() must be called before plot_fields().")
        if self.polarization == "TM":
            fields = ((self.Ea, "E_a", "node"), (self.Ht, "H_t", "node"), (self.Hw, "H_w", "cell"))
        else:
            fields = ((self.Et, "E_t", "cell"), (self.Ha, "H_a", "cell"), (self.Ew, "E_w", "node"))

        fig = Figure(figsize=(12, 3 * self.num_modes), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(self.num_modes, 3, squeeze=False)
        node_coordinate = np.arange(self.N + 1) * self.dt
        cell_coordinate = (np.arange(self.N) + 0.5) * self.dt
        for mode in range(self.num_modes):
            e_norm = max(
                np.max(np.abs(field[:, mode]))
                for field, label, _ in fields
                if label.startswith("E")
            )
            h_norm = max(
                np.max(np.abs(field[:, mode]))
                for field, label, _ in fields
                if label.startswith("H")
            )
            for ax, (field, label, stagger) in zip(axes[mode], fields):
                coordinate = node_coordinate if stagger == "node" else cell_coordinate
                norm = e_norm if label.startswith("E") else h_norm
                profile = field[:, mode] / norm if norm > 0 else field[:, mode]
                ax.plot(coordinate, np.real(profile), label=f"Re({label})")
                ax.plot(coordinate, np.abs(profile), "--", label=f"|{label}|")
                ax.set_title(
                    f"Mode {mode + 1}: {label} ({stagger}), " f"neff={self.complex_neff[mode]:.6g}"
                )
                ax.set_xlabel("t (m)")
                ax.set_ylabel("normalised field")
                ax.set_ylim(-1.05, 1.05)
                ax.grid(True)
                ax.legend(fontsize=8)
        fig.savefig(output_path, dpi=200)
        return output_path

    @staticmethod
    def _passive_positive_neff(neff_squared):
        root = np.sqrt(neff_squared)
        tolerance = 1e-12 * np.maximum(1.0, np.abs(root))
        flip = (np.real(root) < -tolerance) | (
            (np.abs(np.real(root)) <= tolerance) & (np.imag(root) > tolerance)
        )
        neff = np.where(flip, -root, root)

        real = np.real(neff)
        imag = np.imag(neff)
        real = np.where(np.abs(real) <= tolerance, 0.0, real)
        imag = np.where(np.abs(imag) <= tolerance, 0.0, imag)
        return real + 1j * imag

    @staticmethod
    def _max_magnitude(values):
        magnitude = np.abs(np.asarray(values))
        finite = magnitude[np.isfinite(magnitude)]
        return float(np.max(finite)) if finite.size else 0.0

    def _default_guess(self):
        max_epsilon = max(
            self._max_magnitude(values)
            for values in (
                self.eps_r_t,
                self.eps_r_a,
                self.eps_r_w,
            )
        )
        max_permeability = max(
            self._max_magnitude(values)
            for values in (
                self.mu_r_t,
                self.mu_r_a,
                self.mu_r_w,
            )
        )
        return -(max_epsilon * max_permeability)
