import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.linalg import eig
from scipy.sparse import bmat, coo_matrix, diags
from scipy.sparse.linalg import eigs

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import (
    BoundaryAmpereRow,
    BoundaryMagneticTerm,
    FDFDSurfaceBoundary,
)


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

    Electric PEC and magnetic PMC masks constrain the corresponding component
    DOFs. A PEC tangential-E constraint also constrains the collocated,
    surface-normal transverse H component, and a PMC tangential-H constraint
    likewise constrains the collocated, surface-normal transverse E component.
    Non-finite electric and magnetic material entries are interpreted as PEC
    and PMC respectively, then replaced by finite placeholders after the masks
    have been built.

    ``surface_boundary`` supplies impedance-volume topology independently of
    PEC/PMC masks. Its retained-component masks remove interior volume DOFs,
    while each integral Ampere row replaces the corresponding standard curl
    row and electric material coefficient.

    After :meth:`solve`, ``raw_powers`` contains the complex Poynting power
    before normalization. ``forward_power_metrics`` is its real part divided
    by a positive, E/H-balanced transverse field norm; ``power_valid`` applies
    the class tolerance to that signed, scale-independent ratio.
    """

    FORWARD_POWER_METRIC_TOLERANCE = 1e-8

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
        surface_boundary=None,
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
        self.pmc_u_mask = self._component_constraint_mask(self.mu_r_uu, pmc_u_mask, self.shape_hu)
        self.pmc_v_mask = self._component_constraint_mask(self.mu_r_vv, pmc_v_mask, self.shape_hv)
        self.pmc_w_mask = self._component_constraint_mask(self.mu_r_ww, pmc_w_mask, self.shape_hw)

        # H_u is collocated with E_v and is normal to a u-oriented PEC face;
        # H_v is collocated with E_u and is normal to a v-oriented PEC face.
        # Keep tangential H unconstrained so it can represent PEC surface current.
        self.hu_constraint_mask = self.pmc_u_mask | self.pec_v_mask
        self.hv_constraint_mask = self.pmc_v_mask | self.pec_u_mask

        # By electromagnetic duality, tangential PMC H_v constrains normal E_u,
        # and tangential PMC H_u constrains normal E_v. Tangential E remains free.
        self.eu_constraint_mask = self.pec_u_mask | self.pmc_v_mask
        self.ev_constraint_mask = self.pec_v_mask | self.pmc_u_mask

        self.surface_boundary = surface_boundary
        self._prepare_surface_boundary()

        self.eps_r_uu[self.pec_u_mask] = 1.0 + 0j
        self.eps_r_vv[self.pec_v_mask] = 1.0 + 0j
        self.eps_r_ww[self.pec_w_mask] = 1.0 + 0j
        self.mu_r_uu[self.pmc_u_mask] = 1.0 + 0j
        self.mu_r_vv[self.pmc_v_mask] = 1.0 + 0j
        self.mu_r_ww[self.pmc_w_mask] = 1.0 + 0j

        self.free_eu_mask = self.surface_electric_retained[0].ravel(order="F").copy()
        self.free_ev_mask = self.surface_electric_retained[1].ravel(order="F").copy()
        self.free_ew_mask = self.surface_electric_retained[2].ravel(order="F").copy()
        self.free_hu_mask = self.surface_magnetic_retained[0].ravel(order="F").copy()
        self.free_hv_mask = self.surface_magnetic_retained[1].ravel(order="F").copy()
        self.free_hw_mask = self.surface_magnetic_retained[2].ravel(order="F").copy()
        self.free_eu_mask &= ~self.pec_u_mask.ravel(order="F")
        self.free_ev_mask &= ~self.pec_v_mask.ravel(order="F")
        self.free_ew_mask &= ~self.pec_w_mask.ravel(order="F")
        self.free_hu_mask &= ~self.pmc_u_mask.ravel(order="F")
        self.free_hv_mask &= ~self.pmc_v_mask.ravel(order="F")
        self.free_hw_mask &= ~self.pmc_w_mask.ravel(order="F")
        self.free_eu_mask &= ~self.pmc_v_mask.ravel(order="F")
        self.free_ev_mask &= ~self.pmc_u_mask.ravel(order="F")
        self.free_hu_mask &= ~self.pec_v_mask.ravel(order="F")
        self.free_hv_mask &= ~self.pec_u_mask.ravel(order="F")
        self.free_euv_mask = np.concatenate((self.free_eu_mask, self.free_ev_mask))
        self.free_huv_mask = np.concatenate((self.free_hu_mask, self.free_hv_mask))

        self.guess = guess if guess is not None else self._default_guess()
        self.eigenvalues = None
        self.eigenvectors = None
        self.complex_neff = None
        self.real_neff = None
        self.raw_powers = None
        self.forward_power_metrics = None
        self.power_valid = None
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
                raise ValueError(
                    f"{name} shape {actual} does not match expected local Yee shape {shape}."
                )

    def _component_constraint_mask(self, values, explicit_mask, expected_shape):
        mask = ~np.isfinite(values)
        if explicit_mask is not None:
            explicit_mask = np.asarray(explicit_mask, dtype=bool)
            if explicit_mask.shape != expected_shape:
                raise ValueError(
                    f"Constraint mask shape {explicit_mask.shape} does not match expected shape {expected_shape}."
                )
            mask |= explicit_mask
        return mask

    def _prepare_surface_boundary(self):
        """Validate and install impedance-volume topology and boundary rows."""
        electric_shapes = (self.shape_eu, self.shape_ev, self.shape_ew)
        magnetic_shapes = (self.shape_hu, self.shape_hv, self.shape_hw)
        if self.surface_boundary is None:
            self.surface_electric_retained = tuple(
                np.ones(shape, dtype=bool) for shape in electric_shapes
            )
            self.surface_magnetic_retained = tuple(
                np.ones(shape, dtype=bool) for shape in magnetic_shapes
            )
            self.surface_boundary_rows = ()
            return
        if not isinstance(self.surface_boundary, FDFDSurfaceBoundary):
            raise TypeError("surface_boundary must be an FDFDSurfaceBoundary")

        self.surface_electric_retained = self._validated_retained_masks(
            self.surface_boundary.electric_retained,
            electric_shapes,
            "electric",
        )
        self.surface_magnetic_retained = self._validated_retained_masks(
            self.surface_boundary.magnetic_retained,
            magnetic_shapes,
            "magnetic",
        )

        electric_constraints = (
            self.eu_constraint_mask,
            self.ev_constraint_mask,
            self.pec_w_mask,
        )
        magnetic_constraints = (
            self.hu_constraint_mask,
            self.hv_constraint_mask,
            self.pmc_w_mask,
        )
        allowed_magnetic_axes = ({2}, {2}, {0, 1})
        rows = []
        occupied = set()
        for row in self.surface_boundary.rows:
            if not isinstance(row, BoundaryAmpereRow):
                raise TypeError("surface boundary rows must be BoundaryAmpereRow objects")
            axis = self._validated_axis(row.electric_axis, "electric")
            index = self._validated_component_index(
                row.electric_index,
                electric_shapes[axis],
                f"surface electric axis {axis}",
            )
            key = (axis, index)
            if key in occupied:
                raise ValueError(f"duplicate surface Ampere row for electric component {key}")
            occupied.add(key)
            if not self.surface_electric_retained[axis][index]:
                raise ValueError(f"surface Ampere row {key} is marked as excluded")
            if electric_constraints[axis][index]:
                raise ValueError(f"surface Ampere row {key} conflicts with a PEC/PMC constraint")

            area = float(row.retained_dual_area)
            relative_permittivity = complex(row.relative_permittivity)
            if not np.isfinite(area) or area <= 0:
                raise ValueError(f"surface Ampere row {key} has invalid retained dual area")
            if not np.isfinite(relative_permittivity):
                raise ValueError(f"surface Ampere row {key} has non-finite permittivity")
            if not row.magnetic_terms:
                raise ValueError(f"surface Ampere row {key} has no magnetic circulation terms")

            term_weights = {}
            for term in row.magnetic_terms:
                if not isinstance(term, BoundaryMagneticTerm):
                    raise TypeError(
                        "surface boundary magnetic terms must be BoundaryMagneticTerm objects"
                    )
                magnetic_axis = self._validated_axis(term.axis, "magnetic")
                if magnetic_axis not in allowed_magnetic_axes[axis]:
                    raise ValueError(
                        f"surface electric axis {axis} cannot use magnetic axis {magnetic_axis}"
                    )
                magnetic_index = self._validated_component_index(
                    term.index,
                    magnetic_shapes[magnetic_axis],
                    f"surface magnetic axis {magnetic_axis}",
                )
                line_weight = float(term.line_weight)
                if not np.isfinite(line_weight) or line_weight == 0:
                    raise ValueError(f"surface Ampere row {key} has invalid line weight")
                if not self.surface_magnetic_retained[magnetic_axis][magnetic_index]:
                    raise ValueError(
                        f"surface Ampere row {key} references an excluded magnetic component"
                    )
                if magnetic_constraints[magnetic_axis][magnetic_index]:
                    raise ValueError(
                        f"surface Ampere row {key} references a PEC/PMC-constrained magnetic component"
                    )
                term_key = (magnetic_axis, magnetic_index)
                term_weights[term_key] = term_weights.get(term_key, 0.0) + line_weight

            terms = tuple(
                BoundaryMagneticTerm(axis, index, line_weight)
                for (axis, index), line_weight in sorted(term_weights.items())
                if line_weight != 0
            )
            if not terms:
                raise ValueError(f"surface Ampere row {key} has zero net magnetic circulation")

            canonical_row = BoundaryAmpereRow(
                electric_axis=axis,
                electric_index=index,
                retained_dual_area=area,
                relative_permittivity=relative_permittivity,
                magnetic_terms=terms,
            )
            rows.append(canonical_row)
            (self.eps_r_uu, self.eps_r_vv, self.eps_r_ww)[axis][index] = relative_permittivity
        self.surface_boundary_rows = tuple(rows)

    @staticmethod
    def _validated_retained_masks(masks, expected_shapes, field_kind):
        if len(masks) != 3:
            raise ValueError(f"surface {field_kind} retained masks must contain three components")
        validated = []
        for axis, (mask, shape) in enumerate(zip(masks, expected_shapes)):
            array = np.asarray(mask, dtype=bool)
            if array.shape != shape:
                raise ValueError(
                    f"surface {field_kind} axis {axis} retained mask shape {array.shape} "
                    f"does not match expected shape {shape}"
                )
            validated.append(array.copy())
        return tuple(validated)

    @staticmethod
    def _validated_axis(axis, field_kind):
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
            raise ValueError(f"surface {field_kind} axis must be an integer in [0, 2]")
        axis = int(axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"surface {field_kind} axis must be an integer in [0, 2]")
        return axis

    @staticmethod
    def _validated_component_index(index, shape, label):
        if len(index) != 2:
            raise ValueError(f"{label} index must contain two entries")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in index
        ):
            raise ValueError(f"{label} index entries must be integers")
        index = (int(index[0]), int(index[1]))
        if not (0 <= index[0] < shape[0] and 0 <= index[1] < shape[1]):
            raise ValueError(f"{label} index {index} is outside shape {shape}")
        return index

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
                entries = (
                    ((i + 1, j, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i - 1, j, -1.0))
                )
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nu))
                        data.append(value / scale)
        return coo_matrix(
            (data, (rows, cols)), shape=(out_nu * out_nv, in_shape[0] * in_shape[1])
        ).tocsr()

    def _difference_matrix_v(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nu, _ = in_shape
        out_nu, out_nv = out_shape
        for j in range(out_nv):
            for i in range(out_nu):
                row = self._flat_index(i, j, out_nu)
                entries = (
                    ((i, j + 1, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i, j - 1, -1.0))
                )
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nu))
                        data.append(value / scale)
        return coo_matrix(
            (data, (rows, cols)), shape=(out_nu * out_nv, in_shape[0] * in_shape[1])
        ).tocsr()

    def _init_operators(self):
        du = self.normalized_du
        dv = self.normalized_dv

        self.DEU_EW_TO_EU = self._difference_matrix_u(
            self.shape_ew, self.shape_eu, du, forward=True
        )
        self.DEV_EW_TO_EV = self._difference_matrix_v(
            self.shape_ew, self.shape_ev, dv, forward=True
        )
        self.DEU_EV_TO_HW = self._difference_matrix_u(
            self.shape_ev, self.shape_hw, du, forward=True
        )
        self.DEV_EU_TO_HW = self._difference_matrix_v(
            self.shape_eu, self.shape_hw, dv, forward=True
        )

        self.DHU_HV_TO_EW = -self.DEU_EW_TO_EU.conj().T
        self.DHV_HU_TO_EW = -self.DEV_EW_TO_EV.conj().T
        self.DHU_HW_TO_HU = -self.DEU_EV_TO_HW.conj().T
        self.DHV_HW_TO_HV = -self.DEV_EU_TO_HW.conj().T
        self._apply_surface_ampere_rows()

    def _apply_surface_ampere_rows(self):
        """Replace standard rectangular curl rows by clipped integral rows."""
        if not self.surface_boundary_rows:
            return

        replacements = {
            "DHU_HV_TO_EW": {},
            "DHV_HU_TO_EW": {},
            "DHU_HW_TO_HU": {},
            "DHV_HW_TO_HV": {},
        }
        for boundary_row in self.surface_boundary_rows:
            electric_axis = boundary_row.electric_axis
            electric_shape = (self.shape_eu, self.shape_ev, self.shape_ew)[electric_axis]
            row_index = self._flat_index(*boundary_row.electric_index, electric_shape[0])
            denominator = boundary_row.retained_dual_area * self.k0

            # A longitudinal electric row contains both transverse magnetic
            # derivatives; replacing only one would leave part of the old,
            # unclipped rectangular circulation behind.
            if electric_axis == 2:
                replacements["DHU_HV_TO_EW"][row_index] = {}
                replacements["DHV_HU_TO_EW"][row_index] = {}

            for term in boundary_row.magnetic_terms:
                coefficient = term.line_weight / denominator
                if electric_axis == 0:
                    matrix_name = "DHV_HW_TO_HV"
                    column_shape = self.shape_hw
                elif electric_axis == 1:
                    matrix_name = "DHU_HW_TO_HU"
                    column_shape = self.shape_hw
                    coefficient *= -1
                elif term.axis == 0:
                    matrix_name = "DHV_HU_TO_EW"
                    column_shape = self.shape_hu
                    coefficient *= -1
                else:
                    matrix_name = "DHU_HV_TO_EW"
                    column_shape = self.shape_hv
                column = self._flat_index(*term.index, column_shape[0])
                row_values = replacements[matrix_name].setdefault(row_index, {})
                row_values[column] = row_values.get(column, 0.0) + coefficient

        for matrix_name, rows in replacements.items():
            if rows:
                setattr(
                    self,
                    matrix_name,
                    self._replace_sparse_rows(getattr(self, matrix_name), rows),
                )

    @staticmethod
    def _replace_sparse_rows(matrix, replacements):
        editable = matrix.tolil(copy=True)
        for row, values in replacements.items():
            nonzero = sorted((column, value) for column, value in values.items() if value != 0)
            editable.rows[row] = [column for column, _ in nonzero]
            editable.data[row] = [value for _, value in nonzero]
        return editable.tocsr()

    @staticmethod
    def _diag(values):
        return diags(values.ravel(order="F"), format="csr")

    def _inverse_diag_on_free(self, values, free_mask):
        flat = values.ravel(order="F")
        inverse = np.zeros_like(flat, dtype=np.complex128)
        inverse[free_mask] = 1.0 / flat[free_mask]
        return diags(inverse, format="csr")

    def _solve_reduced(self, operator):
        size = operator.shape[0]
        if size <= self.num_modes:
            raise ValueError(
                f"Not enough unconstrained electric DOFs ({size}) to solve {self.num_modes} modes."
            )

        if size <= self.num_modes + 1:
            eigenvalues, eigenvectors = eig(operator.toarray())
            selection = np.argsort(np.abs(eigenvalues - self.guess))[: self.num_modes]
            eigenvalues = eigenvalues[selection]
            eigenvectors = eigenvectors[:, selection]
        else:
            try:
                eigenvalues, eigenvectors = eigs(
                    operator,
                    k=self.num_modes,
                    sigma=self.guess,
                )
            except RuntimeError:
                shifted_guess = self.guess * (1.0 + 1e-9) - 1e-12
                eigenvalues, eigenvectors = eigs(
                    operator,
                    k=self.num_modes,
                    sigma=shifted_guess,
                )

        order = np.argsort(np.real(eigenvalues))
        return eigenvalues[order], eigenvectors[:, order]

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
        eigenvalues, reduced_eigenvectors = self._solve_reduced(omega_matrix)
        eigenvectors = np.zeros((self.n_e_transverse, self.num_modes), dtype=np.complex128)
        eigenvectors[self.free_euv_mask, :] = reduced_eigenvectors

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.complex_neff = self._passive_positive_neff(-self.eigenvalues)
        self.real_neff = np.real(self.complex_neff)

        self._calculate_fields(Q_reduced, eps_ww_inv, mu_ww_inv)
        self._orient_backward_modes_to_forward_power(
            Q_reduced,
            eps_ww_inv,
            mu_ww_inv,
        )
        self._normalize_modes_to_power()
        self._align_modes_for_real_profile_power()
        self._set_modal_fields()

    def _calculate_fields(self, Q_reduced, eps_ww_inv, mu_ww_inv):
        # eigenvalue = -neff^2, so the branch consistent with
        # exp(+j*omega*t - j*beta*w) is sqrt(eigenvalue) = +j*neff.
        # Reuse the selected propagation branch when reconstructing H.
        sqrt_eigenvalues = 1j * self.complex_neff
        if np.any(np.abs(sqrt_eigenvalues) < 1e-300):
            raise ValueError(
                "Encountered a near-zero eigenvalue while reconstructing magnetic fields."
            )
        eigenvalues_inv = diags(1.0 / sqrt_eigenvalues, format="csr")

        eu_flat = np.asarray(self.eigenvectors[: self.n_eu, :], dtype=np.complex128)
        ev_flat = np.asarray(self.eigenvectors[self.n_eu :, :], dtype=np.complex128)
        huv_reduced = Q_reduced @ self.eigenvectors @ eigenvalues_inv
        huv_flat = np.zeros((self.n_h_transverse, self.num_modes), dtype=np.complex128)
        huv_flat[self.free_huv_mask, :] = huv_reduced
        hu_norm = np.asarray(huv_flat[: self.n_hu, :], dtype=np.complex128)
        hv_norm = np.asarray(huv_flat[self.n_hu :, :], dtype=np.complex128)
        ew_flat = np.asarray(
            eps_ww_inv @ (self.DHU_HV_TO_EW @ hv_norm - self.DHV_HU_TO_EW @ hu_norm),
            dtype=np.complex128,
        )
        hw_norm = np.asarray(
            mu_ww_inv @ (self.DEU_EV_TO_HW @ ev_flat - self.DEV_EU_TO_HW @ eu_flat),
            dtype=np.complex128,
        )

        self.Eu = self._unflatten_modes(eu_flat, self.shape_eu)
        self.Ev = self._unflatten_modes(ev_flat, self.shape_ev)
        self.Ew = self._unflatten_modes(ew_flat, self.shape_ew)
        self.Hu = self._unflatten_modes(1j * hu_norm / self.eta0, self.shape_hu)
        self.Hv = self._unflatten_modes(1j * hv_norm / self.eta0, self.shape_hv)
        self.Hw = self._unflatten_modes(1j * hw_norm / self.eta0, self.shape_hw)
        self._zero_constrained_fields()

    @staticmethod
    def _unflatten_modes(flat_modes, shape):
        return np.asarray(flat_modes, dtype=np.complex128).reshape(
            (*shape, flat_modes.shape[1]), order="F"
        )

    def _zero_constrained_fields(self):
        self.Eu[self.eu_constraint_mask | ~self.surface_electric_retained[0], :] = 0.0
        self.Ev[self.ev_constraint_mask | ~self.surface_electric_retained[1], :] = 0.0
        self.Ew[self.pec_w_mask | ~self.surface_electric_retained[2], :] = 0.0
        self.Hu[self.hu_constraint_mask | ~self.surface_magnetic_retained[0], :] = 0.0
        self.Hv[self.hv_constraint_mask | ~self.surface_magnetic_retained[1], :] = 0.0
        self.Hw[self.pmc_w_mask | ~self.surface_magnetic_retained[2], :] = 0.0

    def _set_modal_fields(self):
        self.modal_Eu = self.Eu[:, :, self.mode_index]
        self.modal_Ev = self.Ev[:, :, self.mode_index]
        self.modal_Ew = self.Ew[:, :, self.mode_index]
        self.modal_Hu = self.Hu[:, :, self.mode_index]
        self.modal_Hv = self.Hv[:, :, self.mode_index]
        self.modal_Hw = self.Hw[:, :, self.mode_index]
        self.modal_complex_neff = self.complex_neff[self.mode_index]
        self.modal_real_neff = self.real_neff[self.mode_index]
        self.modal_raw_power = self.raw_powers[self.mode_index]
        self.modal_forward_power_metric = self.forward_power_metrics[self.mode_index]
        self.modal_power_valid = self.power_valid[self.mode_index]
        self.modal_power = self.powers[self.mode_index]

    def _field_to_cells(self, field, component):
        if component in ("u", "hv"):
            return 0.5 * (field[:, : self.Nv] + field[:, 1:])
        if component in ("v", "hu"):
            return 0.5 * (field[: self.Nu, :] + field[1:, :])
        if component == "w_e":
            return 0.25 * (
                field[: self.Nu, : self.Nv]
                + field[1:, : self.Nv]
                + field[: self.Nu, 1:]
                + field[1:, 1:]
            )
        if component == "w_h":
            return field
        raise ValueError(f"Unknown component {component!r}.")

    def _mode_transverse_fields_on_cells(self, mode):
        eu = self._field_to_cells(self.Eu[:, :, mode], "u")
        ev = self._field_to_cells(self.Ev[:, :, mode], "v")
        hu = self._field_to_cells(self.Hu[:, :, mode], "hu")
        hv = self._field_to_cells(self.Hv[:, :, mode], "hv")
        return eu, ev, hu, hv

    def _calculate_mode_complex_power(self, mode):
        eu, ev, hu, hv = self._mode_transverse_fields_on_cells(mode)
        poynting_w = eu * np.conj(hv) - ev * np.conj(hu)
        return (
            0.5
            * self.du
            * self.dv
            * complex(
                math.fsum(np.ravel(np.real(poynting_w))),
                math.fsum(np.ravel(np.imag(poynting_w))),
            )
        )

    def _calculate_mode_power(self, mode):
        return float(np.real(self._calculate_mode_complex_power(mode)))

    def _calculate_mode_balanced_power(self, mode):
        """Return a positive E/H-balanced field scale with units of power."""
        eu, ev, hu, hv = self._mode_transverse_fields_on_cells(mode)
        density = (
            np.square(np.abs(eu))
            + np.square(np.abs(ev))
            + self.eta0**2 * (np.square(np.abs(hu)) + np.square(np.abs(hv)))
        ) / (4.0 * self.eta0)
        return math.fsum(np.ravel(density)) * self.du * self.dv

    def _orient_backward_modes_to_forward_power(
        self,
        Q_reduced,
        eps_ww_inv,
        mu_ww_inv,
    ):
        """Select the passive beta branch whose real power points forward.

        A backward-wave mode requires the opposite phase branch, not an
        H-only sign change. Reconstructing the fields after reversing ``neff``
        preserves Maxwell consistency while orienting real energy flow along
        the solver's forward axis.
        """

        reverse = np.zeros(self.num_modes, dtype=bool)
        for mode in range(self.num_modes):
            balanced_power = self._calculate_mode_balanced_power(mode)
            if not np.isfinite(balanced_power) or balanced_power <= 0:
                continue
            metric = np.real(self._calculate_mode_complex_power(mode)) / balanced_power
            candidate = -self.complex_neff[mode]
            tolerance = 1e-12 * max(1.0, abs(candidate))
            reverse[mode] = (
                np.isfinite(metric)
                and metric < -self.FORWARD_POWER_METRIC_TOLERANCE
                and np.imag(candidate) <= tolerance
            )
        if not np.any(reverse):
            return

        self.complex_neff[reverse] *= -1
        self.real_neff = np.real(self.complex_neff)
        self._calculate_fields(Q_reduced, eps_ww_inv, mu_ww_inv)

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
                    np.isfinite(metric) and metric > self.FORWARD_POWER_METRIC_TOLERANCE
                )
            if not np.isfinite(balanced_power) or balanced_power <= 0:
                raise ValueError(
                    f"Cannot normalize mode {mode}: balanced field power is {balanced_power}."
                )

            normalization_power = (
                float(np.real(raw_power)) if self.power_valid[mode] else balanced_power
            )
            scale = np.sqrt(target_power / normalization_power)
            for field in (self.Eu, self.Ev, self.Ew, self.Hu, self.Hv, self.Hw):
                field[:, :, mode] *= scale
            self.powers[mode] = self._calculate_mode_power(mode)

    def _align_modes_for_real_profile_power(self):
        for mode in range(self.num_modes):
            eu = self.Eu[:, :, mode]
            ev = self.Ev[:, :, mode]
            hu = self.Hu[:, :, mode]
            hv = self.Hv[:, :, mode]
            e_real_h_real = self._real_profile_power_from_fields(
                np.real(eu), np.real(ev), np.real(hu), np.real(hv)
            )
            e_imag_h_imag = self._real_profile_power_from_fields(
                np.imag(eu), np.imag(ev), np.imag(hu), np.imag(hv)
            )
            e_real_h_imag = self._real_profile_power_from_fields(
                np.real(eu), np.real(ev), np.imag(hu), np.imag(hv)
            )
            e_imag_h_real = self._real_profile_power_from_fields(
                np.imag(eu), np.imag(ev), np.real(hu), np.real(hv)
            )
            phase = 0.5 * np.arctan2(
                -0.5 * (e_real_h_imag + e_imag_h_real),
                0.5 * (e_real_h_real - e_imag_h_imag),
            )
            self._rotate_mode(mode, phase)
            if self._calculate_real_profile_power(mode) < 0:
                self._rotate_mode(mode, 0.5 * np.pi)
            self._canonicalize_mode_sign(mode)

    def _canonicalize_mode_sign(self, mode):
        """Fix the remaining plus/minus gauge using tangential electric fields."""
        pivot_vector = np.concatenate((self.Eu[:, :, mode].ravel(), self.Ev[:, :, mode].ravel()))
        pivot = pivot_vector[np.argmax(np.abs(pivot_vector))]
        tolerance = 1e-12 * max(1.0, abs(pivot))
        if np.real(pivot) < -tolerance or (abs(np.real(pivot)) <= tolerance and np.imag(pivot) < 0):
            for field in (self.Eu, self.Ev, self.Ew, self.Hu, self.Hv, self.Hw):
                field[:, :, mode] *= -1

    def _rotate_mode(self, mode, phase):
        phase_factor = np.exp(1j * phase)
        for field in (self.Eu, self.Ev, self.Ew, self.Hu, self.Hv, self.Hw):
            field[:, :, mode] *= phase_factor

    @staticmethod
    def _new_figure(figsize):
        """Build a Figure with its own Agg canvas, bypassing pyplot entirely.

        This keeps plot generation headless-safe without ever touching the
        process-wide matplotlib backend, so importing or using this solver
        cannot override a backend the host application already selected.
        """
        fig = Figure(figsize=figsize, constrained_layout=True)
        FigureCanvasAgg(fig)
        return fig

    def plot_e_fields(self, output_path="fdfd_modes_eu_ev.png"):
        fig = self._new_figure((8, 3 * self.num_modes))
        axes = np.atleast_2d(fig.subplots(self.num_modes, 2))
        for mode in range(self.num_modes):
            for ax, field, component in (
                (axes[mode, 0], np.real(self.Eu[:, :, mode]), "E_u"),
                (axes[mode, 1], np.real(self.Ev[:, :, mode]), "E_v"),
            ):
                image = ax.imshow(field.T, origin="lower", cmap="RdBu_r", aspect="auto")
                ax.set_title(f"Mode {mode + 1} {component}, neff={self.complex_neff[mode]:.6g}")
                ax.set_xlabel("u index")
                ax.set_ylabel("v index")
                fig.colorbar(image, ax=ax)
        fig.savefig(output_path, dpi=200)
        return output_path

    def plot_h_fields(self, output_path="fdfd_modes_hu_hv.png"):
        fig = self._new_figure((8, 3 * self.num_modes))
        axes = np.atleast_2d(fig.subplots(self.num_modes, 2))
        for mode in range(self.num_modes):
            for ax, field, component in (
                (axes[mode, 0], np.real(self.Hu[:, :, mode]), "H_u"),
                (axes[mode, 1], np.real(self.Hv[:, :, mode]), "H_v"),
            ):
                image = ax.imshow(field.T, origin="lower", cmap="RdBu_r", aspect="auto")
                ax.set_title(f"Mode {mode + 1} {component}, neff={self.complex_neff[mode]:.6g}")
                ax.set_xlabel("u index")
                ax.set_ylabel("v index")
                fig.colorbar(image, ax=ax)
        fig.savefig(output_path, dpi=200)
        return output_path

    def plot_pec_component_masks(self, output_path="fdfd_yee_uvw_pec_masks.png"):
        fig = self._new_figure((11, 3))
        axes = fig.subplots(1, 3)
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

    def _default_guess(self):
        max_epsilon = max(
            self._max_magnitude(values) for values in (self.eps_r_uu, self.eps_r_vv, self.eps_r_ww)
        )
        max_permeability = max(
            self._max_magnitude(values) for values in (self.mu_r_uu, self.mu_r_vv, self.mu_r_ww)
        )
        return -(max_epsilon * max_permeability)

    @staticmethod
    def _max_magnitude(values):
        finite = np.isfinite(values)
        if not np.any(finite):
            return 1.0
        return float(np.max(np.abs(values[finite])))
