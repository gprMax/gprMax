"""Residuals for the vacuum-normalized FDFD operators."""

import numpy as np


# The reduced eigenvalue is the negative squared operator effective index.
# This floor sets the absolute-defect scale near zero on that dimensionless
# operator scale, while retaining the relative test away from cutoff.
_CUTOFF_RESIDUAL_SCALE = 1e-2


def normalized_defect(applied, target, reference_norm):
    """Return columnwise defects with a floor relative to the field norm."""
    defect = np.linalg.norm(applied - target, axis=0)
    relative_scale = np.linalg.norm(applied, axis=0) + np.linalg.norm(target, axis=0)
    denominator = np.maximum(relative_scale, _CUTOFF_RESIDUAL_SCALE * reference_norm)
    valid = (reference_norm > 0) & np.isfinite(denominator) & np.isfinite(defect)
    return np.divide(defect, denominator, out=np.full_like(defect, np.inf), where=valid)


def eigenpair_residuals(operator, vectors, eigenvalues):
    """Check reduced vectors against their assigned dimensionless eigenvalues."""
    return normalized_defect(
        operator @ vectors,
        eigenvalues * vectors,
        np.linalg.norm(vectors, axis=0),
    )
