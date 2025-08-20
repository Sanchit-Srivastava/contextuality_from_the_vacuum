'''This module generates the inequalities which define the faces of the Wigner polytop'''

import numpy as np
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from magic.phase_points import phase_point


# Precompute all phase point operators: shape (9, 3, 3) 
A_points = np.zeros((9, 3, 3), dtype=complex)
for i in range(3):
    for j in range(3):
        A_points[i * 3 + j] = phase_point(i, j)


# Function to test inequality violations
def wigner_inequalities(rho, tol=1e-12, include_sum: bool = False):
    """
    Return Wigner inequality violations for a qutrit state.

    Inequality checked: Re Tr[rho A_{i,j}] >= 0 for all phase-point operators A_{i,j}.

    Parameters
    ----------
    rho : numpy.ndarray
            Density matrix.
    tol : float
            Numerical tolerance; values >= -tol are treated as satisfying the inequality.
    include_sum : bool
            When True, also return the sum of the absolute values of all violations.

    Returns
    -------
    tuple
            If include_sum is False (default):
                (max_violation, violating_points)
            If include_sum is True:
                (max_violation, violating_points, sum_abs_violations)

            where
                - max_violation: float, the largest positive amount by which the inequality is violated (0 if none).
                - violating_points: list of ((i, j), value) for each violating phase point with value < -tol.
                - sum_abs_violations: float, sum over all violating points of abs(value) (i.e., -value for value < -tol).
    """
    # Expectation values (real up to numerical error)
    expectation_values = np.real(np.array([np.trace(rho @ A) for A in A_points]))

    # Identify violations (strictly below -tol)
    violations_mask = expectation_values < -tol
    if not np.any(violations_mask):
        return (0.0, [], 0.0) if include_sum else (0.0, [])

    # Map flat indices to (i, j) coordinates
    d = int(np.sqrt(A_points.shape[0]))
    violating_points = []
    for idx in np.flatnonzero(violations_mask):
        i, j = divmod(idx, d)
        violating_points.append(((i, j), expectation_values[idx]))

    # Maximum violation magnitude and total negativity (sum of absolute violations)
    neg_values = -expectation_values[violations_mask]
    max_violation = float(np.max(neg_values))
    sum_abs_violations = float(np.sum(neg_values))

    return (max_violation, violating_points, sum_abs_violations) if include_sum else (max_violation, violating_points)

# Backward-compatible alias for notebooks