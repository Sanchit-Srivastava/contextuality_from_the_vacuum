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
def wigner_ineq(rho):
    """
    Check the Wigner inequality for a given density matrix.

    The Wigner inequality states that for any valid quantum state,
    the expectation value of the phase point operators must satisfy certain bounds.

    Parameters
    ----------
    rho : numpy.ndarray
        The density matrix representing the quantum state.

    Returns
    -------
    bool
        True if the inequality is satisfied, False otherwise.
    """
    # Compute the expectation values of the phase point operators
    expectation_values = np.array([np.trace(rho @ A) for A in A_points])

    # Check the Wigner inequality
    # (This is a placeholder; the actual inequality conditions need to be implemented)
    if np.all(expectation_values >= 0):
        return True
    else:
        return False