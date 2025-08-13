''' This module contains utility functions to validate quantum states'''

import numpy as np

def is_valid_state(rho: np.ndarray) -> bool:
    """Check if a given matrix is a valid density matrix.

    A valid density matrix must be:
    - Hermitian,
    - have a trace of 1,
    - be positive semidefinite.
    """
    # Ensure the matrix is Hermitian
    if not np.allclose(rho, rho.T.conj()):
        return False

    # Check if the trace equals 1
    if not np.isclose(np.trace(rho), 1.0):
        return False

    # Use eigvalsh for Hermitian matrices to check positive semidefiniteness
    try:
        eigenvalues = np.linalg.eigvalsh(rho)
    except np.linalg.LinAlgError:
        return False

    return bool(np.all(eigenvalues >= 0))
