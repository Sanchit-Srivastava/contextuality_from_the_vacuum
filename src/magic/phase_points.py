'''This module contains the function to generate each face of the 
   simulable polytope of a single qutrit'''

import numpy as np

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from utils.state_checks import validate_and_print
from utils.operators import pauli
from utils.measurements import projector


a = np.array([1,0,1,2]) 
b = -1*np.array([0,1,1,1])

def phase_point(x: int, y: int) -> np.ndarray:
    """
    Compute the (unnormalized) qutrit phase-point operator A_{x,y} using modulo-3 arithmetic.

    Given integer phase-space coordinates (x, y) in Z3, this function builds a Hermitian
    operator by summing projectors onto eigenspaces of generalized Pauli operators and
    then subtracting the identity:
    - r_j = (x * a_j + y * b_j) mod 3
    - op_j = pauli([b_j mod 3, a_j mod 3])
    - Π_j = projector(op_j, r_j)
    Returns sum_j Π_j - I.

    Parameters
    ----------
    x : int
        Phase-space coordinate in Z3 multiplying the vector a (interpreted modulo 3).
    y : int
        Phase-space coordinate in Z3 multiplying the vector b (interpreted modulo 3).

    Dependencies
    ------------
    a, b : numpy.ndarray (1-D, integer-valued)
        Global arrays of equal length defining the stabilizer/frame coefficients for each term.
    pauli : callable
        pauli([c, d]) must return a square matrix representing the generalized Pauli  X^cZ^d.
    projector : callable
        projector(op, r) must return the projector onto the eigenspace of op labeled by r in Z3.
    np : module
        NumPy is required (imported as `np`).

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    NameError
        If `a`, `b`, `pauli`, `projector`, or `np` are not defined in the module scope.
    ValueError
        If `a` and `b` do not have the same shape or if helper functions return incompatible shapes.
    TypeError
        If `x` or `y` are not integers or cannot participate in modulo-3 arithmetic.
    RuntimeError
        Propagated from `pauli`/`projector` if they fail to construct operators/projectors.

    Notes
    -----
    - All arithmetic is performed modulo 3 (qutrit case, prime odd dimension).
    - The returned operator is Hermitian by construction and equals (sum of commuting projectors) minus identity.
    - The overall normalization depends on the conventions of `projector`; this function does not rescale the result.

    Examples
    --------
    >>> # Given `a`, `b`, `pauli`, and `projector` are defined appropriately:
    >>> Axy = phase_point(1, 2)
    >>> Axy.shape  # matches the square dimension of the underlying Hilbert space
    """

    P = None
    a = np.array([1,0,1,2]) 
    b = -1*np.array([0,1,1,1])

    r = np.mod(x * a + y * b, 3)

    for j, rj in enumerate(r):
        term = projector(b[j], a[j], int(rj))
        if P is None:
            P = np.zeros_like(term, dtype=complex)
        P += term

    return P - np.eye(P.shape[0], dtype=complex)