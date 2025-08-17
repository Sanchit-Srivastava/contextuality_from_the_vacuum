'''This module contains the function to generate each face of the 
   simulable polytope of a single qutrit'''

import numpy as np

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from utils.state_checks import validate_and_print
from utils.operators import pauli
from utils.measurements import projector



def phase_point(x: int, y: int) -> np.ndarray:
    """
    Compute the qutrit phase-point operator A_{x,y} using modulo-3 arithmetic.
    Returns sum_j Π_j - I.

    Parameters
    ----------
    x,y : int
        Phase-space coordinates in Z3 multiplying the vectors a and b.

    Dependencies
    ------------
    pauli : callable
        pauli([c, d]) must return a square matrix representing the generalized Pauli  X^cZ^d.
    projector : callable
        projector(op, r) must return the projector onto the eigenspace of op labeled by r in Z3.
    Returns
    -------
    numpy.ndarray
    """

    a = np.array([1,0,1,2]) 
    b = np.array([0,-1,-1,-1])

    P = np.zeros((3, 3), dtype=complex)
    r = np.mod((x * a + y * b), 3)

    for j, rj in enumerate(r):
        term = projector(b[j], a[j], int(rj))
        P += term

    return P  - np.eye(P.shape[0], dtype=complex)