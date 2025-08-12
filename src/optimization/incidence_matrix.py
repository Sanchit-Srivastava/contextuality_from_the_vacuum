"""Generating the incidence matrix for all possible global assignments"""

import numpy as np
from scipy.sparse import lil_matrix

try:
    from ..utils.contexts import A, B
    from ..utils.ternary import to_ternary
except ImportError:
    # Fall back to absolute imports when run directly
    from utils.contexts import A, B
    from utils.ternary import to_ternary

rows, cols = 360, 81  # rows = 9*40, cols = 3**4
M_sparse = lil_matrix((rows, cols), dtype=int)  # use lil_matrix for efficient construction

# Create incidence matrix
for g in range(cols):  # for each global assignment
    lam = to_ternary(g)  # convert column number to ternary representation
    for c in range(40):
        a = np.dot(A[c], lam) % 3  # compute outcome for A[c]
        b = np.dot(B[c], lam) % 3  # compute outcome for B[c]
        # position of the joint outcome (a,b) in context c
        row_index = 9*c + (3 * a + b)
        M_sparse[row_index, g] = 1

M = M_sparse.tocsr()  # convert to csr format for efficiency in calculations