"""Generating the incidence matrix for all possible global assignments"""

import numpy as np
from scipy.sparse import lil_matrix

from utils.contexts import A, B
from utils.ternary import to_ternary

rows, cols = 360, 81  # rows = 9*40, cols = 3**4 
# Use lil_matrix for efficient construction
M_sparse = lil_matrix((rows, cols), dtype=int)

M = []  # Initialize incidence matrix
for g in range(cols):  # For each global assignment
    lam = to_ternary(g)  # Convert column number to ternary representation
    for c in range(40):
        a = np.dot(A[c], lam) % 3  # Compute outcome for A[c]
        b = np.dot(B[c], lam) % 3  # Compute outcome for B[c]
        # Position of the joint outcome (a,b) in context c
        row_index = 9*c + (3 * a + b)
        M_sparse[row_index, g] = 1

M = M_sparse.tocsr()  # Convert to csr format for efficiency in calculations