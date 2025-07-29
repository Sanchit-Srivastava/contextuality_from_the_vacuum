"""Generating the incidence matrix for all possible global assignments"""

import numpy as np
from scipy.sparse import lil_matrix

from utils.contexts import A, B
from utils.ternary import to_ternary

rows, cols = 360, 81
M_sparse = lil_matrix((rows, cols), dtype=int)

M = []
for g in range(cols):
    lam = to_ternary(g)
    for c in range(40):
        a = np.dot(A[c], lam)
        b = np.dot(B[c], lam)
        row_index = 9*c + (3 * a + b)
        M_sparse[row_index, g] = 1

M = M_sparse.tocsr()