"""Generating the incidence matrix for all possible global assignments"""

import numpy as np

from utils.contexts import A, B
from utils.ternary import to_ternary

M = []
for g in range(80):
    G = []
    for c in range(40):
        lambda = to_ternary(g)
        a = np.dot(A[c], lambda)
        b = np.dot(B[c], lambda)
        gc = np.zeros((8,1))
        gc[3*a + b] = 1
        G.append(gc)
        G_column = np.vstack(G)
    M.append(G_column)
M = np.hstack(M)