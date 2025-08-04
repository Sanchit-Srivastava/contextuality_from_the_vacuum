"""Contexts for two qutrit Heisenberg-Weyl operators defined in terms of their symplectic vectors"""
import numpy as np

# Each context is specified by two symplectic vectors A and B.
# C[i,0] contains the A vector for context i
# C[i,1] contains the B vector for context i

C = np.array([
    # Context 1
    [[1,0,0,0], [0,0,1,0]],
    # Context 2
    [[1,0,0,0], [0,0,0,1]],
    # Context 3
    [[1,0,0,0], [0,0,1,1]],
    # Context 4
    [[1,0,0,0], [0,0,1,2]],
    # Context 5
    [[0,1,0,0], [0,0,1,0]],
    # Context 6
    [[0,1,0,0], [0,0,0,1]],
    # Context 7
    [[0,1,0,0], [0,0,1,1]],
    # Context 8
    [[0,1,0,0], [0,0,1,2]],
    # Context 9
    [[1,1,0,0], [0,0,1,0]],
    # Context 10
    [[1,1,0,0], [0,0,0,1]],
    # Context 11
    [[1,1,0,0], [0,0,1,1]],
    # Context 12
    [[1,1,0,0], [0,0,1,2]],
    # Context 13
    [[1,2,0,0], [0,0,1,0]],
    # Context 14
    [[1,2,0,0], [0,0,0,1]],
    # Context 15
    [[1,2,0,0], [0,0,1,1]],
    # Context 16
    [[1,2,0,0], [0,0,1,2]],
    # Context 17
    [[1,0,0,1], [0,1,1,0]],
    # Context 18
    [[1,0,0,1], [0,1,1,1]],
    # Context 19
    [[1,0,0,1], [0,1,1,2]],
    # Context 20
    [[0,1,1,0], [1,0,1,1]],
    # Context 21
    [[0,1,1,0], [1,0,2,1]],
    # Context 22
    [[0,1,1,1], [1,1,0,1]],
    # Context 23
    [[0,1,1,1], [1,1,2,0]],
    # Context 24
    [[0,1,1,2], [1,2,1,0]],
    # Context 25
    [[0,1,1,2], [1,2,0,1]],
    # Context 26
    [[1,0,1,1], [1,1,1,0]],
    # Context 27
    [[1,0,1,1], [1,1,0,2]],
    # Context 28
    [[1,0,2,1], [1,2,2,0]],
    # Context 29
    [[1,0,2,1], [1,2,0,2]],
    # Context 30
    [[1,1,2,0], [1,0,0,2]],
    # Context 31
    [[1,1,2,0], [1,0,2,2]],
    # Context 32
    [[1,2,1,0], [1,0,0,2]],
    # Context 33
    [[1,2,1,0], [1,0,1,2]],
    # Context 34
    [[1,1,0,2], [0,1,2,0]],
    # Context 35
    [[1,1,0,2], [0,1,2,2]],
    # Context 36
    [[1,2,0,2], [0,1,2,0]],
    # Context 37
    [[1,2,0,2], [0,1,2,1]],
    # Context 38
    [[1,1,1,2], [1,2,1,1]],
    # Context 39
    [[1,2,2,2], [1,1,2,1]],
    # Context 40
    [[0,1,2,1], [1,0,0,2]],
])

# For backward compatibility, provide A and B as views into C
A = C[:, 0]  # A[i] = C[i, 0]
B = C[:, 1]  # B[i] = C[i, 1]