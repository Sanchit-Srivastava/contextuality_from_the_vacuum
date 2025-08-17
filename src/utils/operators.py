"""Heisenberg-Weyl Operators generated from symplectic vectors"""
# import numpy as np

# #Phase factor
# w = np.exp(2 * np.pi * 1j / 3)

# # Generators of the Heisenberg-Weyl group for qutrits:
# X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) #Pauli X operator
# Z = np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]]) #Pauli Z operator

# #Function to generate the Heisenberg-Weyl operator from symplectic vectors
# def weyl(A):
#     """
#     Generate the Heisenberg-Weyl operator from a symplectic vector.
    
#     The phase is determined by the formula:
#         phase = w**(-2*A[0]*A[1] - 2*A[2]*A[3])

#     Each component is built as the product of integer powers of the matrices X and Z

#     The final operator is constructed as:
#         operator = phase * np.kron(first_component, second_component)

#     Parameters:
#         A (iterable of int): A symplectic vector of length 4. 

#     Returns:
#         numpy.ndarray: The resulting Heisenberg-Weyl operator as a matrix, obtained by applying the 
#     """
#     phase = w**(2 * A[0] * A[1] + 2 * A[2] * A[3])
#     # return phase * np.kron(np.linalg.matrix_power(Z, A[0]) @ np.linalg.matrix_power(X, A[1]), np.linalg.matrix_power(Z, A[2]) @ np.linalg.matrix_power(X, A[3]))
#     return phase * np.kron(np.linalg.matrix_power(X, A[0]) @ np.linalg.matrix_power(Z, A[1]), np.linalg.matrix_power(X, A[2]) @ np.linalg.matrix_power(Z, A[3]))


# # Function to generate 1 qutrit Pauli operator from the symplectic vector.
# def pauli(x: int, z: int) -> np.ndarray:
#     """
#     Construct the Pauli operator associated with the symplectic vector A.
#     """
#     w = np.exp(2 * np.pi * 1j / 3)
#     # A_vec = np.asarray([x, 0, 0, z]) % 3
#     x = int(x) % 3
#     z = int(z) % 3
#     phase = w ** ((2 * x * z) % 3)
#     return phase * np.linalg.matrix_power(Z, x) @ np.linalg.matrix_power(X, z)

import numpy as np

# global constants
w = np.exp(2j * np.pi / 3)  # omega
X = np.array([[0, 0, 1],
              [1, 0, 0],
              [0,1, 0]], dtype=complex)
Z = np.diag([1, w, w**2]).astype(complex)

def D1(z: int, x: int) -> np.ndarray:
    """
    One-qutrit Weyl operator with XZ order:
      D_{x,z} = w^{(1/2) x z} X^x Z^z,   with (1/2) ≡ 2 (mod 3).
    """
    x %= 3; z %= 3
    phase = w ** ((2 * x * z) % 3)
    return phase * (np.linalg.matrix_power(X, z) @ np.linalg.matrix_power(Z, x))

def weyl(A) -> np.ndarray:
    """
    Two-qutrit Weyl operator for A = (x1, z1, x2, z2) with XZ order:
      D(A) = D1(x1, z1) ⊗ D1(x2, z2)
           = w^{(1/2)(x1 z1 + x2 z2)} (X^{x1} Z^{z1}) ⊗ (X^{x2} Z^{z2})
    """
    z1, x1, z2, x2 = (int(A[0]) % 3, int(A[1]) % 3, int(A[2]) % 3, int(A[3]) % 3)
    return np.kron(D1(z1, x1), D1(z2, x2))


pauli = D1