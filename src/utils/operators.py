"""Heisenberg-Weyl Operators generated from symplectic vectors"""
import numpy as np

# Phase factor
w = np.exp(2 * np.pi * 1j / 3)

# Generators of the Heisenberg-Weyl group for qutrits:
X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Pauli X operator
Z = np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]])  # Pauli Z operator


# Function to generate the Heisenberg-Weyl operator from symplectic vectors
def pauli(A):
    """
    Generate the Heisenberg-Weyl operator from a symplectic vector.

    Parameters:
        A (iterable of int): A symplectic vector of length 4. 

    Returns:
        numpy.ndarray: The resulting Heisenberg-Weyl operator as a matrix,
            obtained by applying the Kronocker product to the two 
            single-qudit components.
    """
    phase = w**(2 * A[0] * A[1] + 2 * A[2] * A[3])
    first_component = (np.linalg.matrix_power(X, A[0]) @ 
                      np.linalg.matrix_power(Z, A[1]))
    second_component = (np.linalg.matrix_power(X, A[2]) @ 
                       np.linalg.matrix_power(Z, A[3]))
    return phase * np.kron(first_component, second_component)


 