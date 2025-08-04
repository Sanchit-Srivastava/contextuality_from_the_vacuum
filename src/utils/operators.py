"""Heisenberg-Weyl Operators generated from symplectic vectors"""
import numpy as np

#Phase factor
w = np.exp(2 * np.pi * 1j / 3)

# Generators of the Heisenberg-Weyl group for qutrits:
X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) #Pauli X operator
Z = np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]]) #Pauli Z operator

#Function to generate the Heisenberg-Weyl operator from symplectic vectors
def pauli(A):
    """
    Generate the Heisenberg-Weyl operator from a symplectic vector.

    This function computes the Heisenberg-Weyl operator by applying a phase factor and
    taking the Kronecker product of two operator components. The phase is determined by
    the formula:

        phase = w**(2*A[0]*A[1] + 2*A[2]*A[3])

    Each component is built as the product of integer powers of the matrices X and Z

    The final operator is constructed as:
        operator = phase * np.kron(first_component, second_component)

    Assumptions:
        - The global variables or symbols "w", "X", and "Z" are defined elsewhere.
        - The input A is an iterable with at least four elements, where A[0] through A[3] are
          used as exponents.

    Parameters:
        A (iterable of int): A symplectic vector of length at least 4. The elements A[0] and A[1]
                             determine the powers for the first operator component, while A[2] and A[3]
                             determine the powers for the second operator component.

    Returns:
        numpy.ndarray: The resulting Heisenberg-Weyl operator as a matrix, obtained by applying the 
    """
    phase = w**(2 * A[0] * A[1] + 2 * A[2] * A[3])
    return phase * np.kron(np.linalg.matrix_power(X, A[0]) @ np.linalg.matrix_power(Z, A[1]), np.linalg.matrix_power(X, A[2]) @ np.linalg.matrix_power(Z, A[3]))


 