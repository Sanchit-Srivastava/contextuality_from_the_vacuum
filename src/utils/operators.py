"""Heisenberg-Weyl Operators generated from symplectic vectors"""
import numpy as np

#Phase factor
w = np.exp(2 * np.pi * 1j / 3)

# Generators of the Heisenberg-Weyl group for qutrits:
X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) #Pauli X operator
Z = np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]]) #Pauli Z operator

#Function to generate the Heisenberg-Weyl operator from symplectic vectors
def weyl(A):
    """
    Generate the Heisenberg-Weyl operator from a symplectic vector.
    
    The phase is determined by the formula:
        phase = w**(-2*A[0]*A[1] - 2*A[2]*A[3])

    Each component is built as the product of integer powers of the matrices X and Z

    The final operator is constructed as:
        operator = phase * np.kron(first_component, second_component)

    Parameters:
        A (iterable of int): A symplectic vector of length 4. 

    Returns:
        numpy.ndarray: The resulting Heisenberg-Weyl operator as a matrix, obtained by applying the 
    """
    phase = w**(-2 * A[0] * A[1] - 2 * A[2] * A[3])
    return phase * np.kron(np.linalg.matrix_power(X, A[0]) @ np.linalg.matrix_power(Z, A[1]), np.linalg.matrix_power(X, A[2]) @ np.linalg.matrix_power(Z, A[3]))


# Function to generate 1 qutrit Pauli operator from the symplectic vector.
def pauli(x: int, z: int) -> np.ndarray:
    """
    Construct the Pauli operator associated with the symplectic vector A.
    """
    # A_vec = np.asarray([x, 0, 0, z]) % 3
    x = int(x) % 3
    z = int(z) % 3
    phase = w ** (-2 * x * z)
    return phase * np.linalg.matrix_power(X, x) @ np.linalg.matrix_power(Z, z)