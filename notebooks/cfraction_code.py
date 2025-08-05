# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Contextual fraction calculator
#
# ## Introduction
#
# This notebook provides the code to calculate the contextual fraction of $2$-qutrit states with respect to Heisenber-Weyl operators.
#
# > This notebook contains all the necessary functions and hence can be used as a standalone notebook. For an explanation of the theory and the corresponding relevant functions, use the notebook `src/notebooks/contextual_fraction.ipynb` 

# %% [markdown]
# ## ---Notebook structure---
# Notebook Structure Summary:
# ---------------------------
# 1. Introduction and Overview:
#    - Presents the purpose of the notebook and its relation to contextual fractions.
#
# 2. Heisenberg-Weyl Operators:
#    - Defines qutrit Pauli operators (X, Z) and constructs two-qutrit operators using the pauli() function.
#
# 3. Contexts:
#    - Lists all maximal measurement contexts via symplectic vectors stored in the array C (also accessible via A and B).
#
# 4. Measurements:
#    - Implements functions to compute measurement projectors for each context.
#    - Constructs the empirical model based on the Born rule.
#
# 5. Incidence Matrix:
#    - Creates an incidence matrix connecting global assignments to measurement outcomes.
#
# 6. Linear Program for Contextual Fraction:
#    - Formulates and solves a linear program to compute the contextual fraction of a given quantum state.
#
# 7. Usage Examples:
#    - Provides utility functions to create different quantum states.
#    - Demonstrates the calculation of contextual fractions for these states.

# %% [markdown]
# ## Heisenberg-Weyl operators
#
# The  2-qutrit Heisenber-Weyl operators are defined as

# %%
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



# %% [markdown]
# ## Contexts
#
# The generators for all the maximal contexts of $2$-qutrit Heisenber-Weyl operators are defined below.

# %%
"""Contexts for two qutrit Heisenberg-Weyl operators 
    -defined in terms of their symplectic vectors"""
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

# %% [markdown]
# # Measurements
#
# The measurement statistics for a state $\rho$ are generated as a vectorized empirical model. 
#
# The functions to calculate projectors for each context and generate the empirical model for a given state $\rho$ are given below. 
#
#

# %%
# Measurement projectors for joint measurements on each context

import numpy as np

# from . import operators
# from .contexts import A, B

w = np.exp(2 * np.pi * 1j / 3)  # Primitive cube root of unity


# Function to calculate the measurement projectors for a given context
def projector(c, a, b):
    """
    Calculate the measurement projector for a given context and output (a, b).

    For each pair (p, q), it performs the following steps:
      - Computes an exponent as (p * a + q * b) modulo 3.
      - Constructs an operator using a combination of A[c] and B[c] via the 
        pauli function.
      - Applies a phase factor by raising a global constant w to the power 
        of the negative exponent.
      - Accumulates the resulting operator.

    Parameters:
        c: Identifier/index selecting the context, used to index into A and B.
        a: Outcome of the first operator A in the context (0, 1, or 2).
        b: Outcome of the second operator B in the context (0, 1, or 2).

    Returns:
        The normalized measurement projector.
    """
    P = 0  # Initialize projector
    for p in range(3):
        for q in range(3):
            # if p == 0 and q == 0:
            #   continue
            exponent = (p * a + q * b) % 3  # Exponent for phase factor
            # Operator for (p, q) in context c
            op = pauli(p * A[c] + q * B[c])
            term = w ** (-exponent) * op
            P += term
    return P / 9  


# Precompute all projectors: shape (40, 3, 3)
projectors = [[[projector(c, a, b) for b in range(3)] 
               for a in range(3)] for c in range(40)]


# Function to calculate measurement statistics for all contexts
def empirical_model(rho):
    """
    Calculate the vectorized empirical model for the given quantum state.

    This function computes the probabilities of joint measurement outcomes 
    for each context.
   
    For a given context c:
      - Generates the projectors for all pairs (a, b) in that context.
      - Computes the probabilities using the Born rule: 
        P(a, b) = Tr(rho @ P(a, b)).

    The computed probabilities are stored in a flattened vector of size 360 
    (each context contributes 9 entries).

    If the total probability for any context exceeds 1, a warning message 
    is printed.

    Parameters:
        rho (np.ndarray): The density matrix representing the quantum state. 

    Returns:
        np.ndarray: A 1D numpy array of length 360, where each segment of 
            9 elements corresponds to a measurement context.
    """
    E = np.zeros(360)  # Initialize empirical model vector
    for c in range(40):  # Range over contexts
        for a in range(3):
            for b in range(3):
                # Projectors precomputed using the projector function 
                # outside the loop
                P = projectors[c][a][b]
                E[9*c + (3 * a + b)] = np.trace(rho @ P).real  # Born rule
        tol = 1e-4  # Tolerance for slight numerical deviations
        if np.sum(E[9*c:9*c+9]) > 1 + tol:
            print("Sum of entries for context", c, ":", 
                  np.sum(E[9*c:9*c+9]))
    return E


# %% [markdown]
# ## Incidence matrix
#
# The incidence matrix $M$ required for computing the contextual fraction is defined below. 

# %%
"""Generating the incidence matrix for all possible global assignments"""

import numpy as np
from scipy.sparse import lil_matrix

# from utils.contexts import A, B
# from utils.ternary import to_ternary

def to_ternary(n: int) -> np.ndarray:
    """
    Convert a number smaller than 81 into its ternary (base 3) representation,
    returned as a 4-digit numpy array (with leading zeros if necessary).

    Parameters:
        n (int): The number to convert. Must be in the range 0 <= n < 81.

    Returns:
        np.ndarray: A 4-element numpy array of integers representing the 
            ternary digits.
    """
    if not (0 <= n < 81):
        raise ValueError("Input must be between 0 and 80 (inclusive of 0 "
                        "and exclusive of 81)")

    # Special case for 0
    if n == 0:
        return np.array([0, 0, 0, 0])

    digits = []
    while n:
        digits.append(n % 3)
        n //= 3
    digits.reverse()
    
    # Pad the list with leading zeros to make it 4-digit long
    padded_digits = [0] * (4 - len(digits)) + digits
    return np.array(padded_digits)


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

# %% [markdown]
# ## Linear program to compute the contextual fraction
#
# With the empirical model and the incidence matrix defined, we can compute the contextual fraction using the following linear program.

# %%
import numpy as np
from scipy.optimize import linprog


def contextual_fraction(rho):
    """
    Solve the linear program to maximize the contextual fraction.
    
    Args:
        rho: The density matrix/quantum state
        
    Returns:
        dict: Contains the optimization result with keys:
            - 'success': bool, whether optimization succeeded
            - 'b': array, optimal solution vector (if successful)
            - 'result': scipy.optimize.OptimizeResult object
    """
    # === Linear Program ===
    # maximize 1.b  -> minimize -1.b
    
    c = -np.ones(M.shape[1])  # Objective vector: length 81
    bounds = [(0, 1)] * M.shape[1]  # b >= 0
    
    # Empirical data
    E = empirical_model(rho)
    
    # Solve using HiGHS
    result = linprog(c, A_ub=M, b_ub=E, bounds=bounds, method='highs')
    
    # === Output ===
    output = {
        'success': result.success,
        'result': result
    }
    
    if result.success:
        output['b'] = 1 - np.dot(-c, result.x)
    
    return output


# %% [markdown]
# ## Usage
#
# We can now calculate the contextual fraction for input states $\rho$. Replace the example states in code below with states of interest. 

# %%
"""
Utility functions for creating and analyzing quantum states for 
two-qutrit systems.
"""

import numpy as np

def create_maximally_mixed_state():
    """Create a maximally mixed state for two qutrits (9x9 identity/9)."""
    return np.eye(9) / 9


def create_product_state():
    """Create a product state |0⟩⊗|0⟩ for two qutrits."""
    state_0 = np.array([1, 0, 0])  # |0⟩ state for a qutrit
    product_state = np.kron(state_0, state_0)  # |0⟩⊗|0⟩
    return np.outer(product_state, product_state.conj())


def create_maximally_entangled_state():
    """Create a maximally entangled state for two qutrits."""
    # |ψ⟩ = (|00⟩ + |11⟩ + |22⟩) / √3
    state_00 = np.kron([1, 0, 0], [1, 0, 0])  # |00⟩
    state_11 = np.kron([0, 1, 0], [0, 1, 0])  # |11⟩
    state_22 = np.kron([0, 0, 1], [0, 0, 1])  # |22⟩
    
    psi = (state_00 + state_11 + state_22) / np.sqrt(3)
    return np.outer(psi, psi.conj())


def create_custom_state(alpha=1/np.sqrt(2), beta=1/np.sqrt(2)):
    """
    Create a custom superposition state on the first qutrit, 
    product with |0⟩ on second.
    |ψ⟩ = (α|0⟩ + β|1⟩) ⊗ |0⟩
    """
    # Normalize coefficients
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha/norm, beta/norm
    
    first_qutrit = np.array([alpha, beta, 0])  # Superposition on first qutrit
    second_qutrit = np.array([1, 0, 0])        # |0⟩ on second qutrit
    
    product_state = np.kron(first_qutrit, second_qutrit)
    return np.outer(product_state, product_state.conj())


def print_state_info(state, name):
    """Print information about a quantum state."""
    print(f"\n{'='*50}")
    print(f"State: {name}")
    print(f"{'='*50}")
    print(f"Trace: {np.trace(state):.6f}")
    print(f"Hermitian: {np.allclose(state, state.conj().T)}")
    eigenvals = np.linalg.eigvals(state)
    print(f"Positive semidefinite: {np.all(eigenvals >= -1e-10)}")
    


def get_default_test_states():
    """Return a dictionary of default quantum states for testing."""
    return {
        "Maximally Mixed State": create_maximally_mixed_state(),
        "Product State |00⟩": create_product_state(),
        "Maximally Entangled State": create_maximally_entangled_state(),
        "Custom Superposition": create_custom_state(alpha=1, beta=1j),
    }


# Get test states
states_dict = get_default_test_states()

# Compute contextual fractions for each state
print("Contextual fractions for different quantum states:")
print("=" * 50)
for name, rho in states_dict.items():
    result = contextual_fraction(rho)
    if result['success']:
        cf = result['b']
        print(f"{name}: {cf:.6f}")
    else:
        print(f"{name}: Optimization failed")

