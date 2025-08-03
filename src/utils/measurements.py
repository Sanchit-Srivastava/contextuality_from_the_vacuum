# Measurement projectors for joint measurements on each context

import numpy as np

from utils import operators
from utils.contexts import A, B

w = np.exp(2 * np.pi * 1j / 3) # Primitive cube root of unity

# Function to calculate the measurement projectors for a given context
def projector(c, a, b):
  """
  Calculate the measurement projector for a given context and parameters.

  This function computes a measurement projector by summing over contributions from a double loop on p and q (each ranging from 0 to 2).
  For each pair (p, q), it performs the following steps:
    - Computes an exponent as (p * a + q * b) modulo 3.
    - Constructs an operator using a context-dependent combination of A[c] and B[c] via the pauli function.
    - Applies a phase factor by raising a global constant w to the power of the negative exponent.
    - Accumulates the resulting operator.

  Finally, the accumulated sum is normalized by dividing by 9.

  Parameters:
      c: Identifier/index selecting the context, used to index into A and B.
      a: Coefficient used in the calculation of the exponent.
      b: Coefficient used in the calculation of the exponent.

  Returns:
      The normalized measurement projector as a summed operator (or possibly a matrix) after dividing by 9.
  """
  P = 0
  for p in range(3):
    for q in range(3):
      # if p == 0 and q == 0:
      #   continue
      exponent = (p * a + q * b) % 3
      op = operators.pauli(p * A[c] + q * B[c])
      term = w ** (-exponent) * op
      P += term
  return P / 9
  
# Precompute all projectors: shape (40, 3, 3)
projectors = [[[projector(c, a, b) for b in range(3)] for a in range(3)] for c in range(40)]

# Function to calculate measurement statistics for all contexts
def empirical_model(rho):
    """
    Calculate the vectorized empirical model for the given quantum state.

    This function computes a vector of empirical probabilities associated with different measurement contexts
    for the quantum state provided as a density matrix (rho). For each of the 40 contexts, and for each combination
    of two measurement indices (a and b, each ranging from 0 to 2), it calculates the real trace of the product of
    rho with the corresponding measurement projector. The computed probabilities are stored in a flattened vector
    of size 360 (each context contributes 9 entries). If the total probability for any context exceeds 1, a warning
    message is printed to indicate a potential issue with the probabilities for that context.

    Parameters:
      rho (np.ndarray): The density matrix representing the quantum state. It should be compatible with the
                measurement projectors such that the matrix multiplication and trace operations are valid.

    Returns:
      np.ndarray: A 1D numpy array of length 360, where each segment of 9 elements corresponds to a measurement context.
    """
    E = np.zeros(360)
    for c in range(40):
        for a in range(3):
            for b in range(3):
                # if a == 0 and b == 0:
                #     continue
                # Get the projector for context c, measurement a, b
                P = projectors[c][a][b]
                E[9*c + (3 * a + b)] = np.trace(rho @ P).real
        if np.sum(E[9*c:9*c+9]) > 1:
            print("Sum of entries for context", c, ":", np.sum(E[9*c:9*c+9]))
    return E