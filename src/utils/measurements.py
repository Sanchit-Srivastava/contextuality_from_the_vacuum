# Measurement projectors for joint measurements on each context

import numpy as np

from utils import operators
from utils.contexts import A, B

w = np.exp(2 * np.pi * 1j / 3) # Primitive cube root of unity

# Function to calculate the measurement projectors for a given context
def projector(c, a, b):
  """
  Calculate the measurement projector for a given context and output (a, b).

  For each pair (p, q), it performs the following steps:
    - Computes an exponent as (p * a + q * b) modulo 3.
    - Constructs an operator using a context-dependent combination of A[c] and B[c] via the pauli function.
    - Applies a phase factor by raising a global constant w to the power of the negative exponent.
    - Accumulates the resulting operator.

  Finally, the accumulated sum is normalized by dividing by 9.

  Parameters:
      c: Identifier/index selecting the context, used to index into A and B.
      a: Outcome of the first operator A in the context (0, 1, or 2).
      b: Outcome of the second operator B in the context (0, 1, or 2).

  Returns:
      The normalized measurement projector as a summed operator (or possibly a matrix) after dividing by 9.
  """
  P = 0 # Initialize projector
  for p in range(3):
    for q in range(3):
      # if p == 0 and q == 0:
      #   continue
      exponent = (p * a + q * b) % 3 # Exponent for phase factor
      op = operators.pauli(p * A[c] + q * B[c]) # Operator for (p, q) in context c
      term = w ** (-exponent) * op
      P += term 
  return P / 9 # Normalize by 9
  
# Precompute all projectors: shape (40, 3, 3)
projectors = [[[projector(c, a, b) for b in range(3)] for a in range(3)] for c in range(40)]

# Function to calculate measurement statistics for all contexts
def empirical_model(rho):
    """
    Calculate the vectorized empirical model for the given quantum state.

    This function computes the probabilities of joint measurement outcomes for each context.
   
    For a given context c, it iterates over all pairs of outcomes (a, b) and calculates the probability using the corresponding projector.
    
    The computed probabilities are stored in a flattened vector of size 360 (each context contributes 9 entries).

    If the total probability for any context exceeds 1, a warning message is printed to indicate a potential issue with the probabilities for that context.

    Parameters:
      rho (np.ndarray): The density matrix representing the quantum state. It should be compatible with the
                measurement projectors such that the matrix multiplication and trace operations are valid.

    Returns:
      np.ndarray: A 1D numpy array of length 360, where each segment of 9 elements corresponds to a measurement context.
    """
    E = np.zeros(360) # Initialize empirical model vector
    for c in range(40): #range over contexts
        for a in range(3): 
            for b in range(3):
                P = projectors[c][a][b] # projectors precomputed using the projector function outside the loop
                E[9*c + (3 * a + b)] = np.trace(rho @ P).real # Born rule
        if np.sum(E[9*c:9*c+9]) > 1:
            print("Sum of entries for context", c, ":", np.sum(E[9*c:9*c+9])) # Warning if sum of probabilities > 1
    return E