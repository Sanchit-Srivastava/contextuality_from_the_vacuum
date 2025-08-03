# Measurement projectors for joint measurements on each context

import numpy as np

from utils import operators
from utils.contexts import A, B

w = np.exp(2 * np.pi * 1j / 3)
# Function to calculate the measurement projectors for a given context
def projector(c, a, b):
  """
  Calculate the measurement projector for given context c and parameters a, b.
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
    Calculate the vectorized empirical model for the given state rho.
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