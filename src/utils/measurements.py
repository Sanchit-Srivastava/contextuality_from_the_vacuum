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
      if p == 0 and q == 0:
        continue
      exponent = (p * a + q * b) % 3
      op = operators.pauli(p * A[c], q * B[c])
      term = w ** (-exponent) * op
      P += term
      return P / 8