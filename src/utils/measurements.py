# Measurement projectors for joint measurements on each context

import numpy as np

try:
    from . import operators
    from .contexts import A, B
except ImportError:
    # Fall back to absolute imports when run directly
    import operators
    from contexts import A, B

w = np.exp(2 * np.pi * 1j / 3) # Primitive cube root of unity

# Projector for a single measurement 
def projector(x: int, z: int, r_a: int) -> np.ndarray:
  """
  Construct the normalized spectral projector associated with outcome r_a
  for the operator defined by x,z (arithmetic modulo 3).
  """

  w = np.exp(2 * np.pi * 1j / 3) # Primitive cube root of unity
  r = int(r_a) % 3
  x = int(x) % 3
  z = int(z) % 3

  # Initialize with correct shape and dtype
  P = np.zeros_like(operators.pauli(0, 0), dtype=complex)

  for i in range(3):
    phase = w ** ((-1 * i * r) % 3)
    op = operators.pauli(x, z)  # Pauli operator for (x, z)
    op_power = np.linalg.matrix_power(op, i)
    P += phase * op_power
  # Check if valid projector
  proj = P/3
  if np.allclose(proj@proj,proj):
      return proj
  else:
      raise ValueError(f"Invalid projector for ({x}, {z}, {r_a}): {proj}")



# Function to calculate the measurement projectors for a given context
def context_projector(c, a, b):
  """
  Calculate the measurement projector for a given context and output (a, b).

  For each pair (p, q), it performs the following steps:
    - Computes an exponent as (p * a + q * b) modulo 3.
  - Constructs an operator using a combination of A[c] and B[c] via the weyl function.
    - Applies a phase factor by raising a global constant w to the power of the negative exponent.
    - Accumulates the resulting operator.

  Finally, the accumulated sum is normalized by dividing by 9.

  Parameters:
      c: Identifier/index selecting the context, used to index into A and B.
      a: Outcome of the first operator A in the context (0, 1, or 2).
      b: Outcome of the second operator B in the context (0, 1, or 2).

  Returns:
      The normalized measurement projector.
  """
  P = 0 # Initialize projector
  for p in range(3):
    for q in range(3):
      # if p == 0 and q == 0:
      #   continue
      exponent = (p * a + q * b) % 3 # Exponent for phase factor
      v = (p * A[c] + q * B[c]) % 3
      op = operators.weyl(v) # Operator for (p, q) in context c
      term = w ** (-exponent) * op
      P += term
  proj = P/9
  if np.allclose(proj @ proj, proj):
      return proj
  else:
      raise ValueError(f"Invalid projector for context {c}, a={a}, b={b}: {proj}")

# Precompute all projectors: shape (40, 3, 3)
projectors = [[[context_projector(c, a, b) for b in range(3)] for a in range(3)] for c in range(40)]

# Function to calculate measurement statistics for all contexts
def empirical_model(rho):
    """
    Calculate the vectorized empirical model for the given quantum state.

    This function computes the probabilities of joint measurement outcomes for each context.
   
    For a given context c 
      - Generates the projectors for all pairs (a, b) in that context.
      - Computes the probabilities using the Born rule: P(a, b) = Tr(rho @ P(a, b)).

    The computed probabilities are stored in a flattened vector of size 360 
      -(each context contributes 9 entries).

    If the total probability for any context exceeds 1, a warning message is printed.

    Parameters:
      rho (np.ndarray): The density matrix representing the quantum state. 

    Returns:
      np.ndarray: A 1D numpy array of length 360
      -each segment of 9 elements corresponds to a measurement context.
    """
    E = np.zeros(360) # Initialize empirical model vector
    for c in range(40): #range over contexts
        for a in range(3): 
            for b in range(3):
                P = projectors[c][a][b] # projectors precomputed using the projector function outside the loop
                E[9*c + (3 * a + b)] = np.trace(rho @ P).real # Born rule
        tol = 1e-11  # Tolerance for slight numerical deviations
        if np.sum(E[9*c:9*c+9]) > 1 + tol:
            print("Sum of entries for context", c, ":", np.sum(E[9*c:9*c+9]))
    return E