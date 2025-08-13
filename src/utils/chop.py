''' This module contains the function to chop stuff'''


import numpy as np

def chop(x, tol=1e-10):
    """Like Mathematica's Chop: zero out tiny real/imag parts."""
    x = np.asarray(x)
    r = np.where(np.abs(x.real) < tol, 0.0, x.real)
    i = np.where(np.abs(x.imag) < tol, 0.0, x.imag)
    return r + 1j*i