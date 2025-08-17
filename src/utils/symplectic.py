"""Symplectic utilities for two-qutrit (and general mod-d) vectors.

Provides a reusable symplectic product function used across modules.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence


# def symplectic_product(
#     u: Sequence[int] | np.ndarray,
#     v: Sequence[int] | np.ndarray,
#     *,
#     mod: int = 3,
# ) -> int:
#     """Compute the symplectic product sp(u, v) modulo ``mod``.

#     For 2-qutrit symplectic vectors u=(a,b,c,d), v=(a2,b2,c2,d2),
#     sp(u, v) = a*b2 - b*a2 + c*d2 - d*c2 (mod mod).

#     Returns an int in [0, mod-1]. Inputs are reduced mod ``mod`` first.
#     """
#     u = np.asarray(u, dtype=int) % mod
#     v = np.asarray(v, dtype=int) % mod
#     if u.shape[-1] != 4 or v.shape[-1] != 4:
#         raise ValueError("symplectic_product expects length-4 vectors for two qutrits")
#     a, b, c, d = u
#     a2, b2, c2, d2 = v
#     return int((a * b2 - b * a2 + c * d2 - d * c2) % mod)

def symplectic_product(
    u: Sequence[int] | np.ndarray,
    v: Sequence[int] | np.ndarray,
    *,
    mod: int = 3,
) -> int:
    """
    Symplectic product for two qutrits (XZ convention).

    u = (x1, z1, x2, z2)
    v = (x1', z1', x2', z2')

    Returns:
        [u,v] = (x1*z1' - z1*x1') + (x2*z2' - z2*x2')   (mod mod)
    """
    u = np.asarray(u, dtype=int) % mod
    v = np.asarray(v, dtype=int) % mod
    if u.shape != (4,) or v.shape != (4,):
        raise ValueError("symplectic_product expects length-4 vectors (two qutrits)")

    x1, z1, x2, z2 = u
    x1p, z1p, x2p, z2p = v

    val = (x1 * z1p - z1 * x1p) + (x2 * z2p - z2 * x2p)
    return int(val % mod)


# convenient alias
sp = symplectic_product

