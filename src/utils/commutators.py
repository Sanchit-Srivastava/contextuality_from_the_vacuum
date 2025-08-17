#!/usr/bin/env python3
# For checking pairwise commutators in contexts. 

import numpy as np
import sys
import os

# Support both package and script execution
try:
    from .contexts import A, B  # when imported as a package module
except Exception:  # pragma: no cover - fallback for direct script execution
    sys.path.append(os.path.dirname(__file__))
    from contexts import A, B
try:
    from . import operators  # when imported as a package module
    from .symplectic import symplectic_product as sp
except Exception:  # pragma: no cover - fallback for direct script execution
    sys.path.append(os.path.dirname(__file__))
    import operators
    from symplectic import symplectic_product as sp


def commute_check(A, B):
    """
    Checks if two operators A and B commute in the context of qudit systems.
    The commutation is determined by evaluating (A[0] * B[1] - A[1] * B[0] + A[2] * B[3] - A[3] * B[2]) % 3 == 0.
    """
    return (A[0] * B[1] - A[1] * B[0] + A[2] * B[3] - A[3] * B[2]) % 3 == 0 
    

def context_check(): 
    """
    Checks if a context c is valid by ensuring that all pairs of operators in the context commute.
    """   
    for c in range(len(A)):
        commute_check(A[c], B[c])
        if not commute_check(A[c], B[c]):
            return False
    return True


def main() -> int:
    """Run checks over all contexts (no args):
    1) Pairwise commutation of A[c], B[c]
    2) Weyl operator commutation within each context (W(u)W(v) = W(v)W(u))

    Exit codes: 0 on success for all contexts, 1 otherwise.
    """
    n = len(A)
    commute_failures: list[int] = []
    algebra_failures: list[int] = []

    for c in range(n):
        if not commute_check(A[c], B[c]):
            commute_failures.append(c + 1)
        ok = check_algebra(c, exhaustive=True)
        if not ok:
            algebra_failures.append(c + 1)

    if commute_failures:
        print(f"Non-commuting contexts: {commute_failures}")
    if algebra_failures:
        print(f"Algebra test failed in contexts: {algebra_failures}")

    if commute_failures or algebra_failures:
        return 1

    print("All contexts commute and Weyl(u), Weyl(v) commute within each context.")
    return 0


# ----------------------- Heisenberg–Weyl algebra tests -----------------------

w = np.exp(2j * np.pi / 3)


def _weyl_in_context(c: int, p: int, q: int) -> np.ndarray:
    """Return Weyl operator W(p*A[c] + q*B[c])."""
    u = (p * A[c] + q * B[c]) % 3
    return operators.weyl(u)


def check_weyl_product_rule_in_context(
    c: int,
    *,
    exhaustive: bool = True,
    samples: int = 100,
    atol: float = 1e-12,
    return_failures: bool = False,
):
    """Verify the Weyl product rule within a single context.

    Checks that for all (or many) p,q,p',q' in Z3:
        W(pA+qB) W(p'A+q'B) = w^{-2 sp(u,v)} W((p+p')A + (q+q')B)
    where u=pA+qB, v=p'A+q'B and sp is the symplectic form mod 3.

    Args:
      c: Context index (0-based).
      exhaustive: If True, test all 3^4 pairs. If False, test `samples` random pairs.
      samples: Number of random pairs when exhaustive=False.
      atol: Numerical tolerance for matrix comparison.
      return_failures: If True, also return a list of failing cases.

    Returns:
      bool, or (bool, list): True if all tested pairs satisfy the relation; otherwise False.
      If return_failures is True, also returns a list of details for failing cases.
    """
    n = len(A)
    if c < 0 or c >= n:
        raise ValueError(f"Invalid context index {c}. Must be in [0, {n-1}].")

    failures = []

    def check_pair(p, q, p2, q2):
        u = (p * A[c] + q * B[c]) % 3
        v = (p2 * A[c] + q2 * B[c]) % 3
        lhs = _weyl_in_context(c, p, q) @ _weyl_in_context(c, p2, q2)
        phase = w ** (2 * sp(u, v))
        rhs = phase * _weyl_in_context(c, (p + p2) % 3, (q + q2) % 3)
        ok = np.allclose(lhs, rhs, atol=atol)
        if not ok:
            failures.append({
                "p": int(p), "q": int(q), "p2": int(p2), "q2": int(q2),
                "sp": sp(u, v)
            })
        return ok

    if exhaustive:
        for p in range(3):
            for q in range(3):
                for p2 in range(3):
                    for q2 in range(3):
                        if not check_pair(p, q, p2, q2):
                            # keep collecting to report all, but could early-exit
                            pass
    else:
        rng = np.random.default_rng()
        for _ in range(samples):
            p, q, p2, q2 = rng.integers(0, 3, size=4)
            check_pair(int(p), int(q), int(p2), int(q2))

    ok_all = len(failures) == 0
    return (ok_all, failures) if return_failures else ok_all


def check_algebra(
    c: int,
    *,
    exhaustive: bool = True,
    samples: int = 200,
    atol: float = 1e-12,
    return_failures: bool = False,
):
    """Check Weyl(u) Weyl(v) == Weyl(v) Weyl(u) for u=pA+qB, v=p'A+q'B in context c.

    Args:
      c: Context index (0-based).
      exhaustive: If True, check all 3^4 pairs; else sample `samples` random pairs.
      samples: Number of random pairs when exhaustive is False.
      atol: Numerical tolerance for matrix equality.
      return_failures: If True, return details of failing pairs.

    Returns:
      bool, or (bool, list): True iff all checked pairs commute.
    """
    n = len(A)
    if c < 0 or c >= n:
        raise ValueError(f"Invalid context index {c}. Must be in [0, {n-1}].")

    failures = []

    def check_pair(p, q, p2, q2):
        U = _weyl_in_context(c, p, q)
        V = _weyl_in_context(c, p2, q2)
        lhs = U @ V
        rhs = V @ U
        ok = np.allclose(lhs, rhs, atol=atol)
        if not ok and return_failures:
            failures.append({"p": int(p), "q": int(q), "p2": int(p2), "q2": int(q2)})
        return ok

    if exhaustive:
        for p in range(3):
            for q in range(3):
                for p2 in range(3):
                    for q2 in range(3):
                        check_pair(p, q, p2, q2)
    else:
        rng = np.random.default_rng()
        for _ in range(samples):
            p, q, p2, q2 = rng.integers(0, 3, size=4)
            check_pair(int(p), int(q), int(p2), int(q2))

    ok_all = len(failures) == 0
    return (ok_all, failures) if return_failures else ok_all



if __name__ == "__main__":
    # raise SystemExit(main())
    main()


