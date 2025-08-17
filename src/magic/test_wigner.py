import os
import sys
import numpy as np

# Ensure the project src/ is on sys.path so `magic` package is importable under pytest
SRC = os.path.dirname(os.path.dirname(__file__))  # .../src
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    # Prefer absolute import once src is on sys.path
    from magic.wigner_polytope import wigner_inequalities
    from magic.phase_points import phase_point
except Exception:
    # Fallback when executed as part of the magic package
    from .wigner_polytope import wigner_inequalities   # type: ignore
    from .phase_points import phase_point  # type: ignore

w = np.exp(2j*np.pi/3)

def _rho_from_ket(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())

def _extract(out):
    """Handle both (max_violation, points) and (max_violation, points, W)."""
    if isinstance(out, tuple) and len(out) >= 2:
        mv, pts = out[0], out[1]
        W = out[2] if len(out) >= 3 else None
        return float(mv), pts, W
    raise RuntimeError("wigner_inequalities should return a tuple")


def _wigner_values(rho: np.ndarray) -> np.ndarray:
    """Compute the 9 qutrit Wigner values from phase-point operators.

    W[x,y] = Re Tr[rho A_{x,y}], flattened in row-major order.
    """
    vals = []
    for i in range(3):
        for j in range(3):
            A = phase_point(i, j)
            vals.append(np.real(np.trace(rho @ A)))
    vals = np.asarray(vals, dtype=float)
    return vals / 3.0  # normalize by d=3 so sum W = 1

# --- Test 1: stabilizer states → no negativity ------------------------------

def test_stabilizer_states_nonnegative():
    # Z-eigenstates |0>,|1>,|2>
    e0 = np.array([1,0,0], complex)
    e1 = np.array([0,1,0], complex)
    e2 = np.array([0,0,1], complex)

    # X-eigenstates (Fourier basis) |~k> with components ω^{jk}/√3
    F = (1/np.sqrt(3)) * np.array([[1, 1, 1],
                                   [1, w, w**2],
                                   [1, w**2, w]], complex)
    x0, x1, x2 = F[:,0], F[:,1], F[:,2]

    for psi in [e0,e1,e2,x0,x1,x2]:
        rho = _rho_from_ket(psi)
        mv, pts, W = _extract(wigner_inequalities(rho))
        assert mv == 0.0 and len(pts) == 0, "Stabilizer state showed negativity"
        # Always test Wigner normalization
        Wvals = _wigner_values(rho)
        assert np.isclose(np.sum(Wvals), 1.0, atol=1e-10)

# --- Test 2: a generic pure state should (almost always) show negativity -----

def test_random_pure_often_negative():
    rng = np.random.default_rng(42)
    found_negative = False
    for _ in range(20):
        psi = rng.normal(size=3) + 1j*rng.normal(size=3)
        rho = _rho_from_ket(psi)
        mv, pts, W = _extract(wigner_inequalities(rho))
        # Always test Wigner normalization for each sampled state
        Wvals = _wigner_values(rho)
        assert np.isclose(np.sum(Wvals), 1.0, atol=1e-10)
        if mv > 0:
            found_negative = True
            break
    assert found_negative, "Did not find a negative Wigner among random pure states"