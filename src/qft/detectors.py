'''This module contains the functions to define the detector states'''

import numpy as np
from numpy import pi, sqrt
from scipy.special import erf, erfc  # complex-safe

# ---------- base 9x9 seed ----------
rho0 = np.zeros((9, 9), dtype=complex)
rho0[0, 0] = 1.0

# ---------- scalars (Ω is absorbed as 1 here, matching your defs) ----------
def L(sigma: float) -> complex:
    return (1/(4*pi)) * (
        np.exp(-0.5 * sigma**2)
        - sqrt(pi) * (sigma/sqrt(2)) * erfc(sigma/sqrt(2))
    )

def Lab(sigma: float, d: float) -> complex:
    pref = (1/(4*sqrt(pi))) * (sigma/(sqrt(2)*d)) * np.exp(-(d**2)/(2*sigma**2))
    term = np.exp(-1j*d) * erf(1j*(d/(sqrt(2)*sigma)) - sigma/sqrt(2))
    return pref * (np.imag(term) - np.sin(d))

def M(sigma: float, d: float) -> complex:
    pref = 1j * (1/(4*sqrt(pi))) * (sigma/(sqrt(2)*d))
    pref *= np.exp(-(d**2)/(2*sigma**2)) * np.exp(-0.5*sigma**2)
    return pref * (1 + erf(1j*(d/(sqrt(2)*sigma))))

def QregDelta(sigma: float, a: float) -> complex:
    return np.exp(-0.5*sigma**2) * (1/(8*pi) - 1j * sigma**3 / (8*pi*a*(a**2 + sigma**2)))

def QregHeavsde(sigma: float, a: float) -> complex:
    return np.exp(-0.5*sigma**2) * (1/(8*pi) - 1j * sigma/(8*sqrt(2*pi)*a))

def Qmagic(sigma: float, a: float) -> complex:
    return np.exp(-0.5*sigma**2) * (1/(8*pi))

# ---------- matrices ----------
def rhosecond(sigma: float, d: float, a: float, Qfunc) -> np.ndarray:
    Lval   = L(sigma)
    Labval = Lab(sigma, d)
    Mval   = M(sigma, d)
    Qval   = Qfunc(sigma, a)

    return np.array([
        [-2*Lval, 0, np.conjugate(Qval), 0, np.conjugate(Mval), 0, np.conjugate(Qval), 0, 0],
        [0,       Lval, 0,               Labval, 0, 0, 0, 0, 0],
        [Qval,    0,    0,               0,      0, 0, 0, 0, 0],
        [0,       Labval, 0,             Lval,   0, 0, 0, 0, 0],
        [Mval,    0,    0,               0,      0, 0, 0, 0, 0],
        [0,       0,    0,               0,      0, 0, 0, 0, 0],
        [Qval,    0,    0,               0,      0, 0, 0, 0, 0],
        [0,       0,    0,               0,      0, 0, 0, 0, 0],
        [0,       0,    0,               0,      0, 0, 0, 0, 0],
    ], dtype=complex)

def rho(sigma: float, d: float, a: float, Qfunc, lam: float) -> np.ndarray:
    return rho0 + (lam**2) * rhosecond(sigma, d, a, Qfunc)

# ---------- utilities ----------
def chop(x, tol=1e-12):
    x = np.asarray(x)
    r = np.where(np.abs(x.real) < tol, 0.0, x.real)
    i = np.where(np.abs(x.imag) < tol, 0.0, x.imag)
    return r + 1j*i

# ---------- example (mirrors: Eigenvalues[rho[1,10,10,QregDelta,10^-2]] // N // Chop) ----------
if __name__ == "__main__":
    # Test parameters
    sigma = 1.0
    d = 10.0
    a = 10.0
    lam = 1e-2
    
    print("Quantum Field Theory Detector State Analysis")
    print("=" * 50)
    print(f"Parameters: sigma={sigma}, d={d}, a={a}, lambda={lam}")
    print()
    
    # Test with QregDelta Q-function
    print("Testing with QregDelta Q-function:")
    print("-" * 30)
    
    A = rho(sigma, d, a, QregDelta, lam)
    
    # Print the state matrix
    print("Generated density matrix:")
    print(A)
    print()
    
    # Check if it's a valid density matrix
    print("Density Matrix Validation:")
    print("-" * 25)
    
    # Check trace
    trace = np.trace(A)
    print(f"Trace: {trace:.10f}")
    trace_valid = np.abs(trace - 1.0) < 1e-10
    print(f"Trace = 1: {trace_valid}")
    
    # Check if Hermitian
    is_hermitian = np.allclose(A, A.T.conj())
    print(f"Is Hermitian: {is_hermitian}")
    
    # Check eigenvalues (positive semidefinite)
    evals = np.linalg.eigvals(A)
    evals_real = np.real(evals)  # Should be real for Hermitian matrices
    min_eigenval = np.min(evals_real)
    print(f"Eigenvalues: {chop(np.sort_complex(evals))}")
    print(f"Minimum eigenvalue: {min_eigenval:.10f}")
    positive_semidefinite = min_eigenval >= -1e-10  # Allow small numerical errors
    print(f"Positive semidefinite: {positive_semidefinite}")
    
    # Overall validity
    is_valid_density_matrix = trace_valid and is_hermitian and positive_semidefinite
    print()
    print(f"Valid density matrix: {is_valid_density_matrix}")
    
    if not is_valid_density_matrix:
        print("\nWARNING: Generated state is not a valid density matrix!")
        if not trace_valid:
            print(f"  - Trace issue: {trace} ≠ 1")
        if not is_hermitian:
            print("  - Not Hermitian")
        if not positive_semidefinite:
            print(f"  - Negative eigenvalue: {min_eigenval}")
    else:
        print("\nSUCCESS: Generated state is a valid density matrix!")