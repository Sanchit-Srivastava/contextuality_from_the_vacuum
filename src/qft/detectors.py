# pip install numpy scipy

import numpy as np
from numpy import pi, sqrt
from scipy.special import erf, erfc  # supports complex args


try:
    from utils.state_checks import is_valid_state
    from utils.state_checks import validate_and_print
except ImportError:
    import sys
    from os.path import dirname, abspath
    sys.path.append(dirname(dirname(abspath(__file__))))
    from utils.state_checks import is_valid_state
    from utils.state_checks import validate_and_print

# ---------- base 9x9 seed ----------
rho0 = np.zeros((9, 9), dtype=complex)
rho0[0, 0] = 1.0


# ----- scalar helper functions -----
########### Don't need this for now ####################
# def L(gap: float, switching: float, coupling: float, ) -> complex:
#     """Refactored L[Ω, σ, λ] for improved clarity."""
#     factor = coupling**2 / (4 * pi)
#     om_switching = switching * gap
#     exp_term = np.exp(-0.5 * om_switching**2)
#     erfc_term = sqrt(pi) * (om_switching / sqrt(2)) * erfc(om_switching / sqrt(2))
#     return factor * (exp_term - erfc_term)
########################################################


def L_term(gap: float, switching: float, detector_type: str, group: str) -> float:
    """Compute the L probability."""
    if detector_type != "point_like":
        raise ValueError("Unsupported detector_type (for now)")
    
    if group == "HW":
        effective_gap = gap
    elif group == "SU2":
        effective_gap = -gap
    else:
        raise ValueError("Unsupported group")
    
    expr = np.exp(-0.5 * (switching * effective_gap) ** 2)
    arg = switching * effective_gap / sqrt(2)
    term = (1 / (4 * pi)) * (expr + sqrt(pi) * arg * (2 - erfc(arg)))
    return term
    

def Lab_term(gap: float, switching: float, separation: float, group: str) -> complex:
    """Lab[Ω, σ, separation, λ]"""
    
    # Determine effective gap based on group
    if group == "HW":
        effective_gap = gap
    elif group == "SU2":
        effective_gap = -gap
    else:
        raise ValueError(f"Unsupported group: {group}")

    # Pre-compute common terms
    sqrt_2 = sqrt(2)
    sqrt_pi = sqrt(pi)
    
    # Compute prefix factor
    norm_separation = (separation / (sqrt_2 * switching))
    pref = 1 / (8 * sqrt_pi) * (1 / norm_separation)
    pref *= np.exp(-1 * norm_separation**2)

    # Compute error function argument
    erf_arg = 1j * norm_separation + (effective_gap * switching) / sqrt_2

    # Compute the main term
    exp_term = np.exp(1j * separation * effective_gap)
    erf_term = erf(erf_arg)
    main_term = exp_term * erf_term
    
    # Combine terms
    result = pref * (np.imag(main_term) + np.sin(effective_gap * separation))
    
    return result

def M_term(gap: float, switching: float, separation: float, detector_type: str) -> complex:
    """M[Ω, σ, separation, λ]"""
    
    if detector_type != "point_like":
        raise ValueError("Unsupported detector_type (for now)")
    
    # Pre-compute common terms
    sqrt_2 = sqrt(2)
    sqrt_pi = sqrt(pi)
    
    # Compute normalized separation
    norm_separation = separation / (sqrt_2 * switching)
    
    # Compute prefix factor
    pref = 1j / (8 * sqrt_pi * norm_separation)
    pref *= np.exp(-norm_separation**2 - 0.5 * (switching * gap)**2)
    
    # Compute error function term
    erf_term = 1 + erf(1j * norm_separation)
    
    return pref * erf_term

def Q_term(gap: float, switching: float, a: float, regularization: str) -> complex:
    """Q[Ω, σ, a, λ]"""

    pre_factor = np.exp(-0.5*(switching*gap)**2)

    if regularization == "delta":
        return  pre_factor * (
            1/(8*pi) - 1j * switching**3 / (8*pi*a*(a**2 + switching**2))
        )
    elif regularization == "heaviside":
        return  pre_factor * (
                    1/(8*pi) - 1j * switching / (8*a*sqrt(2*pi))
                )
    elif regularization == "magical":
        return  pre_factor * (
            1/(8*pi)
        )


def QregHeavsde(gap: float, switching: float, a: float, coupling: float) -> complex:
    """QregHeavsde[Ω, σ, a, λ]"""
    return (coupling**2) * np.exp(-0.5*(switching*gap)**2) * (
        1/(8*pi) - 1j * switching/(8*sqrt(2*pi)*a)
    )

def Qmagic(gap: float, switching: float, a: float, coupling: float) -> complex:
    """Qmagic[Ω, σ, a, λ]"""
    return (coupling**2) * np.exp(-0.5*(switching*gap)**2) * (1/(8*pi))


# ----- 9x9 matrix builder -----

def twoqutrits_SUtwo(gap: float, switching: float,separation: float, a: float, Qfunc, coupling: float) -> np.ndarray:
    """
    Python version of:
      twoqutritsSUtwo[Ω, σ,separation, a, Q, λ]
    where Q is a callable: Q(Ω, σ, a, λ)
    """
    Lval   = L(gap, switching, coupling)
    Labval = Lab(gap, switching,separation, coupling)
    Mval   = M(gap, switching,separation, coupling)
    Qval   = Qfunc(gap, switching, a, coupling)

    return np.array([
        [1 - 2*Lval,           0, np.conjugate(Qval), 0, np.conjugate(Mval), 0, np.conjugate(Qval), 0, 0],
        [0,                    Lval, 0,               Labval, 0, 0, 0, 0, 0],
        [Qval,                 0,    0,               0,      0, 0, 0, 0, 0],
        [0,                    Labval, 0,             Lval,   0, 0, 0, 0, 0],
        [Mval,                 0,    0,               0,      0, 0, 0, 0, 0],
        [0,                    0,    0,               0,      0, 0, 0, 0, 0],
        [Qval,                 0,    0,               0,      0, 0, 0, 0, 0],
        [0,                    0,    0,               0,      0, 0, 0, 0, 0],
        [0,                    0,    0,               0,      0, 0, 0, 0, 0],
    ], dtype=complex)

# ----- utilities -----

def chop(x, tol=1e-12):
    """Like Mathematica's Chop: zero out tiny real/imag parts."""
    x = np.asarray(x)
    r = np.where(np.abs(x.real) < tol, 0.0, x.real)
    i = np.where(np.abs(x.imag) < tol, 0.0, x.imag)
    return r + 1j*i


# ----- example usage (mirrors your Eigenvalues[...] // N // Chop) -----
if name == "__main__":
    gap   = 2.0                                  # choose a numeric Ω
    switching = 1/gap
   separation     = 10/gap
    a     = 1e-4/gap
    coupling   = 1e-2
    A = twoqutrits_SUtwo(gap, switching,separation, a, QregDelta, coupling)
    evals = np.linalg.eigvals(A)
    print(chop(np.sort_complex(evals)))