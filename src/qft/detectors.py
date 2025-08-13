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
    

def Lab_term(gap: float, switching: float, separation: float, detector_type: str, group: str) -> complex:
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

def Q_term(gap: float, switching: float, regulator: float, regularization: str) -> complex:
    """Q[Ω, σ, a, λ] - Compute Q term with different regularization schemes."""
    
    # Common exponential factor
    a = regulator
    exp_factor = -1 * np.exp(-0.5 * (switching * gap)**2)
    base_term = 1 / (8 * pi)
    
    if regularization == "delta":
        imaginary_term = 1j * switching**3 / (8 * pi * a * (a**2 + switching**2))
        return exp_factor * (base_term - imaginary_term)
    
    elif regularization == "heaviside":
        imaginary_term = 1j * switching / (8 * a * sqrt(2 * pi))
        return exp_factor * (base_term - imaginary_term)
    
    elif regularization == "magical":
        return exp_factor * base_term
    
    else:
        raise ValueError(f"Unsupported regularization type: {regularization}")



# ----- 9x9 matrix builder -----

# def twoqutrits_SUtwo(gap: float, switching: float,separation: float, a: float, Qfunc, coupling: float) -> np.ndarray:
#     """
#     Python version of:
#       twoqutritsSUtwo[Ω, σ,separation, a, Q, λ]
#     where Q is a callable: Q(Ω, σ, a, λ)
#     """
#     Lval   = L(gap, switching, coupling)
#     Labval = Lab(gap, switching,separation, coupling)
#     Mval   = M(gap, switching,separation, coupling)
#     Qval   = Qfunc(gap, switching, a, coupling)

#     return np.array([
#         [1 - 2*Lval,           0, np.conjugate(Qval), 0, np.conjugate(Mval), 0, np.conjugate(Qval), 0, 0],
#         [0,                    Lval, 0,               Labval, 0, 0, 0, 0, 0],
#         [Qval,                 0,    0,               0,      0, 0, 0, 0, 0],
#         [0,                    Labval, 0,             Lval,   0, 0, 0, 0, 0],
#         [Mval,                 0,    0,               0,      0, 0, 0, 0, 0],
#         [0,                    0,    0,               0,      0, 0, 0, 0, 0],
#         [Qval,                 0,    0,               0,      0, 0, 0, 0, 0],
#         [0,                    0,    0,               0,      0, 0, 0, 0, 0],
#         [0,                    0,    0,               0,      0, 0, 0, 0, 0],
#     ], dtype=complex)

# ---------- matrices ----------
def rho_perturb(gap: float, switching: float, separation: float, regulator: float, regularization: str, detector_type: str, group: str) -> np.ndarray:
    """
    Compute the perturbative correction matrix for the detector state.

    Parameters:
        a (float): Regulator parameter

    Returns:
        np.ndarray: A 9x9 complex matrix representing the perturbative correction.
    """

    # Evaluate the component functions
    L_value = L_term(gap, switching, detector_type, group)
    Lab_value = Lab_term(gap, switching, separation, detector_type, group)
    M_value = M_term(gap, switching, separation, detector_type) # Same for both groups SU2 and HW

    # Initialize a 9x9 zero matrix
    perturb_matrix = np.zeros((9, 9), dtype=complex)

    if group == "SU2":
        # Assign nonzero elements
        Q_value = Q_term(gap, switching, regulator, regularization) # For point like and SU2 only
        perturb_matrix[0, 0] = -2 * L_value
        perturb_matrix[0, 2] = np.conjugate(Q_value)
        perturb_matrix[0, 4] = np.conjugate(M_value)
        perturb_matrix[0, 6] = np.conjugate(Q_value)

        perturb_matrix[1, 1] = L_value
        perturb_matrix[1, 3] = Lab_value

        perturb_matrix[2, 0] = Q_value

        perturb_matrix[3, 1] = Lab_value #not conjugating because already real
        perturb_matrix[3, 3] = L_value

        perturb_matrix[4, 0] = M_value

        perturb_matrix[6, 0] = Q_value

        # Remaining entries are zeros by default
        return perturb_matrix

def detector_state(gap: float, switching: float, separation: float, regulator: float, regularization: str, detector_type: str, group: str, lam: float) -> np.ndarray:
    """
    Compute the density matrix by adding a perturbative correction to the seed state.

    Parameters:
        sigma (float): Standard deviation parameter.
        d (float): Distance parameter.
        a (float): Parameter a.
        Q_func (callable): Function that computes the Q-value.
        lam (float): Perturbation strength parameter.

    Returns:
        np.ndarray: The resulting 9x9 complex density matrix.
    """
    rho_second = lam**2 * rho_perturb(gap, switching, separation, regulator, regularization, detector_type, group)  
    result = rho0 + rho_second

    return result

# ----- utilities -----
def chop(x, tol=1e-12):
    """Like Mathematica's Chop: zero out tiny real/imag parts."""
    x = np.asarray(x)
    r = np.where(np.abs(x.real) < tol, 0.0, x.real)
    i = np.where(np.abs(x.imag) < tol, 0.0, x.imag)
    return r + 1j*i

if __name__ == "__main__":
    # Test parameters
    gap = 10
    switching = 1
    separation = 10
    regulator = 1
    regularization = "magical"
    detector_type = "point_like"
    group = "SU2"
    lam = 1e-2

    print("Detector state")
    print("=" * 50)
    print(f"Parameters: gap={gap}, switching={switching}, separation={separation}, regulator={regulator}, regularization={regularization}, detector_type={detector_type}, group={group}, lambda={lam}")
    print()
    
    # Test with QregDelta Q-function
    print("Testing with QregDelta Q-function:")
    print("-" * 30)


    rho = detector_state(gap, switching, separation, regulator, regularization, detector_type, group, lam)
    # Print the state matrix
    print("Generated density matrix:")
    print(rho)
    print()
    
    # Check if it's a valid density matrix
    print("Density Matrix Validation:")
    print("-" * 25)

    validate_and_print(rho, "Generated Density Matrix")