'''This module contains the functions to define the detector states'''

import numpy as np
from numpy import pi, sqrt
from scipy.special import erf, erfc  # complex-safe

# ---------- base 9x9 seed ----------
rho0 = np.zeros((9, 9), dtype=complex)
rho0[0, 0] = 1.0

# ---------- scalars (Ω is absorbed as 1 here, matching your defs) ----------
def L(sigma: float) -> complex:
    """
    Compute the complex-valued function L(sigma) defined by the following formula:

        L(sigma) = 1/(4*pi) * (exp(-0.5*sigma**2) - sqrt(pi) * (sigma/sqrt(2)) * erfc(sigma/sqrt(2)))

    Parameters:
        sigma (float): The parameter sigma

    Returns:
        complex: The evaluated result of the function L for the given sigma.
    """
    coefficient = 1 / (4 * pi)
    exp_term = np.exp(-0.5 * sigma**2)
    erfc_term = sqrt(pi) * (sigma / sqrt(2)) * erfc(sigma / sqrt(2))
    return coefficient * (exp_term - erfc_term)


def Lab(sigma: float, d: float) -> complex:
    """
    Compute the Lab function for given sigma and d.

    Parameters:
        sigma (float): Standard deviation parameter.
        d (float): Distance parameter.

    Returns:
        complex: Result of the Lab computation.
    """
    # Normalization and ratio factors
    normalization = 1 / (4 * sqrt(pi))
    ratio = sigma / (sqrt(2) * d)
    exp_factor = np.exp(- (d ** 2) / (2 * sigma ** 2))
    prefactor = normalization * ratio * exp_factor

    # Calculate the complex term involving the error function
    phase = np.exp(-1j * d)
    erf_arg = 1j * (d / (sqrt(2) * sigma)) - sigma / sqrt(2)
    error_component = erf(erf_arg)

    # Use the imaginary part of the complex multiplication
    term_imag = np.imag(phase * error_component)
    return prefactor * (term_imag - np.sin(d))


def M(sigma: float, d: float) -> complex:
    """
    Compute the function M defined by:
      M(sigma, d) = (1j/(4*sqrt(pi))) * (sigma/(sqrt(2)*d)) *
                    exp(-(d**2/(2*sigma**2) + 0.5*sigma**2)) *
                    (1 + erf(1j*d/(sqrt(2)*sigma)))
    
    Parameters:
        sigma (float): The sigma parameter.
        d (float): The d parameter.
        
    Returns:
        complex: The computed value of M.
    """
    coefficient = 1j / (4 * sqrt(pi)) * (sigma / (sqrt(2) * d))
    exponent = -((d**2) / (2 * sigma**2) + 0.5 * sigma**2)
    term = np.exp(exponent)
    error_component = 1 + erf(1j * (d / (sqrt(2) * sigma)))
    return coefficient * term * error_component


def QregDelta(sigma: float, a: float) -> complex:
    """
    Compute the QregDelta function:
    
        QregDelta(sigma, a) = exp(-0.5 * sigma^2) / (8*pi) * [1 - 1j * sigma^3 / (a * (a^2 + sigma^2))]
    
    Parameters:
        sigma (float): Standard deviation parameter.
        a (float): Parameter a.
    
    Returns:
        complex: The computed value.
    """
    # Precompute the common scaling factor.
    scale = np.exp(-0.5 * sigma**2) / (8 * pi)
    
    # Separate the expression into real and imaginary parts.
    real_component = 1
    imag_component = -sigma**3 / (a * (a**2 + sigma**2))
    
    return scale * (real_component + 1j * imag_component)


def QregHeaviside(sigma: float, a: float) -> complex:
    # Precompute the exponential damping factor.
    scale = np.exp(-0.5 * sigma**2)
    
    # Compute the real and imaginary coefficients separately.
    real_coeff = 1 / (8 * pi)
    imag_coeff = sigma / (8 * sqrt(2 * pi) * a)
    
    # Combine the terms into one complex number.
    return scale * (real_coeff - 1j * imag_coeff)


def Qmagic(sigma: float, a: float) -> complex:
    """
    Compute the Qmagic function defined as an exponential damping scaled factor.

    Parameters:
        sigma (float): Standard deviation parameter.
        a (float): Parameter a (currently not used in the formula).

    Returns:
        complex: The computed value of Qmagic.
    """
    return np.exp(-0.5 * sigma ** 2) / (8 * pi)



# ---------- matrices ----------
def rho_perturb(sigma: float, d: float, a: float, Q_func) -> np.ndarray:
    """
    Compute the perturbative correction matrix for the detector state.

    Parameters:
        sigma (float): Standard deviation parameter.
        d (float): Distance parameter.
        a (float): Parameter a.
        Qfunc (callable): Function to compute the Q-value.

    Returns:
        np.ndarray: A 9x9 complex matrix representing the perturbative correction.
    """
    # Evaluate the component functions
    L_value   = L(sigma)
    Lab_value = Lab(sigma, d)
    M_value   = M(sigma, d)
    Q_value   = Q_func(sigma, a)

    # Initialize a 9x9 zero matrix
    perturb_matrix = np.zeros((9, 9), dtype=complex)

    # Assign nonzero elements
    perturb_matrix[0, 0] = -2 * L_value
    perturb_matrix[0, 2] = np.conjugate(Q_value)
    perturb_matrix[0, 4] = np.conjugate(M_value)
    perturb_matrix[0, 6] = np.conjugate(Q_value)

    perturb_matrix[1, 1] = L_value
    perturb_matrix[1, 3] = Lab_value

    perturb_matrix[2, 0] = Q_value

    perturb_matrix[3, 1] = Lab_value
    perturb_matrix[3, 3] = L_value

    perturb_matrix[4, 0] = Q_value

    # Remaining entries are zeros by default

    return perturb_matrix

def rho(sigma: float, d: float, a: float, Qfunc, lam: float) -> np.ndarray:
    return rho0 + (lam**2) * rho_perturb(sigma, d, a, Qfunc)

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