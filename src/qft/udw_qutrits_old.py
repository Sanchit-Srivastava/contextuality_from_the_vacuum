'''This module generates the states of the UDW qutrit detectors'''


import numpy as np
from numpy import pi, sqrt
from scipy import integrate, special
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


################ Helper functions ################
# Suggested by AI 
# We separate the real and imaginary parts of the integrand and 
##integrate them separately. 
# --- helper: integrate complex-valued integrands with quad ---

def _quad_complex(f, a, b, **quad_kwargs):
    """Return integral of complex f over [a,b] via separate real/imag quads."""
    fr = lambda x: np.real(f(x))
    fi = lambda x: np.imag(f(x))
    vr, er = integrate.quad(fr, a, b, **quad_kwargs)
    vi, ei = integrate.quad(fi, a, b, **quad_kwargs)
    return vr + 1j*vi

#################################################

def L_term(gap: float, switching: float, smearing: float, detector_type: str, group: str) -> float:
    """Compute the L probability."""
    if detector_type != "point_like" and detector_type != "smeared":
        raise ValueError("Unsupported detector_type")
    
    if group == "HW":
        effective_gap = gap
    elif group == "SU2":
        effective_gap = -gap
    else:
        raise ValueError("Unsupported group")

    if detector_type == "point_like":
        # Point like detectors
        expr = np.exp(-0.5 * (switching * effective_gap) ** 2)
        arg = switching * effective_gap / sqrt(2)
        term = (1 / (4 * pi)) * (expr + sqrt(pi) * arg * (2 - erfc(arg)))
        return term

    elif detector_type == "smeared":
        # smeared detectors
        # def smearedL(Omega, T, R, *, 
        epsabs=1e-10
        epsrel=1e-8
        limit=200
        pre_factor = (9*switching**2) / (4*np.pi*smearing**2)

        def integrand(k):
            # Avoid 0/0 at k=0; the limit is 0 so return 0 explicitly
            if k == 0.0:
                return 0.0
            j1 = special.spherical_jn(1, k*smearing)
            return ((j1*j1) * np.exp(-0.5*(switching**2)*(k - gap)**2)) / k

        val = integrate.quad(integrand, 0.0, np.inf,
                            epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        return pre_factor * val



def Lab_term(gap: float, switching: float, separation: float, smearing: float, detector_type: str, group: str) -> complex:
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
    
    if detector_type == "point_like":   
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

    elif detector_type == "smeared":
        # Smeared detectors
        epsabs=1e-10
        epsrel=1e-8
        limit=200

        pre_factor = (9*switching**2) / (4*np.pi*smearing**2)

        def integrand(k):
            # Avoid 0/0 at k=0; the limit is 0 so return 0 explicitly
            if k == 0.0:
                return 0.0
            j1 = special.spherical_jn(1, k*smearing)
            j0 = special.spherical_jn(0, k*separation)
            return ((j1*j1) * np.exp(-0.5*(switching**2)*(k - gap)**2) * j0)/ k

        val = integrate.quad(integrand, 0.0, np.inf,
                            epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        return pre_factor * val




def M_term(gap: float, switching: float, separation: float, smearing: float, detector_type: str) -> complex:
    """M[Ω, σ, separation, λ]"""
    # M is even w.r.t gap, so we don't differentiate between the groups
    if detector_type != "point_like" and detector_type != "smeared":
        raise ValueError("Unsupported detector_type")
    
    # Pre-compute common terms
    sqrt_2 = sqrt(2)
    sqrt_pi = sqrt(pi)
    
    if detector_type == "point_like":
        # Compute normalized separation
        norm_separation = separation / (sqrt_2 * switching)
        
        # Compute prefix factor
        pref = 1j / (8 * sqrt_pi * norm_separation)
        pref *= np.exp(-norm_separation**2 - 0.5 * (switching * gap)**2)
        
        # Compute error function term
        erf_term = 1 + erf(1j * norm_separation)
        
        return pref * erf_term

    elif detector_type == "smeared":
        epsabs=1e-10
        epsrel=1e-8
        limit=200
        # adding -1 pre factor to eq. (4.15)
        pre_factor = -1 * (9*switching**2) / (4*np.pi*smearing**2) * np.exp(-0.5*smearing**2*gap**2)
        

        def integrand(k):
            # Avoid 0/0 at k=0; the limit is 0 so return 0 explicitly (complex)
            if k == 0.0:
                return 0.0 + 0j
            j1 = special.spherical_jn(1, k*smearing)
            j0 = special.spherical_jn(0, k*separation)
            term =  special.erfc(1j*k*switching/sqrt(2))
            gauss = np.exp(-0.5*switching**2*k**2)
            return ((j1*j1) * term * gauss * j0) / k

        val = _quad_complex(integrand, 0.0, np.inf,
                            epsabs=epsabs, epsrel=epsrel, limit=limit)
        return pre_factor * val




def Q_term(gap: float, switching: float, regulator: float, regularization: str, smearing: float, detector_type: str) -> complex:
    """Q[Ω, σ, a, λ] - Compute Q term with different regularization schemes."""
    # This is only for SU2 group. Don't even try to use it for HW. We will come for you. 

    if detector_type == "point_like":
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

    elif detector_type == "smeared": 
        epsabs=1e-10
        epsrel=1e-8
        limit=200

        # Adding a -1 pre factor to Eq. 4.10
        pre_factor = -1*(9*switching**2) / (8*np.pi*smearing**2) * np.exp(-0.5*switching**2*gap**2)

        def integrand(k):
            # Avoid 0/0 at k=0; the limit is 0 so return 0 explicitly (complex)
            if k == 0.0:
                return 0.0 + 0j
            j1 = special.spherical_jn(1, k*smearing)
            gauss = np.exp(-0.5*switching**2*k**2)
            term = special.erfc(1j*k*switching/np.sqrt(2.0))  
            return ((j1*j1) * term * gauss) / k

        val = _quad_complex(integrand, 0.0, np.inf,
                            epsabs=epsabs, epsrel=epsrel, limit=limit)
        return pre_factor * val


def V_term(gap: float, switching: float, smearing: float) -> complex:
    ''' for smeared detectors only'''
    epsabs=1e-10
    epsrel=1e-8 
    limit=200

    # Adding a prefactor of -1 to eq. 4.11
    pre_factor = -1 * (9*switching**2) / (8*np.pi*smearing**2) * np.exp((-1/8)*switching**2*gap**2)

    def integrand(k):
        # Avoid 0/0 at k=0; the limit is 0 so return 0 explicitly (complex)
        if k == 0.0:
            return 0.0 + 0j
        j1 = special.spherical_jn(1, k*smearing)
        shift = k - 0.5*gap
        term = special.erfc((1j*switching/np.sqrt(2.0))*shift)
        gauss_shift = np.exp(-0.5*switching**2*(k - gap)**2)
        return ((j1*j1) * term * gauss_shift) / k

    val = _quad_complex(integrand, 0.0, np.inf,
                        epsabs=epsabs, epsrel=epsrel, limit=limit)
    return pre_factor * val



# ---------- matrices ---------
def rho_perturb(gap: float, switching: float, separation: float, regulator: float, smearing: float, regularization: str, detector_type: str, group: str) -> np.ndarray:
    """
    Compute the perturbative correction matrix for the detector state.

    Parameters:
        a (float): Regulator parameter

    Returns:
        np.ndarray: A 9x9 complex matrix representing the perturbative correction.
    """

    # Evaluate the component functions
    L_value = L_term(gap, switching, smearing,  detector_type, group)
    Lab_value = Lab_term(gap, switching, separation, smearing,  detector_type, group)
    M_value = M_term(gap, switching, separation, smearing, detector_type) # Same for both groups SU2 and HW
    Q_value = Q_term(gap, switching, regulator, regularization, smearing, detector_type)
    V_value = V_term(gap, switching, smearing)

    # Initialize a 9x9 zero matrix
    perturb_matrix = np.zeros((9, 9), dtype=complex)

    if group == "SU2":
        # Assign nonzero elements
        Q_value = Q_term(gap, switching, regulator, regularization, smearing, detector_type)
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

    elif group == "HW":
        # Row 1
        perturb_matrix[0,0] = -4*L_value
        perturb_matrix[0,1] = np.conjugate(V_value)
        perturb_matrix[0,2] = np.conjugate(V_value)
        perturb_matrix[0,3] = np.conjugate(V_value)
        perturb_matrix[0,4] = np.conjugate(M_value)
        perturb_matrix[0,5] = np.conjugate(M_value)
        perturb_matrix[0,6] = np.conjugate(V_value)
        perturb_matrix[0,7] = np.conjugate(M_value)
        perturb_matrix[0,8] = np.conjugate(M_value)

        #Row 2 
        perturb_matrix[1,0] = V_value
        perturb_matrix[1,1] = L_value
        perturb_matrix[1,2] = L_value
        # Not conjugating Lab_value because they are real
        perturb_matrix[1,3] = Lab_value
        perturb_matrix[1,6] = Lab_value

        #Row 3
        perturb_matrix[2,0] = V_value
        perturb_matrix[2,1] = L_value
        perturb_matrix[2,2] = L_value
        perturb_matrix[2,3] = Lab_value
        perturb_matrix[2,6] = Lab_value

        # Row 4
        perturb_matrix[3,0] = V_value
        perturb_matrix[3,1] = Lab_value
        perturb_matrix[3,2] = Lab_value
        perturb_matrix[3,3] = L_value
        perturb_matrix[3,6] = L_value

        # Row 5
        perturb_matrix[4,0] = M_value

        # Row 6
        perturb_matrix[5,0] = M_value

        # Row 7 #same as row 4
        perturb_matrix[6,0] = V_value
        perturb_matrix[6,1] = Lab_value
        perturb_matrix[6,2] = Lab_value
        perturb_matrix[6,3] = L_value
        perturb_matrix[6,6] = L_value

        # Row 8
        perturb_matrix[7,0] = M_value

        # Row 9
        perturb_matrix[8,0] = M_value

        return perturb_matrix

def detector_state(gap: float, switching: float, separation: float, regulator: float, smearing: float, regularization: str, detector_type: str, group: str, lam: float) -> np.ndarray:
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
    rho_second = lam**2 * rho_perturb(gap, switching, separation, regulator, smearing, regularization, detector_type, group)  
    result = rho0 + rho_second

    return result


if __name__ == "__main__":
    # Test parameters
    gap = 1
    switching = 1
    separation = 10
    regulator = 1
    regularization = "magical"
    detector_type = "smeared"
    smearing = 0.1
    group = "HW"
    lam = 1e-2

    print("Detector state")
    print("=" * 50)
    print(f"Parameters: gap={gap}, switching={switching}, separation={separation}, regulator={regulator}, regularization={regularization}, detector_type={detector_type}, group={group}, lambda={lam}")
    print()
    
    # Test with QregDelta Q-function
    print("Testing with QregDelta Q-function:")
    print("-" * 30)

    rho = detector_state(gap, switching, separation, regulator, smearing, regularization, detector_type, group, lam)
    # Print the state matrix
    print("Generated density matrix:")
    print(rho)
    print()
    
    # Check if it's a valid density matrix
    print("Density Matrix Validation:")
    print("-" * 25)

    validate_and_print(rho, "Generated Density Matrix")