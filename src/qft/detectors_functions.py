def Lab_term(gap: float, switching: float,separation: float, coupling: float) -> complex:
    """Lab[Ω, σ,separation, λ]"""

    if group == "HW":
            effective_gap = gap
        elif group == "SU2":
            effective_gap = -gap
        else:
            raise ValueError("Unsupported group")

    pref = coupling**2 * (1/(4*sqrt(pi))) * (switching/(sqrt(2)*d)) * np.exp(-(d**2)/(2*switching**2))
    term = np.exp(1j*d*gap) * erf(1j*(d/(sqrt(2)*switching)) + (gap*switching)/sqrt(2))
    return pref * (np.imag(term) + np.sin(gap*d))

def M_term(gap: float, switching: float,separation: float, detector_type: str) -> complex:
    """M[Ω, σ,separation, λ]"""
    
    norm_separation = (separation / (sqrt_2 * switching))

    pref =  1j * (1/(8*sqrt(pi))) * (1/norm_separation)
    pref *= np.exp(-(norm_separation**2)) * np.exp(-0.5*(switching*gap)**2)
    return pref * (1 + erf(1j*(norm_separation)))

