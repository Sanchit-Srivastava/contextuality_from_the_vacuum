import numpy as np
from scipy import integrate, special

# --- helper: integrate complex-valued integrands with quad ---
def _quad_complex(f, a, b, **quad_kwargs):
    """Return integral of complex f over [a,b] via separate real/imag quads."""
    fr = lambda x: np.real(f(x))
    fi = lambda x: np.imag(f(x))
    vr, er = integrate.quad(fr, a, b, **quad_kwargs)
    vi, ei = integrate.quad(fi, a, b, **quad_kwargs)
    return vr + 1j*vi

# Eq. (4.9)
def smearedL(Omega, T, R, *, epsabs=1e-10, epsrel=1e-8, limit=200):
    pref = (9*T**2) / (4*np.pi*R**2)

    def integrand(k):
        j1 = special.spherical_jn(1, k*R)
        return (j1*j1) * np.exp(-0.5*T**2*(k - Omega)**2) / k

    val = integrate.quad(integrand, 0.0, np.inf,
                         epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
    return pref * val

# Eq. (4.10)
def smearedW(Omega, T, R, *, epsabs=1e-10, epsrel=1e-8, limit=200):
    # Use erfc(i x) = 1 - i*erfi(x) to avoid needing complex erfc directly
    pref = (9*T**2) / (8*np.pi*R**2) * np.exp(-0.5*T**2*Omega**2)
    rt2 = np.sqrt(2.0)

    def integrand(k):
        j1 = special.spherical_jn(1, k*R)
        gauss = np.exp(-0.5*T**2*k**2)
        erfc_i = 1.0 - 1j*special.erfi((T/rt2)*k)  # equals erfc(1j*k*T/rt2)
        return (j1*j1) * erfc_i * gauss / k

    val = _quad_complex(integrand, 0.0, np.inf,
                        epsabs=epsabs, epsrel=epsrel, limit=limit)
    return pref * val

# Eq. (4.11)
def smearedV(Omega, T, R, *, epsabs=1e-10, epsrel=1e-8, limit=200):
    # erfc(i x) = 1 - i*erfi(x) again
    pref = (9*T**2) / (8*np.pi*R**2) * np.exp(-0.125*T**2*Omega**2)
    rt2 = np.sqrt(2.0)

    def integrand(k):
        j1 = special.spherical_jn(1, k*R)
        shift = k - 0.5*Omega
        erfc_i = 1.0 - 1j*special.erfi((T/rt2)*shift)
        gauss_shift = np.exp(-0.5*T**2*(k - Omega)**2)
        return (j1*j1) * erfc_i * gauss_shift / k

    val = _quad_complex(integrand, 0.0, np.inf,
                        epsabs=epsabs, epsrel=epsrel, limit=limit)
    return pref * val

# Eq. (4.12)
def smearedLab(Omega, T, R, d, *, epsabs=1e-10, epsrel=1e-8, limit=200):
    pref = (9*T**2) / (4*np.pi*R**2)

    def integrand(k):
        j1 = special.spherical_jn(1, k*R)
        j0 = special.spherical_jn(0, k*d)
        return (j1*j1) * np.exp(-0.5*T**2*(k - Omega)**2) * j0 / k

    val = integrate.quad(integrand, 0.0, np.inf,
                         epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
    return pref * val

# Eq. (4.15)
def smearedM(Omega, T, R, d, *, epsabs=1e-10, epsrel=1e-8, limit=200):
    # erfc(i x) = 1 - i*erfi(x)
    pref = (9*T**2) / (4*np.pi*R**2) * np.exp(-0.5*T**2*Omega**2)
    rt2 = np.sqrt(2.0)

    def integrand(k):
        j1 = special.spherical_jn(1, k*R)
        j0 = special.spherical_jn(0, k*d)
        erfc_i = 1.0 - 1j*special.erfi((T/rt2)*k)
        gauss = np.exp(-0.5*T**2*k**2)
        return (j1*j1) * erfc_i * gauss * j0 / k

    val = _quad_complex(integrand, 0.0, np.inf,
                        epsabs=epsabs, epsrel=epsrel, limit=limit)
    return pref * val

# --- Example (remove or adapt) ---
if __name__ == "__main__":
    Omega, T, R, d = 1.0, 0.5, 0.3, 0.2
    print("L  =", smearedL(Omega, T, R))
    print("W  =", smearedW(Omega, T, R))
    print("V  =", smearedV(Omega, T, R))
    print("Lab=", smearedLab(Omega, T, R, d))
    print("M  =", smearedM(Omega, T, R, d))