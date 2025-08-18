
import numpy as np
try:
    from . import operators
    from .symplectic import symplectic_product as sp
except ImportError:
    import operators
    from symplectic import symplectic_product as sp

w = np.exp(2j * np.pi / 3)

# sp now imported from utils.symplectic

def Weyl(u):
    u = np.array(u, int) % 3
    return operators.weyl(u)

def test_weyl_algebra():
    for _ in range(100):
        u = np.random.randint(0, 3, 4)
        v = np.random.randint(0, 3, 4)
        lhs = Weyl(u) @ Weyl(v)
        rhs = (w **  (1*sp(u, v))) * Weyl((v + u) % 3)
        # rhs = (w **  (2*sp(u, v))) * Weyl(v) * Weyl(u)
        assert np.allclose(lhs, rhs, atol=1e-12), f"Failed for u={u}, v={v}"

if __name__ == "__main__":
    test_weyl_algebra()
    print("All Weyl algebra tests passed.")


