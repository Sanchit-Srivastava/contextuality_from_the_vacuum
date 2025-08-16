
import numpy as np
try:
    from . import operators
except ImportError:
    import operators

w = np.exp(2j * np.pi / 3)

def sp(u, v):
    a, b, c, d = u
    a2, b2, c2, d2 = v
    return (a * b2 - b * a2 + c * d2 - d * c2) % 3

def Weyl(u):
    u = np.array(u, int) % 3
    return operators.pauli(u)

def test_weyl_algebra():
    for _ in range(100):
        u = np.random.randint(0, 3, 4)
        v = np.random.randint(0, 3, 4)
        lhs = Weyl(u) @ Weyl(v)
        rhs = (w ** (2 * sp(u, v))) * Weyl((u + v) % 3)
        assert np.allclose(lhs, rhs, atol=1e-12), f"Failed for u={u}, v={v}"

if __name__ == "__main__":
    test_weyl_algebra()
    print("All Weyl algebra tests passed.")
