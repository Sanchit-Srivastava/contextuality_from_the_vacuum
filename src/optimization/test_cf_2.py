# Extend the two-qutrit scenario by adding two more measurements
# M7 = (XZ)⊗I, M8 = I⊗(XZ), and add several commuting contexts that share measurements.
import numpy as np
from itertools import product
from scipy.optimize import linprog

np.set_printoptions(suppress=True, precision=6)

d = 3
omega = np.exp(2j * np.pi / d)

# Rebuild single-qudit X,Z
Z1 = np.diag([1, omega, omega**2]).astype(complex)
X1 = np.roll(np.eye(3, dtype=complex), 1, axis=1)

def kron(A, B):
    return np.kron(A, B)

# All 8 measurements
M1 = kron(Z1, np.eye(3))
M2 = kron(np.eye(3), Z1)
M3 = kron(X1, np.eye(3))
M4 = kron(np.eye(3), X1)
M5 = kron(X1, Z1)
M6 = kron(Z1, X1)
M7 = kron(X1 @ Z1, np.eye(3))  # XZ ⊗ I
M8 = kron(np.eye(3), X1 @ Z1)  # I ⊗ XZ
measurements = [M1, M2, M3, M4, M5, M6, M7, M8]

def spectral_projectors_three_labels(U):
    vals, vecs = np.linalg.eig(U)
    for k in range(vecs.shape[1]):
        v = vecs[:, k]
        idx = np.argmax(np.abs(v))
        phase = np.exp(-1j * np.angle(v[idx]))
        vecs[:, k] = v * phase / np.linalg.norm(v)
    roots = np.array([1, omega, omega**2])
    labels = [int(np.argmin(np.abs(roots - lam))) for lam in vals]
    n = U.shape[0]
    P = [np.zeros((n, n), dtype=complex) for _ in range(3)]
    for col, lbl in enumerate(labels):
        v = vecs[:, col].reshape(-1, 1)
        P[lbl] += v @ v.conj().T
    for i in range(3):
        P[i] = 0.5 * (P[i] + P[i].conj().T)
    return P

proj_by_meas = [spectral_projectors_three_labels(U) for U in measurements]

# Expanded set of commuting contexts with overlaps
contexts = [
    (0, 1),  # {Z⊗I, I⊗Z}
    (0, 3),  # {Z⊗I, I⊗X}
    (2, 1),  # {X⊗I, I⊗Z}
    (2, 3),  # {X⊗I, I⊗X}
    (4, 5),  # {X⊗Z, Z⊗X}
    (0, 7),  # {Z⊗I, I⊗XZ}
    (6, 1),  # {XZ⊗I, I⊗Z}
    (6, 7),  # {XZ⊗I, I⊗XZ}
    (2, 7),  # {X⊗I, I⊗XZ}
    (6, 3),  # {XZ⊗I, I⊗X}
]

# State: |H3><H3| ⊗ I/3
e0 = np.array([1, 0, 0], dtype=complex)
e1 = np.array([0, 1, 0], dtype=complex)
H3 = (e0 + e1) / np.sqrt(2)
rho1 = np.outer(H3, H3.conj())
rho2 = np.eye(3, dtype=complex) / 3.0
rho = kron(rho1, rho2)

# Build v_e
ve = []
row_index = []
for ci, (mi, mj) in enumerate(contexts):
    Pi = proj_by_meas[mi]
    Pj = proj_by_meas[mj]
    for a in range(3):
        for b in range(3):
            Pab = Pi[a] @ Pj[b]
            Pab = 0.5 * (Pab + Pab.conj().T)
            prob = float(np.real_if_close(np.trace(Pab @ rho)))
            prob = max(0.0, min(1.0, prob))
            ve.append(prob)
            row_index.append((ci, a, b))
ve = np.array(ve)
m = len(ve)

# Incidence matrix
num_meas = len(measurements)  # 8
num_globals = 3 ** num_meas   # 6561
M = np.zeros((m, num_globals), dtype=float)

def ternary_digits(n, width):
    digs = []
    for _ in range(width):
        digs.append(n % 3)
        n //= 3
    return digs[::-1]

for g_idx in range(num_globals):
    outcomes = ternary_digits(g_idx, num_meas)
    for r, (ci, a, b) in enumerate(row_index):
        mi, mj = contexts[ci]
        if outcomes[mi] == a and outcomes[mj] == b:
            M[r, g_idx] = 1.0

# Solve LP
c = -np.ones(num_globals)
bounds = [(0, None)] * num_globals
res = linprog(c, A_ub=M, b_ub=ve, bounds=bounds, method="highs", options={"presolve": True})

if not res.success:
    print("LP failed:", res.message)
else:
    NCF = -res.fun
    CF = 1.0 - NCF
    print("Two-qutrit CF with 10 overlapping contexts (8 measurements)")
    print("  #contexts:", len(contexts), "  rows:", m, "  cols:", num_globals)
    print("  Noncontextual fraction (NCF):", float(NCF))
    print("  Contextual fraction (CF):", float(CF))

    # Dual witness (optional)
    cd = ve.copy()
    Adual = -M.T
    bdual = -np.ones(num_globals)
    res_dual = linprog(cd, A_ub=Adual, b_ub=bdual, bounds=[(0, None)] * m, method="highs", options={"presolve": True})
    if res_dual.success:
        y = res_dual.x
        noncontextual_bound = float((M.T @ y).max())
        quantum_value = float(ve @ y)
        print("\nDual-based witness summary:")
        print("  Noncontextual bound R =", noncontextual_bound)
        print("  Value on our model a·v_e =", quantum_value)
        for k in np.argsort(-y)[:12]:
            ci, a, b = row_index[k]
            print(f"   coeff={y[k]:.6f} on context C{ci+1} (meas {contexts[ci][0]} & {contexts[ci][1]}), outcome ({a},{b})")
