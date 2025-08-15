import numpy as np
from scipy.optimize import linprog
from scipy import sparse as sp

try:
    from .incidence_matrix import M
    from ..utils.measurements import empirical_model
except ImportError:
    # Fall back to absolute imports when run directly
    from optimization.incidence_matrix import M
    from utils.measurements import empirical_model


def contextual_fraction(rho):
    """
    Solve the linear program to maximize the contextual fraction.
    
    Args:
        rho: The density matrix/quantum state
        
    Returns:
        dict: Contains the optimization result with keys:
            - 'success': bool, whether optimization succeeded
            - 'b': float, contextual fraction value (if successful)
            - 'result': scipy.optimize.OptimizeResult object
    """
    # === Linear Program ===
    # maximize 1.b  -> minimize -1.b
    
    c = -np.ones(M.shape[1])  # Objective vector: length = number of deterministic models
    bounds = [(0, 1)]  # probabilities in [0,1]

    # Empirical data
    E = empirical_model(rho)

    # Always enforce a probability-mass cap: sum(x) ≤ 1 + tiny_slack
    # This forbids solutions with sum(x) > 1 while still allowing contextual
    # behaviors where the optimum may have sum(x) < 1.
    ones_row = np.ones((1, M.shape[1]))
    if sp.issparse(M):
        A_ub_ext = sp.vstack([M, sp.csr_matrix(ones_row)], format='csr')
    else:
        A_ub_ext = np.vstack([M, ones_row])
    sum_cap_slack = 1e-12
    b_ub_ext = np.concatenate([E, np.array([1.0 + sum_cap_slack])])
    
    # Solve using HiGHS
    result = linprog(
        c,
        A_ub=A_ub_ext,
        b_ub=b_ub_ext,
        bounds=bounds,
        method='highs',
        options={
            'primal_feasibility_tolerance': 1e-10,
            'dual_feasibility_tolerance': 1e-10,
            'ipm_optimality_tolerance': 1e-10,
            'presolve': True,
        },
    )
    
    # === Output ===
    output = {
        'success': result.success,
        'result': result
    }
    
    if result.success:
        x = result.x.copy()
        # Clip tiny numerical negatives/overshoots into [0,1]
        x = np.clip(x, 0.0, 1.0)

        sum_x = x.sum()
        note = None
        # With the cap enforced in the LP, we should not see sum(x) > 1.
        # Guard anyway: normalize tiny overshoot; flag hard overshoot as failure.
        if sum_x > 1.0:
            overshoot = sum_x - 1.0
            if overshoot <= 1e-10:
                x /= sum_x
                sum_x = 1.0
                note = "normalized_post_solve_small_overshoot"
            else:
                # Treat as an invalid solution; do not return an >1 sum.
                return {
                    'success': False,
                    'result': result,
                    'error': f"LP returned sum(x)={sum_x:.12g} exceeding 1 by more than tolerance",
                }

        # Contextual fraction
        output['b'] = 1 + result.fun  # equals 1 - sum(x) if c = -1
        output['x'] = x
        output['sum_x'] = float(sum_x)
        if note:
            output['note'] = note

    return output