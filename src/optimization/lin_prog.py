import numpy as np
from scipy.optimize import linprog

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
    
    c = -np.ones(M.shape[1])  # Objective vector: length 81
    bounds = [(0, 1)] * M.shape[1]  # b >= 0, b <= 1
    
    # Empirical data
    E = empirical_model(rho)
    
    # Solve using HiGHS
    result = linprog(c, A_ub=M, b_ub=E, bounds=bounds, method='highs')
    
    # === Output ===
    output = {
        'success': result.success,
        'result': result
    }
    
    if result.success:
        output['b'] = 1 - np.dot(-c, result.x)
    
    return output