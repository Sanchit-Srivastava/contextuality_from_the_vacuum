''' This module contains utility functions to validate quantum states'''

import numpy as np

def is_valid_state(rho: np.ndarray, tolerance: float = 1e-10, 
                   return_details: bool = False):
    """Check if a given matrix is a valid density matrix.

    A valid density matrix must be:
    - Hermitian,
    - have a trace of 1,
    - be positive semidefinite.
    
    Args:
        rho: The matrix to check
        tolerance: Numerical tolerance for checks
        return_details: If True, return detailed validation info
        
    Returns:
        If return_details=False: bool indicating validity
        If return_details=True: dict with validation details
    """
    validation_results = {
        'is_valid': True,
        'failed_checks': [],
        'details': {}
    }
    
    # Check if matrix is square
    if rho.shape[0] != rho.shape[1]:
        validation_results['is_valid'] = False
        validation_results['failed_checks'].append('not_square')
        validation_results['details']['shape'] = f"Matrix shape {rho.shape} is not square"
        if not return_details:
            return False
        # For non-square matrices, skip other checks that require square matrices
        return validation_results
    
    # Check if the matrix is Hermitian
    hermitian_diff = np.max(np.abs(rho - rho.T.conj()))
    is_hermitian = hermitian_diff <= tolerance
    validation_results['details']['hermitian_deviation'] = hermitian_diff
    
    if not is_hermitian:
        validation_results['is_valid'] = False
        validation_results['failed_checks'].append('not_hermitian')
        validation_results['details']['hermitian_error'] = f"Max deviation from Hermitian: {hermitian_diff:.2e}"
        if not return_details:
            return False

    # Check if the trace equals 1
    trace_value = np.trace(rho)
    trace_deviation = abs(trace_value - 1.0)
    is_trace_one = trace_deviation <= tolerance
    validation_results['details']['trace'] = trace_value
    validation_results['details']['trace_deviation'] = trace_deviation
    
    if not is_trace_one:
        validation_results['is_valid'] = False
        validation_results['failed_checks'].append('incorrect_trace')
        validation_results['details']['trace_error'] = f"Trace = {trace_value:.6f}, deviation = {trace_deviation:.2e}"
        if not return_details:
            return False

    # Check positive semidefiniteness using eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(rho)
        min_eigenvalue = np.min(eigenvalues)
        validation_results['details']['eigenvalues'] = eigenvalues
        validation_results['details']['min_eigenvalue'] = min_eigenvalue
        
        is_positive_semidefinite = min_eigenvalue >= -tolerance
        
        if not is_positive_semidefinite:
            validation_results['is_valid'] = False
            validation_results['failed_checks'].append('not_positive_semidefinite')
            validation_results['details']['eigenvalue_error'] = f"Minimum eigenvalue: {min_eigenvalue:.2e}"
            if not return_details:
                return False
                
    except np.linalg.LinAlgError as e:
        validation_results['is_valid'] = False
        validation_results['failed_checks'].append('eigenvalue_computation_failed')
        validation_results['details']['linalg_error'] = str(e)
        if not return_details:
            return False

    if return_details:
        return validation_results
    else:
        return validation_results['is_valid']


def print_validation_results(validation_results: dict, matrix_name: str = "Matrix"):
    """Print validation results in a readable format.
    
    Args:
        validation_results: Results from is_valid_state with return_details=True
        matrix_name: Name to use when describing the matrix
    """
    print(f"\n{matrix_name} Validation Results:")
    print("=" * 40)
    
    if validation_results['is_valid']:
        print("✓ VALID DENSITY MATRIX")
    else:
        print("✗ INVALID DENSITY MATRIX")
        print(f"Failed checks: {', '.join(validation_results['failed_checks'])}")
    
    print("\nDetailed Analysis:")
    print("-" * 20)
    
    details = validation_results['details']
    
    # Shape check
    if 'shape' in details:
        print(f"Shape: {details['shape']}")
    
    # Hermitian check
    if 'hermitian_deviation' in details:
        hermitian_dev = details['hermitian_deviation']
        status = "✓" if hermitian_dev <= 1e-10 else "✗"
        print(f"Hermitian check {status}: Max deviation = {hermitian_dev:.2e}")
        
    # Trace check
    if 'trace' in details:
        trace_val = details['trace']
        trace_dev = details['trace_deviation']
        status = "✓" if trace_dev <= 1e-10 else "✗"
        print(f"Trace check {status}: Tr(ρ) = {trace_val:.6f}, deviation = {trace_dev:.2e}")
    
    # Eigenvalue check
    if 'eigenvalues' in details:
        eigenvals = details['eigenvalues']
        min_eig = details['min_eigenvalue']
        status = "✓" if min_eig >= -1e-10 else "✗"
        print(f"Positive semidefinite {status}: Min eigenvalue = {min_eig:.2e}")
        print(f"All eigenvalues: {eigenvals}")
    
    # Error messages
    if 'hermitian_error' in details:
        print(f"Hermitian error: {details['hermitian_error']}")
    if 'trace_error' in details:
        print(f"Trace error: {details['trace_error']}")
    if 'eigenvalue_error' in details:
        print(f"Eigenvalue error: {details['eigenvalue_error']}")
    if 'linalg_error' in details:
        print(f"Linear algebra error: {details['linalg_error']}")
    
    print()


def validate_and_print(rho: np.ndarray, matrix_name: str = "Matrix", 
                       tolerance: float = 1e-10):
    """Validate a density matrix and print detailed results.
    
    Args:
        rho: The matrix to validate
        matrix_name: Name for the matrix in output
        tolerance: Numerical tolerance for validation
        
    Returns:
        bool: Whether the matrix is valid
    """
    results = is_valid_state(rho, tolerance=tolerance, return_details=True)
    print_validation_results(results, matrix_name)
    return results['is_valid']
