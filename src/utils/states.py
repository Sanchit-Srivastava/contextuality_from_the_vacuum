"""
Utility functions for creating and analyzing quantum states for 
two-qutrit systems.
"""

import numpy as np
from .measurements import empirical_model


def create_maximally_mixed_state():
    """Create a maximally mixed state for two qutrits (9x9 identity/9)."""
    return np.eye(9) / 9


def create_product_state():
    """Create a product state |0⟩⊗|0⟩ for two qutrits."""
    state_0 = np.array([1, 0, 0])  # |0⟩ state for a qutrit
    product_state = np.kron(state_0, state_0)  # |0⟩⊗|0⟩
    return np.outer(product_state, product_state.conj())


def create_maximally_entangled_state():
    """Create a maximally entangled state for two qutrits."""
    # |ψ⟩ = (|00⟩ + |11⟩ + |22⟩) / √3
    state_00 = np.kron([1, 0, 0], [1, 0, 0])  # |00⟩
    state_11 = np.kron([0, 1, 0], [0, 1, 0])  # |11⟩
    state_22 = np.kron([0, 0, 1], [0, 0, 1])  # |22⟩
    
    psi = (state_00 + state_11 + state_22) / np.sqrt(3)
    return np.outer(psi, psi.conj())


def create_custom_state(alpha=1/np.sqrt(2), beta=1/np.sqrt(2)):
    """
    Create a custom superposition state on the first qutrit, 
    product with |0⟩ on second.
    |ψ⟩ = (α|0⟩ + β|1⟩) ⊗ |0⟩
    """
    # Normalize coefficients
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha/norm, beta/norm
    
    first_qutrit = np.array([alpha, beta, 0])  # Superposition on first qutrit
    second_qutrit = np.array([1, 0, 0])        # |0⟩ on second qutrit
    
    product_state = np.kron(first_qutrit, second_qutrit)
    return np.outer(product_state, product_state.conj())


def print_state_info(state, name):
    """Print information about a quantum state."""
    print(f"\n{'='*50}")
    print(f"State: {name}")
    print(f"{'='*50}")
    print(f"Trace: {np.trace(state):.6f}")
    print(f"Hermitian: {np.allclose(state, state.conj().T)}")
    eigenvals = np.linalg.eigvals(state)
    print(f"Positive semidefinite: {np.all(eigenvals >= -1e-10)}")
    


def get_default_test_states():
    """Return a dictionary of default quantum states for testing."""
    return {
        "Maximally Mixed State": create_maximally_mixed_state(),
        "Product State |00⟩": create_product_state(),
        "Maximally Entangled State": create_maximally_entangled_state(),
        "Custom Superposition": create_custom_state(alpha=1, beta=1j),
    }
