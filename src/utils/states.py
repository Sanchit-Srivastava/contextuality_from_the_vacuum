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
    return product_state[:, np.newaxis] @ product_state.conj().T[np.newaxis, :]


def create_maximally_entangled_state():
    """Create a maximally entangled state for two qutrits."""
    # |ψ⟩ = (|00⟩ + |11⟩ + |22⟩) / √3
    state_00 = np.kron([1, 0, 0], [1, 0, 0])  # |00⟩
    state_11 = np.kron([0, 1, 0], [0, 1, 0])  # |11⟩
    state_22 = np.kron([0, 0, 1], [0, 0, 1])  # |22⟩
    
    psi = (state_00 + state_11 + state_22) / np.sqrt(3)
    return psi[:, np.newaxis] @ psi.conj().T[np.newaxis, :]


def tensor_state(alpha, beta):
    """
    Create a custom superposition state on the first qutrit, 
    product with |0⟩ on second.
    |ψ⟩ = (α|0⟩ + β|1⟩) ⊗ |0⟩
    """
    # Normalize coefficients
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha/norm, beta/norm
    

    first_qutrit = np.array([alpha, beta, 0])  # Superposition on first qutrit
    first_qutrit_rho = first_qutrit[:, np.newaxis] @ first_qutrit.conj().T[np.newaxis, :]
    # second_qutrit = np.array([1, 0, 0])        # |0⟩ on second qutrit
    second_qutrit = np.array([1, 0, 0])
    second_qutrit_rho = second_qutrit[:, np.newaxis] @ second_qutrit.conj().T[np.newaxis, :]
    # second_qutrit_rho = (1/3)*np.eye(3,3)

    product_state = np.kron(first_qutrit_rho, first_qutrit_rho)
    return product_state


def print_state_info(state, name):
    """Print information about a quantum state."""
    print(f"\n{'='*50}")
    print(f"State: {name}")
    print(f"{'='*50}")
    print(f"Trace: {np.trace(state):.6f}")
    print(f"Hermitian: {np.allclose(state, state.conj().T)}")
    eigenvals = np.linalg.eigvals(state)
    print(f"Positive semidefinite: {np.all(eigenvals >= -1e-10)}")
    print(f"Shape of state: {state.shape}")

def magic_test_state(phase):
    """Create a magic test state for two qutrits."""
    # |ψ⟩ = (|00⟩ + |11⟩ + |22⟩) / √3
    state_00 = np.kron([1, 0, 0], [1, 0, 0])  # |00⟩
    state_11 = np.kron([0, 1, 0], [0, 1, 0])  # |11⟩
    state_22 = np.kron([0, 0, 1], [0, 0, 1])  # |22⟩

    psi = (state_00 + state_11 + phase * state_22) / np.sqrt(3)
    return psi[:, np.newaxis] @ psi.conj().T[np.newaxis, :]

def custom_state(p1, p2):
    state = p1* magic_test_state(-1) + p2 * magic_test_state(1j)
    return state


def get_default_test_states():
    """Return a dictionary of default quantum states for testing."""
    return {
        "Maximally Mixed State": create_maximally_mixed_state(),
        "Product State |00⟩": create_product_state(),
        "Maximally Entangled State": create_maximally_entangled_state(),
        "Tensor product state": tensor_state(alpha=1, beta=1),
        "Magic Test State": magic_test_state(-1),
        "Custom State": custom_state(1/4, 3/4)
    }

