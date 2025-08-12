#!/usr/bin/env python3
"""
Simple example of calculating contextual fraction for a specific quantum state.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.lin_prog import contextual_fraction
from utils.states import create_maximally_mixed_state, create_product_state


def example_usage():
    """Example of how to use the contextual_fraction function."""
    
    # Example 1: Maximally mixed state
    print("Example 1: Maximally Mixed State")
    print("-" * 40)
    rho_mixed = create_maximally_mixed_state()
    
    result = contextual_fraction(rho_mixed)
    
    if result['success']:
        print(f"Contextual fraction: {result['b']:.6f}")
    else:
        print("Optimization failed!")
        
    # Example 2: Product state |00⟩
    print("\nExample 2: Product State |00⟩")
    print("-" * 40)
    rho_product = create_product_state()
    
    result = contextual_fraction(rho_product)
    
    if result['success']:
        print(f"Contextual fraction: {result['b']:.6f}")
    else:
        print("Optimization failed!")
        
    # Example 3: Custom state (you can modify this)
    print("\nExample 3: Custom Entangled State")
    print("-" * 40)
    # Create a simple entangled state: (|00⟩ + |11⟩) / √2
    state_00 = np.kron([1, 0, 0], [1, 0, 0])
    state_11 = np.kron([0, 1, 0], [0, 1, 0])
    psi = (state_00 + state_11) / np.sqrt(2)
    rho_entangled = psi[:, np.newaxis] @ psi.conj().T[np.newaxis, :]
    
    result = contextual_fraction(rho_entangled)
    
    if result['success']:
        print(f"Contextual fraction: {result['b']:.6f}")
    else:
        print("Optimization failed!")


if __name__ == "__main__":
    example_usage()
