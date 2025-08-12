#!/usr/bin/env python3
"""
Main script to calculate contextual fraction for quantum states.

This script demonstrates how to calculate the contextual fraction for various
quantum states using the linear programming approach.
"""

import numpy as np
import sys
import os

# Add src directory to path (go up one level from examples/ to project root)
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'src'))

# Import with try/except for better error handling
try:
    from optimization.lin_prog import contextual_fraction
    from utils.commutators import check_context_commutators
    from utils.states import (
        print_state_info,
        get_default_test_states
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def main():
    """
    Main function to calculate contextual fractions for various quantum states.
    """
    print("Contextual Fraction Calculator for Two-Qutrit Systems")
    print("=" * 60)
    
    # First, check if all context pairs commute
    check_context_commutators()
    
    # Get default test states
    states = get_default_test_states()
    
    results = {}
    
    for state_name, rho in states.items():
        print_state_info(rho, state_name)
        
        try:
            # Calculate contextual fraction
            result = contextual_fraction(rho)
            
            if result['success']:
                cf = result['b']
                print(f"Contextual Fraction: {cf:.6f}")
                print("Optimization Status: SUCCESS")
                results[state_name] = cf
            else:
                print("Optimization Status: FAILED")
                print(f"Message: {result['result'].message}")
                results[state_name] = None
                
        except Exception as e:
            print(f"Error calculating contextual fraction: {e}")
            results[state_name] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for state_name, cf in results.items():
        if cf is not None:
            print(f"{state_name:<30}: {cf:.6f}")
        else:
            print(f"{state_name:<30}: FAILED")


if __name__ == "__main__":
    main()
