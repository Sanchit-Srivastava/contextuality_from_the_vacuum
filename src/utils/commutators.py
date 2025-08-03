# For checking pairwise commutators in contexts. 

import numpy as np

def commute_check(A, B):
    """
    Checks if two operators A and B commute in the context of qudit systems.
    The commutation is determined by evaluating (A[0] * B[1] - A[1] * B[0] + A[2] * B[3] - A[3] * B[2]) % 3 == 0.
    """
    return (A[0] * B[1] - A[1] * B[0] + A[2] * B[3] - A[3] * B[2]) % 3 == 0 


def check_context_commutators():
    """Check if each pair A, B in contexts commute."""
    from utils.contexts import A, B
    
    print("\n" + "="*60)
    print("CONTEXT COMMUTATOR CHECK")
    print("="*60)
    
    commuting_contexts = 0
    non_commuting_contexts = []
    
    for c in range(len(A)):
        if commute_check(A[c], B[c]):
            commuting_contexts += 1
        else:
            non_commuting_contexts.append(c + 1)  # +1 for 1-based indexing
    
    print(f"Total contexts: {len(A)}")
    print(f"Commuting contexts: {commuting_contexts}")
    print(f"Non-commuting contexts: {len(non_commuting_contexts)}")
    
    if non_commuting_contexts:
        print(f"Non-commuting context numbers: {non_commuting_contexts}")
        print("WARNING: Some contexts have non-commuting operators!")
    else:
        print("✓ All context pairs (A, B) commute correctly!")
    
    print("="*60)
