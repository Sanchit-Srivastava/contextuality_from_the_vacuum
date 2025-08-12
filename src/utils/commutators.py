# For checking pairwise commutators in contexts. 

import numpy as np

def commute_check(A, B):
    """
    Checks if two operators A and B commute in the context of qudit systems.
    The commutation is determined by evaluating (A[0] * B[1] - A[1] * B[0] + A[2] * B[3] - A[3] * B[2]) % 3 == 0.
    """
    return (A[0] * B[1] - A[1] * B[0] + A[2] * B[3] - A[3] * B[2]) % 3 == 0 
    

