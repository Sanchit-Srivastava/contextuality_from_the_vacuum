import numpy as np

def to_ternary(n: int) -> np.ndarray:
    """
    Convert a number smaller than 81 into its ternary (base 3) representation,
    returned as a 4-digit numpy array (with leading zeros if necessary).

    Parameters:
    n (int): The number to convert. Must be in the range 0 <= n < 81.

    Returns:
    np.ndarray: A 4-element numpy array of integers representing the ternary digits.
    """
    if not (0 <= n < 81):
        raise ValueError("Input must be between 0 and 80 (inclusive of 0 and exclusive of 81)")

    # Special case for 0
    if n == 0:
        return np.array([0, 0, 0, 0])

    digits = []
    while n:
        digits.append(n % 3)
        n //= 3
    digits.reverse()
    
    # Pad the list with leading zeros to make it 4-digit long
    padded_digits = [0] * (4 - len(digits)) + digits
    return np.array(padded_digits)


if __name__ == "__main__":
    # Simple tests
    test_numbers = [0, 1, 2, 3, 4, 80]
    for number in test_numbers:
        print(f"{number} in ternary is {to_ternary(number)}")