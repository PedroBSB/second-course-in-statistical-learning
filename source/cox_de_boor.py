import numpy as np
def bspline_basis_zero(x_val, k, knots):
    """
    Degree-0 B-spline basis:
        B_{k,0}(x) = 1 if knots[k] <= x < knots[k+1], else 0
    Special handling at the right boundary so that x == knots[-1]
    belongs to the last basis function.
    """
    left = knots[k]
    right = knots[k + 1]

    if left <= x_val < right:
        return 1.0

    # Right-endpoint convention: include x == knots[-1] in the last interval
    if x_val == knots[-1] and k == len(knots) - 2:
        return 1.0

    return 0.0
