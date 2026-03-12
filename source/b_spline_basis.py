from source.cox_de_boor import bspline_basis_zero
import numpy as np
def bspline_basis(x_val, k, d, knots):
    """
    Cox-de Boor recursion for B_{k,d}(x), with k using 0-based indexing.
    """
    if d == 0:
        return bspline_basis_zero(x_val, k, knots)

    # First term
    denom1 = knots[k + d] - knots[k]
    if denom1 == 0:
        term1 = 0.0
    else:
        term1 = ((x_val - knots[k]) / denom1) * bspline_basis(x_val, k, d - 1, knots)

    # Second term
    denom2 = knots[k + d + 1] - knots[k + 1]
    if denom2 == 0:
        term2 = 0.0
    else:
        term2 = ((knots[k + d + 1] - x_val) / denom2) * bspline_basis(x_val, k + 1, d - 1, knots)

    return term1 + term2