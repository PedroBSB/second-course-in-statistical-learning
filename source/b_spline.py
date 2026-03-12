import numpy as np
from source.b_spline_basis import bspline_basis
from fractions import Fraction

# Data
x = np.array([0.0, 2.0, 5.0, 7.0, 10.0], dtype=float)

# Clamped cubic knot vector:
# {0,0,0,0, 3,6, 10,10,10,10}
knots = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 10.0, 10.0, 10.0, 10.0], dtype=float)

degree = 3
M = len(knots)
K = M - degree - 1  # number of basis functions

# Build design matrix B
B = np.zeros((len(x), K), dtype=float)

for i, xv in enumerate(x):
    for k in range(K):
        B[i, k] = bspline_basis(xv, k, degree, knots)

# Fix tiny floating-point noise
B[np.abs(B) < 1e-14] = 0.0

# Print results
np.set_printoptions(precision=12, suppress=False)

print("Knot vector:")
print(knots)
print()

print("Degree:", degree)
print("Number of basis functions K:", K)
print()

print("B matrix:")
print(B)
print()

print("Row sums (should be 1):")
print(B.sum(axis=1))
print()

print("Selected row for x=5:")
row_index = np.where(x == 5.0)[0][0]
print(B[row_index])

print("Selected row for x=5 (fractions):")
print("[", ", ".join(str(Fraction(v).limit_denominator()) for v in B[row_index]), "]")