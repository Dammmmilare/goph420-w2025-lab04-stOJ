import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import numpy as np
from goph420_lab04.regression import multi_regression

# Example data (Magnitude M and log-transformed Frequency log10(N))
M = np.array([1, 2, 3, 4, 5])
N = np.array([100, 50, 20, 10, 5])
Y = np.log10(N)

# Prepare matrix Z
Z = np.column_stack((np.ones_like(M), M))  # Add bias term (1s) for intercept

# Perform regression
a, residuals, r_sq = multi_regression(Y, Z)

print(f"Regression Coefficients: {a}")
print(f"Residuals: {residuals}")
print(f"R^2: {r_sq}")
