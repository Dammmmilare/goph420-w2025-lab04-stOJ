import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from goph420_lab04.regression import multi_regression

# Load the earthquake data from CSV
DATA_FILE = "earthquake_data.csv"  # Update this with the correct file path
data = pd.read_csv(DATA_FILE)

# Ensure data has the required columns
if "Magnitude" not in data.columns or "Frequency" not in data.columns:
    raise ValueError("Error: Input CSV must contain 'Magnitude' and 'Frequency' columns.")

# Remove rows where frequency is zero or negative (log10 is undefined for N=0)
data = data[data["Frequency"] > 0].dropna()

# Extract magnitude (M) and frequency (N) values
M = data["Magnitude"].values
N = data["Frequency"].values

# Apply logarithmic transformation to linearize the Gutenberg-Richter relation
Y = np.log10(N)

# Prepare the design matrix Z (adding a bias term for the intercept)
Z = np.column_stack((np.ones_like(M), M))

# Perform multiple linear regression
a, e, r2 = multi_regression(Y, Z)

# Extract regression parameters
A = a[0]  # Intercept (corresponds to 'a' in Gutenberg-Richter)
B = a[1]  # Slope
b_value = -B  # Convert slope to 'b' parameter

# Print results
print(f"Regression Equation: log10(N) = {A:.4f} - {b_value:.4f} * M")
print(f"Coefficient of Determination (R²): {r2:.4f}")

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(M, Y, label="Observed Data", color="blue", alpha=0.7)
plt.plot(M, A + B * M, color="red", label=f"Best Fit (b = {b_value:.2f})", linewidth=2)
plt.xlabel("Magnitude (M)")
plt.ylabel("log10(Frequency)")
plt.title(f"Gutenberg-Richter Law Fit (R² = {r2:.3f})")
plt.legend()
plt.grid(True)
plt.show()

# Save results to CSV
results = pd.DataFrame({
    "Magnitude": M,
    "Observed_log10_N": Y,
    "Predicted_log10_N": A + B * M,
    "Residuals": e
})
results.to_csv("regression_results.csv", index=False)
print("Results saved to regression_results.csv")