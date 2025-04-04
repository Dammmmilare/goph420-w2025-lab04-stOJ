import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import matplotlib.pyplot as plt
from goph420_lab04.regression import multi_regression



def main():

    # Importing the txt file
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/M_data1.txt"))
    print(f"Loading file from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # The data file contains two columns: time (in hours) and magnitude (M)
    data = np.loadtxt(file_path)
    

    # Time and magnitude data
    t = data[:, 0] 
    M_data = data[:, 1] 

    # First time interval  @ 0 < t < 35hrs
    int_1 = np.where(t < 35)[0][-1]  # Obtaining the index for values where t < 35

    m1 = np.linspace(-0.15, 0.8, 20)
    n1 = np.zeros_like(m1)

    for i, m1m in enumerate(m1): 
        n1[i] = np.count_nonzero(M_data[:int_1] > m1m)

    mask1 = n1 > 0
    y1 = np.log10(n1[mask1])
    m1 = m1[mask1]
    z1 = np.vstack((np.ones_like(m1), m1)).T

    a1, residual1, r_sq_1 = multi_regression(y1, z1) 

    print(f"a1: {a1}")
    print(f"Residuals: {residual1}")
    print(f"R^2: {r_sq_1}") 
    

    y_model1 = z1 @ a1

    # Second time interval @ 35hrs < t < 45hrs
    int_2 = np.where(t < 45)[0][-1]  # Obtaining the index for values where t < 45

    m2 = np.linspace(-0.15, 0.8, 20)
    n2 = np.zeros_like(m2) 

    for i, m2m in enumerate(m2):
        n2[i] = np.count_nonzero(M_data[int_1:int_2] > m2m)

    mask2 = n2 > 0
    y2 = np.log10(n2[mask2])
    m2 = m2[mask2]
    z2 = np.vstack((np.ones_like(m2), m2)).T

    a2, residual2, r_sq_2 = multi_regression(y2, z2)

    print(f"a2: {a2}") 
    print(f"Residuals: {residual2}")
    print(f"R^2: {r_sq_2}") 
    

    y_model2 = z2 @ a2

    # Third time interval @ 45hrs < t < 73hrs
    int_3 = np.where(t < 73)[0][-1]  # Obtaining the index for values where t < 73

    m3 = np.linspace(-0.15, 0.8, 20)
    n3 = np.zeros_like(m3) 

    for i, m3m in enumerate(m3):
        n3[i] = np.count_nonzero(M_data[int_2:int_3] > m3m)

    mask3 = n3 > 0
    y3 = np.log10(n3[mask3])
    m3 = m3[mask3]
    z3 = np.vstack((np.ones_like(m3), m3)).T

    a3, residual3, r_sq_3 = multi_regression(y3, z3)

    print(f"a3: {a3}")
    print(f"Residuals: {residual3}")
    print(f"R^2: {r_sq_3}") 
    

    y_model3 = z3 @ a3

    # Fourth time interval @ 73hrs < t < 96hrs
    int_4 = np.where(t < 96)[0][-1]  # Obtaining the index for values where t < 96

    m4 = np.linspace(-0.15, 0.8, 20)
    n4 = np.zeros_like(m4) 

    for i, m4m in enumerate(m4):
        n4[i] = np.count_nonzero(M_data[int_3:int_4] > m4m)

    mask4 = n4 > 0
    y4 = np.log10(n4[mask4])
    m4 = m4[mask4]
    z4 = np.vstack((np.ones_like(m4), m4)).T

    a4, residual4, r_sq_4 = multi_regression(y4, z4) 

    print(f"a4: {a4}")
    print(f"Residuals: {residual4}")
    print(f"R^2: {r_sq_4}") 
    

    y_model4 = z4 @ a4

    # Fifth time interval  @ 96hrs < t < 120hrs
    int_5 = np.where(t < 120)[0][-1]  # Obtaining the index for values where t < 120

    m5 = np.linspace(-0.15, 0.8, 20)
    n = np.zeros_like(m5) 

    for i, m5m in enumerate(m5):
        n[i] = np.count_nonzero(M_data[int_4:int_5] > m5m)

    y5 = np.log(n) 
    z5 = np.vstack((np.ones_like(m5), m5)).T 

    a5, residual5, r_sq_5 = multi_regression(y5, z5)

    print(f"a5: {a5}") 
    print(f"Residuals: {residual5}")
    print(f"R^2: {r_sq_5}") 
    

    y_model5 = z5 @ a5

    # Plotting the results
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    data_list = [
        (y_model1, a1, y1, m1, "0 < t < 35hrs"),
        (y_model2, a2, y2, m2, "35hrs < t < 45hrs"),
        (y_model3, a3, y3, m3, "45hrs < t < 73hrs"),
        (y_model4, a4, y4, m4, "73hrs < t < 96hrs"),
        (y_model5, a5, y5, m5, "96hrs < t < 120hrs"),
    ]

    for ax, (y_model, a, y_scatter, m_scatter, title) in zip(axes, data_list):
        ax.plot(m_scatter, y_model, label=f"y = {a[0]:.2f} + {a[1]:.2f}m")
        ax.scatter(m_scatter, y_scatter)
        ax.set_xlabel("m_data")
        ax.set_ylabel("log10(N)")
        ax.set_title(title)
        ax.legend()
        ax.grid()

    plt.tight_layout()

    # Ensure directory exists before saving
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../figures"))
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, "regression_plots.png"))


if __name__ == "__main__":
    main()