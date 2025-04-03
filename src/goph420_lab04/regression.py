import numpy as np

def multi_regression(y, z):
    """
    Perform multiple regression on y with respect to z.

    Parameters
    ----------
    y : array_like, shape = (n,) or (n,1)
        The vector of dependent variable data
    Z : array_like, shape = (n,m)
        The matrix of independent variable data

    Returns
    -------
    numpy.ndarray, shape = (m,) or (m,1)
    The vector of model coefficients
    numpy.ndarray, shape = (n,) or (n,1)
    The vector of residuals

    float
    The coefficient of determination, r^2
    """
    #Ensure imputa are Numpy arrays and flatten y
    y = np.asarray(y).flatten()
    z = np.asarray(z)

    # check if z is singular or non singular
    if np.linalg.matrix_rank(z) < z.shape[1]:
        raise ValueError("The matrix z is singular or non-invertible.")
    
    # Computing regression coefficients
    Ztz = z.T @ z
    Zty = z.T @ y
    a = np.linalg.solve(Ztz, Zty)

    # computing predicted values
    y_hat = z @ a

    # Calculate residuals
    residuals = y - y_hat

    #Computing R-Squared
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residuals = np.sum(residuals**2)
    r_squared = 1 - (ss_residuals / ss_total)

    return a, residuals, r_squared