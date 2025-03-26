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
    # Add a column of ones to z
    z = np.column_stack((np.ones(len(z)), z))

    # Calculate beta
    beta = np.linalg.inv(z.T @ z) @ z.T @ y

    # Calculate residuals
    residuals = y - z @ beta

    return beta, residuals