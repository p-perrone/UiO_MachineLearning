# python3
# Source code with all the functions for "Applied Data Analysis and Machine Learning" course
# University of Oslo, Fall 2025
#---------------------------------
# author: Pietro Perrone
#

#--------------------------------
# dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import autograd.numpy as np
from autograd import grad

#--------------------------------
# FIRST PART: PREPROCESSING AND OLS
#--------------------------------

# Runge function
def Runge(x, noise=True):
    """ Computes Runge function with or without a normal distributed stochastic noise.
        Parameters:
        :: x (array) = input dataset
        :: noise (bool) = adds some noise if True
    """
    if not isinstance(noise, bool):
        raise TypeError(f"`noise` must be boolean (True or False) and not {type(noise)}")

    if noise == True:
        y = ( 1 / (1 + 25 * x**2) ) + np.random.normal(loc=0, scale=1, size=len(x))
    elif noise == False:
        y = ( 1 / (1 + 25 * x**2) )

    return y

# OLS implementation
def polynomial_features(x, p, intercept=True):
    """ Computes the design matrix for linear regression.
        Parameters:
        :: x (array) = input dataset
        :: p (int) = polynomial degree
        :: intercept (bool) = sets first column at 1 if True, at 0 if False
    """
    if not isinstance(intercept, bool):
        raise TypeError("Intercept must be boolean (True or False)")
        
    n = len(x)
    X = np.zeros((n, p+1))
    
    if intercept == True:
        for j in range(0, p+1):
            X[:, j] = x**(j)
    elif intercept == False:
        for j in range(1, p+1):
            X[:, j] = x**(j)

    return X

def OLS_params(X, y):
    """ Computes optimal parameters for ordinary least squares regression.
        Parameters:
        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: y (array) = true value to be modeled
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y

