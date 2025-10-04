# python3
# Source code with all the functions for "Applied Data Analysis and Machine Learning" master course
# University of Oslo, Autumn 2025
#---------------------------------
# author: Pietro Perrone
#
""" 
=====================
ml_project1.functions
=====================
This module contains functions for:
    - Generating the Runge function with or without noise
    - Creating a design matrix for polynomial regression
    - Computing optimal parameters for Ordinary Least Squares (OLS) and Ridge (L2) regression
    - Performing linear regression predictions
    - Standard scaling of data
    - Gradient descent optimization with different methods (AdaGrad, RMSprop, Adam)
    - Lasso regression using gradient descent with L1 regularization
"""

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
# FIRST PART: PREPROCESSING, OLS AND RIDGE
#--------------------------------


# Runge function
def Runge(x, noise=True, noisescale=1):
    """ Computes Runge function with or without a normal distributed stochastic noise.
        Parameters:
        :: x (array) = input dataset
        :: noise (bool) = adds some noise if True
    """
    if not isinstance(noise, bool):
        raise TypeError(f"`noise` must be boolean (True or False) and not {type(noise)}")

    if noise == True:
        y = ( 1 / (1 + 25 * x**2) ) + np.random.normal(loc=0, scale=noisescale, size=len(x))
    elif noise == False:
        y = ( 1 / (1 + 25 * x**2) )

    return y


# OLS implementation

class LinearRegression_own:
    """
    Own implementation of linear regression with OLS and Ridge methods and their analytical solutions.
    Includes polynomial feature expansion and fitting.
    
    Based on scikit-learn linear regression workflow.
    """

    def __init__(self, intercept=True):
        self.intercept = intercept


    def polynomial_features(self, x, p):  
        """ Computes the design matrix for linear regression.
            Parameters:
            :: x (array) = input dataset
            :: p (int) = polynomial degree
            :: intercept (bool) = sets first column at 1 if True, at 0 if False
        """
        if not isinstance(self.intercept, bool):
            raise TypeError("Intercept must be boolean (True or False)")

        n = len(x)
        X = np.zeros((n, p+1))

        if self.intercept == True:
            for j in range(0, p+1):
                X[:, j] = x**(j)
        
        if not isinstance(self.intercept, bool):
            raise TypeError(f"Intercept must be boolean (True or False), not {type(self.intercept)}")
            
        n = len(x)
        X = np.zeros((n, p+1))
        
        if self.intercept == True:
            for j in range(0, p+1):
                X[:, j] = x**(j)
        elif self.intercept == False:
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


    def Ridge_params(X, y, lbda):
        """ Computes optimal parameters for Ridge regression.
            Parameters:
            :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
            :: y (array) = true value to be modeled
            :: lmbda (scalar) = penalty coefficient
        """
        return np.linalg.inv(X.T @ X + lbda * np.identity(X.shape[1])) @ X.T @ y


    def predict(self, X, beta):
        """ Computes the regression curve (predicted y values).
            Parameters:
            :: X (matrix) = design matrix
            :: beta (array) = optimal OLS parameters obtained with OLS_params(X, y)
        """
        return X @ beta


# standard scaling
def standard_scaler(a, centering=True, stdscaling=True):
    """ Scales input array a.
        Parameters:
        :: a (array) = input 1D array
        :: centering (bool) = substracts a.mean()
        :: scaling (bool) = divides by a.std()
    """
    if not isinstance(a, np.ndarray):
        raise TypeError("'a' must be a 1D array")
    if not isinstance(centering, bool):
        raise TypeError("'centering' must be boolean (True or False)")
    if not isinstance(stdscaling, bool):
        raise TypeError("'stdscaling' must be boolean (True or False)")

    if centering:
        centered = a - a.mean()
    else:
        centered = a

    if stdscaling:
        scaled_a = centered / a.std()
    else:
        scaled_a = centered
        
    return scaled_a


#--------------------------------
# SECOND PART: GRADIENT DESCENT AND LASSO
#--------------------------------


# defining cost function for OLS, Ridge and Lasso
def cost_function(y, X, theta, regression_method='OLS', lbda=0.1):
    """ Cost function for OLS, Ridge or Lasso regression.

        Parameters:
        :: y (array) = true value to be modeled
        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: regression_method (str) = 'OLS', 'Ridge' or 'Lasso'
        :: lbda (scalar) = regularization parameter (only for Ridge and Lasso)
    """
    if regression_method == 'OLS':
        return np.sum((y - X @ theta)**2)
    
    elif regression_method == 'Ridge':
        return np.sum((y - X @ theta)**2) + lbda * np.sum(theta**2)
    
    elif regression_method == 'Lasso':
        return np.sum((y - X @ theta)**2) + lbda * np.sum(np.abs(theta))
    
    else:
        raise ValueError(f"`regression_method` must be 'OLS', 'Ridge' or 'Lasso', not {regression_method}")


# analytical computation of gradient
def grad_analytic(X, y, theta, lbda, regression_method='OLS'):
        """Compute gradient analytically.

            Parameters:
            :: X_batch (matrix) = design matrix for the batch
            :: y_batch (array) = true value to be modeled for the batch
            :: theta (array) = current parameters
            :: regression_method (str) = 'OLS', 'Ridge' or 'Lasso'
        """

        if not isinstance(regression_method, str):
             raise TypeError(f"`regression_method` must be a string, not {type(regression_method)}")

        # predicted values
        y_pred = X @ theta
        # residuals
        residuals = y_pred - y.reshape(-1, 1)

        if regression_method == 'OLS':
            return (2.0 / len(y)) * (X.T @ residuals)
        elif regression_method == 'Ridge':
            return 2.0 * ((X.T @ residuals) / len(y) + (lbda * theta))
        elif regression_method == 'Lasso':
            return 2.0 * ((X.T @ residuals) / len(y) + lbda * np.sign(theta))
        else:
             raise ValueError(f"`regression_method` must be either 'OLS', 'Ridge' or 'Lasso', not {regression_method}")


# optimal parameters with simple gradient descent
def theta_gd(X, y, eta, regression_method='OLS', lbda=1, iterations=2000, converge = 1e-8):
    """ Computes optimal parameters for ordinary least squares regression with gradient descent.
        Parameters:

        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: y (array) = true value to be modeled
        :: eta (scalar) = learning rate
        :: regression_method (str) = 'OLS' or 'Ridge'
        :: lbda (scalar) = regularization parameter (only for Ridge)
        :: iterations (int) = maximum number of iterations
        :: converge (scalar) = convergence criterion
    """
    # error for wrong regression_method input
    if regression_method not in ['OLS', 'Ridge', 'Lasso']:
        raise ValueError("regression_method must be 'OLS', 'Ridge' or 'Lasso'")

    # defining cost function
    if regression_method == 'Ridge':
        cost = lambda theta: cost_function(y, X, theta, regression_method='Ridge', lbda=lbda)
    elif regression_method == 'Lasso':
        cost = lambda theta: cost_function(y, X, theta, regression_method='Lasso', lbda=lbda)
    else:
        cost = lambda theta: cost_function(y, X, theta)

    # initialize random theta
    theta = np.random.randn(X.shape[1], 1)

    # gradient descent loop
    for k in range(iterations):
        gradient = grad(cost)(theta).reshape(-1, 1)  # compute gradient with autograd
        theta -= eta * gradient

        # convergence of the algorithm
        gradient_norm = np.linalg.norm(gradient)
        if eta * gradient_norm <= converge:
            print(f"Stop at epsilon = {gradient_norm:.2e}, iteration = {k}")
            break

    return theta    


# optimal parameters with momentum gradient descent and different eta update methods
def theta_gd_mom(X, y, regression_method='OLS', eta=1e-3, eta_update_method='Simple_momentum', lbdaRidge=0.1, 
                lbdaLasso=0.1, momentum=0.9, iterations=2000, converge = 1e-8):
    """ Computes optimal parameters for ordinary least squares regression with momentum gradient descent.
        Parameters:
        :: X (matrix) = design matrix obtained with polynomial_features(x, p, intercept)
        :: y (array) = true value to be modeled
        :: regression_method (str) = 'OLS', 'Ridge' or 'Lasso'
        :: eta (scalar) = learning rate - default 1e-3 (optimal from experiments results)
        :: lbdaRidge (scalar) = penalty coefficient for Ridge regression
        :: lbdaLasso (scalar) = penalty coefficient for Lasso regression
        :: momentum (scalar) = momentum coefficient for momentum gradient descent
        :: eta_update_method (str) = 'AdaGrad', 'RMSprop', 'ADAM' or 'Simple_momentum'
        :: iterations (int) = maximum number of iterations
        :: converge (scalar) = convergence criterion
    """

    # error for wrong input
    if regression_method != 'OLS' and regression_method != 'Ridge' and regression_method != 'Lasso':
        raise ValueError(f"'regression_method' must be either 'OLS' or 'Ridge' and not {regression_method}")
    if eta_update_method not in ['AdaGrad', 'RMSprop', 'ADAM', 'Simple_momentum']:
        raise ValueError(f"'eta_update_method' must be either 'AdaGrad', 'RMSprop', 'ADAM' or 'Simple_momentum' and not {eta_update_method}")

    # initialize random theta and velocity
    theta = np.random.randn(X.shape[1], 1)
    velocity = np.zeros((X.shape[1], 1))

    # common params
    delta = 1e-8    # to avoid division by zero
    t = 0           # time step for Adam
    Giter = 0.0     # initialize sum of squared gradients for AdaGrad, RMSprop and Adam

    # defining cost function
    if regression_method == 'OLS':
        cost = lambda theta: cost_function(y, X, theta)
    elif regression_method == 'Ridge':
        cost = lambda theta: cost_function(y, X, theta, regression_method='Ridge', lbda=lbdaRidge)
    elif regression_method == 'Lasso':
        cost = lambda theta: cost_function(y, X, theta, regression_method='Lasso', lbda=lbdaLasso)

    for k in range(iterations):
        gradient = grad(cost)(theta).reshape(-1, 1)  # compute gradient with autograd
        if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            print(f"NaN in gradient at iteration {k}")
            print("Gradient stats:", np.min(gradient), np.max(gradient))
            break

        # update theta according to the chosen method
        if eta_update_method == 'Simple_momentum':
            velocity = momentum * velocity - eta * gradient     # simple momentum update
            update = velocity
            theta += update
        
        elif eta_update_method == 'AdaGrad': 
            # AdaGrad tracks the sum of the squares of the previous gradients
            Giter += gradient**2    # square the gradient
            adjusted_eta = eta / (delta + np.sqrt(Giter))
            update = adjusted_eta * gradient
            theta -= update

        elif eta_update_method == 'RMSprop':
            # RMSprop modificates AdaGrad by useing a moving average of squared gradients to normalize the gradient
            beta = 0.99     # discount factor that controls the averaging time of the second moment (variance of the gradient)
            Giter = beta * Giter + (1 - beta) * gradient**2
            adjusted_eta = eta / (delta + np.sqrt(Giter))
            update = adjusted_eta * gradient
            theta -= update

        elif eta_update_method == 'ADAM':
            # Adam (Adaptive Moment Estimation) is an extension to RMSprop that also takes 
            # into account the moving average of the gradient itself
            beta1 = 0.9     # decay rate for the first moment estimate (velocity = weighted average of the past gradients)
            beta2 = 0.99    # decay rate for the second moment estimate (Giter = weighted average of the past squared gradients)
            t += 1          # time step
            
            # update first moment estimate
            velocity = beta1 * velocity + (1 - beta1) * gradient
            # update second moment estimate
            Giter = beta2 * Giter + (1 - beta2) * gradient**2
            
            # compute bias-corrected first moment estimate
            velocity_corrected = velocity / (1 - beta1**t)
            # compute bias-corrected second moment estimate
            Giter_corrected = Giter / (1 - beta2**t)

            adjusted_eta = eta / (delta + np.sqrt(Giter_corrected))
            update = adjusted_eta * velocity_corrected
            theta -= update

        # convergence of the algorithm
        update_norm = np.linalg.norm(update)
        if update_norm <= converge:
            print(f"Stop at epsilon = {update_norm:.2e}, iteration = {k}")
            break

    return theta








#--------------------------------
# TEST PART: TESTING AND VISUALIZATION
#--------------------------------

# matrices for MSE and R2 as a function of p and n
def MSE_R2_pn():
    """ Evaluates MSE and R2 as a function of complexity `p` and dimensionality `n` for OLS regression
    """

    n_range = np.linspace(10, 100000, 10).astype(int)
    p_range = np.arange(1, 21, 1)

    MSE_train_matrix = np.zeros((len(n_range), len(p_range)))
    MSE_test_matrix = np.zeros((len(n_range), len(p_range)))
    Rsquared_train_matrix = np.zeros((len(n_range), len(p_range)))
    Rsquared_test_matrix = np.zeros((len(n_range), len(p_range)))

    for i, n in enumerate(n_range):
        
        xi = np.linspace(-1, 1, n)
        yi = Runge(xi, noise=False)
        xi_train, xi_test, yi_train, yi_test = train_test_split(xi, yi, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        xi_train = scaler.fit_transform(xi_train.reshape(-1, 1)).flatten()
        xi_test = scaler.transform(xi_test.reshape(-1, 1)).flatten()

        for j, pi in enumerate(p_range):
            # suggested by DeepSeek: skip if p > n_train
            n_train = len(xi_train)
            if pi >= n_train:  # p features vs n_train samples
                MSE_train_matrix[i, j] = np.nan
                MSE_test_matrix[i, j] = np.nan
                Rsquared_train_matrix[i, j] = np.nan
                Rsquared_test_matrix[i, j] = np.nan
                continue
            
            Xi_train = polynomial_features(xi_train, pi)
            Xi_test = polynomial_features(xi_test, pi)

            betai = OLS_params(Xi_train, yi_train)

            yi_pred_train = linear_regression(Xi_train, betai)
            yi_pred_test = linear_regression(Xi_test, betai)
            
            MSE_train_matrix[i, j] = mean_squared_error(yi_train, yi_pred_train)
            MSE_test_matrix[i, j] = mean_squared_error(yi_test, yi_pred_test)
            Rsquared_train_matrix[i, j] = r2_score(yi_train, yi_pred_train)
            Rsquared_test_matrix[i, j] = r2_score(yi_test, yi_pred_test)
    
    return MSE_train_matrix, MSE_test_matrix, Rsquared_train_matrix, Rsquared_test_matrix